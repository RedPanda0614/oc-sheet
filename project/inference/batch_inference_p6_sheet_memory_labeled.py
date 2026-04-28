"""
Batch inference for P6 sheet memory on top of a fine-tuned IP-Adapter checkpoint.

Purpose
-------
This script implements the proposal's P6 stage:

1. group labeled validation pairs by sheet_id
2. generate panels sequentially for each sheet
3. after each accepted panel, update sheet memory
4. use the evolving memory state for later panel generation and reranking

Design choice
-------------
The original implementation represented sheet memory as a single collage image
built from the original reference and accepted panels. In practice, feeding a
multi-face collage back into IP-Adapter can amplify artifacts and harm
identity preservation. This script therefore supports multiple memory modes:

- `collage`: original behavior, kept for reproducibility
- `latest_panel`: use the most recent accepted panel as the next reference
- `best_identity`: use the accepted panel most similar to the original reference
- `rerank_only`: always generate from the original reference and use memory only
  during reranking

`rerank_only` is the safest option when sheet-memory drift hurts quality.

Each panel request still uses P5-style reranking:
- generate 4 candidates
- score them for expression / identity / palette / copy
- add extra consistency scores against accepted memory panels
- choose the best one

Output
------
The output directory contains:
- selected panel images at the root
- candidate images under `candidates/...`
- memory collage images under `memory_inputs/...`
- `manifest.json` whose records point to the selected panel for each generated
  sheet panel
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from run_baseline import NEGATIVE_PROMPT, load_all_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from arcface_similarity import ArcFaceEvaluator  # noqa: E402
from copy_score import copy_score, copy_violation  # noqa: E402
from expression_classifier import CLIPControlEvaluator  # noqa: E402
from palette_distance import palette_distance  # noqa: E402


EMOTION_PROMPTS = {
    "neutral": "manga character, neutral expression, calm face, 1girl, high quality",
    "happy": "manga character, smiling, happy expression, 1girl, high quality",
    "sad": "manga character, sad expression, teary eyes, 1girl, high quality",
    "angry": "manga character, angry expression, frowning, 1girl, high quality",
    "surprised": "manga character, surprised expression, wide eyes, 1girl, high quality",
    "crying": "manga character, crying, tears, 1girl, high quality",
    "embarrassed": "manga character, embarrassed, blushing, 1girl, high quality",
}

EMOTION_ORDER = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "crying",
    "embarrassed",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-json",
        default="data/label_pairs/val.json",
        help="Labeled validation pairs JSON with target_emotion.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing image_proj_model.pt and ip_attn_procs.pt (typically P4 output).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/p6_sheet_memory_labeled",
        help="Directory for selected outputs, candidates, memory inputs, and manifest.json.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename written inside output-dir.",
    )
    parser.add_argument("--scale", type=float, default=0.7, help="IP-Adapter scale during inference.")
    parser.add_argument("--sd-path", default="models/sd-v1-5", help="Stable Diffusion base model path.")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter", help="Local IP-Adapter repo path.")
    parser.add_argument(
        "--max-sheets",
        type=int,
        default=0,
        help="How many sheets to process; 0 means all sheets in pairs-json.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Alias for max-sheets, kept for CLI consistency with earlier scripts.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="How many candidates to generate per panel request.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps per candidate.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; candidate k for panel j uses seed + j * num_candidates + k.",
    )
    parser.add_argument(
        "--memory-image-size",
        type=int,
        default=512,
        help="Output size of the memory collage fed into IP-Adapter.",
    )
    parser.add_argument(
        "--max-memory-panels",
        type=int,
        default=4,
        help="Maximum number of images used to build memory: original reference + recent accepted panels.",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["collage", "latest_panel", "best_identity", "rerank_only"],
        default="collage",
        help=(
            "How sheet memory is used during generation. "
            "'collage' keeps the original implementation; "
            "'latest_panel' uses the most recent accepted panel; "
            "'best_identity' uses the accepted panel closest to the original reference; "
            "'rerank_only' always generates from the original reference and uses memory only in reranking."
        ),
    )
    parser.add_argument(
        "--copy-threshold",
        type=float,
        default=0.88,
        help="Threshold used to flag trivial reference copying.",
    )
    parser.add_argument("--w-expr-hit", type=float, default=3.0, help="Weight on exact expression match.")
    parser.add_argument("--w-expr-conf", type=float, default=1.0, help="Weight on expression confidence.")
    parser.add_argument("--w-id", type=float, default=1.0, help="Weight on identity similarity to original reference.")
    parser.add_argument("--w-palette", type=float, default=0.75, help="Weight on palette consistency to original reference.")
    parser.add_argument("--w-copy", type=float, default=0.75, help="Weight on lower copy score to original reference.")
    parser.add_argument(
        "--w-memory-id",
        type=float,
        default=0.75,
        help="Weight on identity consistency with accepted memory panels.",
    )
    parser.add_argument(
        "--w-memory-palette",
        type=float,
        default=0.50,
        help="Weight on palette consistency with accepted memory panels.",
    )
    parser.add_argument(
        "--copy-violation-penalty",
        type=float,
        default=2.0,
        help="Penalty applied if a candidate is flagged as too similar to the original reference.",
    )
    return parser.parse_args()


def resolve_checkpoint_paths(checkpoint_dir: str | Path) -> tuple[Path, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    image_proj_path = checkpoint_dir / "image_proj_model.pt"
    ip_attn_path = checkpoint_dir / "ip_attn_procs.pt"
    missing = [str(path) for path in (image_proj_path, ip_attn_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required checkpoint file(s):\n" + "\n".join(missing))
    return image_proj_path, ip_attn_path


def load_finetuned_ip_adapter_weights(pipe, image_proj_path: Path, ip_attn_path: Path, device: str) -> None:
    if not hasattr(pipe.unet, "encoder_hid_proj") or pipe.unet.encoder_hid_proj is None:
        raise RuntimeError("UNet does not expose encoder_hid_proj; cannot load fine-tuned image projection weights.")

    image_proj_state = torch.load(image_proj_path, map_location=device)
    pipe.unet.encoder_hid_proj.load_state_dict(image_proj_state, strict=True)

    attn_state = torch.load(ip_attn_path, map_location="cpu")
    matched = 0
    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    for name, proc_state in attn_state.items():
        if name not in pipe.unet.attn_processors:
            continue
        state = {
            key: value.to(device=device, dtype=target_dtype)
            for key, value in proc_state.items()
        }
        pipe.unet.attn_processors[name].load_state_dict(state)
        matched += 1

    if matched == 0:
        raise RuntimeError(
            "Loaded checkpoint attention weights, but no UNet attention processors matched. "
            "Please verify the diffusers/IP-Adapter save format."
        )

    print(f"Loaded image projection from: {image_proj_path}")
    print(f"Loaded attention processors from: {ip_attn_path} ({matched}/{len(attn_state)} matched)")


def normalize_metric(values: list[float | None], higher_is_better: bool) -> list[float]:
    present = [value for value in values if value is not None]
    if not present:
        return [0.0 for _ in values]

    min_value = min(present)
    max_value = max(present)
    if abs(max_value - min_value) < 1e-8:
        return [1.0 if value is not None else 0.0 for value in values]

    normalized = []
    for value in values:
        if value is None:
            normalized.append(0.0)
            continue
        if higher_is_better:
            score = (value - min_value) / (max_value - min_value)
        else:
            score = (max_value - value) / (max_value - min_value)
        normalized.append(float(score))
    return normalized


def build_sheet_requests(pairs: list[dict]) -> list[dict]:
    """
    Convert raw labeled pairs into sheet-level generation requests.

    For each sheet:
    - choose one original reference path (the most common reference in that sheet)
    - choose one target panel per unique target emotion
    - order requested emotions using a canonical expression order
    """

    pairs_by_sheet: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        sheet_id = pair.get("sheet_id")
        if sheet_id and pair.get("target_emotion") in EMOTION_PROMPTS:
            pairs_by_sheet[sheet_id].append(pair)

    requests = []
    for sheet_id, sheet_pairs in pairs_by_sheet.items():
        ref_counter = Counter(
            pair["reference_path"]
            for pair in sheet_pairs
            if pair.get("reference_path") and Path(pair["reference_path"]).exists()
        )
        if not ref_counter:
            continue
        base_reference_path = ref_counter.most_common(1)[0][0]

        by_emotion: dict[str, dict] = {}
        for pair in sheet_pairs:
            emotion = pair["target_emotion"]
            current = by_emotion.get(emotion)
            if current is None:
                by_emotion[emotion] = pair
                continue
            # Prefer a pair already aligned to the chosen base reference.
            if current.get("reference_path") != base_reference_path and pair.get("reference_path") == base_reference_path:
                by_emotion[emotion] = pair

        ordered_emotions = [
            emotion for emotion in EMOTION_ORDER if emotion in by_emotion
        ] + sorted(
            emotion for emotion in by_emotion if emotion not in EMOTION_ORDER
        )

        panel_requests = []
        for emotion in ordered_emotions:
            chosen = by_emotion[emotion]
            panel_requests.append(
                {
                    "target_emotion": emotion,
                    "target_path": chosen.get("target_path"),
                }
            )

        requests.append(
            {
                "sheet_id": sheet_id,
                "base_reference_path": base_reference_path,
                "panel_requests": panel_requests,
            }
        )

    requests.sort(key=lambda item: item["sheet_id"])
    return requests


def build_memory_collage(image_paths: list[str | Path], size: int) -> Image.Image:
    """
    Build a square collage summarizing the current memory state.

    The collage is used as a single IP-Adapter reference image, which gives us
    a practical sheet-memory mechanism without changing the model architecture.
    """

    if not image_paths:
        raise ValueError("image_paths must contain at least one image.")

    images = [
        ImageOps.fit(Image.open(path).convert("RGB"), (size, size), Image.LANCZOS)
        for path in image_paths
    ]
    if len(images) == 1:
        return images[0]

    grid_cols = math.ceil(math.sqrt(len(images)))
    grid_rows = math.ceil(len(images) / grid_cols)
    tile_w = size // grid_cols
    tile_h = size // grid_rows

    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        tile = ImageOps.fit(image, (tile_w, tile_h), Image.LANCZOS)
        canvas.paste(tile, (col * tile_w, row * tile_h))
    return canvas


def select_memory_reference_path(
    base_reference_path: str | Path,
    accepted_panel_paths: list[str],
    identity_eval: ArcFaceEvaluator,
    mode: str,
) -> str:
    """
    Pick a single clean reference image instead of constructing a collage.

    For `rerank_only`, we intentionally keep generation anchored to the
    original reference and let memory influence only candidate selection.
    """

    if mode == "rerank_only":
        return str(base_reference_path)

    if not accepted_panel_paths:
        return str(base_reference_path)

    if mode == "latest_panel":
        return accepted_panel_paths[-1]

    if mode == "best_identity":
        best_path = accepted_panel_paths[-1]
        best_score = float("-inf")
        for candidate_path in accepted_panel_paths:
            try:
                score = identity_eval.similarity(base_reference_path, candidate_path)
            except Exception:
                score = None
            if score is not None and score > best_score:
                best_score = float(score)
                best_path = candidate_path
        return best_path

    return str(base_reference_path)


def compute_mean_metric(metric_fn, anchor_path: str | Path, reference_paths: list[str | Path]) -> float | None:
    values = []
    for reference_path in reference_paths:
        try:
            value = metric_fn(reference_path, anchor_path)
        except Exception:
            value = None
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def score_candidate_set(candidate_metrics: list[dict], args) -> int:
    expr_hits = [metric["expression_hit"] for metric in candidate_metrics]
    expr_conf = [metric["prediction_confidence"] for metric in candidate_metrics]
    identities = [metric["identity_similarity"] for metric in candidate_metrics]
    palettes = [metric["palette_distance"] for metric in candidate_metrics]
    copies = [metric["copy_score"] for metric in candidate_metrics]
    memory_ids = [metric["memory_identity_similarity"] for metric in candidate_metrics]
    memory_palettes = [metric["memory_palette_distance"] for metric in candidate_metrics]

    expr_conf_norm = normalize_metric(expr_conf, higher_is_better=True)
    id_norm = normalize_metric(identities, higher_is_better=True)
    palette_norm = normalize_metric(palettes, higher_is_better=False)
    copy_norm = normalize_metric(copies, higher_is_better=False)
    memory_id_norm = normalize_metric(memory_ids, higher_is_better=True)
    memory_palette_norm = normalize_metric(memory_palettes, higher_is_better=False)

    best_idx = 0
    best_key = None

    for idx, metric in enumerate(candidate_metrics):
        rerank_score = (
            args.w_expr_hit * float(expr_hits[idx])
            + args.w_expr_conf * expr_conf_norm[idx]
            + args.w_id * id_norm[idx]
            + args.w_palette * palette_norm[idx]
            + args.w_copy * copy_norm[idx]
            + args.w_memory_id * memory_id_norm[idx]
            + args.w_memory_palette * memory_palette_norm[idx]
            - args.copy_violation_penalty * float(metric["copy_violation"])
        )

        metric["norm_expression_confidence"] = expr_conf_norm[idx]
        metric["norm_identity_similarity"] = id_norm[idx]
        metric["norm_palette_consistency"] = palette_norm[idx]
        metric["norm_copy_avoidance"] = copy_norm[idx]
        metric["norm_memory_identity_similarity"] = memory_id_norm[idx]
        metric["norm_memory_palette_consistency"] = memory_palette_norm[idx]
        metric["rerank_score"] = float(rerank_score)

        tie_break_key = (
            rerank_score,
            float(expr_hits[idx]),
            metric["prediction_confidence"] if metric["prediction_confidence"] is not None else -1.0,
            metric["identity_similarity"] if metric["identity_similarity"] is not None else -1.0,
            metric["memory_identity_similarity"] if metric["memory_identity_similarity"] is not None else -1.0,
            -(metric["copy_score"] if metric["copy_score"] is not None else 1e9),
            -(metric["palette_distance"] if metric["palette_distance"] is not None else 1e9),
            -idx,
        )
        if best_key is None or tie_break_key > best_key:
            best_key = tie_break_key
            best_idx = idx

    return best_idx


def main():
    args = parse_args()
    if args.max_sheets == 0 and args.n > 0:
        args.max_sheets = args.n

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_root = output_dir / "candidates"
    memory_root = output_dir / "memory_inputs"
    candidates_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    pairs = json.loads(Path(args.pairs_json).read_text())
    sheet_requests = build_sheet_requests(pairs)
    if args.max_sheets > 0:
        sheet_requests = sheet_requests[: args.max_sheets]

    image_proj_path, ip_attn_path = resolve_checkpoint_paths(args.checkpoint_dir)
    pipe, _, device = load_all_models(args)
    load_finetuned_ip_adapter_weights(pipe, image_proj_path, ip_attn_path, device)

    identity_eval = ArcFaceEvaluator()
    control_eval = CLIPControlEvaluator()

    manifest_records = []
    global_panel_idx = 0

    for sheet_request in tqdm(sheet_requests, desc="ip-adapter-p6-sheet-memory"):
        sheet_id = sheet_request["sheet_id"]
        base_reference_path = sheet_request["base_reference_path"]
        panel_requests = sheet_request["panel_requests"]

        accepted_panel_paths: list[str] = []
        accepted_panel_labels: list[str] = []

        for panel_step, panel_request in enumerate(panel_requests):
            target_emotion = panel_request["target_emotion"]
            target_path = panel_request.get("target_path")

            # Build the memory context from the original reference and the most
            # recently accepted panels.
            memory_slots = max(args.max_memory_panels - 1, 0)
            recent_memory_paths = (
                accepted_panel_paths[-memory_slots:] if memory_slots > 0 else []
            )
            recent_memory_labels = (
                accepted_panel_labels[-memory_slots:] if memory_slots > 0 else []
            )
            memory_image_paths = [base_reference_path] + recent_memory_paths

            if args.memory_mode == "collage":
                memory_image = build_memory_collage(
                    memory_image_paths,
                    size=args.memory_image_size,
                )
                memory_input_path = (
                    memory_root / f"{sheet_id}_step{panel_step:02d}_{target_emotion}.jpg"
                )
                memory_image.save(memory_input_path)
                memory_reference_paths_for_manifest = [
                    str(Path(path).resolve()) for path in memory_image_paths
                ]
                memory_reference_labels_for_manifest = ["original_reference"] + recent_memory_labels
            else:
                selected_memory_reference = select_memory_reference_path(
                    base_reference_path=base_reference_path,
                    accepted_panel_paths=recent_memory_paths,
                    identity_eval=identity_eval,
                    mode=args.memory_mode,
                )
                memory_image = Image.open(selected_memory_reference).convert("RGB")
                memory_input_path = Path(selected_memory_reference).resolve()
                memory_reference_paths_for_manifest = [str(memory_input_path)]
                if str(memory_input_path) == str(Path(base_reference_path).resolve()):
                    memory_reference_labels_for_manifest = ["original_reference"]
                else:
                    memory_reference_labels_for_manifest = ["selected_memory_panel"]

            prompt = EMOTION_PROMPTS[target_emotion]
            candidate_dir = candidates_root / sheet_id / f"step{panel_step:02d}_{target_emotion}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_metrics = []

            for candidate_idx in range(args.num_candidates):
                candidate_seed = args.seed + global_panel_idx * args.num_candidates + candidate_idx
                generator = torch.Generator(device=device).manual_seed(candidate_seed)

                try:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        ip_adapter_image=[memory_image],
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=generator,
                    ).images[0]
                except Exception as exc:
                    tqdm.write(
                        f"Failed to generate candidate {candidate_idx} for {sheet_id}/{target_emotion}: {exc}"
                    )
                    continue

                candidate_path = candidate_dir / f"candidate_{candidate_idx:02d}.jpg"
                image.save(candidate_path)

                try:
                    prediction = control_eval.predict(candidate_path, label_type="expression")
                    predicted_label = prediction.label
                    prediction_confidence = prediction.confidence
                except Exception as exc:
                    tqdm.write(f"Expression scoring failed for {candidate_path}: {exc}")
                    predicted_label = None
                    prediction_confidence = None

                try:
                    identity_similarity = identity_eval.similarity(base_reference_path, candidate_path)
                except Exception as exc:
                    tqdm.write(f"Identity scoring failed for {candidate_path}: {exc}")
                    identity_similarity = None

                try:
                    palette = palette_distance(base_reference_path, candidate_path)
                except Exception as exc:
                    tqdm.write(f"Palette scoring failed for {candidate_path}: {exc}")
                    palette = None

                try:
                    copied = copy_score(base_reference_path, candidate_path)
                    copy_flag = copy_violation(copied, threshold=args.copy_threshold)
                except Exception as exc:
                    tqdm.write(f"Copy scoring failed for {candidate_path}: {exc}")
                    copied = None
                    copy_flag = False

                memory_identity = compute_mean_metric(
                    identity_eval.similarity,
                    candidate_path,
                    accepted_panel_paths,
                ) if accepted_panel_paths else None

                memory_palette = compute_mean_metric(
                    palette_distance,
                    candidate_path,
                    accepted_panel_paths,
                ) if accepted_panel_paths else None

                candidate_metrics.append(
                    {
                        "candidate_index": candidate_idx,
                        "candidate_path": str(candidate_path.resolve()),
                        "seed": candidate_seed,
                        "predicted_label": predicted_label,
                        "prediction_confidence": prediction_confidence,
                        "expression_hit": float(predicted_label == target_emotion) if predicted_label else 0.0,
                        "identity_similarity": identity_similarity,
                        "palette_distance": palette,
                        "copy_score": copied,
                        "copy_violation": bool(copy_flag),
                        "memory_identity_similarity": memory_identity,
                        "memory_palette_distance": memory_palette,
                    }
                )

            if not candidate_metrics:
                tqdm.write(f"No candidates produced for {sheet_id}/{target_emotion}")
                continue

            best_idx = score_candidate_set(candidate_metrics, args)
            best_candidate = candidate_metrics[best_idx]
            selected_image = Image.open(best_candidate["candidate_path"]).convert("RGB")
            selected_path = output_dir / f"{global_panel_idx:04d}_{sheet_id}_{target_emotion}.jpg"
            selected_image.save(selected_path)

            accepted_panel_paths.append(str(selected_path.resolve()))
            accepted_panel_labels.append(target_emotion)

            manifest_records.append(
                {
                    "index": global_panel_idx,
                    "sheet_id": sheet_id,
                    "reference_path": str(Path(base_reference_path).resolve()),
                    "target_path": str(Path(target_path).resolve()) if target_path else None,
                    "generated_path": str(selected_path.resolve()),
                    "requested_label": target_emotion,
                    "ground_truth_target_emotion": target_emotion,
                    "label_type": "expression",
                    "seed": best_candidate["seed"],
                    "ip_adapter_scale": args.scale,
                    "baseline_type": "ip_adapter_p6_sheet_memory",
                    "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
                    "generation_mode": "sheet_memory_with_reranking",
                    "memory_mode": args.memory_mode,
                    "panel_step": panel_step,
                    "memory_input_path": str(memory_input_path),
                    "memory_reference_paths": memory_reference_paths_for_manifest,
                    "memory_reference_labels": memory_reference_labels_for_manifest,
                    "num_candidates": args.num_candidates,
                    "selected_candidate_index": best_candidate["candidate_index"],
                    "selected_rerank_score": best_candidate["rerank_score"],
                    "candidate_metrics": candidate_metrics,
                }
            )

            global_panel_idx += 1

    manifest_payload = {
        "summary": {
            "n_input_sheets": len(sheet_requests),
            "n_generated_panels": len(manifest_records),
            "num_candidates_per_panel": args.num_candidates,
            "max_memory_panels": args.max_memory_panels,
            "memory_mode": args.memory_mode,
            "rerank_weights": {
                "w_expr_hit": args.w_expr_hit,
                "w_expr_conf": args.w_expr_conf,
                "w_id": args.w_id,
                "w_palette": args.w_palette,
                "w_copy": args.w_copy,
                "w_memory_id": args.w_memory_id,
                "w_memory_palette": args.w_memory_palette,
                "copy_violation_penalty": args.copy_violation_penalty,
            },
        },
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))

    print(f"Saved P6 sheet-memory outputs to {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
