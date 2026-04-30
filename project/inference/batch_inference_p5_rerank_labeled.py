"""
Batch inference for P5 reranking on top of a fine-tuned IP-Adapter checkpoint.

Purpose
-------
This script implements the proposal's P5 stage:

1. load a fine-tuned reference-guided IP-Adapter checkpoint (typically P4)
2. generate multiple candidates per input pair (default: 4)
3. score each candidate for:
   - expression correctness / confidence
   - identity similarity
   - palette consistency
   - copy avoidance
4. select the best candidate automatically

Why this file exists
--------------------
We keep reranking as a standalone inference entrypoint so that:
- P1/P3/P4 training scripts stay unchanged
- zero-shot and single-sample baselines stay unchanged
- P5 can be evaluated with the same manifest / eval format as earlier runs

Output
------
The output directory contains:
- selected final images named like `0000_sheet_00008_happy.jpg`
- all four candidates under `candidates/...`
- `manifest.json` whose `generated_path` points to the selected image, plus
  candidate-level scores for qualitative analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageStat
from tqdm import tqdm

from run_baseline import NEGATIVE_PROMPT, load_all_models
from batch_inference_cfg_labeled import prepare_reference_cfg_embeds


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

FACE_PROMPTS = {
    "neutral": (
        "solo, close-up face, anime portrait, neutral calm expression, "
        "half-lidded eyes, relaxed eyebrows, closed mouth, simple background"
    ),
    "happy": (
        "solo, close-up face, anime portrait, laughing happily, "
        "eyes curved into crescents, wide smile showing teeth, rosy cheeks, simple background"
    ),
    "sad": (
        "solo, close-up face, anime portrait, sad expression, "
        "drooping eyes, eyebrows slanted inward, downturned mouth, simple background"
    ),
    "angry": (
        "solo, close-up face, anime portrait, furious angry expression, "
        "eyes narrowed, eyebrows sharply angled down, clenched teeth, simple background"
    ),
    "surprised": (
        "solo, close-up face, anime portrait, shocked expression, "
        "eyes wide open and round, eyebrows raised high, open mouth, simple background"
    ),
    "crying": (
        "solo, close-up face, anime portrait, crying expression, "
        "tears on cheeks, closed eyes, scrunched brows, simple background"
    ),
}

CLEAN_PROMPT_SUFFIX = (
    "front-facing portrait, centered face, same camera angle, straight-on view, "
    "clean plain white background, simple background, single character, head and shoulders"
)

CLEAN_NEGATIVE_PROMPT_EXTRA = (
    "side view, profile view, three-quarter view, turned head, tilted head, "
    "looking away, cropped face, panel border, comic panel, manga panel, speech bubble, "
    "text, watermark, busy background, detailed background, extra character, multiple people"
)


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
        default="results/p5_rerank_labeled",
        help="Directory for selected images, candidates, and manifest.json.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename written inside output-dir.",
    )
    parser.add_argument("--scale", type=float, default=0.7, help="IP-Adapter scale during inference.")
    parser.add_argument(
        "--image-cfg-scale",
        type=float,
        default=1.0,
        help=(
            "Classifier-free guidance scale applied to the reference-image condition. "
            "Use 1.10 to run P5 on top of the current best P4 cfg110 setting."
        ),
    )
    parser.add_argument(
        "--disable-image-cfg",
        action="store_true",
        help="Use the original IP-Adapter image path instead of precomputed reference CFG embeds.",
    )
    parser.add_argument("--sd-path", default="models/sd-v1-5", help="Stable Diffusion base model path.")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter", help="Local IP-Adapter repo path.")
    parser.add_argument("--n", type=int, default=500, help="Number of samples to run; 0 means full JSON.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; candidate k for sample i uses seed + i * num_candidates + k.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="How many candidates to generate before reranking.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Diffusion steps per candidate.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--prompt-style",
        choices=["default", "clean", "face"],
        default="default",
        help=(
            "Prompt variant. 'clean' appends view/background constraints; "
            "'face' replaces the base emotion prompts with close-up face prompts."
        ),
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Optional extra text appended to each expression prompt after the selected prompt style.",
    )
    parser.add_argument(
        "--negative-prompt-extra",
        default="",
        help="Optional extra negative prompt text appended after the selected prompt style.",
    )
    parser.add_argument(
        "--copy-threshold",
        type=float,
        default=0.88,
        help="Threshold used to flag trivial reference copying.",
    )
    parser.add_argument("--w-expr-hit", type=float, default=3.0, help="Weight on exact expression match.")
    parser.add_argument("--w-expr-conf", type=float, default=1.0, help="Weight on expression confidence.")
    parser.add_argument("--w-id", type=float, default=1.0, help="Weight on identity similarity.")
    parser.add_argument("--w-palette", type=float, default=0.75, help="Weight on palette consistency.")
    parser.add_argument("--w-copy", type=float, default=0.75, help="Weight on lower copy score.")
    parser.add_argument("--target-view", default="front", help="Desired generated view label for view reranking.")
    parser.add_argument("--w-view-hit", type=float, default=0.0, help="Weight on exact target-view match.")
    parser.add_argument("--w-view-conf", type=float, default=0.0, help="Weight on target-view confidence.")
    parser.add_argument("--w-background", type=float, default=0.0, help="Weight on low background clutter.")
    parser.add_argument(
        "--view-mismatch-penalty",
        type=float,
        default=0.0,
        help="Extra penalty when CLIP view prediction does not match --target-view.",
    )
    parser.add_argument(
        "--copy-violation-penalty",
        type=float,
        default=2.0,
        help="Penalty applied if a candidate is flagged as too similar to the reference.",
    )
    parser.add_argument(
        "--skip-target-labels",
        default="",
        help=(
            "Comma-separated target labels to skip during generation. "
            "Use 'neutral' to match the P4 cfg110 validation subset."
        ),
    )
    parser.add_argument(
        "--model-tag",
        default="",
        help="Optional manifest tag describing the backbone/sampling setting.",
    )
    return parser.parse_args()


def resolve_checkpoint_paths(checkpoint_dir: str | Path) -> tuple[Path, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    image_proj_path = checkpoint_dir / "image_proj_model.pt"
    ip_attn_path = checkpoint_dir / "ip_attn_procs.pt"

    missing = [str(p) for p in (image_proj_path, ip_attn_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required checkpoint file(s):\n" + "\n".join(missing)
        )
    return image_proj_path, ip_attn_path


def load_finetuned_ip_adapter_weights(pipe, image_proj_path: Path, ip_attn_path: Path, device: str) -> None:
    """
    Load a fine-tuned IP-Adapter checkpoint saved in the shared project format:
    - image_proj_model.pt
    - ip_attn_procs.pt
    """

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
    """
    Min-max normalize a metric within one candidate set.

    This keeps reranking comparable across metrics with different raw scales.
    Missing values become 0.0, and if all present values tie we assign 1.0 to
    all present entries because the metric offers no preference among them.
    """

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


def background_clutter_score(image_path: str | Path) -> float:
    """
    Estimate how much border/background clutter a candidate contains.

    The score focuses on the outer ring because panel borders, speech bubbles,
    and busy backgrounds usually show up there. Lower is cleaner.
    """

    image = Image.open(image_path).convert("RGB").resize((128, 128), Image.LANCZOS)
    border = 18
    strips = [
        image.crop((0, 0, 128, border)),
        image.crop((0, 128 - border, 128, 128)),
        image.crop((0, border, border, 128 - border)),
        image.crop((128 - border, border, 128, 128 - border)),
    ]

    edge_scores = []
    dark_fractions = []
    color_std_scores = []

    for strip in strips:
        gray = strip.convert("L")
        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge_scores.append(ImageStat.Stat(edge).mean[0] / 255.0)

        hist = gray.histogram()
        total = sum(hist) or 1
        dark_fractions.append(sum(hist[:45]) / total)

        color_std = ImageStat.Stat(strip).stddev
        color_std_scores.append(min(sum(color_std) / (3.0 * 96.0), 1.0))

    score = (
        0.50 * sum(edge_scores) / len(edge_scores)
        + 0.30 * sum(color_std_scores) / len(color_std_scores)
        + 0.20 * sum(dark_fractions) / len(dark_fractions)
    )
    return float(max(0.0, min(score, 1.0)))


def score_candidate_set(candidate_metrics: list[dict], args) -> int:
    """
    Score all candidates for a single input pair and return the selected index.

    Reranking policy:
    - exact expression match has the strongest raw weight
    - continuous metrics are normalized within the 4-way candidate set
    - copy-threshold violations receive an explicit penalty
    """

    expr_hits = [metric["expression_hit"] for metric in candidate_metrics]
    expr_conf = [metric["prediction_confidence"] for metric in candidate_metrics]
    identities = [metric["identity_similarity"] for metric in candidate_metrics]
    palettes = [metric["palette_distance"] for metric in candidate_metrics]
    copies = [metric["copy_score"] for metric in candidate_metrics]
    view_hits = [metric["view_hit"] for metric in candidate_metrics]
    view_conf = [metric["target_view_confidence"] for metric in candidate_metrics]
    background_clutter = [metric["background_clutter_score"] for metric in candidate_metrics]

    expr_conf_norm = normalize_metric(expr_conf, higher_is_better=True)
    id_norm = normalize_metric(identities, higher_is_better=True)
    palette_norm = normalize_metric(palettes, higher_is_better=False)
    copy_norm = normalize_metric(copies, higher_is_better=False)
    view_conf_norm = normalize_metric(view_conf, higher_is_better=True)
    background_clean_norm = normalize_metric(background_clutter, higher_is_better=False)

    best_idx = 0
    best_key = None

    for idx, metric in enumerate(candidate_metrics):
        rerank_score = (
            args.w_expr_hit * float(expr_hits[idx])
            + args.w_expr_conf * expr_conf_norm[idx]
            + args.w_id * id_norm[idx]
            + args.w_palette * palette_norm[idx]
            + args.w_copy * copy_norm[idx]
            + args.w_view_hit * float(view_hits[idx])
            + args.w_view_conf * view_conf_norm[idx]
            + args.w_background * background_clean_norm[idx]
            - args.copy_violation_penalty * float(metric["copy_violation"])
            - args.view_mismatch_penalty * float(metric["predicted_view"] != args.target_view)
        )
        metric["norm_expression_confidence"] = expr_conf_norm[idx]
        metric["norm_identity_similarity"] = id_norm[idx]
        metric["norm_palette_consistency"] = palette_norm[idx]
        metric["norm_copy_avoidance"] = copy_norm[idx]
        metric["norm_view_confidence"] = view_conf_norm[idx]
        metric["norm_background_cleanliness"] = background_clean_norm[idx]
        metric["rerank_score"] = float(rerank_score)

        # Keep a deterministic tie-breaker so behavior is stable across runs.
        tie_break_key = (
            rerank_score,
            float(expr_hits[idx]),
            float(view_hits[idx]),
            metric["prediction_confidence"] if metric["prediction_confidence"] is not None else -1.0,
            metric["target_view_confidence"] if metric["target_view_confidence"] is not None else -1.0,
            metric["identity_similarity"] if metric["identity_similarity"] is not None else -1.0,
            -(metric["background_clutter_score"] if metric["background_clutter_score"] is not None else 1e9),
            -(metric["copy_score"] if metric["copy_score"] is not None else 1e9),
            -(metric["palette_distance"] if metric["palette_distance"] is not None else 1e9),
            -idx,
        )
        if best_key is None or tie_break_key > best_key:
            best_key = tie_break_key
            best_idx = idx

    return best_idx


def parse_label_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def join_prompt_parts(*parts: str) -> str:
    return ", ".join(part.strip(" ,") for part in parts if part and part.strip(" ,"))


def build_prompt(target_emotion: str, args) -> str:
    if args.prompt_style == "face" and target_emotion in FACE_PROMPTS:
        prompt = FACE_PROMPTS[target_emotion]
    else:
        prompt = EMOTION_PROMPTS[target_emotion]
    if args.prompt_style == "clean":
        prompt = join_prompt_parts(prompt, CLEAN_PROMPT_SUFFIX)
    return join_prompt_parts(prompt, args.prompt_suffix)


def build_negative_prompt(args) -> str:
    negative_prompt = NEGATIVE_PROMPT
    if args.prompt_style == "clean":
        negative_prompt = join_prompt_parts(negative_prompt, CLEAN_NEGATIVE_PROMPT_EXTRA)
    return join_prompt_parts(negative_prompt, args.negative_prompt_extra)


def generate_with_reference_condition(pipe, prompt: str, negative_prompt: str, reference_image: Image.Image, args, device: str, generator):
    if args.disable_image_cfg:
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=[reference_image],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]

    image_embeds = prepare_reference_cfg_embeds(
        pipe=pipe,
        reference_image=reference_image,
        device=device,
        image_cfg_scale=args.image_cfg_scale,
    )
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=None,
        ip_adapter_image_embeds=image_embeds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]


def main():
    args = parse_args()
    if args.image_cfg_scale <= 0:
        raise ValueError("--image-cfg-scale must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_root = output_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    pairs = json.loads(Path(args.pairs_json).read_text())
    if args.n > 0:
        pairs = pairs[: args.n]

    image_proj_path, ip_attn_path = resolve_checkpoint_paths(args.checkpoint_dir)
    pipe, _, device = load_all_models(args)
    load_finetuned_ip_adapter_weights(pipe, image_proj_path, ip_attn_path, device)

    identity_eval = ArcFaceEvaluator()
    control_eval = CLIPControlEvaluator()

    skipped_target_labels = parse_label_set(args.skip_target_labels)
    valid_emotions = set(EMOTION_PROMPTS.keys()) - skipped_target_labels
    manifest_records = []
    skipped_invalid_label = 0
    skipped_missing_reference = 0

    for sample_idx, pair in enumerate(tqdm(pairs, desc="ip-adapter-p5-rerank")):
        target_emotion = pair.get("target_emotion")
        reference_path = pair.get("reference_path")
        target_path = pair.get("target_path")

        if target_emotion not in valid_emotions:
            skipped_invalid_label += 1
            continue
        if not reference_path or not Path(reference_path).exists():
            skipped_missing_reference += 1
            continue

        prompt = build_prompt(target_emotion, args)
        negative_prompt = build_negative_prompt(args)
        reference_image = Image.open(reference_path).convert("RGB")
        sample_prefix = f"{sample_idx:04d}_{pair.get('sheet_id', 'none')}_{target_emotion}"
        candidate_dir = candidates_root / sample_prefix
        candidate_dir.mkdir(parents=True, exist_ok=True)

        candidate_metrics = []

        for candidate_idx in range(args.num_candidates):
            candidate_seed = args.seed + sample_idx * args.num_candidates + candidate_idx
            generator = torch.Generator(device=device).manual_seed(candidate_seed)

            try:
                image = generate_with_reference_condition(
                    pipe=pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    reference_image=reference_image,
                    args=args,
                    device=device,
                    generator=generator,
                )
            except Exception as exc:
                tqdm.write(
                    f"Failed to generate candidate {candidate_idx} for {reference_path}: {exc}"
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
                view_prediction = control_eval.predict(candidate_path, label_type="view")
                predicted_view = view_prediction.label
                predicted_view_confidence = view_prediction.confidence
                target_view_confidence = (
                    predicted_view_confidence if predicted_view == args.target_view else 0.0
                )
            except Exception as exc:
                tqdm.write(f"View scoring failed for {candidate_path}: {exc}")
                predicted_view = None
                predicted_view_confidence = None
                target_view_confidence = None

            try:
                identity_similarity = identity_eval.similarity(reference_path, candidate_path)
            except Exception as exc:
                tqdm.write(f"Identity scoring failed for {candidate_path}: {exc}")
                identity_similarity = None

            try:
                palette = palette_distance(reference_path, candidate_path)
            except Exception as exc:
                tqdm.write(f"Palette scoring failed for {candidate_path}: {exc}")
                palette = None

            try:
                copied = copy_score(reference_path, candidate_path)
                copy_flag = copy_violation(copied, threshold=args.copy_threshold)
            except Exception as exc:
                tqdm.write(f"Copy scoring failed for {candidate_path}: {exc}")
                copied = None
                copy_flag = False

            try:
                background_clutter = background_clutter_score(candidate_path)
            except Exception as exc:
                tqdm.write(f"Background scoring failed for {candidate_path}: {exc}")
                background_clutter = None

            candidate_metrics.append(
                {
                    "candidate_index": candidate_idx,
                    "candidate_path": str(candidate_path.resolve()),
                    "seed": candidate_seed,
                    "predicted_label": predicted_label,
                    "prediction_confidence": prediction_confidence,
                    "expression_hit": float(predicted_label == target_emotion) if predicted_label else 0.0,
                    "predicted_view": predicted_view,
                    "predicted_view_confidence": predicted_view_confidence,
                    "target_view": args.target_view,
                    "target_view_confidence": target_view_confidence,
                    "view_hit": float(predicted_view == args.target_view) if predicted_view else 0.0,
                    "background_clutter_score": background_clutter,
                    "identity_similarity": identity_similarity,
                    "palette_distance": palette,
                    "copy_score": copied,
                    "copy_violation": bool(copy_flag),
                    "image_cfg_scale": args.image_cfg_scale,
                    "image_cfg_enabled": not args.disable_image_cfg,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                }
            )

        if not candidate_metrics:
            tqdm.write(f"No candidates produced for {reference_path}")
            continue

        best_idx = score_candidate_set(candidate_metrics, args)
        best_candidate = candidate_metrics[best_idx]

        selected_image = Image.open(best_candidate["candidate_path"]).convert("RGB")
        selected_path = output_dir / f"{sample_prefix}.jpg"
        selected_image.save(selected_path)

        manifest_records.append(
            {
                "index": sample_idx,
                "sheet_id": pair.get("sheet_id", "none"),
                "reference_path": str(Path(reference_path).resolve()),
                "target_path": str(Path(target_path).resolve()) if target_path else None,
                "generated_path": str(selected_path.resolve()),
                "requested_label": target_emotion,
                "ground_truth_target_emotion": target_emotion,
                "label_type": "expression",
                "seed": best_candidate["seed"],
                "ip_adapter_scale": args.scale,
                "image_cfg_scale": args.image_cfg_scale,
                "image_cfg_enabled": not args.disable_image_cfg,
                "text_guidance_scale": args.guidance,
                "prompt_style": args.prompt_style,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "baseline_type": "ip_adapter_p5_rerank",
                "model_tag": args.model_tag.strip() or Path(args.checkpoint_dir).name,
                "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
                "generation_mode": (
                    "reference_image_sampling_with_view_background_reranking"
                    if args.disable_image_cfg
                    else "reference_image_cfg_sampling_with_reranking"
                ),
                "num_candidates": args.num_candidates,
                "selected_candidate_index": best_candidate["candidate_index"],
                "selected_rerank_score": best_candidate["rerank_score"],
                "candidate_metrics": candidate_metrics,
            }
        )

    manifest_payload = {
        "summary": {
            "n_input_pairs": len(pairs),
            "n_generated": len(manifest_records),
            "num_candidates_per_pair": args.num_candidates,
            "image_cfg_scale": args.image_cfg_scale,
            "image_cfg_enabled": not args.disable_image_cfg,
            "target_view": args.target_view,
            "prompt_style": args.prompt_style,
            "prompt_suffix": args.prompt_suffix,
            "negative_prompt_extra": args.negative_prompt_extra,
            "skipped_invalid_label": skipped_invalid_label,
            "skipped_missing_reference": skipped_missing_reference,
            "skipped_target_labels": sorted(skipped_target_labels),
            "rerank_weights": {
                "w_expr_hit": args.w_expr_hit,
                "w_expr_conf": args.w_expr_conf,
                "w_id": args.w_id,
                "w_palette": args.w_palette,
                "w_copy": args.w_copy,
                "w_view_hit": args.w_view_hit,
                "w_view_conf": args.w_view_conf,
                "w_background": args.w_background,
                "copy_violation_penalty": args.copy_violation_penalty,
                "view_mismatch_penalty": args.view_mismatch_penalty,
            },
        },
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))

    print(f"Saved P5 reranked outputs to {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
