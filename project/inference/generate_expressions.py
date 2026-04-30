# inference/generate_expressions.py
"""
Given a reference image, generate all expression variants using a fine-tuned
IP-Adapter and stitch results into a single sheet.

Usage (single, no rerank):
  python inference/generate_expressions.py \
      --reference  path/to/your/face.jpg \
      --checkpoint checkpoints/p4 \
      --output_dir results/single_char

Usage (P4 + P5 rerank):
  python inference/generate_expressions.py \
      --reference     path/to/your/face.jpg \
      --checkpoint    checkpoints/p4 \
      --output_dir    results/single_char_p5 \
      --n_candidates  4
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline

_INFERENCE_DIR = Path(__file__).resolve().parent
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

from batch_inference_p5_rerank_labeled import (          # noqa: E402
    normalize_metric,
    score_candidate_set,
)

PROJECT_ROOT = _INFERENCE_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from arcface_similarity import ArcFaceEvaluator        # noqa: E402
from copy_score import copy_score, copy_violation      # noqa: E402
from expression_classifier import CLIPControlEvaluator  # noqa: E402
from palette_distance import palette_distance          # noqa: E402
# ─────────────────────────────────────────────────────────────────────────────


EMOTIONS = {
    "neutral": (
        "anime portrait, neutral calm face, "
        "half-lidded eyes, relaxed eyebrows, lips closed, white background"
    ),
    "happy": (
        "anime portrait, laughing happily, "
        "eyes curved into crescents upward, wide open smile showing teeth, rosy cheeks, white background"
    ),
    "sad": (
        "anime portrait, sad grieving face, "
        "eyes drooping downward at corners, eyebrows slanted inward, pursed trembling lips, white background"
    ),
    "angry": (
        "anime portrait, furious rage expression, "
        "eyes narrowed to slits, thick eyebrows sharply angled down toward nose, jaw clenched, white background"
    ),
    "surprised": (
        "anime portrait, utterly shocked face, "
        "eyes perfectly round and wide open, eyebrows raised as high as possible, mouth dropped open, white background"
    ),
    "crying": (
        "anime portrait, weeping face, "
        "eyes tightly shut with tears flowing down cheeks, large teardrops, eyebrows raised and scrunched, white background"
    ),
}

EMOTION_SEEDS = {
    "neutral":   42,
    "happy":     123,
    "sad":       456,
    "angry":     789,
    "surprised": 1024,
    "crying":    2048,
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, extra fingers"

EMOTION_COLORS = {
    "neutral":     (160, 160, 160),
    "happy":       (80,  200, 80),
    "sad":         (80,  120, 220),
    "angry":       (220, 60,  60),
    "surprised":   (220, 180, 40),
    "crying":      (100, 160, 240),
    "embarrassed": (220, 100, 180),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference",   required=True,
                   help="reference image path")
    p.add_argument("--checkpoint",  default=None,
                   help="fine-tuned checkpoint dir (image_proj_model.pt + ip_attn_procs.pt); "
                        "omit to run zero-shot")
    p.add_argument("--sd_path",     default="models/sd-v1-5")
    p.add_argument("--ip_repo",     default="models/ip-adapter")
    p.add_argument("--ip_weight",   default="ip-adapter-plus_sd15.bin")
    p.add_argument("--output_dir",  default="results/single_char")
    p.add_argument("--scale",       type=float, default=0.7)
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--guidance",    type=float, default=7.5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--size",        type=int,   default=512)
    # P5 rerank
    p.add_argument("--n_candidates",  type=int,   default=1,
                   help="candidates per emotion; 1 = no rerank")
    p.add_argument("--copy_threshold", type=float, default=0.88,
                   help="copy_violation threshold, consistent with p5_rerank")
    p.add_argument("--w_expr_hit",    type=float, default=3.0)
    p.add_argument("--w_expr_conf",   type=float, default=1.0)
    p.add_argument("--w_id",          type=float, default=1.0)
    p.add_argument("--w_palette",     type=float, default=0.75)
    p.add_argument("--w_copy",        type=float, default=0.75)
    p.add_argument("--copy_violation_penalty", type=float, default=2.0)
    return p.parse_args()


def load_pipeline(args, device):
    print("Loading SD + IP-Adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    pipe.load_ip_adapter(args.ip_repo, subfolder="models",
                         weight_name=args.ip_weight)
    pipe.set_ip_adapter_scale(args.scale)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        proj_path = ckpt / "image_proj_model.pt"
        attn_path = ckpt / "ip_attn_procs.pt"

        print(f"Loading fine-tuned weights from {ckpt}...")
        pipe.unet.encoder_hid_proj.load_state_dict(
            torch.load(proj_path, map_location=device), strict=True
        )

        attn_sd = torch.load(attn_path, map_location="cpu")
        matched = 0
        for name, proc_state in attn_sd.items():
            if name not in pipe.unet.attn_processors:
                continue
            pipe.unet.attn_processors[name].load_state_dict(
                {k: v.to(device=device, dtype=torch.float16)
                 for k, v in proc_state.items()}
            )
            matched += 1
        print(f"  {matched}/{len(attn_sd)} attn processors loaded")
    else:
        print("No checkpoint specified — running zero-shot.")

    return pipe


def score_candidates(
    ref_path: str,
    emotion: str,
    candidates: list[Image.Image],
    args,
    tmp_dir: Path,
    identity_eval: "ArcFaceEvaluator",
    control_eval: "CLIPControlEvaluator",
) -> int:
    """Score candidates and return the index of the best one."""
    candidate_metrics = []

    for k, img in enumerate(candidates):
        cand_path = tmp_dir / f"cand_{k:02d}.jpg"
        img.save(cand_path)

        # expression
        try:
            prediction = control_eval.predict(
                str(cand_path), label_type="expression")
            predicted_label = prediction.label
            prediction_conf = prediction.confidence
        except Exception as e:
            print(f"    [warn] expression eval failed for cand {k}: {e}")
            predicted_label = None
            prediction_conf = None

        # identity
        try:
            id_sim = identity_eval.similarity(ref_path, str(cand_path))
        except Exception as e:
            print(f"    [warn] identity eval failed for cand {k}: {e}")
            id_sim = None

        # palette
        try:
            pal = palette_distance(ref_path, str(cand_path))
        except Exception as e:
            print(f"    [warn] palette eval failed for cand {k}: {e}")
            pal = None

        # copy
        try:
            copied = copy_score(ref_path, str(cand_path))
            copy_flag = copy_violation(copied, threshold=args.copy_threshold)
        except Exception as e:
            print(f"    [warn] copy eval failed for cand {k}: {e}")
            copied = None
            copy_flag = False

        candidate_metrics.append({
            "candidate_index":      k,
            "predicted_label":      predicted_label,
            "prediction_confidence": prediction_conf,
            "expression_hit":       float(predicted_label == emotion) if predicted_label else 0.0,
            "identity_similarity":  id_sim,
            "palette_distance":     pal,
            "copy_score":           copied,
            "copy_violation":       bool(copy_flag),
        })

    best_idx = score_candidate_set(candidate_metrics, args)

    for m in candidate_metrics:
        marker = "★" if m["candidate_index"] == best_idx else " "
        print(
            f"    {marker} cand{m['candidate_index']} "
            f"expr={m['expression_hit']:.0f}({m['prediction_confidence'] or 0:.2f}) "
            f"id={m['identity_similarity'] or 0:.3f} "
            f"pal={m['palette_distance'] or 0:.3f} "
            f"copy={m['copy_score'] or 0:.3f} "
            f"→ rerank={m.get('rerank_score', 0):.3f}"
        )

    return best_idx


def make_sheet(images: dict[str, Image.Image], ref_img: Image.Image,
               size: int = 512) -> Image.Image:
    """Stitch reference and all generated images into one sheet."""
    emotions = list(images.keys())
    n = len(emotions) + 1
    cols = 4
    rows = (n + cols - 1) // cols
    bar_h = 24
    pad = 6

    cell_w = size
    cell_h = size + bar_h
    canvas_w = cols * (cell_w + pad) + pad
    canvas_h = rows * (cell_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    def paste_cell(img, label, color, idx):
        col = idx % cols
        row = idx // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        canvas.paste(img.resize((size, size)), (x, y))
        draw.rectangle([x, y + size, x + cell_w, y + size + bar_h], fill=color)
        bbox = draw.textbbox((0, 0), label)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell_w - tw) // 2, y + size + 4),
                  label, fill=(255, 255, 255))

    paste_cell(ref_img, "REFERENCE", (60, 60, 60), 0)
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, (120, 120, 120))
        paste_cell(images[emotion], emotion, color, i + 1)

    return canvas


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_img = Image.open(args.reference).convert("RGB")
    ref_path = str(Path(args.reference).resolve())
    pipe = load_pipeline(args, device)

    use_rerank = args.n_candidates > 1
    identity_eval = control_eval = None
    if use_rerank:
        print("Initialising eval models for P5 rerank...")
        identity_eval = ArcFaceEvaluator()
        control_eval = CLIPControlEvaluator()

    generated = {}
    for emotion, prompt in EMOTIONS.items():
        base_seed = EMOTION_SEEDS.get(emotion, args.seed)

        if use_rerank:
            print(f"  [{emotion}] generating {args.n_candidates} candidates ...")
            candidates = []
            for k in range(args.n_candidates):
                gen = torch.Generator(device=device).manual_seed(
                    base_seed + k * 1000)
                img = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    ip_adapter_image=[ref_img],
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                    height=args.size,
                    width=args.size,
                ).images[0]
                candidates.append(img)

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                best_idx = score_candidates(
                    ref_path, emotion, candidates, args,
                    tmp_dir, identity_eval, control_eval,
                )
            image = candidates[best_idx]
            print(f"  [{emotion}] selected cand {best_idx}")

            cand_dir = Path(args.output_dir) / f"{emotion}_candidates"
            cand_dir.mkdir(exist_ok=True)
            for k, cand in enumerate(candidates):
                marker = "best_" if k == best_idx else ""
                cand.save(cand_dir / f"{marker}cand{k:02d}.jpg")
        else:
            print(f"  generating: {emotion} ...", end=" ", flush=True)
            gen = torch.Generator(device=device).manual_seed(base_seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[ref_img],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=gen,
                height=args.size,
                width=args.size,
            ).images[0]

        generated[emotion] = image
        save_path = Path(args.output_dir) / f"{emotion}.jpg"
        image.save(save_path)
        print(f"saved → {save_path}")

    sheet = make_sheet(generated, ref_img, size=args.size)
    sheet_path = Path(args.output_dir) / "sheet.jpg"
    sheet.save(sheet_path, quality=95)
    print(f"\nSheet saved → {sheet_path}")


if __name__ == "__main__":
    main()
