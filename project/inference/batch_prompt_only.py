"""
Batch inference for the prompt-only baseline.

Supports two prompt settings:
1. naive: exact prompts from run_baseline.py
2. structured: same prompts plus character attributes
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from run_baseline import EMOTION_PROMPTS, NEGATIVE_PROMPT
from run_prompt_only import build_naive_prompt, build_structured_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/pairs/val.json",
                        help="validation set JSON path")
    parser.add_argument("--pretrained-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", default="results/prompt_only/batch",
                        help="output directory")
    parser.add_argument("--manifest-name", default="manifest.json",
                        help="evaluation manifest JSON filename")
    parser.add_argument("--mode", choices=["naive", "structured", "both"], default="both")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=500, help="number of samples (0 = all)")

    # Character attributes for structured prompts
    parser.add_argument("--hair-color", default="blue hair")
    parser.add_argument("--hair-style", default="long hair")
    parser.add_argument("--eye-color", default="red eyes")
    parser.add_argument("--outfit", default="school uniform")
    parser.add_argument("--accessories", default="")
    parser.add_argument("--style-tags", default="manga style, clean lineart, high quality")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    with open(args.pairs_json) as f:
        pairs = json.load(f)
    if args.n > 0:
        pairs = pairs[:args.n]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    available_emotions = list(EMOTION_PROMPTS.keys())
    modes = ["naive", "structured"] if args.mode == "both" else [args.mode]

    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        manifest_records = []

        for i, pair in enumerate(tqdm(pairs, desc=f"prompt-only:{mode}")):
            target_emotion = pair.get("target_emotion", "unknown")
            if target_emotion not in available_emotions:
                target_emotion = random.choice(available_emotions)

            if mode == "naive":
                prompt = build_naive_prompt(target_emotion, args)
            else:
                prompt = build_structured_prompt(target_emotion, args)

            generator = torch.Generator(device=device).manual_seed(args.seed + i)
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]

            sheet_id = pair.get("sheet_id", "none")
            save_name = f"{i:04d}_{sheet_id}_{target_emotion}.jpg"
            output_path = mode_dir / save_name
            image.save(output_path)

            manifest_records.append(
                {
                    "index": i,
                    "sheet_id": sheet_id,
                    "reference_path": pair.get("reference_path"),
                    "target_path": pair.get("target_path"),
                    "generated_path": str(output_path),
                    "requested_label": target_emotion,
                    "label_type": "expression",
                    "seed": args.seed + i,
                    "baseline_type": "prompt_only",
                    "prompt_mode": mode,
                    "prompt": prompt,
                }
            )

        with open(mode_dir / args.manifest_name, "w") as f:
            json.dump(manifest_records, f, indent=2, ensure_ascii=False)

        print(f"Saved prompt-only batch outputs to {mode_dir}")


if __name__ == "__main__":
    main()
