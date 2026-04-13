"""
Prompt-only baseline inference.

This baseline compares:
1. naive prompt-only generation
2. structured prompt-only generation with character attributes

No reference image is used during generation. This isolates the effect of prompt
design on expression control and coarse identity consistency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


EMOTION_PHRASES = {
    "neutral": "neutral expression",
    "happy": "happy smiling expression",
    "sad": "sad expression",
    "angry": "angry frowning expression",
    "surprised": "surprised expression",
    "crying": "crying expression with tears",
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, ugly"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output-dir", default="results/prompt_only")
    p.add_argument("--mode", choices=["naive", "structured", "both"], default="both")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)

    # Character attributes for structured prompts
    p.add_argument("--subject", default="anime character portrait")
    p.add_argument("--gender-token", default="1girl")
    p.add_argument("--hair-color", default="blue hair")
    p.add_argument("--hair-style", default="long hair")
    p.add_argument("--eye-color", default="red eyes")
    p.add_argument("--outfit", default="school uniform")
    p.add_argument("--accessories", default="")
    p.add_argument("--style-tags", default="manga style, clean lineart, high quality")
    return p.parse_args()


def build_naive_prompt(emotion: str, args) -> str:
    return f"{args.subject}, {args.gender_token}, {EMOTION_PHRASES[emotion]}, high quality"


def build_structured_prompt(emotion: str, args) -> str:
    parts = [
        args.subject,
        args.gender_token,
        args.hair_color,
        args.hair_style,
        args.eye_color,
        args.outfit,
        EMOTION_PHRASES[emotion],
        args.style_tags,
    ]
    if args.accessories.strip():
        parts.insert(6, args.accessories)
    return ", ".join([part for part in parts if part])


def save_prompts(prompts_by_mode: dict[str, dict[str, str]], output_dir: Path):
    (output_dir / "prompts.json").write_text(json.dumps(prompts_by_mode, indent=2))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    prompts_by_mode: dict[str, dict[str, str]] = {}

    modes = ["naive", "structured"] if args.mode == "both" else [args.mode]
    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        prompts_by_mode[mode] = {}

        for emotion in EMOTION_PHRASES:
            if mode == "naive":
                prompt = build_naive_prompt(emotion, args)
            else:
                prompt = build_structured_prompt(emotion, args)

            prompts_by_mode[mode][emotion] = prompt

            generator = torch.Generator(device=device).manual_seed(args.seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
            image.save(mode_dir / f"{emotion}.jpg")
            print(f"[{mode}] saved {emotion}.jpg")

    save_prompts(prompts_by_mode, out_dir)
    print(f"Saved prompt-only comparison to {out_dir}")


if __name__ == "__main__":
    main()
