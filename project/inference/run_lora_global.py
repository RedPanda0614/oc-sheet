"""
Inference script for global LoRA expression-control baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


EMOTION_PROMPTS = {
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
    p.add_argument("--pretrained-model", default="models/sd-v1-5")
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--reference", required=True, help="Reference image (optional for identity check)")
    p.add_argument("--output-dir", default="results/lora_global/outputs")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-prompt", default="anime character portrait, 1girl")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.unet.load_attn_procs(args.lora_dir)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for emotion, emo_prompt in EMOTION_PROMPTS.items():
        prompt = f"{args.base_prompt}, {emo_prompt}"
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        image.save(out_dir / f"lora_global_{emotion}.jpg")

    # Save reference for side-by-side comparison
    ref = Image.open(args.reference).convert("RGB")
    ref.save(out_dir / "reference.jpg")


if __name__ == "__main__":
    main()
