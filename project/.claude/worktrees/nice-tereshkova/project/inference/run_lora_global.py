"""
Inference script for global LoRA expression-control.

Usage — generate all emotions for a reference image:
  python inference/run_lora_global.py \
      --pretrained-model models/sd-v1-5 \
      --lora-dir results/lora_global \
      --reference path/to/ref.jpg \
      --output-dir results/lora_global/outputs

Usage — generate a single emotion:
  python inference/run_lora_global.py ... --emotion happy

How it works
------------
1. The reference image is used as the init_image for img2img diffusion.
   This preserves the character's identity (face structure, style).
2. The LoRA weights + emotion prompt drive the expression change.
3. `--strength` (0–1) controls the tradeoff:
     low  (~0.5) → stays close to reference, subtle expression change
     high (~0.8) → more expressive but may drift from reference identity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


EMOTION_PROMPTS = {
    "neutral": "neutral expression, calm face",
    "happy": "happy smiling expression, joyful",
    "sad": "sad expression, sorrowful",
    "angry": "angry frowning expression, furious",
    "surprised": "surprised expression, wide eyes",
    "crying": "crying expression with tears, weeping",
}

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, ugly, "
    "extra fingers, mutation, poorly drawn face, missing limbs"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="models/sd-v1-5")
    p.add_argument("--lora-dir", required=True, help="Directory containing LoRA attn_procs weights")
    p.add_argument("--reference", required=True, help="Reference image for character identity")
    p.add_argument("--output-dir", default="results/lora_global/outputs")
    p.add_argument(
        "--emotion",
        default=None,
        choices=list(EMOTION_PROMPTS.keys()),
        help="Generate a single emotion (default: all)",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument(
        "--strength",
        type=float,
        default=0.65,
        help="img2img strength (0–1). Lower = more identity; higher = more expression change.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-prompt", default=None, help="Override base prompt (reads from lora-dir by default)")
    p.add_argument("--resolution", type=int, default=512)
    return p.parse_args()


def load_base_prompt(lora_dir: Path, override: str | None) -> str:
    if override:
        return override
    prompt_file = lora_dir / "prompt_template.txt"
    if prompt_file.exists():
        return prompt_file.read_text().strip()
    meta_file = lora_dir / "train_meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text()).get("base_prompt", "anime character portrait")
    return "anime character portrait"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = Path(args.lora_dir)

    # Load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # Load LoRA attention processors
    pipe.unet.load_attn_procs(str(lora_dir))
    print(f"Loaded LoRA from {lora_dir}")

    # Load and preprocess reference image
    ref_image = Image.open(args.reference).convert("RGB").resize(
        (args.resolution, args.resolution), Image.LANCZOS
    )
    ref_image.save(out_dir / "reference.jpg")

    base_prompt = load_base_prompt(lora_dir, args.base_prompt)

    # Determine which emotions to generate
    emotions_to_run = (
        {args.emotion: EMOTION_PROMPTS[args.emotion]}
        if args.emotion
        else EMOTION_PROMPTS
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating {len(emotions_to_run)} emotion(s) | "
          f"strength={args.strength} | steps={args.steps} | guidance={args.guidance}")

    for emotion, emo_desc in emotions_to_run.items():
        prompt = f"{base_prompt}, {emo_desc}"
        result = pipe(
            prompt=prompt,
            image=ref_image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=NEGATIVE_PROMPT,
            generator=generator,
        ).images[0]

        out_path = out_dir / f"lora_global_{emotion}.jpg"
        result.save(out_path)
        print(f"  [{emotion}] → {out_path}")

    print(f"\nDone. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
