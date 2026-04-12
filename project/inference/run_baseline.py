"""
run_baseline.py — Zero-shot IP-Adapter inference on a single reference image.

Generates 6 expression variants (happy / sad / angry / surprised / crying /
embarrassed) using ip-adapter-plus_sd15.bin + stable-diffusion-v1-5.

Model paths (relative to project root):
  models/sd-v1-5/                     ← runwayml/stable-diffusion-v1-5
  models/ip-adapter/models/
      ip-adapter-plus_sd15.bin
      image_encoder/                  ← CLIP ViT-H/14

Usage (from project root, on a GPU node):
  python inference/run_baseline.py path/to/reference.jpg
  python inference/run_baseline.py path/to/reference.jpg \
      --output-dir results/baseline --scale 0.7 --steps 30
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


EMOTION_PROMPTS = {
    "happy":       "manga character, smiling, happy expression, 1girl, high quality",
    "sad":         "manga character, sad expression, teary eyes, 1girl, high quality",
    "angry":       "manga character, angry expression, frowning, 1girl, high quality",
    "surprised":   "manga character, surprised expression, wide eyes, 1girl, high quality",
    "crying":      "manga character, crying, tears, 1girl, high quality",
    "embarrassed": "manga character, embarrassed, blushing, 1girl, high quality",
}

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, worst quality, blurry, "
    "deformed, extra fingers, ugly, text, watermark"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("reference", help="Path to reference image")
    p.add_argument("--output-dir",  default="results/baseline")
    p.add_argument("--scale",       type=float, default=0.7,
                   help="IP-Adapter scale (0.3=text-dominant … 1.0=identity-lock)")
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--guidance",    type=float, default=7.5)
    p.add_argument("--image-size",  type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--sd-path",     default="models/sd-v1-5")
    p.add_argument("--ip-adapter-path", default="models/ip-adapter")
    p.add_argument("--emotions",    nargs="+",  default=list(EMOTION_PROMPTS.keys()),
                   help="Subset of emotions to generate")
    return p.parse_args()


def load_pipeline(sd_path: str, ip_adapter_path: str, scale: float):
    from diffusers import StableDiffusionPipeline

    print(f"Loading SD pipeline from {sd_path} …")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    pipe = pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    print(f"Loading IP-Adapter from {ip_adapter_path} …")
    pipe.load_ip_adapter(
        ip_adapter_path,
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
    )
    pipe.set_ip_adapter_scale(scale)

    return pipe, device


def generate(pipe, ref_image: Image.Image, prompt: str, args) -> Image.Image:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        ip_adapter_image=ref_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.image_size,
        width=args.image_size,
        generator=generator,
    )
    return result.images[0]


def main():
    args = parse_args()
    ref_path = Path(args.reference)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ref_path.exists():
        raise SystemExit(f"Reference image not found: {ref_path}")

    # Validate emotion list
    invalid = [e for e in args.emotions if e not in EMOTION_PROMPTS]
    if invalid:
        raise SystemExit(f"Unknown emotions: {invalid}. "
                         f"Valid: {list(EMOTION_PROMPTS.keys())}")

    ref_image = Image.open(ref_path).convert("RGB")
    print(f"Reference: {ref_path}  ({ref_image.size})")

    pipe, device = load_pipeline(args.sd_path, args.ip_adapter_path, args.scale)

    stem = ref_path.stem
    generated = {}

    for emotion in args.emotions:
        prompt = EMOTION_PROMPTS[emotion]
        print(f"  Generating: {emotion} …")
        img = generate(pipe, ref_image, prompt, args)
        out_path = out_dir / f"{stem}_{emotion}.jpg"
        img.save(out_path, "JPEG", quality=95)
        generated[emotion] = str(out_path)
        print(f"    → {out_path}")

    print(f"\nGenerated {len(generated)} images in {out_dir}/")
    return generated


if __name__ == "__main__":
    main()
