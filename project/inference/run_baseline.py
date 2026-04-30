import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

os.environ["TORCH_SKIP_CHECK_SAFE_SERIALIZATION"] = "True"

EMOTION_PROMPTS = {
    "happy":       "manga character, smiling, happy expression, high quality",
    "sad":         "manga character, sad expression, teary eyes, high quality",
    "angry":       "manga character, angry expression, frowning, high quality",
    "surprised":   "manga character, surprised expression, wide eyes, high quality",
    "crying":      "manga character, crying, tears, high quality",
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, ugly"

def load_all_models(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc_path = os.path.join(args.ip_repo_path, "models", "image_encoder")
    print(f"Loading Image Encoder: {enc_path}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        enc_path, torch_dtype=torch.float16
    ).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained(enc_path)

    print(f"Loading SD Pipeline: {args.sd_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    print("Mounting IP-Adapter Plus...")
    pipe.load_ip_adapter(
        args.ip_repo_path,
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin"
    )
    pipe.set_ip_adapter_scale(args.scale)

    return pipe, feature_extractor, device

def compare_scales(pipe, device, reference_path, out_dir, scales=[0.3, 0.5, 0.7, 1.0]):
    """Generate comparison images at different IP-Adapter scales."""
    print(f"\nGenerating scale comparison images for scales: {scales}")
    comp_dir = Path(out_dir) / "scale_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    raw_image = Image.open(reference_path).convert("RGB")
    prompt = EMOTION_PROMPTS["happy"]

    for scale in scales:
        print(f"  > Testing scale={scale} ...")

        pipe.set_ip_adapter_scale(scale)

        generator = torch.Generator(device=device).manual_seed(42)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=[raw_image],
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        save_path = comp_dir / f"scale_{scale:.1f}.jpg"
        image.save(save_path, quality=95)
        print(f"    done")

    pipe.set_ip_adapter_scale(0.7)
    print(f"Scale comparison images saved to: {comp_dir}/\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--output-dir",  default="results/baseline")
    parser.add_argument("--scale",       type=float, default=0.7)
    parser.add_argument("--sd-path",     default="models/sd-v1-5")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter")
    parser.add_argument("--compare-scales", action="store_true",
                        help="Run IP-Adapter scale comparison test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe, feature_extractor, device = load_all_models(args)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Reference image not found: {ref_path}")
        return

    if args.compare_scales:
        compare_scales(pipe, device, ref_path, out_dir)
    raw_image = Image.open(ref_path).convert("RGB")

    print(f"Running inference: {ref_path.name}")

    for emotion, prompt in EMOTION_PROMPTS.items():
        print(f"  > Generating: {emotion}")

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=[raw_image],
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]

        save_path = out_dir / f"{ref_path.stem}_{emotion}.jpg"
        image.save(save_path)
        print(f"    Saved.")


if __name__ == "__main__":
    main()
