"""
batch_inference.py — Run IP-Adapter baseline over the val set.

For each unique reference in val.json, generates all 6 expression variants.
Saves results to --output-dir/<sheet_id>/<emotion>.jpg.
Writes a manifest: --output-dir/batch_manifest.json

Usage (from project root, on a GPU node):
  python inference/batch_inference.py
  python inference/batch_inference.py --val-json data/pairs/val.json \
      --output-dir results/baseline --n 200 --scale 0.7
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


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
    p.add_argument("--val-json",    default="data/pairs/val.json")
    p.add_argument("--output-dir",  default="results/baseline")
    p.add_argument("--n",           type=int,   default=200,
                   help="Max number of unique references to process")
    p.add_argument("--scale",       type=float, default=0.7)
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--guidance",    type=float, default=7.5)
    p.add_argument("--image-size",  type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--sd-path",     default="models/sd-v1-5")
    p.add_argument("--ip-adapter-path", default="models/ip-adapter")
    p.add_argument("--resume",      action="store_true",
                   help="Skip already-generated sheets")
    return p.parse_args()


def load_pipeline(sd_path, ip_adapter_path, scale):
    from diffusers import StableDiffusionPipeline

    print(f"Loading SD from {sd_path} …")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    pipe = pipe.to(device)
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


def generate_one(pipe, ref_image, prompt, steps, guidance, image_size, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        ip_adapter_image=ref_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=image_size,
        width=image_size,
        generator=generator,
    )
    return result.images[0]


def main():
    args = parse_args()
    val_path = Path(args.val_json)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        raise SystemExit(f"Val json not found: {val_path}")

    pairs = json.loads(val_path.read_text())
    print(f"Val set: {len(pairs)} pairs")

    # Deduplicate references (one reference → one set of outputs)
    seen_refs: dict[str, str] = {}  # reference_path → sheet_id
    for p in pairs:
        ref = p["reference_path"]
        if ref not in seen_refs:
            seen_refs[ref] = p["sheet_id"]

    refs = list(seen_refs.items())[:args.n]
    print(f"Processing {len(refs)} unique references (max={args.n})")

    pipe, device = load_pipeline(args.sd_path, args.ip_adapter_path, args.scale)

    manifest = {}
    n_ok, n_skip, n_err = 0, 0, 0

    for ref_path_str, sheet_id in tqdm(refs, desc="batch inference"):
        ref_path = Path(ref_path_str)
        sheet_out_dir = out_dir / sheet_id
        sheet_out_dir.mkdir(exist_ok=True)

        # Resume: skip if all emotions already done
        if args.resume:
            done = all(
                (sheet_out_dir / f"{emo}.jpg").exists()
                for emo in EMOTION_PROMPTS
            )
            if done:
                n_skip += 1
                continue

        if not ref_path.exists():
            print(f"  missing ref: {ref_path}")
            n_err += 1
            continue

        try:
            ref_image = Image.open(ref_path).convert("RGB")
        except Exception as e:
            print(f"  load error {ref_path}: {e}")
            n_err += 1
            continue

        sheet_results = {}
        for emotion, prompt in EMOTION_PROMPTS.items():
            out_path = sheet_out_dir / f"{emotion}.jpg"
            if args.resume and out_path.exists():
                sheet_results[emotion] = str(out_path)
                continue
            try:
                img = generate_one(pipe, ref_image, prompt,
                                   args.steps, args.guidance,
                                   args.image_size, args.seed)
                img.save(out_path, "JPEG", quality=95)
                sheet_results[emotion] = str(out_path)
            except Exception as e:
                print(f"  gen error {sheet_id}/{emotion}: {e}")

        manifest[sheet_id] = {
            "reference": ref_path_str,
            "generated": sheet_results,
        }
        n_ok += 1

    manifest_path = out_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Batch inference summary ===")
    print(f"  Processed: {n_ok}")
    print(f"  Skipped:   {n_skip} (already done)")
    print(f"  Errors:    {n_err}")
    print(f"  Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
