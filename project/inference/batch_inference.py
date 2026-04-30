import json
import os
import random
import argparse
import torch
from tqdm import tqdm
from PIL import Image

from run_baseline import load_all_models, EMOTION_PROMPTS, NEGATIVE_PROMPT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/pairs/val.json",
                        help="validation set JSON path")
    parser.add_argument("--output-dir", default="results/baseline/batch",
                        help="output directory")
    parser.add_argument("--manifest-name", default="manifest.json",
                        help="evaluation manifest JSON filename")
    parser.add_argument("--scale", type=float, default=0.7,
                        help="IP-Adapter scale")
    parser.add_argument("--sd-path", default="models/sd-v1-5",
                        help="SD base model path")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter",
                        help="IP-Adapter repo path")
    parser.add_argument("--n", type=int, default=500,
                        help="number of samples (0 = all)")
    return parser.parse_args()

def batch_generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, args.manifest_name)

    print(f"Loading pairs: {args.pairs_json}")
    with open(args.pairs_json) as f:
        pairs = json.load(f)

    if args.n > 0:
        pairs = pairs[:args.n]

    pipe, feature_extractor, device = load_all_models(args)

    available_emotions = list(EMOTION_PROMPTS.keys())
    manifest_records = []

    print(f"\nBatch inference: {len(pairs)} images...\n")

    for i, pair in enumerate(tqdm(pairs)):
        ref_path = pair["reference_path"]

        target_emotion = pair.get("target_emotion", "unknown")
        if target_emotion not in available_emotions:
            target_emotion = random.choice(available_emotions)

        prompt = EMOTION_PROMPTS[target_emotion]

        if not os.path.exists(ref_path):
            tqdm.write(f"Reference not found: {ref_path}")
            continue

        try:
            raw_image = Image.open(ref_path).convert("RGB")

            generator = torch.Generator(device=device).manual_seed(42 + i)

            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[raw_image],
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]

            sheet_id = pair.get("sheet_id", "none")
            save_name = f"{i:04d}_{sheet_id}_{target_emotion}.jpg"
            output_path = os.path.join(args.output_dir, save_name)
            image.save(output_path)

            manifest_records.append({
                "index": i,
                "sheet_id": sheet_id,
                "reference_path": ref_path,
                "target_path": pair.get("target_path"),
                "generated_path": output_path,
                "requested_label": target_emotion,
                "label_type": "expression",
                "seed": 42 + i,
                "ip_adapter_scale": args.scale,
            })

        except Exception as e:
            tqdm.write(f"Failed {ref_path}: {e}")

    with open(manifest_path, "w") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    print(f"\nBatch complete. Saved to: {args.output_dir}")
    print(f"Manifest saved: {manifest_path}")

if __name__ == "__main__":
    args = parse_args()
    batch_generate(args)
