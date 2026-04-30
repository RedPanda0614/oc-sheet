"""
Batch inference for the IP-Adapter baseline using ground-truth target_emotion.

This script is the strict, label-driven version of batch_inference.py:
- it expects labeled pairs
- it skips samples with invalid or missing target_emotion
- it never falls back to a random emotion
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from run_baseline import EMOTION_PROMPTS, NEGATIVE_PROMPT, load_all_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/lora/pairs/val.json",
                        help="labeled validation set JSON with target_emotion")
    parser.add_argument("--output-dir", default="results/baseline/batch_labeled",
                        help="output directory")
    parser.add_argument("--manifest-name", default="manifest.json",
                        help="evaluation manifest JSON filename")
    parser.add_argument("--scale", type=float, default=0.7, help="IP-Adapter scale")
    parser.add_argument("--sd-path", default="models/sd-v1-5", help="SD base model path")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter",
                        help="IP-Adapter repo path")
    parser.add_argument("--n", type=int, default=500, help="number of samples (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, args.manifest_name)

    with open(args.pairs_json) as f:
        pairs = json.load(f)
    if args.n > 0:
        pairs = pairs[:args.n]

    pipe, _, device = load_all_models(args)
    valid_emotions = set(EMOTION_PROMPTS.keys())

    manifest_records = []
    skipped_invalid_label = 0
    skipped_missing_reference = 0

    for i, pair in enumerate(tqdm(pairs, desc="ip-adapter-labeled")):
        target_emotion = pair.get("target_emotion")
        ref_path = pair.get("reference_path")

        if target_emotion not in valid_emotions:
            skipped_invalid_label += 1
            continue
        if not ref_path or not os.path.exists(ref_path):
            skipped_missing_reference += 1
            continue

        prompt = EMOTION_PROMPTS[target_emotion]
        raw_image = Image.open(ref_path).convert("RGB")
        generator = torch.Generator(device=device).manual_seed(42 + i)

        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[raw_image],
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
        except Exception as exc:
            tqdm.write(f"Failed {ref_path}: {exc}")
            continue

        sheet_id = pair.get("sheet_id", "none")
        save_name = f"{i:04d}_{sheet_id}_{target_emotion}.jpg"
        output_path = os.path.join(args.output_dir, save_name)
        image.save(output_path)

        manifest_records.append(
            {
                "index": i,
                "sheet_id": sheet_id,
                "reference_path": ref_path,
                "target_path": pair.get("target_path"),
                "generated_path": output_path,
                "requested_label": target_emotion,
                "ground_truth_target_emotion": target_emotion,
                "label_type": "expression",
                "seed": 42 + i,
                "ip_adapter_scale": args.scale,
                "generation_mode": "strict_ground_truth_label",
            }
        )

    with open(manifest_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "n_input_pairs": len(pairs),
                    "n_generated": len(manifest_records),
                    "skipped_invalid_label": skipped_invalid_label,
                    "skipped_missing_reference": skipped_missing_reference,
                },
                "records": manifest_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved labeled IP-Adapter batch outputs to {args.output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
