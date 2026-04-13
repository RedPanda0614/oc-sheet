"""
Batch inference for the global LoRA expression-control baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from run_baseline import EMOTION_PROMPTS, NEGATIVE_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/pairs/val.json", help="验证集 JSON 路径")
    parser.add_argument("--pretrained-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", default="results/lora_global/batch", help="输出文件夹")
    parser.add_argument("--manifest-name", default="manifest.json", help="保存评估元数据的 JSON 文件名")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=500, help="测试数量，设为 0 表示跑完整个 JSON")
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
    pipe.load_lora_weights(args.lora_dir)

    available_emotions = list(EMOTION_PROMPTS.keys())
    manifest_records = []

    for i, pair in enumerate(tqdm(pairs, desc="lora-global")):
        target_emotion = pair.get("target_emotion", "unknown")
        if target_emotion not in available_emotions:
            target_emotion = random.choice(available_emotions)

        prompt = EMOTION_PROMPTS[target_emotion]
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
        output_path = out_dir / save_name
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
                "baseline_type": "lora_global",
                "prompt": prompt,
            }
        )

    with open(out_dir / args.manifest_name, "w") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    print(f"Saved LoRA batch outputs to {out_dir}")


if __name__ == "__main__":
    main()
