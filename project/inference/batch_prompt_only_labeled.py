"""
Batch inference for the prompt-only baseline using ground-truth target_emotion.

This is the strict, label-driven version of batch_prompt_only.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from run_baseline import EMOTION_PROMPTS, NEGATIVE_PROMPT
from run_prompt_only import build_naive_prompt, build_structured_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/lora/pairs/val.json", help="带 target_emotion 标签的验证集 JSON")
    parser.add_argument("--pretrained-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", default="results/prompt_only/batch_labeled", help="输出文件夹")
    parser.add_argument("--manifest-name", default="manifest.json", help="评估元数据 JSON 文件名")
    parser.add_argument("--mode", choices=["naive", "structured", "both"], default="both")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=500, help="测试数量，设为 0 表示跑完整个 JSON")
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

    valid_emotions = set(EMOTION_PROMPTS.keys())
    modes = ["naive", "structured"] if args.mode == "both" else [args.mode]

    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        manifest_records = []
        skipped_invalid_label = 0

        for i, pair in enumerate(tqdm(pairs, desc=f"prompt-only-labeled:{mode}")):
            target_emotion = pair.get("target_emotion")
            if target_emotion not in valid_emotions:
                skipped_invalid_label += 1
                continue

            prompt = (
                build_naive_prompt(target_emotion, args)
                if mode == "naive"
                else build_structured_prompt(target_emotion, args)
            )

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
                    "ground_truth_target_emotion": target_emotion,
                    "label_type": "expression",
                    "seed": args.seed + i,
                    "baseline_type": "prompt_only",
                    "prompt_mode": mode,
                    "prompt": prompt,
                    "generation_mode": "strict_ground_truth_label",
                }
            )

        with open(mode_dir / args.manifest_name, "w") as f:
            json.dump(
                {
                    "summary": {
                        "n_input_pairs": len(pairs),
                        "n_generated": len(manifest_records),
                        "skipped_invalid_label": skipped_invalid_label,
                    },
                    "records": manifest_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved labeled prompt-only batch outputs to {mode_dir}")


if __name__ == "__main__":
    main()
