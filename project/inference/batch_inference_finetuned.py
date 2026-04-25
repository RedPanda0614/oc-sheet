# inference/batch_inference_finetuned.py
"""
用 fine-tuned checkpoint（P1/P3）跑 batch inference。
加载方式与 train/p3_finetune.py 完全一致。
manifest 格式与原版 batch_inference_labeled.py 一致，可直接喂给 run_eval.py。

用法：
  python inference/batch_inference_finetuned.py \
      --image_proj_ckpt checkpoints/p3/image_proj_model.pt \
      --ip_attn_ckpt    checkpoints/p3/ip_attn_procs.pt \
      --pairs_json      data/label_pairs/val.json \
      --output_dir      results/p3/batch_labeled
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline


EMOTION_PROMPTS = {
    "neutral":     "anime character portrait, neutral expression",
    "happy":       "anime character portrait, happy smiling expression",
    "sad":         "anime character portrait, sad expression",
    "angry":       "anime character portrait, angry frowning expression",
    "surprised":   "anime character portrait, surprised expression",
    "crying":      "anime character portrait, crying expression with tears",
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, extra fingers"


def parse_args():
    p = argparse.ArgumentParser()
    # Fine-tuned checkpoint
    p.add_argument("--image_proj_ckpt", required=True,
                   help="image_proj_model.pt 路径")
    p.add_argument("--ip_attn_ckpt",    required=True,
                   help="ip_attn_procs.pt 路径")

    # 模型路径（和原始脚本保持一致）
    p.add_argument("--sd_path",      default="models/sd-v1-5")
    p.add_argument("--ip_repo_path", default="models/ip-adapter")
    p.add_argument("--ip_weight",    default="ip-adapter-plus_sd15.bin")

    # 数据 & 输出
    p.add_argument("--pairs_json",   default="data/label_pairs/val.json")
    p.add_argument("--output_dir",   default="results/finetuned/batch_labeled")
    p.add_argument("--manifest_name",default="manifest.json")

    # 推理参数
    p.add_argument("--scale",        type=float, default=0.7)
    p.add_argument("--steps",        type=int,   default=30)
    p.add_argument("--guidance",     type=float, default=7.5)
    p.add_argument("--n",            type=int,   default=500,
                   help="最多处理多少条，0=全部")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def load_finetuned_pipeline(args, device):
    """
    加载 SD + IP-Adapter，然后覆盖 fine-tuned 权重。
    """
    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.load_ip_adapter(
        args.ip_repo_path,
        subfolder="models",
        weight_name=args.ip_weight,
    )
    pipe.set_ip_adapter_scale(args.scale)
    pipe = pipe.to(device)

    unet = pipe.unet

    # 加载 fine-tuned image_proj_model
    print(f"Loading image_proj ← {args.image_proj_ckpt}")
    proj_sd = torch.load(args.image_proj_ckpt, map_location=device)
    unet.encoder_hid_proj.load_state_dict(proj_sd, strict=True)

    # 加载 fine-tuned attn processors
    print(f"Loading ip_attn   ← {args.ip_attn_ckpt}")
    attn_sd = torch.load(args.ip_attn_ckpt, map_location="cpu")
    current_procs = unet.attn_processors
    matched = 0
    for name, proc_state in attn_sd.items():
        if name not in current_procs:
            continue
        cur = current_procs[name]
        state = {k: v.to(device=device, dtype=torch.float16)
                 for k, v in proc_state.items()}
        cur.load_state_dict(state)
        matched += 1
    print(f"  {matched}/{len(attn_sd)} attn processors loaded")

    pipe.unet.eval()
    return pipe


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    pipe = load_finetuned_pipeline(args, device)

    # 加载 pairs
    with open(args.pairs_json) as f:
        pairs = json.load(f)
    if args.n > 0:
        pairs = pairs[:args.n]

    valid_emotions = set(EMOTION_PROMPTS.keys())
    manifest_records = []
    skipped_invalid  = 0
    skipped_missing  = 0

    for i, pair in enumerate(tqdm(pairs, desc="finetuned-inference")):
        emotion  = pair.get("target_emotion")
        ref_path = pair.get("reference_path")

        if emotion not in valid_emotions:
            skipped_invalid += 1
            continue
        if not ref_path or not Path(ref_path).exists():
            skipped_missing += 1
            continue

        prompt    = EMOTION_PROMPTS[emotion]
        ref_image = Image.open(ref_path).convert("RGB")
        generator = torch.Generator(device=device).manual_seed(args.seed + i)

        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[ref_image],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
        except Exception as exc:
            tqdm.write(f"❌ Failed {ref_path}: {exc}")
            continue

        sheet_id  = pair.get("sheet_id", "none")
        save_name = f"{i:04d}_{sheet_id}_{emotion}.jpg"
        out_path  = os.path.join(args.output_dir, save_name)
        image.save(out_path)

        manifest_records.append({
            "index":                      i,
            "sheet_id":                   sheet_id,
            "reference_path":             ref_path,
            "target_path":                pair.get("target_path"),
            "generated_path":             out_path,
            "requested_label":            emotion,
            "ground_truth_target_emotion":emotion,
            "label_type":                 "expression",
            "seed":                       args.seed + i,
            "ip_adapter_scale":           args.scale,
            "generation_mode":            "finetuned",
            "image_proj_ckpt":            args.image_proj_ckpt,
            "ip_attn_ckpt":               args.ip_attn_ckpt,
        })

    # 保存 manifest
    manifest_path = os.path.join(args.output_dir, args.manifest_name)
    with open(manifest_path, "w") as f:
        json.dump({
            "summary": {
                "n_input_pairs":          len(pairs),
                "n_generated":            len(manifest_records),
                "skipped_invalid_label":  skipped_invalid,
                "skipped_missing_reference": skipped_missing,
            },
            "records": manifest_records,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(manifest_records)} images → {args.output_dir}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
