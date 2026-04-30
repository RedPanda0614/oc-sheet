# inference/generate_expressions.py
"""
给定一张 reference 图，用 fine-tuned IP-Adapter 生成所有表情变体。
结果拼成一张图保存。

用法：
  python inference/generate_expressions.py \
      --reference    path/to/your/face.jpg \
      --checkpoint   checkpoints/p3 \
      --output_dir   results/single_char
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline


EMOTIONS = {
    "neutral": (
        "solo, close-up face, anime portrait, neutral calm expression, "
        "half-lidded eyes, relaxed eyebrows, closed mouth, simple background"
    ),
    "happy": (
        "solo, close-up face, anime portrait, laughing happily, "
        "eyes curved into crescents, wide smile showing teeth, rosy cheeks, simple background"
    ),
    "sad": (
        "solo, close-up face, anime portrait, sad expression, "
        "drooping eyes, eyebrows slanted inward, downturned mouth, simple background"
    ),
    "angry": (
        "solo, close-up face, anime portrait, furious angry expression, "
        "eyes narrowed, eyebrows sharply angled down, clenched teeth, simple background"
    ),
    "surprised": (
        "solo, close-up face, anime portrait, shocked expression, "
        "eyes wide open and round, eyebrows raised high, open mouth, simple background"
    ),
    "crying": (
        "solo, close-up face, anime portrait, crying expression, "
        "tears on cheeks, closed eyes, scrunched brows, simple background"
    ),
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, extra fingers"

EMOTION_COLORS = {
    "neutral":     (160, 160, 160),
    "happy":       (80,  200, 80),
    "sad":         (80,  120, 220),
    "angry":       (220, 60,  60),
    "surprised":   (220, 180, 40),
    "crying":      (100, 160, 240),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference",   required=True,
                   help="Reference 图路径")
    p.add_argument("--checkpoint",  default=None,
                   help="Fine-tuned checkpoint 目录（含 image_proj_model.pt 和 ip_attn_procs.pt）。"
                        "不填则用原始 IP-Adapter zero-shot。")
    p.add_argument("--sd_path",     default="models/sd-v1-5")
    p.add_argument("--ip_repo",     default="models/ip-adapter")
    p.add_argument("--ip_weight",   default="ip-adapter-plus_sd15.bin")
    p.add_argument("--output_dir",  default="results/single_char")
    p.add_argument("--scale",       type=float, default=0.7)
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--guidance",    type=float, default=7.5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--size",        type=int,   default=512)
    return p.parse_args()


def load_pipeline(args, device):
    print("Loading SD + IP-Adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    pipe.load_ip_adapter(args.ip_repo, subfolder="models", weight_name=args.ip_weight)
    pipe.set_ip_adapter_scale(args.scale)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        proj_path = ckpt / "image_proj_model.pt"
        attn_path = ckpt / "ip_attn_procs.pt"

        print(f"Loading fine-tuned weights from {ckpt}...")
        pipe.unet.encoder_hid_proj.load_state_dict(
            torch.load(proj_path, map_location=device), strict=True
        )

        attn_sd = torch.load(attn_path, map_location="cpu")
        matched = 0
        for name, proc_state in attn_sd.items():
            if name not in pipe.unet.attn_processors:
                continue
            pipe.unet.attn_processors[name].load_state_dict(
                {k: v.to(device=device, dtype=torch.float16) for k, v in proc_state.items()}
            )
            matched += 1
        print(f"  {matched}/{len(attn_sd)} attn processors loaded")
    else:
        print("No checkpoint specified — running zero-shot.")

    return pipe


def make_sheet(images: dict[str, Image.Image], ref_img: Image.Image,
               size: int = 512) -> Image.Image:
    """把 reference + 所有生成图拼成一张大图。"""
    emotions = list(images.keys())
    n = len(emotions) + 1          # +1 是 reference
    cols = 4
    rows = (n + cols - 1) // cols
    bar_h = 24
    pad = 6

    cell_w = size
    cell_h = size + bar_h
    canvas_w = cols * (cell_w + pad) + pad
    canvas_h = rows * (cell_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    def paste_cell(img, label, color, idx):
        col = idx % cols
        row = idx // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        canvas.paste(img.resize((size, size)), (x, y))
        draw.rectangle([x, y + size, x + cell_w, y + size + bar_h], fill=color)
        bbox = draw.textbbox((0, 0), label)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell_w - tw) // 2, y + size + 4), label, fill=(255, 255, 255))

    # reference 放第一格
    paste_cell(ref_img, "REFERENCE", (60, 60, 60), 0)

    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, (120, 120, 120))
        paste_cell(images[emotion], emotion, color, i + 1)

    return canvas


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_img = Image.open(args.reference).convert("RGB")
    pipe    = load_pipeline(args, device)

    generated = {}
    for emotion, prompt in EMOTIONS.items():
        print(f"  generating: {emotion} ...", end=" ", flush=True)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=[ref_img],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            height=args.size,
            width=args.size,
        ).images[0]
        generated[emotion] = image

        # 单独保存每个表情
        save_path = Path(args.output_dir) / f"{emotion}.jpg"
        image.save(save_path)
        print(f"saved → {save_path}")

    # 拼成一张 sheet
    sheet = make_sheet(generated, ref_img, size=args.size)
    sheet_path = Path(args.output_dir) / "sheet.jpg"
    sheet.save(sheet_path, quality=95)
    print(f"\nSheet saved → {sheet_path}")


if __name__ == "__main__":
    main()
    