import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

# ── 配置：情绪 Prompt 映射 ─────────────────────────────────────
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
    p.add_argument("reference", help="参考人脸图片的路径")
    p.add_argument("--output-dir",  default="results/baseline")
    p.add_argument("--scale",       type=float, default=0.7, help="参考图强度 (0.5-0.8)")
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--sd-path",     default="models/sd-v1-5")
    # 注意：这里的默认路径指向你克隆的 IP-Adapter 文件夹
    p.add_argument("--ip-repo-path", default="models/ip-adapter")
    return p.parse_args()

def load_all_models(args):
    """根据你的项目结构加载模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 显式加载你本地的 Image Encoder (ViT-H/14)
    # 路径通常在 IP-Adapter/models/image_encoder
    image_encoder_path = os.path.join(args.ip_repo_path, "models", "image_encoder")
    print(f"📦 正在加载 Image Encoder: {image_encoder_path}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_path,
        torch_dtype=torch.float16
    ).to(device)

    # 2. 加载基础 SD v1.5
    print(f"📦 正在加载 SD Pipeline: {args.sd_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # 3. 加载 IP-Adapter 权重
    # 路径在 IP-Adapter/models/ip-adapter-plus_sd15.bin
    print(f"⚓ 正在挂载 IP-Adapter Plus 权重...")
    pipe.load_ip_adapter(
        os.path.join(args.ip_repo_path, "models"), 
        weight_name="ip-adapter-plus_sd15.bin"
    )
    pipe.set_ip_adapter_scale(args.scale)

    # 内存优化
    pipe.enable_attention_slicing()
    
    return pipe, device

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    pipe, device = load_all_models(args)

    # 处理参考图
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ 找不到图片: {ref_path}")
        return
    ref_image = Image.open(ref_path).convert("RGB")

    print(f"🚀 开始生成角色 {ref_path.stem} 的表情变体...")

    for emotion, prompt in EMOTION_PROMPTS.items():
        print(f"  > 生成中: {emotion}")
        
        # 生成图片
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=ref_image,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]

        # 保存结果
        save_path = out_dir / f"{ref_path.stem}_{emotion}.jpg"
        image.save(save_path, quality=95)
        print(f"    ✅ 已保存: {save_path}")

    print(f"\n✨ 完成！结果保存在: {out_dir}")

if __name__ == "__main__":
    main()
