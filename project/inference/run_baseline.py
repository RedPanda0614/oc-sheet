import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# 屏蔽安全警告和损坏检查
os.environ["TORCH_SKIP_CHECK_SAFE_SERIALIZATION"] = "True"

EMOTION_PROMPTS = {
    "happy":       "manga character, smiling, happy expression, 1girl, high quality",
    "sad":         "manga character, sad expression, teary eyes, 1girl, high quality",
    "angry":       "manga character, angry expression, frowning, 1girl, high quality",
    "surprised":   "manga character, surprised expression, wide eyes, 1girl, high quality",
    "crying":      "manga character, crying, tears, 1girl, high quality",
    "embarrassed": "manga character, embarrassed, blushing, 1girl, high quality",
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, ugly"

def load_all_models(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载图像编码器和预处理器
    enc_path = os.path.join(args.ip_repo_path, "models", "image_encoder")
    print(f"📦 加载 Image Encoder: {enc_path}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        enc_path, torch_dtype=torch.float16
    ).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained(enc_path)

    # 2. 加载 SD Pipeline
    print(f"📦 加载 SD Pipeline: {args.sd_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # 3. 加载 IP-Adapter Plus
    print(f"⚓ 挂载 IP-Adapter Plus...")
    # 注意：这里直接传列表 [ref_image] 有时能解决元组报错
    pipe.load_ip_adapter(
        args.ip_repo_path, 
        subfolder="models", 
        weight_name="ip-adapter-plus_sd15.bin"
    )
    pipe.set_ip_adapter_scale(args.scale)
    
    return pipe, feature_extractor, device
def compare_scales(pipe, device, reference_path, out_dir, scales=[0.3, 0.5, 0.7, 1.0]):
    """
    对比不同 IP-Adapter scale 的效果，专为 Midway 报告生成对比图素材
    """
    print(f"\n📊 开始生成 Scale 对比图，测试范围: {scales}")
    comp_dir = Path(out_dir) / "scale_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    raw_image = Image.open(reference_path).convert("RGB")
    # 固定使用 happy 情绪作为测试基准
    prompt = EMOTION_PROMPTS["happy"]

    for scale in scales:
        print(f"  > 正在测试 Scale = {scale} ...")
        
        # 1. 动态设置当前循环的 scale
        pipe.set_ip_adapter_scale(scale)

        # 2. 生成图片（必须固定 seed，确保唯一的变量是 scale）
        generator = torch.Generator(device=device).manual_seed(42)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=[raw_image], # 继续使用安全的列表封装
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        # 3. 保存图片
        save_path = comp_dir / f"scale_{scale:.1f}.jpg"
        image.save(save_path, quality=95)
        print(f"    ✅ done")

    # 测试结束后，恢复默认的 scale (0.7)，以免影响后续其他任务
    pipe.set_ip_adapter_scale(0.7)
    print(f"✨ Scale 对比图已保存在: {comp_dir}/\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--output-dir",  default="results/baseline")
    parser.add_argument("--scale",       type=float, default=0.7)
    parser.add_argument("--sd-path",     default="models/sd-v1-5")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter")
    parser.add_argument("--compare-scales", action="store_true", help="是否运行 Scale 权重对比测试")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe, feature_extractor, device = load_all_models(args)
    
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ 找不到图片: {ref_path}")
        return

    if args.compare_scales:
        compare_scales(pipe, device, ref_path, out_dir)
        # 如果只想跑对比图，跑完可以直接 return，或者继续往下跑基础表情测试
    raw_image = Image.open(ref_path).convert("RGB")

    print(f"🚀 开始推理: {ref_path.name}")

    for emotion, prompt in EMOTION_PROMPTS.items():
        print(f"  > 生成中: {emotion}")
        
        # 核心修正：使用列表包装原始 PIL 图片
        # 在最新版 diffusers 中，对于 Plus 模型，传入 [PIL.Image] 
        # 会触发正确的内部编码流程，避免产生空的元组分量
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=[raw_image], # 重点：中括号包装单张图片
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]

        save_path = out_dir / f"{ref_path.stem}_{emotion}.jpg"
        image.save(save_path)
        print(f"    ✅ 已保存")


if __name__ == "__main__":
    main()
