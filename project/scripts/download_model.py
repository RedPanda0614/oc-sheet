import os
from huggingface_hub import hf_hub_download, snapshot_download

def setup_models():
    # 定义根目录
    base_path = "models"
    os.makedirs(base_path, exist_ok=True)

    # 1. 下载 Stable Diffusion v1.5 基础模型 (仅下载必需部分以节省空间)
    sd_path = os.path.join(base_path, "sd-v1-5")
    if not os.path.exists(sd_path):
        print("📥 正在下载 Stable Diffusion v1.5...")
        # snapshot_download 会下载整个仓库
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=sd_path,
            ignore_patterns=["*.msgpack", "*.ckpt"] # 优先用 diffusers 格式
        )
    else:
        print("✅ SD v1.5 已存在")

    # 2. 下载 IP-Adapter 权重
    ip_path = os.path.join(base_path, "ip-adapter", "models")
    os.makedirs(ip_path, exist_ok=True)
    
    ip_weight_file = "ip-adapter-plus_sd15.bin"
    if not os.path.exists(os.path.join(ip_path, ip_weight_file)):
        print(f"📥 正在下载 IP-Adapter Plus 权重: {ip_weight_file}...")
        hf_hub_download(
            repo_id="h94/IP-Adapter",
            subfolder="models",
            filename=ip_weight_file,
            local_dir=os.path.dirname(ip_path) # 保持其原始层级
        )
    else:
        print(f"✅ {ip_weight_file} 已存在")

    # 3. 下载图像编码器 (CLIP ViT-H/14)
    encoder_path = os.path.join(ip_path, "image_encoder")
    if not os.path.exists(encoder_path):
        print("📥 正在下载 CLIP 图像编码器...")
        snapshot_download(
            repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            local_dir=encoder_path
        )
    else:
        print("✅ 图像编码器已存在")

    print("\n✨ 所有模型文件已就绪！你可以运行 run_baseline.py 了。")

if __name__ == "__main__":
    setup_models()
