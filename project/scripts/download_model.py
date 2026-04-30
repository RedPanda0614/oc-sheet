import os
from huggingface_hub import hf_hub_download, snapshot_download

def setup_models():
    base_path = "models"
    os.makedirs(base_path, exist_ok=True)

    # 1. Download Stable Diffusion v1.5
    sd_path = os.path.join(base_path, "sd-v1-5")
    if not os.path.exists(sd_path):
        print("Downloading Stable Diffusion v1.5...")
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=sd_path,
            ignore_patterns=["*.msgpack", "*.ckpt"]
        )
    else:
        print("SD v1.5 already exists.")

    # 2. Download IP-Adapter weights
    ip_path = os.path.join(base_path, "ip-adapter", "models")
    os.makedirs(ip_path, exist_ok=True)

    ip_weight_file = "ip-adapter-plus_sd15.bin"
    if not os.path.exists(os.path.join(ip_path, ip_weight_file)):
        print(f"Downloading IP-Adapter Plus: {ip_weight_file}...")
        hf_hub_download(
            repo_id="h94/IP-Adapter",
            subfolder="models",
            filename=ip_weight_file,
            local_dir=os.path.dirname(ip_path)
        )
    else:
        print(f"{ip_weight_file} already exists.")

    # 3. Download image encoder (CLIP ViT-H/14)
    encoder_path = os.path.join(ip_path, "image_encoder")
    if not os.path.exists(encoder_path):
        print("Downloading CLIP image encoder...")
        snapshot_download(
            repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            local_dir=encoder_path
        )
    else:
        print("Image encoder already exists.")

    print("\nAll model files ready. You can now run run_baseline.py.")

if __name__ == "__main__":
    setup_models()
