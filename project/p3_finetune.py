# train/p3_finetune.py
"""
P3: Expression-local Mask Loss Fine-tuning
==========================================
从已有的 IP-Adapter fine-tuned checkpoint 出发，
用表情区域加权 loss 继续训练。

不依赖 P1 代码，可独立运行。

用法：
  python train/p3_finetune.py \
      --image_proj_ckpt /ocean/projects/cis260099p/sliu45/project/checkpoints/image_proj_model.pt \
      --ip_attn_ckpt    /ocean/projects/cis260099p/sliu45/project/checkpoints/ip_attn_procs.pt \
      --train_json      data/pairs/train.json \
      --val_json        data/pairs/val.json \
      --output_dir      checkpoints/p3
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from torchvision import transforms

# IP-Adapter 仓库路径（clone 到项目根目录下）
sys.path.insert(0, str(Path(__file__).parent.parent / "IP-Adapter"))
from ip_adapter import IPAdapterPlus


# ── 数据集（自包含，不依赖 p1_dataset.py）────────────────────────────

EMOTION_TO_IDX = {
    "neutral": 0, "happy": 1, "sad": 2,
    "angry": 3, "surprised": 4, "crying": 5, "embarrassed": 6,
}

EMOTION_PROMPTS = {
    "neutral":     "anime character, neutral expression, 1girl",
    "happy":       "anime character, smiling, happy expression, 1girl",
    "sad":         "anime character, sad expression, teary eyes, 1girl",
    "angry":       "anime character, angry expression, frowning, 1girl",
    "surprised":   "anime character, surprised, wide eyes, 1girl",
    "crying":      "anime character, crying, tears streaming, 1girl",
    "embarrassed": "anime character, embarrassed, blushing, 1girl",
}


class ExpressionPairDataset(Dataset):
    def __init__(self, pairs_json: str, size: int = 512):
        with open(pairs_json) as f:
            raw = json.load(f)

        self.pairs = [
            p for p in raw
            if p.get("target_emotion", "unknown") in EMOTION_TO_IDX
            and Path(p["reference_path"]).exists()
            and Path(p["target_path"]).exists()
        ]
        print(f"  {pairs_json}: {len(raw)} total → {len(self.pairs)} usable pairs")

        self.size = size
        self.diffusion_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        ref = Image.open(pair["reference_path"]).convert("RGB")
        tgt = Image.open(pair["target_path"]).convert("RGB")
        emotion = pair["target_emotion"]
        return {
            "reference":   self.clip_transform(ref),
            "target":      self.diffusion_transform(tgt),
            "prompt":      EMOTION_PROMPTS[emotion],
            "emotion_idx": EMOTION_TO_IDX[emotion],
        }


# ── Soft Mask（不依赖关键点检测）────────────────────────────────────

def build_soft_mask(
    latent_size: int = 64,   # 512 图像对应 64×64 latent
    expr_weight: float = 3.0,
    sigma: float = 4.0,      # latent 空间 sigma，对应图像空间约 32px
    device: str = "cpu",
) -> torch.Tensor:
    """
    在 latent 空间构建表情区域软 mask。

    Anime 脸固定比例（从上到下）：
      15%–50%  眉毛 + 眼睛
      58%–82%  嘴巴

    返回: (1, 1, latent_size, latent_size), float32, 均值=1.0
    """
    h = w = latent_size
    mask = np.ones((h, w), dtype=np.float32)
    extra = np.zeros((h, w), dtype=np.float32)

    for top_r, bot_r in [(0.15, 0.50), (0.58, 0.82)]:
        top = int(h * top_r)
        bot = int(h * bot_r)
        extra[top:bot, :] = expr_weight - 1.0

    extra = gaussian_filter(extra, sigma=sigma)
    mask = mask + extra
    mask = mask / mask.mean()   # 均值归一化到 1.0，保持 loss 量级不变

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def masked_mse_loss(noise_pred, noise, mask):
    """mask: (1,1,H,W)，自动 broadcast 到 (B,C,H,W)"""
    return (((noise_pred.float() - noise.float()) ** 2) * mask).mean()


# ── 加载 checkpoint ───────────────────────────────────────────────────

def load_checkpoint(ip_model, unet, image_proj_path: str, ip_attn_path: str, device):
    """
    加载两个 .pt 文件：
      image_proj_model.pt  →  ip_model.image_proj_model
      ip_attn_procs.pt     →  unet 里的 to_k_ip / to_v_ip 参数
    """
    # --- image_proj_model ---
    proj_sd = torch.load(image_proj_path, map_location=device)
    ip_model.image_proj_model.load_state_dict(proj_sd, strict=True)
    print(f"  image_proj_model loaded ← {image_proj_path}")

    # --- ip_attn_procs ---
    # 两种常见保存格式都兼容：
    #   格式A: {full_param_name: tensor}   e.g. "down_blocks.0...to_k_ip.weight"
    #   格式B: {short_key: tensor}         e.g. "to_k_ip.weight"
    attn_sd = torch.load(ip_attn_path, map_location=device)

    matched = 0
    unet_sd = dict(unet.named_parameters())

    for ckpt_key, ckpt_val in attn_sd.items():
        if ckpt_key in unet_sd:
            # 格式 A：完整参数名直接匹配
            unet_sd[ckpt_key].data.copy_(ckpt_val.to(device))
            matched += 1
        else:
            # 格式 B：短 key，用后缀匹配
            for unet_name, unet_param in unet.named_parameters():
                if unet_name.endswith(ckpt_key) and ("to_k_ip" in unet_name or "to_v_ip" in unet_name):
                    unet_param.data.copy_(ckpt_val.to(device))
                    matched += 1
                    break

    total_ip_params = sum(1 for n in unet.named_parameters()
                          if "to_k_ip" in n[0] or "to_v_ip" in n[0])
    print(f"  ip_attn_procs loaded ← {ip_attn_path}")
    print(f"  matched {matched}/{len(attn_sd)} keys  "
          f"(unet has {total_ip_params} IP attn params total)")

    if matched == 0:
        raise RuntimeError(
            "No attn params matched! Check the keys in ip_attn_procs.pt:\n"
            f"  sample keys: {list(attn_sd.keys())[:5]}"
        )


def save_checkpoint(ip_model, unet, output_dir: str, step: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(ip_model.image_proj_model.state_dict(),
               out / "image_proj_model.pt")

    ip_attn = {
        name: param.detach().cpu()
        for name, param in unet.named_parameters()
        if "to_k_ip" in name or "to_v_ip" in name
    }
    torch.save(ip_attn, out / "ip_attn_procs.pt")

    # 带 step 的快照（方便回滚）
    torch.save(ip_model.image_proj_model.state_dict(),
               out / f"image_proj_model_step{step}.pt")
    torch.save(ip_attn, out / f"ip_attn_procs_step{step}.pt")

    print(f"  [ckpt] saved at step {step} → {output_dir}")


# ── 参数 ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # 模型
    p.add_argument("--base_model",       default="models/sd-v1-5")
    p.add_argument("--ip_ckpt",          default="models/ip-adapter/models/ip-adapter-plus_sd15.bin",
                   help="原始 IP-Adapter 权重，仅用于初始化模型结构")
    p.add_argument("--image_encoder",    default="models/ip-adapter/models/image_encoder")

    # 要加载的已有 checkpoint（同学 tune 好的）
    p.add_argument("--image_proj_ckpt",  required=True,
                   help="image_proj_model.pt 路径")
    p.add_argument("--ip_attn_ckpt",     required=True,
                   help="ip_attn_procs.pt 路径")

    # 数据
    p.add_argument("--train_json",       default="data/pairs/train.json")
    p.add_argument("--val_json",         default="data/pairs/val.json")

    # 输出
    p.add_argument("--output_dir",       default="checkpoints/p3")

    # 训练超参
    p.add_argument("--lr",               type=float, default=5e-5)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--num_steps",        type=int,   default=5000)
    p.add_argument("--save_every",       type=int,   default=500)
    p.add_argument("--log_every",        type=int,   default=50)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--image_size",       type=int,   default=512)

    # Mask 超参
    p.add_argument("--expr_weight",      type=float, default=3.0)
    p.add_argument("--mask_sigma",       type=float, default=4.0,
                   help="latent 空间高斯 sigma（图像空间约 sigma×8 px）")

    p.add_argument("--device",           default="cuda")
    return p.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Soft mask（只算一次）
    latent_size = args.image_size // 8   # 512 → 64
    soft_mask = build_soft_mask(
        latent_size=latent_size,
        expr_weight=args.expr_weight,
        sigma=args.mask_sigma,
        device=str(device),
    )
    print(f"Mask: shape={soft_mask.shape}, "
          f"min={soft_mask.min():.3f}, max={soft_mask.max():.3f}")

    # 2. 加载基础模型
    print("Loading base models...")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae"
    ).to(device, dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet"
    ).to(device, dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder
    ).to(device, dtype=torch.float16)
    ip_model = IPAdapterPlus(
        sd_pipe=None,
        image_encoder_path=args.image_encoder,
        ip_ckpt=args.ip_ckpt,   # 原始权重，结构初始化用
        device=device,
        num_tokens=16,
    )

    # 3. 覆盖加载同学 tune 好的权重
    print("Loading fine-tuned checkpoint...")
    load_checkpoint(ip_model, unet, args.image_proj_ckpt, args.ip_attn_ckpt, device)

    # 4. 冻结 / 解冻
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)

    ip_model.image_proj_model.requires_grad_(True)
    ip_model.image_proj_model.train()

    for name, param in unet.named_parameters():
        if "to_k_ip" in name or "to_v_ip" in name:
            param.requires_grad_(True)
    unet.train()

    trainable = (
        list(ip_model.image_proj_model.parameters()) +
        [p for n, p in unet.named_parameters() if "to_k_ip" in n or "to_v_ip" in n]
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )

    # 6. 数据
    train_ds = ExpressionPairDataset(args.train_json, size=args.image_size)
    val_ds   = ExpressionPairDataset(args.val_json,   size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)

    # 7. 保存配置
    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 8. 训练 loop
    print(f"\nStarting P3 training: {args.num_steps} steps, "
          f"lr={args.lr}, expr_weight={args.expr_weight}\n")

    step = 0
    running_loss = 0.0
    pbar = tqdm(total=args.num_steps, desc="P3")

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            ref_pixel    = batch["reference"].to(device, dtype=torch.float16)
            target_pixel = batch["target"].to(device, dtype=torch.float16)
            prompts      = batch["prompt"]

            with torch.no_grad():
                # VAE encode
                latents = vae.encode(target_pixel).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 加噪
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # CLIP encode
                image_embeds = image_encoder(ref_pixel).image_embeds  # (B, 1024)

                # Text encode
                text_input = ip_model.pipe.tokenizer(
                    prompts, padding="max_length",
                    max_length=77, return_tensors="pt"
                ).to(device)
                text_embeds = ip_model.pipe.text_encoder(
                    text_input.input_ids
                ).last_hidden_state  # (B, 77, 768)

            # IP projection（可训练）
            ip_tokens = ip_model.image_proj_model(image_embeds)  # (B, 16, 768)
            encoder_hidden = torch.cat([text_embeds, ip_tokens], dim=1)  # (B, 93, 768)

            # UNet 前向
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden
            ).sample  # (B, 4, 64, 64)

            # P3 核心：masked MSE
            loss = masked_mse_loss(noise_pred, noise, soft_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "lr":   f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })
                running_loss = 0.0

            if step % args.save_every == 0:
                save_checkpoint(ip_model, unet, args.output_dir, step)
                val_loss = compute_val_loss(
                    unet, vae, image_encoder, ip_model,
                    val_loader, noise_scheduler, soft_mask, device
                )
                tqdm.write(f"  step {step:5d} | val_loss={val_loss:.4f}")

    pbar.close()
    save_checkpoint(ip_model, unet, args.output_dir, step)
    print(f"\nDone. Saved to {args.output_dir}")


@torch.no_grad()
def compute_val_loss(unet, vae, image_encoder, ip_model,
                     val_loader, noise_scheduler, soft_mask, device,
                     max_batches: int = 20) -> float:
    unet.eval()
    ip_model.image_proj_model.eval()
    total, n = 0.0, 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ref_pixel    = batch["reference"].to(device, dtype=torch.float16)
        target_pixel = batch["target"].to(device, dtype=torch.float16)
        prompts      = batch["prompt"]

        latents = vae.encode(target_pixel).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        image_embeds = image_encoder(ref_pixel).image_embeds
        ip_tokens    = ip_model.image_proj_model(image_embeds)

        text_input = ip_model.pipe.tokenizer(
            prompts, padding="max_length", max_length=77, return_tensors="pt"
        ).to(device)
        text_embeds = ip_model.pipe.text_encoder(
            text_input.input_ids
        ).last_hidden_state

        encoder_hidden = torch.cat([text_embeds, ip_tokens], dim=1)
        noise_pred = unet(noisy_latents, timesteps,
                          encoder_hidden_states=encoder_hidden).sample

        total += masked_mse_loss(noise_pred, noise, soft_mask).item()
        n += 1

    unet.train()
    ip_model.image_proj_model.train()
    return total / max(n, 1)


if __name__ == "__main__":
    main()
