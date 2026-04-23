# train/p3_finetune.py
"""
P3: Expression-local Mask Loss Fine-tuning
==========================================
在 P1 checkpoint 基础上继续训练，用 build_soft_mask 对表情区域加权。

用法：
  python train/p3_finetune.py \
      --p1_ckpt checkpoints/p1 \
      --train_json data/pairs/train.json \
      --val_json   data/pairs/val.json \
      --output_dir checkpoints/p3

参数说明见 get_args()。
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)

# IP-Adapter 路径（与 p1_finetune.py 保持一致）
sys.path.insert(0, str(Path(__file__).parent.parent / "IP-Adapter"))
from ip_adapter import IPAdapterPlus

# 同目录的工具模块
sys.path.insert(0, str(Path(__file__).parent))
from p1_dataset import ExpressionPairDataset
from p3_mask import build_soft_mask, masked_mse_loss


# ── 参数 ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="P3: Masked MSE fine-tuning")

    # 模型路径
    p.add_argument("--base_model",    default="models/sd-v1-5",
                   help="SD 1.5 根目录")
    p.add_argument("--ip_ckpt",       default="models/ip-adapter/models/ip-adapter-plus_sd15.bin",
                   help="原始 IP-Adapter plus 权重（仅用于模型结构初始化）")
    p.add_argument("--image_encoder", default="models/ip-adapter/models/image_encoder",
                   help="CLIP ViT-H image encoder 目录")

    # P1 checkpoint（从这里 resume）
    p.add_argument("--p1_ckpt",       required=True,
                   help="P1 训练输出目录，包含 image_proj.pt 和 unet_ip.pt")

    # 数据
    p.add_argument("--train_json",    default="data/pairs/train.json")
    p.add_argument("--val_json",      default="data/pairs/val.json")

    # 输出
    p.add_argument("--output_dir",    default="checkpoints/p3")

    # 训练超参
    p.add_argument("--lr",            type=float, default=5e-5,
                   help="P3 比 P1 用更小的 lr，避免破坏已学的特征")
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--num_steps",     type=int,   default=5000,
                   help="P3 在 P1 基础上继续，步数可以少一些")
    p.add_argument("--save_every",    type=int,   default=500)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--num_workers",   type=int,   default=4)

    # Mask 参数
    p.add_argument("--expr_weight",   type=float, default=3.0,
                   help="表情区域（眉眼嘴）的 loss 权重倍数")
    p.add_argument("--mask_sigma",    type=float, default=30.0,
                   help="mask 边界高斯平滑半径（像素）")
    p.add_argument("--image_size",    type=int,   default=512)

    p.add_argument("--device",        default="cuda")

    return p.parse_args()


# ── 工具函数 ──────────────────────────────────────────────────────────

def load_p1_weights(ip_model, unet, p1_ckpt_dir: str, device):
    """
    从 P1 输出目录加载 image_proj 和 unet IP attn 权重。
    P1 保存格式：
      checkpoints/p1/
        ├── image_proj.pt      # ip_model.image_proj_model.state_dict()
        └── unet_ip.pt         # {name: param} for "to_k_ip" / "to_v_ip"
    """
    ckpt_dir = Path(p1_ckpt_dir)

    # 加载 image_proj_model
    proj_path = ckpt_dir / "image_proj.pt"
    if proj_path.exists():
        ip_model.image_proj_model.load_state_dict(
            torch.load(proj_path, map_location=device)
        )
        print(f"  image_proj loaded from {proj_path}")
    else:
        print(f"  [WARN] {proj_path} not found, using original IP-Adapter weights")

    # 加载 unet IP attn weights
    unet_ip_path = ckpt_dir / "unet_ip.pt"
    if unet_ip_path.exists():
        ip_state = torch.load(unet_ip_path, map_location=device)
        # 只加载 to_k_ip / to_v_ip，其余保持 SD1.5 原始权重
        missing = []
        for name, param in unet.named_parameters():
            if name in ip_state:
                param.data.copy_(ip_state[name].to(device))
            elif "to_k_ip" in name or "to_v_ip" in name:
                missing.append(name)
        if missing:
            print(f"  [WARN] {len(missing)} IP attn params not found in checkpoint")
        else:
            print(f"  unet_ip loaded from {unet_ip_path} ({len(ip_state)} params)")
    else:
        print(f"  [WARN] {unet_ip_path} not found, using original IP-Adapter weights")


def save_checkpoint(ip_model, unet, output_dir: str, step: int):
    """
    保存格式与 P1 完全一致，方便 P4 直接 resume。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(
        ip_model.image_proj_model.state_dict(),
        out / "image_proj.pt"
    )

    ip_state = {
        name: param.detach().cpu()
        for name, param in unet.named_parameters()
        if "to_k_ip" in name or "to_v_ip" in name
    }
    torch.save(ip_state, out / "unet_ip.pt")

    # 同时保存一份带 step 编号的快照
    torch.save(ip_model.image_proj_model.state_dict(), out / f"image_proj_step{step}.pt")
    torch.save(ip_state, out / f"unet_ip_step{step}.pt")

    print(f"  [ckpt] saved at step {step} → {output_dir}")


# ── 验证（可选，每隔 save_every 步跑一次）────────────────────────────

@torch.no_grad()
def run_val(unet, vae, image_encoder, ip_model, val_loader,
            soft_mask, device, max_batches=20):
    """
    在 val set 上计算 masked loss，用于监控过拟合。
    """
    unet.eval()
    ip_model.image_proj_model.eval()

    noise_scheduler_val = DDPMScheduler.from_pretrained(
        ip_model.pipe.config._name_or_path if hasattr(ip_model, 'pipe') else "models/sd-v1-5",
        subfolder="scheduler"
    )

    total_loss = 0.0
    n = 0
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
            0, noise_scheduler_val.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()
        noisy_latents = noise_scheduler_val.add_noise(latents, noise, timesteps)

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

        loss = masked_mse_loss(noise_pred, noise, soft_mask)
        total_loss += loss.item()
        n += 1

    unet.train()
    ip_model.image_proj_model.train()
    return total_loss / max(n, 1)


# ── 主训练流程 ────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. 构建 soft mask（只算一次，全程复用）──────────────────────
    print("Building expression mask...")
    soft_mask = build_soft_mask(
        size=args.image_size // 8,   # latent 空间是图像的 1/8
        expr_weight=args.expr_weight,
        sigma=args.mask_sigma / 8,   # sigma 也等比缩小
        device=str(device),
    )
    # soft_mask: (1, 1, 64, 64)，在 latent 空间对应 512 图像
    print(f"  mask shape: {soft_mask.shape}, "
          f"min={soft_mask.min():.3f}, max={soft_mask.max():.3f}, mean={soft_mask.mean():.3f}")

    # ── 2. 加载模型 ──────────────────────────────────────────────────
    print("Loading models...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )
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
        ip_ckpt=args.ip_ckpt,
        device=device,
        num_tokens=16,
    )

    # ── 3. 加载 P1 权重 ──────────────────────────────────────────────
    print(f"Loading P1 checkpoint from {args.p1_ckpt}...")
    load_p1_weights(ip_model, unet, args.p1_ckpt, device)

    # ── 4. 冻结 / 解冻（与 P1 完全一致）────────────────────────────
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    ip_model.image_proj_model.requires_grad_(True)
    ip_model.image_proj_model.train()

    for name, param in unet.named_parameters():
        if "to_k_ip" in name or "to_v_ip" in name:
            param.requires_grad_(True)
    unet.train()

    trainable = (
        list(ip_model.image_proj_model.parameters()) +
        [p for n, p in unet.named_parameters()
         if "to_k_ip" in n or "to_v_ip" in n]
    )
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_params:,}")

    # ── 5. Optimizer（P3 用更小 lr）─────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )

    # ── 6. 数据 ──────────────────────────────────────────────────────
    train_ds = ExpressionPairDataset(args.train_json, size=args.image_size)
    val_ds   = ExpressionPairDataset(args.val_json,   size=args.image_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    # ── 7. 保存训练配置 ──────────────────────────────────────────────
    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 8. 训练 loop ─────────────────────────────────────────────────
    print(f"\nStarting P3 training for {args.num_steps} steps...")
    print(f"  lr={args.lr}, batch={args.batch_size}, "
          f"expr_weight={args.expr_weight}, sigma={args.mask_sigma}\n")

    step = 0
    log_loss = 0.0
    clip_processor = CLIPImageProcessor.from_pretrained(args.image_encoder)

    pbar = tqdm(total=args.num_steps, desc="P3")

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            ref_pixel    = batch["reference"].to(device, dtype=torch.float16)
            target_pixel = batch["target"].to(device, dtype=torch.float16)
            prompts      = batch["prompt"]

            with torch.no_grad():
                # VAE encode → latent
                latents = vae.encode(target_pixel).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 加噪
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # CLIP encode 参考图
                image_embeds = image_encoder(ref_pixel).image_embeds  # (B, 1024)

            # IP-Adapter projection（可训练）
            ip_tokens = ip_model.image_proj_model(image_embeds)  # (B, 16, 768)

            # Text encode
            with torch.no_grad():
                text_input = ip_model.pipe.tokenizer(
                    prompts, padding="max_length",
                    max_length=77, return_tensors="pt"
                ).to(device)
                text_embeds = ip_model.pipe.text_encoder(
                    text_input.input_ids
                ).last_hidden_state  # (B, 77, 768)

            # 拼接 text + ip_tokens → encoder_hidden_states
            encoder_hidden = torch.cat([text_embeds, ip_tokens], dim=1)  # (B, 93, 768)

            # UNet 前向
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden
            ).sample  # (B, 4, 64, 64)

            # ── P3 核心：masked MSE loss ─────────────────────────────
            loss = masked_mse_loss(noise_pred, noise, soft_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            lr_scheduler.step()

            log_loss += loss.item()
            step += 1
            pbar.update(1)

            # 打印 loss
            if step % args.log_every == 0:
                avg_loss = log_loss / args.log_every
                pbar.set_postfix({"loss": f"{avg_loss:.4f}",
                                   "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
                log_loss = 0.0

            # 保存 checkpoint
            if step % args.save_every == 0:
                save_checkpoint(ip_model, unet, args.output_dir, step)

                # val loss
                val_loss = run_val(
                    unet, vae, image_encoder, ip_model,
                    val_loader, soft_mask, device
                )
                tqdm.write(f"  [step {step}] val_loss={val_loss:.4f}")

    pbar.close()

    # 最终保存
    save_checkpoint(ip_model, unet, args.output_dir, step)
    print(f"\nP3 training done. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
