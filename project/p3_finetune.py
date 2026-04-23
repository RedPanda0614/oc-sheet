# train/p3_finetune.py
"""
P3: Expression-local Mask Loss Fine-tuning
==========================================
使用 diffusers 原生 IP-Adapter 接口（适配同学用 diffusers 训练的 checkpoint）。

用法：
  python train/p3_finetune.py \
      --image_proj_ckpt /ocean/projects/cis260099p/sliu45/project/checkpoints/p1/image_proj_model.pt \
      --ip_attn_ckpt    /ocean/projects/cis260099p/sliu45/project/checkpoints/p1/ip_attn_procs.pt \
      --train_json      data/label_pairs/train.json \
      --val_json        data/label_pairs/val.json \
      --output_dir      checkpoints/p3
"""

import os
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
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, DDPMScheduler


# ── 数据集 ────────────────────────────────────────────────────────────

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
        self.diffusion_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        ref = Image.open(pair["reference_path"]).convert("RGB")
        tgt = Image.open(pair["target_path"]).convert("RGB")
        emotion = pair["target_emotion"]
        return {
            "reference":   self.clip_tf(ref),
            "target":      self.diffusion_tf(tgt),
            "prompt":      EMOTION_PROMPTS[emotion],
            "emotion_idx": EMOTION_TO_IDX[emotion],
        }


# ── Soft Mask ─────────────────────────────────────────────────────────

def build_soft_mask(latent_size=64, expr_weight=3.0, sigma=4.0, device="cpu"):
    """
    在 latent 空间构建表情区域软 mask（anime 脸固定比例，无需关键点）。
    返回: (1, 1, latent_size, latent_size), float32, 均值=1.0
    """
    h = w = latent_size
    mask = np.ones((h, w), dtype=np.float32)
    extra = np.zeros((h, w), dtype=np.float32)
    for top_r, bot_r in [(0.15, 0.50), (0.58, 0.82)]:
        extra[int(h * top_r):int(h * bot_r), :] = expr_weight - 1.0
    extra = gaussian_filter(extra, sigma=sigma)
    mask = (mask + extra) / (mask + extra).mean()
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def masked_mse_loss(noise_pred, noise, mask):
    return (((noise_pred.float() - noise.float()) ** 2) * mask).mean()


# ── 加载 checkpoint ───────────────────────────────────────────────────

def load_checkpoint(unet, image_proj_ckpt: str, ip_attn_ckpt: str, device):
    """
    加载 diffusers 风格的 checkpoint：
      image_proj_model.pt  → unet.encoder_hid_proj
      ip_attn_procs.pt     → {path: processor_object}，用 set_attn_processor 加载
    """
    # image_proj
    proj_sd = torch.load(image_proj_ckpt, map_location=device)
    unet.encoder_hid_proj.load_state_dict(proj_sd, strict=True)
    print(f"  encoder_hid_proj loaded ← {image_proj_ckpt}")

    # IP attn processors（只存了 IP 专用的 16 个，合并进当前全部 36 个里）
    saved_procs = torch.load(ip_attn_ckpt, map_location="cpu")
    current_procs = unet.attn_processors   # 引用，直接修改即生效
    matched = 0
    for key, saved_proc in saved_procs.items():
        if key in current_procs:
            cur = current_procs[key]
            if isinstance(saved_proc, dict):
                # 存的是 state_dict（普通 dict of tensors）
                state = {k: v.to(device=device, dtype=torch.float16)
                         for k, v in saved_proc.items()}
            else:
                # 存的是 processor 对象
                state = {k: v.to(device=device, dtype=torch.float16)
                         for k, v in saved_proc.state_dict().items()}
            cur.load_state_dict(state)
            matched += 1
    print(f"  ip_attn_procs loaded ← {ip_attn_ckpt}  "
          f"({matched}/{len(saved_procs)} processors matched)")


def save_checkpoint(unet, output_dir: str, step: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # image_proj
    torch.save(unet.encoder_hid_proj.state_dict(), out / "image_proj_model.pt")

    # 只保存 IP 专用 processor（与 P1 格式一致，key 里含 "ip" 的）
    ip_procs = {k: v for k, v in unet.attn_processors.items()
                if hasattr(v, "to_k_ip") or hasattr(v, "to_v_ip")}
    torch.save(ip_procs, out / "ip_attn_procs.pt")

    # 带 step 快照
    torch.save(unet.encoder_hid_proj.state_dict(), out / f"image_proj_model_step{step}.pt")
    torch.save(ip_procs, out / f"ip_attn_procs_step{step}.pt")
    print(f"  [ckpt] saved at step {step} → {output_dir}")


# ── 参数 ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",      default="models/sd-v1-5")
    p.add_argument("--ip_adapter_dir",  default="models/ip-adapter",
                   help="包含 models/ip-adapter-plus_sd15.bin 的本地目录")
    p.add_argument("--image_encoder",   default="models/ip-adapter/models/image_encoder")

    # 已有 checkpoint
    p.add_argument("--image_proj_ckpt", required=True)
    p.add_argument("--ip_attn_ckpt",    required=True)

    # 数据
    p.add_argument("--train_json",      default="data/label_pairs/train.json")
    p.add_argument("--val_json",        default="data/label_pairs/val.json")

    # 输出
    p.add_argument("--output_dir",      default="checkpoints/p3")

    # 训练超参
    p.add_argument("--lr",              type=float, default=5e-5)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_steps",       type=int,   default=5000)
    p.add_argument("--save_every",      type=int,   default=500)
    p.add_argument("--log_every",       type=int,   default=50)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--image_size",      type=int,   default=512)

    # Mask 超参
    p.add_argument("--expr_weight",     type=float, default=3.0)
    p.add_argument("--mask_sigma",      type=float, default=4.0)

    p.add_argument("--device",          default="cuda")
    return p.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Soft mask（latent 空间，只算一次）
    latent_size = args.image_size // 8
    soft_mask = build_soft_mask(latent_size, args.expr_weight, args.mask_sigma, str(device))
    print(f"Mask: {soft_mask.shape}, min={soft_mask.min():.3f}, max={soft_mask.max():.3f}")

    # 2. 加载 SD pipeline（包含 tokenizer / text_encoder / vae / unet）
    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to(device)
    vae       = pipe.vae
    unet      = pipe.unet
    tokenizer = pipe.tokenizer
    text_enc  = pipe.text_encoder

    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # 3. 加载 IP-Adapter 结构（diffusers 原生接口）
    #    这会在 unet 里注册 encoder_hid_proj 和 IP attn processors
    print("Loading IP-Adapter structure...")
    pipe.load_ip_adapter(
        args.ip_adapter_dir,
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
    )

    # 4. 覆盖加载同学的 fine-tuned 权重
    print("Loading fine-tuned checkpoint...")
    load_checkpoint(unet, args.image_proj_ckpt, args.ip_attn_ckpt, device)

    # 5. 加载 CLIP image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder
    ).to(device, dtype=torch.float16)

    # 6. 冻结 / 解冻
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 只训练 encoder_hid_proj 和 IP attn 参数
    unet.encoder_hid_proj.requires_grad_(True)
    unet.encoder_hid_proj.train()
    for name, param in unet.named_parameters():
        if "to_k_ip" in name or "to_v_ip" in name:
            param.requires_grad_(True)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # 7. Optimizer
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    # 8. 数据
    train_ds = ExpressionPairDataset(args.train_json, args.image_size)
    val_ds   = ExpressionPairDataset(args.val_json,   args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 9. 训练 loop
    print(f"\nP3 training: {args.num_steps} steps, lr={args.lr}, expr_weight={args.expr_weight}\n")
    step, running_loss = 0, 0.0
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
                latents = vae.encode(target_pixel).latent_dist.sample() * vae.config.scaling_factor

                # 加噪
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text encode
                text_ids = tokenizer(
                    prompts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                text_embeds = text_enc(text_ids).last_hidden_state  # (B, 77, 768)

                # CLIP image encode（raw features，UNet 内部会过 encoder_hid_proj）
                image_embeds = image_encoder(ref_pixel).image_embeds  # (B, 1024)

            # UNet 前向（diffusers 原生 IP-Adapter 接口）
            # encoder_hid_proj 在这里被调用（可训练），attn proc 也是（可训练）
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeds,
                added_cond_kwargs={"image_embeds": image_embeds},
            ).sample  # (B, 4, 64, 64)

            # P3 核心：masked MSE
            loss = masked_mse_loss(noise_pred, noise, soft_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            lr_sched.step()

            running_loss += loss.item()
            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                pbar.set_postfix({"loss": f"{avg:.4f}",
                                  "lr": f"{lr_sched.get_last_lr()[0]:.2e}"})
                running_loss = 0.0

            if step % args.save_every == 0:
                save_checkpoint(unet, args.output_dir, step)
                val_loss = compute_val_loss(unet, vae, text_enc, tokenizer,
                                            image_encoder, val_loader,
                                            noise_scheduler, soft_mask, device)
                tqdm.write(f"  step {step:5d} | val_loss={val_loss:.4f}")

    pbar.close()
    save_checkpoint(unet, args.output_dir, step)
    print(f"\nDone. Saved to {args.output_dir}")


@torch.no_grad()
def compute_val_loss(unet, vae, text_enc, tokenizer, image_encoder,
                     val_loader, noise_scheduler, soft_mask, device,
                     max_batches=20):
    unet.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ref_pixel    = batch["reference"].to(device, dtype=torch.float16)
        target_pixel = batch["target"].to(device, dtype=torch.float16)
        prompts      = batch["prompt"]

        latents = vae.encode(target_pixel).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        text_ids = tokenizer(prompts, padding="max_length", max_length=77,
                             truncation=True, return_tensors="pt").input_ids.to(device)
        text_embeds  = text_enc(text_ids).last_hidden_state
        image_embeds = image_encoder(ref_pixel).image_embeds

        noise_pred = unet(noisy_latents, timesteps,
                          encoder_hidden_states=text_embeds,
                          added_cond_kwargs={"image_embeds": image_embeds}).sample
        total += masked_mse_loss(noise_pred, noise, soft_mask).item()
        n += 1

    unet.train()
    return total / max(n, 1)


if __name__ == "__main__":
    main()
