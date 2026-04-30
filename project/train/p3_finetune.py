# train/p3_finetune.py
"""
P3: Expression-local Mask Loss Fine-tuning
  python scripts/p3_finetune.py \
      --image_proj_ckpt /ocean/projects/cis260099p/sliu45/project/results/ip_adapter_finetune/image_proj_model.pt \
      --ip_attn_ckpt    /ocean/projects/cis260099p/sliu45/project/results/ip_adapter_finetune/ip_attn_procs.pt \
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
from diffusers import StableDiffusionPipeline, DDPMScheduler


EMOTION_PROMPTS = {
    "neutral":     "neutral expression",
    "happy":       "happy smiling expression",
    "sad":         "sad expression",
    "angry":       "angry frowning expression",
    "surprised":   "surprised expression",
    "crying":      "crying expression with tears",
    "embarrassed": "embarrassed blushing expression",
}

BASE_PROMPT = "anime character portrait, 1girl"


class ExpressionPairDataset(Dataset):
    def __init__(self, pairs_json: str, size: int = 512):
        with open(pairs_json) as f:
            raw = json.load(f)
        self.pairs = [
            p for p in raw
            if p.get("target_emotion") in EMOTION_PROMPTS
            and Path(p["reference_path"]).exists()
            and Path(p["target_path"]).exists()
        ]
        print(f"  {pairs_json}: {len(raw)} total → {len(self.pairs)} usable pairs")
        self.size = size
        self.target_tf = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair  = self.pairs[idx]
        ref   = Image.open(pair["reference_path"]).convert("RGB")
        tgt   = Image.open(pair["target_path"]).convert("RGB")
        emotion = pair["target_emotion"]
        prompt = f"{BASE_PROMPT}, {EMOTION_PROMPTS[emotion]}"
        return {
            "reference_pil":  ref,
            "target_pixels":  self.target_tf(tgt),
            "prompt":         prompt,
        }


def collate_fn(batch):
    return {
        "reference_pil": [b["reference_pil"] for b in batch],
        "target_pixels": torch.stack([b["target_pixels"] for b in batch]),
        "prompts":       [b["prompt"] for b in batch],
    }


def build_soft_mask(latent_size=64, expr_weight=3.0, sigma=4.0, device="cpu"):
    """Returns (1, 1, latent_size, latent_size), float32, mean=1.0"""
    h = latent_size
    mask  = np.ones((h, h), dtype=np.float32)
    extra = np.zeros((h, h), dtype=np.float32)
    for top_r, bot_r in [(0.15, 0.50), (0.58, 0.82)]:
        extra[int(h * top_r):int(h * bot_r), :] = expr_weight - 1.0
    extra = gaussian_filter(extra, sigma=sigma)
    mask  = (mask + extra)
    mask  = mask / mask.mean()
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def masked_mse_loss(noise_pred, noise, mask):
    return (((noise_pred.float() - noise.float()) ** 2) * mask).mean()


def load_p1_checkpoint(unet, image_proj_ckpt: str, ip_attn_ckpt: str, device):
    """
    Load P1 checkpoint weights:
      image_proj_model.pt  → unet.encoder_hid_proj.state_dict()
      ip_attn_procs.pt     → {attn_name: proc_state_dict}
    """
    proj_sd = torch.load(image_proj_ckpt, map_location=device)
    unet.encoder_hid_proj.load_state_dict(proj_sd, strict=True)
    print(f"  encoder_hid_proj ← {image_proj_ckpt}")

    attn_sd = torch.load(ip_attn_ckpt, map_location="cpu")
    current_procs = unet.attn_processors
    matched = 0
    for name, proc_state in attn_sd.items():
        if name not in current_procs:
            continue
        cur_proc = current_procs[name]
        state = {k: v.to(device=device, dtype=torch.float16)
                 for k, v in proc_state.items()}
        cur_proc.load_state_dict(state)
        matched += 1
    print(f"  ip_attn_procs    ← {ip_attn_ckpt}  ({matched}/{len(attn_sd)} matched)")
    if matched == 0:
        raise RuntimeError(
            f"No processors matched.\n"
            f"  checkpoint keys (sample): {list(attn_sd.keys())[:3]}\n"
            f"  unet proc keys  (sample): {list(current_procs.keys())[:3]}"
        )


def save_checkpoint(unet, output_dir: str, step: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(unet.encoder_hid_proj.state_dict(), out / "image_proj_model.pt")

    attn_state = {}
    for name, proc in unet.attn_processors.items():
        if not hasattr(proc, "state_dict"):
            continue
        sd = proc.state_dict()
        if not sd:
            continue
        if not any(k.startswith(("to_k_ip", "to_v_ip")) for k in sd):
            continue
        attn_state[name] = {k: v.detach().cpu() for k, v in sd.items()}
    torch.save(attn_state, out / "ip_attn_procs.pt")

    torch.save(unet.encoder_hid_proj.state_dict(), out / f"image_proj_model_step{step}.pt")
    torch.save(attn_state, out / f"ip_attn_procs_step{step}.pt")
    print(f"  [ckpt] step {step} → {output_dir}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model",  default="models/sd-v1-5")
    p.add_argument("--ip_repo_path",      default="models/ip-adapter",
                   help="local ip-adapter directory containing models/ip-adapter-plus_sd15.bin")
    p.add_argument("--ip_weight",         default="ip-adapter-plus_sd15.bin")

    p.add_argument("--image_proj_ckpt",   required=True)
    p.add_argument("--ip_attn_ckpt",      required=True)

    p.add_argument("--train_json",        default="data/label_pairs/train.json")
    p.add_argument("--val_json",          default="data/label_pairs/val.json")
    p.add_argument("--output_dir",        default="checkpoints/p3")

    p.add_argument("--lr",                type=float, default=5e-5)
    p.add_argument("--batch_size",        type=int,   default=2)
    p.add_argument("--num_steps",         type=int,   default=5000)
    p.add_argument("--save_every",        type=int,   default=500)
    p.add_argument("--log_every",         type=int,   default=50)
    p.add_argument("--num_workers",       type=int,   default=2)
    p.add_argument("--image_size",        type=int,   default=512)
    p.add_argument("--ip_scale",          type=float, default=0.7)

    p.add_argument("--expr_weight",       type=float, default=3.0)
    p.add_argument("--mask_sigma",        type=float, default=4.0)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    # 1. Soft mask (latent space, computed once)
    latent_size = args.image_size // 8
    soft_mask = build_soft_mask(latent_size, args.expr_weight, args.mask_sigma, device)
    print(f"Mask: {soft_mask.shape}, "
          f"min={soft_mask.min():.3f} max={soft_mask.max():.3f} mean={soft_mask.mean():.3f}")

    # 2. Load pipeline
    print("Loading SD pipeline + IP-Adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16 if use_amp else torch.float32,
        safety_checker=None,
    )
    pipe.load_ip_adapter(
        args.ip_repo_path,
        subfolder="models",
        weight_name=args.ip_weight,
    )
    pipe.set_ip_adapter_scale(args.ip_scale)

    tokenizer    = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae          = pipe.vae
    unet         = pipe.unet
    noise_sched  = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.to(device)

    # 3. Load P1 weights
    print("Loading P1 checkpoint...")
    load_p1_checkpoint(unet, args.image_proj_ckpt, args.ip_attn_ckpt, device)

    # 4. Freeze / unfreeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    trainable = []
    if hasattr(unet, "encoder_hid_proj") and unet.encoder_hid_proj is not None:
        unet.encoder_hid_proj.float()
        unet.encoder_hid_proj.requires_grad_(True)
        unet.encoder_hid_proj.train()
        trainable += list(unet.encoder_hid_proj.parameters())

    for proc in unet.attn_processors.values():
        if hasattr(proc, "float"):    proc.float()
        if hasattr(proc, "requires_grad_"):
            try: proc.requires_grad_(True)
            except: pass
        if hasattr(proc, "parameters"):
            trainable += list(proc.parameters())

    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # 5. Optimizer + AMP scaler
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 6. Data
    train_ds = ExpressionPairDataset(args.train_json, args.image_size)
    val_ds   = ExpressionPairDataset(args.val_json,   args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 7. Training loop
    print(f"\nP3 training: {args.num_steps} steps, "
          f"lr={args.lr}, expr_weight={args.expr_weight}\n")

    unet.train()
    step, running_loss = 0, 0.0
    pbar = tqdm(total=args.num_steps, desc="P3")

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            target_pixels = batch["target_pixels"].to(device=device, dtype=vae.dtype)
            prompts       = batch["prompts"]
            ref_pils      = batch["reference_pil"]

            with torch.no_grad():
                latents = vae.encode(target_pixels).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                text_ids = tokenizer(
                    prompts, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt"
                ).input_ids.to(device)
                text_embeds = text_encoder(text_ids).last_hidden_state

                image_embeds = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[ref_pils],
                    ip_adapter_image_embeds=None,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={"image_embeds": image_embeds},
                ).sample

                loss = masked_mse_loss(noise_pred, noise, soft_mask)

            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
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
                val_loss = compute_val_loss(pipe, unet, vae, text_encoder, tokenizer,
                                            noise_sched, val_loader, soft_mask,
                                            device, use_amp)
                tqdm.write(f"  step {step:5d} | val_loss={val_loss:.4f}")

    pbar.close()
    save_checkpoint(unet, args.output_dir, step)
    print(f"\nDone. Saved to {args.output_dir}")


@torch.no_grad()
def compute_val_loss(pipe, unet, vae, text_encoder, tokenizer,
                     noise_sched, val_loader, soft_mask,
                     device, use_amp, max_batches=20):
    unet.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        target_pixels = batch["target_pixels"].to(device=device, dtype=vae.dtype)
        prompts       = batch["prompts"]
        ref_pils      = batch["reference_pil"]

        latents = vae.encode(target_pixels).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_sched.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

        text_ids = tokenizer(prompts, padding="max_length", truncation=True,
                             max_length=tokenizer.model_max_length,
                             return_tensors="pt").input_ids.to(device)
        text_embeds  = text_encoder(text_ids).last_hidden_state
        image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=[ref_pils],
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            noise_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states=text_embeds,
                              added_cond_kwargs={"image_embeds": image_embeds}).sample
            total += masked_mse_loss(noise_pred, noise, soft_mask).item()
        n += 1

    unet.train()
    return total / max(n, 1)


if __name__ == "__main__":
    main()
