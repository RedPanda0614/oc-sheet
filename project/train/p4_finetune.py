"""
P4: Paired anti-copy fine-tuning
================================

This script continues training from a previous IP-Adapter checkpoint
(typically P3, but P1 also works) and adds the P4 objective:

- the target panel acts as the positive example
- the reference panel acts as a hard negative when the requested emotion differs

Implementation summary
----------------------
We keep the P3 masked diffusion loss, then add a triplet-style anti-copy loss
in latent space:

    triplet = relu(sim(pred_x0, ref_latent) - sim(pred_x0, target_latent) + margin)

The anti-copy term is only active for pairs where:

    reference_emotion != target_emotion

Reference emotions are looked up from faces_emotion.json.

Usage
-----
Resume from P3:

  python scripts/p4_finetune.py \
      --resume_dir checkpoints/p3 \
      --train_json data/label_pairs/train.json \
      --val_json data/label_pairs/val.json \
      --faces_emotion_json data/processed/faces_emotion.json \
      --output_dir checkpoints/p4

Resume directly from P1:

  python scripts/p4_finetune.py \
      --image_proj_ckpt results/ip_adapter_finetune/image_proj_model.pt \
      --ip_attn_ckpt results/ip_adapter_finetune/ip_attn_procs.pt \
      --output_dir checkpoints/p4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler, StableDiffusionPipeline
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


EMOTION_PROMPTS = {
    "neutral": "neutral expression",
    "happy": "happy smiling expression",
    "sad": "sad expression",
    "angry": "angry frowning expression",
    "surprised": "surprised expression",
    "crying": "crying expression with tears",
    "embarrassed": "embarrassed blushing expression",
}

BASE_PROMPT = "anime character portrait, 1girl"


def normalize_path_key(path: str) -> str:
    """Normalize JSON file paths so pair metadata and emotion labels match reliably."""
    return Path(path).as_posix()


def load_face_emotion_map(path: str | Path) -> dict[str, str]:
    records = json.loads(Path(path).read_text())
    mapping = {}
    for record in records:
        face_path = record.get("face_path")
        emotion = record.get("target_emotion")
        if not face_path or not emotion:
            continue
        mapping[normalize_path_key(face_path)] = emotion
    if not mapping:
        raise RuntimeError(f"No usable face emotion labels found in {path}")
    return mapping


class AntiCopyExpressionDataset(Dataset):
    """
    Dataset for P4 paired anti-copy training.

    Each sample returns:
    - reference PIL image for IP-Adapter conditioning
    - target tensor for the diffusion objective
    - reference tensor so we can encode the hard-negative latent
    - anti_copy_active flag, based on reference emotion vs target emotion
    """

    def __init__(self, pairs_json: str, faces_emotion_json: str, size: int = 512):
        raw_pairs = json.loads(Path(pairs_json).read_text())
        face_emotion_map = load_face_emotion_map(faces_emotion_json)

        usable_pairs = []
        missing_ref_emotion = 0
        anti_copy_pairs = 0

        for pair in raw_pairs:
            target_emotion = pair.get("target_emotion")
            reference_path = pair.get("reference_path")
            target_path = pair.get("target_path")
            if target_emotion not in EMOTION_PROMPTS:
                continue
            if not reference_path or not target_path:
                continue
            if not Path(reference_path).exists() or not Path(target_path).exists():
                continue

            reference_emotion = face_emotion_map.get(normalize_path_key(reference_path))
            anti_copy_active = (
                reference_emotion is not None and reference_emotion != target_emotion
            )
            if reference_emotion is None:
                missing_ref_emotion += 1
            if anti_copy_active:
                anti_copy_pairs += 1

            usable_pairs.append(
                {
                    "reference_path": reference_path,
                    "target_path": target_path,
                    "target_emotion": target_emotion,
                    "reference_emotion": reference_emotion,
                    "anti_copy_active": anti_copy_active,
                }
            )

        self.pairs = usable_pairs
        self.size = size
        self.image_tf = transforms.Compose(
            [
                transforms.Resize(
                    (size, size), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print(
            f"  {pairs_json}: {len(raw_pairs)} total -> {len(self.pairs)} usable pairs | "
            f"anti-copy active: {anti_copy_pairs} | missing ref emotion: {missing_ref_emotion}"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        ref_pil = Image.open(pair["reference_path"]).convert("RGB")
        tgt_pil = Image.open(pair["target_path"]).convert("RGB")
        target_emotion = pair["target_emotion"]
        prompt = f"{BASE_PROMPT}, {EMOTION_PROMPTS[target_emotion]}"

        return {
            "reference_pil": ref_pil,
            "reference_pixels": self.image_tf(ref_pil),
            "target_pixels": self.image_tf(tgt_pil),
            "prompt": prompt,
            "anti_copy_active": float(pair["anti_copy_active"]),
        }


def collate_fn(batch):
    return {
        "reference_pil": [b["reference_pil"] for b in batch],
        "reference_pixels": torch.stack([b["reference_pixels"] for b in batch]),
        "target_pixels": torch.stack([b["target_pixels"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "anti_copy_active": torch.tensor(
            [b["anti_copy_active"] for b in batch], dtype=torch.float32
        ),
    }


def build_soft_mask(latent_size=64, expr_weight=3.0, sigma=4.0, device="cpu"):
    """
    Build the same expression-local weighting mask used in P3.

    The mask emphasizes eye / brow and mouth regions so P4 keeps the benefits
    of expression-local control while adding anti-copy regularization.
    """

    height = latent_size
    mask = np.ones((height, height), dtype=np.float32)
    extra = np.zeros((height, height), dtype=np.float32)
    for top_ratio, bottom_ratio in [(0.15, 0.50), (0.58, 0.82)]:
        extra[int(height * top_ratio) : int(height * bottom_ratio), :] = expr_weight - 1.0
    extra = gaussian_filter(extra, sigma=sigma)
    mask = mask + extra
    mask = mask / mask.mean()
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def masked_mse_loss(noise_pred, noise, mask):
    return (((noise_pred.float() - noise.float()) ** 2) * mask).mean()


def predict_x0_from_model_output(
    noisy_latents: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
) -> torch.Tensor:
    """
    Recover the model's estimate of the clean latent x0.

    This lets us compare the prediction against:
    - the target latent (positive)
    - the reference latent (hard negative)
    """

    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sigma_t = (1.0 - alpha_t).sqrt()
    sqrt_alpha_t = alpha_t.sqrt()

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return (noisy_latents - sigma_t * model_output) / sqrt_alpha_t
    if prediction_type == "sample":
        return model_output
    if prediction_type == "v_prediction":
        return sqrt_alpha_t * noisy_latents - sigma_t * model_output
    raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")


def anti_copy_triplet_loss(
    pred_x0: torch.Tensor,
    target_latents: torch.Tensor,
    reference_latents: torch.Tensor,
    active_mask: torch.Tensor,
    margin: float,
):
    """
    Target acts as the positive example and reference acts as the hard negative.

    Only samples whose reference emotion differs from the target emotion
    participate in this loss.
    """

    pred_flat = F.normalize(pred_x0.float().flatten(1), dim=1, eps=1e-6)
    target_flat = F.normalize(target_latents.float().flatten(1), dim=1, eps=1e-6)
    reference_flat = F.normalize(reference_latents.float().flatten(1), dim=1, eps=1e-6)

    positive_similarity = (pred_flat * target_flat).sum(dim=1)
    negative_similarity = (pred_flat * reference_flat).sum(dim=1)

    active_mask = active_mask.bool()
    if active_mask.any():
        loss = F.relu(
            negative_similarity[active_mask]
            - positive_similarity[active_mask]
            + margin
        ).mean()
        pos_mean = positive_similarity[active_mask].mean().detach()
        neg_mean = negative_similarity[active_mask].mean().detach()
        active_count = int(active_mask.sum().item())
    else:
        loss = pred_x0.new_zeros(())
        pos_mean = pred_x0.new_zeros(())
        neg_mean = pred_x0.new_zeros(())
        active_count = 0

    return loss, pos_mean, neg_mean, active_count


def resolve_checkpoint_paths(args) -> tuple[Path, Path]:
    """
    Resolve the checkpoint source for P4.

    Preferred:
    - --resume_dir checkpoints/p3

    Fallback:
    - --image_proj_ckpt path/to/image_proj_model.pt
    - --ip_attn_ckpt path/to/ip_attn_procs.pt
    """

    if args.resume_dir:
        image_proj_ckpt = Path(args.resume_dir) / "image_proj_model.pt"
        ip_attn_ckpt = Path(args.resume_dir) / "ip_attn_procs.pt"
    elif args.image_proj_ckpt and args.ip_attn_ckpt:
        image_proj_ckpt = Path(args.image_proj_ckpt)
        ip_attn_ckpt = Path(args.ip_attn_ckpt)
    else:
        raise ValueError(
            "Provide either --resume_dir or both --image_proj_ckpt and --ip_attn_ckpt."
        )

    missing = [str(p) for p in (image_proj_ckpt, ip_attn_ckpt) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint file(s):\n" + "\n".join(missing))

    return image_proj_ckpt, ip_attn_ckpt


def load_ip_adapter_checkpoint(unet, image_proj_ckpt: Path, ip_attn_ckpt: Path, device):
    proj_state = torch.load(image_proj_ckpt, map_location=device)
    unet.encoder_hid_proj.load_state_dict(proj_state, strict=True)
    print(f"  encoder_hid_proj <- {image_proj_ckpt}")

    attn_state = torch.load(ip_attn_ckpt, map_location="cpu")
    current_processors = unet.attn_processors
    matched = 0
    target_dtype = torch.float16 if device == "cuda" else torch.float32

    for name, processor_state in attn_state.items():
        if name not in current_processors:
            continue
        current_proc = current_processors[name]
        state = {
            key: value.to(device=device, dtype=target_dtype)
            for key, value in processor_state.items()
        }
        current_proc.load_state_dict(state)
        matched += 1

    print(f"  ip_attn_procs    <- {ip_attn_ckpt} ({matched}/{len(attn_state)} matched)")
    if matched == 0:
        raise RuntimeError(
            "No attention processors matched while loading the resume checkpoint."
        )


def save_checkpoint(unet, output_dir: str, step: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(unet.encoder_hid_proj.state_dict(), out / "image_proj_model.pt")

    attn_state = {}
    for name, proc in unet.attn_processors.items():
        if not hasattr(proc, "state_dict"):
            continue
        state_dict = proc.state_dict()
        if not state_dict:
            continue
        if not any(key.startswith(("to_k_ip", "to_v_ip")) for key in state_dict):
            continue
        attn_state[name] = {key: value.detach().cpu() for key, value in state_dict.items()}

    torch.save(attn_state, out / "ip_attn_procs.pt")
    torch.save(unet.encoder_hid_proj.state_dict(), out / f"image_proj_model_step{step}.pt")
    torch.save(attn_state, out / f"ip_attn_procs_step{step}.pt")
    print(f"  [ckpt] step {step} -> {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="models/sd-v1-5")
    parser.add_argument("--ip_repo_path", default="models/ip-adapter")
    parser.add_argument("--ip_weight", default="ip-adapter-plus_sd15.bin")

    parser.add_argument("--resume_dir", default=None)
    parser.add_argument("--image_proj_ckpt", default=None)
    parser.add_argument("--ip_attn_ckpt", default=None)

    parser.add_argument("--train_json", default="data/label_pairs/train.json")
    parser.add_argument("--val_json", default="data/label_pairs/val.json")
    parser.add_argument(
        "--faces_emotion_json",
        default="data/processed/faces_emotion.json",
        help="Emotion labels used to decide whether the reference should be a hard negative.",
    )
    parser.add_argument("--output_dir", default="checkpoints/p4")

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ip_scale", type=float, default=0.7)

    parser.add_argument("--expr_weight", type=float, default=3.0)
    parser.add_argument("--mask_sigma", type=float, default=4.0)
    parser.add_argument(
        "--anti_copy_weight",
        type=float,
        default=0.0,
        help="Weight on the hard-negative anti-copy triplet loss.",
    )
    parser.add_argument(
        "--anti_copy_margin",
        type=float,
        default=0.05,
        help="Margin used in relu(sim(pred, ref) - sim(pred, target) + margin).",
    )
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    image_proj_ckpt, ip_attn_ckpt = resolve_checkpoint_paths(args)

    latent_size = args.image_size // 8
    soft_mask = build_soft_mask(latent_size, args.expr_weight, args.mask_sigma, device)
    print(
        f"Mask: {soft_mask.shape}, min={soft_mask.min():.3f} "
        f"max={soft_mask.max():.3f} mean={soft_mask.mean():.3f}"
    )

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

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.to(device)

    print("Loading previous-stage checkpoint...")
    load_ip_adapter_checkpoint(unet, image_proj_ckpt, ip_attn_ckpt, device)

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
        if hasattr(proc, "float"):
            proc.float()
        if hasattr(proc, "requires_grad_"):
            try:
                proc.requires_grad_(True)
            except Exception:
                pass
        if hasattr(proc, "parameters"):
            trainable += list(proc.parameters())

    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_ds = AntiCopyExpressionDataset(
        args.train_json, args.faces_emotion_json, args.image_size
    )
    val_ds = AntiCopyExpressionDataset(
        args.val_json, args.faces_emotion_json, args.image_size
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    with open(Path(args.output_dir) / "train_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)

    print(
        f"\nP4 training: {args.num_steps} steps, lr={args.lr}, "
        f"anti_copy_weight={args.anti_copy_weight}, margin={args.anti_copy_margin}\n"
    )
    anti_copy_enabled = args.anti_copy_weight > 0.0

    unet.train()
    step = 0
    running = {
        "total": 0.0,
        "diffusion": 0.0,
        "anti_copy": 0.0,
        "active_pairs": 0.0,
    }
    pbar = tqdm(total=args.num_steps, desc="P4")

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            target_pixels = batch["target_pixels"].to(device=device, dtype=vae.dtype)
            reference_pixels = None
            if anti_copy_enabled:
                reference_pixels = batch["reference_pixels"].to(
                    device=device, dtype=vae.dtype
                )
            prompts = batch["prompts"]
            ref_pils = batch["reference_pil"]
            anti_copy_active = None
            if anti_copy_enabled:
                anti_copy_active = batch["anti_copy_active"].to(device=device)

            with torch.no_grad():
                target_latents = vae.encode(target_pixels).latent_dist.sample() * 0.18215
                reference_latents = None
                if anti_copy_enabled:
                    reference_latents = (
                        vae.encode(reference_pixels).latent_dist.sample() * 0.18215
                    )
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(
                    target_latents, noise, timesteps
                )

                text_ids = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                text_embeds = text_encoder(text_ids).last_hidden_state

                image_embeds = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[ref_pils],
                    ip_adapter_image_embeds=None,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={"image_embeds": image_embeds},
                ).sample

                diffusion_loss = masked_mse_loss(noise_pred, noise, soft_mask)
                anti_copy_loss = diffusion_loss.new_zeros(())
                pos_sim = diffusion_loss.new_zeros(())
                neg_sim = diffusion_loss.new_zeros(())
                active_count = 0
                if anti_copy_enabled:
                    pred_x0 = predict_x0_from_model_output(
                        noisy_latents, noise_pred, timesteps, noise_scheduler
                    )
                    anti_copy_loss, pos_sim, neg_sim, active_count = (
                        anti_copy_triplet_loss(
                            pred_x0,
                            target_latents,
                            reference_latents,
                            anti_copy_active,
                            args.anti_copy_margin,
                        )
                    )
                total_loss = diffusion_loss + args.anti_copy_weight * anti_copy_loss

            scaler.scale(total_loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            running["total"] += total_loss.item()
            running["diffusion"] += diffusion_loss.item()
            running["anti_copy"] += anti_copy_loss.item()
            running["active_pairs"] += float(active_count)

            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                avg_total = running["total"] / args.log_every
                avg_diff = running["diffusion"] / args.log_every
                avg_anti = running["anti_copy"] / args.log_every
                avg_active = running["active_pairs"] / args.log_every
                pbar.set_postfix(
                    {
                        "loss": f"{avg_total:.4f}",
                        "diff": f"{avg_diff:.4f}",
                        "anti": f"{avg_anti:.4f}",
                        "active": f"{avg_active:.2f}",
                        "pos": f"{float(pos_sim):.3f}",
                        "neg": f"{float(neg_sim):.3f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                running = {key: 0.0 for key in running}

            if step % args.save_every == 0:
                save_checkpoint(unet, args.output_dir, step)
                val_stats = compute_val_loss(
                    pipe=pipe,
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    val_loader=val_loader,
                    soft_mask=soft_mask,
                    anti_copy_margin=args.anti_copy_margin,
                    anti_copy_weight=args.anti_copy_weight,
                    device=device,
                    use_amp=use_amp,
                )
                tqdm.write(
                    "  step "
                    f"{step:5d} | val_total={val_stats['total']:.4f} "
                    f"| val_diff={val_stats['diffusion']:.4f} "
                    f"| val_anti={val_stats['anti_copy']:.4f}"
                )

    pbar.close()
    save_checkpoint(unet, args.output_dir, step)
    print(f"\nDone. Saved to {args.output_dir}")


@torch.no_grad()
def compute_val_loss(
    pipe,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    val_loader,
    soft_mask,
    anti_copy_margin,
    anti_copy_weight,
    device,
    use_amp,
    max_batches: int = 20,
):
    unet.eval()
    anti_copy_enabled = anti_copy_weight > 0.0
    totals = {"total": 0.0, "diffusion": 0.0, "anti_copy": 0.0}
    num_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        target_pixels = batch["target_pixels"].to(device=device, dtype=vae.dtype)
        reference_pixels = None
        if anti_copy_enabled:
            reference_pixels = batch["reference_pixels"].to(device=device, dtype=vae.dtype)
        prompts = batch["prompts"]
        ref_pils = batch["reference_pil"]
        anti_copy_active = None
        if anti_copy_enabled:
            anti_copy_active = batch["anti_copy_active"].to(device=device)

        target_latents = vae.encode(target_pixels).latent_dist.sample() * 0.18215
        reference_latents = None
        if anti_copy_enabled:
            reference_latents = vae.encode(reference_pixels).latent_dist.sample() * 0.18215
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (target_latents.shape[0],),
            device=device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

        text_ids = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
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
            diffusion_loss = masked_mse_loss(noise_pred, noise, soft_mask)
            anti_copy_loss = diffusion_loss.new_zeros(())
            if anti_copy_enabled:
                pred_x0 = predict_x0_from_model_output(
                    noisy_latents, noise_pred, timesteps, noise_scheduler
                )
                anti_copy_loss, _, _, _ = anti_copy_triplet_loss(
                    pred_x0,
                    target_latents,
                    reference_latents,
                    anti_copy_active,
                    anti_copy_margin,
                )
            total_loss = diffusion_loss + anti_copy_weight * anti_copy_loss

        totals["total"] += total_loss.item()
        totals["diffusion"] += diffusion_loss.item()
        totals["anti_copy"] += anti_copy_loss.item()
        num_batches += 1

    unet.train()
    if num_batches == 0:
        return {key: 0.0 for key in totals}
    return {key: value / num_batches for key, value in totals.items()}


if __name__ == "__main__":
    main()
