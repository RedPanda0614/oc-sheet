"""
IP-Adapter fine-tuning for expression control (reference-guided).

Trains IP-Adapter attention processors (and image projection if available)
on emotion-labeled pairs: (reference image, target image, target_emotion).

This is the main Stage-1 training script aligned with the proposal.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer


EMOTION_PROMPTS = {
    "neutral": "neutral expression",
    "happy": "happy smiling expression",
    "sad": "sad expression",
    "angry": "angry frowning expression",
    "surprised": "surprised expression",
    "crying": "crying expression with tears",
}


def state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def collect_ip_adapter_attn_processor_state(unet) -> dict[str, dict[str, torch.Tensor]]:
    attn_state = {}
    for name, proc in unet.attn_processors.items():
        if not hasattr(proc, "state_dict"):
            continue
        proc_state = proc.state_dict()
        if not proc_state:
            continue
        if not any(key.startswith(("to_k_ip", "to_v_ip")) for key in proc_state):
            continue
        attn_state[name] = state_dict_to_cpu(proc_state)
    return attn_state


def save_finetuned_ip_adapter(unet, output_dir: Path, args) -> None:
    if hasattr(unet, "encoder_hid_proj") and unet.encoder_hid_proj is not None:
        torch.save(state_dict_to_cpu(unet.encoder_hid_proj.state_dict()), output_dir / "image_proj_model.pt")

    attn_state = collect_ip_adapter_attn_processor_state(unet)
    torch.save(attn_state, output_dir / "ip_attn_procs.pt")

    meta = {
        "ip_scale": args.ip_scale,
        "base_prompt": args.base_prompt,
        "base_ip_adapter_weight": args.ip_weight,
        "save_format": "diffusers_state_dict_override",
        "reload_hint": [
            "Load the base IP-Adapter with pipe.load_ip_adapter(...).",
            "Load image_proj_model.pt into pipe.unet.encoder_hid_proj.",
            "Load ip_attn_procs.pt and apply each state_dict to pipe.unet.attn_processors[name].",
        ],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="models/sd-v1-5")
    p.add_argument("--ip-repo-path", default="models/ip-adapter")
    p.add_argument("--ip-weight", default="ip-adapter-plus_sd15.bin")
    p.add_argument("--pairs-json", default="data/label_pairs/train.json")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/ip_adapter_finetune")
    p.add_argument("--base-prompt", default="anime character portrait, 1girl")
    p.add_argument("--ip-scale", type=float, default=0.7)
    return p.parse_args()


class ExpressionPairDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, size: int, base_prompt: str):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.base_prompt = base_prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        ref = Image.open(pair["reference_path"]).convert("RGB")
        tgt = Image.open(pair["target_path"]).convert("RGB")
        emotion = pair["target_emotion"]
        prompt = f"{self.base_prompt}, {EMOTION_PROMPTS.get(emotion, emotion)}"

        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        ref_img = ref
        tgt_img = self.transform(tgt)
        return {
            "reference_pil": ref_img,
            "target_pixels": tgt_img,
            "input_ids": input_ids,
        }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(Path(args.pairs_json).read_text())
    pairs = [
        p for p in pairs
        if p.get("target_emotion") in EMOTION_PROMPTS
        and Path(p.get("reference_path", "")).exists()
        and Path(p.get("target_path", "")).exists()
    ]
    if not pairs:
        raise RuntimeError("No valid pairs with emotion labels found.")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    pipe.load_ip_adapter(
        args.ip_repo_path,
        subfolder="models",
        weight_name=args.ip_weight,
    )
    pipe.set_ip_adapter_scale(args.ip_scale)

    tokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    vae: AutoencoderKL = pipe.vae
    unet = pipe.unet
    def collate_fn(batch):
        return {
            "reference_pil": [b["reference_pil"] for b in batch],
            "target_pixels": torch.stack([b["target_pixels"] for b in batch]),
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
        }

    dataset = ExpressionPairDataset(pairs, tokenizer, size=args.resolution, base_prompt=args.base_prompt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable gradients for IP-Adapter components
    trainable_params = []
    if hasattr(unet, "encoder_hid_proj") and unet.encoder_hid_proj is not None:
        unet.encoder_hid_proj.float()
        unet.encoder_hid_proj.requires_grad_(True)
        trainable_params += list(unet.encoder_hid_proj.parameters())

    # Unfreeze IP-Adapter attention processors if present
    for _, proc in unet.attn_processors.items():
        if hasattr(proc, "float"):
            proc.float()
        if hasattr(proc, "requires_grad_"):
            try:
                proc.requires_grad_(True)
            except Exception:
                pass
        if hasattr(proc, "parameters"):
            trainable_params += list(proc.parameters())

    if not trainable_params:
        raise RuntimeError("No trainable IP-Adapter parameters found. Check diffusers version.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    if hasattr(pipe, "image_encoder"):
        pipe.image_encoder.to(device)

    unet.train()
    if hasattr(unet, "encoder_hid_proj") and unet.encoder_hid_proj is not None:
        unet.encoder_hid_proj.train()

    global_step = 0
    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            target_pixels = batch["target_pixels"].to(device=device, dtype=vae.dtype)
            input_ids = batch["input_ids"].to(device)

            # Encode target to latents
            with torch.no_grad():
                latents = vae.encode(target_pixels).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state

            # Prepare IP-Adapter image embeddings
            ref_pils = batch["reference_pil"]
            if not isinstance(ref_pils, list):
                ref_pils = list(ref_pils)
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
                    encoder_hidden_states,
                    added_cond_kwargs={"image_embeds": image_embeds},
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if global_step % 200 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")
            global_step += 1

    save_finetuned_ip_adapter(unet, output_dir, args)

    print(f"Saved IP-Adapter fine-tune to {output_dir}")


if __name__ == "__main__":
    main()
