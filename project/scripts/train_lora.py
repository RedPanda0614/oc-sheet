"""
LoRA fine-tuning script (few-shot per character).
Baseline for Midway Executive Summary.

Trains LoRA adapters on the UNet cross-attention layers using SD 1.5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="models/sd-v1-5")
    p.add_argument("--pairs-json", default="data/pairs/train.json")
    p.add_argument("--sheet-id", required=True, help="Character id (sheet_id) to personalize")
    p.add_argument("--num-images", type=int, default=8, help="Number of images for few-shot")
    p.add_argument("--token", default="<oc>", help="Optional placeholder token")
    p.add_argument("--instance-prompt", default="a anime character portrait of <oc>")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/lora")
    return p.parse_args()


class CharacterDataset(Dataset):
    def __init__(self, image_paths: list[str], prompt: str, tokenizer, size: int = 512):
        self.image_paths = image_paths
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(img)
        input_ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def load_image_paths(pairs_json: str, sheet_id: str, num_images: int) -> list[str]:
    pairs = json.loads(Path(pairs_json).read_text())
    paths = set()
    for pair in pairs:
        if pair.get("sheet_id") == sheet_id:
            paths.add(pair["reference_path"])
            paths.add(pair["target_path"])
    paths = [p for p in paths if p]
    if not paths:
        raise RuntimeError(f"No images found for sheet_id={sheet_id}")
    random.shuffle(paths)
    return paths[: max(1, num_images)]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / args.sheet_id
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")

    if args.token not in tokenizer.get_vocab():
        tokenizer.add_tokens([args.token])
        text_encoder.resize_token_embeddings(len(tokenizer))

    prompt = args.instance_prompt.replace(args.token, args.token)

    image_paths = load_image_paths(args.pairs_json, args.sheet_id, args.num_images)
    dataset = CharacterDataset(image_paths, prompt, tokenizer, size=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject LoRA adapters into attention processors
    lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            attn_processor.cross_attention_dim
            if hasattr(attn_processor, "cross_attention_dim")
            else unet.config.cross_attention_dim
        )
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=attn_processor.hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )
    unet.set_attn_processor(lora_attn_procs)

    # Only optimize LoRA params
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    unet.train()

    global_step = 0
    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")
            global_step += 1

    # Save LoRA weights
    unet.save_attn_procs(output_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    (output_dir / "prompt.txt").write_text(prompt)
    print(f"Saved LoRA weights to {output_dir}")


if __name__ == "__main__":
    main()
