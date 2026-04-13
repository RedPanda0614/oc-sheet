"""
Global LoRA training for expression control (all characters).

Trains a single LoRA adapter on the full dataset with emotion-conditioned prompts.
Note: This learns expression control in prompt space; it does NOT use reference images.
For reference-guided control, use IP-Adapter fine-tuning.
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

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer


EMOTION_PROMPTS = {
    "neutral": "neutral expression",
    "happy": "happy smiling expression",
    "sad": "sad expression",
    "angry": "angry frowning expression",
    "surprised": "surprised expression",
    "crying": "crying expression with tears",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="models/sd-v1-5")
    p.add_argument("--pairs-json", default="data/lora/pairs/train.json")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/lora_global")
    p.add_argument("--base-prompt", default="anime character portrait, 1girl")
    p.add_argument("--min-confidence", type=float, default=0.0)
    return p.parse_args()


class ExpressionDataset(Dataset):
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
        img = Image.open(pair["target_path"]).convert("RGB")
        pixel_values = self.transform(img)
        emotion = pair["target_emotion"]
        prompt = f"{self.base_prompt}, {EMOTION_PROMPTS.get(emotion, emotion)}"
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(Path(args.pairs_json).read_text())
    pairs = [
        p for p in pairs
        if p.get("target_emotion") in EMOTION_PROMPTS and Path(p.get("target_path", "")).exists()
    ]
    if not pairs:
        raise RuntimeError("No valid pairs found with target_emotion labels.")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")

    dataset = ExpressionDataset(pairs, tokenizer, size=args.resolution, base_prompt=args.base_prompt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

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

            if global_step % 200 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")
            global_step += 1

    unet.save_attn_procs(output_dir)
    (output_dir / "prompt_template.txt").write_text(args.base_prompt)
    print(f"Saved global LoRA to {output_dir}")


if __name__ == "__main__":
    main()
