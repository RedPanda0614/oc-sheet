"""
LoRA fine-tuning on top of an IP-Adapter-conditioned diffusion model.

Purpose
-------
This script trains a UNet LoRA adapter while keeping IP-Adapter active as the
reference-guidance pathway.

Conceptually:
- IP-Adapter supplies character identity from the reference image
- LoRA learns the expression-transfer / generation bias

This makes it possible to test the hybrid setting:

    reference image -> IP-Adapter
    target emotion  -> text prompt
    trainable part  -> LoRA on the UNet

The script supports two starting points:
1. zero-shot base IP-Adapter
2. an existing fine-tuned IP-Adapter checkpoint (P1 / P3 / P4)

Outputs
-------
The output directory contains:
- pytorch_lora_weights.safetensors
- prompt_template.txt
- meta.json

Usage
-----
Train LoRA on top of zero-shot IP-Adapter:

  python scripts/train_ip_adapter_lora.py \
      --pairs-json data/label_pairs/train.json \
      --output-dir results/ip_adapter_lora

Train LoRA on top of a fine-tuned IP-Adapter checkpoint:

  python scripts/train_ip_adapter_lora.py \
      --pairs-json data/label_pairs/train.json \
      --ip-checkpoint-dir checkpoints/p4 \
      --output-dir results/ip_adapter_lora_on_p4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer


EMOTION_PROMPTS = {
    "neutral": "neutral expression",
    "happy": "happy smiling expression",
    "sad": "sad expression",
    "angry": "angry frowning expression",
    "surprised": "surprised expression",
    "crying": "crying expression with tears",
    "embarrassed": "embarrassed blushing expression",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model", default="models/sd-v1-5")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter")
    parser.add_argument("--ip-weight", default="ip-adapter-plus_sd15.bin")
    parser.add_argument(
        "--ip-checkpoint-dir",
        default=None,
        help="Optional fine-tuned IP-Adapter checkpoint dir containing image_proj_model.pt and ip_attn_procs.pt.",
    )
    parser.add_argument("--pairs-json", default="data/label_pairs/train.json")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    steps_group = parser.add_mutually_exclusive_group()
    steps_group.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Train for a fixed number of optimization steps.",
    )
    steps_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Train for a fixed number of full passes through the dataloader.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ip_adapter_lora")
    parser.add_argument("--base-prompt", default="anime character portrait, 1girl")
    parser.add_argument("--ip-scale", type=float, default=0.7)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


class IPAdapterExpressionDataset(Dataset):
    """
    Dataset for reference-guided expression transfer.

    Each sample returns:
    - reference PIL image for IP-Adapter conditioning
    - target image tensor for diffusion training
    - tokenized prompt describing the target emotion
    """

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
        reference = Image.open(pair["reference_path"]).convert("RGB")
        target = Image.open(pair["target_path"]).convert("RGB")
        emotion = pair["target_emotion"]
        prompt = f"{self.base_prompt}, {EMOTION_PROMPTS.get(emotion, emotion)}"

        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "reference_pil": reference,
            "target_pixels": self.transform(target),
            "input_ids": input_ids,
        }


def collate_fn(batch):
    return {
        "reference_pil": [item["reference_pil"] for item in batch],
        "target_pixels": torch.stack([item["target_pixels"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
    }


def add_lora_to_unet(unet, rank: int, lora_alpha: int):
    """
    Attach PEFT LoRA adapters to the UNet attention projections.

    We target the same projections used in the stable global-LoRA script so the
    training recipe stays close to a known-good path.
    """

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is required for IP-Adapter + LoRA training. Install it with `pip install peft`."
        ) from exc

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    trainable = [param for param in unet.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found after `unet.add_adapter(...)`.")

    # Keep LoRA weights in fp32 for a more stable optimizer state.
    for param in trainable:
        param.data = param.data.float()
    return trainable


def load_ip_adapter_checkpoint(unet, checkpoint_dir: str | Path, device: str) -> None:
    """
    Load a previously fine-tuned IP-Adapter checkpoint into the current UNet.

    Expected files inside checkpoint_dir:
    - image_proj_model.pt
    - ip_attn_procs.pt
    """

    checkpoint_dir = Path(checkpoint_dir)
    image_proj_path = checkpoint_dir / "image_proj_model.pt"
    ip_attn_path = checkpoint_dir / "ip_attn_procs.pt"

    missing = [str(path) for path in (image_proj_path, ip_attn_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required IP-Adapter checkpoint file(s):\n" + "\n".join(missing)
        )

    if not hasattr(unet, "encoder_hid_proj") or unet.encoder_hid_proj is None:
        raise RuntimeError(
            "UNet does not expose encoder_hid_proj; cannot load fine-tuned IP-Adapter projection weights."
        )

    image_proj_state = torch.load(image_proj_path, map_location=device)
    unet.encoder_hid_proj.load_state_dict(image_proj_state, strict=True)

    attn_state = torch.load(ip_attn_path, map_location="cpu")
    matched = 0
    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    for name, proc_state in attn_state.items():
        if name not in unet.attn_processors:
            continue
        state = {
            key: value.to(device=device, dtype=target_dtype)
            for key, value in proc_state.items()
        }
        unet.attn_processors[name].load_state_dict(state)
        matched += 1

    if matched == 0:
        raise RuntimeError(
            "Loaded IP-Adapter checkpoint, but no UNet attention processors matched. "
            "Please verify the checkpoint save format."
        )

    print(f"Loaded fine-tuned IP-Adapter projection from: {image_proj_path}")
    print(f"Loaded fine-tuned IP-Adapter attention processors from: {ip_attn_path} ({matched}/{len(attn_state)} matched)")


def save_lora_weights(pipe: StableDiffusionPipeline, output_dir: Path, args) -> None:
    try:
        from peft.utils import get_peft_model_state_dict
        from diffusers.utils import convert_state_dict_to_diffusers
    except ImportError as exc:
        raise RuntimeError(
            "Saving LoRA weights requires PEFT. Install it with `pip install peft`."
        ) from exc

    unet_state_dict = get_peft_model_state_dict(pipe.unet)
    unet_state_dict = convert_state_dict_to_diffusers(unet_state_dict)
    pipe.save_lora_weights(output_dir, unet_lora_layers=unet_state_dict)

    meta = {
        "base_prompt": args.base_prompt,
        "base_ip_adapter_weight": args.ip_weight,
        "ip_scale": args.ip_scale,
        "lora_rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "ip_checkpoint_dir": args.ip_checkpoint_dir,
        "save_format": "diffusers_lora_unet_only",
        "reload_hint": [
            "Load the base SD pipeline.",
            "Load IP-Adapter with pipe.load_ip_adapter(...).",
            "If ip_checkpoint_dir was used during training, reload that IP-Adapter checkpoint first.",
            "Then call pipe.load_lora_weights(output_dir).",
        ],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (output_dir / "prompt_template.txt").write_text(args.base_prompt)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(Path(args.pairs_json).read_text())
    pairs = [
        pair
        for pair in pairs
        if pair.get("target_emotion") in EMOTION_PROMPTS
        and Path(pair.get("reference_path", "")).exists()
        and Path(pair.get("target_path", "")).exists()
    ]
    if not pairs:
        raise RuntimeError("No valid pairs found with emotion labels and existing image paths.")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_ip_adapter(
        args.ip_repo_path,
        subfolder="models",
        weight_name=args.ip_weight,
    )
    pipe.set_ip_adapter_scale(args.ip_scale)

    tokenizer: CLIPTokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    vae: AutoencoderKL = pipe.vae
    unet = pipe.unet

    if args.ip_checkpoint_dir:
        load_ip_adapter_checkpoint(unet, args.ip_checkpoint_dir, device)

    dataset = IPAdapterExpressionDataset(
        pairs, tokenizer, size=args.resolution, base_prompt=args.base_prompt
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    if len(dataloader) == 0:
        raise RuntimeError(
            "Dataloader is empty. Increase the dataset size or reduce batch-size."
        )

    if args.epochs is not None:
        max_steps = args.epochs * len(dataloader)
        training_schedule = f"{args.epochs} epoch(s) = {max_steps} step(s)"
    elif args.max_steps is not None:
        max_steps = args.max_steps
        approx_epochs = max_steps / len(dataloader)
        training_schedule = f"{max_steps} step(s) (~{approx_epochs:.2f} epoch(s))"
    else:
        max_steps = 5000
        approx_epochs = max_steps / len(dataloader)
        training_schedule = f"default {max_steps} step(s) (~{approx_epochs:.2f} epoch(s))"

    print(f"Training schedule: {training_schedule}")

    # Freeze base modules. The trainable parameters will come only from LoRA.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_params = add_lora_to_unet(unet, rank=args.rank, lora_alpha=args.lora_alpha)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.to(device=device, dtype=dtype)
    text_encoder.to(device)
    unet.to(device=device, dtype=dtype)
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.to(device=device, dtype=dtype)
    unet.train()

    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="ip-adapter-lora-train", unit="step")
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            target_pixels = batch["target_pixels"].to(device=device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            reference_pils = batch["reference_pil"]

            with torch.no_grad():
                latents = vae.encode(target_pixels).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids).last_hidden_state.to(
                    device=device,
                    dtype=dtype,
                )
                image_embeds = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=[reference_pils],
                    ip_adapter_image_embeds=None,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                added_cond_kwargs={"image_embeds": image_embeds},
            ).sample
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(),
                noise.float(),
                reduction="mean",
            )

            optimizer.zero_grad()
            if not torch.isfinite(loss):
                print(f"step {global_step} | non-finite loss {loss.item()}, skipping update")
                global_step += 1
                progress_bar.update(1)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
            optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % 100 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")
            global_step += 1

    progress_bar.close()
    save_lora_weights(pipe, output_dir, args)
    print(f"Saved IP-Adapter-conditioned LoRA to {output_dir}")


if __name__ == "__main__":
    main()
