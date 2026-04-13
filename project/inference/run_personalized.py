"""
Inference for Textual Inversion / LoRA personalized baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer


EMOTION_PROMPTS = {
    "happy":       "manga character, smiling, happy expression, 1girl, high quality",
    "sad":         "manga character, sad expression, teary eyes, 1girl, high quality",
    "angry":       "manga character, angry expression, frowning, 1girl, high quality",
    "surprised":   "manga character, surprised expression, wide eyes, 1girl, high quality",
    "crying":      "manga character, crying, tears, 1girl, high quality",
    "embarrassed": "manga character, embarrassed, blushing, 1girl, high quality",
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, ugly"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="models/sd-v1-5")
    p.add_argument("--mode", choices=["textual_inversion", "lora"], required=True)
    p.add_argument("--weights-dir", required=True, help="Path to TI or LoRA weights")
    p.add_argument("--token", default="<oc>")
    p.add_argument("--output-dir", default="results/personalized")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_textual_inversion(pipe, weights_dir: Path, token: str):
    # Load learned embedding
    embed_path = weights_dir / "learned_embeds.pt"
    if not embed_path.exists():
        raise FileNotFoundError(f"Missing TI embedding: {embed_path}")

    learned = torch.load(embed_path, map_location="cpu")
    if token not in learned:
        raise RuntimeError(f"Token {token} not found in {embed_path}")

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])
        text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_id] = learned[token]


def load_lora(pipe, weights_dir: Path):
    pipe.unet.load_attn_procs(weights_dir)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    ).to(device)

    weights_dir = Path(args.weights_dir)
    if args.mode == "textual_inversion":
        load_textual_inversion(pipe, weights_dir, args.token)
    else:
        load_lora(pipe, weights_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for emotion, prompt in EMOTION_PROMPTS.items():
        full_prompt = f"{prompt}, {args.token}"
        image = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        image.save(out_dir / f"{args.mode}_{emotion}.jpg")


if __name__ == "__main__":
    main()
