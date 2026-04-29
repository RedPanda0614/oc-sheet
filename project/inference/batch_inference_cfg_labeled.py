"""
Batch inference for reference-image CFG sampling on top of an IP-Adapter model.

Purpose
-------
This script evaluates a generator-level sampling variant:

- same labeled validation protocol as the existing IP-Adapter scripts
- no reranking (P5) and no sheet memory (P6)
- optional fine-tuned checkpoint loading (P1 / P4 / any compatible directory)
- explicit classifier-free guidance on the IP-Adapter image condition

Why this file exists
--------------------
The user wanted to test reference-image CFG *between* P4 and P5:

    P4 backbone -> P4 + CFG sampling -> P5 -> P6

That means we want a clean inference-only variant that keeps the generator
fixed and only changes how the reference image condition is used at sampling
time.

Implementation note
-------------------
We precompute IP-Adapter image embeddings with
`do_classifier_free_guidance=True`, which gives us unconditional and
conditional image-condition branches. We then apply a separate image-guidance
scale on the conditional branch:

    guided = uncond + image_cfg_scale * (cond - uncond)

When `image_cfg_scale = 1.0`, this reduces to the default behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from run_baseline import EMOTION_PROMPTS, NEGATIVE_PROMPT, load_all_models
from batch_inference_p1_labeled import load_p1_weights, resolve_p1_weight_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-json",
        default="data/label_pairs/val.json",
        help="Labeled validation pairs JSON with target_emotion.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Optional directory containing image_proj_model.pt and ip_attn_procs.pt. "
            "If omitted, runs zero-shot IP-Adapter with image CFG sampling."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/ip_adapter_cfg_labeled",
        help="Directory for generated images and manifest.json.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename written inside output-dir.",
    )
    parser.add_argument("--scale", type=float, default=0.7, help="Base IP-Adapter scale during inference.")
    parser.add_argument(
        "--image-cfg-scale",
        type=float,
        default=1.5,
        help=(
            "Classifier-free guidance scale applied to the reference-image condition. "
            "1.0 reproduces the default conditioning strength; values > 1.0 strengthen "
            "the conditional image branch."
        ),
    )
    parser.add_argument("--guidance", type=float, default=7.5, help="Text classifier-free guidance scale.")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion sampling steps per image.")
    parser.add_argument("--sd-path", default="models/sd-v1-5", help="Stable Diffusion base model path.")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter", help="Local IP-Adapter repo path.")
    parser.add_argument("--n", type=int, default=500, help="Number of samples to run; 0 means full JSON.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; actual seed for sample i is seed + i.",
    )
    parser.add_argument(
        "--model-tag",
        default="",
        help=(
            "Optional manifest tag describing the backbone used. "
            "If omitted, we use the checkpoint directory name or 'zero_shot'."
        ),
    )
    return parser.parse_args()


def apply_reference_image_cfg(image_embeds: list[torch.Tensor], image_cfg_scale: float) -> list[torch.Tensor]:
    """
    Strengthen the conditional IP-Adapter image branch with a CFG-style rule.

    The pipeline expects a concatenated unconditional/conditional layout when
    text CFG is enabled. For each embed tensor:

    - first half: unconditional image condition
    - second half: conditional image condition

    We keep the unconditional branch unchanged and replace the conditional
    branch with:

        uncond + s * (cond - uncond)

    This mirrors classifier-free guidance, but on the reference-image signal.
    """

    if abs(image_cfg_scale - 1.0) < 1e-8:
        return image_embeds

    guided_embeds = []
    for embed in image_embeds:
        if embed.shape[0] % 2 != 0:
            raise ValueError(
                "Expected unconditional/conditional image embeds stacked on batch dimension, "
                f"but got shape {tuple(embed.shape)}."
            )
        half = embed.shape[0] // 2
        uncond = embed[:half]
        cond = embed[half:]
        guided_cond = uncond + image_cfg_scale * (cond - uncond)
        guided_embeds.append(torch.cat([uncond, guided_cond], dim=0))
    return guided_embeds


def prepare_reference_cfg_embeds(pipe, reference_image: Image.Image, device: str, image_cfg_scale: float):
    """
    Build IP-Adapter image embeddings with unconditional + conditional branches,
    then apply a separate reference-image CFG scale.
    """

    image_embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[[reference_image]],
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return apply_reference_image_cfg(image_embeds, image_cfg_scale=image_cfg_scale)


def infer_model_tag(args) -> str:
    if args.model_tag.strip():
        return args.model_tag.strip()
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).name
    return "zero_shot"


def main():
    args = parse_args()
    if args.image_cfg_scale <= 0:
        raise ValueError("--image-cfg-scale must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    pairs = json.loads(Path(args.pairs_json).read_text())
    if args.n > 0:
        pairs = pairs[: args.n]

    pipe, _, device = load_all_models(args)

    image_proj_path = ip_attn_path = None
    if args.checkpoint_dir:
        image_proj_path, ip_attn_path = resolve_p1_weight_paths(args.checkpoint_dir)
        load_p1_weights(pipe, image_proj_path, ip_attn_path, device)

    model_tag = infer_model_tag(args)
    valid_emotions = set(EMOTION_PROMPTS.keys())
    manifest_records = []
    skipped_invalid_label = 0
    skipped_missing_reference = 0

    for i, pair in enumerate(tqdm(pairs, desc="ip-adapter-image-cfg-labeled")):
        target_emotion = pair.get("target_emotion")
        reference_path = pair.get("reference_path")

        if target_emotion not in valid_emotions:
            skipped_invalid_label += 1
            continue
        if not reference_path or not Path(reference_path).exists():
            skipped_missing_reference += 1
            continue

        prompt = EMOTION_PROMPTS[target_emotion]
        raw_image = Image.open(reference_path).convert("RGB")
        generator = torch.Generator(device=device).manual_seed(args.seed + i)

        try:
            image_embeds = prepare_reference_cfg_embeds(
                pipe=pipe,
                reference_image=raw_image,
                device=device,
                image_cfg_scale=args.image_cfg_scale,
            )
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=None,
                ip_adapter_image_embeds=image_embeds,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
        except Exception as exc:
            tqdm.write(f"Failed to generate for {reference_path}: {exc}")
            continue

        sheet_id = pair.get("sheet_id", "none")
        save_name = f"{i:04d}_{sheet_id}_{target_emotion}.jpg"
        generated_path = output_dir / save_name
        image.save(generated_path)

        manifest_records.append(
            {
                "index": i,
                "sheet_id": sheet_id,
                "reference_path": reference_path,
                "target_path": pair.get("target_path"),
                "generated_path": str(generated_path),
                "requested_label": target_emotion,
                "ground_truth_target_emotion": target_emotion,
                "label_type": "expression",
                "seed": args.seed + i,
                "ip_adapter_scale": args.scale,
                "image_cfg_scale": args.image_cfg_scale,
                "text_guidance_scale": args.guidance,
                "baseline_type": "ip_adapter_image_cfg_sampling",
                "model_tag": model_tag,
                "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()) if args.checkpoint_dir else None,
                "generation_mode": "reference_image_cfg_sampling",
            }
        )

    manifest_payload = {
        "summary": {
            "n_input_pairs": len(pairs),
            "n_generated": len(manifest_records),
            "skipped_invalid_label": skipped_invalid_label,
            "skipped_missing_reference": skipped_missing_reference,
            "image_cfg_scale": args.image_cfg_scale,
            "text_guidance_scale": args.guidance,
            "model_tag": model_tag,
        },
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))

    print(f"Saved reference-image CFG batch outputs to {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
