"""
Batch inference for the P1 fine-tuned IP-Adapter model on labeled validation pairs.

Purpose
-------
This script evaluates the Stage-1 supervised IP-Adapter fine-tune ("P1") using
the exact same batch-inference / manifest format as the existing baseline
scripts. The only difference from the zero-shot IP-Adapter baseline is that we
reload the fine-tuned P1 weights:

- `image_proj_model.pt`
- `ip_attn_procs.pt`

These files are produced by `scripts/train_ip_adapter_finetune.py`.

Why this file exists
--------------------
We keep this as a separate script so that:
- zero-shot IP-Adapter baseline inference stays unchanged
- P1 evaluation has a clean, explicit entrypoint
- downstream evaluation scripts (`eval/run_eval.py`,
  `eval/target_emotion_metrics.py`) can consume the outputs without any format
  changes

Output
------
The output directory contains:
- generated images named like `0000_sheet_00008_happy.jpg`
- `manifest.json` with the same schema used by the baseline batch scripts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from run_baseline import EMOTION_PROMPTS, NEGATIVE_PROMPT, load_all_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs-json",
        default="data/label_pairs/val.json",
        help="Labeled validation pairs JSON with target_emotion.",
    )
    parser.add_argument(
        "--finetuned-dir",
        required=True,
        help="Directory containing P1 outputs: image_proj_model.pt and ip_attn_procs.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ip_adapter_finetune_labeled",
        help="Directory for generated images and manifest.json",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename written inside output-dir",
    )
    parser.add_argument("--scale", type=float, default=0.7, help="IP-Adapter scale during inference")
    parser.add_argument("--sd-path", default="models/sd-v1-5", help="Stable Diffusion base model path")
    parser.add_argument("--ip-repo-path", default="models/ip-adapter", help="Local IP-Adapter repo path")
    parser.add_argument("--n", type=int, default=500, help="Number of samples to run; 0 means full JSON")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; actual seed for sample i is seed + i",
    )
    return parser.parse_args()


def resolve_p1_weight_paths(finetuned_dir: str | Path) -> tuple[Path, Path]:
    """
    Resolve the two checkpoint files produced by P1 fine-tuning.

    Expected files inside `finetuned_dir`:
    - image_proj_model.pt
    - ip_attn_procs.pt
    """
    finetuned_dir = Path(finetuned_dir)
    image_proj_path = finetuned_dir / "image_proj_model.pt"
    ip_attn_path = finetuned_dir / "ip_attn_procs.pt"

    missing = [str(p) for p in (image_proj_path, ip_attn_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required P1 checkpoint file(s):\n"
            + "\n".join(missing)
        )

    return image_proj_path, ip_attn_path


def load_p1_weights(pipe, image_proj_path: Path, ip_attn_path: Path, device: str) -> None:
    """
    Load P1 fine-tuned IP-Adapter weights into a base SD1.5 + IP-Adapter pipeline.

    This matches the save format used in `scripts/train_ip_adapter_finetune.py`:
    - `image_proj_model.pt` stores `pipe.unet.encoder_hid_proj.state_dict()`
    - `ip_attn_procs.pt` stores a dict:
        {attn_processor_name: processor_state_dict}
    """
    if not hasattr(pipe.unet, "encoder_hid_proj") or pipe.unet.encoder_hid_proj is None:
        raise RuntimeError("UNet does not expose encoder_hid_proj; cannot load P1 image projection weights.")

    image_proj_state = torch.load(image_proj_path, map_location=device)
    pipe.unet.encoder_hid_proj.load_state_dict(image_proj_state, strict=True)

    attn_state = torch.load(ip_attn_path, map_location="cpu")
    matched = 0
    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    for name, proc_state in attn_state.items():
        if name not in pipe.unet.attn_processors:
            continue

        state = {
            key: value.to(device=device, dtype=target_dtype)
            for key, value in proc_state.items()
        }
        pipe.unet.attn_processors[name].load_state_dict(state)
        matched += 1

    if matched == 0:
        raise RuntimeError(
            "Loaded P1 attention checkpoint, but no UNet attention processors matched. "
            "Please verify the diffusers/IP-Adapter save format."
        )

    print(f"Loaded P1 image projection from: {image_proj_path}")
    print(f"Loaded P1 attention processors from: {ip_attn_path} ({matched}/{len(attn_state)} matched)")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    pairs = json.loads(Path(args.pairs_json).read_text())
    if args.n > 0:
        pairs = pairs[: args.n]

    image_proj_path, ip_attn_path = resolve_p1_weight_paths(args.finetuned_dir)

    # Reuse the same model-loading path as the zero-shot baseline so the only
    # experimental difference is the P1 fine-tuned weights.
    pipe, _, device = load_all_models(args)
    load_p1_weights(pipe, image_proj_path, ip_attn_path, device)

    valid_emotions = set(EMOTION_PROMPTS.keys())
    manifest_records = []
    skipped_invalid_label = 0
    skipped_missing_reference = 0

    for i, pair in enumerate(tqdm(pairs, desc="ip-adapter-p1-labeled")):
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
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[raw_image],
                num_inference_steps=30,
                guidance_scale=7.5,
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
                "baseline_type": "ip_adapter_finetuned_p1",
                "finetuned_dir": str(Path(args.finetuned_dir)),
                "generation_mode": "strict_ground_truth_label",
            }
        )

    manifest_payload = {
        "summary": {
            "n_input_pairs": len(pairs),
            "n_generated": len(manifest_records),
            "skipped_invalid_label": skipped_invalid_label,
            "skipped_missing_reference": skipped_missing_reference,
        },
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False))

    print(f"Saved P1 labeled batch outputs to {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
