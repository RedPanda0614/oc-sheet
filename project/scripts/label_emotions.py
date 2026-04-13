"""
CLIP zero-shot emotion labeling for face crops.

Outputs faces_emotion.json with label + confidence.
Merges sad and crying into a single class: negative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


EMOTION_PROMPTS = {
    "neutral": "an anime character portrait with a neutral expression",
    "happy": "an anime character portrait with a happy smiling expression",
    "angry": "an anime character portrait with an angry frowning expression",
    "surprised": "an anime character portrait with a surprised expression",
    "sad": "an anime character portrait with a sad expression",
    "crying": "an anime character portrait with a crying expression and tears",
}

MERGE_MAP = {
    "sad": "negative",
    "crying": "negative",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--faces-meta", default="data/processed/faces_meta.json")
    p.add_argument("--output", default="data/processed/faces_emotion.json")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--model", default="openai/clip-vit-large-patch14")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    faces = json.loads(Path(args.faces_meta).read_text())
    labels = list(EMOTION_PROMPTS.keys())
    prompts = list(EMOTION_PROMPTS.values())

    output = []
    for entry in faces:
        face_path = entry.get("face_path")
        if not face_path or not Path(face_path).exists():
            continue

        image = Image.open(face_path).convert("RGB")
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
            probs = logits.softmax(dim=0)
            best_idx = int(probs.argmax().item())

        raw_label = labels[best_idx]
        conf = float(probs[best_idx].item())
        label = MERGE_MAP.get(raw_label, raw_label)
        if conf < args.min_confidence:
            label = "unknown"

        output.append(
            {
                "face_path": face_path,
                "sheet_id": entry.get("sheet_id"),
                "face_idx": entry.get("face_idx"),
                "target_emotion": label,
                "raw_emotion": raw_label,
                "confidence": conf,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {len(output)} labels to {out_path}")


if __name__ == "__main__":
    main()
