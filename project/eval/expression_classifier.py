"""
Expression / view control evaluator.

The proposal asks for expression accuracy and optional view accuracy.
This module supports a practical zero-shot CLIP evaluator, and can later be
swapped for a fine-tuned classifier checkpoint without changing run_eval.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


EXPRESSION_PROMPTS = {
    "neutral": "an anime character portrait with a neutral expression",
    "happy": "an anime character portrait with a happy smiling expression",
    "sad": "an anime character portrait with a sad expression",
    "angry": "an anime character portrait with an angry frowning expression",
    "surprised": "an anime character portrait with a surprised expression",
    "crying": "an anime character portrait with a crying expression and tears"
}

VIEW_PROMPTS = {
    "front": "an anime character portrait in front view",
    "side": "an anime character portrait in side profile view",
    "back": "an anime character portrait in back view",
}


@dataclass
class Prediction:
    label: str
    confidence: float


class CLIPControlEvaluator:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        exclude_expression_labels: set[str] | None = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        excluded = exclude_expression_labels or set()
        self.expression_prompts = {
            label: prompt
            for label, prompt in EXPRESSION_PROMPTS.items()
            if label not in excluded
        }

    def predict(self, image_path: str | Path, label_type: str = "expression") -> Prediction:
        if label_type == "expression":
            prompts = self.expression_prompts
        elif label_type == "view":
            prompts = VIEW_PROMPTS
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

        image = Image.open(image_path).convert("RGB")
        texts = list(prompts.values())
        labels = list(prompts.keys())

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0)
            best_idx = int(probs.argmax().item())

        return Prediction(label=labels[best_idx], confidence=float(probs[best_idx].item()))
