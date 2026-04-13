"""
Copy-score utilities.

The proposal explicitly penalizes outputs that are too similar to the reference
when a different expression or view is requested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def _load_grayscale(image_path: str | Path, size: int = 256) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def copy_score(reference_path: str | Path, generated_path: str | Path) -> float:
    ref = _load_grayscale(reference_path)
    gen = _load_grayscale(generated_path)
    return float(ssim(ref, gen, data_range=255.0))


def copy_violation(score: float, threshold: float = 0.88) -> bool:
    return score >= threshold
