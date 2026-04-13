"""
Anime palette consistency metric.

The proposal calls for a color-consistency metric based on dominant Lab-space
palette distance. Here we use a practical proxy: Earth Mover-style distance
computed per Lab channel and averaged across channels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wasserstein_distance


def _load_lab_pixels(image_path: str | Path, max_pixels: int = 65536) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = img.reshape(-1, 3).astype(np.float32)
    if len(pixels) > max_pixels:
        idx = np.linspace(0, len(pixels) - 1, max_pixels, dtype=np.int64)
        pixels = pixels[idx]
    return pixels


def palette_distance(image_a: str | Path, image_b: str | Path) -> float:
    pixels_a = _load_lab_pixels(image_a)
    pixels_b = _load_lab_pixels(image_b)

    distances = []
    support = np.arange(256, dtype=np.float32)
    for channel in range(3):
        hist_a, _ = np.histogram(pixels_a[:, channel], bins=256, range=(0, 255), density=True)
        hist_b, _ = np.histogram(pixels_b[:, channel], bins=256, range=(0, 255), density=True)
        distances.append(wasserstein_distance(support, support, hist_a, hist_b))

    return float(np.mean(distances))
