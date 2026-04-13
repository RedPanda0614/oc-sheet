"""
ArcFace-based identity similarity utilities.

This module measures the proposal's primary identity metric:
cosine similarity between reference and generated face embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceEmbedding:
    embedding: np.ndarray
    bbox_area: float


class ArcFaceEvaluator:
    def __init__(self, model_name: str = "buffalo_l", providers: list[str] | None = None):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is required for ArcFace evaluation. "
                "Install it with `pip install insightface onnxruntime-gpu`."
            ) from exc

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _extract_embedding(self, image_path: str | Path) -> FaceEmbedding | None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        faces = self.app.get(img)
        if not faces:
            return None

        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        bbox = best_face.bbox
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        emb = np.asarray(best_face.normed_embedding, dtype=np.float32)
        return FaceEmbedding(embedding=emb, bbox_area=area)

    def similarity(self, image_a: str | Path, image_b: str | Path) -> float | None:
        fa = self._extract_embedding(image_a)
        fb = self._extract_embedding(image_b)
        if fa is None or fb is None:
            return None
        return float(np.dot(fa.embedding, fb.embedding))
