"""
FID utilities for generated-vs-real panel comparison.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def compute_fid(real_paths: list[str], fake_paths: list[str]) -> float | None:
    try:
        from cleanfid import fid
    except ImportError:
        return None

    if not real_paths or not fake_paths:
        return None

    with tempfile.TemporaryDirectory(prefix="ocsheet_fid_real_") as real_dir, tempfile.TemporaryDirectory(
        prefix="ocsheet_fid_fake_"
    ) as fake_dir:
        for idx, src in enumerate(real_paths):
            shutil.copy2(src, Path(real_dir) / f"real_{idx:05d}{Path(src).suffix or '.png'}")
        for idx, src in enumerate(fake_paths):
            shutil.copy2(src, Path(fake_dir) / f"fake_{idx:05d}{Path(src).suffix or '.png'}")
        return float(fid.compute_fid(real_dir, fake_dir))
