"""
05_build_pairs.py — Group face crops by sheet_id into (reference, target) pairs.

Split strategy: by sheet_id (not by pair) to prevent data leakage — the same
character never appears in both train and val.

Only sheets with >= 2 detected faces are included.

Usage (from project root):
  python scripts/05_build_pairs.py
  python scripts/05_build_pairs.py --meta data/processed/faces_meta.json \
      --output-dir data/pairs --val-ratio 0.2
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta",       default="data/processed/faces_meta.json")
    p.add_argument("--output-dir", default="data/pairs")
    p.add_argument("--val-ratio",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def build_pairs_from_sheet(faces: list[dict]) -> list[dict]:
    """For a sheet with N faces, use face_0 as reference and each other
    face as a target.  Also add face_1 as reference → face_0 as target
    for diversity.  Returns list of pair dicts."""
    if len(faces) < 2:
        return []

    pairs = []
    # face 0 → all others
    ref = faces[0]
    for target in faces[1:]:
        pairs.append({
            "reference_path":   ref["face_path"],
            "target_path":      target["face_path"],
            "sheet_id":         ref["sheet_id"],
            "target_emotion":   "unknown",
        })
    # face 1 → face 0 (extra diversity)
    pairs.append({
        "reference_path":   faces[1]["face_path"],
        "target_path":      faces[0]["face_path"],
        "sheet_id":         faces[0]["sheet_id"],
        "target_emotion":   "unknown",
    })
    return pairs


def main():
    args = parse_args()
    meta_path = Path(args.meta)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise SystemExit(f"Metadata not found: {meta_path}\n"
                         "Run 03_detect_and_crop.py first.")

    all_faces: list[dict] = json.loads(meta_path.read_text())
    print(f"Loaded {len(all_faces)} face entries from {meta_path}")

    # Group by sheet_id
    by_sheet: dict[str, list[dict]] = {}
    for face in all_faces:
        sid = face["sheet_id"]
        by_sheet.setdefault(sid, []).append(face)

    # Sort faces within each sheet by face_idx for determinism
    for sid in by_sheet:
        by_sheet[sid].sort(key=lambda f: f["face_idx"])

    sheets = sorted(by_sheet.keys())
    random.seed(args.seed)
    random.shuffle(sheets)

    n_val   = max(1, int(len(sheets) * args.val_ratio))
    val_ids = set(sheets[:n_val])
    trn_ids = set(sheets[n_val:])

    train_pairs, val_pairs = [], []
    for sid, faces in by_sheet.items():
        pairs = build_pairs_from_sheet(faces)
        if sid in val_ids:
            val_pairs.extend(pairs)
        else:
            train_pairs.extend(pairs)

    (out_dir / "train.json").write_text(json.dumps(train_pairs, indent=2))
    (out_dir / "val.json").write_text(json.dumps(val_pairs, indent=2))

    print(f"\n=== Pair summary ===")
    print(f"  Total sheets:  {len(sheets)}")
    print(f"  Train sheets:  {len(trn_ids)}  →  {len(train_pairs)} pairs")
    print(f"  Val sheets:    {len(val_ids)}  →  {len(val_pairs)} pairs")
    print(f"  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
