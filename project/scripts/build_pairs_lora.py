"""
build_pairs_lora.py

Build supervised pairs for LoRA / Textual Inversion from faces_emotion.json.
Outputs to a separate directory to avoid changing the baseline dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--faces-meta", default="data/processed/faces_meta.json")
    p.add_argument("--faces-emotion", default="data/processed/faces_emotion.json")
    p.add_argument("--output-dir", default="data/lora/pairs")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-faces", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    faces_meta = json.loads(Path(args.faces_meta).read_text())
    emotion_meta = {f["face_path"]: f for f in json.loads(Path(args.faces_emotion).read_text())}

    # Group by sheet_id
    by_sheet: dict[str, list[dict]] = {}
    for face in faces_meta:
        sid = face["sheet_id"]
        by_sheet.setdefault(sid, []).append(face)

    # Sort faces within each sheet by face_idx for determinism
    for sid in by_sheet:
        by_sheet[sid].sort(key=lambda f: f["face_idx"])

    sheets = sorted(by_sheet.keys())
    random.shuffle(sheets)
    n_val = max(1, int(len(sheets) * args.val_ratio))
    val_ids = set(sheets[:n_val])

    train_pairs, val_pairs = [], []
    for sid, faces in by_sheet.items():
        if len(faces) < args.min_faces:
            continue

        # face_0 as reference
        ref = faces[0]
        for target in faces[1:]:
            emotion = emotion_meta.get(target["face_path"], {}).get("target_emotion", "unknown")
            pair = {
                "reference_path": ref["face_path"],
                "target_path": target["face_path"],
                "sheet_id": sid,
                "target_emotion": emotion,
            }
            if sid in val_ids:
                val_pairs.append(pair)
            else:
                train_pairs.append(pair)

        # add a reverse pair for diversity
        if len(faces) > 1:
            target = faces[0]
            ref = faces[1]
            emotion = emotion_meta.get(target["face_path"], {}).get("target_emotion", "unknown")
            pair = {
                "reference_path": ref["face_path"],
                "target_path": target["face_path"],
                "sheet_id": sid,
                "target_emotion": emotion,
            }
            if sid in val_ids:
                val_pairs.append(pair)
            else:
                train_pairs.append(pair)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(train_pairs, indent=2))
    (out_dir / "val.json").write_text(json.dumps(val_pairs, indent=2))

    print(f"Saved train pairs: {len(train_pairs)}")
    print(f"Saved val pairs:   {len(val_pairs)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
