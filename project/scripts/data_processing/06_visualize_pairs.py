"""
06_visualize_pairs.py — Sanity check: render sample (reference, target) pairs.

Saves side-by-side images to results/pair_checks/.
Red flags to look for: different characters, half-cropped faces, blank images.

Usage (from project root):
  python scripts/06_visualize_pairs.py
  python scripts/06_visualize_pairs.py --pairs data/pairs/val.json \
      --output results/pair_checks --n 30
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs",  default="data/pairs/val.json")
    p.add_argument("--output", default="results/pair_checks")
    p.add_argument("--n",      type=int, default=20,
                   help="Number of sample pairs to visualize")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--size",   type=int, default=256,
                   help="Display size for each face thumbnail")
    return p.parse_args()


def add_label(img: Image.Image, text: str, size: int) -> Image.Image:
    """Add a text label at the bottom of an image."""
    canvas = Image.new("RGB", (size, size + 20), (30, 30, 30))
    canvas.paste(img.resize((size, size), Image.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, size + 2), text, fill=(220, 220, 220))
    return canvas


def main():
    args = parse_args()
    pairs_path = Path(args.pairs)
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        raise SystemExit(f"Pairs file not found: {pairs_path}\n"
                         "Run 05_build_pairs.py first.")

    pairs = json.loads(pairs_path.read_text())
    print(f"Loaded {len(pairs)} pairs from {pairs_path}")

    random.seed(args.seed)
    sample = random.sample(pairs, min(args.n, len(pairs)))

    s = args.size
    ok, failed = 0, 0

    for i, pair in enumerate(tqdm(sample, desc="visualizing")):
        try:
            ref    = Image.open(pair["reference_path"]).convert("RGB")
            target = Image.open(pair["target_path"]).convert("RGB")
        except Exception as e:
            print(f"  skip {pair['sheet_id']}: {e}")
            failed += 1
            continue

        ref_lbl    = add_label(ref,    "reference", s)
        target_lbl = add_label(target, "target",    s)

        canvas = Image.new("RGB", (s * 2 + 10, s + 20 + 30), (50, 50, 50))
        canvas.paste(ref_lbl,    (0, 30))
        canvas.paste(target_lbl, (s + 10, 30))

        # sheet id header
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"sheet: {pair['sheet_id'][:40]}", fill=(255, 255, 100))

        out_path = out_dir / f"pair_{i:04d}_{pair['sheet_id'][:20]}.jpg"
        canvas.save(out_path, "JPEG", quality=90)
        ok += 1

    print(f"\n=== Visualization summary ===")
    print(f"  Saved:  {ok} images to {out_dir}/")
    if failed:
        print(f"  Failed: {failed} (missing files?)")
    print(f"\nOpen {out_dir}/ and verify:")
    print("  - Reference and target look like the same character")
    print("  - Faces are well-cropped (not half off-screen)")
    print("  - No completely blank / solid-color images")


if __name__ == "__main__":
    main()
