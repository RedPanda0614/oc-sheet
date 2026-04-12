"""
03_detect_and_crop.py — Detect anime faces and save 512x512 crops.

Detector priority:
  1. anime-face-detector (yolov3)  — best for manga/anime
  2. lbpcascade_animeface (OpenCV) — reliable fallback
  3. face_alignment (blazeface)    — last resort

Each detected face from the same source image is assumed to be the same
character. sheet_id is derived from the source filename.

Usage (from project root):
  python scripts/03_detect_and_crop.py
  python scripts/03_detect_and_crop.py --input data/filtered \
      --output data/processed/faces \
      --meta data/processed/faces_meta.json \
      --score-thresh 0.5 --crop-size 512
"""

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

CASCADE_URL = (
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface"
    "/master/lbpcascade_animeface.xml"
)
CASCADE_PATH = Path("models/cascades/lbpcascade_animeface.xml")


# ── detector backends ────────────────────────────────────────────────────────

def load_anime_face_detector():
    """Try to load the yolov3-based anime-face-detector."""
    try:
        from anime_face_detector import create_detector
        det = create_detector("yolov3")
        return det
    except Exception:
        return None


def load_cascade():
    """Load OpenCV LBP cascade for anime faces, downloading if needed."""
    if not CASCADE_PATH.exists():
        CASCADE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading cascade to {CASCADE_PATH} …")
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from {CASCADE_PATH}")
    return cascade


def load_face_alignment():
    """Load face_alignment blazeface detector as last resort."""
    try:
        import face_alignment
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device="cpu",
        )
        return fa
    except Exception:
        return None


# ── detection functions ───────────────────────────────────────────────────────

def detect_with_yolov3(detector, img_bgr: np.ndarray, score_thresh: float):
    """Returns list of (x1, y1, x2, y2, score) tuples."""
    preds = detector(img_bgr)
    boxes = []
    for pred in preds:
        bb = pred["bbox"]            # [x1, y1, x2, y2, score]
        score = float(bb[4])
        if score >= score_thresh:
            boxes.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]), score))
    return boxes


def detect_with_cascade(cascade, img_bgr: np.ndarray):
    """Returns list of (x1, y1, x2, y2, score=1.0) tuples."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64),
    )
    if len(faces) == 0:
        return []
    return [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces]


def detect_with_fa(fa, img_rgb: np.ndarray):
    """Returns list of (x1, y1, x2, y2, score=1.0) using bounding boxes
    derived from detected landmarks."""
    try:
        landmarks, _, bboxes = fa.get_landmarks_from_image(
            img_rgb, return_bboxes=True
        )
        if bboxes is None:
            return []
        results = []
        for bb in bboxes:
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            results.append((x1, y1, x2, y2, float(bb[4]) if len(bb) > 4 else 1.0))
        return results
    except Exception:
        return []


# ── crop helper ───────────────────────────────────────────────────────────────

def crop_face(img_pil: Image.Image, x1: int, y1: int, x2: int, y2: int,
              pad: float = 0.3, crop_size: int = 512) -> Image.Image:
    """Crop a face with padding, then resize to crop_size×crop_size."""
    w_img, h_img = img_pil.size
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * pad)
    pad_y = int(bh * pad)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w_img, x2 + pad_x)
    cy2 = min(h_img, y2 + pad_y)

    face = img_pil.crop((cx1, cy1, cx2, cy2))
    face = face.resize((crop_size, crop_size), Image.LANCZOS)
    if face.mode != "RGB":
        face = face.convert("RGB")
    return face


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="data/filtered")
    p.add_argument("--output",       default="data/processed/faces")
    p.add_argument("--meta",         default="data/processed/faces_meta.json")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--crop-size",    type=int,   default=512)
    p.add_argument("--pad",          type=float, default=0.3,
                   help="Fractional padding around each detected box")
    return p.parse_args()


def main():
    args = parse_args()
    src  = Path(args.input)
    dst  = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)

    # Select detector
    yolo = load_anime_face_detector()
    if yolo:
        print("Detector: anime-face-detector (yolov3)")
        backend = "yolov3"
    else:
        try:
            cascade = load_cascade()
            print("Detector: lbpcascade_animeface (OpenCV)")
            backend = "cascade"
        except Exception as e:
            print(f"Cascade failed ({e}), falling back to face_alignment")
            fa = load_face_alignment()
            if fa is None:
                raise SystemExit("No face detector available. Install one of:\n"
                                 "  pip install anime-face-detector\n"
                                 "  pip install face-alignment")
            backend = "fa"

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = sorted([f for f in src.iterdir() if f.suffix.lower() in exts])
    print(f"Processing {len(all_files)} images …")

    meta = []
    n_faces_total = 0
    n_no_face = 0

    for img_path in tqdm(all_files, desc="detecting faces"):
        sheet_id = img_path.stem  # unique per source image

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Detect
        if backend == "yolov3":
            boxes = detect_with_yolov3(yolo, img_bgr, args.score_thresh)
        elif backend == "cascade":
            boxes = detect_with_cascade(cascade, img_bgr)
            # retry with lower threshold if nothing found
            if not boxes:
                old_thresh = args.score_thresh
                boxes = detect_with_cascade.__wrapped__(cascade, img_bgr) \
                    if hasattr(detect_with_cascade, "__wrapped__") else []
        else:
            boxes = detect_with_fa(fa, np.array(img_pil))

        if not boxes:
            n_no_face += 1
            continue

        for face_idx, (x1, y1, x2, y2, score) in enumerate(boxes):
            crop = crop_face(img_pil, x1, y1, x2, y2,
                             pad=args.pad, crop_size=args.crop_size)
            out_name = f"{sheet_id}_face{face_idx:02d}.jpg"
            out_path = dst / out_name
            crop.save(out_path, "JPEG", quality=95)

            meta.append({
                "face_path":  str(out_path),
                "sheet_id":   sheet_id,
                "face_idx":   face_idx,
                "bbox":       [x1, y1, x2, y2],
                "score":      round(score, 4),
                "source_img": str(img_path),
            })
            n_faces_total += 1

    # Save metadata
    Path(args.meta).write_text(json.dumps(meta, indent=2))

    print(f"\n=== Detection summary ===")
    print(f"  Images processed: {len(all_files)}")
    print(f"  Faces found:      {n_faces_total}")
    print(f"  No-face images:   {n_no_face}")
    if len(all_files) > 0:
        avg = n_faces_total / max(1, len(all_files) - n_no_face)
        print(f"  Avg faces/sheet:  {avg:.1f}")
    print(f"  Metadata saved:   {args.meta}")


if __name__ == "__main__":
    main()
