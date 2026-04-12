import os
from PIL import Image
from tqdm import tqdm
import shutil

def is_valid(img_path, min_size=400):
    """过滤掉太小、损坏、非RGB的图"""
    try:
        img = Image.open(img_path)
        w, h = img.size
        if w < min_size or h < min_size:
            return False, f"too small: {w}x{h}"
        if img.mode not in ["RGB", "RGBA"]:
            return False, f"wrong mode: {img.mode}"
        return True, "ok"
    except Exception as e:
        return False, str(e)

def filter_directory(src_dir, dst_dir, min_size=400):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.png', '.webp'))]

    kept, dropped = 0, 0
    for fname in tqdm(files, desc=f"Filtering {src_dir}"):
        src = os.path.join(src_dir, fname)
        valid, reason = is_valid(src, min_size)
        if valid:
            shutil.copy(src, os.path.join(dst_dir, fname))
            kept += 1
        else:
            dropped += 1

    print(f"保留: {kept} | 丢弃: {dropped}")

if __name__ == "__main__":
    filter_directory("data/raw/expression_sheet", "data/filtered/expression_sheet")
    filter_directory("data/raw/character_sheet",  "data/filtered/character_sheet")