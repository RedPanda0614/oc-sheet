import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ================= 配置区 =================
SRC_DIR = "data/filtered/expression_sheet"  # 原始图目录
OUT_DIR = "data/processed/faces"            # 头像保存目录
META_FILE = "data/processed/faces_meta.json" # JSON 保存路径
TARGET_SIZE = 512                           
PADDING = 0.4                               
CONFIDENCE_THRESHOLD = 0.5                  
# =========================================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(META_FILE) if os.path.dirname(META_FILE) else '.', exist_ok=True)

print("🚀 正在加载 YOLOv8 二次元人脸模型...")
model_path = hf_hub_download(repo_id="Fuyucchi/yolov8_animeface", filename="yolov8x6_animeface.pt")
model = YOLO(model_path)

def is_duplicate(box, existing_boxes, iou_threshold=0.3):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    for ex1, ey1, ex2, ey2 in existing_boxes:
        ix1, iy1 = max(x1, ex1), max(y1, ey1)
        ix2, iy2 = min(x2, ex2), min(y2, ey2)
        if ix2 <= ix1 or iy2 <= iy1: continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        ex_area = (ex2 - ex1) * (ey2 - ey1)
        iou = inter / (area + ex_area - inter + 1e-6)
        if iou > iou_threshold: return True
    return False

def crop_with_padding(pil_img, box, padding=0.4, target_size=512):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px1 = max(0, x1 - int(w * padding))
    py1 = max(0, y1 - int(h * padding))
    px2 = min(pil_img.width, x2 + int(w * padding))
    py2 = min(pil_img.height, y2 + int(h * padding))
    face = pil_img.crop((px1, py1, px2, py2))
    face = face.resize((target_size, target_size), Image.LANCZOS)
    return face

if __name__ == "__main__":
    files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.png', '.webp'))]
    all_faces = [] # 所有的 meta 数据都存入这个列表
    no_face_count = 0

    print("✅ 开始处理...")
    for i, fname in enumerate(tqdm(files, desc="YOLO Detecting")):
        sheet_id = f"sheet_{i:05d}"
        img_path = os.path.join(SRC_DIR, fname)
        
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            results = model(img_cv, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            raw_boxes = []
            for box in results[0].boxes:
                coords = map(int, box.xyxy[0].tolist())
                raw_boxes.append(tuple(coords))
            
            kept_boxes = []
            for box in raw_boxes:
                if not is_duplicate(box, kept_boxes):
                    kept_boxes.append(box)
            
            if not kept_boxes:
                no_face_count += 1
                continue
                
            for idx, box in enumerate(kept_boxes):
                face_img = crop_with_padding(img_pil, box, padding=PADDING, target_size=TARGET_SIZE)
                face_name = f"{sheet_id}__face{idx:02d}.jpg"
                face_path = os.path.join(OUT_DIR, face_name)
                face_img.save(face_path, quality=95)
                
                # 统一存入 all_faces
                all_faces.append({
                    "face_path": face_path,
                    "sheet_id": sheet_id,
                    "face_idx": idx,
                    "bbox": list(box), 
                    "source_img": img_path,
                })
                
        except Exception as e:
            tqdm.write(f"❌ 处理 {fname} 出错: {e}")
            no_face_count += 1

    # --- 核心保存步骤：只保存一次，确保数据完整 ---
    print(f"\n💾 正在保存 Meta 数据到 {META_FILE}...")
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_faces, f, indent=2, ensure_ascii=False)

    print(f"🎉 全部处理完成！共检测到 {len(all_faces)} 张人脸，未检出 {no_face_count} 张。")