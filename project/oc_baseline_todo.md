# OC Character Sheet — IP-Adapter Baseline 完整执行清单

> 目标：用 IP-Adapter 跑通"输入一张 OC 参考图 → 生成多种表情变体"的完整 pipeline，作为 Midway 的 Baseline 结果。
> 预计总时间：10天（3人并行）

---

## 项目目录结构

```
oc-character-sheet/
├── data/
│   ├── raw/                      # 从 SafeBooru 下载的原始图
│   │   ├── expression_sheet/
│   │   └── character_sheet/
│   ├── processed/                # 处理后的单张图
│   │   ├── expressions/          # 按表情分类的人脸图
│   │   └── multiview/            # 按视角分类的图
│   └── pairs/
│       ├── train.json            # 训练对
│       └── val.json              # 验证对
│
├── scripts/
│   ├── 01_download.sh            # 下载数据
│   ├── 02_filter.py              # 过滤低质量图
│   ├── 03_split_sheets.py        # 切割设定图为单格
│   ├── 04_crop_faces.py          # 人脸检测与裁剪
│   ├── 05_build_pairs.py         # 构建训练对 JSON
│   └── 06_visualize_pairs.py     # 抽样可视化，人工核查
│
├── inference/
│   ├── run_baseline.py           # 零样本 IP-Adapter 推理
│   └── batch_inference.py        # 批量推理，用于评估
│
├── eval/
│   ├── arcface_similarity.py     # 人脸身份一致性
│   ├── palette_distance.py       # 颜色一致性
│   ├── fid_score.py              # FID 分数
│   └── run_eval.py               # 统一评估入口
│
├── results/
│   ├── baseline/                 # 零样本推理结果图
│   └── metrics.json              # 所有评估数字
│
├── requirements.txt
└── README.md
```

---

## Phase 1：环境搭建（Day 1 上午，成员B）

### TODO 1.1 — 创建 conda 环境

```bash
conda create -n oc-sheet python=3.10
conda activate oc-sheet
```

### TODO 1.2 — 安装依赖

```bash
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.25.0 transformers accelerate
pip install face-alignment opencv-python Pillow tqdm
pip install insightface onnxruntime-gpu   # ArcFace 用
pip install clean-fid                     # FID 用
pip install gallery-dl                    # 数据下载用
```

创建 `requirements.txt`：

```
torch==2.1.0
diffusers==0.25.0
transformers
accelerate
face-alignment
opencv-python
Pillow
tqdm
insightface
onnxruntime-gpu
clean-fid
gallery-dl
```

### TODO 1.3 — 克隆 IP-Adapter 仓库并下载权重

```bash
git clone https://github.com/tencent-ailab/IP-Adapter
cd IP-Adapter

# 下载模型权重（约 10GB，提前开始）
# 从 HuggingFace 下载
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
# IP-Adapter 权重
snapshot_download(
    repo_id="h94/IP-Adapter",
    local_dir="./models/ip-adapter"
)
# SD1.5 基础模型
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="./models/sd-v1-5"
)
EOF
```

**验证环境：** 跑官方 demo notebook，能生成图即为成功。

---

## Phase 2：数据下载（Day 1 下午，成员A）

### TODO 2.1 — 下载 expression_sheet 数据

创建 `scripts/01_download.sh`：

```bash
#!/bin/bash
mkdir -p data/raw/expression_sheet
mkdir -p data/raw/character_sheet

# 下载表情合集图（优先，训练用）
gallery-dl \
  --range 1-3000 \
  --directory data/raw/expression_sheet \
  "https://safebooru.org/index.php?page=post&s=list&tags=expressions+multiple_views"

# 下载多视角设定图（后续视角实验用）
gallery-dl \
  --range 1-3000 \
  --directory data/raw/character_sheet \
  "https://safebooru.org/index.php?page=post&s=list&tags=character_sheet+multiple_views"

echo "下载完成"
echo "expression_sheet: $(ls data/raw/expression_sheet/*.jpg 2>/dev/null | wc -l) 张"
echo "character_sheet:  $(ls data/raw/character_sheet/*.jpg 2>/dev/null | wc -l) 张"
```

```bash
chmod +x scripts/01_download.sh
bash scripts/01_download.sh
```

---

## Phase 3：数据处理（Day 2-3，成员A）

### TODO 3.1 — 过滤低质量图

创建 `scripts/02_filter.py`：

```python
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
```

```bash
python scripts/02_filter.py
```

### ⚠️ 为什么放弃格子切割

原始方案（等分切割→按位置推断情绪标签）在真实数据上会大量失败：
- 表情顺序因作者而异，neutral 不一定在第一格
- 很多设定图是不规则排列（人物大小不同、带边框线、有文字标注区域）
- 面板之间可能有留白或装饰导致切割错位

**新方案：直接在整张设定图上跑人脸检测，用 bounding box 定位每张脸，不依赖任何格子假设。**
同一张设定图检出的所有脸 = 同一角色的不同表情，互相构成训练对。情绪标签对 IP-Adapter zero-shot 推理不是必须的（情绪通过文字 prompt 控制），留到 fine-tune 阶段再处理。

---

### TODO 3.2 — 从整图直接检测并裁剪所有人脸（替代原 03+04 两步）

安装动漫人脸检测器（对漫画图比 face_alignment 默认检测器好得多）：

```bash
pip install anime-face-detector
# 该库基于 mmdet，如遇安装问题可用备选方案（见下方常见问题）
```

创建 `scripts/03_detect_and_crop.py`：

```python
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── 检测器选择 ─────────────────────────────────────
# 优先用动漫专用检测器，失败则回退到通用检测器
try:
    from anime_face_detector import create_detector
    detector = create_detector('yolov3')
    DETECTOR_MODE = 'anime'
    print("使用 anime-face-detector")
except ImportError:
    import face_alignment
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cuda',
        face_detector='blazeface'   # blazeface 比 sfd 对漫画脸更友好
    )
    DETECTOR_MODE = 'fa'
    print("回退到 face_alignment (blazeface)")


def detect_faces_anime(img_np):
    """用 anime-face-detector 返回 bounding boxes 列表"""
    preds = detector(img_np)   # 返回 [{'bbox': [x1,y1,x2,y2,score], ...}]
    boxes = []
    for pred in preds:
        x1, y1, x2, y2, score = pred['bbox']
        if score > 0.5:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def detect_faces_fa(img_np):
    """用 face_alignment 返回 bounding boxes（从 landmarks 推断）"""
    landmarks_list = fa.get_landmarks(img_np)
    if landmarks_list is None:
        return []
    boxes = []
    for lm in landmarks_list:
        x_min, y_min = lm.min(axis=0).astype(int)
        x_max, y_max = lm.max(axis=0).astype(int)
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def crop_with_padding(img, box, padding=0.4, target_size=512):
    """根据 bounding box 加 padding 后裁剪，resize 到目标尺寸"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    x1 = max(0, x1 - int(bw * padding))
    y1 = max(0, y1 - int(bh * padding))
    x2 = min(img.width,  x2 + int(bw * padding))
    y2 = min(img.height, y2 + int(bh * padding))

    face = img.crop((x1, y1, x2, y2))
    face = face.resize((target_size, target_size), Image.LANCZOS)
    return face


def is_duplicate(box, existing_boxes, iou_threshold=0.3):
    """检查当前框是否和已有框重叠（过滤重复检测）"""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    for ex1, ey1, ex2, ey2 in existing_boxes:
        ix1, iy1 = max(x1, ex1), max(y1, ey1)
        ix2, iy2 = min(x2, ex2), min(y2, ey2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        ex_area = (ex2 - ex1) * (ey2 - ey1)
        iou = inter / (area + ex_area - inter + 1e-6)
        if iou > iou_threshold:
            return True
    return False


def process_sheet(img_path, out_dir, sheet_id, target_size=512):
    """处理一张设定图：检测所有人脸，裁剪保存，返回元数据列表"""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # 检测
    if DETECTOR_MODE == 'anime':
        boxes = detect_faces_anime(img_np)
    else:
        boxes = detect_faces_fa(img_np)

    if not boxes:
        return []

    # 过滤重叠框
    kept_boxes = []
    for box in boxes:
        if not is_duplicate(box, kept_boxes):
            kept_boxes.append(box)

    results = []
    for idx, box in enumerate(kept_boxes):
        face = crop_with_padding(img, box, padding=0.4, target_size=target_size)
        fname = f"{sheet_id}__face{idx:02d}.jpg"
        out_path = os.path.join(out_dir, fname)
        face.save(out_path, quality=95)

        results.append({
            "face_path": out_path,
            "sheet_id": sheet_id,
            "face_idx": idx,
            "bbox": box,
            "source_img": img_path,
        })

    return results


if __name__ == "__main__":
    src_dir = "data/filtered/expression_sheet"
    out_dir = "data/processed/faces"
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.png', '.webp'))]

    all_faces = []
    no_face_count = 0

    for i, fname in enumerate(tqdm(files, desc="Detecting faces")):
        sheet_id = f"sheet_{i:05d}"
        img_path = os.path.join(src_dir, fname)
        try:
            faces = process_sheet(img_path, out_dir, sheet_id)
            if faces:
                all_faces.extend(faces)
            else:
                no_face_count += 1
        except Exception as e:
            no_face_count += 1

    with open("data/processed/faces_meta.json", "w") as f:
        json.dump(all_faces, f, indent=2, ensure_ascii=False)

    print(f"\n共检测到 {len(all_faces)} 张人脸")
    print(f"未检出任何人脸的设定图：{no_face_count} 张")

    # 打印每张设定图平均检出人脸数
    from collections import Counter
    sheet_counts = Counter(f["sheet_id"] for f in all_faces)
    avg = sum(sheet_counts.values()) / len(sheet_counts) if sheet_counts else 0
    print(f"平均每张设定图检出 {avg:.1f} 张人脸")
```

```bash
python scripts/03_detect_and_crop.py
```

**检验输出是否合理：**
- 平均每张设定图检出 3~8 张脸为正常
- 如果平均只有 1 张，说明检测率太低，换检测器
- 如果平均超过 15 张，说明有误检，调高 score 阈值（`> 0.5` 改为 `> 0.65`）

### TODO 3.4 — 构建训练对

逻辑变得更简单：同一张设定图检出的所有人脸两两配对，face_0 当 reference，其余当 target。不需要情绪标签。

创建 `scripts/05_build_pairs.py`：

```python
import json
import random
import os
from collections import defaultdict

def build_pairs(meta_path, out_train, out_val,
                val_ratio=0.1, min_faces_per_sheet=2):
    with open(meta_path) as f:
        faces = json.load(f)

    # 按 sheet_id 分组
    by_sheet = defaultdict(list)
    for f in faces:
        by_sheet[f["sheet_id"]].append(f)

    pairs = []
    skipped_sheets = 0

    for sheet_id, sheet_faces in by_sheet.items():
        # 同一张设定图少于2张脸则跳过（无法构成对）
        if len(sheet_faces) < min_faces_per_sheet:
            skipped_sheets += 1
            continue

        # face_idx=0 的脸当 reference（检测时通常是最大/最显眼的脸）
        # 也可以随机选，两种方案都可以
        reference = sheet_faces[0]

        for target in sheet_faces[1:]:
            pairs.append({
                "reference_path": reference["face_path"],
                "target_path":    target["face_path"],
                "sheet_id":       sheet_id,
                # 情绪标签暂时留空，zero-shot 阶段不需要
                # fine-tune 阶段再用情绪分类器补充
                "target_emotion": "unknown",
            })

    print(f"有效设定图: {len(by_sheet) - skipped_sheets}")
    print(f"跳过（人脸数不足）: {skipped_sheets}")
    print(f"总训练对: {len(pairs)}")

    # 按 sheet_id 分割（保证同一张设定图不会同时出现在 train 和 val）
    all_sheet_ids = list(by_sheet.keys())
    random.shuffle(all_sheet_ids)
    split = int(len(all_sheet_ids) * (1 - val_ratio))
    train_sheets = set(all_sheet_ids[:split])

    train_pairs = [p for p in pairs if p["sheet_id"] in train_sheets]
    val_pairs   = [p for p in pairs if p["sheet_id"] not in train_sheets]

    os.makedirs("data/pairs", exist_ok=True)
    with open(out_train, "w") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)
    with open(out_val, "w") as f:
        json.dump(val_pairs, f, indent=2, ensure_ascii=False)

    print(f"训练对: {len(train_pairs)} | 验证对: {len(val_pairs)}")

if __name__ == "__main__":
    build_pairs(
        meta_path="data/processed/faces_meta.json",
        out_train="data/pairs/train.json",
        out_val="data/pairs/val.json",
    )
```

```bash
python scripts/05_build_pairs.py
```

> **注意：** 分割时按 sheet_id 而不是按 pair 随机切，是为了避免"同一张设定图的正面出现在 train、侧面出现在 val"，否则评估时会有数据泄露。

### TODO 3.5 — 可视化抽查（必做，别跳过）

创建 `scripts/06_visualize_pairs.py`：

```python
import json
import random
from PIL import Image, ImageDraw, ImageFont
import os

def visualize_pairs(pairs_path, out_dir, n_samples=20):
    os.makedirs(out_dir, exist_ok=True)
    with open(pairs_path) as f:
        pairs = json.load(f)

    samples = random.sample(pairs, min(n_samples, len(pairs)))

    for i, pair in enumerate(samples):
        ref = Image.open(pair["reference_path"]).resize((256, 256))
        tgt = Image.open(pair["target_path"]).resize((256, 256))

        canvas = Image.new("RGB", (512 + 20, 256 + 40), (240, 240, 240))
        canvas.paste(ref, (0, 40))
        canvas.paste(tgt, (256 + 20, 40))

        draw = ImageDraw.Draw(canvas)
        draw.text((80, 10),  "Reference (neutral)", fill=(0,0,0))
        draw.text((320, 10), f"Target ({pair['target_emotion']})", fill=(0,0,0))

        canvas.save(os.path.join(out_dir, f"pair_{i:03d}.jpg"))

    print(f"已保存 {len(samples)} 张可视化到 {out_dir}")

if __name__ == "__main__":
    visualize_pairs("data/pairs/val.json", "results/pair_checks")
```

```bash
python scripts/06_visualize_pairs.py
# 打开 results/pair_checks/ 目录，人工看看配对质量
```

**人工检查标准：**
- [ ] reference 和 target 确实是同一个角色
- [ ] 人脸裁剪没有切到只有一半
- [ ] 没有大量空白/纯色图混入
- [ ] 表情标签大致正确（不要求完美）

---

## Phase 4：零样本 IP-Adapter 推理（Day 4，成员B+C）

### TODO 4.1 — 基础推理脚本

创建 `inference/run_baseline.py`：

```python
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from ip_adapter import IPAdapter   # IP-Adapter 仓库提供
import os

# ── 配置 ──────────────────────────────────────────
BASE_MODEL    = "models/sd-v1-5"
IP_CKPT       = "models/ip-adapter/models/ip-adapter_sd15.bin"
IMAGE_ENCODER = "models/ip-adapter/models/image_encoder"
DEVICE        = "cuda"

EMOTION_PROMPTS = {
    "happy":      "manga character, smiling, happy expression, 1girl, high quality",
    "sad":        "manga character, sad expression, teary eyes, 1girl, high quality",
    "angry":      "manga character, angry expression, frowning, 1girl, high quality",
    "surprised":  "manga character, surprised expression, wide eyes, 1girl, high quality",
    "crying":     "manga character, crying, tears, 1girl, high quality",
    "embarrassed":"manga character, embarrassed, blushing, 1girl, high quality",
}

NEG_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed"

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
    ).to(DEVICE)

    ip_model = IPAdapter(pipe, IMAGE_ENCODER, IP_CKPT, DEVICE)
    return ip_model

def generate_expressions(reference_path, out_dir, scale=0.7, n_per_emotion=2):
    """给定一张参考图，生成所有情绪变体"""
    os.makedirs(out_dir, exist_ok=True)

    ip_model = load_model()
    reference = Image.open(reference_path).convert("RGB")
    reference.save(os.path.join(out_dir, "00_reference.jpg"))

    results = {}
    for emotion, prompt in EMOTION_PROMPTS.items():
        images = ip_model.generate(
            pil_image=reference,
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            scale=scale,           # IP-Adapter 强度：0=忽略参考图，1=完全依赖参考图
            num_samples=n_per_emotion,
            num_inference_steps=30,
            seed=42,
        )
        for i, img in enumerate(images):
            fname = f"{emotion}_{i:02d}.jpg"
            img.save(os.path.join(out_dir, fname))

        results[emotion] = images
        print(f"✓ {emotion}")

    return results

if __name__ == "__main__":
    import sys
    ref_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/faces/sheet_00001__neutral__00_face.jpg"
    generate_expressions(ref_path, out_dir="results/baseline/single_test")
    print("完成！查看 results/baseline/single_test/")
```

```bash
# 先用一张图测试
python inference/run_baseline.py data/processed/faces/sheet_00001__neutral__00_face.jpg
```

### TODO 4.2 — 不同 scale 参数对比实验

在 `inference/run_baseline.py` 末尾添加一个 scale 对比函数，测试 scale=0.3 / 0.5 / 0.7 / 1.0 的效果差异。这个对比图是 Midway 里的重要图表。

```python
def compare_scales(reference_path, out_dir, scales=[0.3, 0.5, 0.7, 1.0]):
    """对比不同 IP-Adapter scale 的效果"""
    os.makedirs(out_dir, exist_ok=True)
    ip_model = load_model()
    reference = Image.open(reference_path).convert("RGB")

    for scale in scales:
        images = ip_model.generate(
            pil_image=reference,
            prompt=EMOTION_PROMPTS["happy"],
            negative_prompt=NEG_PROMPT,
            scale=scale,
            num_samples=1,
            num_inference_steps=30,
            seed=42,
        )
        images[0].save(os.path.join(out_dir, f"scale_{scale:.1f}.jpg"))
        print(f"scale={scale} done")
```

---

## Phase 5：批量评估（Day 5，成员C）

### TODO 5.1 — ArcFace 相似度

创建 `eval/arcface_similarity.py`：

```python
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

def get_embedding(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    faces = app.get(img)
    if not faces:
        return None
    # 取最大的脸
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    return face.embedding  # 512维向量

def arcface_similarity(path1, path2):
    """返回两张图的人脸相似度（-1到1，越高越好）"""
    emb1 = get_embedding(path1)
    emb2 = get_embedding(path2)
    if emb1 is None or emb2 is None:
        return None
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(cos_sim)

if __name__ == "__main__":
    import sys
    score = arcface_similarity(sys.argv[1], sys.argv[2])
    print(f"ArcFace similarity: {score:.4f}")
```

### TODO 5.2 — 颜色一致性

创建 `eval/palette_distance.py`：

```python
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance

def get_color_histogram(img_path, bins=32):
    """提取 RGB 三通道颜色直方图"""
    img = np.array(Image.open(img_path).convert("RGB"))
    hists = []
    for c in range(3):
        hist, _ = np.histogram(img[:,:,c], bins=bins, range=(0, 256), density=True)
        hists.append(hist)
    return hists

def palette_distance(path1, path2):
    """Earth Mover Distance，越低越好"""
    h1 = get_color_histogram(path1)
    h2 = get_color_histogram(path2)
    bins = np.arange(len(h1[0]))
    dists = [wasserstein_distance(bins, bins, h1[c], h2[c]) for c in range(3)]
    return float(np.mean(dists))

if __name__ == "__main__":
    import sys
    d = palette_distance(sys.argv[1], sys.argv[2])
    print(f"Palette distance: {d:.4f}")
```

### TODO 5.3 — 统一评估入口

创建 `eval/run_eval.py`：

```python
import json
import os
from tqdm import tqdm
from arcface_similarity import arcface_similarity
from palette_distance import palette_distance

def evaluate_results(pairs_json, generated_dir, out_json="results/metrics.json"):
    """
    pairs_json:    验证对 JSON（reference + target ground truth）
    generated_dir: 生成图所在目录（命名规则：{emotion}_00.jpg）
    """
    with open(pairs_json) as f:
        pairs = json.load(f)

    arc_scores, pal_scores = [], []
    failed = 0

    for pair in tqdm(pairs[:200], desc="Evaluating"):   # 先跑200对
        ref_path = pair["reference_path"]
        emotion  = pair["target_emotion"]
        gen_path = os.path.join(generated_dir, f"{emotion}_00.jpg")

        if not os.path.exists(gen_path):
            failed += 1
            continue

        arc = arcface_similarity(ref_path, gen_path)
        pal = palette_distance(ref_path, gen_path)

        if arc is not None:
            arc_scores.append(arc)
        pal_scores.append(pal)

    metrics = {
        "arcface_mean":       float(sum(arc_scores) / len(arc_scores)) if arc_scores else 0,
        "arcface_std":        float(float(np.std(arc_scores))) if arc_scores else 0,
        "palette_dist_mean":  float(sum(pal_scores) / len(pal_scores)) if pal_scores else 0,
        "n_evaluated":        len(arc_scores),
        "n_failed":           failed,
    }

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    import numpy as np
    evaluate_results(
        pairs_json="data/pairs/val.json",
        generated_dir="results/baseline/batch",
        out_json="results/metrics_baseline.json"
    )
```

### TODO 5.4 — 批量推理（用于评估）

创建 `inference/batch_inference.py`：

```python
import json
import os
from run_baseline import load_model, EMOTION_PROMPTS, NEG_PROMPT
from PIL import Image
from tqdm import tqdm

def batch_generate(pairs_json, out_dir, scale=0.7, n=200):
    os.makedirs(out_dir, exist_ok=True)

    with open(pairs_json) as f:
        pairs = json.load(f)

    ip_model = load_model()

    for i, pair in enumerate(tqdm(pairs[:n])):
        ref_path = pair["reference_path"]
        emotion  = pair["target_emotion"]

        if emotion not in EMOTION_PROMPTS:
            continue

        reference = Image.open(ref_path).convert("RGB")
        images = ip_model.generate(
            pil_image=reference,
            prompt=EMOTION_PROMPTS[emotion],
            negative_prompt=NEG_PROMPT,
            scale=scale,
            num_samples=1,
            num_inference_steps=30,
            seed=i,
        )
        images[0].save(os.path.join(out_dir, f"{i:04d}_{emotion}_00.jpg"))

if __name__ == "__main__":
    batch_generate("data/pairs/val.json", "results/baseline/batch", scale=0.7, n=200)
```

```bash
python inference/batch_inference.py
python eval/run_eval.py
```

---

## Phase 6：整理结果（Day 6，全员）

### TODO 6.1 — 生成对比大图

把所有情绪的生成结果拼成一张展示图：

```python
# 在 inference/run_baseline.py 里加一个函数
def make_result_grid(ref_path, results_dir, out_path):
    """生成 1(参考) + 6(表情) 的展示大图"""
    emotions = list(EMOTION_PROMPTS.keys())
    ref = Image.open(ref_path).resize((256,256))

    total_w = 256 * (len(emotions) + 1) + 10 * len(emotions)
    canvas = Image.new("RGB", (total_w, 300), (255,255,255))
    canvas.paste(ref, (0, 44))

    draw = ImageDraw.Draw(canvas)
    draw.text((80, 10), "Reference", fill=(0,0,0))

    for i, emotion in enumerate(emotions):
        img_path = os.path.join(results_dir, f"{emotion}_00.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((256,256))
            x = 256 * (i+1) + 10 * (i+1)
            canvas.paste(img, (x, 44))
            draw.text((x + 60, 10), emotion, fill=(0,0,0))

    canvas.save(out_path)
    print(f"展示图已保存到 {out_path}")
```

### TODO 6.2 — 汇总 metrics.json

确保 `results/metrics_baseline.json` 里包含以下字段，供 Midway 直接引用：

```json
{
  "method": "IP-Adapter (zero-shot, scale=0.7)",
  "base_model": "SD1.5",
  "n_evaluated": 200,
  "arcface_mean": 0.XX,
  "arcface_std": 0.XX,
  "palette_dist_mean": 0.XX,
  "notes": "零样本，无任何微调"
}
```

---

## Midway 时需要准备的成果清单

| 类型 | 内容 | 存放位置 |
|---|---|---|
| 数据 | 处理好的训练对，数量 ≥ 5000 对 | `data/pairs/train.json` |
| 定性结果 | 至少 3 个不同 OC 的 6 表情展示图 | `results/baseline/` |
| 定量结果 | ArcFace 相似度 + Palette Distance | `results/metrics_baseline.json` |
| Scale 对比 | 4种 scale 的效果对比图 | `results/baseline/scale_compare/` |
| 代码 | 以上所有脚本跑通，有 README | 整个项目目录 |

---

## 常见问题与解决方案

**Q: `anime-face-detector` 安装失败怎么办？**
A: 该库依赖 `mmdet`，安装略复杂。按以下顺序尝试备选方案：

```bash
# 备选1：直接装 mmdet 后再装
pip install openmim && mim install mmdet && pip install anime-face-detector

# 备选2：用 face_alignment 的 blazeface 检测器（代码已自动回退）
pip install face-alignment
# 在 03_detect_and_crop.py 里把 DETECTOR_MODE 强制改为 'fa'

# 备选3：用 OpenCV 的 lbpcascade_animeface（最轻量，检测率稍低）
# 下载 lbpcascade_animeface.xml：
# https://github.com/nagadomi/lbpcascade_animeface
import cv2
cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24,24))
# faces 返回 [(x, y, w, h), ...] 格式
```

**Q: 检测到的脸有重叠（同一张脸被检出两次）怎么办？**
A: 代码里已有 `is_duplicate()` 函数做 IoU 过滤，阈值默认 0.3。如果仍然有重复，把阈值降到 0.2，或者在检测后按 bbox 面积排序只保留最大的 N 个框。

**Q: IP-Adapter 生成的图根本不像参考角色怎么办？**
A: 调高 `scale` 参数（试 0.8~1.0）；检查参考图质量（模糊图效果差）；换用 `ip-adapter-plus_sd15.bin`（plus 版对细节保留更好）。

**Q: GPU 显存不够（OOM）怎么办？**
A: 在 `StableDiffusionPipeline` 加 `enable_attention_slicing()`；把 `num_inference_steps` 降到 20；把参考图 resize 到 256 再送入。

**Q: SafeBooru 下载很慢或中断怎么办？**
A: `gallery-dl` 支持断点续传，重新跑同一命令即可。或者切换到 Danbooru 的 HuggingFace 镜像（`nyanko7/danbooru2023-subset`）直接用代码下载。
