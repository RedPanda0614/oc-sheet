#!/bin/bash
#SBATCH --job-name=oc-p5-5expr
#SBATCH --account=cis260099p
#SBATCH --partition=GPU-small
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_p5_cfg110_single_ref_5expr.log
#SBATCH --error=logs/%j_p5_cfg110_single_ref_5expr.err

set -euo pipefail

PROJECT_DIR="/ocean/projects/cis260099p/ezhang13/oc-sheet/project"
ENV_NAME="oc-sheet"

CHECKPOINT_DIR="checkpoints/p4_on_p1_3epochs_no"
RESULT_DIR="results/p5_cfg110_single_ref_sheet00008_5expr"
PAIRS_JSON="${RESULT_DIR}/pairs.json"
MANIFEST_PATH="${RESULT_DIR}/manifest.json"
MAIN_OUTPUT_JSON="${RESULT_DIR}/eval_metrics_skipfid.json"
TARGET_OUTPUT_JSON="${RESULT_DIR}/target_emotion_metrics.json"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module load anaconda3
if module --ignore-cache load cuda/11.8; then
  echo "Loaded cuda/11.8 module"
else
  echo "cuda/11.8 module unavailable; continuing with conda/PyTorch CUDA runtime"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

required_paths=(
  "models/sd-v1-5"
  "models/ip-adapter"
  "${PAIRS_JSON}"
  "${CHECKPOINT_DIR}/image_proj_model.pt"
  "${CHECKPOINT_DIR}/ip_attn_procs.pt"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${PROJECT_DIR}/${path}"
    exit 1
  fi
done

python inference/batch_inference_p5_rerank_labeled.py \
  --pairs-json "${PAIRS_JSON}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${RESULT_DIR}" \
  --n 0 \
  --scale 0.7 \
  --image-cfg-scale 1.10 \
  --guidance 7.5 \
  --steps 30 \
  --seed 20260430 \
  --num-candidates 4 \
  --model-tag "p5_cfg110_single_ref_5expr"

python eval/run_eval.py \
  --pairs-json "${PAIRS_JSON}" \
  --generated-dir "${RESULT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --output-json "${MAIN_OUTPUT_JSON}" \
  --exclude-expression-labels "embarrassed" \
  --skip-fid

python eval/target_emotion_metrics.py \
  --pairs-json "${PAIRS_JSON}" \
  --generated-dir "${RESULT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --output-json "${TARGET_OUTPUT_JSON}" \
  --exclude-expression-labels "embarrassed"

python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path("results/p5_cfg110_single_ref_sheet00008_5expr")
reference = Image.open("data/processed/faces/sheet_00008__face00.jpg").convert("RGB")
images = [("reference", reference)]
for path in sorted(root.glob("*.jpg")):
    images.append((path.stem.split("_")[-1], Image.open(path).convert("RGB")))

thumb_w = 192
label_h = 28
pad = 12
canvas_w = pad + len(images) * (thumb_w + pad)
canvas_h = pad + thumb_w + label_h + pad
canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

for idx, (label, image) in enumerate(images):
    image = image.resize((thumb_w, thumb_w), Image.LANCZOS)
    x = pad + idx * (thumb_w + pad)
    canvas.paste(image, (x, pad))
    draw.text((x, pad + thumb_w + 6), label, fill=(0, 0, 0))

canvas.save(root / "contact_sheet.jpg", quality=95)
PY

echo
echo "Saved single-reference P5 CFG110 outputs to: ${PROJECT_DIR}/${RESULT_DIR}"
echo "Contact sheet: ${PROJECT_DIR}/${RESULT_DIR}/contact_sheet.jpg"
echo "Manifest: ${PROJECT_DIR}/${MANIFEST_PATH}"
