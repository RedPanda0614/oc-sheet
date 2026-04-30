#!/bin/bash
#SBATCH --job-name=oc-p5-demo
#SBATCH --account=cis260099p
#SBATCH --partition=GPU-small
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_p5_cfg110_multi_ref_5expr_demo.log
#SBATCH --error=logs/%j_p5_cfg110_multi_ref_5expr_demo.err

set -euo pipefail

PROJECT_DIR="/ocean/projects/cis260099p/ezhang13/oc-sheet/project"
ENV_NAME="oc-sheet"

CHECKPOINT_DIR="checkpoints/p4_on_p1_3epochs_no"
RESULT_DIR="results/p5_cfg110_multi_ref_5expr_demo"
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
  --seed 20260431 \
  --num-candidates 4 \
  --model-tag "p5_cfg110_multi_ref_5expr_demo"

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
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("results/p5_cfg110_multi_ref_5expr_demo")
records = json.loads((root / "manifest.json").read_text())["records"]
by_sheet = defaultdict(list)
for record in records:
    by_sheet[record["sheet_id"]].append(record)

emotions = ["happy", "sad", "angry", "surprised", "crying"]
thumb = 160
label_h = 24
pad = 10
cols = 1 + len(emotions)
rows = len(by_sheet)
canvas = Image.new(
    "RGB",
    (pad + cols * (thumb + pad), pad + rows * (thumb + label_h + pad)),
    "white",
)
draw = ImageDraw.Draw(canvas)

for row, (sheet_id, sheet_records) in enumerate(sorted(by_sheet.items())):
    y = pad + row * (thumb + label_h + pad)
    ref_path = sheet_records[0]["reference_path"]
    ref = Image.open(ref_path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
    canvas.paste(ref, (pad, y))
    draw.text((pad, y + thumb + 5), sheet_id.replace("_demo", " ref"), fill=(0, 0, 0))

    by_label = {record["requested_label"]: record for record in sheet_records}
    for col, emotion in enumerate(emotions, start=1):
        x = pad + col * (thumb + pad)
        record = by_label[emotion]
        image = Image.open(record["generated_path"]).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        canvas.paste(image, (x, y))
        draw.text((x, y + thumb + 5), emotion, fill=(0, 0, 0))

canvas.save(root / "contact_sheet.jpg", quality=95)
PY

echo
echo "Saved multi-reference P5 CFG110 demo to: ${PROJECT_DIR}/${RESULT_DIR}"
echo "Contact sheet: ${PROJECT_DIR}/${RESULT_DIR}/contact_sheet.jpg"
