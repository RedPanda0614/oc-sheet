#!/bin/bash
#SBATCH --job-name=oc-p5-face-vbg
#SBATCH --account=cis260099p
#SBATCH --partition=GPU-small
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_p5_nocfg_viewbg_face_prompt_6expr_demo.log
#SBATCH --error=logs/%j_p5_nocfg_viewbg_face_prompt_6expr_demo.err

set -euo pipefail

PROJECT_DIR="/ocean/projects/cis260099p/ezhang13/oc-sheet/project"
ENV_NAME="oc-sheet"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/p4_on_p1_3epochs_no}"
RESULT_DIR="${RESULT_DIR:-results/p5_nocfg_viewbg_face_prompt_multi_ref_6expr_demo}"
SOURCE_PAIRS_JSON="${SOURCE_PAIRS_JSON:-results/p5_cfg110_multi_ref_5expr_demo/pairs.json}"
REF_SHEET_IDS="${REF_SHEET_IDS:-}"
IP_SCALE="${IP_SCALE:-0.7}"
STEPS="${STEPS:-30}"
SEED="${SEED:-20260430}"
PAIRS_JSON="${RESULT_DIR}/pairs.json"
MANIFEST_PATH="${RESULT_DIR}/manifest.json"
MAIN_OUTPUT_JSON="${RESULT_DIR}/eval_metrics_skipfid.json"
TARGET_OUTPUT_JSON="${RESULT_DIR}/target_emotion_metrics.json"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/${RESULT_DIR}"
cd "${PROJECT_DIR}"

export RESULT_DIR SOURCE_PAIRS_JSON REF_SHEET_IDS
python3 - <<'PY'
import json
from collections import OrderedDict
import os
from pathlib import Path

refs = OrderedDict()
ref_sheet_ids = [s.strip() for s in os.environ["REF_SHEET_IDS"].split(",") if s.strip()]
if ref_sheet_ids:
    for sheet_id in ref_sheet_ids:
        clean_sheet_id = sheet_id[:-5] if sheet_id.endswith("_demo") else sheet_id
        reference_path = Path("data/processed/faces") / f"{clean_sheet_id}__face00.jpg"
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        refs[f"{clean_sheet_id}_demo"] = str(reference_path)
else:
    source = Path(os.environ["SOURCE_PAIRS_JSON"])
    source_pairs = json.loads(source.read_text())
    for pair in source_pairs:
        refs.setdefault(pair["sheet_id"], pair["reference_path"])

emotions = ["neutral", "happy", "sad", "angry", "surprised", "crying"]
pairs = []
for sheet_id, reference_path in refs.items():
    for emotion in emotions:
        pairs.append(
            {
                "reference_path": str(reference_path),
                "target_path": None,
                "sheet_id": sheet_id,
                "target_emotion": emotion,
            }
        )

out = Path(os.environ["RESULT_DIR"]) / "pairs.json"
out.write_text(json.dumps(pairs, indent=2))
PY

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
  --scale "${IP_SCALE}" \
  --disable-image-cfg \
  --image-cfg-scale 1.0 \
  --guidance 7.5 \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --num-candidates 4 \
  --prompt-style face \
  --target-view front \
  --w-view-hit 1.5 \
  --w-view-conf 0.5 \
  --view-mismatch-penalty 1.0 \
  --w-background 1.5 \
  --model-tag "p5_nocfg_viewbg_face_prompt_multi_ref_6expr_demo"

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
import os
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(os.environ["RESULT_DIR"])
records = json.loads((root / "manifest.json").read_text())["records"]
by_sheet = defaultdict(list)
for record in records:
    by_sheet[record["sheet_id"]].append(record)

emotions = ["neutral", "happy", "sad", "angry", "surprised", "crying"]
thumb = 176
label_h = 26
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
    draw.text((pad, y + thumb + 6), sheet_id.replace("_demo", " ref"), fill=(0, 0, 0))

    by_label = {record["requested_label"]: record for record in sheet_records}
    for col, emotion in enumerate(emotions, start=1):
        x = pad + col * (thumb + pad)
        record = by_label[emotion]
        image = Image.open(record["generated_path"]).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        canvas.paste(image, (x, y))
        draw.text((x, y + thumb + 6), emotion, fill=(0, 0, 0))

canvas.save(root / "contact_sheet.jpg", quality=95)
PY

echo
echo "Saved P5 no-CFG view/background face-prompt demo to: ${PROJECT_DIR}/${RESULT_DIR}"
echo "Contact sheet: ${PROJECT_DIR}/${RESULT_DIR}/contact_sheet.jpg"
