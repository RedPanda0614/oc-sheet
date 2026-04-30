#!/bin/bash
#SBATCH --job-name=oc-p5-nocfg-vbg
#SBATCH --account=cis260099p
#SBATCH --partition=GPU-small
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_p5_nocfg_viewbg_eval500.log
#SBATCH --error=logs/%j_p5_nocfg_viewbg_eval500.err

set -euo pipefail

PROJECT_DIR="/ocean/projects/cis260099p/ezhang13/oc-sheet/project"
ENV_NAME="oc-sheet"

CHECKPOINT_DIR="checkpoints/p4_on_p1_3epochs_no"
RESULT_DIR="results/p5_p4_nocfg_viewbg_rerank_n500"
PAIRS_JSON="data/label_pairs/val.json"
MANIFEST_PATH="${RESULT_DIR}/manifest.json"
SKIP_TARGET_LABELS="neutral"
EXCLUDE_LABELS="embarrassed"

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
  --n 500 \
  --scale 0.7 \
  --disable-image-cfg \
  --image-cfg-scale 1.0 \
  --guidance 7.5 \
  --steps 30 \
  --seed 42 \
  --num-candidates 4 \
  --skip-target-labels "${SKIP_TARGET_LABELS}" \
  --target-view front \
  --w-view-hit 1.5 \
  --w-view-conf 0.5 \
  --view-mismatch-penalty 1.0 \
  --w-background 1.5 \
  --model-tag "p5_p4_nocfg_viewbg_rerank"

python eval/run_eval.py \
  --pairs-json "${PAIRS_JSON}" \
  --generated-dir "${RESULT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --output-json "${MAIN_OUTPUT_JSON}" \
  --exclude-expression-labels "${EXCLUDE_LABELS}" \
  --skip-fid

python eval/target_emotion_metrics.py \
  --pairs-json "${PAIRS_JSON}" \
  --generated-dir "${RESULT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --output-json "${TARGET_OUTPUT_JSON}" \
  --exclude-expression-labels "${EXCLUDE_LABELS}"

echo
echo "Saved P5 no-CFG view/background-reranked outputs to: ${PROJECT_DIR}/${RESULT_DIR}"
echo "Saved main metrics to: ${PROJECT_DIR}/${MAIN_OUTPUT_JSON}"
echo "Saved target-emotion metrics to: ${PROJECT_DIR}/${TARGET_OUTPUT_JSON}"
