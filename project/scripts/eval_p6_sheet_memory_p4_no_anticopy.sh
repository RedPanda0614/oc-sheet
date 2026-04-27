#!/bin/bash
#SBATCH --job-name=oc-eval-p6-noac
#SBATCH --account=cis260099p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j_eval_p6_sheet_memory_p4_no_anticopy.log
#SBATCH --error=logs/%j_eval_p6_sheet_memory_p4_no_anticopy.err

set -euo pipefail

# Bridges2 batch entrypoint for evaluating the P6 sheet-memory results
# generated from the P4 no-anticopy checkpoint.
#
# This launches both:
# 1. the main proposal-aligned evaluation (`run_eval.py`)
# 2. the target-emotion accuracy evaluation (`target_emotion_metrics.py`)
#
# Notes:
# - CLIP-based expression evaluation will use GPU if available.
# - ArcFace / insightface may still fall back to CPU if the local
#   onnxruntime-gpu build does not match the node's CUDA driver.
# - We exclude "embarrassed" from the CLIP label set because the current
#   zero-shot CLIP evaluator only defines neutral/happy/sad/angry/surprised/crying.

PROJECT_DIR="/ocean/projects/cis260099p/ezhang13/oc-sheet/project"
ENV_NAME="oc-sheet"

RESULT_DIR="results/p6_sheet_memory_labeled_p4_no_anticopy"
PAIRS_JSON="data/label_pairs/val.json"
MANIFEST_PATH="${RESULT_DIR}/manifest.json"
OUTPUT_DIR="results/eval"
EXCLUDE_LABELS="embarrassed"

MAIN_OUTPUT_JSON="${OUTPUT_DIR}/metrics_p6_sheet_memory_labeled_p4_no_anticopy.json"
TARGET_OUTPUT_JSON="${OUTPUT_DIR}/target_emotion_metrics_p6_sheet_memory_labeled_p4_no_anticopy.json"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module load cuda/11.8
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
nvidia-smi

required_paths=(
  "${PAIRS_JSON}"
  "${RESULT_DIR}"
  "${MANIFEST_PATH}"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${PROJECT_DIR}/${path}"
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

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
echo "Saved main metrics to: ${PROJECT_DIR}/${MAIN_OUTPUT_JSON}"
echo "Saved target-emotion metrics to: ${PROJECT_DIR}/${TARGET_OUTPUT_JSON}"
