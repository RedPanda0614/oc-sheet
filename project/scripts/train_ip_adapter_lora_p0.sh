#!/bin/bash
#SBATCH --job-name=oc-iplora-p0
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_ip_adapter_lora_p0.log
#SBATCH --error=logs/%j_ip_adapter_lora_p0.err

set -euo pipefail

# Bridges2 batch entrypoint for the P0-based IP-Adapter + LoRA experiment.
# This keeps the training itself unchanged and only handles:
# - requesting a GPU node
# - activating the user's conda env
# - checking required local model/data paths
# - launching the 3-epoch run

PROJECT_DIR="/ocean/projects/cis260099p/ezhang13/oc-sheet/project"
ENV_NAME="oc-sheet"

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
  "models/sd-v1-5"
  "models/ip-adapter"
  "data/label_pairs/train.json"
  "data/processed/faces"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${PROJECT_DIR}/${path}"
    exit 1
  fi
done

python scripts/train_ip_adapter_lora.py \
  --pairs-json data/label_pairs/train.json \
  --batch-size 2 \
  --epochs 3 \
  --lr 5e-5 \
  --seed 42 \
  --output-dir results/ip_adapter_lora_p0_3epochs
