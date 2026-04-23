#!/bin/bash
# jobs/p3_train.sh
#SBATCH --job-name=oc-p3
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j_p3.log
#SBATCH --error=logs/%j_p3.err

set -e
mkdir -p logs

module load cuda/11.8
module load anaconda3
conda activate /ocean/projects/cis260099p/sliu45/myenv

cd /ocean/projects/cis260099p/sliu45/oc-character-sheet

# P3 should resume from the Stage-1 IP-Adapter fine-tune outputs (P1).
P1_DIR=/ocean/projects/cis260099p/sliu45/project/results/ip_adapter_finetune

python scripts/p3_finetune.py \
    --pretrained_model models/sd-v1-5 \
    --ip_repo_path     models/ip-adapter \
    --ip_weight        ip-adapter-plus_sd15.bin \
    --image_proj_ckpt  ${P1_DIR}/image_proj_model.pt \
    --ip_attn_ckpt     ${P1_DIR}/ip_attn_procs.pt \
    --train_json       data/label_pairs/train.json \
    --val_json         data/label_pairs/val.json \
    --output_dir       checkpoints/p3 \
    --lr               5e-5 \
    --batch_size       2 \
    --num_steps        5000 \
    --save_every       500 \
    --log_every        50 \
    --expr_weight      3.0 \
    --mask_sigma       4.0 \
    --image_size       512

echo "P3 training finished."
