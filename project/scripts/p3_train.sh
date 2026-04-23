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

python train/p3_finetune.py \
    --base_model    models/sd-v1-5 \
    --ip_ckpt       models/ip-adapter/models/ip-adapter-plus_sd15.bin \
    --image_encoder models/ip-adapter/models/image_encoder \
    --p1_ckpt       checkpoints/p1 \
    --train_json    data/pairs/train.json \
    --val_json      data/pairs/val.json \
    --output_dir    checkpoints/p3 \
    --lr            5e-5 \
    --batch_size    4 \
    --num_steps     5000 \
    --save_every    500 \
    --log_every     50 \
    --expr_weight   3.0 \
    --mask_sigma    30.0 \
    --image_size    512

echo "P3 training finished."
