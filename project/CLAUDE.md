# CLAUDE.md — OC Character Sheet Generation

## Project Overview

This project builds a system for **manga OC (Original Character) character sheet generation**: given 1–3 reference images of a character, generate a complete set of face crops showing different expressions, views, or poses — all consistent in identity.

**Current phase:** Midway baseline — run IP-Adapter zero-shot to establish a quantitative baseline before any fine-tuning.

**Course context:** 10-423/623/723 Generative AI, Spring 2026, CMU. Midway Executive Summary due April 13.

---

## Repository Structure

```
oc-character-sheet/
├── CLAUDE.md                     # ← this file
├── requirements.txt
│
├── data/
│   ├── raw/                      # downloaded from SafeBooru, do not edit
│   │   ├── expression_sheet/
│   │   └── character_sheet/
│   ├── filtered/                 # after 02_filter.py
│   ├── processed/
│   │   ├── faces/                # individual face crops (512x512)
│   │   └── faces_meta.json       # [{face_path, sheet_id, face_idx, bbox, source_img}]
│   └── pairs/
│       ├── train.json            # [{reference_path, target_path, sheet_id}]
│       └── val.json
│
├── scripts/                      # run in order: 01 → 02 → 03 → 05 → 06
│   ├── 01_download.sh            # gallery-dl from SafeBooru
│   ├── 02_filter.py              # drop images < 400px or corrupted
│   ├── 03_detect_and_crop.py     # face detection on full sheet → 512x512 crops
│   ├── 05_build_pairs.py         # group by sheet_id → (reference, target) pairs
│   └── 06_visualize_pairs.py     # sanity check: render sample pairs side-by-side
│
├── inference/
│   ├── run_baseline.py           # single reference image → 6 expression variants
│   └── batch_inference.py        # run over val set for evaluation
│
├── eval/
│   ├── arcface_similarity.py     # identity consistency metric
│   ├── palette_distance.py       # color consistency metric
│   └── run_eval.py               # unified evaluation entry point → results/metrics.json
│
└── results/
    ├── baseline/                 # generated images from zero-shot IP-Adapter
    └── metrics_baseline.json     # final numbers for Midway report
```

---

## Key Design Decisions

### Why face detection instead of grid-splitting
Expression sheets have no consistent panel order, irregular sizes, and overlapping borders. Instead of slicing by grid, we run a face detector on the full image and crop each detected face independently. All faces from the same sheet are assumed to be the same character.

### Why no emotion labels for now
IP-Adapter zero-shot controls expression via **text prompt**, not training labels. Labels are not needed until fine-tuning. For now every pair has `"target_emotion": "unknown"`.

### Pair splitting strategy
Pairs are split by `sheet_id`, not randomly by pair. This prevents the same character appearing in both train and val (data leakage).

### Detector priority
1. `anime-face-detector` (yolov3) — best for manga/anime faces
2. `face_alignment` with `blazeface` — fallback if mmdet fails
3. `lbpcascade_animeface` (OpenCV) — last resort, lightest

---

## Compute: PSC Bridges-2

All training and heavy inference runs on **PSC Bridges-2** via SSH.
Reference: https://www.psc.edu/about-using-ssh/

### Login

```bash
ssh sliu45@bridges2.psc.edu
```

Add this to `~/.ssh/config` locally to avoid typing the full hostname every time:

```
Host bridges2
    HostName bridges2.psc.edu
    User sliu45
    ServerAliveInterval 60
```

Then simply: `ssh bridges2`

### Storage on Bridges-2

| Location | Path | Quota | Use for |
|---|---|---|---|
| Home | `$HOME` | 25 GB | code, scripts, CLAUDE.md |
| Project | `$PROJECT` | shared allocation | datasets, model weights |
| Ocean (scratch) | `/ocean/projects/cis260099p/sliu45` | large | generated images, checkpoints |

**Always store large data (raw images, model weights, results) in `/ocean/`, not `$HOME`.**

Sync code from local to Bridges-2:
```bash
# From your local machine
rsync -avz --exclude='data/' --exclude='results/' \
  ./oc-character-sheet/ bridges2:/ocean/projects/cis260099p/sliu45/oc-character-sheet/
```

### Job Scheduler: SLURM

Bridges-2 uses SLURM. Never run heavy compute on the login node.

#### Interactive GPU session (for debugging, quick tests)

```bash
# Request 1 GPU for 2 hours (GPU-shared partition, good for inference)
interact --gpu --ntasks-per-node=1 --gres=gpu:v100-32:1 -t 02:00:00 -p GPU-shared

# For training (full V100-32GB node)
interact --gpu --ntasks-per-node=1 --gres=gpu:v100-32:1 -t 04:00:00 -p GPU
```

#### Batch job (for long training runs)

Create `jobs/run_inference.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=oc-baseline
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_inference.log
#SBATCH --error=logs/%j_inference.err

# Load modules
module load cuda/11.8
module load anaconda3

# Activate environment
conda activate oc-sheet

# Run
cd /ocean/projects/cis260099p/sliu45/oc-character-sheet
python inference/batch_inference.py
```

```bash
mkdir -p logs
sbatch jobs/run_inference.sh

# Monitor jobs
squeue -u $USER          # list your running jobs
squeue -j <job_id>       # specific job status
scancel <job_id>         # cancel a job
```

#### Useful SLURM commands

```bash
sinfo -p GPU-shared      # check GPU-shared partition availability
sacct -u $USER           # job history
seff <job_id>            # efficiency report after job finishes (CPU/GPU utilization)
```

### Module System

```bash
# Always load these before running anything GPU-related
module load cuda/11.8
module load anaconda3

# Check available modules
module avail cuda
module avail python

# Save current module state (so you don't have to reload each session)
module save my_default
module restore my_default
```

### Environment Setup on Bridges-2 (first time only)

```bash
# SSH into Bridges-2
ssh bridges2

# Load modules
module load cuda/11.8 anaconda3

# Create conda environment in Ocean (not $HOME — quota is too small)
conda create --prefix /ocean/projects/cis260099p/sliu45/envs/oc-sheet python=3.10
conda activate /ocean/projects/cis260099p/sliu45/envs/oc-sheet

# Install dependencies
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.25.0 transformers accelerate
pip install face-alignment opencv-python Pillow tqdm
pip install insightface onnxruntime-gpu
pip install clean-fid gallery-dl

# Clone IP-Adapter
cd /ocean/projects/cis260099p/sliu45/

git clone https://github.com/tencent-ailab/IP-Adapter

# Download model weights (do this in an interactive session, not login node)
interact --gpu -t 01:00:00 -p GPU-shared --gres=gpu:v100-32:1
python -c "
from huggingface_hub import snapshot_download
snapshot_download('h94/IP-Adapter', local_dir='models/ip-adapter')
snapshot_download('runwayml/stable-diffusion-v1-5', local_dir='models/sd-v1-5')
"
```

### Tips for PSC

- **Login node is for editing files and submitting jobs only.** Running Python on the login node will get your account flagged.
- **Always check your allocation balance:** `projects` command on Bridges-2.
- The `GPU-shared` partition shares a single GPU between multiple users — fine for inference, may be slow for training. Use `GPU` partition for dedicated access.
- V100-32GB GPUs are sufficient for SD1.5 + IP-Adapter inference and fine-tuning.
- Use `tmux` on the login node so your session persists if you disconnect:
  ```bash
  tmux new -s main      # start session
  tmux attach -t main   # reattach after disconnect
  ```

---

## Local Environment (for code editing only)

```bash
conda activate oc-sheet   # local env for linting/editing, no GPU needed locally

# GPU check (on Bridges-2 only)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True, Tesla V100-SXM2-32GB
```

**Python:** 3.10
**PyTorch:** 2.1.0 + cu118
**Key libs:** diffusers==0.25.0, transformers, face-alignment, insightface, clean-fid

---

## Running the Pipeline

### Step 1 — Download data
```bash
bash scripts/01_download.sh
# Target: ~3000 expression_sheet images in data/raw/expression_sheet/
```

### Step 2 — Filter
```bash
python scripts/02_filter.py
# Drops: images < 400px, corrupted files, non-RGB
```

### Step 3 — Detect and crop faces
```bash
python scripts/03_detect_and_crop.py
# Output: data/processed/faces/*.jpg (512x512)
#         data/processed/faces_meta.json
# Healthy output: avg 3–8 faces per sheet
```

### Step 4 — Build pairs
```bash
python scripts/05_build_pairs.py
# Output: data/pairs/train.json, data/pairs/val.json
```

### Step 5 — Sanity check (always do this before training/inference)
```bash
python scripts/06_visualize_pairs.py
# Opens results/pair_checks/ — manually verify ~20 pairs look correct
# Red flags: different characters paired together, half-cropped faces, blank images
```

### Step 6 — Run zero-shot baseline
```bash
# Single test (fast, do this first)
python inference/run_baseline.py path/to/reference.jpg

# Batch inference over val set
python inference/batch_inference.py
```

### Step 7 — Evaluate
```bash
python eval/run_eval.py
# Output: results/metrics_baseline.json
# Key numbers: arcface_mean, palette_dist_mean
```

---

## IP-Adapter Configuration

Models live in `models/` (downloaded from HuggingFace):
```
models/
├── sd-v1-5/                      # runwayml/stable-diffusion-v1-5
└── ip-adapter/
    ├── models/
    │   ├── ip-adapter_sd15.bin        # standard version
    │   ├── ip-adapter-plus_sd15.bin   # better identity retention ← prefer this
    │   └── image_encoder/             # CLIP ViT-H/14
```

**Scale parameter guide:**
| scale | effect |
|---|---|
| 0.3 | mostly follows text prompt, weak identity |
| 0.5 | balanced |
| 0.7 | strong identity, text prompt still works |
| 1.0 | maximal identity lock, may ignore emotion prompt |

Default for baseline experiments: `scale=0.7`

**Negative prompt (always use this):**
```
lowres, bad anatomy, bad hands, worst quality, blurry, deformed, extra fingers
```

---

## Evaluation Metrics

| Metric | File | Direction | Meaning |
|---|---|---|---|
| ArcFace Similarity | `eval/arcface_similarity.py` | ↑ higher = better | identity preserved |
| Palette Distance | `eval/palette_distance.py` | ↓ lower = better | colors consistent |
| FID | `eval/fid_score.py` | ↓ lower = better | image quality |

**Minimum required for Midway:** ArcFace + Palette Distance over 200 val pairs.
FID is nice-to-have but needs more generated images (~1000) to be meaningful.

---

## Emotion Prompts

```python
EMOTION_PROMPTS = {
    "happy":       "manga character, smiling, happy expression, 1girl, high quality",
    "sad":         "manga character, sad expression, teary eyes, 1girl, high quality",
    "angry":       "manga character, angry expression, frowning, 1girl, high quality",
    "surprised":   "manga character, surprised expression, wide eyes, 1girl, high quality",
    "crying":      "manga character, crying, tears, 1girl, high quality",
    "embarrassed": "manga character, embarrassed, blushing, 1girl, high quality",
}
```

If the character is male, swap `1girl` → `1boy`. Can also add style prompts like `anime style`, `manga lineart`.

---

## Current Status & Next Steps

### Done
- [ ] Environment setup
- [ ] Data download
- [ ] Data processing pipeline (filter → detect → pairs)
- [ ] Zero-shot inference
- [ ] Evaluation scripts
- [ ] Baseline metrics

### For Midway (April 13)
- [ ] At least 5,000 processed face crops
- [ ] 3 different OC reference images tested
- [ ] Scale comparison figure (0.3 / 0.5 / 0.7 / 1.0)
- [ ] Quantitative table: ArcFace + Palette Distance
- [ ] `results/metrics_baseline.json` filled

### After Midway (fine-tuning phase)
- [ ] Emotion classifier to label pairs (for supervised fine-tuning)
- [ ] Fine-tune IP-Adapter on expression pairs
- [ ] Add ArcFace identity loss to training objective
- [ ] Extend to multi-view (front/side) using ControlNet

---

## Common Issues & Fixes

**OOM during inference**
```python
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
# Also reduce: num_inference_steps=20, image size 512→384
```

**anime-face-detector install fails**
```bash
pip install openmim && mim install mmdet && pip install anime-face-detector
# If still fails, the script auto-falls back to face_alignment (blazeface)
```

**Low detection rate on manga faces**
- Try lowering score threshold in `03_detect_and_crop.py`: `if score > 0.5` → `if score > 0.3`
- Or switch to `lbpcascade_animeface` (OpenCV cascade)

**Generated images don't look like the reference character**
- Switch from `ip-adapter_sd15.bin` to `ip-adapter-plus_sd15.bin`
- Increase `scale` to 0.8–1.0
- Make sure reference image is clean (no background clutter, face centered)

---

## Code Style Preferences

- Python 3.10+, type hints optional but appreciated for function signatures
- Use `tqdm` for all loops over data
- Save intermediate results to JSON at each pipeline stage (so crashes don't lose work)
- Print a summary at the end of every script (`n processed`, `n skipped`, etc.)
- Prefer explicit file paths over glob patterns in scripts
- No Jupyter notebooks — use plain `.py` scripts for reproducibility

---

## Notes for Claude Code

- **Do not modify files in `data/raw/`** — these are original downloads
- When writing new scripts, follow the `XX_name.py` numbering convention
- Inference scripts should always accept a command-line argument for the reference image path
- When adding new evaluation metrics, add them to `eval/run_eval.py` as a new function and include results in `metrics.json`
- The `results/` directory should always be gitignored (large generated images)
- If asked to "run the pipeline", execute steps in order: 01 → 02 → 03 → 05 → 06 → inference → eval
