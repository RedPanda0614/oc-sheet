# OC Expression Sheet Generation

Given a single reference face image of an anime original character (OC), automatically generate that character across a set of target expressions (neutral, happy, sad, angry, surprised, crying) and stitch them into an expression sheet.

---

## Data Processing

Raw data is scraped from Safebooru (expression-sheet / character-sheet tags) and processed through four stages:

```
scripts/data_processing/
  01_download.sh          # gallery-dl scrape from Safebooru
  02_filter.py            # drop corrupted / too-small / non-RGB images
  03_detect_and_crop.py   # YOLOv8 anime-face detection → 512×512 crops
  build_pairs_lora.py     # group crops by sheet_id, build reference-target pairs
scripts/
  label_emotions.py       # CLIP zero-shot emotion label for each crop
```

**Step by step:**

```bash
# 1. Download (~3000 expression sheets)
bash scripts/data_processing/01_download.sh

# 2. Filter bad images
python scripts/data_processing/02_filter.py

# 3. Detect and crop faces (saves to data/processed/faces/ + faces_meta.json)
python scripts/data_processing/03_detect_and_crop.py

# 4. CLIP emotion labeling
python scripts/label_emotions.py \
  --faces-meta data/processed/faces_meta.json \
  --output     data/processed/faces_emotion.json \
  --min-confidence 0.22

# 5. Build train/val pairs
python scripts/data_processing/build_pairs_lora.py \
  --faces-meta    data/processed/faces_meta.json \
  --faces-emotion data/processed/faces_emotion.json \
  --output-dir    data/lora/pairs \
  --val-ratio 0.2
```

Pairs are stored in `data/lora/pairs/{train,val}.json`. Each entry records `sheet_id`, `reference_path`, `target_path`, and `target_emotion`.

---

## Models

| Component | Source |
|---|---|
| Diffusion backbone | Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`) |
| Image encoder | CLIP ViT-H/14 (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) |
| Identity conditioning | IP-Adapter Plus (`h94/IP-Adapter`, `ip-adapter-plus_sd15.bin`) |

Download all weights:
```bash
python scripts/download_model.py
```
Files land under `models/`.

---

## Evaluation Metrics

All metrics are computed by `eval/run_eval.py` against a manifest JSON produced by any batch inference script.

| Metric | Tool | Direction |
|---|---|---|
| **Identity similarity** | ArcFace cosine similarity (ref vs. generated) | ↑ higher is better |
| **Palette distance** | Mean Lab-space distance of top-5 palette colours | ↓ lower is better |
| **Copy score** | CLIP embedding similarity (ref vs. generated) | ↓ lower is better |
| **Copy violation rate** | % images with copy score > 0.88 | ↓ lower is better |
| **Expression accuracy** | CLIP zero-shot classifier | ↑ higher is better |
| **FID** | clean-fid against ground-truth targets | ↓ lower is better |
| **Sheet coverage** | fraction of 6 emotions generated per character | ↑ higher is better |
| **Sheet panel accuracy** | per-sheet mean expression accuracy | ↑ higher is better |

Run evaluation:
```bash
cd eval
python run_eval.py \
  --manifest      ../results/<run>/manifest.json \
  --generated-dir ../results/<run> \
  --output-json   ../results/metrics_<run>.json
```

---

## Baselines

Three zero-shot / training-free baselines, used to establish lower bounds:

**Prompt-only** — pure SD text generation, no reference image.  
Two modes: `naive` (emotion keyword only) and `structured` (emotion + character attributes).
```bash
python inference/batch_prompt_only_labeled.py --mode both
```

**Global LoRA** — a single LoRA trained across all characters on emotion labels; conditions on prompt only at inference (no reference image).
```bash
python train/baseline/train_lora_global.py \
  --pairs-json data/lora/pairs/train.json \
  --output-dir results/lora_global

python inference/batch_lora_global_labeled.py \
  --lora-dir results/lora_global
```

**Vanilla IP-Adapter (P0)** — off-the-shelf IP-Adapter Plus with no fine-tuning, conditioned on the reference image.
```bash
python inference/batch_inference_labeled.py
```

## Our Methods

We build on IP-Adapter Plus and fine-tune it progressively.

### P1 — IP-Adapter Fine-tuning

Standard supervised fine-tuning of the IP-Adapter cross-attention layers and image projection head on our labeled pairs. This teaches the model to use the reference image for identity while following the emotion prompt.

```bash
python train/train_ip_adapter_finetune.py \
  --train_json data/label_pairs/train.json \
  --output_dir checkpoints/p1
```

Inference uses the same pipeline as P0 but loads the fine-tuned weights:
```bash
python inference/batch_inference_p1_labeled.py \
  --checkpoint checkpoints/p1
```

---

### P3 — Expression-local Mask Loss

Observation: expression changes are concentrated in the brow/eye (15–50%) and mouth (58–82%) regions. P3 adds a spatial soft mask to the diffusion loss so those regions receive 3× higher gradient signal, while identity-preserving regions (hair, chin) are down-weighted.

The mask is mean-normalised to 1.0 so total loss magnitude stays the same as P1.

```bash
python train/p3_finetune.py \
  --image_proj_ckpt checkpoints/p1/image_proj_model.pt \
  --ip_attn_ckpt    checkpoints/p1/ip_attn_procs.pt \
  --output_dir      checkpoints/p3
```

**P3 results:** identity 0.455, expr. acc. 0.398, FID 64.2 — expression accuracy slightly lower than P1; the mask alone does not resolve expression confusion.

---

### P4 — Anti-copy Triplet Loss

Problem: the model sometimes copies the reference expression rather than switching to the target. P4 adds a triplet-style penalty in latent space:

```
loss_anticopy = relu(sim(pred_x0, ref_latent) - sim(pred_x0, target_latent) + margin)
```

This penalty is only active for pairs where `reference_emotion ≠ target_emotion`, pushing the prediction toward the target and away from the reference in latent space.

```bash
python train/p4_finetune.py \
  --resume_dir              checkpoints/p1 \
  --train_json              data/label_pairs/train.json \
  --faces_emotion_json      data/processed/faces_emotion.json \
  --output_dir              checkpoints/p4
```

---

### P5 — Candidate Reranking

At inference time, generate N candidates per emotion (default 4) and score each one on expression accuracy, identity similarity, palette consistency, and copy avoidance. The highest-scoring candidate is kept.

Scoring formula:
```
score = w_expr_hit * expr_hit + w_expr_conf * expr_conf
      + w_id * identity - w_palette * palette - w_copy * copy
      - copy_violation_penalty * (copy_score > threshold)
```

```bash
python inference/batch_inference_p5_rerank_labeled.py \
  --checkpoint  checkpoints/p4 \
  --n_candidates 4 \
  --output-dir  results/p5
```

---

### Single-image Inference

To generate an expression sheet for one character:

```bash
python inference/generate_expressions.py \
  --reference  path/to/face.jpg \
  --checkpoint checkpoints/p4 \
  --output_dir results/my_character
```

Add `--n_candidates 4` to enable P5 reranking. Output is individual expression images plus `sheet.jpg`.

---

## Repository Layout

```
project/
  scripts/
    data_processing/   # download → filter → crop → pair-building
    download_model.py  # fetch SD/IP-Adapter weights from HuggingFace
    label_emotions.py  # CLIP emotion labeling
  train/
    train_ip_adapter_finetune.py   # P1
    p3_finetune.py                 # P3 masked loss
    p3_mask.py                     # soft mask construction utility
    p4_finetune.py                 # P4 anti-copy loss
    baseline/                      # LoRA / Textual Inversion baselines
  inference/
    run_baseline.py                # single-image baseline
    batch_inference_labeled.py     # P0 batch
    batch_inference_p1_labeled.py  # P1 batch
    batch_inference_p5_rerank_labeled.py  # P4+P5 batch
    generate_expressions.py        # single-character sheet generation
  eval/
    run_eval.py             # main evaluation runner
    arcface_similarity.py
    expression_classifier.py
    palette_distance.py
    copy_score.py
    fid_score.py
  data/
    lora/pairs/{train,val}.json    # labeled reference-target pairs
    processed/faces_emotion.json   # per-face emotion labels
  checkpoints/                     # saved P3/P4 weights
  results/                         # inference outputs + metric JSONs
```
