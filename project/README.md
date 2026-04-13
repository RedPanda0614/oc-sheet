# oc-sheet

This repo supports the Midway baseline pipeline for OC sheet generation.

## What We Have
- Expression-sheet preprocessing (download, filter, face crop)
- Baseline inference (vanilla IP-Adapter)
- Baseline evaluation (identity, palette, FID, expression accuracy, copy score)
- Textual Inversion + LoRA training scripts (few-shot per character)

## New: CLIP Pseudo-Labeling for Supervised Emotion
We use CLIP zero-shot to assign an emotion label to each face crop.

Supported labels for Stage-1:
`neutral, happy, sad, angry, surprised, crying`

### 1) Label emotions for face crops
```bash
python scripts/label_emotions.py \
  --faces-meta data/processed/faces_meta.json \
  --output data/processed/faces_emotion.json \
  --min-confidence 0.22
```

### 2) Build LoRA/TI pairs using emotion labels
```bash
python scripts/build_pairs_lora.py \
  --faces-meta data/processed/faces_meta.json \
  --faces-emotion data/processed/faces_emotion.json \
  --output-dir data/lora/pairs \
  --val-ratio 0.2
```

## LoRA Baseline (few-shot per character)
Train:
```bash
python scripts/train_lora.py \
  --pairs-json data/lora/pairs/train.json \
  --sheet-id sheet_00008 \
  --num-images 8 \
  --token "<oc>" \
  --output-dir results/lora
```

Infer:
```bash
python inference/run_personalized.py \
  --mode lora \
  --weights-dir results/lora/sheet_00008 \
  --token "<oc>" \
  --output-dir results/lora/sheet_00008/outputs
```

## Textual Inversion Baseline (few-shot per character)
Train:
```bash
python scripts/train_textual_inversion.py \
  --pairs-json data/lora/pairs/train.json \
  --sheet-id sheet_00008 \
  --num-images 8 \
  --token "<oc>" \
  --output-dir results/textual_inversion
```

Infer:
```bash
python inference/run_personalized.py \
  --mode textual_inversion \
  --weights-dir results/textual_inversion/sheet_00008 \
  --token "<oc>" \
  --output-dir results/textual_inversion/sheet_00008/outputs
```
