# Storifai — Training Setup (V2)

## Overview

Two models trained on identical data with identical training protocol for a
fair comparison. Both produce a 5-sentence story given 5 photos.

## Dataset

- Source: VIST Story-in-Sequence (SIS) annotations + dyliu/VIST images
- Filtering: only stories where all 5 photos are present
- Splits (re-split from original VIST val to ensure all images are accessible):
  - Train: 3,482 stories (17,410 images)
  - Val: 746 stories (3,730 images)
  - Test: 747 stories (3,735 images)
- Vocab: 2,565 words (min count 5)
- Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`

## Architectures

### Baseline — ResNet-50 + LSTM
- **Encoder**: ResNet-50 pretrained on ImageNet
  - Frozen for first 5 epochs, then unfrozen with 0.1× learning rate
  - Output: 2048-dim feature per image
- **Decoder**: 1-layer LSTM (hidden=512, embed=512)
  - Initial hidden/cell state from projected image feature
  - Trained per-image (each photo's caption decoded independently)
- ~24.5M trainable parameters

### Improved — CLIP + Cross-Image Attention + Transformer
- **Encoder**: CLIP ViT-B/32 (frozen)
  - Output: 512-dim feature per image
- **Cross-Image Attention**: 2 layers of multi-head self-attention across
  the 5 image features (8 heads, d=512)
  - This is the novelty: each image is contextualized by the others
  - Enables narrative coherence across the photo sequence
- **Decoder**: 4-layer Transformer decoder (8 heads, d=512, ff=2048)
- ~22M trainable parameters (CLIP frozen)

## Training protocol (identical for both)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| LR schedule | 1-epoch warmup + cosine decay |
| Batch size | 8 stories (= 40 images per batch) |
| Max tokens | 30 per sentence |
| Loss | Cross-entropy + label smoothing 0.1 |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | FP16 (autocast + GradScaler) |
| Max epochs | 15 |
| Early stopping | patience=3 on val BLEU-4 |
| Image transforms (train) | Resize-256 + RandomCrop-224 + HFlip |
| Image transforms (eval) | Resize-256 + CenterCrop-224 |
| Normalization | ImageNet stats (used for both for fairness) |

## Evaluation

- BLEU-1, BLEU-2, BLEU-3, BLEU-4 (corpus-level, NLTK with smoothing method 1)
- METEOR (averaged per-story)
- Full-story scoring: 5 sentences concatenated → compared to reference 5 sentences
- Greedy decoding (no beam search)

## Files

- `storifai_dataset.py` — dataset loader, GCS image streaming with LRU cache
- `storifai_models.py` — both architectures
- `storifai_eval.py` — BLEU/METEOR computation
- `storifai_train.py` — training loop with early stopping + checkpointing
- `MASTER_COLAB_CELL.py` — single self-contained cell to paste into Colab

## Outputs (saved to GCP)

```
gs://hoshmand-ai-datasets/storifai/checkpoints/
├── improved_v2_best.pth         # Best improved model checkpoint
├── improved_v2_results.json     # Metrics + samples + history
├── baseline_v2_best.pth         # Best baseline model checkpoint
└── baseline_v2_results.json     # Metrics + samples + history
```

## Expected runtimes on A100

- Improved: ~2-3 hours for 15 epochs
- Baseline: ~1.5 hours for 15 epochs (smaller, faster encoder)
- Total: ~4 hours

Both should converge well before epoch 15 thanks to early stopping.
