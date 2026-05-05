# Storifai training pipeline

Implementation of the baseline and improved visual storytelling models
described in the project proposal.

## Architectures

**Baseline.** ResNet-50 (ImageNet pretrained) image encoder followed by an
LSTM caption decoder. The CNN backbone is frozen for the first five epochs
and unfrozen with a lower learning rate afterwards.

**Improved.** OpenAI CLIP ViT-B/32 visual encoder (frozen) followed by a
two-layer Transformer encoder applied across the five images of a story
(cross-image attention) and a four-layer Transformer decoder (8 heads,
d_model=512). Each per-image sentence is generated conditioned on the
corresponding attended visual feature.

Both models share an identical training/evaluation pipeline; only the
architecture differs.

## Data

Trained on the VIST validation split with images stored in
`gs://hoshmand-ai-datasets/storifai/images/val/`. Stories with all five
images available are re-split 70/15/15 into train/val/test (4,975 stories
total).

## Training

- Optimizer: AdamW, weight decay 0.01
- Learning rate: 1e-4 with linear warmup (10% of steps) and cosine decay
- Batch size: 8 stories
- Max caption length: 30 tokens
- Mixed precision (FP16 autocast) on CUDA
- Gradient clipping at 1.0
- Early stopping: patience=3 on validation BLEU-4
- Up to 15 epochs

## Evaluation

Corpus-level BLEU-1, BLEU-2, BLEU-3, BLEU-4 (NLTK with smoothing method 1)
and METEOR. Stories are formed by concatenating the five generated
sentences before scoring.

## Files

- `storifai_dataset.py` — VIST dataset loader and GCS image fetcher
- `storifai_models.py` — model architectures
- `storifai_eval.py` — BLEU / METEOR scoring
- `storifai_train.py` — training loop and checkpointing
- `MASTER_COLAB_CELL.py` — single-cell Colab driver

## Reproducing results

Switch the Colab runtime to A100/V100/L4 with high RAM, then run
`MASTER_COLAB_CELL.py` as one cell. It installs dependencies,
authenticates with GCP, fetches images, runs a single-batch smoke test,
trains both models, and saves checkpoints + results to
`gs://hoshmand-ai-datasets/storifai/checkpoints/`.
