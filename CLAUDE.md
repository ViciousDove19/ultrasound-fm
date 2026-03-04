# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ultrasound-fm` is a research project building a foundation model for ultrasound images. The core hypothesis is that ultrasound is inherently 1D in its acquisition direction (top-to-bottom, skin to organ), so images should be tokenized as 256×1 row vectors rather than the 16×16 patches used in standard ViT. The pretraining objective is **next-row prediction**: given a sequence of row tokens, predict the next row. This is analogous to causal language modeling but applied to ultrasound image rows.

Each token carries three types of information:
1. Spatial position (row index)
2. Organ identity
3. Disease presence (row-level label)

Target venue: *Medical Image Analysis* journal. The model should outperform existing ultrasound foundation models.

## Architecture Design

- **Tokenizer**: Slice input image into rows of shape `(256,)` → embed to token dimension
- **Positional encoding**: Learned or sinusoidal, combined with organ embedding and disease embedding
- **Backbone**: Transformer encoder/decoder (ViT-style), adapted for 1D row tokens
- **Pretraining head**: Next-row prediction (autoregressive) — predicts pixel values or embeddings of the next row
- **Fine-tuning heads**: Swappable heads for segmentation, classification, detection

All model components live in `src/models/`. Pretraining and fine-tuning loops are separate in `src/training/`.

## Experiment Configuration

Every experiment is fully specified by a YAML config in `configs/`. A config covers: model architecture hyperparameters, dataset selection, data splits, augmentations, optimizer, scheduler, and evaluation settings. Never hardcode hyperparameters in source files — always read from config.

`data/splits/` contains JSON/CSV split files and **is tracked in git** for reproducibility. `data/raw/` and `data/processed/` are gitignored.

## Compute

Single node with an **L40S GPU**. Use PyTorch with mixed precision (`torch.cuda.amp`) and gradient accumulation for large effective batch sizes. Checkpoints are saved to `checkpoints/` (gitignored) and must be backed up externally.

## Key Conventions

- All public datasets used for pretraining are organ-imbalanced — the data pipeline must handle class-weighted sampling or rebalancing
- `data/splits/` files are the source of truth for which samples belong to train/val/test — never split inside dataset classes
- Metrics for benchmarking: AUROC, Dice coefficient, IoU, accuracy (task-dependent)
- Baseline comparisons: existing ultrasound foundation models (document these in `docs/` as they are identified)

## Documentation

`docs/project_structure.md` describes the full directory layout and the purpose of each component. Update it when the structure changes.
