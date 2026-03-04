# Project Structure

This document describes the directory layout of the `ultrasound-fm` project and the purpose of each component.

```
ultrasound-fm/
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── configs/
├── scripts/
├── notebooks/
├── experiments/
├── checkpoints/
├── logs/
└── docs/
```

---

## `src/`

All production source code. Organized by concern so each module has a single responsibility.

### `src/models/`
Model architecture definitions.
- Custom ViT variant with 1D row-based tokenization (256×1 tokens)
- Positional encodings (spatial + organ-type + disease label)
- Attention modules, transformer blocks
- Pretraining heads (next-row prediction, masked prediction, etc.)
- Fine-tuning heads for downstream tasks (segmentation, classification)

### `src/data/`
Everything related to loading and preparing data.
- PyTorch `Dataset` and `DataLoader` classes per dataset source
- Preprocessing pipelines (resizing, normalization, artifact removal)
- Augmentation strategies specific to ultrasound (speckle noise, gain simulation)
- Tokenization: slicing images into 256×1 row tokens
- Organ label and disease label mapping utilities

### `src/training/`
Training infrastructure.
- Pretraining loop (next-row prediction objective)
- Fine-tuning loop for downstream tasks
- Loss functions
- Optimizer and learning rate scheduler setup
- Mixed precision, gradient accumulation, multi-GPU utilities

### `src/evaluation/`
Evaluation and benchmarking.
- Metric implementations (AUROC, Dice coefficient, IoU, accuracy)
- Benchmark runners for downstream tasks
- Comparison utilities against baseline and existing foundation models

### `src/utils/`
Shared utilities used across modules.
- Config loading and validation (YAML parsing)
- Logging setup (console + file + WandB/TensorBoard)
- Checkpoint save/load helpers
- Visualization tools (attention maps, token overlays, prediction plots)
- Reproducibility helpers (seed setting, deterministic mode)

---

## `data/`

Data management. Raw and processed data are **gitignored** (too large); split files are **tracked** for reproducibility.

### `data/raw/`
Original downloaded datasets, never modified. Each dataset gets its own subdirectory.
- Gitignored.

### `data/processed/`
Preprocessed data ready for training (resized, normalized, tokenized).
- Gitignored.

### `data/splits/`
JSON or CSV files defining train/val/test splits per dataset and experiment.
- **Tracked in git** — ensures all experiments use identical splits and results are reproducible.

---

## `configs/`

YAML configuration files, one per experiment or model variant. Each config fully specifies a run:
model architecture, dataset, training hyperparameters, augmentations, evaluation settings.
- Tracked in git so every experiment is reproducible from its config alone.

---

## `scripts/`

Executable scripts for common workflows:
- Downloading and verifying public datasets
- Preprocessing raw data into `data/processed/`
- Launching pretraining and fine-tuning jobs (with SLURM or direct)
- Running evaluations and generating benchmark tables

---

## `notebooks/`

Jupyter notebooks for exploration and analysis. Not production code.
- Dataset exploration and statistics
- Attention map visualization
- Result analysis and figure generation for papers

---

## `experiments/`

Per-run output directory: metrics, plots, config snapshots, evaluation results.
Small text/JSON outputs may be tracked; large binary outputs are gitignored.

---

## `checkpoints/`

Saved model weights (`.pt`, `.ckpt`). **Gitignored** due to file size.
Managed separately — backed up to shared storage or versioned via DVC.

---

## `logs/`

Training logs, TensorBoard event files, WandB cache. **Gitignored**.

---

## `docs/`

Project documentation.
- `project_structure.md` — this file
- Additional docs added as the project grows (data sources, model design, training recipes, etc.)
