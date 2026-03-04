"""
Generate and save dummy ultrasound image data to data/processed/dummy/.
Run from repo root: python scripts/create_dummy_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dummy_dataset import DummyUltrasoundDataset

OUT_DIR = ROOT / "data" / "processed" / "dummy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {"train": 800, "val": 100, "test": 100}

for split, n in SPLITS.items():
    ds = DummyUltrasoundDataset(num_samples=n, seed={"train": 0, "val": 1, "test": 2}[split])
    torch.save(
        {"images": ds.images, "organ_labels": ds.organ_labels, "disease_labels": ds.disease_labels},
        OUT_DIR / f"{split}.pt",
    )
    print(f"Saved {split}: {n} samples → {OUT_DIR / f'{split}.pt'}")

print("Done.")
