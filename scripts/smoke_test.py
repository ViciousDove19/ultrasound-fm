"""
Smoke test: forward pass through all ViT variants on a dummy batch.
Run from repo root: python scripts/smoke_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import build_vit

VARIANTS = ["vit_ti16", "vit_s16", "vit_b16", "vit_l16"]
IMG_SIZE = 224
BATCH_SIZE = 2
IN_CHANNELS = 1   # grayscale ultrasound
NUM_CLASSES = 6   # organs

x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

for name in VARIANTS:
    model = build_vit(name, img_size=IMG_SIZE, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name:10s}  params={params:.1f}M  output={tuple(out.shape)}")
