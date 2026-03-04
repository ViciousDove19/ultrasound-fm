"""Dummy ultrasound dataset for scaffolding and smoke tests."""

import torch
from torch.utils.data import Dataset


ORGAN_LABELS = ["liver", "kidney", "thyroid", "breast", "heart", "bladder"]


class DummyUltrasoundDataset(Dataset):
    """
    Returns random grayscale images with random organ and disease labels.
    Mimics the structure the real dataset class will follow.
    """

    def __init__(self, num_samples: int = 256, img_size: int = 224, num_classes: int = 6, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

        rng = torch.Generator().manual_seed(seed)
        # Grayscale US images: 1 channel
        self.images = torch.rand(num_samples, 1, img_size, img_size, generator=rng)
        self.organ_labels = torch.randint(0, len(ORGAN_LABELS), (num_samples,), generator=rng)
        self.disease_labels = torch.randint(0, 2, (num_samples,), generator=rng)  # binary: normal/diseased

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "organ": self.organ_labels[idx],
            "disease": self.disease_labels[idx],
        }
