# src/transforms.py
# All normalization and augmentation logic lives here

import numpy as np
import torch
from torchvision import transforms


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a single MRI volume.
    Each exam normalized independently — MRI intensity
    is not globally consistent across scanners.
    """
    mean = volume.mean()
    std  = volume.std()
    if std == 0:
        return volume - mean          # flat scan edge case
    return (volume - mean) / std


def to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """
    Convert single-channel MRI slice to 3-channel RGB.
    Required because ResNet50 expects 3-channel input.
    Strategy: replicate the grayscale channel 3 times.
    """
    return np.stack([slice_2d] * 3, axis=0)   # (3, H, W)


# ── Training augmentation ─────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # ImageNet stats — used because ResNet was pretrained on it
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Validation / inference (no augmentation) ─────────────────
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
