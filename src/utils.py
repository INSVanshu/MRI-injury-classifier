# src/utils.py
# Shared helper functions

import yaml
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.
    Call this at the top of every training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Seed set to {seed}")


def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list = ['Normal', 'Sprain', 'Fracture'],
    n: int = 8
) -> None:
    """
    Visualize a batch of preprocessed MRI slices with labels.
    Unnormalizes ImageNet normalization for display.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(1, min(n, len(images)), figsize=(18, 3))
    fig.suptitle('Preprocessed MRI Batch', fontsize=12, fontweight='bold')

    for i, ax in enumerate(axes):
        img = images[i].cpu() * std + mean     # unnormalize
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        label_idx = int(labels[i].item())
        label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
        ax.imshow(img, cmap='gray')
        ax.set_title(label_name, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return trainable
