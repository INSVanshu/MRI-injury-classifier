# src/gradcam.py
# Grad-CAM implementation for MRI slice explainability

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional, Tuple

from src.model import MRNetClassifier
from src.transforms import val_transforms
from src.utils import load_config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet50.

    Registers forward and backward hooks on the target layer
    to capture activations and gradients during inference.
    These are combined to produce a spatial heatmap showing
    which regions of the MRI influenced the prediction most.

    Args:
        model        : Trained MRNetClassifier
        target_layer : Layer to extract activations from
                       ResNet50 layer4 is ideal — deepest
                       spatial features before global pooling
    """

    def __init__(
        self,
        model: MRNetClassifier,
        target_layer: Optional[nn.Module] = None,
    ):
        self.model = model
        self.model.eval()

        # Default to layer4 of ResNet50 backbone
        self.target_layer = (
            target_layer or model.backbone.layer4
        )

        # Storage for hooks
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to target layer."""

        def forward_hook(module, input, output):
            # Save feature maps from forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Save gradients flowing back through this layer
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        device: torch.device,
    ) -> Tuple[float, np.ndarray]:
        """
        Run forward + backward pass and generate heatmap.

        Args:
            input_tensor : (1, 3, 224, 224) preprocessed MRI slice
            device       : torch device

        Returns:
            probability  : float — model's confidence (0–1)
            heatmap      : np.ndarray (224, 224) — normalized [0, 1]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)           # (1, 1)
        probability = torch.sigmoid(logits).item()

        # Backward pass — gradient w.r.t. positive class
        logits.backward()

        # ── Compute Grad-CAM weights ──────────────────────────
        # Global average pool the gradients: (C, H, W) → (C,)
        weights = self.gradients.mean(dim=[2, 3])   # (1, C)

        # Weighted sum of activation maps
        activations = self.activations.squeeze(0)   # (C, H, W)
        weights     = weights.squeeze(0)            # (C,)

        # cam: weighted combination of feature maps
        cam = torch.zeros(
            activations.shape[1:], dtype=torch.float32
        )
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive contributions
        cam = torch.clamp(cam, min=0)

        # Normalize to [0, 1]
        cam = cam.numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Upsample to input resolution (224×224)
        heatmap = cv2.resize(cam, (224, 224))

        return probability, heatmap

    def overlay(
        self,
        original_slice: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original MRI slice.

        Args:
            original_slice : (H, W) float MRI slice, any range
            heatmap        : (H, W) float in [0, 1]
            alpha          : heatmap opacity (0=invisible, 1=opaque)
            colormap       : OpenCV colormap (JET is standard)

        Returns:
            overlay : (H, W, 3) uint8 RGB image
        """
        # Normalize MRI slice to [0, 255] uint8
        s = original_slice.copy().astype(np.float32)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8) * 255
        s = s.astype(np.uint8)

        # Convert grayscale → RGB
        mri_rgb = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)

        # Apply colormap to heatmap
        heatmap_uint8  = (heatmap * 255).astype(np.uint8)
        heatmap_color  = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb    = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(mri_rgb, 1 - alpha,
                                   heatmap_rgb, alpha, 0)
        return overlay


# ── Visualization helpers ─────────────────────────────────────

def visualize_gradcam(
    model      : MRNetClassifier,
    dataset,
    device     : torch.device,
    indices    : list,
    save_path  : Optional[str] = None,
) -> None:
    """
    Visualize Grad-CAM for a list of dataset indices.
    Shows: original slice | heatmap | overlay | prediction

    Args:
        model     : trained MRNetClassifier
        dataset   : MRNetDataset instance (val set recommended)
        device    : torch device
        indices   : list of sample indices to visualize
        save_path : optional path to save the figure
    """
    gradcam = GradCAM(model)
    n       = len(indices)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        'Original MRI Slice',
        'Grad-CAM Heatmap',
        'Overlay',
        'Prediction'
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    class_names = ['Normal', 'Sprain / ACL Tear', 'Fracture']

    for row, idx in enumerate(indices):
        image_tensor, label = dataset[idx]
        input_batch = image_tensor.unsqueeze(0)     # (1, 3, 224, 224)

        # Generate heatmap
        prob, heatmap = gradcam.generate(input_batch, device)
        pred_class    = 1 if prob >= 0.5 else 0
        true_class    = int(label.item())

        # Recover original MRI slice for display
        # Unnormalize ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig = (image_tensor.cpu() * std + mean)
        orig = orig.permute(1, 2, 0).numpy()        # (224, 224, 3)
        orig_gray = orig.mean(axis=2)               # collapse to grayscale

        # Build overlay
        overlay = gradcam.overlay(orig_gray, heatmap)

        # ── Plot row ──────────────────────────────────────────
        # Col 0: Original
        axes[row, 0].imshow(orig_gray, cmap='gray')
        axes[row, 0].axis('off')

        # Col 1: Heatmap only
        axes[row, 1].imshow(heatmap, cmap='jet')
        axes[row, 1].axis('off')

        # Col 2: Overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis('off')

        # Col 3: Prediction card
        correct = pred_class == true_class
        color   = '#2ecc71' if correct else '#e74c3c'
        axes[row, 3].set_facecolor(color)
        axes[row, 3].text(
            0.5, 0.6,
            f"Pred: {class_names[pred_class]}",
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white',
            transform=axes[row, 3].transAxes
        )
        axes[row, 3].text(
            0.5, 0.35,
            f"True: {class_names[true_class]}",
            ha='center', va='center',
            fontsize=10, color='white',
            transform=axes[row, 3].transAxes
        )
        axes[row, 3].text(
            0.5, 0.15,
            f"Conf: {prob:.1%}",
            ha='center', va='center',
            fontsize=9, color='white',
            transform=axes[row, 3].transAxes
        )
        axes[row, 3].axis('off')

    plt.suptitle(
        'Grad-CAM — Model Attention on MRI Slices',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")

    plt.show()


def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
) -> MRNetClassifier:
    """Load a saved checkpoint and return ready model."""
    config = load_config('configs/config.yaml')
    model  = MRNetClassifier(
        pretrained   = False,   # weights come from checkpoint
        freeze_until = 'layer3',
        dropout      = config['model']['dropout'],
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f})")
    return model
