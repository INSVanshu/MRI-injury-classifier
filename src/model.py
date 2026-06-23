# src/model.py
# ResNet50 fine-tuned for MRI knee injury classification

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class MRNetClassifier(nn.Module):
    """
    Transfer-learned ResNet50 for binary MRI classification.

    Architecture:
        ResNet50 (ImageNet pretrained)
            → freeze early layers
            → replace final FC layer
            → custom classifier head
            → single sigmoid output (binary per label)

    Why binary and not 3-class softmax?
        MRNet has 3 separate binary labels (abnormal/acl/meniscus).
        We train one model per label and combine predictions.
        This matches the original MRNet paper's approach.

    Args:
        pretrained   : Use ImageNet weights
        freeze_until : Freeze all layers up to this layer name
        dropout      : Dropout rate before final layer
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_until: str = 'layer3',
        dropout: float = 0.5,
    ):
        super(MRNetClassifier, self).__init__()

        # Load pretrained backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ── Selective layer freezing ──────────────────────────
        # Freeze early layers (learn low-level features like edges)
        # Unfreeze later layers (learn MRI-specific high-level features)
        self._freeze_layers(backbone, freeze_until)

        # ── Remove original FC head ───────────────────────────
        in_features = backbone.fc.in_features      # 2048 for ResNet50
        backbone.fc = nn.Identity()                # strip the classifier
        self.backbone = backbone

        # ── Custom classification head ────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),                     # binary output
        )

        print(f"[MRNetClassifier] pretrained={pretrained} | "
              f"frozen_until={freeze_until} | dropout={dropout}")
        print(f"  Backbone output features : {in_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, 3, 224, 224)
        Returns:
            logits : (batch, 1) — raw scores before sigmoid
        """
        features = self.backbone(x)        # (batch, 2048)
        logits   = self.classifier(features)  # (batch, 1)
        return logits

    def _freeze_layers(
        self, model: nn.Module, freeze_until: str
    ) -> None:
        """
        Freeze all layers up to and including freeze_until.
        Layers after freeze_until remain trainable.
        """
        freeze = True
        for name, param in model.named_parameters():
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

        frozen = sum(
            1 for p in model.parameters() if not p.requires_grad
        )
        total  = sum(1 for p in model.parameters())
        print(f"  Frozen layers: {frozen}/{total} parameter groups")

    def unfreeze_all(self) -> None:
        """Unfreeze all layers — call after initial warmup epochs."""
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen for fine-tuning")


def build_model(config: dict, device: torch.device) -> MRNetClassifier:
    """
    Build and return model from config dict.
    Single entry point — always use this in training scripts.
    """
    model = MRNetClassifier(
        pretrained   = config['model']['pretrained'],
        freeze_until = 'layer3',
        dropout      = config['model']['dropout'],
    )
    model = model.to(device)
    return model
