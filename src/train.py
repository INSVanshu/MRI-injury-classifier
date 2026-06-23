# src/train.py
# Full training loop with validation, early stopping, W&B logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import wandb

from src.dataset import MRNetDataset
from src.model import build_model
from src.transforms import train_transforms, val_transforms
from src.utils import set_seed, get_device, load_config


# ── Metric helpers ────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute accuracy, sensitivity, specificity from batch.
    These are the clinically meaningful metrics for medical AI.
    """
    probs = torch.sigmoid(logits).squeeze()
    preds = (probs >= threshold).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)   # recall — critical for medical AI
    specificity = tn / (tn + fp + 1e-8)

    return {
        'accuracy'   : accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


# ── Single epoch functions ────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    criterion : nn.Module,
    device    : torch.device,
    epoch     : int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)   # (B,) → (B, 1)

        optimizer.zero_grad()
        logits = model(images)                    # (B, 1)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — stabilizes training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"loss={loss.item():.4f}")

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def validate(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: nn.Module,
    device   : torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics    = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


# ── Main training function ────────────────────────────────────

def train(
    config_path : str = 'configs/config.yaml',
    label_type  : str = 'acl',
    plane       : str = 'sagittal',
) -> None:
    """
    Full training pipeline.

    Args:
        config_path : Path to config.yaml
        label_type  : 'acl', 'meniscus', or 'abnormal'
        plane       : 'sagittal', 'coronal', or 'axial'
    """

    # ── Setup ─────────────────────────────────────────────────
    config = load_config(config_path)
    set_seed(42)
    device = get_device()

    # ── W&B init ──────────────────────────────────────────────
    wandb.init(
        project = config['wandb']['project'],
        name    = f"{label_type}_{plane}",
        config  = {
            'label_type'   : label_type,
            'plane'        : plane,
            'architecture' : config['model']['architecture'],
            'epochs'       : config['training']['epochs'],
            'lr'           : config['training']['learning_rate'],
            'batch_size'   : config['training']['batch_size'],
        }
    )

    # ── Datasets ──────────────────────────────────────────────
    data_dir = config['data']['raw_dir']

    train_dataset = MRNetDataset(
        data_dir   = data_dir,
        split      = 'train',
        plane      = plane,
        label_type = label_type,
        transform  = train_transforms,
    )
    val_dataset = MRNetDataset(
        data_dir   = data_dir,
        split      = 'valid',
        plane      = plane,
        label_type = label_type,
        transform  = val_transforms,
    )

    # ── Weighted sampler for class imbalance ──────────────────
    labels      = train_dataset.labels_df['label'].values
    class_counts = np.bincount(labels)
    weights     = 1.0 / class_counts[labels]
    sampler     = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = config['training']['batch_size'],
        sampler    = sampler,
        num_workers= 2,
        pin_memory = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = config['training']['batch_size'],
        shuffle    = False,
        num_workers= 2,
        pin_memory = True,
    )

    # ── Model, loss, optimizer ────────────────────────────────
    model     = build_model(config, device)
    pos_weight = train_dataset.get_class_weights().to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config['training']['learning_rate'],
        weight_decay = config['training']['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    save_dir = Path('models')
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*50}")

        # Unfreeze all layers after warmup
        if epoch == 10:
            model.unfreeze_all()
            print("Warmup complete — all layers unfrozen")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_metrics = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step on val loss
        scheduler.step(val_metrics['loss'])

        # Print epoch summary
        print(f"\n  Train → loss={train_metrics['loss']:.4f} | "
              f"acc={train_metrics['accuracy']:.3f} | "
              f"sens={train_metrics['sensitivity']:.3f} | "
              f"spec={train_metrics['specificity']:.3f}")
        print(f"  Val   → loss={val_metrics['loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.3f} | "
              f"sens={val_metrics['sensitivity']:.3f} | "
              f"spec={val_metrics['specificity']:.3f}")

        # W&B logging
        wandb.log({
            'epoch'           : epoch,
            'train/loss'      : train_metrics['loss'],
            'train/accuracy'  : train_metrics['accuracy'],
            'train/sensitivity': train_metrics['sensitivity'],
            'train/specificity': train_metrics['specificity'],
            'val/loss'        : val_metrics['loss'],
            'val/accuracy'    : val_metrics['accuracy'],
            'val/sensitivity' : val_metrics['sensitivity'],
            'val/specificity' : val_metrics['specificity'],
            'lr'              : optimizer.param_groups[0]['lr'],
        })

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            ckpt_path = save_dir / f'best_{label_type}_{plane}.pth'
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'   : best_val_loss,
                'val_metrics': val_metrics,
                'config'     : config,
            }, ckpt_path)
            print(f"  ✓ Best model saved → {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/"
                  f"{config['training']['patience']})")

        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    train(
        config_path = 'configs/config.yaml',
        label_type  = 'acl',
        plane       = 'sagittal',
    )
