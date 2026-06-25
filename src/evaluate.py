# src/evaluate.py
# Full evaluation pipeline — clinical metrics, ROC-AUC,
# confusion matrix, per-threshold analysis

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve,
)
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from src.model import MRNetClassifier
from src.dataset import MRNetDataset
from src.transforms import val_transforms
from src.gradcam import load_trained_model
from src.utils import get_device, load_config


# ── Inference ─────────────────────────────────────────────────

def run_inference(
    model  : MRNetClassifier,
    loader : DataLoader,
    device : torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on entire dataloader.

    Returns:
        probs  : (N,) float — predicted probabilities
        preds  : (N,) int   — binary predictions at threshold 0.5
        labels : (N,) int   — ground truth labels
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)                      # (B, 1)
            probs  = torch.sigmoid(logits).squeeze(1)  # (B,)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    return probs, preds, labels.astype(int)


# ── Core metrics ──────────────────────────────────────────────

def compute_clinical_metrics(
    probs  : np.ndarray,
    preds  : np.ndarray,
    labels : np.ndarray,
    label_name: str = 'ACL',
) -> Dict[str, float]:
    """
    Compute the full set of clinically relevant metrics.

    In medical AI, sensitivity (catching real injuries) is
    usually prioritized over specificity (avoiding false alarms).
    A missed fracture is worse than an unnecessary follow-up.
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn + 1e-8)   # true positive rate
    specificity = tn / (tn + fp + 1e-8)   # true negative rate
    precision   = tp / (tp + fp + 1e-8)   # positive predictive value
    npv         = tn / (tn + fn + 1e-8)   # negative predictive value
    f1          = 2 * (precision * sensitivity) / \
                  (precision + sensitivity + 1e-8)
    auc         = roc_auc_score(labels, probs)
    ap          = average_precision_score(labels, probs)

    metrics = {
        'label'      : label_name,
        'accuracy'   : round(accuracy,    4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'precision'  : round(precision,   4),
        'npv'        : round(npv,         4),
        'f1_score'   : round(f1,          4),
        'roc_auc'    : round(auc,         4),
        'avg_precision': round(ap,        4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }

    return metrics


def find_optimal_threshold(
    probs : np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Find the probability threshold that maximizes Youden's J.
    Youden's J = sensitivity + specificity - 1
    This is the standard clinical threshold selection method.

    Returns:
        threshold   : optimal cutoff
        sensitivity : at optimal threshold
        specificity : at optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    specificity = 1 - fpr
    j_scores    = tpr + specificity - 1
    best_idx    = np.argmax(j_scores)

    return (
        float(thresholds[best_idx]),
        float(tpr[best_idx]),
        float(specificity[best_idx]),
    )


# ── Plotting functions ────────────────────────────────────────

def plot_roc_curve(
    results   : List[Dict],
    save_path : Optional[str] = None,
) -> None:
    """
    Plot ROC curves for all evaluated label types.
    Each label gets its own curve with AUC in the legend.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ['#E87B4C', '#4C9BE8', '#4CE8A0']

    for result, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(result['labels'], result['probs'])
        auc         = result['metrics']['roc_auc']
        ax.plot(
            fpr, tpr,
            label=f"{result['metrics']['label']} "
                  f"(AUC = {auc:.3f})",
            color=color, linewidth=2.5,
        )

    # Diagonal — random classifier baseline
    ax.plot(
        [0, 1], [0, 1],
        linestyle='--', color='gray',
        linewidth=1, label='Random classifier'
    )

    ax.set_xlabel('False Positive Rate (1 - Specificity)',
                  fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves — MRI Knee Injury Classifier',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(
    preds     : np.ndarray,
    labels    : np.ndarray,
    label_name: str = 'ACL',
    save_path : Optional[str] = None,
) -> None:
    """Plot a clean, annotated confusion matrix."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    # Annotations
    classes = ['Negative', 'Positive']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(
        f'Confusion Matrix — {label_name}',
        fontsize=13, fontweight='bold'
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            cell_labels = [['TN', 'FP'], ['FN', 'TP']]
            ax.text(
                j, i,
                f"{cell_labels[i][j]}\n{cm[i, j]}",
                ha='center', va='center', fontsize=14,
                fontweight='bold',
                color='white' if cm[i, j] > thresh else 'black',
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


def plot_precision_recall(
    results  : List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot precision-recall curves — better than ROC for imbalanced data."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = ['#E87B4C', '#4C9BE8', '#4CE8A0']

    for result, color in zip(results, colors):
        precision, recall, _ = precision_recall_curve(
            result['labels'], result['probs']
        )
        ap = result['metrics']['avg_precision']
        ax.plot(
            recall, precision,
            label=f"{result['metrics']['label']} "
                  f"(AP = {ap:.3f})",
            color=color, linewidth=2.5,
        )

    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(
        'Precision-Recall Curves — MRI Knee Injury Classifier',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


def plot_metrics_summary(
    metrics_list: List[Dict],
    save_path   : Optional[str] = None,
) -> None:
    """Bar chart comparing key metrics across all label types."""
    labels    = [m['label'] for m in metrics_list]
    metric_keys = [
        'accuracy', 'sensitivity', 'specificity',
        'f1_score', 'roc_auc'
    ]
    colors    = ['#4C9BE8', '#E87B4C', '#4CE8A0', '#9B4CE8', '#E84C9B']

    x   = np.arange(len(labels))
    w   = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (key, color) in enumerate(zip(metric_keys, colors)):
        values = [m[key] for m in metrics_list]
        bars   = ax.bar(
            x + i * w, values, w,
            label=key.replace('_', ' ').title(),
            color=color, alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}', ha='center',
                va='bottom', fontsize=7.5,
            )

    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.15])
    ax.set_title(
        'Clinical Metrics Summary',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.axhline(y=0.8, color='red', linestyle='--',
               alpha=0.4, label='0.8 threshold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()


# ── Full evaluation pipeline ──────────────────────────────────

def evaluate_all(
    checkpoint_dir: str = 'models',
    data_dir      : str = 'data/raw',
    report_dir    : str = 'reports',
) -> pd.DataFrame:
    """
    Run full evaluation across all label types.
    Loads each checkpoint, runs inference, computes all metrics.

    Returns:
        DataFrame with metrics for all label types
    """
    config     = load_config('configs/config.yaml')
    device     = get_device()
    Path(report_dir).mkdir(exist_ok=True)

    label_types = ['abnormal', 'acl', 'meniscus']
    plane       = 'sagittal'
    all_results = []

    for label_type in label_types:
        print(f"\n{'─'*40}")
        print(f"Evaluating: {label_type.upper()}")
        print(f"{'─'*40}")

        # Load model
        ckpt_path = (
            f"{checkpoint_dir}/best_{label_type}_{plane}.pth"
        )
        if not Path(ckpt_path).exists():
            print(f"  ⚠ Checkpoint not found: {ckpt_path}")
            continue

        model = load_trained_model(ckpt_path, device)

        # Load validation dataset
        val_dataset = MRNetDataset(
            data_dir   = data_dir,
            split      = 'valid',
            plane      = plane,
            label_type = label_type,
            transform  = val_transforms,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=8,
            shuffle=False, num_workers=2,
        )

        # Run inference
        probs, preds, labels = run_inference(
            model, val_loader, device
        )

        # Compute metrics
        label_name = {
            'abnormal': 'Abnormal',
            'acl'     : 'ACL Tear',
            'meniscus': 'Meniscus',
        }[label_type]

        metrics = compute_clinical_metrics(
            probs, preds, labels, label_name
        )

        # Optimal threshold
        opt_thresh, opt_sens, opt_spec = find_optimal_threshold(
            probs, labels
        )
        metrics['optimal_threshold']    = round(opt_thresh, 3)
        metrics['optimal_sensitivity']  = round(opt_sens,   3)
        metrics['optimal_specificity']  = round(opt_spec,   3)

        all_results.append({
            'label_type': label_type,
            'metrics'   : metrics,
            'probs'     : probs,
            'preds'     : preds,
            'labels'    : labels,
        })

        # Print summary
        print(f"  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
        print(f"  Specificity : {metrics['specificity']:.4f}")
        print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"  F1 Score    : {metrics['f1_score']:.4f}")
        print(f"  Optimal threshold: {opt_thresh:.3f} "
              f"(sens={opt_sens:.3f}, spec={opt_spec:.3f})")

        # Save confusion matrix per label
        plot_confusion_matrix(
            preds, labels,
            label_name=label_name,
            save_path=f"{report_dir}/cm_{label_type}.png",
        )

    if not all_results:
        print("No checkpoints found. Train models first.")
        return pd.DataFrame()

    # Combined plots
    plot_roc_curve(
        all_results,
        save_path=f"{report_dir}/roc_curves.png"
    )
    plot_precision_recall(
        all_results,
        save_path=f"{report_dir}/precision_recall.png"
    )

    metrics_list = [r['metrics'] for r in all_results]
    plot_metrics_summary(
        metrics_list,
        save_path=f"{report_dir}/metrics_summary.png"
    )

    # Return as DataFrame
    df = pd.DataFrame(metrics_list).set_index('label')
    df.to_csv(f"{report_dir}/metrics.csv")
    print(f"\nMetrics saved → {report_dir}/metrics.csv")

    return df
