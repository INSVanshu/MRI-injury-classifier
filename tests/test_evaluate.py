# tests/test_evaluate.py
# Run with: python -m pytest tests/test_evaluate.py -v

import pytest
import numpy as np
from src.evaluate import (
    compute_clinical_metrics,
    find_optimal_threshold,
)


@pytest.fixture
def perfect_predictions():
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    probs  = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95, 0.05, 0.85])
    preds  = (probs >= 0.5).astype(int)
    return probs, preds, labels


@pytest.fixture
def imbalanced_predictions():
    # 80% negative, 20% positive — like real MRNet
    labels = np.array([0]*80 + [1]*20)
    probs  = np.random.rand(100)
    preds  = (probs >= 0.5).astype(int)
    return probs, preds, labels


def test_perfect_sensitivity(perfect_predictions):
    probs, preds, labels = perfect_predictions
    metrics = compute_clinical_metrics(probs, preds, labels)
    assert metrics['sensitivity'] == pytest.approx(1.0, abs=0.01)


def test_perfect_specificity(perfect_predictions):
    probs, preds, labels = perfect_predictions
    metrics = compute_clinical_metrics(probs, preds, labels)
    assert metrics['specificity'] == pytest.approx(1.0, abs=0.01)


def test_perfect_auc(perfect_predictions):
    probs, preds, labels = perfect_predictions
    metrics = compute_clinical_metrics(probs, preds, labels)
    assert metrics['roc_auc'] > 0.95


def test_metrics_keys(perfect_predictions):
    probs, preds, labels = perfect_predictions
    metrics = compute_clinical_metrics(probs, preds, labels)
    required = [
        'accuracy', 'sensitivity', 'specificity',
        'precision', 'f1_score', 'roc_auc',
        'tp', 'tn', 'fp', 'fn',
    ]
    for key in required:
        assert key in metrics, f"Missing metric: {key}"


def test_optimal_threshold_range(perfect_predictions):
    probs, _, labels = perfect_predictions
    threshold, sens, spec = find_optimal_threshold(probs, labels)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= sens <= 1.0
    assert 0.0 <= spec <= 1.0


def test_imbalanced_metrics(imbalanced_predictions):
    probs, preds, labels = imbalanced_predictions
    metrics = compute_clinical_metrics(probs, preds, labels)
    # All metrics should be computable without errors
    assert 0.0 <= metrics['roc_auc'] <= 1.0
