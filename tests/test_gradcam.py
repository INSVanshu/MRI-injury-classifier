# tests/test_gradcam.py
# Run with: python -m pytest tests/test_gradcam.py -v

import pytest
import torch
import numpy as np
from src.gradcam import GradCAM
from src.model import MRNetClassifier


@pytest.fixture
def model():
    m = MRNetClassifier(pretrained=False)
    m.eval()
    return m


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 224, 224)


def test_gradcam_output_shape(model, device, dummy_input):
    """Heatmap should match input spatial resolution."""
    gradcam = GradCAM(model)
    prob, heatmap = gradcam.generate(dummy_input, device)
    assert heatmap.shape == (224, 224), \
        f"Expected (224, 224), got {heatmap.shape}"


def test_probability_range(model, device, dummy_input):
    """Probability should be in [0, 1]."""
    gradcam = GradCAM(model)
    prob, _ = gradcam.generate(dummy_input, device)
    assert 0.0 <= prob <= 1.0, \
        f"Probability {prob} out of range [0, 1]"


def test_heatmap_normalized(model, device, dummy_input):
    """Heatmap values should be in [0, 1]."""
    gradcam = GradCAM(model)
    _, heatmap = gradcam.generate(dummy_input, device)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_shape(model, device, dummy_input):
    """Overlay should be (224, 224, 3) uint8."""
    gradcam = GradCAM(model)
    _, heatmap = gradcam.generate(dummy_input, device)
    original   = np.random.rand(224, 224).astype(np.float32)
    overlay    = gradcam.overlay(original, heatmap)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_hooks_registered(model):
    """Hooks should be attached after GradCAM init."""
    gradcam = GradCAM(model)
    # Forward hooks are stored on the layer
    assert len(gradcam.target_layer._forward_hooks) > 0
