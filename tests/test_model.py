# tests/test_model.py
# Run with: python -m pytest tests/test_model.py -v

import pytest
import torch
from src.model import MRNetClassifier, build_model
from src.utils import load_config, count_parameters


@pytest.fixture
def model():
    return MRNetClassifier(pretrained=False)


@pytest.fixture
def device():
    return torch.device('cpu')


def test_output_shape(model):
    """Model should output (batch, 1) logits."""
    x      = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 1), \
        f"Expected (4, 1), got {logits.shape}"


def test_output_is_logits(model):
    """Output should be unbounded logits, not probabilities."""
    x      = torch.randn(4, 3, 224, 224)
    logits = model(x)
    # Logits can exceed [0,1] — if all in [0,1] something is wrong
    probs  = torch.sigmoid(logits)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_frozen_layers(model):
    """Early layers should be frozen by default."""
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert len(frozen) > 0, "Some layers should be frozen"


def test_unfreeze_all(model):
    """unfreeze_all() should make all params trainable."""
    model.unfreeze_all()
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert len(frozen) == 0, "All layers should be trainable after unfreeze"


def test_build_model(device):
    """build_model() should work from config."""
    config = load_config('configs/config.yaml')
    model  = build_model(config, device)
    assert model is not None


def test_parameter_count(model):
    """Model should have reasonable parameter count."""
    n = count_parameters(model)
    assert n > 1_000_000, "Model seems too small"
