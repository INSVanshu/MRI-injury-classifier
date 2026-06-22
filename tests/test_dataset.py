# tests/test_dataset.py
# Run with: python -m pytest tests/ -v

import pytest
import torch
from torch.utils.data import DataLoader
from src.dataset import MRNetDataset
from src.transforms import train_transforms, val_transforms


DATA_DIR = 'data/raw'


@pytest.fixture
def train_dataset():
    return MRNetDataset(
        data_dir=DATA_DIR,
        split='train',
        plane='sagittal',
        label_type='acl',
        transform=val_transforms,
    )


def test_dataset_length(train_dataset):
    assert len(train_dataset) > 0, "Dataset should not be empty"


def test_item_shape(train_dataset):
    image, label = train_dataset[0]
    assert image.shape == (3, 224, 224), \
        f"Expected (3, 224, 224), got {image.shape}"


def test_label_is_binary(train_dataset):
    _, label = train_dataset[0]
    assert label.item() in [0.0, 1.0], \
        f"Label should be 0 or 1, got {label.item()}"


def test_dataloader_batch(train_dataset):
    loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)


def test_class_weights(train_dataset):
    weights = train_dataset.get_class_weights()
    assert weights.item() > 0, "Class weight should be positive"
