# src/dataset.py
# PyTorch Dataset class for Stanford MRNet

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
from src.transforms import zscore_normalize, to_rgb


class MRNetDataset(Dataset):
    """
    PyTorch Dataset for the Stanford MRNet knee MRI dataset.

    Each exam is a 3D volume: (num_slices, H, W)
    Strategy: extract the single most informative slice
    (the one with maximum intensity variance — likely
    the slice closest to the injury site).

    Args:
        data_dir   : Path to data/raw/
        split      : 'train' or 'valid'
        plane      : 'sagittal', 'coronal', or 'axial'
        label_type : 'abnormal', 'acl', or 'meniscus'
        transform  : torchvision transforms to apply
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        plane: str = 'sagittal',
        label_type: str = 'acl',
        transform=None,
    ):
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.plane      = plane
        self.label_type = label_type
        self.transform  = transform

        # Load labels
        label_path = self.data_dir / 'labels' / f'{split}-{label_type}.csv'
        self.labels_df = pd.read_csv(
            label_path, header=None, names=['id', 'label']
        )

        print(f"[MRNetDataset] {split} | plane={plane} | "
              f"label={label_type} | n={len(self.labels_df)}")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row      = self.labels_df.iloc[idx]
        exam_id  = int(row['id'])
        label    = int(row['label'])

        # Load volume: shape (num_slices, H, W)
        vol_path = (self.data_dir / self.split /
                    self.plane / f'{exam_id:0>4}.npy')
        volume   = np.load(vol_path)                # (S, H, W)

        # Z-score normalize the full volume
        volume = zscore_normalize(volume)

        # Select most informative slice
        slice_2d = self._select_slice(volume)       # (H, W)

        # Convert to uint8 for torchvision transforms
        slice_2d = self._to_uint8(slice_2d)         # (H, W) uint8

        # Convert grayscale → RGB: (H, W) → (H, W, 3)
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)

        # Apply transforms
        if self.transform:
            tensor = self.transform(slice_rgb)      # (3, 224, 224)
        else:
            tensor = torch.from_numpy(
                slice_rgb.transpose(2, 0, 1)
            ).float()

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return tensor, label_tensor

    def _select_slice(self, volume: np.ndarray) -> np.ndarray:
        """
        Select the slice with the highest variance.
        High variance slices contain more structural detail
        and are most likely to show injury regions.
        """
        variances = [volume[i].var() for i in range(volume.shape[0])]
        best_idx  = int(np.argmax(variances))
        return volume[best_idx]

    def _to_uint8(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Scale normalized float slice to uint8 [0, 255].
        Required for torchvision PIL transforms.
        """
        s_min, s_max = slice_2d.min(), slice_2d.max()
        if s_max - s_min == 0:
            return np.zeros_like(slice_2d, dtype=np.uint8)
        scaled = (slice_2d - s_min) / (s_max - s_min) * 255
        return scaled.astype(np.uint8)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights to handle class imbalance.
        Used with BCEWithLogitsLoss pos_weight parameter.
        """
        n_pos = self.labels_df['label'].sum()
        n_neg = len(self.labels_df) - n_pos
        weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        print(f"  pos_weight = {weight.item():.2f} "
              f"(neg={int(n_neg)}, pos={int(n_pos)})")
        return weight
