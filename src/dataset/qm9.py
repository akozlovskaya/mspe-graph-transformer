"""QM9 dataset loader."""

from typing import Optional, Dict, Any, Tuple
import torch
from torch_geometric.datasets import QM9 as PyGQM9
from torch_geometric.data import Dataset

from .base import BaseGraphDataset
from .transforms import CompositeTransform, NormalizeTargets


class QM9Dataset(BaseGraphDataset):
    """QM9 molecular dataset with MSPE support."""

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
        target_idx: int = 0,
        normalize_targets: bool = True,
    ):
        """
        Initialize QM9 dataset.

        Args:
            root: Root directory where dataset should be saved.
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
            target_idx: Index of target property (0-18).
            normalize_targets: Whether to normalize target values.
        """
        super().__init__(root, transform, pre_transform, pe_config)
        self.target_idx = target_idx
        self.normalize_targets = normalize_targets

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(
                pe_config=pe_config, normalize_targets=normalize_targets
            )

        # Initialize PyG dataset
        self.full_dataset = PyGQM9(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
        )

        # QM9 has 19 targets, we select one
        # Extract target and create splits
        self._create_splits()

    def _create_splits(self):
        """Create train/val/test splits from full dataset."""
        # QM9 official split: first 100k train, next 10k val, rest test
        train_size = 100000
        val_size = 10000

        # Extract target
        targets = []
        for data in self.full_dataset:
            if hasattr(data, "y") and data.y is not None:
                target = data.y[:, self.target_idx].item()
                targets.append(target)
            else:
                targets.append(0.0)

        # Compute normalization stats from training set
        train_targets = targets[:train_size]
        if self.normalize_targets:
            self.target_mean = sum(train_targets) / len(train_targets)
            self.target_std = (
                sum((t - self.target_mean) ** 2 for t in train_targets) / len(train_targets)
            ) ** 0.5
        else:
            self.target_mean = None
            self.target_std = None

        # Create split datasets
        from .base import InMemoryGraphDataset

        train_data = []
        val_data = []
        test_data = []

        for i, data in enumerate(self.full_dataset):
            # Set target
            if hasattr(data, "y") and data.y is not None:
                data.y = torch.tensor([[data.y[:, self.target_idx].item()]], dtype=torch.float32)
            else:
                data.y = torch.tensor([[0.0]], dtype=torch.float32)

            if i < train_size:
                train_data.append(data)
            elif i < train_size + val_size:
                val_data.append(data)
            else:
                test_data.append(data)

        self.train_dataset = InMemoryGraphDataset(train_data)
        self.val_dataset = InMemoryGraphDataset(val_data)
        self.test_dataset = InMemoryGraphDataset(test_data)

    def load(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train, val, and test splits."""
        return self.train_dataset, self.val_dataset, self.test_dataset

    @property
    def num_features(self) -> int:
        """Number of node features."""
        if self._num_features is None:
            sample = self.train_dataset[0]
            if hasattr(sample, "x") and sample.x is not None:
                self._num_features = sample.x.size(1)
            else:
                self._num_features = 11  # Default for QM9
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Number of classes (1 for regression)."""
        return 1

    @property
    def train(self) -> Dataset:
        """Training split."""
        return self.train_dataset

    @property
    def val(self) -> Dataset:
        """Validation split."""
        return self.val_dataset

    @property
    def test(self) -> Dataset:
        """Test split."""
        return self.test_dataset

