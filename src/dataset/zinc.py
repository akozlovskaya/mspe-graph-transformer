"""ZINC dataset loader."""

from typing import Optional, Dict, Any, Tuple
import torch
from torch_geometric.datasets import ZINC as PyGZINC
from torch_geometric.data import Dataset

from .base import BaseGraphDataset
from .transforms import CompositeTransform


class ZINCDataset(BaseGraphDataset):
    """ZINC molecular dataset with MSPE support."""

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
        subset: bool = True,
    ):
        """
        Initialize ZINC dataset.

        Args:
            root: Root directory where dataset should be saved.
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
            subset: If True, use subset (10k train, 1k val, 1k test).
        """
        super().__init__(root, transform, pre_transform, pe_config)
        self.subset = subset

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Initialize PyG datasets
        self.train_dataset = PyGZINC(
            root=root,
            split="train",
            subset=subset,
            transform=transform,
            pre_transform=pre_transform,
        )
        self.val_dataset = PyGZINC(
            root=root,
            split="val",
            subset=subset,
            transform=transform,
            pre_transform=pre_transform,
        )
        self.test_dataset = PyGZINC(
            root=root,
            split="test",
            subset=subset,
            transform=transform,
            pre_transform=pre_transform,
        )

    def load(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train, val, and test splits."""
        return self.train_dataset, self.val_dataset, self.test_dataset

    @property
    def num_features(self) -> int:
        """Number of node features."""
        if self._num_features is None:
            # ZINC uses one-hot encoding of atom types
            # Default: 9 atom types
            sample = self.train_dataset[0]
            if hasattr(sample, "x") and sample.x is not None:
                self._num_features = sample.x.size(1)
            else:
                self._num_features = 9  # Default for ZINC
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

