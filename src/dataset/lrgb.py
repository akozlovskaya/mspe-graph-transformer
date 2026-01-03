"""LRGB dataset loader."""

from typing import Optional, Dict, Any, Tuple
import torch
from torch_geometric.datasets import LRGBDataset as PyGLRGBDataset
from torch_geometric.data import Dataset

from .base import BaseGraphDataset
from .transforms import CompositeTransform


class LRGBDataset(BaseGraphDataset):
    """LRGB dataset with MSPE support."""

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LRGB dataset.

        Args:
            root: Root directory where dataset should be saved.
            name: Dataset name (e.g., "Peptides-func", "Peptides-struct",
                  "PascalVOC-SP", "CIFAR10-SP").
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
        """
        super().__init__(root, transform, pre_transform, pe_config)
        self.name = name
        self._num_classes = None

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Initialize PyG dataset
        self.full_dataset = PyGLRGBDataset(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
        )

        # LRGB datasets have built-in splits
        self._extract_splits()

    def _extract_splits(self):
        """Extract train/val/test splits from full dataset."""
        from .base import InMemoryGraphDataset

        train_data = []
        val_data = []
        test_data = []

        # Try to extract splits from data.split attribute
        for data in self.full_dataset:
            # LRGB datasets have split attribute
            if hasattr(data, "split"):
                split = data.split
                if split == "train":
                    train_data.append(data)
                elif split == "val":
                    val_data.append(data)
                elif split == "test":
                    test_data.append(data)

        # If no split info found or some splits are missing, create manual split
        if not train_data or not val_data or not test_data:
            # Create manual split from all data
            total = len(self.full_dataset)
            train_size = int(0.8 * total)
            val_size = int(0.1 * total)

            train_data = [self.full_dataset[i] for i in range(train_size)]
            val_data = [
                self.full_dataset[i] for i in range(train_size, train_size + val_size)
            ]
            test_data = [self.full_dataset[i] for i in range(train_size + val_size, total)]

        # Create datasets (should never be empty after manual split)
        self.train_dataset = InMemoryGraphDataset(train_data) if train_data else None
        self.val_dataset = InMemoryGraphDataset(val_data) if val_data else None
        self.test_dataset = InMemoryGraphDataset(test_data) if test_data else None

        # Final fallback: if still None, raise error
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise ValueError(
                f"Failed to create dataset splits for {self.name}. "
                f"Train: {self.train_dataset is not None}, "
                f"Val: {self.val_dataset is not None}, "
                f"Test: {self.test_dataset is not None}"
            )

    def load(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train, val, and test splits."""
        return self.train_dataset, self.val_dataset, self.test_dataset

    @property
    def num_features(self) -> int:
        """Number of node features."""
        if self._num_features is None:
            sample = self.train_dataset[0] if self.train_dataset else self.full_dataset[0]
            if hasattr(sample, "x") and sample.x is not None:
                self._num_features = sample.x.size(1)
            else:
                # Default based on dataset name
                if "Peptides" in self.name:
                    self._num_features = 9
                elif "PascalVOC" in self.name or "CIFAR10" in self.name:
                    self._num_features = 3  # RGB superpixels
                else:
                    self._num_features = 1
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        if self._num_classes is None:
            sample = self.train_dataset[0] if self.train_dataset else self.full_dataset[0]
            if hasattr(sample, "y") and sample.y is not None:
                if sample.y.dim() == 0:
                    # Classification: count unique classes
                    all_labels = []
                    for data in self.train_dataset[:100]:  # Sample first 100
                        if hasattr(data, "y") and data.y is not None:
                            all_labels.append(data.y.item())
                    self._num_classes = len(set(all_labels)) if all_labels else 1
                else:
                    # Multi-label or regression
                    self._num_classes = sample.y.size(-1) if sample.y.dim() > 0 else 1
            else:
                # Default based on dataset name
                if "func" in self.name.lower():
                    self._num_classes = 10  # Peptides-func: 10 binary tasks
                elif "struct" in self.name.lower():
                    self._num_classes = 11  # Peptides-struct: 11 classes
                elif "PascalVOC" in self.name:
                    self._num_classes = 21
                elif "CIFAR10" in self.name:
                    self._num_classes = 10
                else:
                    self._num_classes = 1
        return self._num_classes

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

