"""PCQM dataset loaders."""

from typing import Optional, Dict, Any, Tuple
import torch
from torch_geometric.data import Dataset

from .base import BaseGraphDataset, InMemoryGraphDataset
from .transforms import CompositeTransform

try:
    from ogb.lsc import PCQM4MDataset, PCQMContactDataset
    OGB_LSC_AVAILABLE = True
except ImportError:
    OGB_LSC_AVAILABLE = False
    PCQM4MDataset = None
    PCQMContactDataset = None


class PCQM4MDatasetWrapper(BaseGraphDataset):
    """PCQM4M dataset wrapper with MSPE support."""

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
        subset: Optional[int] = None,
    ):
        """
        Initialize PCQM4M dataset.

        Args:
            root: Root directory where dataset should be saved.
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
            subset: If provided, use only first N samples (for testing).
        """
        if not OGB_LSC_AVAILABLE:
            raise ImportError(
                "OGB LSC is not installed. Install it with: pip install ogb"
            )

        super().__init__(root, transform, pre_transform, pe_config)
        self.subset = subset
        self._num_classes = None

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Initialize OGB dataset
        self.full_dataset = PCQM4MDataset(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
        )

        # Extract splits
        self._extract_splits()

    def _extract_splits(self):
        """Extract train/val/test splits."""
        split_idx = self.full_dataset.get_idx_split()

        train_indices = split_idx["train"]
        val_indices = split_idx["valid"]
        test_indices = split_idx["test-dev"]

        if self.subset is not None:
            train_indices = train_indices[: self.subset]
            val_indices = val_indices[: min(self.subset // 10, len(val_indices))]
            test_indices = test_indices[: min(self.subset // 10, len(test_indices))]

        train_data = [self.full_dataset[i] for i in train_indices]
        val_data = [self.full_dataset[i] for i in val_indices]
        test_data = [self.full_dataset[i] for i in test_indices]

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
                self._num_features = 9  # Default for molecular graphs
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


class PCQMContactDatasetWrapper(BaseGraphDataset):
    """PCQM-Contact dataset wrapper with MSPE support."""

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
        subset: Optional[int] = None,
    ):
        """
        Initialize PCQM-Contact dataset.

        Args:
            root: Root directory where dataset should be saved.
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
            subset: If provided, use only first N samples (for testing).
        """
        if not OGB_LSC_AVAILABLE:
            raise ImportError(
                "OGB LSC is not installed. Install it with: pip install ogb"
            )

        super().__init__(root, transform, pre_transform, pe_config)
        self.subset = subset
        self._num_classes = None

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Initialize OGB dataset
        self.full_dataset = PCQMContactDataset(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
        )

        # Extract splits
        self._extract_splits()

    def _extract_splits(self):
        """Extract train/val/test splits."""
        split_idx = self.full_dataset.get_idx_split()

        train_indices = split_idx["train"]
        val_indices = split_idx["valid"]
        test_indices = split_idx["test"]

        if self.subset is not None:
            train_indices = train_indices[: self.subset]
            val_indices = val_indices[: min(self.subset // 10, len(val_indices))]
            test_indices = test_indices[: min(self.subset // 10, len(test_indices))]

        train_data = [self.full_dataset[i] for i in train_indices]
        val_data = [self.full_dataset[i] for i in val_indices]
        test_data = [self.full_dataset[i] for i in test_indices]

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
                self._num_features = 9  # Default for molecular graphs
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Number of classes (1 for edge prediction regression)."""
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

