"""OGB molecular dataset loaders."""

from typing import Optional, Dict, Any, Tuple
import torch
from torch_geometric.data import Dataset

from .base import BaseGraphDataset, InMemoryGraphDataset
from .transforms import CompositeTransform

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.graphproppred import Evaluator as GraphPropEvaluator
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False
    PygGraphPropPredDataset = None
    GraphPropEvaluator = None


class OGBMolDataset(BaseGraphDataset):
    """OGB molecular dataset with MSPE support."""

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OGB molecular dataset.

        Args:
            root: Root directory where dataset should be saved.
            name: Dataset name (e.g., "ogbg-molhiv", "ogbg-molpcba").
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
        """
        if not OGB_AVAILABLE:
            raise ImportError(
                "OGB is not installed. Install it with: pip install ogb"
            )

        super().__init__(root, transform, pre_transform, pe_config)
        self.name = name
        self._num_classes = None

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Initialize OGB dataset
        self.full_dataset = PygGraphPropPredDataset(
            name=name,
            root=root,
            transform=transform,
            pre_transform=pre_transform,
        )

        # Get evaluator
        self.evaluator = GraphPropEvaluator(name)

        # Extract splits
        self._extract_splits()

    def _extract_splits(self):
        """Extract train/val/test splits from OGB dataset."""
        split_idx = self.full_dataset.get_idx_split()

        train_indices = split_idx["train"]
        val_indices = split_idx["valid"]
        test_indices = split_idx["test"]

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
                # Default for OGB molecular datasets
                self._num_features = 9  # Atom types
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        if self._num_classes is None:
            sample = self.train_dataset[0]
            if hasattr(sample, "y") and sample.y is not None:
                if sample.y.dim() == 1:
                    self._num_classes = 1
                else:
                    self._num_classes = sample.y.size(-1)
            else:
                # Default based on dataset name
                if "molhiv" in self.name:
                    self._num_classes = 1  # Binary classification
                elif "molpcba" in self.name:
                    self._num_classes = 128  # 128 binary tasks
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

    def get_evaluator(self):
        """Get OGB evaluator for this dataset."""
        return self.evaluator

