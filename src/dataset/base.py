"""Base classes for graph datasets with MSPE support."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import torch
from torch_geometric.data import Dataset, Data, InMemoryDataset


class BaseGraphDataset(ABC):
    """Base class for graph datasets with unified interface."""

    def __init__(
        self,
        root: str,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base dataset.

        Args:
            root: Root directory where dataset should be saved.
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
        """
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pe_config = pe_config or {}
        self._num_features = None
        self._num_classes = None

    @abstractmethod
    def load(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load train, val, and test splits.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of node features."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes (1 for regression)."""
        pass

    def get_splits(self, splits: str = "official") -> Tuple[Dataset, Dataset, Dataset]:
        """
        Get dataset splits.

        Args:
            splits: Split type ("official" or "random")

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train, val, test = self.load()
        return train, val, test


class InMemoryGraphDataset(Dataset):
    """In-memory dataset wrapper with PE support."""

    def __init__(
        self,
        data_list: list,
        root: str = "",
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
    ):
        """
        Initialize in-memory dataset.

        Args:
            data_list: List of Data objects.
            root: Root directory (not used for in-memory).
            transform: Transform to apply.
            pre_transform: Pre-transform to apply.
        """
        super().__init__(root, transform, pre_transform)
        self.data_list = data_list

    def len(self) -> int:
        """Return dataset length."""
        return len(self.data_list) if self.data_list else 0

    def get(self, idx: int) -> Data:
        """Get item by index."""
        if self.data_list and 0 <= idx < len(self.data_list):
            data = self.data_list[idx]
            # Apply transform if provided
            if self.transform is not None:
                data = self.transform(data)
            return data
        else:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list) if self.data_list else 0}")


class PreprocessedPEDatasetWrapper(BaseGraphDataset):
    """Wrapper for datasets with preprocessed PE."""

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        base_dataset: BaseGraphDataset,
    ):
        """
        Initialize wrapper with preprocessed PE datasets.

        Args:
            train_dataset: Train dataset with preprocessed PE.
            val_dataset: Validation dataset with preprocessed PE.
            test_dataset: Test dataset with preprocessed PE.
            base_dataset: Original dataset (for metadata).
        """
        super().__init__(
            root=base_dataset.root,
            transform=None,  # PE already applied
            pre_transform=None,
            pe_config=base_dataset.pe_config,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self._base_dataset = base_dataset

    def load(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load preprocessed splits."""
        return self.train_dataset, self.val_dataset, self.test_dataset

    @property
    def num_features(self) -> int:
        """Number of node features."""
        return self._base_dataset.num_features

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self._base_dataset.num_classes

