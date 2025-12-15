"""Dataset loaders and utilities for Multi-Scale Positional Encodings."""

from .factory import get_dataset, list_available_datasets
from .base import BaseGraphDataset, InMemoryGraphDataset
from .transforms import (
    ApplyNodePE,
    ApplyRelativePE,
    CompositeTransform,
    NormalizeTargets,
    CastDataTypes,
)
from .utils import (
    compute_dataset_stats,
    create_random_split,
    normalize_targets,
    ensure_data_has_pe,
    collate_with_pe,
)

__all__ = [
    # Factory
    "get_dataset",
    "list_available_datasets",
    # Base classes
    "BaseGraphDataset",
    "InMemoryGraphDataset",
    # Transforms
    "ApplyNodePE",
    "ApplyRelativePE",
    "CompositeTransform",
    "NormalizeTargets",
    "CastDataTypes",
    # Utils
    "compute_dataset_stats",
    "create_random_split",
    "normalize_targets",
    "ensure_data_has_pe",
    "collate_with_pe",
]
