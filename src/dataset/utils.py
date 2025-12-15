"""Utility functions for dataset processing."""

from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
from torch_geometric.data import Data, Batch


def compute_dataset_stats(dataset) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.

    Args:
        dataset: PyG Dataset

    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_graphs": len(dataset),
        "num_nodes": [],
        "num_edges": [],
        "num_features": None,
        "num_classes": None,
    }

    targets = []

    for data in dataset:
        stats["num_nodes"].append(data.num_nodes)
        stats["num_edges"].append(data.num_edges)

        if stats["num_features"] is None and hasattr(data, "x") and data.x is not None:
            stats["num_features"] = data.x.size(1)

        if stats["num_classes"] is None and hasattr(data, "y") and data.y is not None:
            if data.y.dim() == 0:
                stats["num_classes"] = 1
            elif data.y.dim() == 1:
                stats["num_classes"] = data.y.size(0)
            else:
                stats["num_classes"] = data.y.size(-1)

        if hasattr(data, "y") and data.y is not None:
            targets.append(data.y.cpu().numpy())

    stats["avg_num_nodes"] = np.mean(stats["num_nodes"])
    stats["avg_num_edges"] = np.mean(stats["num_edges"])
    stats["min_num_nodes"] = np.min(stats["num_nodes"])
    stats["max_num_nodes"] = np.max(stats["num_nodes"])

    if targets:
        targets = np.concatenate([t.flatten() for t in targets])
        stats["target_mean"] = float(np.mean(targets))
        stats["target_std"] = float(np.std(targets))
        stats["target_min"] = float(np.min(targets))
        stats["target_max"] = float(np.max(targets))

    return stats


def create_random_split(
    dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create random split indices.

    Args:
        dataset: PyG Dataset
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size : train_size + val_size].tolist()
    test_indices = indices[train_size + val_size :].tolist()

    return train_indices, val_indices, test_indices


def normalize_targets(dataset, mean: Optional[float] = None, std: Optional[float] = None):
    """
    Normalize target values in dataset.

    Args:
        dataset: PyG Dataset (will be modified in-place)
        mean: Mean for normalization (computed if None)
        std: Std for normalization (computed if None)
    """
    if mean is None or std is None:
        targets = []
        for data in dataset:
            if hasattr(data, "y") and data.y is not None:
                targets.append(data.y.cpu().numpy())
        if targets:
            targets = np.concatenate([t.flatten() for t in targets])
            mean = float(np.mean(targets))
            std = float(np.std(targets))

    for data in dataset:
        if hasattr(data, "y") and data.y is not None:
            data.y = (data.y - mean) / (std + 1e-8)

    return mean, std


def ensure_data_has_pe(data: Data) -> Data:
    """
    Ensure data object has node_pe and edge_pe attributes.

    Args:
        data: PyG Data object

    Returns:
        Data object with PE attributes (zeros if missing)
    """
    if not hasattr(data, "node_pe") or data.node_pe is None:
        data.node_pe = torch.zeros(data.num_nodes, 32, dtype=torch.float32)

    if not hasattr(data, "edge_pe") or data.edge_pe is None:
        num_edges = data.edge_index.size(1) if hasattr(data, "edge_index") else 0
        data.edge_pe = torch.zeros(num_edges, 16, dtype=torch.float32)

    return data


def collate_with_pe(batch: List[Data]) -> Batch:
    """
    Collate function that handles PE attributes.

    Args:
        batch: List of Data objects

    Returns:
        Batched Data object
    """
    # Ensure all graphs have PE
    batch = [ensure_data_has_pe(data) for data in batch]

    # Use PyG's default collate
    from torch_geometric.data import Batch

    return Batch.from_data_list(batch)

