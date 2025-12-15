"""Base class for relative (pairwise) positional encodings."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from torch_geometric.data import Data


class BaseRelativePE(ABC):
    """Abstract base class for relative pairwise positional encodings."""

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
    ):
        """
        Initialize base relative PE.

        Args:
            num_buckets: Number of discrete buckets or embedding channels.
            max_distance: Maximum graph distance to consider.
            normalization: Normalization mode: 'graph', 'pair', or None.
            symmetric: Whether PE(i,j) == PE(j,i) should hold.
            cache: Whether to cache computed PE in data.
        """
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.normalization = normalization
        self.symmetric = symmetric
        self.cache = cache

    @abstractmethod
    def compute(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relative pairwise positional encodings.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: LongTensor [2, num_pairs] with pair indices
                - edge_attr_pe: Tensor [num_pairs, num_buckets] with PE values
        """
        raise NotImplementedError

    def __call__(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PE with caching support.

        Args:
            data: PyG Data object.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe).
        """
        # Check cache
        if self.cache and hasattr(data, "edge_pe_index") and hasattr(data, "edge_pe"):
            if (
                data.edge_pe_index is not None
                and data.edge_pe is not None
                and data.edge_pe.shape[1] == self.num_buckets
            ):
                return data.edge_pe_index, data.edge_pe

        # Compute PE
        edge_index_pe, edge_attr_pe = self.compute(data)

        # Apply normalization
        edge_attr_pe = self._normalize(edge_attr_pe)

        # Ensure symmetric if required
        if self.symmetric:
            edge_index_pe, edge_attr_pe = self._make_symmetric(
                edge_index_pe, edge_attr_pe
            )

        # Cache if requested
        if self.cache:
            data.edge_pe_index = edge_index_pe
            data.edge_pe = edge_attr_pe

        return edge_index_pe, edge_attr_pe

    def _normalize(self, edge_attr_pe: torch.Tensor) -> torch.Tensor:
        """
        Normalize relative positional encodings.

        Args:
            edge_attr_pe: Tensor of shape [num_pairs, num_buckets].

        Returns:
            Normalized tensor.
        """
        if self.normalization is None:
            return edge_attr_pe

        if self.normalization == "graph":
            # Zero mean, unit std across all pairs
            mean = edge_attr_pe.mean(dim=0, keepdim=True)
            std = edge_attr_pe.std(dim=0, keepdim=True) + 1e-8
            return (edge_attr_pe - mean) / std

        elif self.normalization == "pair":
            # Normalize per pair (across buckets)
            mean = edge_attr_pe.mean(dim=1, keepdim=True)
            std = edge_attr_pe.std(dim=1, keepdim=True) + 1e-8
            return (edge_attr_pe - mean) / std

        else:
            raise ValueError(f"Unknown normalization mode: {self.normalization}")

    def _make_symmetric(
        self, edge_index_pe: torch.Tensor, edge_attr_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make PE symmetric by adding reverse edges.

        Args:
            edge_index_pe: Tensor [2, num_pairs].
            edge_attr_pe: Tensor [num_pairs, num_buckets].

        Returns:
            Symmetric (edge_index_pe, edge_attr_pe).
        """
        # Check if already symmetric (has both (i,j) and (j,i))
        # For now, simple approach: add reverse edges if not present
        # More sophisticated check could be added

        # Add reverse edges
        reverse_idx = torch.stack([edge_index_pe[1], edge_index_pe[0]], dim=0)
        combined_idx = torch.cat([edge_index_pe, reverse_idx], dim=1)

        # Combine attributes (should be same for symmetric PE)
        combined_attr = torch.cat([edge_attr_pe, edge_attr_pe], dim=0)

        # Remove duplicates (keep first occurrence)
        # Convert to hashable format for duplicate removal
        num_pairs = combined_idx.size(1)
        pairs_set = {}
        unique_idx_list = []
        unique_attr_list = []

        for i in range(num_pairs):
            pair = (combined_idx[0, i].item(), combined_idx[1, i].item())
            if pair not in pairs_set:
                pairs_set[pair] = True
                unique_idx_list.append(combined_idx[:, i])
                unique_attr_list.append(combined_attr[i])

        if unique_idx_list:
            edge_index_pe = torch.stack(unique_idx_list, dim=1)
            edge_attr_pe = torch.stack(unique_attr_list, dim=0)
        else:
            edge_index_pe = combined_idx
            edge_attr_pe = combined_attr

        return edge_index_pe, edge_attr_pe

