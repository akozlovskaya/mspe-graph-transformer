"""Shortest-Path Distance (SPD) buckets implementation."""

from typing import Optional, Tuple
import torch
from torch_geometric.data import Data

from .base import BaseRelativePE
from .utils import (
    compute_shortest_path_distances,
    distances_to_one_hot,
    distances_to_embeddings,
)


class SPDBuckets(BaseRelativePE):
    """
    Shortest-Path Distance (SPD) Buckets.

    Computes pairwise shortest path distances and discretizes them into buckets.

    The distance d(i,j) between nodes i and j is computed using shortest paths,
    then mapped to discrete buckets for use in attention mechanisms.

    Reference:
        "Graph Transformer Networks" (Dwivedi & Bresson, 2020)
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
        use_one_hot: bool = True,
    ):
        """
        Initialize SPDBuckets.

        Args:
            num_buckets: Number of distance buckets.
            max_distance: Maximum distance to consider (distances > max_distance go to last bucket).
            normalization: Normalization mode: 'graph', 'pair', or None.
            symmetric: Whether to ensure symmetric PE (PE(i,j) == PE(j,i)).
            cache: Whether to cache computed PE.
            use_one_hot: If True, returns one-hot encodings; if False, returns bucket indices.
        """
        super().__init__(
            num_buckets=num_buckets,
            max_distance=max_distance,
            normalization=normalization,
            symmetric=symmetric,
            cache=cache,
        )
        self.use_one_hot = use_one_hot

    def compute(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SPD bucket encodings for all pairs.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs] with all pairs (i,j)
                - edge_attr_pe: [num_pairs, num_buckets] (one-hot) or [num_pairs] (indices)
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Compute shortest path distance matrix
        distances = compute_shortest_path_distances(
            edge_index, num_nodes, max_distance=self.max_distance
        )

        # Create all pairs
        i_indices, j_indices = torch.meshgrid(
            torch.arange(num_nodes, device=edge_index.device),
            torch.arange(num_nodes, device=edge_index.device),
            indexing="ij",
        )
        edge_index_pe = torch.stack([i_indices.flatten(), j_indices.flatten()], dim=0)

        # Get distances for all pairs
        pair_distances = distances[i_indices.flatten(), j_indices.flatten()]

        # Convert to buckets
        if self.use_one_hot:
            edge_attr_pe = distances_to_one_hot(
                pair_distances, self.num_buckets, self.max_distance
            )
        else:
            edge_attr_pe = distances_to_embeddings(
                pair_distances, self.num_buckets, self.max_distance
            )
            # Add dimension for consistency
            edge_attr_pe = edge_attr_pe.unsqueeze(1)

        return edge_index_pe, edge_attr_pe


class SPDBucketsSparse(BaseRelativePE):
    """
    Sparse version of SPDBuckets (only stores pairs within max_distance).

    More memory-efficient for large graphs with limited connectivity.
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
        use_one_hot: bool = True,
    ):
        """
        Initialize sparse SPDBuckets.

        Args:
            num_buckets: Number of distance buckets.
            max_distance: Maximum distance to store (pairs with d > max_distance are excluded).
            normalization: Normalization mode: 'graph', 'pair', or None.
            symmetric: Whether to ensure symmetric PE.
            cache: Whether to cache computed PE.
            use_one_hot: If True, returns one-hot encodings.
        """
        super().__init__(
            num_buckets=num_buckets,
            max_distance=max_distance,
            normalization=normalization,
            symmetric=symmetric,
            cache=cache,
        )
        self.use_one_hot = use_one_hot

    def compute(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse SPD bucket encodings (only pairs within max_distance).

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs] with pairs within max_distance
                - edge_attr_pe: [num_pairs, num_buckets] or [num_pairs, 1]
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Compute shortest path distance matrix
        distances = compute_shortest_path_distances(
            edge_index, num_nodes, max_distance=self.max_distance
        )

        # Find pairs within max_distance
        valid_mask = distances <= self.max_distance

        # Get valid pairs
        i_indices, j_indices = torch.meshgrid(
            torch.arange(num_nodes, device=edge_index.device),
            torch.arange(num_nodes, device=edge_index.device),
            indexing="ij",
        )
        valid_i = i_indices[valid_mask]
        valid_j = j_indices[valid_mask]

        edge_index_pe = torch.stack([valid_i, valid_j], dim=0)

        # Get distances for valid pairs
        pair_distances = distances[valid_i, valid_j]

        # Convert to buckets
        if self.use_one_hot:
            edge_attr_pe = distances_to_one_hot(
                pair_distances, self.num_buckets, self.max_distance
            )
        else:
            edge_attr_pe = distances_to_embeddings(
                pair_distances, self.num_buckets, self.max_distance
            )
            edge_attr_pe = edge_attr_pe.unsqueeze(1)

        return edge_index_pe, edge_attr_pe

