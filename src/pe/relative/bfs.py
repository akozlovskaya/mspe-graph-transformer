"""Truncated BFS distance implementation."""

from typing import Optional, Tuple
import torch
from torch_geometric.data import Data

from .base import BaseRelativePE
from .utils import (
    compute_bfs_distances_truncated,
    distances_to_one_hot,
    distances_to_embeddings,
)


class BFSDistance(BaseRelativePE):
    """
    Truncated BFS Distance.

    Computes distances using BFS limited to k hops, storing only pairs within k.
    More memory-efficient than full SPD for sparse graphs.

    Only pairs (i,j) with d(i,j) ≤ k are stored.
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
        Initialize BFSDistance.

        Args:
            num_buckets: Number of distance buckets.
            max_distance: Maximum BFS depth (k).
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
        Compute truncated BFS distances.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs] with pairs within max_distance
                - edge_attr_pe: [num_pairs, num_buckets] or [num_pairs, 1]
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Compute truncated BFS distances
        edge_index_pe, distances = compute_bfs_distances_truncated(
            edge_index, num_nodes, k=self.max_distance
        )

        if edge_index_pe.size(1) == 0:
            # No pairs found, return empty tensors
            if self.use_one_hot:
                edge_attr_pe = torch.empty((0, self.num_buckets), dtype=torch.float32)
            else:
                edge_attr_pe = torch.empty((0, 1), dtype=torch.long)
            return edge_index_pe, edge_attr_pe

        # Convert distances to buckets
        if self.use_one_hot:
            edge_attr_pe = distances_to_one_hot(
                distances, self.num_buckets, self.max_distance
            )
        else:
            edge_attr_pe = distances_to_embeddings(
                distances, self.num_buckets, self.max_distance
            )
            edge_attr_pe = edge_attr_pe.unsqueeze(1)

        return edge_index_pe, edge_attr_pe

