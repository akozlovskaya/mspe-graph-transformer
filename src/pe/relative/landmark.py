"""Landmark-based SPD approximation implementation."""

from typing import Optional, Tuple
import torch
from torch_geometric.data import Data

from .base import BaseRelativePE
from .utils import (
    compute_shortest_path_distances,
    select_landmarks,
    distances_to_one_hot,
    distances_to_embeddings,
)


class LandmarkSPD(BaseRelativePE):
    """
    Landmark-based Shortest-Path Distance Approximation.

    Approximates pairwise distances using distances to landmark nodes.

    Formula:
        d(i,j) ≈ min_ℓ |d(i,ℓ) - d(j,ℓ)|

    where ℓ are landmark nodes.

    This provides a space-efficient approximation of full SPD, especially useful
    for large graphs where storing all pairwise distances is infeasible.

    Reference:
        "Graph Attention Networks" (Velickovic et al., 2018)
        Various landmark-based graph algorithms
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
        num_landmarks: int = 10,
        landmark_method: str = "random",
        use_one_hot: bool = True,
        approximation_method: str = "min_diff",
        seed: int = 42,
    ):
        """
        Initialize LandmarkSPD.

        Args:
            num_buckets: Number of distance buckets.
            max_distance: Maximum distance to consider.
            normalization: Normalization mode: 'graph', 'pair', or None.
            symmetric: Whether to ensure symmetric PE.
            cache: Whether to cache computed PE.
            num_landmarks: Number of landmark nodes to use.
            landmark_method: Method for selecting landmarks: 'random' or 'degree'.
            use_one_hot: If True, returns one-hot encodings.
            approximation_method: Approximation method:
                - 'min_diff': d(i,j) ≈ min_ℓ |d(i,ℓ) - d(j,ℓ)|
                - 'max_diff': d(i,j) ≈ max_ℓ |d(i,ℓ) - d(j,ℓ)|
                - 'mean_diff': d(i,j) ≈ mean_ℓ |d(i,ℓ) - d(j,ℓ)|
            seed: Random seed for landmark selection.
        """
        super().__init__(
            num_buckets=num_buckets,
            max_distance=max_distance,
            normalization=normalization,
            symmetric=symmetric,
            cache=cache,
        )
        self.num_landmarks = num_landmarks
        self.landmark_method = landmark_method
        self.use_one_hot = use_one_hot
        self.approximation_method = approximation_method
        self.seed = seed

    def compute(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute landmark-based approximate SPD.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs] with all pairs or subset
                - edge_attr_pe: [num_pairs, num_buckets] or [num_pairs, 1]
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Select landmarks
        landmarks = select_landmarks(
            num_nodes,
            self.num_landmarks,
            method=self.landmark_method,
            edge_index=edge_index,
            seed=self.seed,
        )
        landmarks = landmarks.to(edge_index.device)

        # Compute distances from all nodes to landmarks
        # distance_matrix[node, landmark]
        landmark_distances = []
        for landmark in landmarks:
            # Compute shortest paths from landmark to all nodes
            landmark_idx = landmark.item()
            distances_from_landmark = compute_shortest_path_distances(
                edge_index, num_nodes, max_distance=self.max_distance
            )[landmark_idx]
            landmark_distances.append(distances_from_landmark)

        # Stack: [num_nodes, num_landmarks]
        landmark_distances = torch.stack(landmark_distances, dim=1)

        # Create all pairs
        i_indices, j_indices = torch.meshgrid(
            torch.arange(num_nodes, device=edge_index.device),
            torch.arange(num_nodes, device=edge_index.device),
            indexing="ij",
        )
        edge_index_pe = torch.stack([i_indices.flatten(), j_indices.flatten()], dim=0)

        # Get distances from nodes to landmarks
        dist_i = landmark_distances[i_indices.flatten()]  # [num_pairs, num_landmarks]
        dist_j = landmark_distances[j_indices.flatten()]  # [num_pairs, num_landmarks]

        # Compute approximate pairwise distance
        # d(i,j) ≈ min/max/mean_ℓ |d(i,ℓ) - d(j,ℓ)|
        diff = torch.abs(dist_i - dist_j)  # [num_pairs, num_landmarks]

        if self.approximation_method == "min_diff":
            approx_distances = diff.min(dim=1)[0]  # [num_pairs]
        elif self.approximation_method == "max_diff":
            approx_distances = diff.max(dim=1)[0]  # [num_pairs]
        elif self.approximation_method == "mean_diff":
            approx_distances = diff.mean(dim=1)  # [num_pairs]
        else:
            raise ValueError(
                f"Unknown approximation method: {self.approximation_method}"
            )

        # Clip to max_distance
        approx_distances = torch.clamp(approx_distances, 0, self.max_distance)

        # Convert to buckets
        if self.use_one_hot:
            edge_attr_pe = distances_to_one_hot(
                approx_distances, self.num_buckets, self.max_distance
            )
        else:
            edge_attr_pe = distances_to_embeddings(
                approx_distances, self.num_buckets, self.max_distance
            )
            edge_attr_pe = edge_attr_pe.unsqueeze(1)

        return edge_index_pe, edge_attr_pe


class LandmarkSPDSparse(BaseRelativePE):
    """
    Sparse version of LandmarkSPD (only stores pairs with estimated distance <= max_distance).
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
        num_landmarks: int = 10,
        landmark_method: str = "random",
        use_one_hot: bool = True,
        approximation_method: str = "min_diff",
        seed: int = 42,
    ):
        """
        Initialize sparse LandmarkSPD.

        Args are the same as LandmarkSPD, but only pairs within max_distance are stored.
        """
        super().__init__(
            num_buckets=num_buckets,
            max_distance=max_distance,
            normalization=normalization,
            symmetric=symmetric,
            cache=cache,
        )
        self.num_landmarks = num_landmarks
        self.landmark_method = landmark_method
        self.use_one_hot = use_one_hot
        self.approximation_method = approximation_method
        self.seed = seed

    def compute(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse landmark-based approximate SPD (only pairs within max_distance).
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Select landmarks
        landmarks = select_landmarks(
            num_nodes,
            self.num_landmarks,
            method=self.landmark_method,
            edge_index=edge_index,
            seed=self.seed,
        )
        landmarks = landmarks.to(edge_index.device)

        # Compute distances from all nodes to landmarks
        landmark_distances = []
        for landmark in landmarks:
            landmark_idx = landmark.item()
            distances_from_landmark = compute_shortest_path_distances(
                edge_index, num_nodes, max_distance=self.max_distance
            )[landmark_idx]
            landmark_distances.append(distances_from_landmark)

        landmark_distances = torch.stack(landmark_distances, dim=1)

        # Create all pairs
        i_indices, j_indices = torch.meshgrid(
            torch.arange(num_nodes, device=edge_index.device),
            torch.arange(num_nodes, device=edge_index.device),
            indexing="ij",
        )

        # Get distances from nodes to landmarks for all pairs
        # landmark_distances: [num_nodes, num_landmarks]
        # For each pair (i, j), get distances to landmarks
        dist_i = landmark_distances[i_indices.flatten()]  # [num_nodes^2, num_landmarks]
        dist_j = landmark_distances[j_indices.flatten()]  # [num_nodes^2, num_landmarks]

        # Compute approximate pairwise distance
        diff = torch.abs(dist_i - dist_j)  # [num_nodes^2, num_landmarks]

        if self.approximation_method == "min_diff":
            approx_distances = diff.min(dim=1)[0]  # [num_nodes^2]
        elif self.approximation_method == "max_diff":
            approx_distances = diff.max(dim=1)[0]
        elif self.approximation_method == "mean_diff":
            approx_distances = diff.mean(dim=1)
        else:
            raise ValueError(
                f"Unknown approximation method: {self.approximation_method}"
            )

        # Filter pairs within max_distance (excluding self-loops)
        pair_mask = i_indices.flatten() != j_indices.flatten()
        distance_mask = approx_distances <= self.max_distance
        valid_mask = pair_mask & distance_mask

        valid_i = i_indices.flatten()[valid_mask]
        valid_j = j_indices.flatten()[valid_mask]
        valid_distances = approx_distances[valid_mask]

        edge_index_pe = torch.stack([valid_i, valid_j], dim=0)

        # Convert to buckets
        if self.use_one_hot:
            edge_attr_pe = distances_to_one_hot(
                valid_distances, self.num_buckets, self.max_distance
            )
        else:
            edge_attr_pe = distances_to_embeddings(
                valid_distances, self.num_buckets, self.max_distance
            )
            edge_attr_pe = edge_attr_pe.unsqueeze(1)

        return edge_index_pe, edge_attr_pe

