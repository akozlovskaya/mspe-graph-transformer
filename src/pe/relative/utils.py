"""Utility functions for relative positional encodings."""

from typing import Optional, Tuple, List
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx
import networkx as nx


def compute_shortest_path_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_distance: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute shortest path distance matrix using NetworkX.

    Args:
        edge_index: Edge indices tensor [2, num_edges].
        num_nodes: Number of nodes.
        max_distance: Maximum distance to compute (None = all pairs).

    Returns:
        Distance matrix of shape [num_nodes, num_nodes].
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)

    # Compute shortest paths
    if max_distance is not None:
        # Use BFS with depth limit for efficiency
        distances = np.full((num_nodes, num_nodes), np.inf)
        for source in range(num_nodes):
            bfs_tree = nx.bfs_tree(G, source, depth_limit=max_distance)
            for target in range(num_nodes):
                try:
                    dist = nx.shortest_path_length(bfs_tree, source, target)
                    if dist <= max_distance:
                        distances[source, target] = dist
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        distances = np.where(distances == np.inf, max_distance + 1, distances)
    else:
        distances = nx.floyd_warshall_numpy(G)

    return torch.from_numpy(distances).float()


def bucketize_distances(
    distances: torch.Tensor,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """
    Convert distances to bucket indices.

    Args:
        distances: Distance matrix or values [..., num_nodes] or [num_pairs].
        num_buckets: Number of buckets.
        max_distance: Maximum distance (distances > max_distance go to last bucket).

    Returns:
        Bucket indices tensor of same shape as distances.
    """
    # Clip distances to [0, max_distance]
    clipped = torch.clamp(distances, 0, max_distance)

    # Map to buckets [0, num_buckets-1]
    # Linear mapping: distance -> bucket
    bucket_size = max_distance / (num_buckets - 1)
    buckets = (clipped / bucket_size).long()
    buckets = torch.clamp(buckets, 0, num_buckets - 1)

    return buckets


def distances_to_one_hot(
    distances: torch.Tensor,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """
    Convert distances to one-hot bucket encodings.

    Args:
        distances: Distance tensor [num_pairs] or [num_nodes, num_nodes].
        num_buckets: Number of buckets.
        max_distance: Maximum distance.

    Returns:
        One-hot encodings of shape [..., num_buckets].
    """
    buckets = bucketize_distances(distances, num_buckets, max_distance)
    one_hot = torch.nn.functional.one_hot(
        buckets, num_classes=num_buckets
    ).float()

    return one_hot


def distances_to_embeddings(
    distances: torch.Tensor,
    num_buckets: int,
    max_distance: int,
    embedding_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert distances to learnable embedding format (bucket indices).

    Args:
        distances: Distance tensor [num_pairs].
        num_buckets: Number of buckets.
        max_distance: Maximum distance.
        embedding_dim: If provided, returns embedding_dim for lookup table size.

    Returns:
        Bucket indices of shape [num_pairs] (for embedding lookup).
    """
    buckets = bucketize_distances(distances, num_buckets, max_distance)
    return buckets.long()


def compute_bfs_distances_truncated(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute truncated BFS distances (only pairs within k hops).

    Args:
        edge_index: Edge indices tensor [2, num_edges].
        num_nodes: Number of nodes.
        k: Maximum BFS depth.

    Returns:
        Tuple of (edge_index_pairs, distances):
            - edge_index_pairs: [2, num_pairs] with pairs within k hops
            - distances: [num_pairs] with distances
    """
    # Convert to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().numpy().T
    G.add_edges_from(edges)

    # Collect all pairs within k hops
    pair_list = []
    dist_list = []

    for source in range(num_nodes):
        # BFS tree with depth limit
        bfs_tree = nx.bfs_tree(G, source, depth_limit=k)
        for target in range(num_nodes):
            if target != source:
                try:
                    dist = nx.shortest_path_length(bfs_tree, source, target)
                    if dist <= k:
                        pair_list.append([source, target])
                        dist_list.append(dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

    if pair_list:
        edge_index_pairs = torch.tensor(pair_list, dtype=torch.long).t()
        distances = torch.tensor(dist_list, dtype=torch.float32)
    else:
        edge_index_pairs = torch.empty((2, 0), dtype=torch.long)
        distances = torch.empty((0,), dtype=torch.float32)

    return edge_index_pairs, distances


def get_top_k_eigenpairs(
    laplacian: torch.Tensor,
    k: int,
    exclude_zero: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k eigenpairs of Laplacian.

    Args:
        laplacian: Dense Laplacian matrix [num_nodes, num_nodes].
        k: Number of eigenpairs.
        exclude_zero: Whether to exclude zero eigenvalue.

    Returns:
        Tuple of (eigenvalues, eigenvectors):
            - eigenvalues: [k]
            - eigenvectors: [num_nodes, k]
    """
    from src.pe.node.utils import compute_eigenvectors

    return compute_eigenvectors(laplacian, k=k, exclude_zero=exclude_zero)


def build_attention_bias(
    edge_index_pe: torch.Tensor,
    edge_attr_pe: torch.Tensor,
    num_nodes: int,
    num_heads: Optional[int] = None,
    mode: str = "dense",
    gating: bool = False,
    dropout: float = 0.0,
    learnable_gating: bool = False,
) -> torch.Tensor:
    """
    Build attention bias from relative PE.

    Args:
        edge_index_pe: Pair indices [2, num_pairs].
        edge_attr_pe: PE values [num_pairs, num_buckets].
        num_nodes: Number of nodes.
        num_heads: Number of attention heads (if None, uses 1).
        mode: 'dense' or 'sparse'.
        gating: Whether to use gating mechanism.
        dropout: Dropout rate on bias.
        learnable_gating: Whether gating is learnable (requires returning gate params).

    Returns:
        Attention bias tensor:
            - Dense: [num_heads, num_nodes, num_nodes] or [num_nodes, num_nodes]
            - Sparse: SparseTensor
    """
    if num_heads is None:
        num_heads = 1

    # Aggregate PE values to single scalar per pair
    # Simple approach: sum or mean across buckets
    if edge_attr_pe.dim() == 2:
        if edge_attr_pe.size(1) == 1:
            pair_values = edge_attr_pe.squeeze(1)  # [num_pairs]
        else:
            # Sum or mean across buckets
            pair_values = edge_attr_pe.sum(dim=1)  # [num_pairs]
    else:
        pair_values = edge_attr_pe  # [num_pairs]

    if mode == "dense":
        # Create dense bias matrix
        bias = torch.zeros(num_nodes, num_nodes, device=edge_index_pe.device)

        # Fill in PE values
        bias[edge_index_pe[0], edge_index_pe[1]] = pair_values

        # Apply gating if requested
        if gating:
            # Learnable or fixed gate
            if learnable_gating:
                # Would need to return gate parameter for training
                # For now, use fixed sigmoid gate
                bias = torch.sigmoid(bias) * bias
            else:
                bias = torch.sigmoid(bias) * bias

        # Expand to num_heads if needed
        if num_heads > 1:
            bias = bias.unsqueeze(0).repeat(num_heads, 1, 1)

        # Apply dropout
        if dropout > 0 and bias.requires_grad:
            bias = torch.nn.functional.dropout(bias, p=dropout, training=True)

        return bias

    elif mode == "sparse":
        # Return sparse representation
        from torch_sparse import SparseTensor

        row, col = edge_index_pe
        values = pair_values

        # Apply gating
        if gating:
            values = torch.sigmoid(values) * values

        sparse_bias = SparseTensor(
            row=row,
            col=col,
            value=values,
            sparse_sizes=(num_nodes, num_nodes),
        )

        return sparse_bias

    else:
        raise ValueError(f"Unknown mode: {mode}")


def select_landmarks(
    num_nodes: int,
    num_landmarks: int,
    method: str = "random",
    edge_index: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Select landmark nodes.

    Args:
        num_nodes: Total number of nodes.
        num_landmarks: Number of landmarks to select.
        method: Selection method: 'random', 'degree', 'kmeans'.
        edge_index: Edge indices (needed for degree-based selection).
        seed: Random seed.

    Returns:
        Tensor of landmark node indices [num_landmarks].
    """
    num_landmarks = min(num_landmarks, num_nodes)

    if method == "random":
        torch.manual_seed(seed)
        landmarks = torch.randperm(num_nodes)[:num_landmarks]
        return landmarks

    elif method == "degree":
        if edge_index is None:
            # Fallback to random
            torch.manual_seed(seed)
            return torch.randperm(num_nodes)[:num_landmarks]

        from torch_geometric.utils import degree

        deg = degree(edge_index[0], num_nodes, dtype=torch.float32)
        # Select top-degree nodes
        _, top_indices = torch.topk(deg, num_landmarks)
        return top_indices

    else:
        raise ValueError(f"Unknown landmark selection method: {method}")

