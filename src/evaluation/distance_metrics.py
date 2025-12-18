"""Distance computation utilities for long-range evaluation."""

from typing import Optional, Dict, Tuple, List
from collections import deque

import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix


def compute_shortest_path_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_distance: int = 20,
    directed: bool = False,
) -> torch.Tensor:
    """
    Compute shortest-path distances using BFS.

    Args:
        edge_index: Edge index [2, num_edges].
        num_nodes: Number of nodes.
        max_distance: Maximum distance to compute (truncated).
        directed: Whether graph is directed.

    Returns:
        Distance matrix [num_nodes, num_nodes] with -1 for unreachable pairs.
    """
    # Build adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    edge_index_np = edge_index.cpu().numpy()

    for i in range(edge_index.size(1)):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_list[src].append(dst)
        if not directed:
            adj_list[dst].append(src)

    # BFS from each node
    distances = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)

    for source in range(num_nodes):
        dist = [-1] * num_nodes
        dist[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            if dist[node] >= max_distance:
                continue

            for neighbor in adj_list[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        distances[source] = torch.tensor(dist, dtype=torch.long)

    return distances


def compute_shortest_path_distances_sparse(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_distance: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shortest-path distances in sparse format (only pairs within max_distance).

    Args:
        edge_index: Edge index [2, num_edges].
        num_nodes: Number of nodes.
        max_distance: Maximum distance to store.

    Returns:
        Tuple of:
            - pair_indices: [2, num_pairs] source-target pairs
            - pair_distances: [num_pairs] distances
    """
    adj_list = [[] for _ in range(num_nodes)]
    edge_index_np = edge_index.cpu().numpy()

    for i in range(edge_index.size(1)):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    sources = []
    targets = []
    distances = []

    for source in range(num_nodes):
        dist = [-1] * num_nodes
        dist[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            current_dist = dist[node]

            if current_dist > 0:  # Don't store self-loops
                sources.append(source)
                targets.append(node)
                distances.append(current_dist)

            if current_dist >= max_distance:
                continue

            for neighbor in adj_list[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = current_dist + 1
                    queue.append(neighbor)

    pair_indices = torch.tensor([sources, targets], dtype=torch.long)
    pair_distances = torch.tensor(distances, dtype=torch.long)

    return pair_indices, pair_distances


def compute_landmark_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_landmarks: int = 10,
    max_distance: int = 50,
    selection: str = "degree",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distances to landmark nodes for approximate distance computation.

    Args:
        edge_index: Edge index [2, num_edges].
        num_nodes: Number of nodes.
        num_landmarks: Number of landmark nodes.
        max_distance: Maximum distance to compute.
        selection: Landmark selection method: 'degree', 'random', 'central'.

    Returns:
        Tuple of:
            - landmark_indices: [num_landmarks] indices of landmarks
            - landmark_distances: [num_nodes, num_landmarks] distances to landmarks
    """
    num_landmarks = min(num_landmarks, num_nodes)

    # Select landmarks
    if selection == "degree":
        # Select high-degree nodes
        edge_index_np = edge_index.cpu().numpy()
        degrees = np.bincount(edge_index_np[0], minlength=num_nodes)
        degrees += np.bincount(edge_index_np[1], minlength=num_nodes)
        landmark_indices = torch.tensor(
            np.argsort(degrees)[-num_landmarks:], dtype=torch.long
        )
    elif selection == "random":
        perm = torch.randperm(num_nodes)
        landmark_indices = perm[:num_landmarks]
    else:
        # Central nodes (first num_landmarks)
        landmark_indices = torch.arange(num_landmarks)

    # Compute distances from landmarks
    adj_list = [[] for _ in range(num_nodes)]
    edge_index_np = edge_index.cpu().numpy()

    for i in range(edge_index.size(1)):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    landmark_distances = torch.full(
        (num_nodes, num_landmarks), max_distance + 1, dtype=torch.long
    )

    for l_idx, landmark in enumerate(landmark_indices.tolist()):
        dist = [-1] * num_nodes
        dist[landmark] = 0
        queue = deque([landmark])

        while queue:
            node = queue.popleft()
            if dist[node] >= max_distance:
                continue

            for neighbor in adj_list[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        for i in range(num_nodes):
            if dist[i] >= 0:
                landmark_distances[i, l_idx] = dist[i]

    return landmark_indices, landmark_distances


def approximate_pairwise_distance(
    landmark_distances: torch.Tensor,
    source: int,
    target: int,
) -> int:
    """
    Approximate distance between two nodes using landmark distances.

    Uses triangle inequality: d(s,t) ≈ min_l |d(s,l) + d(t,l)|

    Args:
        landmark_distances: [num_nodes, num_landmarks] distances to landmarks.
        source: Source node index.
        target: Target node index.

    Returns:
        Approximate distance.
    """
    # Upper bound via triangle inequality
    upper_bound = (
        landmark_distances[source] + landmark_distances[target]
    ).min().item()

    return upper_bound


def add_distance_info_to_data(
    data: Data,
    max_distance: int = 20,
    sparse: bool = True,
    use_landmarks: bool = False,
    num_landmarks: int = 10,
) -> Data:
    """
    Add distance information to a PyG Data object.

    Args:
        data: PyG Data object.
        max_distance: Maximum distance to compute.
        sparse: Whether to use sparse storage.
        use_landmarks: Whether to use landmark-based approximation.
        num_landmarks: Number of landmarks if using approximation.

    Returns:
        Data object with distance information added.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if use_landmarks:
        landmark_indices, landmark_distances = compute_landmark_distances(
            edge_index, num_nodes, num_landmarks, max_distance
        )
        data.landmark_indices = landmark_indices
        data.landmark_distances = landmark_distances
    elif sparse:
        pair_indices, pair_distances = compute_shortest_path_distances_sparse(
            edge_index, num_nodes, max_distance
        )
        data.distance_pairs = pair_indices
        data.distance_values = pair_distances
    else:
        distances = compute_shortest_path_distances(
            edge_index, num_nodes, max_distance
        )
        data.distance_matrix = distances

    data.max_distance_computed = max_distance

    return data


def get_node_pair_distances(
    data: Data,
    source_nodes: torch.Tensor,
    target_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Get distances between specified node pairs from cached data.

    Args:
        data: PyG Data object with distance info.
        source_nodes: Source node indices.
        target_nodes: Target node indices.

    Returns:
        Distances tensor.
    """
    if hasattr(data, "distance_matrix"):
        # Dense storage
        return data.distance_matrix[source_nodes, target_nodes]
    elif hasattr(data, "distance_pairs"):
        # Sparse storage - need to look up
        pair_idx = data.distance_pairs
        pair_val = data.distance_values

        distances = torch.full((len(source_nodes),), -1, dtype=torch.long)

        for i, (src, tgt) in enumerate(zip(source_nodes.tolist(), target_nodes.tolist())):
            # Find pair in sparse storage
            mask = (pair_idx[0] == src) & (pair_idx[1] == tgt)
            if mask.any():
                distances[i] = pair_val[mask][0]

        return distances
    elif hasattr(data, "landmark_distances"):
        # Approximate using landmarks
        distances = torch.zeros(len(source_nodes), dtype=torch.long)
        for i, (src, tgt) in enumerate(zip(source_nodes.tolist(), target_nodes.tolist())):
            distances[i] = approximate_pairwise_distance(
                data.landmark_distances, src, tgt
            )
        return distances
    else:
        raise ValueError("Data object has no distance information")


def compute_distance_histogram(
    data: Data,
    max_distance: int = 20,
) -> torch.Tensor:
    """
    Compute histogram of pairwise distances in graph.

    Args:
        data: PyG Data object.
        max_distance: Maximum distance for histogram.

    Returns:
        Histogram tensor [max_distance + 2] (including unreachable).
    """
    if hasattr(data, "distance_matrix"):
        distances = data.distance_matrix.view(-1)
    elif hasattr(data, "distance_values"):
        distances = data.distance_values
    else:
        # Compute on the fly
        distances = compute_shortest_path_distances(
            data.edge_index, data.num_nodes, max_distance
        ).view(-1)

    histogram = torch.zeros(max_distance + 2, dtype=torch.long)

    for d in range(max_distance + 1):
        histogram[d] = (distances == d).sum()

    # Unreachable pairs
    histogram[max_distance + 1] = (distances == -1).sum()

    return histogram


def get_distance_to_label_nodes(
    data: Data,
    label_node_mask: torch.Tensor,
    max_distance: int = 20,
) -> torch.Tensor:
    """
    Compute minimum distance from each node to any labeled/target node.

    Useful for measuring how far information needs to travel.

    Args:
        data: PyG Data object.
        label_node_mask: Boolean mask indicating labeled nodes.
        max_distance: Maximum distance.

    Returns:
        Tensor [num_nodes] with minimum distance to any labeled node.
    """
    label_nodes = label_node_mask.nonzero(as_tuple=True)[0]
    num_nodes = data.num_nodes

    # BFS from all label nodes simultaneously
    adj_list = [[] for _ in range(num_nodes)]
    edge_index_np = data.edge_index.cpu().numpy()

    for i in range(data.edge_index.size(1)):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    distances = torch.full((num_nodes,), -1, dtype=torch.long)
    queue = deque()

    for node in label_nodes.tolist():
        distances[node] = 0
        queue.append(node)

    while queue:
        node = queue.popleft()
        if distances[node] >= max_distance:
            continue

        for neighbor in adj_list[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances

