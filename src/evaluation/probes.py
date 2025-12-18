"""Synthetic probing tasks for long-range evaluation."""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from .distance_metrics import compute_shortest_path_distances


class DistantNodeFeatureProbe:
    """
    Probing task: predict features of distant nodes.

    Given a source node, predict the features of a node at distance k.
    """

    def __init__(
        self,
        distance: int = 5,
        feature_dim: int = 16,
        aggregation: str = "mean",
    ):
        """
        Initialize probe.

        Args:
            distance: Target distance for reconstruction.
            feature_dim: Dimension of node features.
            aggregation: How to aggregate if multiple nodes at distance k.
        """
        self.distance = distance
        self.feature_dim = feature_dim
        self.aggregation = aggregation

    def generate_task(
        self,
        data: Data,
        num_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate probing task for a graph.

        Args:
            data: PyG Data object with node features.
            num_samples: Number of source nodes to sample.

        Returns:
            Tuple of (source_indices, target_features, distances).
        """
        num_nodes = data.num_nodes

        # Compute distances
        distances = compute_shortest_path_distances(
            data.edge_index, num_nodes, self.distance + 1
        )

        sources = []
        targets = []
        target_features = []

        for src in range(num_nodes):
            # Find nodes at target distance
            nodes_at_dist = (distances[src] == self.distance).nonzero(as_tuple=True)[0]

            if len(nodes_at_dist) > 0:
                sources.append(src)

                if self.aggregation == "mean":
                    feat = data.x[nodes_at_dist].mean(dim=0)
                elif self.aggregation == "first":
                    feat = data.x[nodes_at_dist[0]]
                elif self.aggregation == "random":
                    idx = torch.randint(len(nodes_at_dist), (1,)).item()
                    feat = data.x[nodes_at_dist[idx]]
                else:
                    feat = data.x[nodes_at_dist].mean(dim=0)

                target_features.append(feat)
                targets.append(self.distance)

        if num_samples and len(sources) > num_samples:
            indices = torch.randperm(len(sources))[:num_samples]
            sources = [sources[i] for i in indices]
            target_features = [target_features[i] for i in indices]
            targets = [targets[i] for i in indices]

        if not sources:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        return (
            torch.tensor(sources, dtype=torch.long),
            torch.stack(target_features),
            torch.tensor(targets, dtype=torch.long),
        )


class LabelPropagationProbe:
    """
    Probing task: propagate labels over k hops.

    Given initial labels on some nodes, predict labels after k-hop propagation.
    """

    def __init__(
        self,
        num_hops: int = 5,
        num_classes: int = 2,
        label_fraction: float = 0.1,
    ):
        """
        Initialize probe.

        Args:
            num_hops: Number of hops for label propagation.
            num_classes: Number of label classes.
            label_fraction: Fraction of nodes with initial labels.
        """
        self.num_hops = num_hops
        self.num_classes = num_classes
        self.label_fraction = label_fraction

    def generate_task(
        self,
        data: Data,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate label propagation task.

        Args:
            data: PyG Data object.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (initial_labels, propagated_labels, node_distances).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        num_nodes = data.num_nodes
        num_labeled = max(1, int(num_nodes * self.label_fraction))

        # Select labeled nodes
        labeled_indices = torch.randperm(num_nodes)[:num_labeled]

        # Assign random labels
        initial_labels = torch.full((num_nodes,), -1, dtype=torch.long)
        initial_labels[labeled_indices] = torch.randint(
            0, self.num_classes, (num_labeled,)
        )

        # Build adjacency for propagation
        adj_list = [[] for _ in range(num_nodes)]
        edge_index = data.edge_index.cpu()

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
            adj_list[dst].append(src)

        # Propagate labels via majority voting
        current_labels = initial_labels.clone()
        distances_to_label = torch.full((num_nodes,), -1, dtype=torch.long)
        distances_to_label[labeled_indices] = 0

        for hop in range(self.num_hops):
            new_labels = current_labels.clone()
            new_distances = distances_to_label.clone()

            for node in range(num_nodes):
                if current_labels[node] >= 0:
                    continue

                neighbors = adj_list[node]
                if not neighbors:
                    continue

                # Count neighbor labels
                label_counts = torch.zeros(self.num_classes)
                min_dist = float("inf")

                for neighbor in neighbors:
                    if current_labels[neighbor] >= 0:
                        label_counts[current_labels[neighbor]] += 1
                        if distances_to_label[neighbor] >= 0:
                            min_dist = min(min_dist, distances_to_label[neighbor] + 1)

                if label_counts.sum() > 0:
                    new_labels[node] = label_counts.argmax()
                    if min_dist < float("inf"):
                        new_distances[node] = int(min_dist)

            current_labels = new_labels
            distances_to_label = new_distances

        return initial_labels, current_labels, distances_to_label


class PathParityProbe:
    """
    Probing task: compute parity (XOR) of features along a path.

    Tests ability to propagate and aggregate information over long distances.
    """

    def __init__(
        self,
        path_length: int = 10,
    ):
        """
        Initialize probe.

        Args:
            path_length: Length of path for parity computation.
        """
        self.path_length = path_length

    def generate_path_graph(
        self,
        seed: int = 42,
    ) -> Data:
        """
        Generate a path graph with parity labels.

        Args:
            seed: Random seed.

        Returns:
            PyG Data object with path graph and parity labels.
        """
        torch.manual_seed(seed)

        num_nodes = self.path_length

        # Path edges
        src = torch.arange(num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ])

        # Binary features
        x = torch.randint(0, 2, (num_nodes, 1)).float()

        # Parity labels: XOR of all previous nodes
        parity = torch.zeros(num_nodes, dtype=torch.long)
        running_parity = 0
        for i in range(num_nodes):
            running_parity ^= int(x[i].item())
            parity[i] = running_parity

        # Distance from start
        distances = torch.arange(num_nodes, dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            y=parity,
            distances=distances,
        )

    def generate_batch(
        self,
        batch_size: int = 32,
        base_seed: int = 42,
    ) -> List[Data]:
        """
        Generate batch of path graphs.

        Args:
            batch_size: Number of graphs.
            base_seed: Base seed for reproducibility.

        Returns:
            List of Data objects.
        """
        return [
            self.generate_path_graph(seed=base_seed + i)
            for i in range(batch_size)
        ]


class NodeCountingProbe:
    """
    Probing task: count nodes within k hops.

    Tests receptive field and aggregation capabilities.
    """

    def __init__(
        self,
        max_hops: int = 5,
    ):
        """
        Initialize probe.

        Args:
            max_hops: Maximum number of hops to count.
        """
        self.max_hops = max_hops

    def generate_task(
        self,
        data: Data,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate node counting task.

        Args:
            data: PyG Data object.

        Returns:
            Tuple of (counts_per_hop, distances_matrix).
        """
        num_nodes = data.num_nodes

        distances = compute_shortest_path_distances(
            data.edge_index, num_nodes, self.max_hops
        )

        # Count nodes within each hop distance for each source
        counts = torch.zeros(num_nodes, self.max_hops + 1, dtype=torch.float)

        for src in range(num_nodes):
            for hop in range(self.max_hops + 1):
                counts[src, hop] = (
                    (distances[src] >= 0) & (distances[src] <= hop)
                ).sum()

        return counts, distances


class SyntheticLongRangeTask:
    """
    Synthetic task specifically designed to test long-range capabilities.

    Creates graphs where correct prediction requires information
    from nodes at distance k.
    """

    def __init__(
        self,
        graph_size: int = 100,
        signal_distance: int = 10,
        noise_level: float = 0.1,
    ):
        """
        Initialize task.

        Args:
            graph_size: Number of nodes.
            signal_distance: Distance of informative nodes.
            noise_level: Amount of noise in features.
        """
        self.graph_size = graph_size
        self.signal_distance = signal_distance
        self.noise_level = noise_level

    def generate(
        self,
        seed: int = 42,
    ) -> Data:
        """
        Generate synthetic long-range task.

        The target node's label depends on features of nodes at signal_distance.

        Args:
            seed: Random seed.

        Returns:
            PyG Data object with task.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        num_nodes = self.graph_size

        # Create random connected graph (Barabási-Albert style)
        edge_list = []
        for i in range(1, num_nodes):
            # Connect to random earlier node
            target = np.random.randint(0, i)
            edge_list.append((i, target))
            edge_list.append((target, i))

            # Additional edges for connectivity
            if i > 2 and np.random.random() < 0.3:
                target2 = np.random.randint(0, i)
                edge_list.append((i, target2))
                edge_list.append((target2, i))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        # Compute distances
        distances = compute_shortest_path_distances(
            edge_index, num_nodes, self.signal_distance + 5
        )

        # Generate features with signal at distance
        x = torch.randn(num_nodes, 16) * self.noise_level

        # For each node, find nodes at signal_distance and embed signal
        y = torch.zeros(num_nodes, dtype=torch.float)

        for node in range(num_nodes):
            signal_nodes = (distances[node] == self.signal_distance).nonzero(as_tuple=True)[0]

            if len(signal_nodes) > 0:
                # Signal is mean of first feature at signal distance
                signal = x[signal_nodes, 0].mean()
                y[node] = (signal > 0).float()

                # Add corresponding signal to distant nodes
                x[signal_nodes, 0] += y[node] * 2 - 1
            else:
                y[node] = 0.5  # Ambiguous if no nodes at signal_distance

        return Data(
            x=x,
            edge_index=edge_index,
            y=y.unsqueeze(-1),
            distances=distances[0],  # Distances from node 0
            signal_distance=self.signal_distance,
        )


def create_probing_dataset(
    probe_type: str,
    num_samples: int = 100,
    **kwargs,
) -> List[Data]:
    """
    Create dataset of probing tasks.

    Args:
        probe_type: Type of probe: 'parity', 'counting', 'feature', 'propagation'.
        num_samples: Number of samples.
        **kwargs: Probe-specific arguments.

    Returns:
        List of Data objects.
    """
    if probe_type == "parity":
        probe = PathParityProbe(path_length=kwargs.get("path_length", 10))
        return probe.generate_batch(num_samples)

    elif probe_type == "long_range":
        probe = SyntheticLongRangeTask(
            graph_size=kwargs.get("graph_size", 100),
            signal_distance=kwargs.get("signal_distance", 10),
        )
        return [probe.generate(seed=i) for i in range(num_samples)]

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

