"""Role/Structural features for node positional encodings."""

from typing import Optional, List, Dict
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
import networkx as nx

from .base import BaseNodePE
from .utils import aggregate_multi_scale


class RolePE(BaseNodePE):
    """
    Role-based Positional Encoding.

    Computes structural node features such as:
    - Node degree (log-scaled)
    - Clustering coefficient
    - k-core number
    - Optional: betweenness/closeness centrality (for small graphs)

    Reference:
        Structural features commonly used in graph representation learning.
    """

    def __init__(
        self,
        dim: int,
        features: Optional[List[str]] = None,
        normalization: str = "graph",
        sign_invariant: bool = False,  # Not applicable to role features
        cache: bool = True,
        use_betweenness: bool = False,
        use_closeness: bool = False,
        use_eigenvector_centrality: bool = False,
    ):
        """
        Initialize RolePE.

        Args:
            dim: Output embedding dimension per node.
            features: List of features to compute. Options:
                'degree', 'clustering', 'core', 'betweenness', 'closeness', 'eigenvector'
                If None, uses default: ['degree', 'clustering', 'core'].
            normalization: Normalization mode: 'graph', 'node', or None.
            sign_invariant: Not used for role features (kept for API consistency).
            cache: Whether to cache PE in data.node_pe.
            use_betweenness: Whether to compute betweenness centrality (slow for large graphs).
            use_closeness: Whether to compute closeness centrality (slow for large graphs).
            use_eigenvector_centrality: Whether to compute eigenvector centrality.
        """
        if features is None:
            features = ["degree", "clustering", "core"]

        super().__init__(
            dim=dim,
            scales=None,  # Role features don't use scales
            normalization=normalization,
            sign_invariant=False,
            cache=cache,
        )

        self.features = features
        self.use_betweenness = use_betweenness
        self.use_closeness = use_closeness
        self.use_eigenvector_centrality = use_eigenvector_centrality

    def compute(self, data: Data) -> torch.Tensor:
        """
        Compute role-based positional encodings.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tensor of shape [num_nodes, dim] with positional encodings.
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        device = edge_index.device

        # Convert to NetworkX for some computations
        # Only if needed (clustering, core, centrality measures)
        needs_nx = any(
            f in self.features
            for f in ["clustering", "core", "betweenness", "closeness", "eigenvector"]
        ) or self.use_betweenness or self.use_closeness or self.use_eigenvector_centrality

        if needs_nx:
            try:
                G = to_networkx(data, to_undirected=True, remove_self_loops=True)
            except Exception:
                # Fallback: create simple graph
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                edges = edge_index.cpu().numpy().T
                G.add_edges_from(edges)

        feature_list = []

        # Node degree (log-scaled)
        if "degree" in self.features:
            deg = degree(edge_index[0], num_nodes, dtype=torch.float32)
            # Log-scale to handle large degrees
            deg_log = torch.log(deg + 1.0)
            feature_list.append(deg_log.unsqueeze(1))

        # Clustering coefficient
        if "clustering" in self.features:
            if needs_nx:
                clustering = nx.clustering(G)
                clustering_values = torch.tensor(
                    [clustering.get(i, 0.0) for i in range(num_nodes)],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                # Simple approximation using triangles
                clustering_values = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            feature_list.append(clustering_values.unsqueeze(1))

        # k-core number
        if "core" in self.features:
            if needs_nx:
                core_number = nx.core_number(G)
                core_values = torch.tensor(
                    [core_number.get(i, 0) for i in range(num_nodes)],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                # Simple approximation: use degree as proxy
                deg = degree(edge_index[0], num_nodes, dtype=torch.float32)
                core_values = deg
            feature_list.append(core_values.unsqueeze(1))

        # Betweenness centrality (expensive - only for small graphs)
        if "betweenness" in self.features or self.use_betweenness:
            if needs_nx and num_nodes < 1000:  # Only for small graphs
                betweenness = nx.betweenness_centrality(G)
                betweenness_values = torch.tensor(
                    [betweenness.get(i, 0.0) for i in range(num_nodes)],
                    dtype=torch.float32,
                    device=device,
                )
                feature_list.append(betweenness_values.unsqueeze(1))
            else:
                # Skip or use zero
                betweenness_values = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                feature_list.append(betweenness_values.unsqueeze(1))

        # Closeness centrality (expensive - only for small graphs)
        if "closeness" in self.features or self.use_closeness:
            if needs_nx and num_nodes < 1000:  # Only for small graphs
                closeness = nx.closeness_centrality(G)
                closeness_values = torch.tensor(
                    [closeness.get(i, 0.0) for i in range(num_nodes)],
                    dtype=torch.float32,
                    device=device,
                )
                feature_list.append(closeness_values.unsqueeze(1))
            else:
                # Skip or use zero
                closeness_values = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                feature_list.append(closeness_values.unsqueeze(1))

        # Eigenvector centrality
        if "eigenvector" in self.features or self.use_eigenvector_centrality:
            if needs_nx and num_nodes < 1000:  # Only for small graphs
                try:
                    eigenvector = nx.eigenvector_centrality(G, max_iter=100)
                    eigenvector_values = torch.tensor(
                        [eigenvector.get(i, 0.0) for i in range(num_nodes)],
                        dtype=torch.float32,
                        device=device,
                    )
                    feature_list.append(eigenvector_values.unsqueeze(1))
                except Exception:
                    # Fallback: use zeros
                    eigenvector_values = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                    feature_list.append(eigenvector_values.unsqueeze(1))
            else:
                eigenvector_values = torch.zeros(num_nodes, dtype=torch.float32, device=device)
                feature_list.append(eigenvector_values.unsqueeze(1))

        # Concatenate all features
        if feature_list:
            node_pe = torch.cat(feature_list, dim=1)
        else:
            # Fallback: use degree
            deg = degree(edge_index[0], num_nodes, dtype=torch.float32)
            node_pe = torch.log(deg + 1.0).unsqueeze(1)

        return node_pe


def create_role_pe_with_default_features(
    dim: int = 8,
    normalization: str = "graph",
) -> RolePE:
    """
    Create RolePE with default features (degree, clustering, core).

    Args:
        dim: Output dimension (will be padded/truncated to match feature count).
        normalization: Normalization mode.

    Returns:
        RolePE instance with default features.
    """
    return RolePE(
        dim=dim,
        features=["degree", "clustering", "core"],
        normalization=normalization,
    )

