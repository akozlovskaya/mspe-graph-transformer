"""Utility functions for node positional encodings."""

from typing import List, Optional, Tuple
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import networkx as nx
from torch_geometric.utils import to_networkx


def create_log_spaced_scales(min_scale: float, max_scale: float, num_scales: int) -> List[float]:
    """
    Create logarithmically spaced scales.

    Args:
        min_scale: Minimum scale value.
        max_scale: Maximum scale value.
        num_scales: Number of scales to generate.

    Returns:
        List of log-spaced scale values.
    """
    return np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales).tolist()


def create_linear_spaced_scales(min_scale: float, max_scale: float, num_scales: int) -> List[float]:
    """
    Create linearly spaced scales.

    Args:
        min_scale: Minimum scale value.
        max_scale: Maximum scale value.
        num_scales: Number of scales to generate.

    Returns:
        List of linearly spaced scale values.
    """
    return np.linspace(min_scale, max_scale, num_scales).tolist()


def aggregate_multi_scale(
    scale_embeddings: List[torch.Tensor], method: str = "concat"
) -> torch.Tensor:
    """
    Aggregate multi-scale embeddings.

    Args:
        scale_embeddings: List of tensors, each of shape [num_nodes, scale_dim].
        method: Aggregation method: 'concat' or 'mean'.

    Returns:
        Aggregated tensor of shape [num_nodes, aggregated_dim].
    """
    if method == "concat":
        return torch.cat(scale_embeddings, dim=1)
    elif method == "mean":
        return torch.stack(scale_embeddings, dim=0).mean(dim=0)
    elif method == "sum":
        return torch.stack(scale_embeddings, dim=0).sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def apply_sign_invariance(
    eigenvectors: torch.Tensor, method: str = "abs"
) -> torch.Tensor:
    """
    Apply sign-invariant processing to eigenvectors.

    Args:
        eigenvectors: Tensor of shape [num_nodes, num_eigenvectors].
        method: Sign-invariance method:
            - 'abs': Take absolute value
            - 'flip': Concatenate [φ, -φ]
            - 'square': Square the values

    Returns:
        Sign-invariant tensor.
    """
    if method == "abs":
        return torch.abs(eigenvectors)
    elif method == "flip":
        return torch.cat([eigenvectors, -eigenvectors], dim=1)
    elif method == "square":
        return eigenvectors ** 2
    else:
        raise ValueError(f"Unknown sign-invariance method: {method}")


def get_normalized_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute normalized graph Laplacian matrix.

    Args:
        edge_index: Edge indices tensor of shape [2, num_edges].
        num_nodes: Number of nodes.
        edge_weight: Optional edge weights.

    Returns:
        Dense normalized Laplacian matrix of shape [num_nodes, num_nodes].
    """
    from torch_geometric.utils import degree, add_self_loops

    # Add self-loops if not present
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=num_nodes
    )

    # Compute degree
    if edge_weight is None:
        deg = degree(edge_index[0], num_nodes, dtype=torch.float32)
    else:
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], edge_weight)

    # Avoid division by zero
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    # Build adjacency matrix
    adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze(0)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg_mat = torch.diag(deg_inv_sqrt)
    laplacian = torch.eye(num_nodes, device=edge_index.device) - deg_mat @ adj @ deg_mat

    return laplacian


def compute_eigenvectors(
    laplacian: torch.Tensor,
    k: int,
    exclude_zero: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k eigenvectors and eigenvalues of Laplacian.

    Args:
        laplacian: Dense Laplacian matrix of shape [num_nodes, num_nodes].
        k: Number of eigenvectors to compute.
        exclude_zero: Whether to exclude zero eigenvalue.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
        eigenvalues: Tensor of shape [k]
        eigenvectors: Tensor of shape [num_nodes, k]
    """
    # Compute eigendecomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    except RuntimeError:
        # Fallback to numpy if torch fails
        laplacian_np = laplacian.cpu().numpy()
        eigenvalues_np, eigenvectors_np = np.linalg.eigh(laplacian_np)
        eigenvalues = torch.from_numpy(eigenvalues_np).to(laplacian.device)
        eigenvectors = torch.from_numpy(eigenvectors_np).to(laplacian.device)

    # Sort by eigenvalue (ascending)
    sort_idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Limit k to available eigenvalues
    k = min(k, len(eigenvalues))

    # Exclude zero eigenvalue if requested
    if exclude_zero:
        # Find first non-zero eigenvalue
        nonzero_idx = (eigenvalues > 1e-8).nonzero(as_tuple=True)[0]
        if len(nonzero_idx) > 0:
            start_idx = nonzero_idx[0].item()
            # Take up to k non-zero eigenvalues
            available_k = min(k, len(eigenvalues) - start_idx)
            eigenvalues = eigenvalues[start_idx : start_idx + available_k]
            eigenvectors = eigenvectors[:, start_idx : start_idx + available_k]
        else:
            # All eigenvalues are zero (disconnected graph) - take first k
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
    else:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    return eigenvalues, eigenvectors


def get_transition_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute random walk transition matrix P = D^{-1} A.

    Args:
        edge_index: Edge indices tensor of shape [2, num_edges].
        num_nodes: Number of nodes.
        edge_weight: Optional edge weights.

    Returns:
        Dense transition matrix of shape [num_nodes, num_nodes].
    """
    from torch_geometric.utils import degree, add_self_loops

    # Add self-loops
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=num_nodes
    )

    # Compute degree
    if edge_weight is None:
        deg = degree(edge_index[0], num_nodes, dtype=torch.float32)
    else:
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], edge_weight)

    # Avoid division by zero
    deg_inv = deg.pow(-1.0)
    deg_inv[deg_inv == float("inf")] = 0

    # Build adjacency matrix
    adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze(0)

    # Transition matrix: P = D^{-1} A
    deg_mat = torch.diag(deg_inv)
    transition = deg_mat @ adj

    return transition


def compute_power_matrix(
    matrix: torch.Tensor,
    power: int,
) -> torch.Tensor:
    """
    Compute matrix power efficiently.

    Args:
        matrix: Dense matrix of shape [n, n].
        power: Power to raise matrix to.

    Returns:
        Matrix raised to power.
    """
    if power == 0:
        return torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    elif power == 1:
        return matrix
    else:
        result = matrix
        for _ in range(power - 1):
            result = result @ matrix
        return result

