"""Effective Resistance (approximate) implementation."""

from typing import Optional, Tuple
import torch
from torch_geometric.data import Data

from .base import BaseRelativePE
from .utils import get_top_k_eigenpairs


class EffectiveResistancePE(BaseRelativePE):
    """
    Effective Resistance Positional Encoding (Low-rank Approximation).

    Computes approximate effective resistance between node pairs using low-rank
    approximation of the Laplacian pseudoinverse.

    Formula:
        R(i,j) = L^+_{ii} + L^+_{jj} - 2*L^+_{ij}

    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.

    Low-rank approximation:
        L^+ ≈ Σ_{k=1}^K (1/λ_k) * φ_k * φ_k^T

    Reference:
        "Graph Attention Networks" (Velickovic et al., 2018)
        "Understanding Attention and Generalization in Graph Neural Networks" (Knyazev et al., 2019)
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,  # Not used for resistance, but kept for API consistency
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
        k_eigenvectors: int = 50,
        use_sparse: bool = True,
        max_pairs: Optional[int] = None,
    ):
        """
        Initialize EffectiveResistancePE.

        Args:
            num_buckets: Number of output channels (always 1 for resistance).
            max_distance: Not used, kept for API consistency.
            normalization: Normalization mode: 'graph', 'pair', or None.
            symmetric: Whether to ensure symmetric PE (always True for resistance).
            cache: Whether to cache computed PE.
            k_eigenvectors: Number of eigenvectors to use for low-rank approximation.
            use_sparse: Whether to use sparse computation (only store non-zero pairs).
            max_pairs: Maximum number of pairs to store (for memory efficiency).
        """
        super().__init__(
            num_buckets=1,  # Resistance is scalar per pair
            max_distance=max_distance,
            normalization=normalization,
            symmetric=True,  # Resistance is always symmetric
            cache=cache,
        )
        self.k_eigenvectors = k_eigenvectors
        self.use_sparse = use_sparse
        self.max_pairs = max_pairs

    def compute(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute approximate effective resistance.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs]
                - edge_attr_pe: [num_pairs, 1] with resistance values
        """
        from src.pe.node.utils import get_normalized_laplacian

        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Limit k to available nodes
        k = min(self.k_eigenvectors, num_nodes)

        # Compute normalized Laplacian
        laplacian = get_normalized_laplacian(edge_index, num_nodes)

        # Get top-k eigenpairs (excluding zero eigenvalue)
        eigenvalues, eigenvectors = get_top_k_eigenpairs(
            laplacian, k=k, exclude_zero=True
        )

        # Avoid division by zero
        epsilon = 1e-8
        inv_eigenvalues = 1.0 / (eigenvalues + epsilon)  # [k]

        # Compute low-rank approximation of L^+
        # L^+ ≈ Σ_k (1/λ_k) * φ_k * φ_k^T
        # For each pair (i,j): L^+_{ij} = Σ_k (1/λ_k) * φ_k(i) * φ_k(j)

        # Compute L^+ matrix using low-rank form
        weighted_eigenvec = eigenvectors * torch.sqrt(inv_eigenvalues).unsqueeze(0)
        # L^+ = (weighted_eigenvec) @ (weighted_eigenvec)^T
        L_plus = weighted_eigenvec @ weighted_eigenvec.t()  # [num_nodes, num_nodes]

        # Compute effective resistance R(i,j) = L^+_{ii} + L^+_{jj} - 2*L^+_{ij}
        diag_L_plus = torch.diag(L_plus)  # [num_nodes]

        if self.use_sparse and self.max_pairs is not None:
            # Compute resistance for all pairs, then keep top max_pairs
            i_indices, j_indices = torch.meshgrid(
                torch.arange(num_nodes, device=edge_index.device),
                torch.arange(num_nodes, device=edge_index.device),
                indexing="ij",
            )

            # Compute resistance values
            resistance = (
                diag_L_plus[i_indices]
                + diag_L_plus[j_indices]
                - 2 * L_plus[i_indices, j_indices]
            )

            # Filter out self-loops and select top pairs
            mask = i_indices != j_indices
            valid_i = i_indices[mask]
            valid_j = j_indices[mask]
            valid_resistance = resistance[mask]

            # Keep top max_pairs by magnitude
            if len(valid_resistance) > self.max_pairs:
                _, top_indices = torch.topk(
                    torch.abs(valid_resistance), self.max_pairs
                )
                edge_index_pe = torch.stack(
                    [valid_i[top_indices], valid_j[top_indices]], dim=0
                )
                edge_attr_pe = valid_resistance[top_indices].unsqueeze(1)
            else:
                edge_index_pe = torch.stack([valid_i, valid_j], dim=0)
                edge_attr_pe = valid_resistance.unsqueeze(1)
        else:
            # Store all pairs (excluding self-loops)
            i_indices, j_indices = torch.meshgrid(
                torch.arange(num_nodes, device=edge_index.device),
                torch.arange(num_nodes, device=edge_index.device),
                indexing="ij",
            )

            # Compute resistance values
            resistance = (
                diag_L_plus[i_indices]
                + diag_L_plus[j_indices]
                - 2 * L_plus[i_indices, j_indices]
            )

            # Filter out self-loops
            mask = i_indices != j_indices
            edge_index_pe = torch.stack([i_indices[mask], j_indices[mask]], dim=0)
            edge_attr_pe = resistance[mask].unsqueeze(1)

        return edge_index_pe, edge_attr_pe

