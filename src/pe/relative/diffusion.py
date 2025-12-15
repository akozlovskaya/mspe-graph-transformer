"""Diffusion / Heat Kernel pairwise encoding implementation."""

from typing import Optional, List, Tuple
import torch
from torch_geometric.data import Data
import numpy as np

from .base import BaseRelativePE
from .utils import get_top_k_eigenpairs


def create_log_spaced_times(
    min_time: float, max_time: float, num_scales: int
) -> List[float]:
    """Create logarithmically spaced diffusion times."""
    return np.logspace(np.log10(min_time), np.log10(max_time), num_scales).tolist()


class DiffusionPE(BaseRelativePE):
    """
    Diffusion / Heat Kernel Pairwise Encoding.

    Computes pairwise diffusion kernel values using heat diffusion on the graph.

    Formula:
        K_t(i,j) = Σ_k exp(-λ_k * t) * φ_k(i) * φ_k(j)

    where:
        λ_k, φ_k are eigenvalues and eigenvectors of normalized Laplacian
        t is diffusion time

    Reference:
        "A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion"
        (Sun et al., 2009)
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,  # Not used for diffusion, but kept for API consistency
        normalization: str = "graph",
        symmetric: bool = True,
        cache: bool = True,
        scales: Optional[List[float]] = None,
        k_eigenvectors: int = 50,
        cutoff_threshold: Optional[float] = None,
        use_low_rank: bool = True,
        min_time: float = 0.01,
        max_time: float = 10.0,
    ):
        """
        Initialize DiffusionPE.

        Args:
            num_buckets: Number of output channels (one per scale).
            max_distance: Not used, kept for API consistency.
            normalization: Normalization mode: 'graph', 'pair', or None.
            symmetric: Whether to ensure symmetric PE.
            cache: Whether to cache computed PE.
            scales: List of diffusion times. If None, generates log-spaced scales.
            k_eigenvectors: Number of eigenvectors to use.
            cutoff_threshold: Threshold for cutting off small values (for sparsity).
            use_low_rank: Whether to use low-rank approximation.
            min_time: Minimum diffusion time when generating default scales.
            max_time: Maximum diffusion time when generating default scales.
        """
        if scales is None:
            scales = create_log_spaced_times(min_time, max_time, num_buckets)

        super().__init__(
            num_buckets=len(scales),
            max_distance=max_distance,
            normalization=normalization,
            symmetric=symmetric,
            cache=cache,
        )
        self.scales = scales
        self.k_eigenvectors = k_eigenvectors
        self.cutoff_threshold = cutoff_threshold
        self.use_low_rank = use_low_rank

    def compute(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion pairwise encodings.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs] with all pairs or filtered pairs
                - edge_attr_pe: [num_pairs, num_buckets] with diffusion values per scale
        """
        from src.pe.node.utils import get_normalized_laplacian

        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Limit k to available nodes
        k = min(self.k_eigenvectors, num_nodes)

        # Compute normalized Laplacian
        laplacian = get_normalized_laplacian(edge_index, num_nodes)

        # Get top-k eigenpairs
        eigenvalues, eigenvectors = get_top_k_eigenpairs(
            laplacian, k=k, exclude_zero=True
        )

        # Compute diffusion kernel for each scale
        # For all pairs (i,j): K_t(i,j) = Σ_k exp(-λ_k * t) * φ_k(i) * φ_k(j)
        # This can be written as: (eigenvectors @ diag(exp(-λ*t)) @ eigenvectors^T)[i,j]

        # Create all pairs
        i_indices, j_indices = torch.meshgrid(
            torch.arange(num_nodes, device=edge_index.device),
            torch.arange(num_nodes, device=edge_index.device),
            indexing="ij",
        )
        edge_index_pe = torch.stack([i_indices.flatten(), j_indices.flatten()], dim=0)

        # Compute diffusion values for each scale
        scale_values = []
        for t in self.scales:
            # Compute exp(-λ_k * t) weights
            exp_weights = torch.exp(-eigenvalues * t)  # [k]

            # Compute K_t = eigenvectors @ diag(exp_weights) @ eigenvectors^T
            # More efficient: compute row-wise
            weighted_eigenvec = eigenvectors * exp_weights.unsqueeze(0)  # [num_nodes, k]
            diffusion_kernel = weighted_eigenvec @ eigenvectors.t()  # [num_nodes, num_nodes]

            # Extract values for all pairs
            pair_values = diffusion_kernel[i_indices.flatten(), j_indices.flatten()]

            # Apply cutoff if requested
            if self.cutoff_threshold is not None:
                # Only keep pairs above threshold
                valid_mask = torch.abs(pair_values) > self.cutoff_threshold
                # For now, keep all pairs but set small values to 0
                pair_values = torch.where(
                    torch.abs(pair_values) > self.cutoff_threshold,
                    pair_values,
                    torch.zeros_like(pair_values),
                )

            scale_values.append(pair_values.unsqueeze(1))

        # Stack scales: [num_pairs, num_scales]
        edge_attr_pe = torch.cat(scale_values, dim=1)

        # Apply cutoff: filter pairs if requested
        if self.cutoff_threshold is not None:
            # Keep pairs where at least one scale has value above threshold
            max_per_pair = torch.abs(edge_attr_pe).max(dim=1)[0]
            valid_mask = max_per_pair > self.cutoff_threshold

            edge_index_pe = edge_index_pe[:, valid_mask]
            edge_attr_pe = edge_attr_pe[valid_mask]

        return edge_index_pe, edge_attr_pe

