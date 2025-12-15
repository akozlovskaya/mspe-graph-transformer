"""Heat Kernel Signatures (HKS) implementation."""

from typing import Optional, List
import torch
from torch_geometric.data import Data

from .base import BaseNodePE
from .utils import (
    get_normalized_laplacian,
    compute_eigenvectors,
    apply_sign_invariance,
    aggregate_multi_scale,
    create_log_spaced_scales,
)


class HKS(BaseNodePE):
    """
    Heat Kernel Signatures.

    Computes node encodings based on heat diffusion on the graph.

    Formula:
        HKS_t(i) = sum_k exp(-λ_k * t) * φ_k(i)^2

    where:
        λ_k are eigenvalues of Laplacian
        φ_k are eigenvectors of Laplacian
        t is diffusion time

    Reference:
        "A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion"
        (Sun et al., 2009)
    """

    def __init__(
        self,
        dim: int,
        scales: Optional[List[float]] = None,
        normalization: str = "graph",
        sign_invariant: bool = True,
        sign_invariance_method: str = "square",
        cache: bool = True,
        k_eigenvectors: int = 50,
        exclude_zero_eigenvalue: bool = True,
        min_time: float = 0.01,
        max_time: float = 10.0,
        learnable_scales: bool = False,
    ):
        """
        Initialize HKS.

        Args:
            dim: Output embedding dimension per node.
            scales: List of diffusion times (t values). If None, generates log-spaced scales.
            normalization: Normalization mode: 'graph', 'node', or None.
            sign_invariant: Whether to apply sign-invariant processing (default: True).
            sign_invariance_method: Method for sign-invariance (default: 'square' for HKS).
            cache: Whether to cache PE in data.node_pe.
            k_eigenvectors: Number of eigenvectors to use for computation.
            exclude_zero_eigenvalue: Whether to exclude zero eigenvalue.
            min_time: Minimum diffusion time when generating default scales.
            max_time: Maximum diffusion time when generating default scales.
            learnable_scales: Whether scales are learnable (placeholder for future).
        """
        if scales is None:
            # Generate log-spaced diffusion times
            num_scales = dim // 2 if dim > 1 else 1
            scales = create_log_spaced_scales(min_time, max_time, num_scales)

        super().__init__(
            dim=dim,
            scales=scales,
            normalization=normalization,
            sign_invariant=sign_invariant,
            cache=cache,
        )

        self.sign_invariance_method = sign_invariance_method
        self.k_eigenvectors = k_eigenvectors
        self.exclude_zero_eigenvalue = exclude_zero_eigenvalue
        self.min_time = min_time
        self.max_time = max_time
        self.learnable_scales = learnable_scales

        # Placeholder for learnable scales (would be torch.nn.Parameter if implemented)
        if learnable_scales:
            # For now, just store as regular list
            # In future, could use nn.Parameter for learnable scales
            pass

    def compute(self, data: Data) -> torch.Tensor:
        """
        Compute Heat Kernel Signatures.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tensor of shape [num_nodes, dim] with positional encodings.
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # Limit k to available nodes
        k = min(self.k_eigenvectors, num_nodes)

        # Compute normalized Laplacian
        laplacian = get_normalized_laplacian(edge_index, num_nodes, edge_weight)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = compute_eigenvectors(
            laplacian, k=k, exclude_zero=self.exclude_zero_eigenvalue
        )

        # Compute HKS for each diffusion time
        scale_embeddings = []
        for t in self.scales:
            # HKS_t(i) = sum_k exp(-λ_k * t) * φ_k(i)^2
            # Compute exp(-λ_k * t) for all eigenvalues
            exp_weights = torch.exp(-eigenvalues * t)  # [k]

            # Compute φ_k(i)^2 for all nodes and eigenvectors
            # Apply sign-invariance by squaring (default for HKS)
            if self.sign_invariant and self.sign_invariance_method == "square":
                eigenvec_squared = eigenvectors ** 2  # [num_nodes, k]
            else:
                # Still square for HKS formula, but could use other methods
                eigenvec_squared = apply_sign_invariance(
                    eigenvectors, method=self.sign_invariance_method
                )
                if self.sign_invariance_method != "square":
                    # If not square, we need to adjust the formula
                    # For now, use square as per HKS definition
                    eigenvec_squared = eigenvectors ** 2

            # Weight by exp(-λ_k * t) and sum over k
            hks_t = (eigenvec_squared * exp_weights.unsqueeze(0)).sum(dim=1, keepdim=True)

            scale_embeddings.append(hks_t)

        # Aggregate multi-scale embeddings
        if len(scale_embeddings) > 1:
            node_pe = aggregate_multi_scale(scale_embeddings, method="concat")
        else:
            node_pe = scale_embeddings[0]

        return node_pe


def create_hks_with_default_scales(
    dim: int = 16,
    normalization: str = "graph",
    k_eigenvectors: int = 50,
    min_time: float = 0.01,
    max_time: float = 10.0,
) -> HKS:
    """
    Create HKS with default log-spaced diffusion times.

    Args:
        dim: Output dimension.
        normalization: Normalization mode.
        k_eigenvectors: Number of eigenvectors to use.
        min_time: Minimum diffusion time.
        max_time: Maximum diffusion time.

    Returns:
        HKS instance with default scales.
    """
    return HKS(
        dim=dim,
        scales=None,  # Will generate log-spaced
        normalization=normalization,
        k_eigenvectors=k_eigenvectors,
        min_time=min_time,
        max_time=max_time,
    )

