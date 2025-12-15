"""Laplacian Positional Encoding (LapPE) implementation."""

from typing import Optional, List, Union
import torch
from torch_geometric.data import Data

from .base import BaseNodePE
from .utils import (
    get_normalized_laplacian,
    compute_eigenvectors,
    apply_sign_invariance,
    aggregate_multi_scale,
)


class LapPE(BaseNodePE):
    """
    Laplacian Positional Encoding.

    Computes positional encodings from eigenvectors of the normalized graph Laplacian.
    Supports sign-invariant processing and multi-scale modes.

    Formula:
        Uses top-k eigenvectors of normalized Laplacian L = I - D^{-1/2} A D^{-1/2}

    Reference:
        "Graph Transformer Networks" (Dwivedi & Bresson, 2020)
    """

    def __init__(
        self,
        dim: int,
        k: Optional[int] = None,
        scales: Optional[List[int]] = None,
        normalization: str = "graph",
        sign_invariant: bool = True,
        sign_invariance_method: str = "abs",
        cache: bool = True,
        exclude_zero_eigenvalue: bool = True,
    ):
        """
        Initialize LapPE.

        Args:
            dim: Output embedding dimension per node.
            k: Number of eigenvectors to use. If None, uses dim.
               If scales is provided, this is the number per scale.
            scales: List of k values for multi-scale mode. If None, uses single k.
            normalization: Normalization mode: 'graph', 'node', or None.
            sign_invariant: Whether to apply sign-invariant processing.
            sign_invariance_method: Method for sign-invariance: 'abs', 'flip', or 'square'.
            cache: Whether to cache PE in data.node_pe.
            exclude_zero_eigenvalue: Whether to exclude zero eigenvalue.
        """
        # Use scales as list of k values, or default to [k] or [dim]
        if scales is not None:
            self.ks = scales
        elif k is not None:
            self.ks = [k]
        else:
            self.ks = [dim]

        super().__init__(
            dim=dim,
            scales=self.ks,  # Pass as scales for base class
            normalization=normalization,
            sign_invariant=sign_invariant,
            cache=cache,
        )

        self.sign_invariance_method = sign_invariance_method
        self.exclude_zero_eigenvalue = exclude_zero_eigenvalue

    def compute(self, data: Data) -> torch.Tensor:
        """
        Compute Laplacian positional encodings.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tensor of shape [num_nodes, dim] with positional encodings.
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # Compute normalized Laplacian
        laplacian = get_normalized_laplacian(edge_index, num_nodes, edge_weight)

        # Multi-scale: use different k values
        scale_embeddings = []
        for k in self.ks:
            # Compute top-k eigenvectors
            eigenvalues, eigenvectors = compute_eigenvectors(
                laplacian, k=k, exclude_zero=self.exclude_zero_eigenvalue
            )

            # Apply sign-invariant processing if enabled
            if self.sign_invariant:
                pe = apply_sign_invariance(eigenvectors, method=self.sign_invariance_method)
            else:
                pe = eigenvectors

            scale_embeddings.append(pe)

        # Aggregate multi-scale embeddings
        if len(scale_embeddings) > 1:
            node_pe = aggregate_multi_scale(scale_embeddings, method="concat")
        else:
            node_pe = scale_embeddings[0]

        return node_pe


def create_multi_scale_lappe(
    dim: int,
    k_scales: List[int] = [8, 16, 32],
    normalization: str = "graph",
    sign_invariant: bool = True,
) -> LapPE:
    """
    Create multi-scale LapPE with different k values.

    Args:
        dim: Target output dimension.
        k_scales: List of k values (eigenvectors) to use per scale.
        normalization: Normalization mode.
        sign_invariant: Whether to apply sign-invariant processing.

    Returns:
        LapPE instance configured for multi-scale.
    """
    return LapPE(
        dim=dim,
        scales=k_scales,
        normalization=normalization,
        sign_invariant=sign_invariant,
    )

