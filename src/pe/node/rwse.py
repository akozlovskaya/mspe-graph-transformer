"""Random-Walk Structural Encoding (RWSE) implementation."""

from typing import Optional, List
import torch
from torch_geometric.data import Data

from .base import BaseNodePE
from .utils import (
    get_transition_matrix,
    compute_power_matrix,
    aggregate_multi_scale,
    create_log_spaced_scales,
)


class RWSE(BaseNodePE):
    """
    Random-Walk Structural Encoding.

    Computes node encodings based on return probabilities in random walks.

    Formula:
        RWSE_t(i) = P^t(i, i)
        where P is the transition matrix D^{-1} A

    Reference:
        "Transformers are Graph Neural Networks" (Dwivedi et al., 2021)
    """

    def __init__(
        self,
        dim: int,
        scales: Optional[List[int]] = None,
        normalization: str = "graph",
        sign_invariant: bool = False,  # RWSE doesn't need sign-invariance
        cache: bool = True,
        log_spaced: bool = True,
        max_scale: int = 32,
    ):
        """
        Initialize RWSE.

        Args:
            dim: Output embedding dimension per node.
            scales: List of RW steps (t values). If None, generates log-spaced scales.
            normalization: Normalization mode: 'graph', 'node', or None.
            sign_invariant: Not used for RWSE (kept for API consistency).
            cache: Whether to cache PE in data.node_pe.
            log_spaced: Whether to use log-spaced scales when scales=None.
            max_scale: Maximum RW step when generating default scales.
        """
        if scales is None:
            if log_spaced:
                # Generate log-spaced integer scales
                num_scales = dim // 2 if dim > 1 else 1
                scales_float = create_log_spaced_scales(1.0, float(max_scale), num_scales)
                scales = [int(max(1, round(s))) for s in scales_float]
            else:
                # Linear spacing
                scales = list(range(1, min(dim + 1, max_scale + 1)))

        super().__init__(
            dim=dim,
            scales=scales,
            normalization=normalization,
            sign_invariant=False,  # RWSE doesn't need sign-invariance
            cache=cache,
        )

        self.max_scale = max_scale

    def compute(self, data: Data) -> torch.Tensor:
        """
        Compute Random-Walk Structural Encodings.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tensor of shape [num_nodes, dim] with positional encodings.
        """
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # Compute transition matrix P = D^{-1} A
        transition = get_transition_matrix(edge_index, num_nodes, edge_weight)

        # Compute RWSE for each scale (RW step)
        scale_embeddings = []
        for t in self.scales:
            # Compute P^t
            if t == 1:
                p_t = transition
            else:
                p_t = compute_power_matrix(transition, t)

            # Extract diagonal: P^t(i, i) = return probability
            rwse_t = torch.diag(p_t).unsqueeze(1)  # [num_nodes, 1]

            scale_embeddings.append(rwse_t)

        # Aggregate multi-scale embeddings
        if len(scale_embeddings) > 1:
            node_pe = aggregate_multi_scale(scale_embeddings, method="concat")
        else:
            node_pe = scale_embeddings[0]

        return node_pe


def create_rwse_with_default_scales(
    dim: int = 16,
    normalization: str = "graph",
    max_scale: int = 32,
) -> RWSE:
    """
    Create RWSE with default log-spaced scales.

    Args:
        dim: Output dimension.
        normalization: Normalization mode.
        max_scale: Maximum RW step.

    Returns:
        RWSE instance with default scales.
    """
    return RWSE(
        dim=dim,
        scales=None,  # Will generate log-spaced
        normalization=normalization,
        max_scale=max_scale,
    )

