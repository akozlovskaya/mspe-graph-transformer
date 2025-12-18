"""FLOPs estimation utilities.

Assumptions:
- Matrix multiplication: A (m x k) @ B (k x n) = 2 * m * k * n FLOPs
- Element-wise operations: 1 FLOP per element
- Softmax: ~5N FLOPs per row of length N
- Layer normalization: ~5N FLOPs
- Activation functions: 1 FLOP per element

All estimates are approximate and consistent for comparisons.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FLOPsEstimate:
    """Container for FLOPs estimate."""

    total: int
    breakdown: Dict[str, int]
    unit: str = "GFLOPs"

    def __repr__(self):
        gflops = self.total / 1e9
        return f"{gflops:.2f} {self.unit}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "total_gflops": self.total / 1e9,
            "breakdown": self.breakdown,
        }


def estimate_linear_flops(
    in_features: int,
    out_features: int,
    batch_size: int = 1,
    sequence_length: int = 1,
    bias: bool = True,
) -> int:
    """
    Estimate FLOPs for linear layer.

    Args:
        in_features: Input features.
        out_features: Output features.
        batch_size: Batch size.
        sequence_length: Sequence/node length.
        bias: Whether bias is used.

    Returns:
        Estimated FLOPs.
    """
    # Matrix multiplication: 2 * m * k * n
    num_elements = batch_size * sequence_length
    flops = 2 * num_elements * in_features * out_features

    # Bias addition
    if bias:
        flops += num_elements * out_features

    return flops


def estimate_attention_flops(
    num_nodes: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int = 1,
    include_projections: bool = True,
    sparse_ratio: float = 1.0,
) -> Dict[str, int]:
    """
    Estimate FLOPs for multi-head attention.

    Args:
        num_nodes: Number of nodes (sequence length).
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        batch_size: Batch size.
        include_projections: Include Q, K, V, O projections.
        sparse_ratio: Ratio of non-zero attention weights (1.0 = dense).

    Returns:
        Dictionary with FLOPs breakdown.
    """
    head_dim = hidden_dim // num_heads
    N = num_nodes
    B = batch_size
    H = num_heads
    D = head_dim

    flops = {}

    if include_projections:
        # Q, K, V projections: 3 * Linear(hidden_dim, hidden_dim)
        flops["qkv_projection"] = 3 * estimate_linear_flops(
            hidden_dim, hidden_dim, B, N
        )

    # Attention scores: Q @ K^T per head
    # [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
    # FLOPs = B * H * N * D * N * 2
    num_pairs = int(N * N * sparse_ratio)
    flops["attention_scores"] = 2 * B * H * N * D * int(N * sparse_ratio)

    # Softmax: ~5N per row for N rows per head
    flops["softmax"] = 5 * B * H * N * int(N * sparse_ratio)

    # Attention @ V
    # [B, H, N, N] @ [B, H, N, D] -> [B, H, N, D]
    flops["attention_values"] = 2 * B * H * N * int(N * sparse_ratio) * D

    if include_projections:
        # Output projection: Linear(hidden_dim, hidden_dim)
        flops["output_projection"] = estimate_linear_flops(
            hidden_dim, hidden_dim, B, N
        )

    flops["total"] = sum(flops.values())

    return flops


def estimate_mpnn_flops(
    num_nodes: int,
    num_edges: int,
    in_features: int,
    out_features: int,
    batch_size: int = 1,
    mpnn_type: str = "gin",
) -> Dict[str, int]:
    """
    Estimate FLOPs for MPNN layer.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of edges.
        in_features: Input features.
        out_features: Output features.
        batch_size: Batch size.
        mpnn_type: Type of MPNN: 'gin', 'gcn', 'gat'.

    Returns:
        Dictionary with FLOPs breakdown.
    """
    N = num_nodes * batch_size
    E = num_edges * batch_size

    flops = {}

    if mpnn_type == "gin":
        # GIN: MLP on aggregated features
        # Aggregation: sum over neighbors (E additions)
        flops["aggregation"] = E * in_features

        # MLP: typically 2 layers
        flops["mlp"] = estimate_linear_flops(in_features, out_features, 1, N)
        flops["mlp"] += estimate_linear_flops(out_features, out_features, 1, N)

        # Activation
        flops["activation"] = N * out_features

    elif mpnn_type == "gcn":
        # GCN: Linear + aggregation
        flops["linear"] = estimate_linear_flops(in_features, out_features, 1, N)
        flops["aggregation"] = E * out_features
        flops["activation"] = N * out_features

    elif mpnn_type == "gat":
        # GAT: attention-weighted aggregation
        flops["linear"] = estimate_linear_flops(in_features, out_features, 1, N)
        flops["attention_scores"] = 2 * E * out_features  # Attention computation
        flops["softmax"] = 5 * E  # Per-edge softmax
        flops["aggregation"] = E * out_features
        flops["activation"] = N * out_features

    flops["total"] = sum(flops.values())

    return flops


def estimate_ffn_flops(
    hidden_dim: int,
    ffn_dim: int,
    num_nodes: int,
    batch_size: int = 1,
) -> Dict[str, int]:
    """
    Estimate FLOPs for feed-forward network.

    Args:
        hidden_dim: Input/output dimension.
        ffn_dim: Intermediate dimension.
        num_nodes: Number of nodes.
        batch_size: Batch size.

    Returns:
        Dictionary with FLOPs breakdown.
    """
    N = num_nodes * batch_size

    flops = {}

    # First linear: hidden_dim -> ffn_dim
    flops["linear1"] = estimate_linear_flops(hidden_dim, ffn_dim, 1, N)

    # Activation (GELU ~4 ops)
    flops["activation"] = 4 * N * ffn_dim

    # Second linear: ffn_dim -> hidden_dim
    flops["linear2"] = estimate_linear_flops(ffn_dim, hidden_dim, 1, N)

    flops["total"] = sum(flops.values())

    return flops


def estimate_layer_norm_flops(
    hidden_dim: int,
    num_nodes: int,
    batch_size: int = 1,
) -> int:
    """
    Estimate FLOPs for layer normalization.

    Args:
        hidden_dim: Feature dimension.
        num_nodes: Number of nodes.
        batch_size: Batch size.

    Returns:
        Estimated FLOPs.
    """
    N = num_nodes * batch_size

    # Mean: N * hidden_dim additions
    # Variance: N * hidden_dim subtractions + squares + additions
    # Normalize: N * hidden_dim divisions
    # Scale & shift: 2 * N * hidden_dim multiplications/additions

    return 5 * N * hidden_dim


def estimate_pe_flops(
    num_nodes: int,
    pe_dim: int,
    pe_type: str = "lap",
    num_edges: int = 0,
    k_eigenvectors: int = 16,
) -> Dict[str, int]:
    """
    Estimate FLOPs for positional encoding computation.

    Args:
        num_nodes: Number of nodes.
        pe_dim: PE output dimension.
        pe_type: Type of PE: 'lap', 'rwse', 'spd'.
        num_edges: Number of edges.
        k_eigenvectors: Number of eigenvectors for spectral PE.

    Returns:
        Dictionary with FLOPs breakdown.
    """
    N = num_nodes
    E = num_edges if num_edges > 0 else N * 5  # Assume avg degree 5

    flops = {}

    if pe_type == "lap":
        # Laplacian computation: O(E)
        flops["laplacian"] = 3 * E

        # Eigendecomposition: O(N * k^2) for iterative methods
        flops["eigendecomposition"] = N * k_eigenvectors * k_eigenvectors * 10

        # Projection to pe_dim
        flops["projection"] = estimate_linear_flops(k_eigenvectors, pe_dim, 1, N)

    elif pe_type == "rwse":
        # Random walk: matrix-vector products per scale
        num_scales = 5  # Typical
        # Each scale: sparse matrix-vector product O(E)
        flops["random_walk"] = num_scales * E * 2

        # Projection
        flops["projection"] = estimate_linear_flops(num_scales, pe_dim, 1, N)

    elif pe_type == "spd":
        # BFS from each node: O(N * E) worst case, typically O(N * avg_path_length * avg_degree)
        avg_path_length = min(10, int(N ** 0.5))
        flops["bfs"] = N * avg_path_length * (E // N)

        # One-hot encoding
        flops["encoding"] = N * N  # Simplified

    elif pe_type == "hks":
        # Heat kernel: similar to LapPE + exponentials
        flops["laplacian"] = 3 * E
        flops["eigendecomposition"] = N * k_eigenvectors * k_eigenvectors * 10

        num_scales = 5
        # Exponentials and sums per scale
        flops["heat_kernel"] = num_scales * N * k_eigenvectors * 3

        flops["projection"] = estimate_linear_flops(num_scales, pe_dim, 1, N)

    flops["total"] = sum(flops.values())

    return flops


def estimate_model_flops(
    num_nodes: int,
    num_edges: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    batch_size: int = 1,
    include_mpnn: bool = True,
    sparse_attention: float = 1.0,
    node_pe_dim: int = 0,
    relative_pe_buckets: int = 0,
) -> FLOPsEstimate:
    """
    Estimate total FLOPs for Graph Transformer model.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of edges.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        ffn_dim: FFN intermediate dimension.
        batch_size: Batch size.
        include_mpnn: Whether model includes MPNN.
        sparse_attention: Attention sparsity ratio.
        node_pe_dim: Node PE dimension (0 if not used).
        relative_pe_buckets: Relative PE buckets (0 if not used).

    Returns:
        FLOPsEstimate with total and breakdown.
    """
    breakdown = {}

    # Input projection
    in_features = hidden_dim  # Assume already projected
    breakdown["input_projection"] = estimate_linear_flops(
        in_features, hidden_dim, batch_size, num_nodes
    )

    # PE projection (if used)
    if node_pe_dim > 0:
        breakdown["node_pe_projection"] = estimate_linear_flops(
            node_pe_dim, hidden_dim, batch_size, num_nodes
        )

    # Per-layer computations
    attention_per_layer = estimate_attention_flops(
        num_nodes, hidden_dim, num_heads, batch_size,
        sparse_ratio=sparse_attention
    )["total"]

    ffn_per_layer = estimate_ffn_flops(
        hidden_dim, ffn_dim, num_nodes, batch_size
    )["total"]

    layer_norm_per_layer = 2 * estimate_layer_norm_flops(
        hidden_dim, num_nodes, batch_size
    )  # Pre-norm for attention and FFN

    breakdown["attention"] = num_layers * attention_per_layer
    breakdown["ffn"] = num_layers * ffn_per_layer
    breakdown["layer_norm"] = num_layers * layer_norm_per_layer

    if include_mpnn:
        mpnn_per_layer = estimate_mpnn_flops(
            num_nodes, num_edges, hidden_dim, hidden_dim, batch_size
        )["total"]
        breakdown["mpnn"] = num_layers * mpnn_per_layer

    # Relative PE bias (if used)
    if relative_pe_buckets > 0:
        # Embedding lookup and projection
        num_pairs = int(num_nodes * num_nodes * sparse_attention)
        breakdown["relative_pe_bias"] = num_layers * num_pairs * relative_pe_buckets

    # Output projection
    breakdown["output_projection"] = estimate_linear_flops(
        hidden_dim, 1, batch_size, num_nodes  # Assume single output
    )

    total = sum(breakdown.values())

    return FLOPsEstimate(total=total, breakdown=breakdown)


class FLOPsEstimator:
    """FLOPs estimator for Graph Transformer models."""

    def __init__(self, model: nn.Module):
        """
        Initialize FLOPs estimator.

        Args:
            model: Model to estimate FLOPs for.
        """
        self.model = model
        self._extract_model_config()

    def _extract_model_config(self):
        """Extract model configuration from architecture."""
        self.config = {
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "ffn_dim": 512,
        }

        # Try to extract from model
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "ffn" in name or "mlp" in name:
                    self.config["ffn_dim"] = max(
                        self.config["ffn_dim"], module.out_features
                    )
                else:
                    self.config["hidden_dim"] = module.out_features

            if isinstance(module, nn.MultiheadAttention):
                self.config["num_heads"] = module.num_heads
                self.config["hidden_dim"] = module.embed_dim

        # Count layers
        num_layers = 0
        for name, _ in self.model.named_modules():
            if "layer" in name.lower() and name.count(".") == 1:
                num_layers += 1
        if num_layers > 0:
            self.config["num_layers"] = num_layers

    def estimate(
        self,
        num_nodes: int,
        num_edges: int,
        batch_size: int = 1,
        **kwargs,
    ) -> FLOPsEstimate:
        """
        Estimate FLOPs for given input size.

        Args:
            num_nodes: Number of nodes.
            num_edges: Number of edges.
            batch_size: Batch size.
            **kwargs: Additional config overrides.

        Returns:
            FLOPsEstimate.
        """
        config = {**self.config, **kwargs}

        return estimate_model_flops(
            num_nodes=num_nodes,
            num_edges=num_edges,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            batch_size=batch_size,
            **{k: v for k, v in kwargs.items() if k not in config},
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

