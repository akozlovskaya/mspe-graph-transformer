"""Positional encoding integration for Graph Transformers."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data


class NodePEIntegration(nn.Module):
    """
    Integration module for node-wise positional encodings.

    Combines node features with positional encodings through concatenation
    and projection, with optional residual mixing and dropout.

    Architecture:
        1. Concatenate node features and PE: [x, pe]
        2. Linear projection to hidden_dim
        3. Optional residual mixing with learnable gate
        4. Dropout regularization
    """

    def __init__(
        self,
        node_dim: int,
        pe_dim: int,
        hidden_dim: int,
        use_residual_mixing: bool = True,
        dropout: float = 0.1,
        pe_dropout: float = 0.1,
    ):
        """
        Initialize NodePEIntegration.

        Args:
            node_dim: Dimension of input node features.
            pe_dim: Dimension of positional encodings.
            hidden_dim: Output hidden dimension.
            use_residual_mixing: Whether to use residual PE mixing.
            dropout: Dropout rate for output.
            pe_dropout: Dropout rate for PE (before concatenation).
        """
        super().__init__()
        self.node_dim = node_dim
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.use_residual_mixing = use_residual_mixing

        # Projection layers
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None

        # Combined projection (for concatenation mode)
        self.combined_proj = nn.Linear(node_dim + pe_dim, hidden_dim) if pe_dim > 0 else None

        # Residual mixing gate
        if use_residual_mixing and pe_dim > 0:
            # Learnable scalar gate for PE contribution
            self.gate = nn.Parameter(torch.tensor(0.5))
        else:
            self.gate = None

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.pe_dropout = nn.Dropout(pe_dropout)

        # LayerNorm for stabilization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        node_pe: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Integrate node features with positional encodings.

        Args:
            x: Node features [N, node_dim].
            node_pe: Positional encodings [N, pe_dim] or None.

        Returns:
            Integrated features [N, hidden_dim].
        """
        if node_pe is None or self.pe_dim == 0:
            # No PE, just project node features
            h = self.node_proj(x)
            h = self.norm(h)
            h = self.dropout(h)
            return h

        # Apply dropout to PE to prevent over-reliance
        node_pe = self.pe_dropout(node_pe)

        if self.use_residual_mixing and self.gate is not None:
            # Residual mixing: h = gate * pe_proj(pe) + (1-gate) * node_proj(x)
            h_node = self.node_proj(x)
            h_pe = self.pe_proj(node_pe)

            # Sigmoid gate to ensure values in [0, 1]
            gate = torch.sigmoid(self.gate)
            h = (1 - gate) * h_node + gate * h_pe
        else:
            # Concatenation mode: h = proj([x, pe])
            h = torch.cat([x, node_pe], dim=-1)
            h = self.combined_proj(h)

        h = self.norm(h)
        h = self.dropout(h)
        return h


class RelativePEIntegration(nn.Module):
    """
    Integration module for relative (pairwise) positional encodings.

    Converts edge-based PE to attention bias format.
    """

    def __init__(
        self,
        pe_dim: int,
        num_heads: int,
        use_gating: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize RelativePEIntegration.

        Args:
            pe_dim: Dimension of edge PE.
            num_heads: Number of attention heads.
            use_gating: Whether to use learnable gating.
            dropout: Dropout rate for bias.
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.num_heads = num_heads
        self.use_gating = use_gating

        # Project PE to per-head bias
        self.proj = nn.Linear(pe_dim, num_heads)

        # Optional gating
        if use_gating:
            self.gate = nn.Parameter(torch.ones(num_heads) * 0.5)
        else:
            self.gate = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        edge_pe_index: Optional[torch.Tensor],
        edge_pe: Optional[torch.Tensor],
        num_nodes: int,
    ) -> Optional[torch.Tensor]:
        """
        Convert edge PE to attention bias.

        Args:
            edge_pe_index: Edge indices [2, num_pairs].
            edge_pe: Edge PE values [num_pairs, pe_dim].
            num_nodes: Number of nodes.

        Returns:
            Attention bias [num_heads, num_nodes, num_nodes] or None.
        """
        if edge_pe is None or edge_pe_index is None:
            return None

        # Project to num_heads
        bias_values = self.proj(edge_pe)  # [num_pairs, num_heads]

        # Apply gating
        if self.use_gating and self.gate is not None:
            gate = torch.sigmoid(self.gate)  # [num_heads]
            bias_values = bias_values * gate.unsqueeze(0)

        # Build dense bias matrix
        bias = torch.zeros(
            self.num_heads, num_nodes, num_nodes,
            device=edge_pe.device, dtype=edge_pe.dtype
        )

        # Fill in bias values
        row, col = edge_pe_index
        for h in range(self.num_heads):
            bias[h, row, col] = bias_values[:, h]

        # Apply dropout
        bias = self.dropout(bias)

        return bias

