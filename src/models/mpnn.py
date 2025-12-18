"""Message Passing Neural Network layers for Graph Transformers."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops


class MPNNBlock(nn.Module):
    """
    Unified MPNN block supporting GIN, GAT, and GCN.

    All MPNN variants follow the same interface:
        - Input: [N, hidden_dim], edge_index [2, E]
        - Output: [N, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        mpnn_type: str = "gin",
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize MPNNBlock.

        Args:
            hidden_dim: Hidden dimension.
            mpnn_type: Type of MPNN: 'gin', 'gat', or 'gcn'.
            num_heads: Number of heads for GAT.
            dropout: Dropout rate.
            edge_dim: Edge feature dimension (for edge-aware MPNNs).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mpnn_type = mpnn_type.lower()
        self.dropout = nn.Dropout(dropout)

        if self.mpnn_type == "gin":
            self.conv = GINBlock(hidden_dim, dropout=dropout)
        elif self.mpnn_type == "gat":
            self.conv = GATBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        elif self.mpnn_type == "gcn":
            self.conv = GCNBlock(hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown MPNN type: {mpnn_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through MPNN.

        Args:
            x: Node features [N, hidden_dim].
            edge_index: Edge indices [2, E].
            edge_attr: Edge attributes [E, edge_dim] (optional).

        Returns:
            Updated node features [N, hidden_dim].
        """
        return self.conv(x, edge_index, edge_attr)


class GINBlock(nn.Module):
    """
    Graph Isomorphism Network (GIN) block.

    Formula:
        h_v = MLP((1 + ε) * h_v + Σ_{u∈N(v)} h_u)
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        """
        Initialize GINBlock.

        Args:
            hidden_dim: Hidden dimension.
            dropout: Dropout rate.
            eps: Initial epsilon value.
            train_eps: Whether epsilon is trainable.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # MLP for GIN
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.conv = GINConv(mlp, eps=eps, train_eps=train_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GIN."""
        out = self.conv(x, edge_index)
        out = self.dropout(out)
        return out


class GATBlock(nn.Module):
    """
    Graph Attention Network (GAT) block.

    Uses multi-head attention over neighbors.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = False,
    ):
        """
        Initialize GATBlock.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            concat: Whether to concatenate heads (if False, average).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat = concat

        if concat:
            assert hidden_dim % num_heads == 0
            head_dim = hidden_dim // num_heads
            out_channels = head_dim
        else:
            out_channels = hidden_dim

        self.conv = GATConv(
            in_channels=hidden_dim,
            out_channels=out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GAT."""
        out = self.conv(x, edge_index)
        out = self.dropout(out)
        return out


class GCNBlock(nn.Module):
    """
    Graph Convolutional Network (GCN) block.

    Formula:
        h_v = σ(D^{-1/2} A D^{-1/2} h W)
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize GCNBlock.

        Args:
            hidden_dim: Hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GCN."""
        out = self.conv(x, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        return out


def get_mpnn(
    mpnn_type: str,
    hidden_dim: int,
    num_heads: int = 4,
    dropout: float = 0.1,
    edge_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory function to create MPNN block.

    Args:
        mpnn_type: Type of MPNN: 'gin', 'gat', or 'gcn'.
        hidden_dim: Hidden dimension.
        num_heads: Number of heads for GAT.
        dropout: Dropout rate.
        edge_dim: Edge feature dimension.

    Returns:
        MPNN module.
    """
    return MPNNBlock(
        hidden_dim=hidden_dim,
        mpnn_type=mpnn_type,
        num_heads=num_heads,
        dropout=dropout,
        edge_dim=edge_dim,
    )

