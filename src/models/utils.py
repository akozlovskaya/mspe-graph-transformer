"""Utility modules for Graph Transformers."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Feed-forward network (FFN) for Transformer layers.

    Architecture: Linear -> Activation -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize FeedForward.

        Args:
            hidden_dim: Input/output dimension.
            expansion_factor: Factor to expand hidden dimension.
            dropout: Dropout rate.
            activation: Activation function: 'gelu', 'relu', 'swish'.
        """
        super().__init__()
        ff_dim = hidden_dim * expansion_factor

        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class GatingMechanism(nn.Module):
    """
    Learnable gating mechanism for mixing local and global features.

    Supports scalar, vector, or MLP-based gating.
    """

    def __init__(
        self,
        hidden_dim: int,
        gate_type: str = "scalar",
        init_value: float = 0.5,
    ):
        """
        Initialize GatingMechanism.

        Args:
            hidden_dim: Hidden dimension.
            gate_type: Type of gate: 'scalar', 'vector', or 'mlp'.
            init_value: Initial gate value.
        """
        super().__init__()
        self.gate_type = gate_type

        if gate_type == "scalar":
            # Single learnable scalar
            self.gate = nn.Parameter(torch.tensor(init_value))
        elif gate_type == "vector":
            # Per-dimension gate
            self.gate = nn.Parameter(torch.full((hidden_dim,), init_value))
        elif gate_type == "mlp":
            # MLP-based gate
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    def forward(
        self,
        h_local: torch.Tensor,
        h_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix local and global features.

        Args:
            h_local: Local features [N, hidden_dim].
            h_global: Global features [N, hidden_dim].

        Returns:
            Mixed features [N, hidden_dim].
        """
        if self.gate_type in ["scalar", "vector"]:
            g = torch.sigmoid(self.gate)
            return g * h_global + (1 - g) * h_local
        else:
            # MLP-based gating
            combined = torch.cat([h_local, h_global], dim=-1)
            g = self.gate_mlp(combined)
            return g * h_global + (1 - g) * h_local


class DropPath(nn.Module):
    """
    Drop path (stochastic depth) for regularization.

    Randomly drops entire residual paths during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath.

        Args:
            drop_prob: Drop probability.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path."""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Shape for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def create_batch_mask(batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Create attention mask for PyG batch.

    Args:
        batch: Batch assignment [N].
        num_nodes: Total number of nodes.

    Returns:
        Boolean mask [N, N] where True = masked (different graphs).
    """
    batch_i = batch.unsqueeze(1)
    batch_j = batch.unsqueeze(0)
    return batch_i != batch_j

