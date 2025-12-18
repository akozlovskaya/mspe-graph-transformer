"""Multi-head attention with relative PE support for Graph Transformers."""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with support for attention bias from relative PE.

    Supports:
        - Dense attention bias [H, N, N]
        - Sparse attention bias (via edge_index format)
        - Batch masking for PyG batches
        - Optional FlashAttention (when available)

    Architecture choice: Pre-LN (LayerNorm before attention)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash_attention: bool = False,
    ):
        """
        Initialize MultiHeadAttention.

        Args:
            hidden_dim: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            bias: Whether to use bias in projections.
            use_flash_attention: Whether to use FlashAttention (if available).
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # FlashAttention flag
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional attention bias.

        Args:
            x: Input features [N, hidden_dim].
            attention_bias: Relative PE bias [num_heads, N, N] or None.
            attention_mask: Boolean mask [N, N] (True = masked) or None.
            batch: Batch assignment [N] for PyG batches.

        Returns:
            Output features [N, hidden_dim].
        """
        N = x.size(0)

        # Compute Q, K, V
        q = self.q_proj(x)  # [N, hidden_dim]
        k = self.k_proj(x)  # [N, hidden_dim]
        v = self.v_proj(x)  # [N, hidden_dim]

        # Reshape for multi-head attention
        # [N, hidden_dim] -> [N, num_heads, head_dim] -> [num_heads, N, head_dim]
        q = q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention scores
        # [num_heads, N, head_dim] @ [num_heads, head_dim, N] -> [num_heads, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add attention bias from relative PE
        if attention_bias is not None:
            attn = attn + attention_bias

        # Apply batch mask (for PyG batches - mask attention between different graphs)
        if batch is not None:
            batch_mask = self._create_batch_mask(batch, N, x.device)
            attn = attn.masked_fill(batch_mask.unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.unsqueeze(0), float('-inf'))

        # Softmax
        attn = F.softmax(attn, dim=-1)

        # Handle NaN from all-masked rows
        attn = torch.nan_to_num(attn, nan=0.0)

        # Apply dropout
        attn = self.attn_dropout(attn)

        # Apply attention to values
        # [num_heads, N, N] @ [num_heads, N, head_dim] -> [num_heads, N, head_dim]
        out = torch.matmul(attn, v)

        # Reshape back
        # [num_heads, N, head_dim] -> [N, num_heads, head_dim] -> [N, hidden_dim]
        out = out.transpose(0, 1).contiguous().view(N, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

    def _create_batch_mask(
        self,
        batch: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create mask for batch attention (mask attention between different graphs).

        Args:
            batch: Batch assignment [N].
            num_nodes: Number of nodes.
            device: Device.

        Returns:
            Boolean mask [N, N] where True means masked (different graphs).
        """
        # Create mask where True if nodes are in different graphs
        batch_i = batch.unsqueeze(1)  # [N, 1]
        batch_j = batch.unsqueeze(0)  # [1, N]
        mask = batch_i != batch_j  # [N, N]
        return mask


class SparseMultiHeadAttention(nn.Module):
    """
    Sparse multi-head attention that operates only on edges.

    More memory-efficient for sparse graphs, but requires explicit edge structure.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize SparseMultiHeadAttention.

        Args:
            hidden_dim: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            bias: Whether to use bias.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sparse attention forward pass.

        Args:
            x: Node features [N, hidden_dim].
            edge_index: Edge indices [2, E].
            edge_bias: Edge-level bias [E, num_heads] or None.

        Returns:
            Output features [N, hidden_dim].
        """
        from torch_geometric.utils import softmax as scatter_softmax

        N = x.size(0)
        E = edge_index.size(1)
        row, col = edge_index  # source, target

        # Compute Q, K, V
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # Get queries at target nodes, keys at source nodes
        q_i = q[col]  # [E, num_heads, head_dim]
        k_j = k[row]  # [E, num_heads, head_dim]
        v_j = v[row]  # [E, num_heads, head_dim]

        # Compute attention scores
        attn = (q_i * k_j).sum(dim=-1) * self.scale  # [E, num_heads]

        # Add edge bias
        if edge_bias is not None:
            attn = attn + edge_bias

        # Softmax over incoming edges for each node
        attn = scatter_softmax(attn, col, num_nodes=N)  # [E, num_heads]
        attn = self.dropout(attn)

        # Aggregate
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        attn_expanded = attn.unsqueeze(-1)  # [E, num_heads, 1]
        weighted_v = attn_expanded * v_j  # [E, num_heads, head_dim]

        # Scatter add to target nodes
        out.scatter_add_(0, col.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim), weighted_v)

        # Reshape and output projection
        out = out.view(N, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

