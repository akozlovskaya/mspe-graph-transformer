"""GPS (General, Powerful, Scalable) Layer for Graph Transformers.

The GPS Layer combines local message passing with global self-attention,
following the architecture from "Recipe for a General, Powerful, Scalable
Graph Transformer" (Rampášek et al., 2022).

Architecture:
    1. Local MPNN block (GIN/GAT/GCN)
    2. Global Transformer block (multi-head attention)
    3. Gating mechanism to mix local and global
    4. Feed-forward network
    Each sub-block has residual connections and layer normalization.

Normalization: Pre-LN (LayerNorm before each sub-block) for stability.
"""

from typing import Optional
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .mpnn import MPNNBlock
from .utils import FeedForward, GatingMechanism, DropPath


class GPSLayer(nn.Module):
    """
    GPS Layer: Local MPNN + Global Attention + FFN.

    This is the main building block for Graph Transformers, combining
    the locality inductive bias of MPNNs with the global receptive field
    of Transformers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mpnn_type: str = "gin",
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_expansion: int = 4,
        gate_type: str = "scalar",
        use_local: bool = True,
        use_global: bool = True,
        drop_path: float = 0.0,
    ):
        """
        Initialize GPSLayer.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            mpnn_type: Type of local MPNN: 'gin', 'gat', 'gcn'.
            dropout: General dropout rate.
            attn_dropout: Attention dropout rate.
            ffn_expansion: Expansion factor for FFN.
            gate_type: Gating mechanism type: 'scalar', 'vector', 'mlp'.
            use_local: Whether to use local MPNN block.
            use_global: Whether to use global attention block.
            drop_path: Drop path rate for stochastic depth.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_local = use_local
        self.use_global = use_global

        # Pre-LN for stability
        # Local block
        if use_local:
            self.norm_local = nn.LayerNorm(hidden_dim)
            self.local_mpnn = MPNNBlock(
                hidden_dim=hidden_dim,
                mpnn_type=mpnn_type,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Global block
        if use_global:
            self.norm_global = nn.LayerNorm(hidden_dim)
            self.global_attn = MultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
            )

        # Gating mechanism (only if both local and global are used)
        if use_local and use_global:
            self.gate = GatingMechanism(
                hidden_dim=hidden_dim,
                gate_type=gate_type,
            )
        else:
            self.gate = None

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            expansion_factor=ffn_expansion,
            dropout=dropout,
        )

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GPS layer.

        Args:
            x: Node features [N, hidden_dim].
            edge_index: Edge indices [2, E].
            edge_attr: Edge attributes [E, edge_dim] (optional).
            attention_bias: Relative PE attention bias [H, N, N] or None.
            batch: Batch assignment [N] for PyG batches.

        Returns:
            Updated node features [N, hidden_dim].
        """
        # Local MPNN block
        if self.use_local:
            h_local = self.local_mpnn(self.norm_local(x), edge_index, edge_attr)
            h_local = self.drop_path(h_local)
        else:
            h_local = None

        # Global attention block
        if self.use_global:
            h_global = self.global_attn(
                self.norm_global(x),
                attention_bias=attention_bias,
                batch=batch,
            )
            h_global = self.drop_path(h_global)
        else:
            h_global = None

        # Combine local and global
        if self.use_local and self.use_global:
            # Gated mixing
            h = x + self.gate(h_local, h_global)
        elif self.use_local:
            h = x + h_local
        elif self.use_global:
            h = x + h_global
        else:
            h = x

        # Feed-forward network with residual
        h = h + self.drop_path(self.ffn(self.norm_ffn(h)))

        return h


class GPSLayerStack(nn.Module):
    """
    Stack of GPS layers with optional stochastic depth.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        mpnn_type: str = "gin",
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_expansion: int = 4,
        gate_type: str = "scalar",
        use_local: bool = True,
        use_global: bool = True,
        drop_path_rate: float = 0.0,
    ):
        """
        Initialize GPSLayerStack.

        Args:
            num_layers: Number of GPS layers.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            mpnn_type: Type of local MPNN.
            dropout: General dropout rate.
            attn_dropout: Attention dropout rate.
            ffn_expansion: Expansion factor for FFN.
            gate_type: Gating mechanism type.
            use_local: Whether to use local MPNN.
            use_global: Whether to use global attention.
            drop_path_rate: Maximum drop path rate (linearly increases).
        """
        super().__init__()

        # Linearly increasing drop path rate
        drop_path_rates = [
            drop_path_rate * i / (num_layers - 1) if num_layers > 1 else 0.0
            for i in range(num_layers)
        ]

        self.layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mpnn_type=mpnn_type,
                dropout=dropout,
                attn_dropout=attn_dropout,
                ffn_expansion=ffn_expansion,
                gate_type=gate_type,
                use_local=use_local,
                use_global=use_global,
                drop_path=drop_path_rates[i],
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all GPS layers.

        Args:
            x: Node features [N, hidden_dim].
            edge_index: Edge indices [2, E].
            edge_attr: Edge attributes (optional).
            attention_bias: Relative PE attention bias (optional).
            batch: Batch assignment (optional).

        Returns:
            Updated node features [N, hidden_dim].
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, attention_bias, batch)
        return x

