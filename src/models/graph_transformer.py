"""Graph Transformer model (GraphGPS-style) with multi-scale PE support.

Main model class that integrates:
    - Node-wise positional encodings (LapPE, RWSE, HKS)
    - Relative pairwise positional encodings (SPD, diffusion, resistance)
    - GPS layers (local MPNN + global attention)
    - Readout for graph-level or node-level prediction

Reference:
    "Recipe for a General, Powerful, Scalable Graph Transformer"
    (Rampášek et al., NeurIPS 2022)
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from .pe_integration import NodePEIntegration, RelativePEIntegration
from .gps_layer import GPSLayerStack
from .utils import FeedForward


class GraphTransformer(nn.Module):
    """
    Graph Transformer with multi-scale positional encodings support.

    Architecture:
        1. Node embedding: project node features + PE to hidden_dim
        2. GPS layers: L layers of (local MPNN + global attention + FFN)
        3. Readout: graph-level or node-level prediction head

    Features:
        - Native support for node-wise PE (data.node_pe)
        - Native support for relative PE (data.edge_pe, data.edge_pe_index)
        - Stable training up to 16+ layers with Pre-LN and drop path
        - Modular design for easy experimentation
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        out_dim: int = 1,
        mpnn_type: str = "gin",
        node_pe_dim: int = 0,
        use_relative_pe: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_expansion: int = 4,
        gate_type: str = "scalar",
        use_local: bool = True,
        use_global: bool = True,
        drop_path_rate: float = 0.0,
        readout: str = "mean",
        task: str = "graph",
    ):
        """
        Initialize GraphTransformer.

        Args:
            node_dim: Input node feature dimension.
            hidden_dim: Hidden dimension for all layers.
            num_layers: Number of GPS layers.
            num_heads: Number of attention heads.
            out_dim: Output dimension (num_classes or 1 for regression).
            mpnn_type: Type of local MPNN: 'gin', 'gat', 'gcn'.
            node_pe_dim: Dimension of node positional encodings (0 = no PE).
            use_relative_pe: Whether to use relative PE for attention bias.
            dropout: General dropout rate.
            attn_dropout: Attention dropout rate.
            ffn_expansion: Expansion factor for FFN.
            gate_type: Gating mechanism: 'scalar', 'vector', 'mlp'.
            use_local: Whether to use local MPNN in GPS layers.
            use_global: Whether to use global attention in GPS layers.
            drop_path_rate: Maximum drop path rate for stochastic depth.
            readout: Graph-level readout: 'mean', 'add', 'max'.
            task: Task type: 'graph' or 'node'.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.node_pe_dim = node_pe_dim
        self.use_relative_pe = use_relative_pe
        self.task = task

        # Node embedding with PE integration
        self.node_embed = NodePEIntegration(
            node_dim=node_dim,
            pe_dim=node_pe_dim,
            hidden_dim=hidden_dim,
            use_residual_mixing=True,
            dropout=dropout,
        )

        # Relative PE integration
        if use_relative_pe:
            # Assume default PE dim of 16 (will be projected)
            self.relative_pe = RelativePEIntegration(
                pe_dim=16,  # Default, will handle varying dims
                num_heads=num_heads,
                use_gating=True,
                dropout=dropout,
            )
        else:
            self.relative_pe = None

        # GPS layer stack
        self.gps_layers = GPSLayerStack(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mpnn_type=mpnn_type,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ffn_expansion=ffn_expansion,
            gate_type=gate_type,
            use_local=use_local,
            use_global=use_global,
            drop_path_rate=drop_path_rate,
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Readout function
        if readout == "mean":
            self.readout_fn = global_mean_pool
        elif readout == "add":
            self.readout_fn = global_add_pool
        elif readout == "max":
            self.readout_fn = global_max_pool
        else:
            raise ValueError(f"Unknown readout: {readout}")

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass through Graph Transformer.

        Args:
            data: PyG Data or Batch object with:
                - x: Node features [N, node_dim]
                - edge_index: Edge indices [2, E]
                - node_pe: Node PE [N, pe_dim] (optional)
                - edge_pe_index: Relative PE indices [2, P] (optional)
                - edge_pe: Relative PE values [P, pe_dim] (optional)
                - batch: Batch assignment [N] (for Batch)

        Returns:
            Predictions:
                - Graph task: [B, out_dim] where B is batch size
                - Node task: [N, out_dim]
        """
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        # Get optional PE
        node_pe = getattr(data, "node_pe", None)
        edge_pe_index = getattr(data, "edge_pe_index", None)
        edge_pe = getattr(data, "edge_pe", None)
        edge_attr = getattr(data, "edge_attr", None)

        # Embed nodes with PE
        h = self.node_embed(x, node_pe)

        # Build attention bias from relative PE
        if self.use_relative_pe and edge_pe is not None and edge_pe_index is not None:
            # Handle varying PE dimensions
            if edge_pe.size(1) != self.relative_pe.pe_dim:
                # Project to expected dimension
                if not hasattr(self, "_pe_proj"):
                    self._pe_proj = nn.Linear(
                        edge_pe.size(1), self.relative_pe.pe_dim
                    ).to(edge_pe.device)
                edge_pe = self._pe_proj(edge_pe)

            attention_bias = self.relative_pe(edge_pe_index, edge_pe, h.size(0))
        else:
            attention_bias = None

        # Pass through GPS layers
        h = self.gps_layers(
            h, edge_index, edge_attr,
            attention_bias=attention_bias,
            batch=batch,
        )

        # Final normalization
        h = self.final_norm(h)

        # Readout and prediction
        if self.task == "graph":
            if batch is None:
                # Single graph
                h = h.mean(dim=0, keepdim=True)
            else:
                # Batch of graphs
                h = self.readout_fn(h, batch)
            out = self.pred_head(h)
        else:
            # Node-level task
            out = self.pred_head(h)

        return out

    def get_node_embeddings(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get node embeddings before prediction head.

        Args:
            data: PyG Data or Batch object.

        Returns:
            Node embeddings [N, hidden_dim].
        """
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        node_pe = getattr(data, "node_pe", None)
        edge_pe_index = getattr(data, "edge_pe_index", None)
        edge_pe = getattr(data, "edge_pe", None)
        edge_attr = getattr(data, "edge_attr", None)

        # Embed nodes with PE
        h = self.node_embed(x, node_pe)

        # Build attention bias
        if self.use_relative_pe and edge_pe is not None and edge_pe_index is not None:
            attention_bias = self.relative_pe(edge_pe_index, edge_pe, h.size(0))
        else:
            attention_bias = None

        # Pass through GPS layers
        h = self.gps_layers(
            h, edge_index, edge_attr,
            attention_bias=attention_bias,
            batch=batch,
        )

        # Final normalization
        h = self.final_norm(h)

        return h


class GraphTransformerForNodeClassification(GraphTransformer):
    """Graph Transformer specialized for node classification."""

    def __init__(self, *args, **kwargs):
        kwargs["task"] = "node"
        super().__init__(*args, **kwargs)


class GraphTransformerForGraphClassification(GraphTransformer):
    """Graph Transformer specialized for graph classification."""

    def __init__(self, *args, **kwargs):
        kwargs["task"] = "graph"
        super().__init__(*args, **kwargs)

