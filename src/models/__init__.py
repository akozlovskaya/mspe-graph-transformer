"""Graph Transformer models with multi-scale positional encodings."""

from typing import Optional

from .graph_transformer import (
    GraphTransformer,
    GraphTransformerForNodeClassification,
    GraphTransformerForGraphClassification,
)
from .gps_layer import GPSLayer, GPSLayerStack
from .attention import MultiHeadAttention, SparseMultiHeadAttention
from .mpnn import MPNNBlock, GINBlock, GATBlock, GCNBlock, get_mpnn
from .pe_integration import NodePEIntegration, RelativePEIntegration
from .utils import FeedForward, GatingMechanism, DropPath


def get_model(
    name: str = "graph_transformer",
    num_features: int = 16,
    num_classes: int = 1,
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    node_pe_dim: Optional[int] = None,
    relative_pe_dim: Optional[int] = None,
    task: str = "graph",
    **kwargs,
):
    """
    Factory function to create models.

    Args:
        name: Model name: 'graph_transformer', 'gps'.
        num_features: Number of input features.
        num_classes: Number of output classes/targets.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        node_pe_dim: Dimension of node positional encodings.
        relative_pe_dim: Dimension of relative positional encodings.
        task: Task type: 'graph', 'node', 'edge'.
        **kwargs: Additional model arguments.

    Returns:
        Model instance.
    """
    name = name.lower()

    # Map to GraphTransformer expected argument names
    use_relative_pe = relative_pe_dim is not None and relative_pe_dim > 0
    
    # Convert ffn_dim to ffn_expansion if provided
    if "ffn_dim" in kwargs:
        ffn_dim = kwargs.pop("ffn_dim")
        if "ffn_expansion" not in kwargs:
            # Calculate expansion factor: ffn_dim = hidden_dim * expansion
            kwargs["ffn_expansion"] = ffn_dim // hidden_dim if hidden_dim > 0 else 4
    
    # Filter out unsupported parameters
    supported_kwargs = {
        "mpnn_type", "attn_dropout", "ffn_expansion", "gate_type",
        "use_local", "use_global", "drop_path_rate", "readout"
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

    if name in ["graph_transformer", "gt"]:
        if task == "node":
            model = GraphTransformerForNodeClassification(
                node_dim=num_features,
                out_dim=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                node_pe_dim=node_pe_dim or 0,
                use_relative_pe=use_relative_pe,
                **filtered_kwargs,
            )
        else:
            model = GraphTransformerForGraphClassification(
                node_dim=num_features,
                out_dim=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                node_pe_dim=node_pe_dim or 0,
                use_relative_pe=use_relative_pe,
                **filtered_kwargs,
            )
    elif name == "gps":
        # GPS uses GraphTransformer as well
        # Use same filtered kwargs
        model = GraphTransformerForGraphClassification(
            node_dim=num_features,
            out_dim=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            node_pe_dim=node_pe_dim or 0,
            use_relative_pe=use_relative_pe,
            **filtered_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return model


__all__ = [
    # Factory
    "get_model",
    # Main models
    "GraphTransformer",
    "GraphTransformerForNodeClassification",
    "GraphTransformerForGraphClassification",
    # Layers
    "GPSLayer",
    "GPSLayerStack",
    # Attention
    "MultiHeadAttention",
    "SparseMultiHeadAttention",
    # MPNN
    "MPNNBlock",
    "GINBlock",
    "GATBlock",
    "GCNBlock",
    "get_mpnn",
    # PE Integration
    "NodePEIntegration",
    "RelativePEIntegration",
    # Utils
    "FeedForward",
    "GatingMechanism",
    "DropPath",
]
