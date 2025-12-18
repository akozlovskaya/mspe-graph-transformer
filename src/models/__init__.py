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

    if name in ["graph_transformer", "gt"]:
        if task == "node":
            model = GraphTransformerForNodeClassification(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                node_pe_dim=node_pe_dim,
                relative_pe_dim=relative_pe_dim,
                **kwargs,
            )
        else:
            model = GraphTransformerForGraphClassification(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                node_pe_dim=node_pe_dim,
                relative_pe_dim=relative_pe_dim,
                **kwargs,
            )
    elif name == "gps":
        model = GPSLayerStack(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            node_pe_dim=node_pe_dim,
            **kwargs,
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
