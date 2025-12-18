"""Tests for Graph Transformer model."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

from src.models import GraphTransformer, GPSLayer, MPNNBlock
from src.models.attention import MultiHeadAttention
from src.models.pe_integration import NodePEIntegration, RelativePEIntegration


def create_test_graph(num_nodes: int = 10, node_dim: int = 8, seed: int = 42) -> Data:
    """Create a simple test graph."""
    torch.manual_seed(seed)

    # Create ring graph
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    x = torch.randn(num_nodes, node_dim)
    node_pe = torch.randn(num_nodes, 16)

    # Create relative PE (SPD-like)
    edge_pe_index = edge_index  # Use edges for simplicity
    edge_pe = torch.randn(edge_index.size(1), 16)

    return Data(
        x=x,
        edge_index=edge_index,
        node_pe=node_pe,
        edge_pe_index=edge_pe_index,
        edge_pe=edge_pe,
        y=torch.tensor([[0.5]]),  # Graph-level target
        num_nodes=num_nodes,
    )


def create_test_batch(num_graphs: int = 4, num_nodes: int = 8, node_dim: int = 8) -> Batch:
    """Create a batch of test graphs."""
    graphs = []
    for i in range(num_graphs):
        graph = create_test_graph(num_nodes=num_nodes, node_dim=node_dim, seed=42 + i)
        graphs.append(graph)
    return Batch.from_data_list(graphs)


class TestGraphTransformer:
    """Test suite for GraphTransformer."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformer(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            mpnn_type="gin",
            node_pe_dim=16,
            use_relative_pe=True,
        )

        out = model(data)

        assert out.shape == (1, 1)  # [1, out_dim] for graph task
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_pe(self):
        """Test forward pass without PE."""
        data = create_test_graph(num_nodes=10, node_dim=8)
        data.node_pe = None
        data.edge_pe = None
        data.edge_pe_index = None

        model = GraphTransformer(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,  # No PE
            use_relative_pe=False,
        )

        out = model(data)

        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_forward_batch(self):
        """Test forward pass with batch."""
        batch = create_test_batch(num_graphs=4, num_nodes=8, node_dim=8)

        model = GraphTransformer(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            mpnn_type="gin",
            node_pe_dim=16,
        )

        out = model(batch)

        assert out.shape == (4, 1)  # [batch_size, out_dim]
        assert not torch.isnan(out).any()

    def test_node_task(self):
        """Test node-level task."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformer(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=3,  # 3 classes
            mpnn_type="gin",
            node_pe_dim=16,
            task="node",
        )

        out = model(data)

        assert out.shape == (10, 3)  # [num_nodes, out_dim]
        assert not torch.isnan(out).any()

    def test_different_mpnn_types(self):
        """Test different MPNN types."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for mpnn_type in ["gin", "gat", "gcn"]:
            model = GraphTransformer(
                node_dim=8,
                hidden_dim=32,
                num_layers=2,
                num_heads=4,
                out_dim=1,
                mpnn_type=mpnn_type,
                node_pe_dim=16,
            )

            out = model(data)
            assert out.shape == (1, 1)
            assert not torch.isnan(out).any()

    def test_deep_model(self):
        """Test deep model (12 layers)."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformer(
            node_dim=8,
            hidden_dim=64,
            num_layers=12,
            num_heads=8,
            out_dim=1,
            mpnn_type="gin",
            node_pe_dim=16,
            drop_path_rate=0.1,
        )

        out = model(data)

        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic_output(self):
        """Test deterministic output with fixed seed."""
        data = create_test_graph(num_nodes=10, node_dim=8, seed=42)

        torch.manual_seed(123)
        model = GraphTransformer(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
        )
        model.eval()

        out1 = model(data)
        out2 = model(data)

        assert torch.allclose(out1, out2)

    def test_get_node_embeddings(self):
        """Test getting node embeddings."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformer(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
        )

        embeddings = model.get_node_embeddings(data)

        assert embeddings.shape == (10, 32)  # [num_nodes, hidden_dim]
        assert not torch.isnan(embeddings).any()


class TestGPSLayer:
    """Test suite for GPSLayer."""

    def test_gps_layer_basic(self):
        """Test basic GPS layer."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        layer = GPSLayer(
            hidden_dim=32,
            num_heads=4,
            mpnn_type="gin",
            dropout=0.1,
        )

        out = layer(x, edge_index)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gps_layer_with_bias(self):
        """Test GPS layer with attention bias."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        attention_bias = torch.randn(4, 10, 10)  # [num_heads, N, N]

        layer = GPSLayer(
            hidden_dim=32,
            num_heads=4,
            mpnn_type="gin",
        )

        out = layer(x, edge_index, attention_bias=attention_bias)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gps_layer_local_only(self):
        """Test GPS layer with only local MPNN."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        layer = GPSLayer(
            hidden_dim=32,
            num_heads=4,
            mpnn_type="gin",
            use_local=True,
            use_global=False,
        )

        out = layer(x, edge_index)

        assert out.shape == x.shape

    def test_gps_layer_global_only(self):
        """Test GPS layer with only global attention."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        layer = GPSLayer(
            hidden_dim=32,
            num_heads=4,
            use_local=False,
            use_global=True,
        )

        out = layer(x, edge_index)

        assert out.shape == x.shape


class TestAttention:
    """Test suite for attention modules."""

    def test_multi_head_attention(self):
        """Test multi-head attention."""
        x = torch.randn(10, 32)

        attn = MultiHeadAttention(
            hidden_dim=32,
            num_heads=4,
            dropout=0.1,
        )

        out = attn(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_attention_with_bias(self):
        """Test attention with bias."""
        x = torch.randn(10, 32)
        bias = torch.randn(4, 10, 10)  # [num_heads, N, N]

        attn = MultiHeadAttention(
            hidden_dim=32,
            num_heads=4,
        )

        out = attn(x, attention_bias=bias)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_attention_with_batch(self):
        """Test attention with batch mask."""
        x = torch.randn(20, 32)  # 2 graphs of 10 nodes each
        batch = torch.cat([torch.zeros(10), torch.ones(10)]).long()

        attn = MultiHeadAttention(
            hidden_dim=32,
            num_heads=4,
        )

        out = attn(x, batch=batch)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestMPNN:
    """Test suite for MPNN modules."""

    def test_mpnn_gin(self):
        """Test GIN MPNN."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        mpnn = MPNNBlock(hidden_dim=32, mpnn_type="gin")
        out = mpnn(x, edge_index)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_mpnn_gat(self):
        """Test GAT MPNN."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        mpnn = MPNNBlock(hidden_dim=32, mpnn_type="gat", num_heads=4)
        out = mpnn(x, edge_index)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_mpnn_gcn(self):
        """Test GCN MPNN."""
        x = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        mpnn = MPNNBlock(hidden_dim=32, mpnn_type="gcn")
        out = mpnn(x, edge_index)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestPEIntegration:
    """Test suite for PE integration modules."""

    def test_node_pe_integration(self):
        """Test node PE integration."""
        x = torch.randn(10, 8)
        pe = torch.randn(10, 16)

        integrator = NodePEIntegration(
            node_dim=8,
            pe_dim=16,
            hidden_dim=32,
        )

        out = integrator(x, pe)

        assert out.shape == (10, 32)
        assert not torch.isnan(out).any()

    def test_node_pe_integration_no_pe(self):
        """Test node PE integration without PE."""
        x = torch.randn(10, 8)

        integrator = NodePEIntegration(
            node_dim=8,
            pe_dim=16,
            hidden_dim=32,
        )

        out = integrator(x, None)

        assert out.shape == (10, 32)
        assert not torch.isnan(out).any()

    def test_relative_pe_integration(self):
        """Test relative PE integration."""
        edge_pe_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_pe = torch.randn(3, 16)

        integrator = RelativePEIntegration(
            pe_dim=16,
            num_heads=4,
        )

        bias = integrator(edge_pe_index, edge_pe, num_nodes=10)

        assert bias.shape == (4, 10, 10)  # [num_heads, N, N]
        assert not torch.isnan(bias).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

