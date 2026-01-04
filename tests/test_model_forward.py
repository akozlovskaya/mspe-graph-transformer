"""Tests for model forward pass."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import GraphTransformer, get_model
from src.models.graph_transformer import (
    GraphTransformerForGraphClassification,
    GraphTransformerForNodeClassification,
)


def create_test_graph(num_nodes: int = 10, node_dim: int = 8, seed: int = 42) -> Data:
    """Create a simple test graph."""
    torch.manual_seed(seed)

    # Create ring graph
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return Data(
        x=torch.randn(num_nodes, node_dim),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


def create_test_batch(num_graphs: int = 4, num_nodes: int = 8, node_dim: int = 8) -> Batch:
    """Create a batch of test graphs."""
    graphs = []
    for i in range(num_graphs):
        graph = create_test_graph(num_nodes=num_nodes, node_dim=node_dim, seed=42 + i)
        graphs.append(graph)
    return Batch.from_data_list(graphs)


class TestModelForward:
    """Test suite for model forward pass."""

    def test_forward_graph_classification(self):
        """Test forward pass for graph classification task."""
        data = create_test_graph(num_nodes=10, node_dim=8)
        
        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=3,  # 3 classes
            node_pe_dim=0,
            use_relative_pe=False,
        )

        out = model(data)

        assert out.shape == (1, 3)  # [batch_size, num_classes]
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_node_classification(self):
        """Test forward pass for node classification task."""
        data = create_test_graph(num_nodes=10, node_dim=8)
        
        model = GraphTransformerForNodeClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=5,  # 5 classes
            node_pe_dim=0,
            use_relative_pe=False,
        )

        out = model(data)

        assert out.shape == (10, 5)  # [num_nodes, num_classes]
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_with_node_pe(self):
        """Test forward pass with node positional encodings."""
        data = create_test_graph(num_nodes=10, node_dim=8)
        data.node_pe = torch.randn(10, 16)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=16,
            use_relative_pe=False,
        )

        out = model(data)

        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_forward_with_relative_pe(self):
        """Test forward pass with relative positional encodings."""
        data = create_test_graph(num_nodes=10, node_dim=8)
        # Create relative PE (all pairs)
        num_pairs = 10 * 10
        data.edge_pe_index = torch.randint(0, 10, (2, num_pairs))
        data.edge_pe = torch.randn(num_pairs, 16)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=True,
        )

        out = model(data)

        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_forward_with_both_pe(self):
        """Test forward pass with both node and relative PE."""
        data = create_test_graph(num_nodes=10, node_dim=8)
        data.node_pe = torch.randn(10, 16)
        num_pairs = 10 * 10
        data.edge_pe_index = torch.randint(0, 10, (2, num_pairs))
        data.edge_pe = torch.randn(num_pairs, 16)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=16,
            use_relative_pe=True,
        )

        out = model(data)

        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_forward_batch_graph_classification(self):
        """Test forward pass with batch for graph classification."""
        batch = create_test_batch(num_graphs=4, num_nodes=8, node_dim=8)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=2,
            node_pe_dim=0,
            use_relative_pe=False,
        )

        out = model(batch)

        assert out.shape == (4, 2)  # [batch_size, num_classes]
        assert not torch.isnan(out).any()

    def test_forward_batch_node_classification(self):
        """Test forward pass with batch for node classification."""
        batch = create_test_batch(num_graphs=3, num_nodes=6, node_dim=8)

        model = GraphTransformerForNodeClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=4,
            node_pe_dim=0,
            use_relative_pe=False,
        )

        out = model(batch)

        # Total nodes = 3 graphs * 6 nodes = 18
        assert out.shape == (18, 4)  # [total_nodes, num_classes]
        assert not torch.isnan(out).any()

    def test_forward_different_output_dims(self):
        """Test forward pass with different output dimensions."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for out_dim in [1, 2, 5, 10]:
            model = GraphTransformerForGraphClassification(
                node_dim=8,
                hidden_dim=32,
                num_layers=2,
                num_heads=4,
                out_dim=out_dim,
                node_pe_dim=0,
                use_relative_pe=False,
            )

            out = model(data)
            assert out.shape == (1, out_dim)
            assert not torch.isnan(out).any()

    def test_forward_different_hidden_dims(self):
        """Test forward pass with different hidden dimensions."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for hidden_dim in [16, 32, 64, 128]:
            model = GraphTransformerForGraphClassification(
                node_dim=8,
                hidden_dim=hidden_dim,
                num_layers=2,
                num_heads=4,
                out_dim=1,
                node_pe_dim=0,
                use_relative_pe=False,
            )

            out = model(data)
            assert out.shape == (1, 1)
            assert not torch.isnan(out).any()

    def test_forward_different_num_layers(self):
        """Test forward pass with different numbers of layers."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for num_layers in [1, 2, 4, 6]:
            model = GraphTransformerForGraphClassification(
                node_dim=8,
                hidden_dim=32,
                num_layers=num_layers,
                num_heads=4,
                out_dim=1,
                node_pe_dim=0,
                use_relative_pe=False,
            )

            out = model(data)
            assert out.shape == (1, 1)
            assert not torch.isnan(out).any()

    def test_forward_different_num_heads(self):
        """Test forward pass with different numbers of attention heads."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for num_heads in [1, 2, 4, 8]:
            model = GraphTransformerForGraphClassification(
                node_dim=8,
                hidden_dim=32,
                num_layers=2,
                num_heads=num_heads,
                out_dim=1,
                node_pe_dim=0,
                use_relative_pe=False,
            )

            out = model(data)
            assert out.shape == (1, 1)
            assert not torch.isnan(out).any()

    def test_forward_different_readout(self):
        """Test forward pass with different readout methods."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for readout in ["mean", "add", "max"]:
            model = GraphTransformerForGraphClassification(
                node_dim=8,
                hidden_dim=32,
                num_layers=2,
                num_heads=4,
                out_dim=1,
                node_pe_dim=0,
                use_relative_pe=False,
                readout=readout,
            )

            out = model(data)
            assert out.shape == (1, 1)
            assert not torch.isnan(out).any()

    def test_forward_different_mpnn_types(self):
        """Test forward pass with different MPNN types."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        for mpnn_type in ["gin", "gat", "gcn"]:
            model = GraphTransformerForGraphClassification(
                node_dim=8,
                hidden_dim=32,
                num_layers=2,
                num_heads=4,
                out_dim=1,
                node_pe_dim=0,
                use_relative_pe=False,
                mpnn_type=mpnn_type,
            )

            out = model(data)
            assert out.shape == (1, 1)
            assert not torch.isnan(out).any()

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with fixed seed."""
        data = create_test_graph(num_nodes=10, node_dim=8, seed=42)

        torch.manual_seed(123)
        model1 = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
        )
        model1.eval()
        out1 = model1(data)

        torch.manual_seed(123)
        model2 = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
        )
        model2.eval()
        out2 = model2(data)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the model."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
        )

        out = model(data)
        loss = out.sum()
        loss.backward()

        # Check that gradients exist for at least some parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any()
                break

        assert has_grad, "No gradients found in model parameters"

    def test_forward_with_dropout_training(self):
        """Test forward pass with dropout in training mode."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
            dropout=0.5,  # High dropout
        )
        model.train()

        out1 = model(data)
        out2 = model(data)

        # With dropout, outputs should be different (but not always due to randomness)
        # Just check that model runs without errors
        assert out1.shape == (1, 1)
        assert out2.shape == (1, 1)

    def test_forward_with_dropout_eval(self):
        """Test forward pass with dropout in eval mode (should be deterministic)."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
            dropout=0.5,
        )
        model.eval()

        out1 = model(data)
        out2 = model(data)

        # In eval mode, outputs should be identical
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_get_model_factory(self):
        """Test model factory function."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = get_model(
            name="graph_transformer",
            num_features=8,
            num_classes=1,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            task="graph",
        )

        out = model(data)
        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_get_model_factory_node_task(self):
        """Test model factory function for node task."""
        data = create_test_graph(num_nodes=10, node_dim=8)

        model = get_model(
            name="graph_transformer",
            num_features=8,
            num_classes=5,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            task="node",
        )

        out = model(data)
        assert out.shape == (10, 5)
        assert not torch.isnan(out).any()

    def test_forward_single_node_graph(self):
        """Test forward pass with single node graph."""
        data = Data(
            x=torch.randn(1, 8),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
        )

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
        )
        model.eval()  # Use eval mode to avoid BatchNorm issues with single node

        out = model(data)
        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()

    def test_forward_empty_graph(self):
        """Test forward pass with empty graph (no edges)."""
        data = Data(
            x=torch.randn(5, 8),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=5,
        )

        model = GraphTransformerForGraphClassification(
            node_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            out_dim=1,
            node_pe_dim=0,
            use_relative_pe=False,
        )

        out = model(data)
        assert out.shape == (1, 1)
        assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
