"""Integration tests for positional encoding modules."""

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pe.node import LapPE, RWSE, HKS, RolePE
from src.pe.relative import SPDBuckets, DiffusionPE, EffectiveResistancePE
from src.dataset.transforms import ApplyNodePE, ApplyRelativePE, CompositeTransform
from src.models import GraphTransformerForGraphClassification


def create_test_graph(num_nodes: int = 10, seed: int = 42) -> Data:
    """Create a simple test graph."""
    torch.manual_seed(seed)

    # Create ring graph
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return Data(
        x=torch.randn(num_nodes, 8),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


class TestPEIntegration:
    """Integration tests for PE modules."""

    def test_node_pe_combinations(self):
        """Test combining different node PE types."""
        data = create_test_graph(num_nodes=10)

        # Test LapPE + RWSE
        lappe = LapPE(dim=16, k=8, sign_invariant=True)
        rwse = RWSE(dim=16, scales=[1, 2, 4, 8])

        pe1 = lappe(data)
        pe2 = rwse(data)

        # Combine manually
        combined = torch.cat([pe1, pe2], dim=1)

        assert combined.shape == (10, 32)  # 16 + 16
        assert not torch.isnan(combined).any()

    def test_node_pe_with_model(self):
        """Test node PE integration with model."""
        data = create_test_graph(num_nodes=10)
        
        # Compute node PE
        pe = LapPE(dim=16, k=8, sign_invariant=True)
        data.node_pe = pe(data)

        # Create model with node PE
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

    def test_relative_pe_with_model(self):
        """Test relative PE integration with model."""
        data = create_test_graph(num_nodes=10)

        # Compute relative PE
        pe = SPDBuckets(num_buckets=16, max_distance=10, use_one_hot=True)
        edge_index_pe, edge_attr_pe = pe(data)

        data.edge_pe_index = edge_index_pe
        data.edge_pe = edge_attr_pe

        # Create model with relative PE
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

    def test_both_pe_with_model(self):
        """Test both node and relative PE with model."""
        data = create_test_graph(num_nodes=10)

        # Compute node PE
        node_pe = LapPE(dim=16, k=8, sign_invariant=True)
        data.node_pe = node_pe(data)

        # Compute relative PE
        rel_pe = SPDBuckets(num_buckets=16, max_distance=10, use_one_hot=True)
        edge_index_pe, edge_attr_pe = rel_pe(data)
        data.edge_pe_index = edge_index_pe
        data.edge_pe = edge_attr_pe

        # Create model with both PE
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

    def test_apply_node_pe_transform(self):
        """Test ApplyNodePE transform."""
        data = create_test_graph(num_nodes=10)

        transform = ApplyNodePE(
            node_pe_config={
                "enabled": True,
                "types": ["lap_pe"],
                "dim": 16,
                "scales": [8],
                "sign_invariant": True,
            }
        )

        transformed = transform(data)

        assert hasattr(transformed, "node_pe")
        assert transformed.node_pe.shape == (10, 16)
        assert not torch.isnan(transformed.node_pe).any()

    def test_apply_relative_pe_transform(self):
        """Test ApplyRelativePE transform."""
        data = create_test_graph(num_nodes=10)

        transform = ApplyRelativePE(
            relative_pe_config={
                "enabled": True,
                "types": ["spd"],
                "max_distance": 10,
                "num_buckets": 16,
            }
        )

        transformed = transform(data)

        assert hasattr(transformed, "edge_pe_index")
        assert hasattr(transformed, "edge_pe")
        assert transformed.edge_pe.shape[1] == 16
        assert not torch.isnan(transformed.edge_pe).any()

    def test_composite_transform(self):
        """Test CompositeTransform with both PE types."""
        data = create_test_graph(num_nodes=10)

        transform = CompositeTransform(
            pe_config={
                "node": {
                    "enabled": True,
                    "types": ["lap_pe"],
                    "dim": 16,
                    "scales": [8],
                    "sign_invariant": True,
                },
                "relative": {
                    "enabled": True,
                    "types": ["spd"],
                    "max_distance": 10,
                    "num_buckets": 16,
                },
            }
        )

        transformed = transform(data)

        assert hasattr(transformed, "node_pe")
        assert hasattr(transformed, "edge_pe_index")
        assert hasattr(transformed, "edge_pe")
        assert transformed.node_pe.shape == (10, 16)
        assert not torch.isnan(transformed.node_pe).any()
        assert not torch.isnan(transformed.edge_pe).any()

    def test_multi_scale_node_pe(self):
        """Test multi-scale node PE."""
        data = create_test_graph(num_nodes=10)

        # Multi-scale LapPE
        pe = LapPE(dim=32, scales=[4, 8, 16], sign_invariant=True)
        node_pe = pe(data)

        assert node_pe.shape[0] == 10
        assert node_pe.shape[1] >= 32  # May be larger due to concatenation
        assert not torch.isnan(node_pe).any()

        # Multi-scale RWSE
        pe2 = RWSE(dim=32, scales=[1, 2, 4, 8, 16])
        node_pe2 = pe2(data)

        assert node_pe2.shape == (10, 32)
        assert not torch.isnan(node_pe2).any()

    def test_different_node_pe_types(self):
        """Test different node PE types produce valid outputs."""
        data = create_test_graph(num_nodes=10)

        pe_types = [
            ("lap_pe", LapPE(dim=16, k=8, sign_invariant=True)),
            ("rwse", RWSE(dim=16, scales=[1, 2, 4, 8])),
            ("hks", HKS(dim=16, scales=[0.1, 1.0, 10.0], k_eigenvectors=20)),
            ("role", RolePE(dim=8, features=["degree", "clustering", "core"])),
        ]

        for name, pe in pe_types:
            node_pe = pe(data)
            assert node_pe.shape[0] == 10, f"{name} failed shape check"
            assert not torch.isnan(node_pe).any(), f"{name} has NaN values"
            assert not torch.isinf(node_pe).any(), f"{name} has Inf values"

    def test_different_relative_pe_types(self):
        """Test different relative PE types produce valid outputs."""
        data = create_test_graph(num_nodes=10)

        pe_types = [
            ("spd", SPDBuckets(num_buckets=16, max_distance=10, use_one_hot=True)),
        ]

        for name, pe in pe_types:
            edge_index_pe, edge_attr_pe = pe(data)
            assert edge_index_pe.shape[0] == 2, f"{name} failed edge_index shape"
            assert edge_attr_pe.shape[1] == 16, f"{name} failed edge_attr shape"
            assert not torch.isnan(edge_attr_pe).any(), f"{name} has NaN values"

    def test_pe_caching(self):
        """Test PE caching functionality."""
        data = create_test_graph(num_nodes=10)

        pe = LapPE(dim=16, k=8, sign_invariant=True, cache=True)
        
        # First call
        node_pe1 = pe(data)
        
        # Second call should use cache
        node_pe2 = pe(data)

        assert torch.allclose(node_pe1, node_pe2, atol=1e-6)
        assert hasattr(data, "node_pe")

    def test_pe_with_batch(self):
        """Test PE computation with batched graphs."""
        graphs = []
        for i in range(3):
            graphs.append(create_test_graph(num_nodes=8, seed=42 + i))

        batch = Batch.from_data_list(graphs)

        # Node PE
        pe = LapPE(dim=16, k=8, sign_invariant=True)
        node_pe = pe(batch)

        assert node_pe.shape[0] == 24  # 3 graphs * 8 nodes
        assert node_pe.shape[1] == 16
        assert not torch.isnan(node_pe).any()

    def test_pe_dimension_consistency(self):
        """Test that PE dimensions are consistent across different graph sizes."""
        for num_nodes in [5, 10, 20, 50]:
            data = create_test_graph(num_nodes=num_nodes)

            pe = LapPE(dim=16, k=8, sign_invariant=True)
            node_pe = pe(data)

            assert node_pe.shape == (num_nodes, 16)
            assert not torch.isnan(node_pe).any()

    def test_pe_with_disconnected_graph(self):
        """Test PE computation on disconnected graph."""
        # Create graph with two components
        edge_list = []
        # First component: 5 nodes in ring
        for i in range(5):
            edge_list.append([i, (i + 1) % 5])
        # Second component: 5 nodes in ring
        for i in range(5, 10):
            edge_list.append([i, (i + 1) % 10 if (i + 1) < 10 else 5])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)

        data = Data(
            x=torch.randn(10, 8),
            edge_index=edge_index,
            num_nodes=10,
        )

        pe = LapPE(dim=16, k=8, sign_invariant=True)
        node_pe = pe(data)

        assert node_pe.shape == (10, 16)
        assert not torch.isnan(node_pe).any()

    def test_pe_normalization(self):
        """Test PE normalization options."""
        data = create_test_graph(num_nodes=10)

        for normalization in ["graph", "node", None]:
            pe = LapPE(
                dim=16,
                k=8,
                sign_invariant=True,
                normalization=normalization,
            )
            node_pe = pe(data)

            assert node_pe.shape == (10, 16)
            assert not torch.isnan(node_pe).any()

    def test_pe_sign_invariance_methods(self):
        """Test different sign invariance methods."""
        data = create_test_graph(num_nodes=10)

        for method in ["abs", "flip", "square"]:
            pe = LapPE(
                dim=16,
                k=8,
                sign_invariant=True,
                sign_invariance_method=method,
            )
            node_pe = pe(data)

            assert node_pe.shape[0] == 10
            assert not torch.isnan(node_pe).any()
            assert not torch.isinf(node_pe).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
