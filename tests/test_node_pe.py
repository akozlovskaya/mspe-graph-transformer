"""Tests for node positional encodings."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.pe.node import LapPE, RWSE, HKS, RolePE, BaseNodePE


def create_connected_graph(num_nodes: int = 10, seed: int = 42) -> Data:
    """Create a simple connected graph."""
    torch.manual_seed(seed)
    # Create a ring graph (connected)
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    
    return Data(
        x=torch.randn(num_nodes, 5),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


def create_disconnected_graph(num_nodes: int = 10, seed: int = 42) -> Data:
    """Create a disconnected graph with two components."""
    torch.manual_seed(seed)
    
    # First component: 5 nodes in a ring
    edge_list = []
    for i in range(5):
        edge_list.append([i, (i + 1) % 5])
    
    # Second component: 5 nodes in a ring
    for i in range(5, num_nodes):
        edge_list.append([i, (i + 1) % num_nodes if (i + 1) < num_nodes else 5])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    
    return Data(
        x=torch.randn(num_nodes, 5),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


def test_lappe_basic():
    """Test basic LapPE functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = LapPE(dim=16, k=8, sign_invariant=True)
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)
    assert not torch.isnan(node_pe).any()
    assert not torch.isinf(node_pe).any()


def test_lappe_disconnected():
    """Test LapPE on disconnected graph."""
    data = create_disconnected_graph(num_nodes=10)
    pe = LapPE(dim=16, k=8, sign_invariant=True)
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)
    assert not torch.isnan(node_pe).any()


def test_lappe_multi_scale():
    """Test multi-scale LapPE."""
    data = create_connected_graph(num_nodes=10)
    pe = LapPE(dim=32, scales=[4, 8, 16], sign_invariant=True)
    
    node_pe = pe(data)
    
    # Multi-scale should produce larger dimension
    assert node_pe.shape[0] == data.num_nodes
    assert node_pe.shape[1] >= 32  # May be larger due to concat


def test_lappe_sign_invariance():
    """Test sign-invariance in LapPE."""
    data = create_connected_graph(num_nodes=10)
    
    pe_abs = LapPE(dim=16, k=8, sign_invariant=True, sign_invariance_method="abs")
    pe_flip = LapPE(dim=32, k=8, sign_invariant=True, sign_invariance_method="flip")
    
    node_pe_abs = pe_abs(data)
    node_pe_flip = pe_flip(data)
    
    assert node_pe_abs.shape == (data.num_nodes, 16)
    assert node_pe_flip.shape[0] == data.num_nodes
    # Flip method doubles the dimension
    assert node_pe_flip.shape[1] >= 32


def test_rwse_basic():
    """Test basic RWSE functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=[1, 2, 4, 8])
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)
    assert not torch.isnan(node_pe).any()
    assert not torch.isinf(node_pe).any()


def test_rwse_disconnected():
    """Test RWSE on disconnected graph."""
    data = create_disconnected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=[1, 2, 4])
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)
    assert not torch.isnan(node_pe).any()


def test_rwse_default_scales():
    """Test RWSE with default log-spaced scales."""
    data = create_connected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=None, log_spaced=True)
    
    node_pe = pe(data)
    
    assert node_pe.shape[0] == data.num_nodes
    assert node_pe.shape[1] > 0


def test_hks_basic():
    """Test basic HKS functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = HKS(dim=16, scales=[0.1, 1.0, 10.0], k_eigenvectors=10)
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)
    assert not torch.isnan(node_pe).any()
    assert not torch.isinf(node_pe).any()


def test_hks_disconnected():
    """Test HKS on disconnected graph."""
    data = create_disconnected_graph(num_nodes=10)
    pe = HKS(dim=16, scales=[0.1, 1.0], k_eigenvectors=10)
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)
    assert not torch.isnan(node_pe).any()


def test_hks_default_scales():
    """Test HKS with default log-spaced scales."""
    data = create_connected_graph(num_nodes=10)
    pe = HKS(dim=16, scales=None, k_eigenvectors=10)
    
    node_pe = pe(data)
    
    assert node_pe.shape[0] == data.num_nodes
    assert node_pe.shape[1] > 0


def test_role_pe_basic():
    """Test basic RolePE functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = RolePE(dim=8, features=["degree", "clustering", "core"])
    
    node_pe = pe(data)
    
    assert node_pe.shape[0] == data.num_nodes
    assert node_pe.shape[1] >= 3  # At least degree, clustering, core
    assert not torch.isnan(node_pe).any()
    assert not torch.isinf(node_pe).any()


def test_role_pe_features():
    """Test RolePE with different features."""
    data = create_connected_graph(num_nodes=10)
    
    pe_degree = RolePE(dim=4, features=["degree"])
    pe_full = RolePE(dim=8, features=["degree", "clustering", "core"])
    
    node_pe_degree = pe_degree(data)
    node_pe_full = pe_full(data)
    
    assert node_pe_degree.shape[0] == data.num_nodes
    assert node_pe_full.shape[0] == data.num_nodes
    assert node_pe_full.shape[1] >= node_pe_degree.shape[1]


def test_normalization_graph():
    """Test graph-level normalization."""
    data = create_connected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=[1, 2, 4], normalization="graph")
    
    node_pe = pe(data)
    
    # Check that normalization is applied (mean should be close to 0 per feature)
    mean_per_feature = node_pe.mean(dim=0)
    assert torch.allclose(mean_per_feature, torch.zeros_like(mean_per_feature), atol=1e-5)


def test_normalization_node():
    """Test node-level normalization."""
    data = create_connected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=[1, 2, 4], normalization="node")
    
    node_pe = pe(data)
    
    # Check that normalization is applied per node
    mean_per_node = node_pe.mean(dim=1)
    std_per_node = node_pe.std(dim=1)
    # Mean should be close to 0, std should be close to 1
    assert torch.allclose(mean_per_node, torch.zeros_like(mean_per_node), atol=1e-5)
    assert torch.allclose(std_per_node, torch.ones_like(std_per_node), atol=1e-1)


def test_normalization_none():
    """Test no normalization."""
    data = create_connected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=[1, 2, 4], normalization=None)
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 16)


def test_caching():
    """Test PE caching."""
    data = create_connected_graph(num_nodes=10)
    pe = RWSE(dim=16, scales=[1, 2, 4], cache=True)
    
    # First call
    node_pe1 = pe(data)
    
    # Second call should use cache
    node_pe2 = pe(data)
    
    assert torch.allclose(node_pe1, node_pe2)
    assert hasattr(data, "node_pe")
    assert torch.allclose(data.node_pe, node_pe1)


def test_dimension_projection():
    """Test dimension projection when PE dim doesn't match target."""
    data = create_connected_graph(num_nodes=10)
    
    # Request dim=8 but PE will produce different dimension
    pe = LapPE(dim=8, k=4, sign_invariant=False)
    
    node_pe = pe(data)
    
    assert node_pe.shape == (data.num_nodes, 8)


def test_small_graph():
    """Test on very small graph."""
    data = create_connected_graph(num_nodes=3)
    
    pe = RWSE(dim=4, scales=[1, 2])
    node_pe = pe(data)
    
    assert node_pe.shape == (3, 4)
    assert not torch.isnan(node_pe).any()


def test_large_graph():
    """Test on larger graph."""
    data = create_connected_graph(num_nodes=50)
    
    pe = RWSE(dim=16, scales=[1, 2, 4, 8])
    node_pe = pe(data)
    
    assert node_pe.shape == (50, 16)
    assert not torch.isnan(node_pe).any()


def test_multi_scale_aggregation():
    """Test multi-scale aggregation."""
    data = create_connected_graph(num_nodes=10)
    
    # RWSE with multiple scales
    pe = RWSE(dim=32, scales=[1, 2, 4, 8, 16])
    node_pe = pe(data)
    
    assert node_pe.shape[0] == data.num_nodes
    # Should have dimension >= number of scales (since we concat)
    assert node_pe.shape[1] >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

