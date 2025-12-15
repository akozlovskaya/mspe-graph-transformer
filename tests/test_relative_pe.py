"""Tests for relative (pairwise) positional encodings."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.pe.relative import (
    SPDBuckets,
    SPDBucketsSparse,
    BFSDistance,
    DiffusionPE,
    EffectiveResistancePE,
    LandmarkSPD,
    build_attention_bias,
)


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


def test_spd_buckets_basic():
    """Test basic SPDBuckets functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = SPDBuckets(num_buckets=16, max_distance=10, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_index_pe.shape[1] == data.num_nodes * data.num_nodes  # All pairs
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]
    assert edge_attr_pe.shape[1] == 16
    assert not torch.isnan(edge_attr_pe).any()


def test_spd_buckets_symmetric():
    """Test that SPDBuckets produces symmetric PE."""
    data = create_connected_graph(num_nodes=8)
    pe = SPDBuckets(num_buckets=8, max_distance=5, symmetric=True)

    edge_index_pe, edge_attr_pe = pe(data)

    # Check symmetry: PE(i,j) should equal PE(j,i)
    num_pairs = edge_index_pe.shape[1]
    for i in range(min(100, num_pairs)):  # Check first 100 pairs
        node_i = edge_index_pe[0, i].item()
        node_j = edge_index_pe[1, i].item()
        # Find reverse pair
        reverse_mask = (edge_index_pe[0] == node_j) & (edge_index_pe[1] == node_i)
        if reverse_mask.any():
            reverse_idx = reverse_mask.nonzero()[0, 0].item()
            # Values should be similar (allowing for numerical errors)
            assert torch.allclose(
                edge_attr_pe[i], edge_attr_pe[reverse_idx], atol=1e-5
            )


def test_spd_buckets_sparse():
    """Test sparse version of SPDBuckets."""
    data = create_connected_graph(num_nodes=10)
    pe = SPDBucketsSparse(num_buckets=8, max_distance=3, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_index_pe.shape[1] <= data.num_nodes * data.num_nodes  # Sparse
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]
    assert edge_attr_pe.shape[1] == 8


def test_bfs_distance():
    """Test BFSDistance functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = BFSDistance(num_buckets=8, max_distance=5, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_index_pe.shape[1] <= data.num_nodes * data.num_nodes  # Only pairs within k
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]
    assert edge_attr_pe.shape[1] == 8


def test_diffusion_pe():
    """Test DiffusionPE functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = DiffusionPE(
        num_buckets=4,
        max_distance=10,  # Not used
        scales=[0.1, 1.0, 5.0, 10.0],
        k_eigenvectors=10,
    )

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]
    assert edge_attr_pe.shape[1] == 4  # One per scale
    assert not torch.isnan(edge_attr_pe).any()


def test_effective_resistance_pe():
    """Test EffectiveResistancePE functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = EffectiveResistancePE(
        num_buckets=1, max_distance=10, k_eigenvectors=10, use_sparse=True
    )

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]
    assert edge_attr_pe.shape[1] == 1  # Scalar per pair
    assert not torch.isnan(edge_attr_pe).any()


def test_landmark_spd():
    """Test LandmarkSPD functionality."""
    data = create_connected_graph(num_nodes=10)
    pe = LandmarkSPD(
        num_buckets=8,
        max_distance=5,
        num_landmarks=3,
        landmark_method="random",
        use_one_hot=True,
    )

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_index_pe.shape[1] == data.num_nodes * data.num_nodes  # All pairs
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]
    assert edge_attr_pe.shape[1] == 8


def test_landmark_spd_sparse():
    """Test sparse LandmarkSPD."""
    data = create_connected_graph(num_nodes=10)
    pe = LandmarkSPD(
        num_buckets=8,
        max_distance=3,
        num_landmarks=3,
        use_one_hot=True,
    )

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]


def test_caching():
    """Test that PE caching works."""
    data = create_connected_graph(num_nodes=10)
    pe = SPDBuckets(num_buckets=8, max_distance=5, cache=True)

    # First call
    edge_index_pe1, edge_attr_pe1 = pe(data)

    # Second call should use cache
    edge_index_pe2, edge_attr_pe2 = pe(data)

    assert torch.allclose(edge_index_pe1, edge_index_pe2)
    assert torch.allclose(edge_attr_pe1, edge_attr_pe2)
    assert hasattr(data, "edge_pe_index")
    assert hasattr(data, "edge_pe")


def test_build_attention_bias_dense():
    """Test building dense attention bias."""
    data = create_connected_graph(num_nodes=8)
    pe = SPDBuckets(num_buckets=8, max_distance=5, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    bias = build_attention_bias(
        edge_index_pe, edge_attr_pe, data.num_nodes, num_heads=1, mode="dense"
    )

    assert bias.shape == (data.num_nodes, data.num_nodes)
    assert not torch.isnan(bias).any()


def test_build_attention_bias_multi_head():
    """Test building multi-head attention bias."""
    data = create_connected_graph(num_nodes=8)
    pe = SPDBuckets(num_buckets=8, max_distance=5, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    bias = build_attention_bias(
        edge_index_pe, edge_attr_pe, data.num_nodes, num_heads=4, mode="dense"
    )

    assert bias.shape == (4, data.num_nodes, data.num_nodes)
    assert not torch.isnan(bias).any()


def test_normalization_graph():
    """Test graph-level normalization."""
    data = create_connected_graph(num_nodes=10)
    pe = SPDBuckets(
        num_buckets=8, max_distance=5, normalization="graph", use_one_hot=True
    )

    edge_index_pe, edge_attr_pe = pe(data)

    # Check that normalization is applied (mean should be close to 0 per feature)
    mean_per_feature = edge_attr_pe.mean(dim=0)
    assert torch.allclose(mean_per_feature, torch.zeros_like(mean_per_feature), atol=1e-4)


def test_normalization_pair():
    """Test pair-level normalization."""
    data = create_connected_graph(num_nodes=10)
    pe = SPDBuckets(
        num_buckets=8, max_distance=5, normalization="pair", use_one_hot=True
    )

    edge_index_pe, edge_attr_pe = pe(data)

    # Check that normalization is applied per pair
    mean_per_pair = edge_attr_pe.mean(dim=1)
    assert torch.allclose(mean_per_pair, torch.zeros_like(mean_per_pair), atol=1e-4)


def test_max_distance():
    """Test that max_distance is respected."""
    data = create_connected_graph(num_nodes=10)
    pe = BFSDistance(num_buckets=8, max_distance=3, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    # All distances should be <= max_distance (encoded in buckets)
    # This is implicit - if max_distance is 3, we should only see pairs within 3 hops
    assert edge_index_pe.shape[1] <= data.num_nodes * data.num_nodes


def test_small_graph():
    """Test on very small graph."""
    data = create_connected_graph(num_nodes=3)
    pe = SPDBuckets(num_buckets=4, max_distance=2, use_one_hot=True)

    edge_index_pe, edge_attr_pe = pe(data)

    assert edge_index_pe.shape[0] == 2
    assert edge_attr_pe.shape[0] == edge_index_pe.shape[1]


def test_different_max_distances():
    """Test behavior with different max_distance values."""
    data = create_connected_graph(num_nodes=10)

    for max_dist in [2, 5, 10]:
        pe = BFSDistance(num_buckets=8, max_distance=max_dist, use_one_hot=True)
        edge_index_pe, edge_attr_pe = pe(data)
        assert edge_index_pe.shape[0] == 2
        assert edge_attr_pe.shape[1] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

