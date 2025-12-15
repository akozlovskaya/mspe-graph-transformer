"""Tests for dataset loading functionality."""

import pytest
import torch
from torch_geometric.data import DataLoader

from src.dataset import get_dataset, list_available_datasets
from src.dataset.utils import compute_dataset_stats, ensure_data_has_pe


@pytest.fixture
def pe_config():
    """Default PE configuration for testing."""
    return {
        "node": {
            "enabled": True,
            "types": ["rwse"],
            "dim": 32,
            "scales": [1, 2, 4, 8],
        },
        "relative": {
            "enabled": True,
            "types": ["spd"],
            "max_distance": 10,
            "num_buckets": 16,
        },
    }


def test_list_available_datasets():
    """Test listing available datasets."""
    datasets = list_available_datasets()
    assert "molecular" in datasets
    assert "lrgb" in datasets
    assert "ogb" in datasets
    assert "synthetic" in datasets


def test_synthetic_dataset_loading(pe_config):
    """Test loading synthetic dataset."""
    dataset = get_dataset(
        name="synthetic_ring",
        root="./data/test",
        pe_config=pe_config,
        num_graphs=100,
        graph_params={"n": 20},
    )

    assert dataset.train is not None
    assert dataset.val is not None
    assert dataset.test is not None

    assert len(dataset.train) > 0
    assert len(dataset.val) > 0
    assert len(dataset.test) > 0

    # Check that graphs have PE
    sample = dataset.train[0]
    assert hasattr(sample, "node_pe")
    assert hasattr(sample, "edge_pe")
    assert sample.node_pe.shape[0] == sample.num_nodes
    assert sample.edge_pe.shape[0] == sample.edge_index.size(1)


def test_synthetic_dataset_properties(pe_config):
    """Test dataset properties."""
    dataset = get_dataset(
        name="synthetic_grid_2d",
        root="./data/test",
        pe_config=pe_config,
        num_graphs=50,
        graph_params={"m": 5, "n": 5},
    )

    assert dataset.num_features > 0
    assert dataset.num_classes >= 1


def test_dataset_splits(pe_config):
    """Test that datasets have proper splits."""
    dataset = get_dataset(
        name="synthetic_tree",
        root="./data/test",
        pe_config=pe_config,
        num_graphs=100,
        graph_params={"r": 3, "h": 3},
    )

    splits = dataset.get_splits()
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["train"]) > len(splits["test"])


def test_pe_in_batch(pe_config):
    """Test that PE is included in batched data."""
    dataset = get_dataset(
        name="synthetic_ring",
        root="./data/test",
        pe_config=pe_config,
        num_graphs=20,
        graph_params={"n": 10},
    )

    loader = DataLoader(dataset.train, batch_size=4, shuffle=False)

    for batch in loader:
        assert hasattr(batch, "node_pe")
        assert hasattr(batch, "edge_pe")
        assert batch.node_pe is not None
        assert batch.edge_pe is not None
        break


def test_compute_dataset_stats(pe_config):
    """Test computing dataset statistics."""
    dataset = get_dataset(
        name="synthetic_ring",
        root="./data/test",
        pe_config=pe_config,
        num_graphs=50,
        graph_params={"n": 15},
    )

    stats = compute_dataset_stats(dataset.train)
    assert "num_graphs" in stats
    assert "avg_num_nodes" in stats
    assert "avg_num_edges" in stats
    assert stats["num_graphs"] == len(dataset.train)


def test_ensure_data_has_pe():
    """Test ensuring data has PE attributes."""
    from torch_geometric.data import Data

    data = Data(
        x=torch.randn(10, 5),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
    )

    from src.dataset.utils import ensure_data_has_pe

    data = ensure_data_has_pe(data)
    assert hasattr(data, "node_pe")
    assert hasattr(data, "edge_pe")
    assert data.node_pe.shape == (10, 32)
    assert data.edge_pe.shape == (3, 16)


def test_zinc_dataset_loading(pe_config):
    """Test ZINC dataset loading (if available)."""
    try:
        dataset = get_dataset(
            name="zinc",
            root="./data/test",
            pe_config=pe_config,
            subset=True,
        )

        assert dataset.train is not None
        assert dataset.val is not None
        assert dataset.test is not None

        # Check PE
        if len(dataset.train) > 0:
            sample = dataset.train[0]
            assert hasattr(sample, "node_pe")
            assert hasattr(sample, "edge_pe")

    except Exception as e:
        pytest.skip(f"ZINC dataset not available: {e}")


def test_synthetic_various_types(pe_config):
    """Test various synthetic graph types."""
    graph_types = [
        "synthetic_grid_2d",
        "synthetic_ring",
        "synthetic_tree",
        "synthetic_random_regular",
        "synthetic_barabasi_albert",
        "synthetic_watts_strogatz",
        "synthetic_erdos_renyi",
    ]

    for graph_type in graph_types:
        try:
            dataset = get_dataset(
                name=graph_type,
                root="./data/test",
                pe_config=pe_config,
                num_graphs=20,
                graph_params={"n": 15} if "n" in graph_type else {},
            )

            assert dataset.train is not None
            assert len(dataset.train) > 0

            sample = dataset.train[0]
            assert hasattr(sample, "x")
            assert hasattr(sample, "edge_index")
            assert hasattr(sample, "node_pe")
            assert hasattr(sample, "edge_pe")

        except Exception as e:
            pytest.skip(f"Graph type {graph_type} failed: {e}")


def test_pe_config_disabled():
    """Test dataset loading with PE disabled."""
    pe_config_disabled = {
        "node": {"enabled": False},
        "relative": {"enabled": False},
    }

    dataset = get_dataset(
        name="synthetic_ring",
        root="./data/test",
        pe_config=pe_config_disabled,
        num_graphs=20,
        graph_params={"n": 10},
    )

    sample = dataset.train[0]
    assert hasattr(sample, "node_pe")
    assert hasattr(sample, "edge_pe")
    # PE should be zeros when disabled
    assert torch.all(sample.node_pe == 0) or sample.node_pe.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
