"""Test script for synthetic datasets."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataset

def test_synthetic_dataset():
    """Test synthetic dataset creation and loading."""
    
    # Test Task A: Pairwise Distance Classification
    print("Testing Task A: Pairwise Distance Classification")
    dataset = get_dataset(
        name="synthetic_erdos_renyi",
        root="./data",
        num_graphs=10,
        graph_params={"n": 20, "p": 0.3},
        task="pairwise_distance_classification",
        task_params={"distance_threshold": 3, "num_pairs": 10},
        seed=42,
        use_node_features=True,
    )
    
    train, val, test = dataset.get_splits()
    print(f"  Train: {len(train)} graphs")
    print(f"  Val: {len(val)} graphs")
    print(f"  Test: {len(test)} graphs")
    print(f"  Num features: {dataset.num_features}")
    print(f"  Num classes: {dataset.num_classes}")
    
    # Check first graph
    sample = train[0]
    print(f"  Sample graph: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
    print(f"  Sample target: {sample.y}")
    if hasattr(sample, "pair_sources"):
        print(f"  Sample pairs: {len(sample.pair_sources)} pairs")
    
    # Test Task C: Structural Role
    print("\nTesting Task C: Structural Role")
    dataset = get_dataset(
        name="synthetic_sbm",
        root="./data",
        num_graphs=10,
        graph_params={"n": 30, "n_blocks": 3, "p_in": 0.3, "p_out": 0.05},
        task="structural_role",
        seed=42,
        use_node_features=False,
    )
    
    train, val, test = dataset.get_splits()
    print(f"  Train: {len(train)} graphs")
    print(f"  Num features: {dataset.num_features}")
    print(f"  Num classes: {dataset.num_classes}")
    
    sample = train[0]
    print(f"  Sample graph: {sample.num_nodes} nodes")
    print(f"  Sample target shape: {sample.y.shape}")
    
    # Test SBM generation
    print("\nTesting SBM generation")
    dataset = get_dataset(
        name="synthetic_sbm",
        root="./data",
        num_graphs=5,
        graph_params={"n": 20, "n_blocks": 2, "p_in": 0.5, "p_out": 0.1},
        seed=42,
    )
    
    train, val, test = dataset.get_splits()
    sample = train[0]
    print(f"  Sample graph: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
    
    # Test Random Geometric
    print("\nTesting Random Geometric Graph")
    dataset = get_dataset(
        name="synthetic_random_geometric",
        root="./data",
        num_graphs=5,
        graph_params={"n": 20, "radius": 0.3, "dim": 2},
        seed=42,
    )
    
    train, val, test = dataset.get_splits()
    sample = train[0]
    print(f"  Sample graph: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_synthetic_dataset()

