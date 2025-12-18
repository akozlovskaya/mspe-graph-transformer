"""Tests for long-range evaluation framework."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.distance_metrics import (
    compute_shortest_path_distances,
    compute_shortest_path_distances_sparse,
    compute_landmark_distances,
    add_distance_info_to_data,
    get_node_pair_distances,
    compute_distance_histogram,
)
from src.evaluation.stratification import (
    create_distance_buckets,
    stratify_by_distance,
    DistanceStratifier,
    aggregate_stratified_results,
)
from src.evaluation.long_range import (
    compute_metrics_per_bucket,
    compute_relative_performance_drop,
    compute_area_under_distance_curve,
    find_effective_receptive_field,
    LongRangeEvaluator,
)
from src.evaluation.probes import (
    PathParityProbe,
    NodeCountingProbe,
    SyntheticLongRangeTask,
    create_probing_dataset,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def path_graph():
    """Create a simple path graph: 0 - 1 - 2 - 3 - 4."""
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ], dtype=torch.long)
    x = torch.randn(5, 16)
    y = torch.randn(5, 1)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def star_graph():
    """Create a star graph with center node 0."""
    # Center node 0 connected to 1, 2, 3, 4
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0, 0, 0, 0],
    ], dtype=torch.long)
    x = torch.randn(5, 16)
    y = torch.randn(5, 1)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def grid_graph():
    """Create a 3x3 grid graph."""
    # 0 - 1 - 2
    # |   |   |
    # 3 - 4 - 5
    # |   |   |
    # 6 - 7 - 8
    edges = [
        (0, 1), (1, 2),
        (3, 4), (4, 5),
        (6, 7), (7, 8),
        (0, 3), (3, 6),
        (1, 4), (4, 7),
        (2, 5), (5, 8),
    ]
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(9, 16)
    y = torch.randn(9, 1)
    return Data(x=x, edge_index=edge_index, y=y)


# ============================================================================
# Distance Metrics Tests
# ============================================================================

class TestDistanceMetrics:
    """Tests for distance computation."""

    def test_path_graph_distances(self, path_graph):
        """Test distances in path graph are correct."""
        distances = compute_shortest_path_distances(
            path_graph.edge_index, 5, max_distance=10
        )

        # Distance from 0 to other nodes
        assert distances[0, 0] == 0
        assert distances[0, 1] == 1
        assert distances[0, 2] == 2
        assert distances[0, 3] == 3
        assert distances[0, 4] == 4

        # Symmetry
        assert torch.equal(distances, distances.t())

    def test_star_graph_distances(self, star_graph):
        """Test distances in star graph."""
        distances = compute_shortest_path_distances(
            star_graph.edge_index, 5, max_distance=10
        )

        # All peripheral nodes are distance 1 from center
        for i in range(1, 5):
            assert distances[0, i] == 1
            assert distances[i, 0] == 1

        # Peripheral nodes are distance 2 from each other
        for i in range(1, 5):
            for j in range(1, 5):
                if i != j:
                    assert distances[i, j] == 2

    def test_max_distance_truncation(self, path_graph):
        """Test that distances beyond max_distance are truncated."""
        distances = compute_shortest_path_distances(
            path_graph.edge_index, 5, max_distance=2
        )

        # Distances beyond 2 should be -1 (unreachable within limit)
        assert distances[0, 3] == -1
        assert distances[0, 4] == -1

    def test_sparse_distances(self, path_graph):
        """Test sparse distance computation."""
        pair_indices, pair_distances = compute_shortest_path_distances_sparse(
            path_graph.edge_index, 5, max_distance=10
        )

        # Check we got all pairs
        assert pair_indices.size(1) == 5 * 4  # 5 nodes, 4 non-self pairs each

        # Check specific distance
        mask = (pair_indices[0] == 0) & (pair_indices[1] == 4)
        assert pair_distances[mask].item() == 4

    def test_landmark_distances(self, grid_graph):
        """Test landmark-based distance computation."""
        landmark_indices, landmark_distances = compute_landmark_distances(
            grid_graph.edge_index, 9, num_landmarks=3, max_distance=10
        )

        assert len(landmark_indices) == 3
        assert landmark_distances.shape == (9, 3)

        # Distance from landmark to itself should be 0
        for i, landmark in enumerate(landmark_indices):
            assert landmark_distances[landmark, i] == 0

    def test_add_distance_info(self, path_graph):
        """Test adding distance info to Data object."""
        data = add_distance_info_to_data(path_graph, max_distance=5, sparse=False)

        assert hasattr(data, "distance_matrix")
        assert data.distance_matrix.shape == (5, 5)
        assert data.max_distance_computed == 5

    def test_distance_histogram(self, grid_graph):
        """Test distance histogram computation."""
        data = add_distance_info_to_data(grid_graph, max_distance=10, sparse=False)
        histogram = compute_distance_histogram(data, max_distance=10)

        # Self-loops (distance 0)
        assert histogram[0] == 9

        # All pairs should be counted
        total_pairs = histogram.sum()
        assert total_pairs == 9 * 9


# ============================================================================
# Stratification Tests
# ============================================================================

class TestStratification:
    """Tests for distance-based stratification."""

    def test_create_buckets(self):
        """Test bucket creation."""
        buckets = create_distance_buckets(max_distance=5, bucket_size=2)

        assert (0, 1) in buckets
        assert (2, 3) in buckets
        assert (4, 5) in buckets
        assert (-1, -1) in buckets  # Unreachable

    def test_stratify_by_distance(self):
        """Test stratification logic."""
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        distances = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, -1, -1])

        buckets = [(0, 0), (1, 1), (2, 2), (3, 3), (-1, -1)]
        stratified = stratify_by_distance(predictions, targets, distances, buckets)

        assert stratified[(0, 0)]["count"] == 2
        assert stratified[(1, 1)]["count"] == 2
        assert stratified[(2, 2)]["count"] == 2
        assert stratified[(3, 3)]["count"] == 2
        assert stratified[(-1, -1)]["count"] == 2

    def test_stratifier_accumulation(self):
        """Test DistanceStratifier accumulates correctly."""
        stratifier = DistanceStratifier(max_distance=5, bucket_size=1)

        # Add multiple batches
        for _ in range(3):
            pred = torch.randn(10, 1)
            target = torch.randn(10, 1)
            dist = torch.randint(0, 5, (10,))
            stratifier.update(pred, target, dist)

        result = stratifier.compute()

        # Total count should be 30
        total = sum(data["count"] for data in result.values())
        assert total == 30

    def test_aggregate_stratified_results(self):
        """Test aggregation of multiple stratified results."""
        buckets = [(0, 1), (2, 3)]

        results1 = {
            (0, 1): {"predictions": torch.randn(5, 1), "targets": torch.randn(5, 1), "count": 5},
            (2, 3): {"predictions": torch.randn(3, 1), "targets": torch.randn(3, 1), "count": 3},
        }
        results2 = {
            (0, 1): {"predictions": torch.randn(4, 1), "targets": torch.randn(4, 1), "count": 4},
            (2, 3): {"predictions": torch.randn(2, 1), "targets": torch.randn(2, 1), "count": 2},
        }

        aggregated = aggregate_stratified_results([results1, results2])

        assert aggregated[(0, 1)]["count"] == 9
        assert aggregated[(2, 3)]["count"] == 5


# ============================================================================
# Long-Range Metrics Tests
# ============================================================================

class TestLongRangeMetrics:
    """Tests for long-range evaluation metrics."""

    def test_compute_metrics_per_bucket_regression(self):
        """Test metrics computation for regression."""
        stratified = {
            (0, 1): {
                "predictions": torch.tensor([[1.0], [2.0]]),
                "targets": torch.tensor([[1.1], [1.9]]),
                "count": 2,
            },
            (2, 3): {
                "predictions": torch.tensor([[3.0], [4.0]]),
                "targets": torch.tensor([[2.5], [4.5]]),
                "count": 2,
            },
        }

        metrics = compute_metrics_per_bucket(stratified, task_type="regression")

        assert "mae" in metrics[(0, 1)]
        assert "mse" in metrics[(0, 1)]
        assert metrics[(0, 1)]["mae"] < metrics[(2, 3)]["mae"]  # Closer predictions

    def test_compute_metrics_per_bucket_classification(self):
        """Test metrics computation for classification."""
        stratified = {
            (0, 1): {
                "predictions": torch.tensor([[0.9], [0.1]]),
                "targets": torch.tensor([[1], [0]]),
                "count": 2,
            },
        }

        metrics = compute_metrics_per_bucket(stratified, task_type="binary")

        assert "accuracy" in metrics[(0, 1)]
        assert metrics[(0, 1)]["accuracy"] == 1.0  # Perfect predictions

    def test_relative_performance_drop(self):
        """Test relative performance drop computation."""
        metrics_per_bucket = {
            (0, 0): {"metric": 0.9, "count": 10},
            (1, 1): {"metric": 0.8, "count": 10},
            (2, 2): {"metric": 0.5, "count": 10},
        }

        drops = compute_relative_performance_drop(
            metrics_per_bucket,
            higher_is_better=True,
        )

        assert drops[(0, 0)] == pytest.approx(0.0, abs=1e-6)  # Baseline
        assert drops[(1, 1)] > 0  # Performance dropped
        assert drops[(2, 2)] > drops[(1, 1)]  # Larger drop

    def test_area_under_distance_curve(self):
        """Test AUC computation."""
        # Constant performance
        metrics_constant = {
            (i, i): {"metric": 0.8, "count": 10}
            for i in range(5)
        }
        auc_constant = compute_area_under_distance_curve(
            metrics_constant, max_distance=4, normalize=True
        )

        # Decreasing performance
        metrics_decreasing = {
            (i, i): {"metric": 0.8 - i * 0.1, "count": 10}
            for i in range(5)
        }
        auc_decreasing = compute_area_under_distance_curve(
            metrics_decreasing, max_distance=4, normalize=True
        )

        # Constant should have higher AUC
        assert auc_constant > auc_decreasing

    def test_effective_receptive_field(self):
        """Test ERF computation."""
        metrics_per_bucket = {
            (0, 0): {"metric": 1.0, "count": 10},
            (1, 1): {"metric": 0.8, "count": 10},
            (2, 2): {"metric": 0.6, "count": 10},
            (3, 3): {"metric": 0.4, "count": 10},  # Below 50%
            (4, 4): {"metric": 0.2, "count": 10},
        }

        erf = find_effective_receptive_field(
            metrics_per_bucket,
            threshold=0.5,
            relative_to_baseline=True,
        )

        # Should be 2 (last bucket with metric >= 0.5)
        assert erf == 2


# ============================================================================
# LongRangeEvaluator Tests
# ============================================================================

class TestLongRangeEvaluator:
    """Tests for LongRangeEvaluator class."""

    def test_evaluator_basic(self):
        """Test basic evaluator functionality."""
        evaluator = LongRangeEvaluator(
            max_distance=5,
            bucket_size=1,
            task_type="regression",
        )

        # Add some data
        for _ in range(3):
            pred = torch.randn(10, 1)
            target = torch.randn(10, 1)
            dist = torch.randint(0, 5, (10,))
            evaluator.update(pred, target, dist)

        results = evaluator.compute()

        assert "metrics_per_bucket" in results
        assert "relative_drops" in results
        assert "auc" in results
        assert "effective_receptive_field" in results

    def test_evaluator_summary(self):
        """Test evaluator summary generation."""
        evaluator = LongRangeEvaluator(max_distance=5, task_type="regression")

        pred = torch.randn(50, 1)
        target = torch.randn(50, 1)
        dist = torch.randint(0, 5, (50,))
        evaluator.update(pred, target, dist)

        summary = evaluator.get_summary()

        assert "auc" in summary
        assert "effective_receptive_field" in summary

    def test_evaluator_reset(self):
        """Test evaluator reset."""
        evaluator = LongRangeEvaluator(max_distance=5)

        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        dist = torch.randint(0, 5, (10,))
        evaluator.update(pred, target, dist)

        evaluator.reset()
        results = evaluator.compute()

        # All buckets should be empty after reset
        total = sum(
            results["metrics_per_bucket"].get(b, {}).get("count", 0)
            for b in results["metrics_per_bucket"]
        )
        assert total == 0


# ============================================================================
# Probes Tests
# ============================================================================

class TestProbes:
    """Tests for synthetic probing tasks."""

    def test_path_parity_probe(self):
        """Test path parity probe generates valid data."""
        probe = PathParityProbe(path_length=10)
        data = probe.generate_path_graph(seed=42)

        assert data.num_nodes == 10
        assert hasattr(data, "y")
        assert hasattr(data, "distances")

        # Verify parity is correct
        running_parity = 0
        for i in range(10):
            running_parity ^= int(data.x[i].item())
            assert data.y[i].item() == running_parity

    def test_path_parity_batch(self):
        """Test batch generation."""
        probe = PathParityProbe(path_length=5)
        batch = probe.generate_batch(batch_size=10, base_seed=0)

        assert len(batch) == 10
        for data in batch:
            assert data.num_nodes == 5

    def test_node_counting_probe(self, grid_graph):
        """Test node counting probe."""
        probe = NodeCountingProbe(max_hops=3)
        counts, distances = probe.generate_task(grid_graph)

        assert counts.shape == (9, 4)  # 9 nodes, 4 hop levels (0-3)

        # At hop 0, only the node itself
        assert (counts[:, 0] == 1).all()

        # Counts should be non-decreasing with hops
        for i in range(9):
            for h in range(3):
                assert counts[i, h] <= counts[i, h + 1]

    def test_synthetic_long_range_task(self):
        """Test synthetic long-range task generation."""
        task = SyntheticLongRangeTask(
            graph_size=50,
            signal_distance=5,
            noise_level=0.1,
        )
        data = task.generate(seed=42)

        assert data.num_nodes == 50
        assert hasattr(data, "x")
        assert hasattr(data, "y")
        assert hasattr(data, "distances")

    def test_create_probing_dataset(self):
        """Test probing dataset creation."""
        dataset = create_probing_dataset(
            probe_type="parity",
            num_samples=5,
            path_length=8,
        )

        assert len(dataset) == 5
        for data in dataset:
            assert data.num_nodes == 8


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_evaluation_pipeline(self, grid_graph):
        """Test full evaluation pipeline on grid graph."""
        # Add distance info
        data = add_distance_info_to_data(grid_graph, max_distance=10, sparse=False)

        # Create evaluator
        evaluator = LongRangeEvaluator(
            max_distance=5,
            bucket_size=1,
            task_type="regression",
        )

        # Simulate predictions
        pred = data.y + torch.randn_like(data.y) * 0.1
        distances = data.distance_matrix[0]  # From node 0

        evaluator.update(pred, data.y, distances)
        results = evaluator.compute()

        assert results["metrics_per_bucket"][(0, 0)]["count"] == 1  # Node 0 itself

    def test_no_gradients_during_evaluation(self, path_graph):
        """Test that no gradients are computed during evaluation."""
        evaluator = LongRangeEvaluator(max_distance=5, task_type="regression")

        pred = torch.randn(5, 1, requires_grad=True)
        target = torch.randn(5, 1)
        dist = torch.tensor([0, 1, 2, 3, 4])

        evaluator.update(pred, target, dist)
        results = evaluator.compute()

        # No gradients should be stored in results
        for bucket, data in results["metrics_per_bucket"].items():
            if data["count"] > 0:
                assert not data.get("predictions", torch.tensor([])).requires_grad

    def test_batch_graph_compatibility(self):
        """Test evaluation with batched graphs."""
        # Create batch of graphs
        graphs = []
        for i in range(4):
            num_nodes = 5 + i
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            x = torch.randn(num_nodes, 16)
            y = torch.randn(num_nodes, 1)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        batch = Batch.from_data_list(graphs)

        evaluator = LongRangeEvaluator(max_distance=5, task_type="regression")

        # Use batch info
        pred = torch.randn(batch.num_nodes, 1)
        distances = torch.randint(0, 5, (batch.num_nodes,))

        evaluator.update(pred, batch.y, distances)
        results = evaluator.compute()

        total = sum(
            d["count"] for d in results["metrics_per_bucket"].values()
        )
        assert total == batch.num_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

