"""Tests for profiling framework."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling.runtime import (
    RuntimeProfiler,
    RuntimeStats,
    benchmark_function,
    profile_forward,
    profile_backward,
)
from src.profiling.memory import (
    MemoryProfiler,
    MemoryStats,
    profile_memory_usage,
    reset_memory_stats,
    get_peak_memory,
    estimate_attention_memory,
)
from src.profiling.flops import (
    FLOPsEstimator,
    FLOPsEstimate,
    estimate_linear_flops,
    estimate_attention_flops,
    estimate_mpnn_flops,
    estimate_ffn_flops,
    estimate_model_flops,
)
from src.profiling.scaling import (
    ScalingResult,
    generate_scaling_graphs,
)
from src.profiling.utils import (
    get_hardware_info,
    get_model_info,
    ProfilingContext,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleGNN(nn.Module):
        def __init__(self, in_dim=16, hidden_dim=32, out_dim=1):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, out_dim)
            self.dropout = nn.Dropout(0.1)
            self.relu = nn.ReLU()

        def forward(self, batch):
            x = batch.x
            x = self.relu(self.lin1(x))
            x = self.dropout(x)
            x = self.lin2(x)
            if hasattr(batch, 'batch'):
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, batch.batch)
            else:
                x = x.mean(dim=0, keepdim=True)
            return x

    return SimpleGNN()


@pytest.fixture
def sample_batch():
    """Create a sample batch."""
    num_nodes = 100
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, 300))
    y = torch.randn(1)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


# ============================================================================
# Runtime Tests
# ============================================================================

class TestRuntime:
    """Tests for runtime profiling."""

    def test_benchmark_function(self):
        """Test basic function benchmarking."""
        def simple_fn():
            return torch.randn(100, 100) @ torch.randn(100, 100)

        stats = benchmark_function(simple_fn, num_runs=10, warmup_runs=2)

        assert isinstance(stats, RuntimeStats)
        assert stats.mean > 0
        assert stats.std >= 0
        assert stats.min > 0
        assert stats.min <= stats.mean <= stats.max
        assert stats.num_runs == 10

    def test_profile_forward(self, simple_model, sample_batch):
        """Test forward pass profiling."""
        stats = profile_forward(
            simple_model, sample_batch, num_runs=10, warmup_runs=2
        )

        assert stats.mean > 0
        assert stats.num_runs == 10

    def test_profile_backward(self, simple_model, sample_batch):
        """Test backward pass profiling."""
        stats = profile_backward(
            simple_model, sample_batch, num_runs=5, warmup_runs=1
        )

        assert stats.mean > 0
        assert stats.num_runs == 5

    def test_runtime_profiler(self, simple_model, sample_batch, device):
        """Test RuntimeProfiler class."""
        profiler = RuntimeProfiler(
            simple_model, device, num_runs=10, warmup_runs=2
        )

        forward_stats = profiler.profile_forward(sample_batch)
        assert forward_stats.mean > 0

        backward_stats = profiler.profile_backward(sample_batch)
        assert backward_stats.mean > 0

    def test_runtime_stats_to_dict(self):
        """Test RuntimeStats serialization."""
        stats = RuntimeStats(
            mean=10.5, std=1.2, min=8.0, max=15.0, num_runs=100
        )
        d = stats.to_dict()

        assert d["mean"] == 10.5
        assert d["std"] == 1.2
        assert d["unit"] == "ms"


# ============================================================================
# Memory Tests
# ============================================================================

class TestMemory:
    """Tests for memory profiling."""

    def test_reset_memory_stats(self):
        """Test memory stats reset."""
        reset_memory_stats()

        if torch.cuda.is_available():
            assert get_peak_memory() == 0 or get_peak_memory() < 1  # ~0 MB

    def test_profile_memory_usage(self, simple_model, sample_batch):
        """Test memory usage profiling."""
        def forward():
            return simple_model(sample_batch)

        stats = profile_memory_usage(forward)

        assert isinstance(stats, MemoryStats)
        # On CPU, GPU memory will be 0
        assert stats.peak_mb >= 0

    def test_memory_profiler(self, simple_model, sample_batch, device):
        """Test MemoryProfiler class."""
        profiler = MemoryProfiler(simple_model, device)

        forward_stats = profiler.profile_forward(sample_batch)
        assert forward_stats.peak_mb >= 0

        backward_stats = profiler.profile_backward(sample_batch)
        assert backward_stats.peak_mb >= 0

    def test_estimate_attention_memory(self):
        """Test attention memory estimation."""
        mem = estimate_attention_memory(
            num_nodes=100,
            num_heads=8,
            hidden_dim=256,
            batch_size=1,
        )

        assert mem > 0
        assert mem < 1000  # Should be reasonable

    def test_memory_stats_to_dict(self):
        """Test MemoryStats serialization."""
        stats = MemoryStats(
            peak_mb=100.5, allocated_mb=80.0, reserved_mb=120.0, device="cuda:0"
        )
        d = stats.to_dict()

        assert d["peak_mb"] == 100.5
        assert d["device"] == "cuda:0"


# ============================================================================
# FLOPs Tests
# ============================================================================

class TestFLOPs:
    """Tests for FLOPs estimation."""

    def test_estimate_linear_flops(self):
        """Test linear layer FLOPs estimation."""
        flops = estimate_linear_flops(
            in_features=256, out_features=512, batch_size=1, sequence_length=100
        )

        # 2 * 100 * 256 * 512 + bias
        expected_approx = 2 * 100 * 256 * 512
        assert flops > expected_approx * 0.9
        assert flops < expected_approx * 1.1

    def test_estimate_attention_flops(self):
        """Test attention FLOPs estimation."""
        flops = estimate_attention_flops(
            num_nodes=100,
            hidden_dim=256,
            num_heads=8,
            batch_size=1,
        )

        assert "total" in flops
        assert flops["total"] > 0
        assert "qkv_projection" in flops
        assert "attention_scores" in flops
        assert "softmax" in flops

    def test_estimate_mpnn_flops(self):
        """Test MPNN FLOPs estimation."""
        for mpnn_type in ["gin", "gcn", "gat"]:
            flops = estimate_mpnn_flops(
                num_nodes=100,
                num_edges=300,
                in_features=256,
                out_features=256,
                mpnn_type=mpnn_type,
            )

            assert "total" in flops
            assert flops["total"] > 0

    def test_estimate_ffn_flops(self):
        """Test FFN FLOPs estimation."""
        flops = estimate_ffn_flops(
            hidden_dim=256,
            ffn_dim=512,
            num_nodes=100,
        )

        assert "total" in flops
        assert flops["total"] > 0
        assert "linear1" in flops
        assert "linear2" in flops

    def test_estimate_model_flops(self):
        """Test model FLOPs estimation."""
        estimate = estimate_model_flops(
            num_nodes=100,
            num_edges=300,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            ffn_dim=512,
        )

        assert isinstance(estimate, FLOPsEstimate)
        assert estimate.total > 0
        assert "attention" in estimate.breakdown
        assert "ffn" in estimate.breakdown

    def test_flops_estimator(self, simple_model):
        """Test FLOPsEstimator class."""
        estimator = FLOPsEstimator(simple_model)

        estimate = estimator.estimate(num_nodes=100, num_edges=300)
        assert estimate.total > 0

        params = estimator.count_parameters()
        assert params["total"] > 0

    def test_flops_are_finite(self):
        """Test that FLOPs estimates are finite."""
        estimate = estimate_model_flops(
            num_nodes=1000,
            num_edges=5000,
            hidden_dim=512,
            num_layers=12,
            num_heads=16,
            ffn_dim=2048,
        )

        assert not torch.tensor(estimate.total).isnan()
        assert not torch.tensor(estimate.total).isinf()
        assert estimate.total > 0


# ============================================================================
# Scaling Tests
# ============================================================================

class TestScaling:
    """Tests for scaling experiments."""

    def test_generate_scaling_graphs(self):
        """Test scaling graph generation."""
        graphs = generate_scaling_graphs(
            num_nodes_list=[50, 100, 200],
            avg_degree=5,
            feature_dim=16,
        )

        assert len(graphs) == 3
        assert graphs[0].num_nodes == 50
        assert graphs[1].num_nodes == 100
        assert graphs[2].num_nodes == 200

        for g in graphs:
            assert g.x.shape[1] == 16
            assert g.edge_index.size(0) == 2

    def test_scaling_result_to_dict(self):
        """Test ScalingResult serialization."""
        result = ScalingResult(
            parameter_name="num_nodes",
            parameter_values=[100, 200],
            runtime_stats=[
                RuntimeStats(10.0, 1.0, 8.0, 12.0, 10),
                RuntimeStats(20.0, 2.0, 16.0, 24.0, 10),
            ],
        )

        d = result.to_dict()
        assert d["parameter_name"] == "num_nodes"
        assert len(d["parameter_values"]) == 2
        assert len(d["runtime"]) == 2


# ============================================================================
# Utils Tests
# ============================================================================

class TestUtils:
    """Tests for profiling utilities."""

    def test_get_hardware_info(self):
        """Test hardware info retrieval."""
        info = get_hardware_info()

        assert "platform" in info
        assert "torch_version" in info
        assert "cuda_available" in info

    def test_get_model_info(self, simple_model):
        """Test model info retrieval."""
        info = get_model_info(simple_model)

        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "parameter_memory_mb" in info
        assert info["total_parameters"] > 0

    def test_profiling_context(self, simple_model, sample_batch):
        """Test ProfilingContext."""
        # Get original dropout state
        original_p = None
        for m in simple_model.modules():
            if isinstance(m, nn.Dropout):
                original_p = m.p

        with ProfilingContext(simple_model, disable_dropout=True, eval_mode=True):
            # Dropout should be disabled
            for m in simple_model.modules():
                if isinstance(m, nn.Dropout):
                    assert m.p == 0.0

            # Model should be in eval mode
            assert not simple_model.training

        # Should be restored
        if original_p is not None:
            for m in simple_model.modules():
                if isinstance(m, nn.Dropout):
                    assert m.p == original_p

    def test_profiling_does_not_modify_parameters(self, simple_model, sample_batch):
        """Test that profiling doesn't modify model parameters."""
        # Store original parameters
        original_params = {
            name: param.clone()
            for name, param in simple_model.named_parameters()
        }

        # Run profiling
        profiler = RuntimeProfiler(simple_model, torch.device("cpu"), num_runs=5)
        profiler.profile_forward(sample_batch)

        # Check parameters unchanged
        for name, param in simple_model.named_parameters():
            assert torch.equal(param, original_params[name]), f"Parameter {name} was modified"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for profiling framework."""

    def test_full_profiling_pipeline(self, simple_model, sample_batch, device):
        """Test complete profiling pipeline."""
        # Runtime
        rt_profiler = RuntimeProfiler(simple_model, device, num_runs=5)
        rt_results = rt_profiler.profile_all(sample_batch)
        assert "forward" in rt_results
        assert "backward" in rt_results

        # Memory
        mem_profiler = MemoryProfiler(simple_model, device)
        mem_results = mem_profiler.profile_all(sample_batch)
        assert "parameter_memory_mb" in mem_results
        assert "forward" in mem_results

        # FLOPs
        flops_est = FLOPsEstimator(simple_model)
        flops_results = flops_est.estimate(
            sample_batch.num_nodes,
            sample_batch.edge_index.size(1) // 2
        )
        assert flops_results.total > 0

    def test_profiling_with_batch_graphs(self):
        """Test profiling with batched graphs."""
        # Create batch of graphs
        graphs = []
        for i in range(4):
            num_nodes = 20 + i * 5
            x = torch.randn(num_nodes, 16)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
            y = torch.randn(1)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        batch = Batch.from_data_list(graphs)

        # Create simple model
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Profile forward
        def forward():
            return model(batch.x).mean()

        stats = benchmark_function(forward, num_runs=5, warmup_runs=1)
        assert stats.mean > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

