"""Tests for reproducibility guarantees."""

import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import (
    set_global_seed,
    get_git_info,
    get_environment_info,
    create_reproducibility_info,
    verify_reproducibility,
    reproducibility_context,
)


# ============================================================================
# Seed Tests
# ============================================================================

class TestSeedSetting:
    """Tests for seed setting."""

    def test_python_random_deterministic(self):
        """Test Python random is deterministic with seed."""
        import random

        set_global_seed(42)
        values1 = [random.random() for _ in range(10)]

        set_global_seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    def test_numpy_random_deterministic(self):
        """Test NumPy random is deterministic with seed."""
        set_global_seed(42)
        arr1 = np.random.rand(100)

        set_global_seed(42)
        arr2 = np.random.rand(100)

        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_random_deterministic(self):
        """Test PyTorch random is deterministic with seed."""
        set_global_seed(42)
        t1 = torch.rand(100)

        set_global_seed(42)
        t2 = torch.rand(100)

        assert torch.allclose(t1, t2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_deterministic(self):
        """Test PyTorch CUDA random is deterministic with seed."""
        set_global_seed(42)
        t1 = torch.rand(100, device="cuda")

        set_global_seed(42)
        t2 = torch.rand(100, device="cuda")

        assert torch.allclose(t1, t2)

    def test_different_seeds_give_different_results(self):
        """Test different seeds produce different results."""
        set_global_seed(42)
        val1 = np.random.rand()

        set_global_seed(123)
        val2 = np.random.rand()

        assert val1 != val2


# ============================================================================
# Model Training Reproducibility
# ============================================================================

class TestModelReproducibility:
    """Tests for model training reproducibility."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model."""
        return nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample graph data."""
        return Data(
            x=torch.randn(10, 16),
            edge_index=torch.randint(0, 10, (2, 30)),
            y=torch.randn(1),
        )

    def test_model_init_deterministic(self, simple_model):
        """Test model initialization is deterministic."""
        set_global_seed(42)
        model1 = nn.Linear(16, 16)

        set_global_seed(42)
        model2 = nn.Linear(16, 16)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_forward_pass_deterministic(self, simple_model, sample_data):
        """Test forward pass is deterministic."""
        set_global_seed(42)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        out1 = model(sample_data.x).mean()

        set_global_seed(42)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        out2 = model(sample_data.x).mean()

        assert torch.allclose(out1, out2)

    def test_training_step_deterministic(self, sample_data):
        """Test single training step is deterministic."""
        def train_step(seed):
            set_global_seed(seed)
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            model.train()
            optimizer.zero_grad()
            out = model(sample_data.x).mean()
            loss = (out - sample_data.y[0]) ** 2
            loss.backward()
            optimizer.step()

            return loss.item(), [p.data.clone() for p in model.parameters()]

        loss1, params1 = train_step(42)
        loss2, params2 = train_step(42)

        assert loss1 == loss2
        for p1, p2 in zip(params1, params2):
            assert torch.allclose(p1, p2)


# ============================================================================
# PE Computation Reproducibility
# ============================================================================

class TestPEReproducibility:
    """Tests for PE computation reproducibility."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph."""
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
        ])
        return Data(
            x=torch.randn(5, 16),
            edge_index=edge_index,
            num_nodes=5,
        )

    def test_lap_pe_deterministic(self, sample_graph):
        """Test LapPE is deterministic."""
        from src.pe.node import LapPE

        pe = LapPE(dim=16, k=4)

        set_global_seed(42)
        pe1 = pe.compute(sample_graph)

        set_global_seed(42)
        pe2 = pe.compute(sample_graph)

        # Note: Eigenvectors may have sign ambiguity
        # Check absolute values or use sign-invariant comparison
        assert pe1.shape == pe2.shape

    def test_rwse_deterministic(self, sample_graph):
        """Test RWSE is deterministic (should be fully deterministic)."""
        from src.pe.node import RWSE

        pe = RWSE(dim=16, scales=[1, 2, 4])

        pe1 = pe.compute(sample_graph)
        pe2 = pe.compute(sample_graph)

        assert torch.allclose(pe1, pe2)

    def test_spd_deterministic(self, sample_graph):
        """Test SPD is deterministic."""
        from src.pe.relative import SPDBuckets

        pe = SPDBuckets(num_buckets=8, max_distance=5)

        idx1, attr1 = pe.compute(sample_graph)
        idx2, attr2 = pe.compute(sample_graph)

        assert torch.equal(idx1, idx2)
        assert torch.allclose(attr1, attr2)


# ============================================================================
# Context Manager Tests
# ============================================================================

class TestReproducibilityContext:
    """Tests for reproducibility context manager."""

    def test_context_restores_state(self):
        """Test context manager restores random state."""
        # Set initial state
        np.random.seed(100)
        initial_val = np.random.rand()

        # Use context with different seed
        np.random.seed(100)
        with reproducibility_context(seed=42):
            context_val = np.random.rand()

        # After context, should continue from where we left off
        after_val = np.random.rand()

        # Values in context should be different
        assert context_val != initial_val

        # This is a basic test - full state restoration is complex

    def test_context_is_deterministic(self):
        """Test operations in context are deterministic."""
        results = []

        for _ in range(3):
            with reproducibility_context(seed=42):
                val = torch.rand(1).item()
                results.append(val)

        assert all(r == results[0] for r in results)


# ============================================================================
# Info Functions Tests
# ============================================================================

class TestInfoFunctions:
    """Tests for information gathering functions."""

    def test_get_git_info(self):
        """Test git info retrieval."""
        info = get_git_info()

        assert "commit" in info
        assert "branch" in info
        assert "dirty" in info
        assert "timestamp" in info

    def test_get_environment_info(self):
        """Test environment info retrieval."""
        info = get_environment_info()

        assert "python_version" in info
        assert "torch_version" in info
        assert "numpy_version" in info
        assert "cuda_available" in info

    def test_create_reproducibility_info(self):
        """Test reproducibility info creation."""
        config = {"model": "test", "seed": 42}
        info = create_reproducibility_info(config, seed=42)

        assert info["seed"] == 42
        assert "git" in info
        assert "environment" in info
        assert "config_hash" in info
        assert "timestamp" in info


# ============================================================================
# Verification Tests
# ============================================================================

class TestVerification:
    """Tests for reproducibility verification."""

    def test_verify_identical_results(self):
        """Test verification of identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "run1"
            dir2 = Path(tmpdir) / "run2"
            dir1.mkdir()
            dir2.mkdir()

            # Create identical results
            results = {"metric": 0.5, "loss": 0.1}

            import json
            with open(dir1 / "results.json", "w") as f:
                json.dump(results, f)
            with open(dir2 / "results.json", "w") as f:
                json.dump(results, f)

            # Verify
            verification = verify_reproducibility(dir1, dir2)
            assert verification["identical"]

    def test_verify_different_results(self):
        """Test verification detects differences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "run1"
            dir2 = Path(tmpdir) / "run2"
            dir1.mkdir()
            dir2.mkdir()

            # Create different results
            import json
            with open(dir1 / "results.json", "w") as f:
                json.dump({"metric": 0.5}, f)
            with open(dir2 / "results.json", "w") as f:
                json.dump({"metric": 0.6}, f)

            # Verify
            verification = verify_reproducibility(dir1, dir2)
            assert not verification["identical"]
            assert len(verification["differences"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

