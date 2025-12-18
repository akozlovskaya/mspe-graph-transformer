"""Tests for experiment orchestration framework."""

import pytest
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.registry import (
    ExperimentConfig,
    ExperimentRegistry,
    register_experiment,
    get_experiment,
    list_experiments,
    validate_config,
)
from src.experiments.runner import ExperimentRunner
from src.experiments.sweeps import (
    SweepConfig,
    GridSweep,
    RandomSweep,
    SeedSweep,
    SweepState,
)
from src.experiments.logging import (
    ExperimentLogger,
    setup_experiment_logging,
)
from src.experiments.utils import (
    generate_experiment_id,
    get_output_dir,
    save_experiment_results,
    load_experiment_results,
    aggregate_results,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_config():
    """Create sample experiment config."""
    return ExperimentConfig(
        name="test_experiment",
        dataset={"name": "zinc", "root": "./data"},
        model={"name": "graph_transformer", "hidden_dim": 64, "num_layers": 2},
        pe={"node": {"type": "lap", "dim": 8}, "relative": {"type": "none"}},
        training={"epochs": 2, "batch_size": 4, "lr": 1e-3},
        seed=42,
        tags=["test"],
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 1)

        def forward(self, batch):
            return self.lin(batch.x).mean(dim=0, keepdim=True)

    return SimpleModel()


# ============================================================================
# Registry Tests
# ============================================================================

class TestRegistry:
    """Tests for experiment registry."""

    def test_experiment_config_creation(self, sample_config):
        """Test ExperimentConfig creation."""
        assert sample_config.name == "test_experiment"
        assert sample_config.dataset["name"] == "zinc"
        assert sample_config.seed == 42

    def test_config_to_dict(self, sample_config):
        """Test config serialization."""
        d = sample_config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_experiment"
        assert "dataset" in d
        assert "model" in d

    def test_config_from_dict(self, sample_config):
        """Test config deserialization."""
        d = sample_config.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.name == sample_config.name
        assert restored.seed == sample_config.seed

    def test_config_get_id(self, sample_config):
        """Test unique ID generation."""
        id1 = sample_config.get_id()
        id2 = sample_config.get_id()
        assert id1 == id2  # Same config should give same ID
        assert len(id1) == 12

    def test_config_with_overrides(self, sample_config):
        """Test config override."""
        modified = sample_config.with_overrides(seed=123)
        assert modified.seed == 123
        assert sample_config.seed == 42  # Original unchanged

    def test_config_nested_override(self, sample_config):
        """Test nested config override."""
        modified = sample_config.with_overrides(**{"model.num_layers": 8})
        assert modified.model["num_layers"] == 8

    def test_registry_register(self):
        """Test experiment registration."""
        registry = ExperimentRegistry()
        config = registry.register(
            "test_reg",
            dataset={"name": "zinc"},
            model={"name": "graph_transformer"},
        )
        assert config.name == "test_reg"

    def test_registry_get(self):
        """Test experiment retrieval."""
        registry = ExperimentRegistry()
        registry.register("test_get", dataset={"name": "zinc"})
        retrieved = registry.get("test_get")
        assert retrieved.name == "test_get"

    def test_registry_list(self):
        """Test experiment listing."""
        registry = ExperimentRegistry()
        registry.register("exp1", dataset={"name": "zinc"}, tags=["tag1"])
        registry.register("exp2", dataset={"name": "qm9"}, tags=["tag2"])

        all_exps = registry.list()
        assert "exp1" in all_exps
        assert "exp2" in all_exps

        tag1_exps = registry.list(tags=["tag1"])
        assert "exp1" in tag1_exps
        assert "exp2" not in tag1_exps

    def test_global_registry(self):
        """Test global registry functions."""
        # Note: This modifies global state
        experiments = list_experiments()
        assert isinstance(experiments, list)

    def test_validate_config(self, sample_config):
        """Test config validation."""
        assert validate_config(sample_config) == True

        # Invalid config
        invalid = ExperimentConfig(
            name="",  # Empty name
            dataset={},
            model={},
            pe={},
            training={},
        )
        registry = ExperimentRegistry()
        assert registry.validate(invalid) == False


# ============================================================================
# Sweep Tests
# ============================================================================

class TestSweeps:
    """Tests for sweep management."""

    def test_sweep_config_creation(self):
        """Test SweepConfig creation."""
        sweep = SweepConfig(
            name="test_sweep",
            base_config={"name": "base", "seed": 42},
            parameters={"seed": [1, 2, 3]},
            sweep_type="grid",
        )
        assert sweep.name == "test_sweep"
        assert sweep.get_num_experiments() == 3

    def test_grid_sweep_generation(self, sample_config):
        """Test grid sweep config generation."""
        sweep = GridSweep(
            sample_config,
            parameters={
                "seed": [1, 2],
                "model.num_layers": [2, 4],
            },
        )

        configs = list(sweep.generate())
        assert len(configs) == 4  # 2 seeds × 2 depths

    def test_random_sweep_generation(self, sample_config):
        """Test random sweep config generation."""
        sweep = RandomSweep(
            sample_config,
            parameters={
                "seed": [1, 2, 3, 4, 5],
                "model.num_layers": [2, 4, 6, 8],
            },
            num_samples=5,
            seed=42,
        )

        configs = list(sweep.generate())
        assert len(configs) == 5

    def test_seed_sweep_generation(self, sample_config):
        """Test seed sweep config generation."""
        sweep = SeedSweep(
            sample_config,
            seeds=[42, 123, 456],
        )

        configs = list(sweep.generate())
        assert len(configs) == 3

        seeds = [c.seed for c in configs]
        assert 42 in seeds
        assert 123 in seeds
        assert 456 in seeds

    def test_sweep_state_save_load(self, temp_dir):
        """Test sweep state persistence."""
        state = SweepState(
            sweep_name="test",
            total_experiments=10,
            completed=["exp1", "exp2"],
            failed=["exp3"],
        )

        state_path = temp_dir / "state.json"
        state.save(state_path)

        loaded = SweepState.load(state_path)
        assert loaded.sweep_name == "test"
        assert len(loaded.completed) == 2
        assert len(loaded.failed) == 1


# ============================================================================
# Logging Tests
# ============================================================================

class TestLogging:
    """Tests for experiment logging."""

    def test_experiment_logger_creation(self, temp_dir):
        """Test ExperimentLogger creation."""
        logger = ExperimentLogger(temp_dir)
        assert logger.output_dir == temp_dir

    def test_log_config(self, temp_dir, sample_config):
        """Test config logging."""
        logger = ExperimentLogger(temp_dir)
        logger.log_config(sample_config.to_dict())

        # Check file exists
        config_files = list(temp_dir.glob("config.*"))
        assert len(config_files) == 1

    def test_log_metrics(self, temp_dir):
        """Test metrics logging."""
        logger = ExperimentLogger(temp_dir)
        logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
        logger.log_metrics({"loss": 0.3, "acc": 0.95}, step=2)

        metrics_path = temp_dir / "metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert len(metrics) == 2

    def test_log_artifact(self, temp_dir):
        """Test artifact logging."""
        logger = ExperimentLogger(temp_dir)
        logger.log_artifact("test_data", {"key": "value"}, artifact_type="json")

        artifact_path = temp_dir / "artifacts" / "test_data.json"
        assert artifact_path.exists()

    def test_setup_experiment_logging(self, temp_dir):
        """Test logging setup."""
        logger = setup_experiment_logging(temp_dir)
        assert logger is not None


# ============================================================================
# Utils Tests
# ============================================================================

class TestUtils:
    """Tests for experiment utilities."""

    def test_generate_experiment_id(self, sample_config):
        """Test experiment ID generation."""
        id1 = generate_experiment_id(sample_config.to_dict())
        id2 = generate_experiment_id(sample_config.to_dict())

        # Same config should give same hash prefix
        assert id1[:8] == id2[:8]

    def test_get_output_dir(self, sample_config, temp_dir):
        """Test output directory generation."""
        output_dir = get_output_dir(sample_config, str(temp_dir))
        assert output_dir.exists()
        assert sample_config.name in str(output_dir)

    def test_save_load_results(self, temp_dir):
        """Test results save and load."""
        results = {
            "experiment_name": "test",
            "status": "completed",
            "evaluation": {"test_metrics": {"loss": 0.5}},
        }

        save_experiment_results(results, temp_dir)
        loaded = load_experiment_results(temp_dir)

        assert loaded["experiment_name"] == "test"
        assert loaded["status"] == "completed"

    def test_aggregate_results(self):
        """Test results aggregation."""
        results_list = [
            {
                "experiment_name": "exp1",
                "status": "completed",
                "evaluation": {"test_metrics": {"loss": 0.5, "mae": 0.3}},
            },
            {
                "experiment_name": "exp2",
                "status": "completed",
                "evaluation": {"test_metrics": {"loss": 0.4, "mae": 0.25}},
            },
        ]

        aggregated = aggregate_results(results_list)

        assert aggregated["num_experiments"] == 2
        assert "loss" in aggregated["aggregated_metrics"]
        assert aggregated["aggregated_metrics"]["loss"]["mean"] == 0.45


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""

    def test_config_composition(self):
        """Test that config composition works correctly."""
        registry = ExperimentRegistry()

        # Register base
        base = registry.register(
            "base",
            dataset={"name": "zinc"},
            model={"name": "graph_transformer", "num_layers": 6},
            tags=["base"],
        )

        # Create ablation
        ablation = registry.create_ablation(
            "base",
            "ablation_depth_8",
            **{"model.num_layers": 8},
        )

        assert ablation.model["num_layers"] == 8
        assert "ablation" in ablation.tags

    def test_results_format(self, temp_dir):
        """Test that results are saved in correct format."""
        results = {
            "experiment_id": "abc123",
            "experiment_name": "test",
            "config": {"seed": 42},
            "status": "completed",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T01:00:00",
            "training": {"best_epoch": 50},
            "evaluation": {"test_metrics": {"loss": 0.5}},
        }

        save_experiment_results(results, temp_dir)

        # Verify JSON structure
        with open(temp_dir / "results.json") as f:
            loaded = json.load(f)

        assert "experiment_id" in loaded
        assert "config" in loaded
        assert "training" in loaded
        assert "evaluation" in loaded

    def test_ablation_toggles(self, sample_config):
        """Test that ablation toggles change behavior."""
        # Toggle PE off
        no_pe = sample_config.with_overrides(**{
            "pe.node.type": "none",
            "pe.relative.type": "none",
        })

        assert no_pe.pe["node"]["type"] == "none"
        assert no_pe.pe["relative"]["type"] == "none"

        # Different configs should have different IDs
        assert no_pe.get_id() != sample_config.get_id()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

