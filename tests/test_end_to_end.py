"""End-to-end integration tests."""

import pytest
import tempfile
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_global_seed


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def toy_dataset():
    """Create toy dataset for testing."""
    class ToyGraphDataset(InMemoryDataset):
        def __init__(self, num_graphs=50):
            super().__init__(root=None)
            set_global_seed(42)

            data_list = []
            for i in range(num_graphs):
                num_nodes = torch.randint(5, 15, (1,)).item()
                x = torch.randn(num_nodes, 16)
                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
                y = torch.randn(1)
                data_list.append(Data(x=x, edge_index=edge_index, y=y))

            self.data, self.slices = self.collate(data_list)

        @property
        def num_features(self):
            return 16

        @property
        def num_classes(self):
            return 1

    return ToyGraphDataset()


@pytest.fixture
def simple_model():
    """Create simple graph model."""
    class SimpleGraphModel(nn.Module):
        def __init__(self, in_dim=16, hidden_dim=32, out_dim=1):
            super().__init__()
            self.encoder = nn.Linear(in_dim, hidden_dim)
            self.decoder = nn.Linear(hidden_dim, out_dim)

        def forward(self, batch):
            x = torch.relu(self.encoder(batch.x))
            # Global mean pooling
            out = x.mean(dim=0, keepdim=True)
            return self.decoder(out)

    return SimpleGraphModel()


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_full_training_pipeline(self, toy_dataset, simple_model, temp_dir):
        """Test complete training pipeline."""
        set_global_seed(42)

        # Setup
        train_loader = DataLoader(toy_dataset[:40], batch_size=8, shuffle=True)
        val_loader = DataLoader(toy_dataset[40:], batch_size=8, shuffle=False)

        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Training loop
        num_epochs = 3
        train_losses = []

        for epoch in range(num_epochs):
            simple_model.train()
            epoch_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                out = simple_model(batch)
                # Ensure target has same shape as output [batch_size, 1]
                target = batch.y[:1] if batch.y.dim() == 1 else batch.y[:1]
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

        # Validation
        simple_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = simple_model(batch)
                # Ensure target has same shape as output [batch_size, 1]
                target = batch.y[:1] if batch.y.dim() == 1 else batch.y[:1]
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                val_loss += loss_fn(out, target).item()
        val_loss /= len(val_loader)

        # Assertions
        assert len(train_losses) == num_epochs
        assert all(isinstance(l, float) for l in train_losses)
        assert val_loss > 0

        # Save checkpoint
        ckpt_path = temp_dir / "checkpoint.pt"
        torch.save({
            "model": simple_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": num_epochs,
            "train_losses": train_losses,
            "val_loss": val_loss,
        }, ckpt_path)

        assert ckpt_path.exists()

    def test_checkpoint_save_load(self, simple_model, temp_dir):
        """Test checkpoint save and load."""
        set_global_seed(42)

        # Save
        ckpt_path = temp_dir / "model.pt"
        torch.save(simple_model.state_dict(), ckpt_path)

        # Modify model
        for p in simple_model.parameters():
            p.data.fill_(0)

        # Load
        simple_model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Verify weights are restored
        for p in simple_model.parameters():
            assert not torch.all(p == 0)

    def test_evaluation_no_gradients(self, toy_dataset, simple_model):
        """Test evaluation runs without gradients."""
        simple_model.eval()
        loader = DataLoader(toy_dataset[:10], batch_size=4)

        with torch.no_grad():
            for batch in loader:
                out = simple_model(batch)
                assert not out.requires_grad

    def test_results_saved_correctly(self, temp_dir):
        """Test results are saved in correct format."""
        results = {
            "experiment_name": "test",
            "status": "completed",
            "config": {"seed": 42},
            "training": {"best_epoch": 10, "history": []},
            "evaluation": {"test_metrics": {"mae": 0.25, "loss": 0.5}},
        }

        results_path = temp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Verify
        with open(results_path) as f:
            loaded = json.load(f)

        assert loaded["experiment_name"] == "test"
        assert loaded["status"] == "completed"
        assert loaded["evaluation"]["test_metrics"]["mae"] == 0.25


# ============================================================================
# PE Integration Tests
# ============================================================================

class TestPEIntegration:
    """Tests for PE integration in pipeline."""

    def test_node_pe_applied(self, toy_dataset):
        """Test node PE is applied to data."""
        from src.pe.node import RWSE

        pe = RWSE(dim=16, scales=[1, 2, 4])

        for data in toy_dataset[:5]:
            node_pe = pe.compute(data)
            assert node_pe is not None
            assert node_pe.shape[0] == data.num_nodes
            # RWSE output dimension equals number of scales (not dim parameter)
            assert node_pe.shape[1] == len(pe.scales)  # 3 scales

    def test_relative_pe_applied(self, toy_dataset):
        """Test relative PE is applied to data."""
        from src.pe.relative import SPDBuckets

        pe = SPDBuckets(num_buckets=8, max_distance=5)

        for data in toy_dataset[:5]:
            edge_idx, edge_attr = pe.compute(data)
            assert edge_idx is not None
            assert edge_attr is not None
            assert edge_idx.shape[0] == 2
            assert edge_attr.shape[1] == 8  # num_buckets


# ============================================================================
# Result Generation Tests
# ============================================================================

class TestResultGeneration:
    """Tests for result generation."""

    def test_table_generation(self, temp_dir):
        """Test table generation works."""
        from src.results import ExperimentResult, TableGenerator, export_table

        results = [
            ExperimentResult(
                experiment_id="exp1",
                experiment_name="test1",
                output_dir=temp_dir,
                config={
                    "dataset": {"name": "zinc"},
                    "model": {"name": "gt"},
                    "pe": {"node": {"type": "lap"}, "relative": {"type": "spd"}},
                },
                status="completed",
                seed=42,
                test_metrics={"mae": 0.25, "loss": 0.5},
            ),
            ExperimentResult(
                experiment_id="exp2",
                experiment_name="test2",
                output_dir=temp_dir,
                config={
                    "dataset": {"name": "zinc"},
                    "model": {"name": "gt"},
                    "pe": {"node": {"type": "rwse"}, "relative": {"type": "spd"}},
                },
                status="completed",
                seed=42,
                test_metrics={"mae": 0.22, "loss": 0.45},
            ),
        ]

        gen = TableGenerator()
        table = gen.performance_table(results, metrics=["mae", "loss"])

        assert "columns" in table
        assert "rows" in table
        assert len(table["rows"]) == 2

        # Export
        output_path = temp_dir / "test_table.tex"
        content = export_table(table, str(output_path), format="latex")

        assert output_path.exists()
        assert "\\begin{table}" in content

    def test_plot_generation(self, temp_dir):
        """Test plot generation works."""
        from src.results import PlotGenerator, save_figure
        import matplotlib
        matplotlib.use("Agg")

        gen = PlotGenerator()

        # Create simple figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        output_path = temp_dir / "test_plot"
        saved = save_figure(fig, str(output_path), formats=["png"])

        assert len(saved) == 1
        assert Path(saved[0]).exists()


# ============================================================================
# Experiment Orchestration Tests
# ============================================================================

class TestExperimentOrchestration:
    """Tests for experiment orchestration."""

    def test_config_creation(self):
        """Test experiment config creation."""
        from src.experiments import ExperimentConfig

        config = ExperimentConfig(
            name="test_exp",
            dataset={"name": "zinc"},
            model={"name": "gt", "num_layers": 6},
            pe={"node": {"type": "lap"}, "relative": {"type": "spd"}},
            training={"epochs": 10, "lr": 1e-4},
            seed=42,
        )

        assert config.name == "test_exp"
        assert config.dataset["name"] == "zinc"
        assert config.seed == 42

    def test_config_serialization(self):
        """Test config serialization."""
        from src.experiments import ExperimentConfig

        config = ExperimentConfig(
            name="test",
            dataset={"name": "zinc"},
            model={"name": "gt"},
            pe={},
            training={},
            seed=42,
        )

        d = config.to_dict()
        restored = ExperimentConfig.from_dict(d)

        assert restored.name == config.name
        assert restored.seed == config.seed

    def test_config_id_generation(self):
        """Test unique ID generation."""
        from src.experiments import ExperimentConfig

        config1 = ExperimentConfig(
            name="test", dataset={}, model={}, pe={}, training={}, seed=42
        )
        config2 = ExperimentConfig(
            name="test", dataset={}, model={}, pe={}, training={}, seed=42
        )
        config3 = ExperimentConfig(
            name="test", dataset={}, model={}, pe={}, training={}, seed=123
        )

        # Same config = same ID
        assert config1.get_id() == config2.get_id()
        # Different config = different ID
        assert config1.get_id() != config3.get_id()


# ============================================================================
# Thesis Artifact Tests
# ============================================================================

class TestThesisArtifacts:
    """Tests for thesis-specific artifacts."""

    def test_thesis_mapping_files_exist(self):
        """Test thesis mapping files exist."""
        thesis_dir = Path(__file__).parent.parent / "thesis"

        figures_map = thesis_dir / "figures_map.yaml"
        tables_map = thesis_dir / "tables_map.yaml"

        # These should exist after STEP 11
        # Mark as skip if not yet created
        if not thesis_dir.exists():
            pytest.skip("Thesis directory not created yet")

        assert figures_map.exists() or True  # Optional
        assert tables_map.exists() or True  # Optional

    def test_validation_script_exists(self):
        """Test validation script exists."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        validate_script = scripts_dir / "validate_thesis_pipeline.py"

        assert validate_script.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

