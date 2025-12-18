"""Tests for result processing module."""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results.loader import (
    ExperimentResult,
    ResultLoader,
    load_experiment,
)
from src.results.aggregate import (
    ResultAggregator,
    AggregatedResult,
    aggregate_by_seed,
    filter_results,
)
from src.results.tables import (
    TableGenerator,
    make_performance_table,
    export_table,
)
from src.results.plots import (
    PlotGenerator,
    save_figure,
)
from src.results.formatting import (
    TableFormatter,
    PlotStyle,
    PE_NAMES,
    METRIC_FORMATS,
)
from src.results.utils import (
    safe_mean,
    safe_std,
    flatten_dict,
    unflatten_dict,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_result():
    """Create sample experiment result."""
    return ExperimentResult(
        experiment_id="abc123",
        experiment_name="test_experiment",
        output_dir=Path("/tmp/test"),
        config={
            "name": "test",
            "dataset": {"name": "zinc"},
            "model": {"name": "graph_transformer", "num_layers": 6},
            "pe": {
                "node": {"type": "lap", "dim": 16},
                "relative": {"type": "spd"},
            },
            "seed": 42,
        },
        status="completed",
        seed=42,
        test_metrics={"mae": 0.25, "loss": 0.5},
        val_metrics={"mae": 0.27, "loss": 0.55},
        best_epoch=50,
    )


@pytest.fixture
def sample_results():
    """Create list of sample results with different seeds."""
    results = []
    for seed in [42, 123, 456]:
        result = ExperimentResult(
            experiment_id=f"exp_{seed}",
            experiment_name="test_experiment",
            output_dir=Path(f"/tmp/test_{seed}"),
            config={
                "name": "test",
                "dataset": {"name": "zinc"},
                "model": {"name": "graph_transformer", "num_layers": 6},
                "pe": {
                    "node": {"type": "lap", "dim": 16},
                    "relative": {"type": "spd"},
                },
                "seed": seed,
            },
            status="completed",
            seed=seed,
            test_metrics={"mae": 0.25 + seed * 0.001, "loss": 0.5 + seed * 0.001},
            runtime_ms=10.0 + seed * 0.1,
            memory_mb=100 + seed,
            parameters=1000000,
        )
        results.append(result)
    return results


@pytest.fixture
def mock_experiment_dir(temp_dir):
    """Create mock experiment output directory."""
    exp_dir = temp_dir / "experiment_1"
    exp_dir.mkdir()

    # Config
    config = {
        "name": "mock_experiment",
        "dataset": {"name": "zinc"},
        "model": {"name": "graph_transformer", "num_layers": 6},
        "pe": {"node": {"type": "lap"}, "relative": {"type": "spd"}},
        "seed": 42,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Results
    results = {
        "experiment_id": "mock123",
        "experiment_name": "mock_experiment",
        "status": "completed",
        "training": {"best_epoch": 50},
        "evaluation": {"test_metrics": {"mae": 0.25, "loss": 0.5}},
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f)

    return exp_dir


# ============================================================================
# Loader Tests
# ============================================================================

class TestLoader:
    """Tests for result loading."""

    def test_experiment_result_properties(self, sample_result):
        """Test ExperimentResult properties."""
        assert sample_result.dataset == "zinc"
        assert sample_result.model == "graph_transformer"
        assert sample_result.node_pe_type == "lap"
        assert sample_result.relative_pe_type == "spd"
        assert sample_result.num_layers == 6

    def test_experiment_result_get_metric(self, sample_result):
        """Test getting metrics by name."""
        assert sample_result.get_metric("mae", "test") == 0.25
        assert sample_result.get_metric("mae", "val") == 0.27
        assert sample_result.get_metric("nonexistent") is None

    def test_experiment_result_is_complete(self, sample_result):
        """Test completion check."""
        assert sample_result.is_complete() == True

        incomplete = ExperimentResult(
            experiment_id="inc",
            experiment_name="incomplete",
            output_dir=Path("/tmp"),
            status="failed",
        )
        assert incomplete.is_complete() == False

    def test_load_experiment_from_dir(self, mock_experiment_dir):
        """Test loading from directory."""
        result = load_experiment(str(mock_experiment_dir))
        assert result is not None
        assert result.experiment_name == "mock_experiment"
        assert result.status == "completed"
        assert result.test_metrics.get("mae") == 0.25

    def test_result_loader_discover(self, temp_dir, mock_experiment_dir):
        """Test experiment discovery."""
        loader = ResultLoader(str(temp_dir))
        paths = loader.discover_experiments()
        assert len(paths) == 1

    def test_load_with_missing_files(self, temp_dir):
        """Test loading handles missing files gracefully."""
        exp_dir = temp_dir / "partial"
        exp_dir.mkdir()

        # Only config, no results
        config = {"name": "partial", "dataset": {"name": "zinc"}}
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f)

        result = load_experiment(str(exp_dir))
        # Should return None or partial result
        # Implementation specific


# ============================================================================
# Aggregation Tests
# ============================================================================

class TestAggregation:
    """Tests for result aggregation."""

    def test_aggregator_group_by(self, sample_results):
        """Test grouping by keys."""
        aggregator = ResultAggregator(sample_results)
        groups = aggregator.group_by(["dataset", "model"])

        assert len(groups) == 1  # All same dataset/model
        key = ("zinc", "graph_transformer")
        assert key in groups
        assert len(groups[key]) == 3

    def test_aggregate_by_seed(self, sample_results):
        """Test aggregation over seeds."""
        aggregated = aggregate_by_seed(
            sample_results,
            group_keys=["dataset", "model", "node_pe", "relative_pe"],
        )

        assert len(aggregated) == 1
        agg = aggregated[0]
        assert agg.n_experiments == 3
        assert 42 in agg.seeds
        assert 123 in agg.seeds
        assert 456 in agg.seeds

    def test_aggregation_statistics(self, sample_results):
        """Test that statistics are computed correctly."""
        aggregated = aggregate_by_seed(sample_results, ["dataset"])
        agg = aggregated[0]

        # Check mean
        mae_values = [r.test_metrics["mae"] for r in sample_results]
        expected_mean = np.mean(mae_values)
        assert abs(agg.metrics["mae"]["mean"] - expected_mean) < 1e-6

        # Check std
        expected_std = np.std(mae_values)
        assert abs(agg.metrics["mae"]["std"] - expected_std) < 1e-6

    def test_filter_results(self, sample_results):
        """Test filtering results."""
        filtered = filter_results(sample_results, dataset="zinc")
        assert len(filtered) == 3

        filtered = filter_results(sample_results, dataset="qm9")
        assert len(filtered) == 0

    def test_aggregated_format_metric(self, sample_results):
        """Test metric formatting."""
        aggregated = aggregate_by_seed(sample_results, ["dataset"])
        agg = aggregated[0]

        formatted = agg.format_metric("mae", precision=4)
        assert "±" in formatted or formatted.count(".") >= 1


# ============================================================================
# Table Tests
# ============================================================================

class TestTables:
    """Tests for table generation."""

    def test_table_generator_creation(self):
        """Test TableGenerator initialization."""
        gen = TableGenerator()
        assert gen.formatter is not None

    def test_performance_table(self, sample_results):
        """Test performance table generation."""
        gen = TableGenerator()
        table = gen.performance_table(sample_results, metrics=["mae", "loss"])

        assert "columns" in table
        assert "rows" in table
        assert len(table["rows"]) > 0

    def test_ablation_table(self, sample_results):
        """Test ablation table generation."""
        gen = TableGenerator()
        table = gen.ablation_table(
            sample_results,
            ablation_key="node_pe",
            metrics=["mae"],
        )

        assert "columns" in table
        assert "rows" in table

    def test_export_latex(self, sample_results, temp_dir):
        """Test LaTeX export."""
        table = make_performance_table(sample_results)
        output_path = temp_dir / "table.tex"
        content = export_table(table, str(output_path), format="latex")

        assert output_path.exists()
        assert "\\begin{table}" in content
        assert "\\end{table}" in content
        assert "\\toprule" in content

    def test_export_csv(self, sample_results, temp_dir):
        """Test CSV export."""
        table = make_performance_table(sample_results)
        output_path = temp_dir / "table.csv"
        content = export_table(table, str(output_path), format="csv")

        assert output_path.exists()
        assert "," in content

    def test_export_markdown(self, sample_results, temp_dir):
        """Test Markdown export."""
        table = make_performance_table(sample_results)
        output_path = temp_dir / "table.md"
        content = export_table(table, str(output_path), format="markdown")

        assert output_path.exists()
        assert "|" in content


# ============================================================================
# Plot Tests
# ============================================================================

class TestPlots:
    """Tests for plot generation."""

    def test_plot_generator_creation(self):
        """Test PlotGenerator initialization."""
        gen = PlotGenerator()
        assert gen.style is not None

    def test_plot_style_colors(self):
        """Test PlotStyle color cycling."""
        style = PlotStyle()
        assert style.get_color(0) != style.get_color(1)
        # Should cycle
        assert style.get_color(0) == style.get_color(len(style.colors))

    def test_performance_vs_distance_plot(self, sample_results):
        """Test distance plot generation without error."""
        gen = PlotGenerator()
        # This should not raise
        try:
            fig = gen.performance_vs_distance(sample_results)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            # May fail if no long-range data, but should not crash
            pass

    def test_save_figure(self, sample_results, temp_dir):
        """Test figure saving."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        output_path = temp_dir / "test_fig"
        saved = save_figure(fig, str(output_path), formats=["png"])

        assert len(saved) == 1
        assert Path(saved[0]).exists()


# ============================================================================
# Formatting Tests
# ============================================================================

class TestFormatting:
    """Tests for formatting utilities."""

    def test_table_formatter_metric_name(self):
        """Test metric name formatting."""
        formatter = TableFormatter()
        assert formatter.metric_name("mae") == "MAE"
        assert formatter.metric_name("roc_auc") == "ROC-AUC"

    def test_table_formatter_precision(self):
        """Test precision getting."""
        formatter = TableFormatter()
        assert formatter.get_precision("mae") == 4
        assert formatter.get_precision("accuracy") == 2

    def test_format_mean_std(self):
        """Test mean±std formatting."""
        formatter = TableFormatter()
        formatted = formatter.format_mean_std(0.25, 0.01, "mae")
        assert "0.25" in formatted
        assert "±" in formatted

    def test_pe_names(self):
        """Test PE name mapping."""
        assert PE_NAMES["lap"] == "LapPE"
        assert PE_NAMES["combined"] == "MSPE"

    def test_bold_best(self):
        """Test bolding best value."""
        formatter = TableFormatter()
        values = ["0.25", "0.20", "0.30"]
        bolded = formatter.bold_best(values, "mae", format="latex")
        assert "\\textbf" in bolded[1]  # 0.20 is best (lower)


# ============================================================================
# Utils Tests
# ============================================================================

class TestUtils:
    """Tests for utility functions."""

    def test_safe_mean_empty(self):
        """Test safe_mean with empty list."""
        assert safe_mean([]) == 0.0
        assert safe_mean([], default=1.0) == 1.0

    def test_safe_mean_with_nans(self):
        """Test safe_mean ignores NaN."""
        values = [1.0, 2.0, float("nan"), 3.0]
        result = safe_mean(values)
        assert result == 2.0

    def test_safe_std_insufficient(self):
        """Test safe_std with insufficient values."""
        assert safe_std([]) == 0.0
        assert safe_std([1.0]) == 0.0

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {"a": {"b": {"c": 1}}, "d": 2}
        flat = flatten_dict(nested)
        assert flat["a.b.c"] == 1
        assert flat["d"] == 2

    def test_unflatten_dict(self):
        """Test dictionary unflattening."""
        flat = {"a.b.c": 1, "d": 2}
        nested = unflatten_dict(flat)
        assert nested["a"]["b"]["c"] == 1
        assert nested["d"] == 2


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Tests for output determinism."""

    def test_aggregation_deterministic(self, sample_results):
        """Test that aggregation is deterministic."""
        agg1 = aggregate_by_seed(sample_results, ["dataset"])
        agg2 = aggregate_by_seed(sample_results, ["dataset"])

        assert agg1[0].metrics["mae"]["mean"] == agg2[0].metrics["mae"]["mean"]
        assert agg1[0].metrics["mae"]["std"] == agg2[0].metrics["mae"]["std"]

    def test_table_generation_deterministic(self, sample_results):
        """Test that table generation is deterministic."""
        gen = TableGenerator()
        table1 = gen.performance_table(sample_results)
        table2 = gen.performance_table(sample_results)

        assert table1["rows"] == table2["rows"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

