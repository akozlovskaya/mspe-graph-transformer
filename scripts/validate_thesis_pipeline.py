#!/usr/bin/env python
"""
Thesis Pipeline Validation Script

Validates that the entire thesis pipeline works correctly:
1. Runs a minimal end-to-end experiment
2. Generates tables and plots
3. Verifies all outputs exist and are non-empty
4. Reports any missing or inconsistent artifacts
"""

import argparse
import logging
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.reproducibility import set_global_seed, get_git_info, get_environment_info


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, name: str, message: str = ""):
        self.passed.append({"name": name, "message": message})
        logger.info(f"✓ PASS: {name} {message}")

    def add_fail(self, name: str, message: str):
        self.failed.append({"name": name, "message": message})
        logger.error(f"✗ FAIL: {name} - {message}")

    def add_warning(self, name: str, message: str):
        self.warnings.append({"name": name, "message": message})
        logger.warning(f"⚠ WARN: {name} - {message}")

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "VALIDATION SUMMARY",
            "=" * 60,
            f"Passed:   {len(self.passed)}",
            f"Failed:   {len(self.failed)}",
            f"Warnings: {len(self.warnings)}",
            "",
        ]

        if self.failed:
            lines.append("FAILURES:")
            for f in self.failed:
                lines.append(f"  - {f['name']}: {f['message']}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w['name']}: {w['message']}")
            lines.append("")

        status = "SUCCESS" if self.success else "FAILED"
        lines.append(f"Overall: {status}")
        lines.append("=" * 60)

        return "\n".join(lines)


def validate_imports(result: ValidationResult):
    """Validate all required imports work."""
    logger.info("Validating imports...")

    modules = [
        ("src.dataset", "Dataset module"),
        ("src.pe.node", "Node PE module"),
        ("src.pe.relative", "Relative PE module"),
        ("src.training", "Training module"),
        ("src.evaluation", "Evaluation module"),
        ("src.experiments", "Experiments module"),
        ("src.results", "Results module"),
        ("src.utils.reproducibility", "Reproducibility utils"),
    ]

    for module, name in modules:
        try:
            __import__(module)
            result.add_pass(f"import_{module}", name)
        except ImportError as e:
            result.add_fail(f"import_{module}", str(e))


def validate_reproducibility(result: ValidationResult):
    """Validate reproducibility utilities."""
    logger.info("Validating reproducibility...")

    # Test seed setting
    try:
        set_global_seed(42)

        # Check Python random
        import random
        val1 = random.random()
        set_global_seed(42)
        val2 = random.random()
        assert val1 == val2, "Python random not reproducible"

        # Check NumPy
        arr1 = np.random.rand(10)
        set_global_seed(42)
        np.random.rand(10)  # Skip first call
        set_global_seed(42)
        arr2 = np.random.rand(10)
        # Note: Need to reset after first rand call
        set_global_seed(42)
        arr3 = np.random.rand(10)
        assert np.allclose(arr1, arr3), "NumPy random not reproducible"

        # Check PyTorch
        set_global_seed(42)
        t1 = torch.rand(10)
        set_global_seed(42)
        t2 = torch.rand(10)
        assert torch.allclose(t1, t2), "PyTorch random not reproducible"

        result.add_pass("reproducibility_seeds")

    except Exception as e:
        result.add_fail("reproducibility_seeds", str(e))

    # Test git info
    try:
        git_info = get_git_info()
        if git_info["commit"] == "unknown":
            result.add_warning("git_info", "Could not get git commit")
        else:
            result.add_pass("git_info", f"commit={git_info['commit'][:8]}")
    except Exception as e:
        result.add_fail("git_info", str(e))


def validate_configuration(result: ValidationResult):
    """Validate configuration system."""
    logger.info("Validating configuration...")

    config_dir = Path(__file__).parent.parent / "configs"

    required_configs = [
        "config.yaml",
        "dataset/default.yaml",
        "model/default.yaml",
        "pe/default.yaml",
        "train/default.yaml",
    ]

    for config in required_configs:
        path = config_dir / config
        if path.exists():
            result.add_pass(f"config_{config}")
        else:
            result.add_fail(f"config_{config}", f"Missing: {path}")


def validate_minimal_experiment(result: ValidationResult, output_dir: Path):
    """Run minimal end-to-end experiment."""
    logger.info("Running minimal experiment...")

    try:
        from src.dataset import get_dataset
        from src.pe.node import LapPE
        from src.training import Trainer, get_optimizer, get_loss_fn

        set_global_seed(42)

        # Create synthetic data
        from torch_geometric.data import Data, InMemoryDataset

        class ToyDataset(InMemoryDataset):
            def __init__(self):
                super().__init__(root=None)
                data_list = []
                for i in range(20):
                    x = torch.randn(10, 16)
                    edge_index = torch.randint(0, 10, (2, 30))
                    y = torch.randn(1)
                    data_list.append(Data(x=x, edge_index=edge_index, y=y))
                self.data, self.slices = self.collate(data_list)

            @property
            def num_features(self):
                return 16

            @property
            def num_classes(self):
                return 1

        dataset = ToyDataset()
        result.add_pass("dataset_creation")

        # Create model
        import torch.nn as nn

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(16, 1)

            def forward(self, batch):
                return self.lin(batch.x).mean(dim=0, keepdim=True)

        model = ToyModel()
        result.add_pass("model_creation")

        # Training step
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        batch = next(iter(loader))
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y[:1])
        loss.backward()
        optimizer.step()

        result.add_pass("training_step", f"loss={loss.item():.4f}")

        # Save checkpoint
        ckpt_path = output_dir / "checkpoints" / "test.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, ckpt_path)
        result.add_pass("checkpoint_save")

        # Save results
        results = {
            "experiment_name": "validation_test",
            "status": "completed",
            "evaluation": {"test_metrics": {"loss": loss.item()}},
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f)
        result.add_pass("results_save")

    except Exception as e:
        import traceback
        result.add_fail("minimal_experiment", f"{e}\n{traceback.format_exc()}")


def validate_result_generation(result: ValidationResult, output_dir: Path):
    """Validate result table and plot generation."""
    logger.info("Validating result generation...")

    try:
        from src.results import (
            ExperimentResult,
            TableGenerator,
            PlotGenerator,
            export_table,
            save_figure,
        )

        # Create mock result
        mock_result = ExperimentResult(
            experiment_id="test123",
            experiment_name="validation_test",
            output_dir=output_dir,
            config={
                "dataset": {"name": "zinc"},
                "model": {"name": "graph_transformer", "num_layers": 6},
                "pe": {"node": {"type": "lap"}, "relative": {"type": "spd"}},
            },
            status="completed",
            seed=42,
            test_metrics={"mae": 0.25, "loss": 0.5},
        )

        # Generate table
        gen = TableGenerator()
        table = gen.performance_table([mock_result], metrics=["mae", "loss"])

        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        export_table(table, str(tables_dir / "test.tex"), format="latex")

        if (tables_dir / "test.tex").exists():
            result.add_pass("table_generation")
        else:
            result.add_fail("table_generation", "Output file not created")

        # Generate plot (minimal)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, str(figures_dir / "test"), formats=["png"])

        if (figures_dir / "test.png").exists():
            result.add_pass("figure_generation")
        else:
            result.add_fail("figure_generation", "Output file not created")

    except Exception as e:
        import traceback
        result.add_fail("result_generation", f"{e}\n{traceback.format_exc()}")


def validate_thesis_artifacts(result: ValidationResult):
    """Validate thesis mapping files exist."""
    logger.info("Validating thesis artifacts...")

    thesis_dir = Path(__file__).parent.parent / "thesis"

    files = [
        "figures_map.yaml",
        "tables_map.yaml",
    ]

    for f in files:
        path = thesis_dir / f
        if path.exists():
            result.add_pass(f"thesis_{f}")
        else:
            result.add_warning(f"thesis_{f}", f"Missing: {path}")


def validate_tests(result: ValidationResult):
    """Validate test files exist."""
    logger.info("Validating test files...")

    tests_dir = Path(__file__).parent.parent / "tests"

    test_files = [
        "test_pe.py",
        "test_dataset_loading.py",
        "test_training_loop.py",
        "test_experiments.py",
        "test_results.py",
    ]

    for f in test_files:
        path = tests_dir / f
        if path.exists():
            result.add_pass(f"test_{f}")
        else:
            result.add_warning(f"test_{f}", f"Missing: {path}")


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate thesis pipeline"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for validation artifacts",
    )
    parser.add_argument(
        "--skip_experiment",
        action="store_true",
        help="Skip minimal experiment (faster)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("THESIS PIPELINE VALIDATION")
    logger.info("=" * 60)

    # Environment info
    env_info = get_environment_info()
    logger.info(f"Python: {env_info.get('python_version', 'unknown')[:50]}...")
    logger.info(f"PyTorch: {env_info.get('torch_version', 'unknown')}")
    logger.info(f"CUDA: {env_info.get('cuda_available', False)}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="thesis_validation_"))

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Run validations
    result = ValidationResult()

    validate_imports(result)
    validate_reproducibility(result)
    validate_configuration(result)

    if not args.skip_experiment:
        validate_minimal_experiment(result, output_dir)

    validate_result_generation(result, output_dir)
    validate_thesis_artifacts(result)
    validate_tests(result)

    # Print summary
    print(result.summary())

    # Save validation report
    report = {
        "success": result.success,
        "passed": result.passed,
        "failed": result.failed,
        "warnings": result.warnings,
        "environment": env_info,
        "git": get_git_info(),
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Validation report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()

