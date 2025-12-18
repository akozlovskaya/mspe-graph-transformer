"""Result loading utilities for experiment outputs."""

import json
import logging
from typing import Dict, Any, Optional, List, Iterator
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for a single experiment's results."""

    # Identifiers
    experiment_id: str
    experiment_name: str
    output_dir: Path

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Core results
    status: str = "unknown"
    seed: int = 42

    # Metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    # Training info
    best_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)

    # Long-range results
    long_range: Dict[str, Any] = field(default_factory=dict)
    distance_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Profiling results
    profiling: Dict[str, Any] = field(default_factory=dict)
    runtime_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    flops: Optional[int] = None
    parameters: Optional[int] = None

    # Metadata
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def dataset(self) -> str:
        """Get dataset name from config."""
        return self.config.get("dataset", {}).get("name", "unknown")

    @property
    def model(self) -> str:
        """Get model name from config."""
        return self.config.get("model", {}).get("name", "unknown")

    @property
    def node_pe_type(self) -> str:
        """Get node PE type from config."""
        return self.config.get("pe", {}).get("node", {}).get("type", "none")

    @property
    def relative_pe_type(self) -> str:
        """Get relative PE type from config."""
        return self.config.get("pe", {}).get("relative", {}).get("type", "none")

    @property
    def num_layers(self) -> int:
        """Get number of layers from config."""
        return self.config.get("model", {}).get("num_layers", 0)

    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension from config."""
        return self.config.get("model", {}).get("hidden_dim", 0)

    def get_metric(self, name: str, split: str = "test") -> Optional[float]:
        """Get a specific metric value."""
        if split == "test":
            return self.test_metrics.get(name)
        elif split == "val":
            return self.val_metrics.get(name)
        elif split == "train":
            return self.train_metrics.get(name)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "config": self.config,
            "status": self.status,
            "seed": self.seed,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
            "long_range": self.long_range,
            "profiling": self.profiling,
            "runtime_ms": self.runtime_ms,
            "memory_mb": self.memory_mb,
            "parameters": self.parameters,
        }

    def is_complete(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == "completed" and len(self.test_metrics) > 0


class ResultLoader:
    """Loader for experiment results."""

    def __init__(self, base_dir: str = "./outputs"):
        """
        Initialize result loader.

        Args:
            base_dir: Base directory containing experiment outputs.
        """
        self.base_dir = Path(base_dir)

    def discover_experiments(
        self,
        pattern: str = "**/results.json",
        recursive: bool = True,
    ) -> List[Path]:
        """
        Discover experiment output directories.

        Args:
            pattern: Glob pattern for results files.
            recursive: Whether to search recursively.

        Returns:
            List of paths to results files.
        """
        if recursive:
            results = list(self.base_dir.glob(pattern))
        else:
            results = list(self.base_dir.glob("*/results.json"))

        logger.info(f"Discovered {len(results)} experiments in {self.base_dir}")
        return sorted(results)

    def load_experiment(self, output_dir: Path) -> Optional[ExperimentResult]:
        """
        Load a single experiment from its output directory.

        Args:
            output_dir: Path to experiment output directory.

        Returns:
            ExperimentResult or None if loading fails.
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # If path points to a file, get parent directory
        if output_dir.is_file():
            output_dir = output_dir.parent

        try:
            result = ExperimentResult(
                experiment_id="",
                experiment_name="",
                output_dir=output_dir,
            )

            # Load config
            config = self._load_config(output_dir)
            result.config = config
            result.seed = config.get("seed", 42)
            result.experiment_name = config.get("name", output_dir.name)

            # Load main results
            results_data = self._load_json(output_dir / "results.json")
            if results_data:
                result.experiment_id = results_data.get("experiment_id", "")
                result.status = results_data.get("status", "unknown")
                result.start_time = results_data.get("start_time")
                result.end_time = results_data.get("end_time")
                result.errors = results_data.get("errors", [])

                # Training results
                training = results_data.get("training", {})
                result.best_epoch = training.get("best_epoch")
                result.training_history = training.get("history", [])

                # Evaluation results
                evaluation = results_data.get("evaluation", {})
                result.test_metrics = evaluation.get("test_metrics", {})

            # Load metrics history
            metrics_data = self._load_json(output_dir / "metrics.json")
            if metrics_data:
                result.training_history = metrics_data

            # Load long-range results
            lr_data = self._load_json(output_dir / "long_range.json")
            if lr_data:
                result.long_range = lr_data
                result.distance_metrics = lr_data.get("metrics_per_bucket", {})

            # Load profiling results
            prof_data = self._load_json(output_dir / "profiling.json")
            if prof_data:
                result.profiling = prof_data
                result.runtime_ms = prof_data.get("runtime_forward_ms", {}).get("mean")
                result.memory_mb = prof_data.get("memory", {}).get("peak_mb")
                result.parameters = prof_data.get("parameters")
                result.flops = prof_data.get("flops", {}).get("total")

            return result

        except Exception as e:
            logger.error(f"Failed to load experiment from {output_dir}: {e}")
            return None

    def load_all(
        self,
        filter_fn: Optional[callable] = None,
    ) -> List[ExperimentResult]:
        """
        Load all experiments from base directory.

        Args:
            filter_fn: Optional filter function.

        Returns:
            List of ExperimentResult objects.
        """
        results = []
        paths = self.discover_experiments()

        for path in paths:
            result = self.load_experiment(path.parent)
            if result is not None:
                if filter_fn is None or filter_fn(result):
                    results.append(result)

        logger.info(f"Loaded {len(results)} experiments")
        return results

    def load_by_name(self, name: str) -> List[ExperimentResult]:
        """Load all experiments with matching name."""
        return self.load_all(filter_fn=lambda r: r.experiment_name == name)

    def load_by_dataset(self, dataset: str) -> List[ExperimentResult]:
        """Load all experiments for a dataset."""
        return self.load_all(filter_fn=lambda r: r.dataset == dataset)

    def _load_config(self, output_dir: Path) -> Dict[str, Any]:
        """Load experiment configuration."""
        # Try YAML first
        yaml_path = output_dir / "config.yaml"
        if yaml_path.exists() and HAS_YAML:
            try:
                with open(yaml_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass

        # Fall back to JSON
        json_path = output_dir / "config.json"
        return self._load_json(json_path) or {}

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file safely."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None


def discover_experiments(
    base_dir: str = "./outputs",
    **kwargs,
) -> List[Path]:
    """Discover experiment output directories."""
    loader = ResultLoader(base_dir)
    return loader.discover_experiments(**kwargs)


def load_experiment(output_dir: str) -> Optional[ExperimentResult]:
    """Load a single experiment."""
    loader = ResultLoader()
    return loader.load_experiment(Path(output_dir))


def load_all_experiments(
    base_dir: str = "./outputs",
    filter_fn: Optional[callable] = None,
) -> List[ExperimentResult]:
    """Load all experiments from directory."""
    loader = ResultLoader(base_dir)
    return loader.load_all(filter_fn)

