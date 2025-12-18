"""Sweep management for systematic experiments."""

import itertools
import json
import logging
import random
from typing import Dict, Any, List, Optional, Iterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from .registry import ExperimentConfig, ExperimentRegistry
from .runner import ExperimentRunner
from .utils import generate_experiment_id


logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for a sweep."""

    name: str
    base_config: Dict[str, Any]
    parameters: Dict[str, List[Any]]
    sweep_type: str = "grid"  # grid, random, seed
    num_samples: int = 10  # For random sweep
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def get_num_experiments(self) -> int:
        """Get total number of experiments in sweep."""
        if self.sweep_type == "grid":
            return len(list(self._generate_grid_configs()))
        elif self.sweep_type == "random":
            return self.num_samples
        elif self.sweep_type == "seed":
            return len(self.seeds)
        return 0

    def _generate_grid_configs(self) -> Iterator[Dict[str, Any]]:
        """Generate configurations for grid sweep."""
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        for values in itertools.product(*param_values):
            config = deepcopy(self.base_config)
            for name, value in zip(param_names, values):
                _set_nested_value(config, name, value)
            yield config

    def _generate_random_configs(self) -> Iterator[Dict[str, Any]]:
        """Generate configurations for random sweep."""
        for _ in range(self.num_samples):
            config = deepcopy(self.base_config)
            for name, values in self.parameters.items():
                value = random.choice(values)
                _set_nested_value(config, name, value)
            yield config

    def _generate_seed_configs(self) -> Iterator[Dict[str, Any]]:
        """Generate configurations for seed sweep."""
        for seed in self.seeds:
            config = deepcopy(self.base_config)
            config["seed"] = seed
            yield config

    def generate_configs(self) -> Iterator[Dict[str, Any]]:
        """Generate all configurations for sweep."""
        if self.sweep_type == "grid":
            yield from self._generate_grid_configs()
        elif self.sweep_type == "random":
            yield from self._generate_random_configs()
        elif self.sweep_type == "seed":
            yield from self._generate_seed_configs()


def _set_nested_value(d: Dict, key: str, value: Any):
    """Set value in nested dictionary using dot notation."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_nested_value(d: Dict, key: str, default=None) -> Any:
    """Get value from nested dictionary using dot notation."""
    keys = key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


class GridSweep:
    """Grid sweep over parameter combinations."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        parameters: Dict[str, List[Any]],
        name_template: str = "{base}_{params}",
    ):
        """
        Initialize grid sweep.

        Args:
            base_config: Base experiment configuration.
            parameters: Dict mapping parameter paths to value lists.
            name_template: Template for experiment names.
        """
        self.base_config = base_config
        self.parameters = parameters
        self.name_template = name_template

    def generate(self) -> Iterator[ExperimentConfig]:
        """Generate all experiment configurations."""
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        for values in itertools.product(*param_values):
            config_dict = self.base_config.to_dict()

            # Apply parameter values
            param_str_parts = []
            for name, value in zip(param_names, values):
                _set_nested_value(config_dict, name, value)
                short_name = name.split(".")[-1]
                param_str_parts.append(f"{short_name}={value}")

            # Generate name
            param_str = "_".join(param_str_parts)
            config_dict["name"] = self.name_template.format(
                base=self.base_config.name,
                params=param_str,
            )

            yield ExperimentConfig.from_dict(config_dict)

    def __len__(self):
        """Get number of experiments."""
        lengths = [len(v) for v in self.parameters.values()]
        total = 1
        for l in lengths:
            total *= l
        return total


class RandomSweep:
    """Random sweep over parameter space."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        parameters: Dict[str, List[Any]],
        num_samples: int = 10,
        seed: int = 42,
    ):
        """
        Initialize random sweep.

        Args:
            base_config: Base experiment configuration.
            parameters: Dict mapping parameter paths to value lists.
            num_samples: Number of random samples.
            seed: Random seed.
        """
        self.base_config = base_config
        self.parameters = parameters
        self.num_samples = num_samples
        self.seed = seed

    def generate(self) -> Iterator[ExperimentConfig]:
        """Generate random experiment configurations."""
        random.seed(self.seed)

        for i in range(self.num_samples):
            config_dict = self.base_config.to_dict()

            param_str_parts = []
            for name, values in self.parameters.items():
                value = random.choice(values)
                _set_nested_value(config_dict, name, value)
                short_name = name.split(".")[-1]
                param_str_parts.append(f"{short_name}={value}")

            param_str = "_".join(param_str_parts)
            config_dict["name"] = f"{self.base_config.name}_random_{i}_{param_str}"

            yield ExperimentConfig.from_dict(config_dict)

    def __len__(self):
        return self.num_samples


class SeedSweep:
    """Sweep over random seeds for reproducibility studies."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        seeds: List[int] = None,
    ):
        """
        Initialize seed sweep.

        Args:
            base_config: Base experiment configuration.
            seeds: List of seeds to sweep over.
        """
        self.base_config = base_config
        self.seeds = seeds or [42, 123, 456, 789, 1024]

    def generate(self) -> Iterator[ExperimentConfig]:
        """Generate experiments with different seeds."""
        for seed in self.seeds:
            config_dict = self.base_config.to_dict()
            config_dict["seed"] = seed
            config_dict["name"] = f"{self.base_config.name}_seed_{seed}"
            yield ExperimentConfig.from_dict(config_dict)

    def __len__(self):
        return len(self.seeds)


@dataclass
class SweepState:
    """State for tracking sweep progress."""

    sweep_name: str
    total_experiments: int
    completed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def save(self, path: Path):
        """Save state to file."""
        state_dict = {
            "sweep_name": self.sweep_name,
            "total_experiments": self.total_experiments,
            "completed": self.completed,
            "failed": self.failed,
            "pending": self.pending,
            "results": self.results,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SweepState":
        """Load state from file."""
        with open(path) as f:
            state_dict = json.load(f)
        return cls(**state_dict)


class SweepRunner:
    """Runner for executing sweeps."""

    def __init__(
        self,
        sweep: SweepConfig,
        output_dir: str = "./outputs/sweeps",
        parallel: bool = False,
        max_workers: int = 4,
    ):
        """
        Initialize sweep runner.

        Args:
            sweep: Sweep configuration.
            output_dir: Base output directory.
            parallel: Whether to run experiments in parallel.
            max_workers: Maximum parallel workers.
        """
        self.sweep = sweep
        self.output_dir = Path(output_dir) / sweep.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize state
        self.state_path = self.output_dir / "sweep_state.json"
        if self.state_path.exists():
            self.state = SweepState.load(self.state_path)
        else:
            self.state = SweepState(
                sweep_name=sweep.name,
                total_experiments=sweep.get_num_experiments(),
            )

    def run(
        self,
        on_experiment_complete: Optional[Callable] = None,
        on_experiment_failed: Optional[Callable] = None,
    ) -> SweepState:
        """
        Run the sweep.

        Args:
            on_experiment_complete: Callback when experiment completes.
            on_experiment_failed: Callback when experiment fails.

        Returns:
            Final sweep state.
        """
        logger.info(f"Starting sweep: {self.sweep.name}")
        logger.info(f"Total experiments: {self.state.total_experiments}")

        self.state.start_time = datetime.now().isoformat()

        configs = list(self.sweep.generate_configs())

        for i, config_dict in enumerate(configs):
            exp_name = config_dict.get("name", f"exp_{i}")

            # Skip if already completed or failed
            if exp_name in self.state.completed:
                logger.info(f"Skipping completed: {exp_name}")
                continue
            if exp_name in self.state.failed:
                logger.info(f"Skipping failed: {exp_name}")
                continue

            logger.info(f"Running experiment {i + 1}/{len(configs)}: {exp_name}")

            try:
                config = ExperimentConfig.from_dict(config_dict)
                exp_output_dir = self.output_dir / exp_name

                runner = ExperimentRunner(config, str(exp_output_dir))
                results = runner.run_all(
                    run_long_range=True,
                    run_profiling=False,
                )

                self.state.completed.append(exp_name)
                self.state.results[exp_name] = {
                    "status": results["status"],
                    "evaluation": results.get("evaluation", {}),
                }

                if on_experiment_complete:
                    on_experiment_complete(exp_name, results)

            except Exception as e:
                logger.error(f"Experiment failed: {exp_name} - {e}")
                self.state.failed.append(exp_name)

                if on_experiment_failed:
                    on_experiment_failed(exp_name, e)

            # Save state after each experiment
            self.state.save(self.state_path)

        self.state.end_time = datetime.now().isoformat()
        self.state.save(self.state_path)

        # Aggregate results
        self._aggregate_results()

        return self.state

    def _aggregate_results(self):
        """Aggregate sweep results."""
        from .utils import aggregate_results

        all_results = []
        for exp_name in self.state.completed:
            results_path = self.output_dir / exp_name / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    all_results.append(json.load(f))

        if all_results:
            aggregated = aggregate_results(all_results)
            with open(self.output_dir / "aggregated_results.json", "w") as f:
                json.dump(aggregated, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get sweep summary."""
        return {
            "sweep_name": self.sweep.name,
            "total": self.state.total_experiments,
            "completed": len(self.state.completed),
            "failed": len(self.state.failed),
            "pending": self.state.total_experiments - len(self.state.completed) - len(self.state.failed),
        }


def run_sweep(
    sweep: SweepConfig,
    output_dir: str = "./outputs/sweeps",
    **kwargs,
) -> SweepState:
    """
    Run a sweep.

    Args:
        sweep: Sweep configuration.
        output_dir: Output directory.
        **kwargs: Additional runner arguments.

    Returns:
        Final sweep state.
    """
    runner = SweepRunner(sweep, output_dir, **kwargs)
    return runner.run()


def resume_sweep(sweep_dir: str) -> SweepState:
    """
    Resume an incomplete sweep.

    Args:
        sweep_dir: Path to sweep directory.

    Returns:
        Final sweep state.
    """
    sweep_dir = Path(sweep_dir)
    state_path = sweep_dir / "sweep_state.json"

    if not state_path.exists():
        raise ValueError(f"No sweep state found in {sweep_dir}")

    # Load sweep config
    # For simplicity, we'll need the original sweep config
    # In practice, this should be saved with the state
    logger.info(f"Resuming sweep from {sweep_dir}")

    state = SweepState.load(state_path)
    logger.info(f"Completed: {len(state.completed)}, Failed: {len(state.failed)}")

    return state

