"""Experiment orchestration and ablation management."""

from .registry import (
    ExperimentRegistry,
    ExperimentConfig,
    register_experiment,
    get_experiment,
    list_experiments,
    validate_config,
)

from .runner import (
    ExperimentRunner,
    run_experiment,
    run_training_only,
    run_evaluation_only,
)

from .sweeps import (
    SweepConfig,
    GridSweep,
    RandomSweep,
    SeedSweep,
    run_sweep,
    resume_sweep,
)

from .logging import (
    ExperimentLogger,
    setup_experiment_logging,
    log_config,
    log_metrics,
    log_artifact,
)

from .utils import (
    generate_experiment_id,
    get_output_dir,
    save_experiment_results,
    load_experiment_results,
    aggregate_results,
    results_to_dataframe,
)


__all__ = [
    # Registry
    "ExperimentRegistry",
    "ExperimentConfig",
    "register_experiment",
    "get_experiment",
    "list_experiments",
    "validate_config",
    # Runner
    "ExperimentRunner",
    "run_experiment",
    "run_training_only",
    "run_evaluation_only",
    # Sweeps
    "SweepConfig",
    "GridSweep",
    "RandomSweep",
    "SeedSweep",
    "run_sweep",
    "resume_sweep",
    # Logging
    "ExperimentLogger",
    "setup_experiment_logging",
    "log_config",
    "log_metrics",
    "log_artifact",
    # Utils
    "generate_experiment_id",
    "get_output_dir",
    "save_experiment_results",
    "load_experiment_results",
    "aggregate_results",
    "results_to_dataframe",
]

