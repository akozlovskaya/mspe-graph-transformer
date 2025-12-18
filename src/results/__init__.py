"""Results processing and visualization module for thesis-ready outputs."""

from .loader import (
    ExperimentResult,
    ResultLoader,
    load_experiment,
    load_all_experiments,
    discover_experiments,
)

from .aggregate import (
    ResultAggregator,
    aggregate_by_seed,
    aggregate_by_group,
    group_results,
    filter_results,
)

from .tables import (
    TableGenerator,
    make_performance_table,
    make_ablation_table,
    make_long_range_table,
    make_efficiency_table,
    export_table,
)

from .plots import (
    PlotGenerator,
    plot_performance_vs_distance,
    plot_performance_vs_size,
    plot_runtime_vs_accuracy,
    plot_memory_vs_accuracy,
    plot_depth_vs_distance,
    save_figure,
)

from .formatting import (
    TableFormatter,
    PlotStyle,
    METRIC_FORMATS,
    PE_NAMES,
    DATASET_NAMES,
    MODEL_NAMES,
)

from .utils import (
    ensure_output_dir,
    get_experiment_id,
    safe_mean,
    safe_std,
)


__all__ = [
    # Loader
    "ExperimentResult",
    "ResultLoader",
    "load_experiment",
    "load_all_experiments",
    "discover_experiments",
    # Aggregate
    "ResultAggregator",
    "aggregate_by_seed",
    "aggregate_by_group",
    "group_results",
    "filter_results",
    # Tables
    "TableGenerator",
    "make_performance_table",
    "make_ablation_table",
    "make_long_range_table",
    "make_efficiency_table",
    "export_table",
    # Plots
    "PlotGenerator",
    "plot_performance_vs_distance",
    "plot_performance_vs_size",
    "plot_runtime_vs_accuracy",
    "plot_memory_vs_accuracy",
    "plot_depth_vs_distance",
    "save_figure",
    # Formatting
    "TableFormatter",
    "PlotStyle",
    "METRIC_FORMATS",
    "PE_NAMES",
    "DATASET_NAMES",
    "MODEL_NAMES",
    # Utils
    "ensure_output_dir",
    "get_experiment_id",
    "safe_mean",
    "safe_std",
]

