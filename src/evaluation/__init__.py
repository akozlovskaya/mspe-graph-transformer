"""Evaluation framework for Graph Transformers."""

from .distance_metrics import (
    compute_shortest_path_distances,
    compute_shortest_path_distances_sparse,
    compute_landmark_distances,
    approximate_pairwise_distance,
    add_distance_info_to_data,
    get_node_pair_distances,
    compute_distance_histogram,
    get_distance_to_label_nodes,
)

from .stratification import (
    create_distance_buckets,
    stratify_by_distance,
    stratify_node_predictions,
    stratify_edge_predictions,
    stratify_graph_predictions,
    compute_bucket_statistics,
    aggregate_stratified_results,
    DistanceStratifier,
)

from .long_range import (
    compute_metrics_per_bucket,
    compute_relative_performance_drop,
    compute_area_under_distance_curve,
    find_effective_receptive_field,
    LongRangeEvaluator,
    evaluate_layer_wise_long_range,
    compare_pe_configurations,
)

from .probes import (
    DistantNodeFeatureProbe,
    LabelPropagationProbe,
    PathParityProbe,
    NodeCountingProbe,
    SyntheticLongRangeTask,
    create_probing_dataset,
)

from .utils import (
    compute_model_config_hash,
    save_evaluation_results,
    print_evaluation_summary,
    plot_distance_performance,
    plot_layer_wise_analysis,
    plot_pe_comparison,
    compute_graph_diameter,
    add_graph_statistics,
    EvaluationLogger,
)


__all__ = [
    # Distance metrics
    "compute_shortest_path_distances",
    "compute_shortest_path_distances_sparse",
    "compute_landmark_distances",
    "approximate_pairwise_distance",
    "add_distance_info_to_data",
    "get_node_pair_distances",
    "compute_distance_histogram",
    "get_distance_to_label_nodes",
    # Stratification
    "create_distance_buckets",
    "stratify_by_distance",
    "stratify_node_predictions",
    "stratify_edge_predictions",
    "stratify_graph_predictions",
    "compute_bucket_statistics",
    "aggregate_stratified_results",
    "DistanceStratifier",
    # Long-range metrics
    "compute_metrics_per_bucket",
    "compute_relative_performance_drop",
    "compute_area_under_distance_curve",
    "find_effective_receptive_field",
    "LongRangeEvaluator",
    "evaluate_layer_wise_long_range",
    "compare_pe_configurations",
    # Probes
    "DistantNodeFeatureProbe",
    "LabelPropagationProbe",
    "PathParityProbe",
    "NodeCountingProbe",
    "SyntheticLongRangeTask",
    "create_probing_dataset",
    # Utils
    "compute_model_config_hash",
    "save_evaluation_results",
    "print_evaluation_summary",
    "plot_distance_performance",
    "plot_layer_wise_analysis",
    "plot_pe_comparison",
    "compute_graph_diameter",
    "add_graph_statistics",
    "EvaluationLogger",
]
