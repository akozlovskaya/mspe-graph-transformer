"""Utility functions for evaluation."""

from typing import Dict, List, Tuple, Optional, Any
import json
import hashlib
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def compute_model_config_hash(config: Dict) -> str:
    """
    Compute hash of model configuration for result tracking.

    Args:
        config: Configuration dictionary.

    Returns:
        Hash string.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str,
    format: str = "json",
):
    """
    Save evaluation results to file.

    Args:
        results: Results dictionary.
        output_path: Output file path.
        format: Output format: 'json' or 'csv'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert(results), f, indent=2)

    elif format == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Flatten results for CSV
            flat_results = flatten_dict(results)
            writer.writerow(["key", "value"])
            for key, value in flat_results.items():
                writer.writerow([key, value])


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print formatted evaluation summary.

    Args:
        results: Evaluation results dictionary.
    """
    print("\n" + "=" * 60)
    print("Long-Range Evaluation Summary")
    print("=" * 60)

    # Config
    if "config" in results:
        print(f"\nConfiguration:")
        for key, value in results["config"].items():
            print(f"  {key}: {value}")

    # Overall metrics
    print(f"\nOverall Metrics:")
    if "auc" in results:
        print(f"  AUC (distance-performance): {results['auc']:.4f}")
    if "effective_receptive_field" in results:
        print(f"  Effective Receptive Field: {results['effective_receptive_field']}")

    # Aggregate metrics
    if "aggregate_metrics" in results:
        print(f"\nAggregate Metrics:")
        for bucket, metrics in results["aggregate_metrics"].items():
            for name, value in metrics.items():
                if name != "count":
                    print(f"  {name}: {value:.4f}")

    # Per-bucket metrics
    if "metrics_per_bucket" in results:
        print(f"\nMetrics per Distance Bucket:")
        print(f"  {'Bucket':<15} {'Count':<10} {'Metric':<12} {'Drop':<10}")
        print(f"  {'-'*47}")

        drops = results.get("relative_drops", {})

        for bucket in sorted(results["metrics_per_bucket"].keys()):
            metrics = results["metrics_per_bucket"][bucket]
            drop = drops.get(bucket, float("nan"))

            bucket_str = f"[{bucket[0]}, {bucket[1]}]" if bucket[0] >= 0 else "unreachable"
            metric_val = metrics.get("metric", float("nan"))

            if not np.isnan(metric_val):
                print(
                    f"  {bucket_str:<15} {metrics['count']:<10} "
                    f"{metric_val:<12.4f} {drop:<10.2%}"
                )

    print("\n" + "=" * 60)


def plot_distance_performance(
    metrics_per_bucket: Dict[Tuple[int, int], Dict[str, float]],
    metric_name: str = "metric",
    title: str = "Performance vs Distance",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot performance vs distance curve.

    Args:
        metrics_per_bucket: Metrics per distance bucket.
        metric_name: Name of metric to plot.
        title: Plot title.
        output_path: Path to save figure. If None, displays plot.
        figsize: Figure size.
    """
    # Extract data
    distances = []
    values = []
    counts = []

    for bucket in sorted(metrics_per_bucket.keys()):
        if bucket[0] < 0:  # Skip unreachable
            continue

        value = metrics_per_bucket[bucket].get(metric_name)
        if value is not None and not np.isnan(value):
            distances.append((bucket[0] + bucket[1]) / 2)
            values.append(value)
            counts.append(metrics_per_bucket[bucket]["count"])

    if not distances:
        return

    fig, ax1 = plt.subplots(figsize=figsize)

    # Performance curve
    color = "tab:blue"
    ax1.set_xlabel("Graph Distance")
    ax1.set_ylabel(metric_name.capitalize(), color=color)
    ax1.plot(distances, values, "o-", color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis="y", labelcolor=color)

    # Sample count bars
    ax2 = ax1.twinx()
    color = "tab:gray"
    ax2.set_ylabel("Sample Count", color=color)
    ax2.bar(distances, counts, alpha=0.3, color=color, width=0.8)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_layer_wise_analysis(
    layer_results: Dict[int, Dict[str, Any]],
    metric_name: str = "auc",
    title: str = "Layer-wise Long-Range Performance",
    output_path: Optional[str] = None,
):
    """
    Plot layer-wise long-range performance.

    Args:
        layer_results: Results per layer from evaluate_layer_wise_long_range.
        metric_name: Metric to plot.
        title: Plot title.
        output_path: Path to save figure.
    """
    layers = sorted(layer_results.keys())
    values = []

    for layer in layers:
        if metric_name == "auc":
            values.append(layer_results[layer].get("auc", float("nan")))
        elif metric_name == "erf":
            values.append(layer_results[layer].get("effective_receptive_field", -1))
        else:
            summary = layer_results[layer].get("aggregate_metrics", {})
            val = float("nan")
            for bucket, metrics in summary.items():
                val = metrics.get(metric_name, float("nan"))
                break
            values.append(val)

    plt.figure(figsize=(10, 6))
    plt.plot(layers, values, "o-", linewidth=2, markersize=10)
    plt.xlabel("Layer")
    plt.ylabel(metric_name.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pe_comparison(
    comparison_results: Dict[str, float],
    title: str = "PE Configuration Comparison",
    output_path: Optional[str] = None,
):
    """
    Plot comparison of different PE configurations.

    Args:
        comparison_results: Values dict from compare_pe_configurations.
        title: Plot title.
        output_path: Path to save figure.
    """
    names = list(comparison_results.keys())
    values = [comparison_results[n] for n in names]

    # Filter out NaN values
    valid = [(n, v) for n, v in zip(names, values) if not np.isnan(v)]
    if not valid:
        return

    names, values = zip(*sorted(valid, key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values)

    # Color best bar differently
    bars[0].set_color("tab:green")

    plt.xlabel("Score")
    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_graph_diameter(data) -> int:
    """
    Compute diameter of a graph.

    Args:
        data: PyG Data object.

    Returns:
        Graph diameter (max shortest path distance).
    """
    from .distance_metrics import compute_shortest_path_distances

    distances = compute_shortest_path_distances(
        data.edge_index, data.num_nodes, max_distance=data.num_nodes
    )

    # Diameter is max finite distance
    valid_distances = distances[distances >= 0]
    if len(valid_distances) == 0:
        return 0

    return valid_distances.max().item()


def add_graph_statistics(data) -> None:
    """
    Add graph statistics to Data object in-place.

    Args:
        data: PyG Data object.
    """
    # Number of nodes and edges
    data.num_nodes_stat = data.num_nodes
    data.num_edges_stat = data.edge_index.size(1) // 2  # Undirected

    # Average degree
    degrees = torch.zeros(data.num_nodes)
    for i in range(data.edge_index.size(1)):
        degrees[data.edge_index[0, i]] += 1

    data.avg_degree = degrees.mean().item()
    data.max_degree = degrees.max().item()

    # Diameter (expensive for large graphs)
    if data.num_nodes <= 1000:
        data.diameter = compute_graph_diameter(data)
    else:
        data.diameter = -1  # Too expensive


class EvaluationLogger:
    """Logger for tracking evaluation progress."""

    def __init__(self, verbose: bool = True):
        """
        Initialize logger.

        Args:
            verbose: Whether to print progress.
        """
        self.verbose = verbose
        self.logs = []

    def log(self, message: str):
        """Log a message."""
        self.logs.append(message)
        if self.verbose:
            print(message)

    def log_batch(self, batch_idx: int, total: int, metrics: Dict[str, float]):
        """Log batch progress."""
        if self.verbose:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Batch {batch_idx + 1}/{total} | {metrics_str}")

    def save(self, path: str):
        """Save logs to file."""
        with open(path, "w") as f:
            f.write("\n".join(self.logs))

