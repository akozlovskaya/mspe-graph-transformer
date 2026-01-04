"""Plot generation for thesis-ready figures."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from .loader import ExperimentResult
from .aggregate import AggregatedResult, aggregate_by_seed
from .formatting import PlotStyle, PE_NAMES, DATASET_NAMES, MODEL_NAMES


logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generator for thesis-ready plots."""

    def __init__(self, style: Optional[PlotStyle] = None):
        """
        Initialize plot generator.

        Args:
            style: Plot style configuration.
        """
        self.style = style or PlotStyle()
        self._apply_style()

    def _apply_style(self):
        """Apply matplotlib style settings."""
        plt.rcParams.update({
            "font.family": self.style.font_family,
            "font.size": self.style.font_size,
            "axes.labelsize": self.style.label_size,
            "axes.titlesize": self.style.title_size,
            "legend.fontsize": self.style.legend_size,
            "xtick.labelsize": self.style.tick_size,
            "ytick.labelsize": self.style.tick_size,
            "figure.figsize": self.style.figure_size,
            "figure.dpi": self.style.dpi,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def performance_vs_distance(
        self,
        results: List[ExperimentResult],
        metric: str = "mae",
        max_distance: int = 20,
        group_by: str = "node_pe",
    ) -> plt.Figure:
        """
        Plot performance vs graph distance.

        Args:
            results: Experiment results.
            metric: Metric to plot.
            max_distance: Maximum distance to show.
            group_by: Key to group curves by.

        Returns:
            Matplotlib figure.
        """
        aggregated = aggregate_by_seed(results, ["dataset", group_by])

        fig, ax = plt.subplots()

        for i, agg in enumerate(aggregated):
            group_value = agg.group_values.get(group_by, "unknown")
            label = PE_NAMES.get(group_value, group_value)
            color = self.style.get_color(i)
            marker = self.style.get_marker(i)

            distances = []
            values = []
            errors = []

            for dist in range(1, max_distance + 1):
                dist_key = dist
                if dist_key in agg.distance_metrics:
                    dist_data = agg.distance_metrics[dist_key]
                    if metric in dist_data:
                        distances.append(dist)
                        values.append(dist_data[metric]["mean"])
                        errors.append(dist_data[metric].get("std", 0))

            if distances:
                ax.errorbar(
                    distances, values, yerr=errors,
                    label=label, color=color, marker=marker,
                    capsize=3, markersize=6, linewidth=1.5,
                )

        ax.set_xlabel("Graph Distance")
        ax.set_ylabel(self._format_metric_label(metric))
        ax.set_title("Performance vs Distance")
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", framealpha=0.9)
        ax.set_xlim(0, max_distance + 1)

        fig.tight_layout()
        return fig

    def performance_vs_size(
        self,
        results: List[ExperimentResult],
        metric: str = "mae",
        size_metric: str = "parameters",
        group_by: str = "model",
    ) -> plt.Figure:
        """
        Plot performance vs model size.

        Args:
            results: Experiment results.
            metric: Performance metric.
            size_metric: Size metric ('parameters', 'runtime', 'memory').
            group_by: Key to group points by.

        Returns:
            Matplotlib figure.
        """
        aggregated = aggregate_by_seed(results, ["dataset", "model", "node_pe", "relative_pe"])

        fig, ax = plt.subplots()

        # Group by specified key
        groups = {}
        for agg in aggregated:
            group_value = agg.group_values.get(group_by, "unknown")
            if group_value not in groups:
                groups[group_value] = {"x": [], "y": [], "yerr": []}

            # Get size
            if size_metric == "parameters":
                size = agg.parameters
            elif size_metric == "runtime":
                size = agg.runtime_ms["mean"] if agg.runtime_ms else None
            elif size_metric == "memory":
                size = agg.memory_mb["mean"] if agg.memory_mb else None
            else:
                size = None

            # Get performance
            if metric in agg.metrics and size is not None:
                groups[group_value]["x"].append(size)
                groups[group_value]["y"].append(agg.metrics[metric]["mean"])
                groups[group_value]["yerr"].append(agg.metrics[metric].get("std", 0))

        for i, (group, data) in enumerate(groups.items()):
            if data["x"]:
                color = self.style.get_color(i)
                marker = self.style.get_marker(i)
                label = MODEL_NAMES.get(group, group)

                ax.errorbar(
                    data["x"], data["y"], yerr=data["yerr"],
                    label=label, color=color, marker=marker,
                    capsize=3, markersize=8, linestyle="none",
                )

        ax.set_xlabel(self._format_size_label(size_metric))
        ax.set_ylabel(self._format_metric_label(metric))
        ax.set_title(f"Performance vs {size_metric.capitalize()}")
        ax.legend(loc="best", framealpha=0.9)

        # Log scale for parameters
        if size_metric == "parameters":
            ax.set_xscale("log")

        fig.tight_layout()
        return fig

    def runtime_vs_accuracy(
        self,
        results: List[ExperimentResult],
        accuracy_metric: str = "mae",
        lower_is_better: bool = True,
    ) -> plt.Figure:
        """
        Plot runtime vs accuracy trade-off.

        Args:
            results: Experiment results.
            accuracy_metric: Accuracy metric.
            lower_is_better: Whether lower metric is better.

        Returns:
            Matplotlib figure.
        """
        aggregated = aggregate_by_seed(
            results, ["model", "node_pe", "relative_pe"]
        )

        fig, ax = plt.subplots()

        for i, agg in enumerate(aggregated):
            if agg.runtime_ms is None or accuracy_metric not in agg.metrics:
                continue

            runtime = agg.runtime_ms["mean"]
            accuracy = agg.metrics[accuracy_metric]["mean"]
            acc_err = agg.metrics[accuracy_metric].get("std", 0)

            # Label
            pe_label = PE_NAMES.get(agg.group_values.get("node_pe", ""), "")
            rel_label = PE_NAMES.get(agg.group_values.get("relative_pe", ""), "")
            label = f"{pe_label}+{rel_label}" if rel_label else pe_label

            color = self.style.get_color(i)
            marker = self.style.get_marker(i)

            ax.errorbar(
                runtime, accuracy, yerr=acc_err,
                label=label, color=color, marker=marker,
                capsize=3, markersize=10, linestyle="none",
            )

        ax.set_xlabel("Runtime (ms)")
        ax.set_ylabel(self._format_metric_label(accuracy_metric))
        ax.set_title("Runtime vs Accuracy Trade-off")
        ax.legend(loc="best", framealpha=0.9, ncol=2)

        # Indicate better direction
        if lower_is_better:
            ax.annotate("Better →", xy=(0.95, 0.05), xycoords="axes fraction",
                       fontsize=10, ha="right")
        else:
            ax.annotate("Better →", xy=(0.95, 0.95), xycoords="axes fraction",
                       fontsize=10, ha="right")

        fig.tight_layout()
        return fig

    def memory_vs_accuracy(
        self,
        results: List[ExperimentResult],
        accuracy_metric: str = "mae",
    ) -> plt.Figure:
        """
        Plot memory vs accuracy trade-off.

        Args:
            results: Experiment results.
            accuracy_metric: Accuracy metric.

        Returns:
            Matplotlib figure.
        """
        aggregated = aggregate_by_seed(
            results, ["model", "node_pe", "relative_pe"]
        )

        fig, ax = plt.subplots()

        for i, agg in enumerate(aggregated):
            if agg.memory_mb is None or accuracy_metric not in agg.metrics:
                continue

            memory = agg.memory_mb["mean"]
            accuracy = agg.metrics[accuracy_metric]["mean"]
            acc_err = agg.metrics[accuracy_metric].get("std", 0)

            pe_label = PE_NAMES.get(agg.group_values.get("node_pe", ""), "")
            label = pe_label

            color = self.style.get_color(i)
            marker = self.style.get_marker(i)

            ax.errorbar(
                memory, accuracy, yerr=acc_err,
                label=label, color=color, marker=marker,
                capsize=3, markersize=10, linestyle="none",
            )

        ax.set_xlabel("Memory (MB)")
        ax.set_ylabel(self._format_metric_label(accuracy_metric))
        ax.set_title("Memory vs Accuracy Trade-off")
        ax.legend(loc="best", framealpha=0.9)

        fig.tight_layout()
        return fig

    def depth_vs_distance(
        self,
        results: List[ExperimentResult],
        metric: str = "mae",
        distances: List[int] = None,
    ) -> plt.Figure:
        """
        Plot layer depth vs effective distance.

        Args:
            results: Experiment results.
            metric: Metric to plot.
            distances: Distances to show.

        Returns:
            Matplotlib figure.
        """
        if distances is None:
            distances = [1, 5, 10, 15, 20]

        aggregated = aggregate_by_seed(results, ["num_layers", "node_pe"])

        fig, ax = plt.subplots()

        # Group by PE type
        pe_groups = {}
        for agg in aggregated:
            pe = agg.group_values.get("node_pe", "none")
            if pe not in pe_groups:
                pe_groups[pe] = []
            pe_groups[pe].append(agg)

        for i, (pe, aggs) in enumerate(pe_groups.items()):
            color = self.style.get_color(i)
            label = PE_NAMES.get(pe, pe)

            depths = []
            effective_distances = []

            for agg in sorted(aggs, key=lambda x: x.group_values.get("num_layers", 0)):
                depth = agg.group_values.get("num_layers", 0)
                depths.append(depth)

                # Find effective distance (where performance drops significantly)
                if agg.long_range and "effective_receptive_field" in agg.long_range:
                    eff_dist = agg.long_range["effective_receptive_field"].get("mean", 0)
                else:
                    eff_dist = 0
                effective_distances.append(eff_dist)

            if depths:
                ax.plot(depths, effective_distances, label=label,
                       color=color, marker="o", linewidth=2, markersize=8)

        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Effective Receptive Field")
        ax.set_title("Depth vs Effective Distance")
        ax.legend(loc="best", framealpha=0.9)

        fig.tight_layout()
        return fig

    def ablation_heatmap(
        self,
        results: List[ExperimentResult],
        row_key: str = "node_pe",
        col_key: str = "relative_pe",
        metric: str = "mae",
    ) -> plt.Figure:
        """
        Plot ablation heatmap.

        Args:
            results: Experiment results.
            row_key: Key for rows.
            col_key: Key for columns.
            metric: Metric to show.

        Returns:
            Matplotlib figure.
        """
        aggregated = aggregate_by_seed(results, [row_key, col_key])

        # Collect unique values
        row_values = sorted(set(agg.group_values.get(row_key, "") for agg in aggregated))
        col_values = sorted(set(agg.group_values.get(col_key, "") for agg in aggregated))

        # Build matrix
        matrix = np.full((len(row_values), len(col_values)), np.nan)

        for agg in aggregated:
            row_val = agg.group_values.get(row_key, "")
            col_val = agg.group_values.get(col_key, "")
            if metric in agg.metrics:
                r = row_values.index(row_val)
                c = col_values.index(col_val)
                matrix[r, c] = agg.metrics[metric]["mean"]

        fig, ax = plt.subplots()

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

        # Labels
        ax.set_xticks(range(len(col_values)))
        ax.set_xticklabels([PE_NAMES.get(v, v) for v in col_values], rotation=45, ha="right")
        ax.set_yticks(range(len(row_values)))
        ax.set_yticklabels([PE_NAMES.get(v, v) for v in row_values])

        ax.set_xlabel(col_key.replace("_", " ").title())
        ax.set_ylabel(row_key.replace("_", " ").title())
        ax.set_title(f"{metric.upper()} Ablation")

        # Add values
        for i in range(len(row_values)):
            for j in range(len(col_values)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}",
                           ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, label=metric.upper())
        fig.tight_layout()
        return fig

    def _format_metric_label(self, metric: str) -> str:
        """Format metric name for axis label."""
        labels = {
            "mae": "MAE ↓",
            "mse": "MSE ↓",
            "rmse": "RMSE ↓",
            "loss": "Loss ↓",
            "accuracy": "Accuracy ↑",
            "roc_auc": "ROC-AUC ↑",
            "ap": "AP ↑",
        }
        return labels.get(metric, metric.upper())

    def _format_size_label(self, size_metric: str) -> str:
        """Format size metric for axis label."""
        labels = {
            "parameters": "Parameters",
            "runtime": "Runtime (ms)",
            "memory": "Memory (MB)",
        }
        return labels.get(size_metric, size_metric)


def plot_performance_vs_distance(
    results: List[ExperimentResult],
    **kwargs,
) -> plt.Figure:
    """Generate performance vs distance plot."""
    gen = PlotGenerator()
    return gen.performance_vs_distance(results, **kwargs)


def plot_performance_vs_size(
    results: List[ExperimentResult],
    **kwargs,
) -> plt.Figure:
    """Generate performance vs size plot."""
    gen = PlotGenerator()
    return gen.performance_vs_size(results, **kwargs)


def plot_runtime_vs_accuracy(
    results: List[ExperimentResult],
    **kwargs,
) -> plt.Figure:
    """Generate runtime vs accuracy plot."""
    gen = PlotGenerator()
    return gen.runtime_vs_accuracy(results, **kwargs)


def plot_memory_vs_accuracy(
    results: List[ExperimentResult],
    **kwargs,
) -> plt.Figure:
    """Generate memory vs accuracy plot."""
    gen = PlotGenerator()
    return gen.memory_vs_accuracy(results, **kwargs)


def plot_depth_vs_distance(
    results: List[ExperimentResult],
    **kwargs,
) -> plt.Figure:
    """Generate depth vs distance plot."""
    gen = PlotGenerator()
    return gen.depth_vs_distance(results, **kwargs)


def save_figure(
    fig: plt.Figure,
    output_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> List[str]:
    """
    Save figure to file(s).

    Args:
        fig: Matplotlib figure.
        output_path: Base output path (without extension).
        formats: List of formats ('pdf', 'png').
        dpi: Resolution for raster formats.

    Returns:
        List of saved file paths.
    """
    if formats is None:
        formats = ["pdf", "png"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(str(path))
        logger.info(f"Saved figure to {path}")

    plt.close(fig)
    return saved

