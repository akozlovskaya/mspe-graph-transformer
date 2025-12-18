"""Result aggregation utilities."""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from .loader import ExperimentResult


logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Aggregated results across multiple experiments."""

    # Group identifiers
    group_key: Tuple[str, ...]
    group_values: Dict[str, Any]

    # Number of experiments
    n_experiments: int
    seeds: List[int]

    # Aggregated metrics (mean ± std)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Long-range aggregation
    long_range: Dict[str, Dict[str, float]] = field(default_factory=dict)
    distance_metrics: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Profiling aggregation
    runtime_ms: Optional[Dict[str, float]] = None
    memory_mb: Optional[Dict[str, float]] = None
    parameters: Optional[int] = None

    # Source experiments
    experiment_ids: List[str] = field(default_factory=list)

    def get_metric(
        self,
        name: str,
        stat: str = "mean",
    ) -> Optional[float]:
        """Get aggregated metric value."""
        if name in self.metrics:
            return self.metrics[name].get(stat)
        return None

    def format_metric(
        self,
        name: str,
        precision: int = 4,
    ) -> str:
        """Format metric as 'mean ± std'."""
        if name not in self.metrics:
            return "—"

        m = self.metrics[name]
        mean = m.get("mean", 0)
        std = m.get("std", 0)

        if std > 0:
            return f"{mean:.{precision}f} ± {std:.{precision}f}"
        return f"{mean:.{precision}f}"


class ResultAggregator:
    """Aggregator for experiment results."""

    def __init__(self, results: List[ExperimentResult]):
        """
        Initialize aggregator.

        Args:
            results: List of experiment results.
        """
        self.results = results

    def group_by(
        self,
        keys: List[str],
    ) -> Dict[Tuple, List[ExperimentResult]]:
        """
        Group results by specified keys.

        Args:
            keys: List of attribute names to group by.

        Returns:
            Dictionary mapping group keys to results.
        """
        groups = defaultdict(list)

        for result in self.results:
            group_key = tuple(self._get_value(result, key) for key in keys)
            groups[group_key].append(result)

        return dict(groups)

    def aggregate(
        self,
        group_keys: List[str],
        metrics: Optional[List[str]] = None,
    ) -> List[AggregatedResult]:
        """
        Aggregate results by group keys.

        Args:
            group_keys: Keys to group by.
            metrics: Specific metrics to aggregate (None for all).

        Returns:
            List of AggregatedResult objects.
        """
        groups = self.group_by(group_keys)
        aggregated = []

        for group_key, group_results in groups.items():
            agg = self._aggregate_group(
                group_key,
                group_keys,
                group_results,
                metrics,
            )
            aggregated.append(agg)

        return aggregated

    def _aggregate_group(
        self,
        group_key: Tuple,
        group_keys: List[str],
        results: List[ExperimentResult],
        metrics: Optional[List[str]],
    ) -> AggregatedResult:
        """Aggregate a single group of results."""
        # Group values
        group_values = {k: v for k, v in zip(group_keys, group_key)}

        # Collect metrics
        metric_values = defaultdict(list)
        for result in results:
            for name, value in result.test_metrics.items():
                if metrics is None or name in metrics:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metric_values[name].append(value)

        # Compute statistics
        agg_metrics = {}
        for name, values in metric_values.items():
            if values:
                arr = np.array(values)
                agg_metrics[name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": len(values),
                }

        # Aggregate long-range
        lr_metrics = self._aggregate_long_range(results)

        # Aggregate profiling
        runtime_agg, memory_agg = self._aggregate_profiling(results)

        # Get parameters (should be same for all)
        parameters = results[0].parameters if results else None

        return AggregatedResult(
            group_key=group_key,
            group_values=group_values,
            n_experiments=len(results),
            seeds=[r.seed for r in results],
            metrics=agg_metrics,
            long_range=lr_metrics.get("summary", {}),
            distance_metrics=lr_metrics.get("by_distance", {}),
            runtime_ms=runtime_agg,
            memory_mb=memory_agg,
            parameters=parameters,
            experiment_ids=[r.experiment_id for r in results],
        )

    def _aggregate_long_range(
        self,
        results: List[ExperimentResult],
    ) -> Dict[str, Any]:
        """Aggregate long-range metrics."""
        summary_values = defaultdict(list)
        distance_values = defaultdict(lambda: defaultdict(list))

        for result in results:
            lr = result.long_range

            # Summary metrics
            for key in ["auc", "effective_receptive_field", "relative_drop"]:
                if key in lr and lr[key] is not None:
                    summary_values[key].append(lr[key])

            # Per-distance metrics
            for dist, metrics in result.distance_metrics.items():
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        distance_values[dist][name].append(value)

        # Compute summary stats
        summary = {}
        for key, values in summary_values.items():
            if values:
                arr = np.array(values)
                summary[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }

        # Compute per-distance stats
        by_distance = {}
        for dist, metrics in distance_values.items():
            by_distance[dist] = {}
            for name, values in metrics.items():
                if values:
                    arr = np.array(values)
                    by_distance[dist][name] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                    }

        return {"summary": summary, "by_distance": by_distance}

    def _aggregate_profiling(
        self,
        results: List[ExperimentResult],
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Aggregate profiling metrics."""
        runtimes = [r.runtime_ms for r in results if r.runtime_ms is not None]
        memories = [r.memory_mb for r in results if r.memory_mb is not None]

        runtime_agg = None
        if runtimes:
            arr = np.array(runtimes)
            runtime_agg = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        memory_agg = None
        if memories:
            arr = np.array(memories)
            memory_agg = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        return runtime_agg, memory_agg

    def _get_value(self, result: ExperimentResult, key: str) -> Any:
        """Get value from result by key path."""
        if key == "dataset":
            return result.dataset
        elif key == "model":
            return result.model
        elif key == "node_pe":
            return result.node_pe_type
        elif key == "relative_pe":
            return result.relative_pe_type
        elif key == "num_layers":
            return result.num_layers
        elif key == "seed":
            return result.seed
        elif "." in key:
            # Nested config access
            parts = key.split(".")
            value = result.config
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value
        else:
            return getattr(result, key, None)


def group_results(
    results: List[ExperimentResult],
    keys: List[str],
) -> Dict[Tuple, List[ExperimentResult]]:
    """Group results by keys."""
    aggregator = ResultAggregator(results)
    return aggregator.group_by(keys)


def aggregate_by_seed(
    results: List[ExperimentResult],
    group_keys: List[str] = None,
) -> List[AggregatedResult]:
    """
    Aggregate results over seeds.

    Args:
        results: List of results.
        group_keys: Keys to group by (excluding seed).

    Returns:
        Aggregated results.
    """
    if group_keys is None:
        group_keys = ["dataset", "model", "node_pe", "relative_pe"]

    aggregator = ResultAggregator(results)
    return aggregator.aggregate(group_keys)


def aggregate_by_group(
    results: List[ExperimentResult],
    group_keys: List[str],
    metrics: Optional[List[str]] = None,
) -> List[AggregatedResult]:
    """Aggregate results by specified group keys."""
    aggregator = ResultAggregator(results)
    return aggregator.aggregate(group_keys, metrics)


def filter_results(
    results: List[ExperimentResult],
    filter_fn: Optional[Callable[[ExperimentResult], bool]] = None,
    **filters,
) -> List[ExperimentResult]:
    """
    Filter results by criteria.

    Args:
        results: List of results.
        filter_fn: Custom filter function.
        **filters: Attribute filters (e.g., dataset="zinc").

    Returns:
        Filtered results.
    """
    filtered = results

    if filter_fn is not None:
        filtered = [r for r in filtered if filter_fn(r)]

    for key, value in filters.items():
        if key == "dataset":
            filtered = [r for r in filtered if r.dataset == value]
        elif key == "model":
            filtered = [r for r in filtered if r.model == value]
        elif key == "node_pe":
            filtered = [r for r in filtered if r.node_pe_type == value]
        elif key == "relative_pe":
            filtered = [r for r in filtered if r.relative_pe_type == value]
        elif key == "status":
            filtered = [r for r in filtered if r.status == value]
        elif key == "complete":
            filtered = [r for r in filtered if r.is_complete() == value]

    return filtered

