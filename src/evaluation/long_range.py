"""Long-range evaluation metrics and analysis."""

from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)

from .stratification import (
    DistanceStratifier,
    create_distance_buckets,
    stratify_by_distance,
)


def compute_metrics_per_bucket(
    stratified: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    task_type: str = "regression",
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute task-specific metrics for each distance bucket.

    Args:
        stratified: Stratified predictions from DistanceStratifier.
        task_type: Task type: 'regression', 'classification', 'binary'.

    Returns:
        Dictionary mapping buckets to metric dictionaries.
    """
    metrics_per_bucket = {}

    for bucket, data in stratified.items():
        if data["count"] == 0:
            metrics_per_bucket[bucket] = {
                "count": 0,
                "metric": float("nan"),
            }
            continue

        pred = data["predictions"].numpy()
        target = data["targets"].numpy()

        metrics = {"count": data["count"]}

        if task_type == "regression":
            metrics["mae"] = float(mean_absolute_error(target, pred))
            metrics["mse"] = float(mean_squared_error(target, pred))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["metric"] = metrics["mae"]  # Primary metric

        elif task_type == "classification":
            pred_labels = pred.argmax(axis=1) if pred.ndim > 1 else (pred > 0.5).astype(int)
            target_labels = target.flatten().astype(int)
            metrics["accuracy"] = float(accuracy_score(target_labels, pred_labels))
            metrics["metric"] = metrics["accuracy"]

        elif task_type == "binary":
            pred_flat = pred.flatten()
            target_flat = target.flatten()

            # Apply sigmoid if needed
            if pred_flat.min() < 0 or pred_flat.max() > 1:
                pred_probs = 1 / (1 + np.exp(-pred_flat))
            else:
                pred_probs = pred_flat

            pred_labels = (pred_probs > 0.5).astype(int)
            metrics["accuracy"] = float(accuracy_score(target_flat, pred_labels))

            try:
                metrics["roc_auc"] = float(roc_auc_score(target_flat, pred_probs))
            except ValueError:
                metrics["roc_auc"] = float("nan")

            try:
                metrics["ap"] = float(average_precision_score(target_flat, pred_probs))
            except ValueError:
                metrics["ap"] = float("nan")

            metrics["metric"] = metrics["roc_auc"]

        metrics_per_bucket[bucket] = metrics

    return metrics_per_bucket


def compute_relative_performance_drop(
    metrics_per_bucket: Dict[Tuple[int, int], Dict[str, float]],
    metric_name: str = "metric",
    baseline_bucket: Tuple[int, int] = (0, 0),
    higher_is_better: bool = True,
) -> Dict[Tuple[int, int], float]:
    """
    Compute relative performance drop compared to short-range baseline.

    Args:
        metrics_per_bucket: Metrics per bucket from compute_metrics_per_bucket.
        metric_name: Name of metric to compare.
        baseline_bucket: Bucket to use as baseline (typically shortest distance).
        higher_is_better: Whether higher metric values are better.

    Returns:
        Dictionary mapping buckets to relative drop (0 = same, 1 = complete collapse).
    """
    if baseline_bucket not in metrics_per_bucket:
        # Find first non-empty bucket
        for bucket in sorted(metrics_per_bucket.keys()):
            if metrics_per_bucket[bucket]["count"] > 0:
                baseline_bucket = bucket
                break

    baseline_value = metrics_per_bucket.get(baseline_bucket, {}).get(metric_name)

    if baseline_value is None or np.isnan(baseline_value):
        return {bucket: float("nan") for bucket in metrics_per_bucket}

    drops = {}

    for bucket, metrics in metrics_per_bucket.items():
        value = metrics.get(metric_name)

        if value is None or np.isnan(value):
            drops[bucket] = float("nan")
        elif higher_is_better:
            # Drop = (baseline - current) / baseline
            drops[bucket] = (baseline_value - value) / max(abs(baseline_value), 1e-8)
        else:
            # For metrics where lower is better (e.g., MAE)
            drops[bucket] = (value - baseline_value) / max(abs(baseline_value), 1e-8)

    return drops


def compute_area_under_distance_curve(
    metrics_per_bucket: Dict[Tuple[int, int], Dict[str, float]],
    metric_name: str = "metric",
    max_distance: int = 20,
    normalize: bool = True,
) -> float:
    """
    Compute area under the distance-performance curve.

    Higher AUC indicates better long-range performance.

    Args:
        metrics_per_bucket: Metrics per bucket.
        metric_name: Metric to use.
        max_distance: Maximum distance for normalization.
        normalize: Whether to normalize by max possible area.

    Returns:
        AUC value.
    """
    # Sort buckets by distance
    sorted_buckets = sorted(
        [b for b in metrics_per_bucket.keys() if b[0] >= 0],
        key=lambda x: x[0]
    )

    if not sorted_buckets:
        return float("nan")

    # Compute AUC using trapezoidal rule
    auc = 0.0
    prev_dist = 0
    prev_value = None

    for bucket in sorted_buckets:
        value = metrics_per_bucket[bucket].get(metric_name)

        if value is None or np.isnan(value):
            continue

        dist = (bucket[0] + bucket[1]) / 2  # Bucket midpoint

        if prev_value is not None:
            # Trapezoidal area
            auc += (dist - prev_dist) * (prev_value + value) / 2

        prev_dist = dist
        prev_value = value

    if normalize and max_distance > 0:
        # Normalize by max possible area (assuming metric ranges 0-1)
        auc = auc / max_distance

    return auc


def find_effective_receptive_field(
    metrics_per_bucket: Dict[Tuple[int, int], Dict[str, float]],
    metric_name: str = "metric",
    threshold: float = 0.5,
    relative_to_baseline: bool = True,
) -> int:
    """
    Find the maximum distance where performance is still above threshold.

    Args:
        metrics_per_bucket: Metrics per bucket.
        metric_name: Metric to use.
        threshold: Performance threshold (absolute or relative).
        relative_to_baseline: If True, threshold is relative to short-range performance.

    Returns:
        Maximum effective distance, or -1 if not found.
    """
    sorted_buckets = sorted(
        [b for b in metrics_per_bucket.keys() if b[0] >= 0],
        key=lambda x: x[0]
    )

    if not sorted_buckets:
        return -1

    # Get baseline
    baseline = None
    for bucket in sorted_buckets:
        value = metrics_per_bucket[bucket].get(metric_name)
        if value is not None and not np.isnan(value):
            baseline = value
            break

    if baseline is None:
        return -1

    if relative_to_baseline:
        threshold_value = baseline * threshold
    else:
        threshold_value = threshold

    max_effective_distance = -1

    for bucket in sorted_buckets:
        value = metrics_per_bucket[bucket].get(metric_name)

        if value is not None and not np.isnan(value) and value >= threshold_value:
            max_effective_distance = bucket[1]

    return max_effective_distance


class LongRangeEvaluator:
    """Comprehensive long-range evaluation class."""

    def __init__(
        self,
        max_distance: int = 20,
        bucket_size: int = 1,
        task_type: str = "regression",
    ):
        """
        Initialize evaluator.

        Args:
            max_distance: Maximum distance to consider.
            bucket_size: Size of distance buckets.
            task_type: Task type for metrics.
        """
        self.max_distance = max_distance
        self.bucket_size = bucket_size
        self.task_type = task_type
        self.stratifier = DistanceStratifier(max_distance, bucket_size)

    def reset(self):
        """Reset accumulated results."""
        self.stratifier.reset()

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        distances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Add batch of results.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            distances: Distance values for stratification.
            mask: Optional validity mask.
        """
        self.stratifier.update(predictions, targets, distances, mask)

    def compute(self) -> Dict[str, Any]:
        """
        Compute all long-range metrics.

        Returns:
            Dictionary with comprehensive evaluation results.
        """
        stratified = self.stratifier.compute()

        # Metrics per bucket
        metrics_per_bucket = compute_metrics_per_bucket(
            stratified, self.task_type
        )

        # Relative drops
        higher_is_better = self.task_type in ["classification", "binary"]
        relative_drops = compute_relative_performance_drop(
            metrics_per_bucket,
            higher_is_better=higher_is_better,
        )

        # AUC
        auc = compute_area_under_distance_curve(
            metrics_per_bucket, max_distance=self.max_distance
        )

        # Effective receptive field
        erf = find_effective_receptive_field(
            metrics_per_bucket, threshold=0.5
        )

        # Aggregate metrics
        all_preds = []
        all_targets = []
        for data in stratified.values():
            if data["count"] > 0:
                all_preds.append(data["predictions"])
                all_targets.append(data["targets"])

        if all_preds:
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            aggregate_stratified = {
                (0, self.max_distance): {
                    "predictions": all_preds,
                    "targets": all_targets,
                    "count": len(all_preds),
                }
            }
            aggregate_metrics = compute_metrics_per_bucket(
                aggregate_stratified, self.task_type
            )
        else:
            aggregate_metrics = {}

        return {
            "metrics_per_bucket": metrics_per_bucket,
            "relative_drops": relative_drops,
            "auc": auc,
            "effective_receptive_field": erf,
            "aggregate_metrics": aggregate_metrics,
            "bucket_counts": self.stratifier.get_bucket_counts(),
            "config": {
                "max_distance": self.max_distance,
                "bucket_size": self.bucket_size,
                "task_type": self.task_type,
            },
        }

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        results = self.compute()

        summary = {
            "auc": results["auc"],
            "effective_receptive_field": results["effective_receptive_field"],
        }

        # Add aggregate metric
        if results["aggregate_metrics"]:
            for bucket, metrics in results["aggregate_metrics"].items():
                summary["aggregate_metric"] = metrics.get("metric", float("nan"))
                break

        # Short-range vs long-range comparison
        short_range = None
        long_range = None

        for bucket in sorted(results["metrics_per_bucket"].keys()):
            if bucket[0] >= 0 and results["metrics_per_bucket"][bucket]["count"] > 0:
                value = results["metrics_per_bucket"][bucket].get("metric")
                if value is not None and not np.isnan(value):
                    if short_range is None:
                        short_range = value
                    long_range = value

        if short_range is not None:
            summary["short_range_metric"] = short_range
        if long_range is not None:
            summary["long_range_metric"] = long_range
        if short_range is not None and long_range is not None:
            summary["performance_ratio"] = long_range / max(short_range, 1e-8)

        return summary


def evaluate_layer_wise_long_range(
    hidden_states: List[torch.Tensor],
    targets: torch.Tensor,
    distances: torch.Tensor,
    linear_probes: Optional[List[torch.nn.Module]] = None,
    max_distance: int = 20,
    task_type: str = "regression",
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate long-range performance at each layer.

    Args:
        hidden_states: List of hidden states per layer [L x (N, D)].
        targets: Target values.
        distances: Distance values for stratification.
        linear_probes: Optional list of linear probes for each layer.
        max_distance: Maximum distance for stratification.
        task_type: Task type for metrics.

    Returns:
        Dictionary mapping layer index to evaluation results.
    """
    results = {}

    for layer_idx, hidden in enumerate(hidden_states):
        evaluator = LongRangeEvaluator(
            max_distance=max_distance,
            task_type=task_type,
        )

        if linear_probes is not None and layer_idx < len(linear_probes):
            # Apply linear probe
            with torch.no_grad():
                predictions = linear_probes[layer_idx](hidden)
        else:
            # Use hidden states directly
            predictions = hidden

        evaluator.update(predictions, targets, distances)
        results[layer_idx] = evaluator.compute()

    return results


def compare_pe_configurations(
    results_dict: Dict[str, Dict[str, Any]],
    metric_name: str = "auc",
) -> Dict[str, float]:
    """
    Compare long-range performance across PE configurations.

    Args:
        results_dict: Dictionary mapping PE config name to evaluation results.
        metric_name: Metric to compare.

    Returns:
        Dictionary with comparison statistics.
    """
    comparison = {}

    for config_name, results in results_dict.items():
        if metric_name == "auc":
            comparison[config_name] = results.get("auc", float("nan"))
        elif metric_name == "erf":
            comparison[config_name] = results.get("effective_receptive_field", -1)
        else:
            summary = results.get("aggregate_metrics", {})
            if summary:
                for bucket, metrics in summary.items():
                    comparison[config_name] = metrics.get(metric_name, float("nan"))
                    break

    # Rank configurations
    valid_configs = [(k, v) for k, v in comparison.items() if not np.isnan(v)]
    sorted_configs = sorted(valid_configs, key=lambda x: x[1], reverse=True)

    return {
        "values": comparison,
        "ranking": [c[0] for c in sorted_configs],
        "best": sorted_configs[0][0] if sorted_configs else None,
    }

