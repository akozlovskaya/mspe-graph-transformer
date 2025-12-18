"""Stratification utilities for distance-based evaluation."""

from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np


def create_distance_buckets(
    max_distance: int,
    bucket_size: int = 1,
    include_unreachable: bool = True,
) -> List[Tuple[int, int]]:
    """
    Create distance bucket intervals.

    Args:
        max_distance: Maximum distance to consider.
        bucket_size: Size of each bucket.
        include_unreachable: Whether to include unreachable pairs bucket.

    Returns:
        List of (start, end) tuples defining bucket intervals.
    """
    buckets = []

    for start in range(0, max_distance + 1, bucket_size):
        end = min(start + bucket_size - 1, max_distance)
        buckets.append((start, end))

    if include_unreachable:
        buckets.append((-1, -1))  # Unreachable bucket

    return buckets


def stratify_by_distance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    distances: torch.Tensor,
    buckets: List[Tuple[int, int]],
    mask: Optional[torch.Tensor] = None,
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Stratify predictions and targets by distance buckets.

    Args:
        predictions: Model predictions [N, ...].
        targets: Ground truth targets [N, ...].
        distances: Distance values for each sample [N].
        buckets: List of (start, end) bucket intervals.
        mask: Optional mask for valid samples [N].

    Returns:
        Dictionary mapping bucket intervals to {predictions, targets, count}.
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
        distances = distances[mask]

    stratified = {}

    for bucket in buckets:
        start, end = bucket

        if start == -1:
            # Unreachable bucket
            bucket_mask = distances == -1
        else:
            bucket_mask = (distances >= start) & (distances <= end)

        if bucket_mask.sum() > 0:
            stratified[bucket] = {
                "predictions": predictions[bucket_mask],
                "targets": targets[bucket_mask],
                "count": bucket_mask.sum().item(),
            }
        else:
            stratified[bucket] = {
                "predictions": torch.tensor([]),
                "targets": torch.tensor([]),
                "count": 0,
            }

    return stratified


def stratify_node_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    distances_to_target: torch.Tensor,
    buckets: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Stratify node-level predictions by distance to target/label nodes.

    Args:
        predictions: Node predictions [num_nodes, ...].
        targets: Node targets [num_nodes, ...].
        distances_to_target: Distance from each node to nearest target [num_nodes].
        buckets: Distance bucket intervals.

    Returns:
        Stratified predictions per bucket.
    """
    return stratify_by_distance(predictions, targets, distances_to_target, buckets)


def stratify_edge_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    edge_distances: torch.Tensor,
    buckets: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Stratify edge-level predictions by edge distance.

    Args:
        predictions: Edge predictions [num_edges, ...].
        targets: Edge targets [num_edges, ...].
        edge_distances: Distance for each edge [num_edges].
        buckets: Distance bucket intervals.

    Returns:
        Stratified predictions per bucket.
    """
    return stratify_by_distance(predictions, targets, edge_distances, buckets)


def stratify_graph_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    graph_diameters: torch.Tensor,
    buckets: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Stratify graph-level predictions by graph diameter.

    Args:
        predictions: Graph predictions [num_graphs, ...].
        targets: Graph targets [num_graphs, ...].
        graph_diameters: Diameter of each graph [num_graphs].
        buckets: Diameter bucket intervals.

    Returns:
        Stratified predictions per bucket.
    """
    return stratify_by_distance(predictions, targets, graph_diameters, buckets)


def compute_bucket_statistics(
    stratified: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
) -> Dict[str, Dict[Tuple[int, int], float]]:
    """
    Compute basic statistics for each bucket.

    Args:
        stratified: Output from stratify_by_distance.

    Returns:
        Dictionary with statistics per bucket.
    """
    stats = {
        "count": {},
        "mean_pred": {},
        "mean_target": {},
        "std_pred": {},
        "std_target": {},
    }

    for bucket, data in stratified.items():
        stats["count"][bucket] = data["count"]

        if data["count"] > 0:
            pred = data["predictions"].float()
            target = data["targets"].float()

            stats["mean_pred"][bucket] = pred.mean().item()
            stats["mean_target"][bucket] = target.mean().item()
            stats["std_pred"][bucket] = pred.std().item() if len(pred) > 1 else 0.0
            stats["std_target"][bucket] = target.std().item() if len(target) > 1 else 0.0
        else:
            stats["mean_pred"][bucket] = float("nan")
            stats["mean_target"][bucket] = float("nan")
            stats["std_pred"][bucket] = float("nan")
            stats["std_target"][bucket] = float("nan")

    return stats


def aggregate_stratified_results(
    results_list: List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]],
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    Aggregate stratified results from multiple batches/graphs.

    Args:
        results_list: List of stratified results.

    Returns:
        Aggregated stratified results.
    """
    if not results_list:
        return {}

    # Get all buckets
    all_buckets = set()
    for results in results_list:
        all_buckets.update(results.keys())

    aggregated = {}

    for bucket in all_buckets:
        all_preds = []
        all_targets = []
        total_count = 0

        for results in results_list:
            if bucket in results and results[bucket]["count"] > 0:
                all_preds.append(results[bucket]["predictions"])
                all_targets.append(results[bucket]["targets"])
                total_count += results[bucket]["count"]

        if all_preds:
            aggregated[bucket] = {
                "predictions": torch.cat(all_preds, dim=0),
                "targets": torch.cat(all_targets, dim=0),
                "count": total_count,
            }
        else:
            aggregated[bucket] = {
                "predictions": torch.tensor([]),
                "targets": torch.tensor([]),
                "count": 0,
            }

    return aggregated


class DistanceStratifier:
    """Class for managing distance-based stratification across batches."""

    def __init__(
        self,
        max_distance: int = 20,
        bucket_size: int = 1,
        include_unreachable: bool = True,
    ):
        """
        Initialize stratifier.

        Args:
            max_distance: Maximum distance to consider.
            bucket_size: Size of each bucket.
            include_unreachable: Whether to include unreachable pairs.
        """
        self.max_distance = max_distance
        self.bucket_size = bucket_size
        self.buckets = create_distance_buckets(
            max_distance, bucket_size, include_unreachable
        )
        self.accumulated = []

    def reset(self):
        """Reset accumulated results."""
        self.accumulated = []

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
            predictions: Predictions tensor.
            targets: Targets tensor.
            distances: Distances tensor.
            mask: Optional validity mask.
        """
        stratified = stratify_by_distance(
            predictions.detach().cpu(),
            targets.detach().cpu(),
            distances.detach().cpu(),
            self.buckets,
            mask.detach().cpu() if mask is not None else None,
        )
        self.accumulated.append(stratified)

    def compute(self) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """
        Compute aggregated stratified results.

        Returns:
            Aggregated results per bucket.
        """
        return aggregate_stratified_results(self.accumulated)

    def get_bucket_counts(self) -> Dict[Tuple[int, int], int]:
        """Get sample counts per bucket."""
        aggregated = self.compute()
        return {bucket: data["count"] for bucket, data in aggregated.items()}

