"""Metrics for evaluation."""

from typing import Dict, List, Optional, Union
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


class MetricAccumulator:
    """Accumulates predictions and targets for batch-wise metric computation."""

    def __init__(self):
        """Initialize accumulator."""
        self.predictions = []
        self.targets = []
        self.masks = []

    def reset(self):
        """Reset accumulated values."""
        self.predictions = []
        self.targets = []
        self.masks = []

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Add batch predictions and targets.

        Args:
            pred: Predictions tensor.
            target: Targets tensor.
            mask: Optional mask tensor.
        """
        self.predictions.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
        if mask is not None:
            self.masks.append(mask.detach().cpu())

    def get_all(self) -> tuple:
        """
        Get all accumulated predictions and targets.

        Returns:
            Tuple of (predictions, targets, mask or None).
        """
        pred = torch.cat(self.predictions, dim=0)
        target = torch.cat(self.targets, dim=0)
        mask = torch.cat(self.masks, dim=0) if self.masks else None
        return pred, target, mask


class RegressionMetrics:
    """Metrics for regression tasks."""

    def __init__(self):
        """Initialize regression metrics."""
        self.accumulator = MetricAccumulator()

    def reset(self):
        """Reset metrics."""
        self.accumulator.reset()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update with batch data."""
        self.accumulator.update(pred, target, mask)

    def compute(self) -> Dict[str, float]:
        """
        Compute all regression metrics.

        Returns:
            Dictionary of metric names and values.
        """
        pred, target, mask = self.accumulator.get_all()

        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        pred_np = pred.numpy().flatten()
        target_np = target.numpy().flatten()

        mse = mean_squared_error(target_np, pred_np)
        mae = mean_absolute_error(target_np, pred_np)
        rmse = np.sqrt(mse)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
        }


class ClassificationMetrics:
    """Metrics for classification tasks."""

    def __init__(self, num_classes: int = 2, task: str = "binary"):
        """
        Initialize classification metrics.

        Args:
            num_classes: Number of classes.
            task: Task type: 'binary', 'multiclass', 'multilabel'.
        """
        self.num_classes = num_classes
        self.task = task
        self.accumulator = MetricAccumulator()

    def reset(self):
        """Reset metrics."""
        self.accumulator.reset()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update with batch data."""
        self.accumulator.update(pred, target, mask)

    def compute(self) -> Dict[str, float]:
        """
        Compute all classification metrics.

        Returns:
            Dictionary of metric names and values.
        """
        pred, target, mask = self.accumulator.get_all()

        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        metrics = {}

        if self.task == "binary":
            # Binary classification
            pred_np = pred.numpy().flatten()
            target_np = target.numpy().flatten()

            # Apply sigmoid if needed
            if pred_np.min() < 0 or pred_np.max() > 1:
                pred_probs = torch.sigmoid(pred).numpy().flatten()
            else:
                pred_probs = pred_np

            pred_labels = (pred_probs > 0.5).astype(int)

            metrics["accuracy"] = float(accuracy_score(target_np, pred_labels))
            metrics["f1"] = float(f1_score(target_np, pred_labels, zero_division=0))

            try:
                metrics["roc_auc"] = float(roc_auc_score(target_np, pred_probs))
            except ValueError:
                metrics["roc_auc"] = 0.0

            try:
                metrics["ap"] = float(average_precision_score(target_np, pred_probs))
            except ValueError:
                metrics["ap"] = 0.0

        elif self.task == "multiclass":
            # Multiclass classification
            pred_np = pred.numpy()
            target_np = target.numpy().flatten().astype(int)

            pred_labels = pred_np.argmax(axis=1)
            metrics["accuracy"] = float(accuracy_score(target_np, pred_labels))
            metrics["f1_macro"] = float(
                f1_score(target_np, pred_labels, average="macro", zero_division=0)
            )

            try:
                pred_probs = torch.softmax(pred, dim=1).numpy()
                metrics["roc_auc"] = float(
                    roc_auc_score(target_np, pred_probs, multi_class="ovr")
                )
            except ValueError:
                metrics["roc_auc"] = 0.0

        elif self.task == "multilabel":
            # Multilabel classification
            pred_np = pred.numpy()
            target_np = target.numpy()

            # Apply sigmoid
            if pred_np.min() < 0 or pred_np.max() > 1:
                pred_probs = torch.sigmoid(pred).numpy()
            else:
                pred_probs = pred_np

            pred_labels = (pred_probs > 0.5).astype(int)

            metrics["accuracy"] = float(accuracy_score(target_np, pred_labels))

            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(target_np, pred_probs, average="macro")
                )
            except ValueError:
                metrics["roc_auc"] = 0.0

            try:
                metrics["ap"] = float(
                    average_precision_score(target_np, pred_probs, average="macro")
                )
            except ValueError:
                metrics["ap"] = 0.0

        return metrics


class EdgeMetrics:
    """Metrics for edge prediction tasks."""

    def __init__(self):
        """Initialize edge metrics."""
        self.accumulator = MetricAccumulator()

    def reset(self):
        """Reset metrics."""
        self.accumulator.reset()

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Update with batch data."""
        self.accumulator.update(pred, target, mask)

    def compute(self) -> Dict[str, float]:
        """
        Compute all edge prediction metrics.

        Returns:
            Dictionary of metric names and values.
        """
        pred, target, mask = self.accumulator.get_all()

        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        pred_np = pred.numpy().flatten()
        target_np = target.numpy().flatten()

        # Apply sigmoid if needed
        if pred_np.min() < 0 or pred_np.max() > 1:
            pred_probs = torch.sigmoid(pred).numpy().flatten()
        else:
            pred_probs = pred_np

        metrics = {}

        try:
            metrics["roc_auc"] = float(roc_auc_score(target_np, pred_probs))
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["ap"] = float(average_precision_score(target_np, pred_probs))
        except ValueError:
            metrics["ap"] = 0.0

        # Precision at k
        k = min(100, len(pred_probs))
        top_k_indices = np.argsort(pred_probs)[-k:]
        metrics["precision@100"] = float(target_np[top_k_indices].mean())

        return metrics


def get_metrics(
    task_type: str, num_classes: int = 2
) -> Union[RegressionMetrics, ClassificationMetrics, EdgeMetrics]:
    """
    Factory function to get appropriate metrics.

    Args:
        task_type: Task type: 'regression', 'classification', 'edge_prediction'.
        num_classes: Number of classes for classification.

    Returns:
        Metrics object.
    """
    if task_type == "regression":
        return RegressionMetrics()
    elif task_type == "classification":
        if num_classes == 2:
            return ClassificationMetrics(num_classes=2, task="binary")
        else:
            return ClassificationMetrics(num_classes=num_classes, task="multiclass")
    elif task_type == "multilabel":
        return ClassificationMetrics(task="multilabel")
    elif task_type == "edge_prediction":
        return EdgeMetrics()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

