"""Loss functions for different task types."""

from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """Loss for regression tasks (MSE, MAE, Huber)."""

    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        """
        Initialize regression loss.

        Args:
            loss_type: Type of loss: 'mse', 'mae', 'huber'.
            reduction: Reduction method: 'mean', 'sum', 'none'.
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown regression loss type: {loss_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute regression loss.

        Args:
            pred: Predictions [B, 1] or [B, num_targets].
            target: Targets [B, 1] or [B, num_targets].
            mask: Optional mask for valid targets.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        return self.loss_fn(pred, target)


class ClassificationLoss(nn.Module):
    """Loss for classification tasks (CrossEntropy, BCE)."""

    def __init__(
        self,
        loss_type: str = "cross_entropy",
        num_classes: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize classification loss.

        Args:
            loss_type: Type of loss: 'cross_entropy', 'bce', 'bce_logits'.
            num_classes: Number of classes (for cross_entropy).
            class_weights: Optional class weights for imbalanced data.
            reduction: Reduction method.
        """
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes

        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        elif loss_type == "bce":
            self.loss_fn = nn.BCELoss(reduction=reduction)
        elif loss_type == "bce_logits":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown classification loss type: {loss_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            pred: Predictions [B, num_classes] or [B, 1] for binary.
            target: Targets [B] for CE or [B, 1] for BCE.
            mask: Optional mask for valid targets.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if self.loss_type == "cross_entropy":
            # Ensure target is long for CE
            if target.dtype != torch.long:
                target = target.long()
            if target.dim() > 1:
                target = target.squeeze(-1)
        elif self.loss_type in ["bce", "bce_logits"]:
            # Ensure target is float for BCE
            if target.dtype != torch.float:
                target = target.float()
            # For BCE, if pred is [B, 2] and target is [B], convert to [B, 1] format
            if pred.dim() == 2 and pred.size(1) == 2 and target.dim() == 1:
                # Use only the first class logit for binary classification
                pred = pred[:, 0:1]  # [B, 2] -> [B, 1]
                target = target.unsqueeze(1)  # [B] -> [B, 1]

        return self.loss_fn(pred, target)


class EdgePredictionLoss(nn.Module):
    """Loss for edge prediction tasks (link prediction, contact prediction)."""

    def __init__(self, reduction: str = "mean"):
        """
        Initialize edge prediction loss.

        Args:
            reduction: Reduction method.
        """
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge prediction loss.

        Args:
            pred: Edge predictions [num_edges] or [num_edges, 1].
            target: Edge targets [num_edges] or [num_edges, 1].
            mask: Optional mask for valid edges.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if target.dtype != torch.float:
            target = target.float()

        return self.loss_fn(pred.view(-1), target.view(-1))


def get_loss_fn(
    task_type: str,
    loss_type: Optional[str] = None,
    num_classes: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory function to get appropriate loss function.

    Args:
        task_type: Task type: 'regression', 'classification', 'multilabel', 'edge_prediction'.
        loss_type: Specific loss type (optional, defaults based on task).
        num_classes: Number of classes for classification.
        class_weights: Optional class weights.

    Returns:
        Loss function module.
    """
    if task_type == "regression":
        return RegressionLoss(loss_type=loss_type or "mse")
    elif task_type == "classification":
        if num_classes == 2 or loss_type in ["bce", "bce_logits"]:
            return ClassificationLoss(
                loss_type=loss_type or "bce_logits",
                num_classes=num_classes,
                class_weights=class_weights,
            )
        else:
            return ClassificationLoss(
                loss_type=loss_type or "cross_entropy",
                num_classes=num_classes,
                class_weights=class_weights,
            )
    elif task_type == "multilabel":
        # Multilabel classification uses BCE with logits
        return ClassificationLoss(
            loss_type=loss_type or "bce_logits",
            num_classes=num_classes,
            class_weights=class_weights,
        )
    elif task_type == "edge_prediction":
        return EdgePredictionLoss()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

