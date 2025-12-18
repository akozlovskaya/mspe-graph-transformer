"""Training and evaluation loop utilities."""

from typing import Dict, Any, Optional, Callable, List
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from .metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    EdgeMetrics,
    MetricAccumulator,
)


logger = logging.getLogger(__name__)


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Execute single training step.

    Args:
        model: PyTorch model.
        batch: PyG batch object.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device.
        grad_clip: Gradient clipping norm.
        scaler: GradScaler for mixed precision.

    Returns:
        Dictionary with loss and optional grad_norm.
    """
    model.train()
    batch = batch.to(device)
    target = batch.y.to(device)

    optimizer.zero_grad()

    results = {}

    if scaler is not None:
        # Mixed precision
        with autocast():
            pred = model(batch)
            loss = loss_fn(pred, target)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            results["grad_norm"] = grad_norm.item()

        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard precision
        pred = model(batch)
        loss = loss_fn(pred, target)

        loss.backward()

        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            results["grad_norm"] = grad_norm.item()

        optimizer.step()

    results["loss"] = loss.item()
    return results


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Execute single evaluation step.

    Args:
        model: PyTorch model.
        batch: PyG batch object.
        loss_fn: Loss function.
        device: Device.

    Returns:
        Dictionary with loss and predictions.
    """
    model.eval()
    batch = batch.to(device)
    target = batch.y.to(device)

    pred = model(batch)
    loss = loss_fn(pred, target)

    return {
        "loss": loss.item(),
        "pred": pred.detach(),
        "target": target.detach(),
    }


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    metrics,
    device: torch.device,
    epochs: int,
    grad_clip: Optional[float] = None,
    mixed_precision: bool = False,
    log_every: int = 50,
    callbacks: Optional[List[Callable]] = None,
) -> Dict[str, List]:
    """
    Run complete training loop.

    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        metrics: Metrics object.
        device: Device.
        epochs: Number of epochs.
        grad_clip: Gradient clipping norm.
        mixed_precision: Use mixed precision.
        log_every: Log every N batches.
        callbacks: Optional list of callback functions.

    Returns:
        Training history.
    """
    scaler = GradScaler() if mixed_precision else None
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            result = train_step(
                model, batch, optimizer, loss_fn, device, grad_clip, scaler
            )
            train_loss += result["loss"]
            num_batches += 1

            if batch_idx % log_every == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx} | Loss: {result['loss']:.4f}"
                )

        train_loss /= num_batches
        history["train_loss"].append(train_loss)

        # Validation
        val_metrics = run_eval_loop(model, val_loader, loss_fn, metrics, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_metrics"].append(val_metrics)

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Logging
        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}"
        )

        # Callbacks
        if callbacks:
            for callback in callbacks:
                callback(epoch, model, optimizer, val_metrics)

    return history


@torch.no_grad()
def run_eval_loop(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run evaluation loop.

    Args:
        model: PyTorch model.
        loader: Data loader.
        loss_fn: Loss function.
        metrics: Metrics object.
        device: Device.

    Returns:
        Dictionary of metrics.
    """
    model.eval()
    metrics.reset()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        result = eval_step(model, batch, loss_fn, device)
        metrics.update(result["pred"], result["target"])
        total_loss += result["loss"]
        num_batches += 1

    eval_metrics = metrics.compute()
    eval_metrics["loss"] = total_loss / num_batches

    return eval_metrics


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Run inference loop (no loss computation).

    Args:
        model: PyTorch model.
        loader: Data loader.
        device: Device.

    Returns:
        List of prediction tensors.
    """
    model.eval()
    predictions = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        predictions.append(pred.cpu())

    return predictions


class EarlyStopping:
    """Early stopping utility."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max' for metric optimization.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class GradientAccumulator:
    """Gradient accumulation utility for large batch training."""

    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate gradients.
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def reset(self):
        """Reset step counter."""
        self.current_step = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps

