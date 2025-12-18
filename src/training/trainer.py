"""Main Trainer class for Graph Transformer training."""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from .metrics import get_metrics, RegressionMetrics, ClassificationMetrics, EdgeMetrics
from .losses import get_loss_fn


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for Graph Transformers.

    Supports:
    - Multiple task types (regression, classification, edge prediction)
    - Mixed precision training
    - Gradient clipping
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[Union[Dict, Any]] = None,
        device: torch.device = None,
        task_type: str = "regression",
        num_classes: int = 2,
        grad_clip: Optional[float] = None,
        mixed_precision: bool = False,
        log_every: int = 50,
        checkpoint_dir: Optional[str] = None,
        experiment_name: str = "experiment",
    ):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            scheduler: Learning rate scheduler (optional).
            loss_fn: Loss function. If None, inferred from task_type.
            metrics: Metrics object or dict. If None, inferred from task_type.
            device: Device to use. Defaults to CUDA if available.
            task_type: Task type: 'regression', 'classification', 'edge_prediction'.
            num_classes: Number of classes for classification.
            grad_clip: Max gradient norm for clipping. None to disable.
            mixed_precision: Whether to use mixed precision training.
            log_every: Log every N batches.
            checkpoint_dir: Directory for saving checkpoints.
            experiment_name: Name of the experiment.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_type = task_type
        self.num_classes = num_classes
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.log_every = log_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.experiment_name = experiment_name

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Loss function
        if loss_fn is None:
            self.loss_fn = get_loss_fn(task_type, num_classes=num_classes)
        else:
            self.loss_fn = loss_fn

        # Metrics
        if metrics is None:
            self.metrics = get_metrics(task_type, num_classes=num_classes)
        else:
            self.metrics = metrics

        # Mixed precision
        self.scaler = GradScaler() if mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.best_epoch = 0

        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging hooks
        self.log_hooks = []

    def add_log_hook(self, hook: Callable):
        """Add a logging hook (e.g., TensorBoard, W&B)."""
        self.log_hooks.append(hook)

    def _log(self, metrics: Dict[str, Any], step: int, prefix: str = "train"):
        """Log metrics to all hooks."""
        for hook in self.log_hooks:
            hook(metrics, step, prefix)

    def _get_target(self, batch) -> torch.Tensor:
        """Extract target from batch based on task type."""
        if hasattr(batch, "y"):
            target = batch.y
        elif hasattr(batch, "target"):
            target = batch.target
        else:
            raise ValueError("Batch has no target attribute (y or target)")
        
        # Ensure target has correct shape for loss computation
        # For regression with single output, target should be [batch_size, 1]
        if self.task_type == "regression" and target.dim() == 1:
            target = target.unsqueeze(-1)
        
        return target

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            loader: Training data loader.

        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)
            target = self._get_target(batch).to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if self.mixed_precision and self.scaler is not None:
                with autocast():
                    pred = self.model(batch)
                    loss = self.loss_fn(pred, target)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(batch)
                loss = self.loss_fn(pred, target)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.grad_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            # Update metrics
            self.metrics.update(pred.detach(), target.detach())
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.log_every == 0:
                logger.info(
                    f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics["loss"] = total_loss / num_batches
        epoch_metrics["time"] = time.time() - epoch_start

        return epoch_metrics

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run evaluation epoch.

        Args:
            loader: Evaluation data loader.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            target = self._get_target(batch).to(self.device)

            pred = self.model(batch)
            loss = self.loss_fn(pred, target)

            self.metrics.update(pred, target)
            total_loss += loss.item()
            num_batches += 1

        # Compute metrics
        eval_metrics = self.metrics.compute()
        eval_metrics["loss"] = total_loss / num_batches

        return eval_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        metric_for_best: str = "loss",
        minimize_metric: bool = True,
        early_stopping: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs.
            metric_for_best: Metric to track for best model.
            minimize_metric: Whether to minimize the metric.
            early_stopping: Stop if no improvement for N epochs. None to disable.

        Returns:
            Training history.
        """
        history = {"train": [], "val": []}
        no_improve_count = 0

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train"].append(train_metrics)

            # Validation
            val_metrics = self.eval_epoch(val_loader)
            history["val"].append(val_metrics)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Log additional metrics
            self._log(train_metrics, epoch, prefix="train")
            self._log(val_metrics, epoch, prefix="val")

            # Check for best model
            current_metric = val_metrics.get(metric_for_best, val_metrics["loss"])

            is_best = False
            if self.best_metric is None:
                is_best = True
            elif minimize_metric and current_metric < self.best_metric:
                is_best = True
            elif not minimize_metric and current_metric > self.best_metric:
                is_best = True

            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                no_improve_count = 0
                if self.checkpoint_dir:
                    self.save_checkpoint("best.pt", val_metrics)
                logger.info(f"New best {metric_for_best}: {current_metric:.4f}")
            else:
                no_improve_count += 1

            # Save last checkpoint
            if self.checkpoint_dir:
                self.save_checkpoint("last.pt", val_metrics)

            # Early stopping
            if early_stopping is not None and no_improve_count >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(
            f"Training finished. Best {metric_for_best}: {self.best_metric:.4f} at epoch {self.best_epoch}"
        )

        return history

    def save_checkpoint(self, filename: str, metrics: Dict[str, float] = None):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename.
            metrics: Current metrics to save.
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics": metrics or {},
        }

        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file.
            load_optimizer: Whether to load optimizer state.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric")
        self.best_epoch = checkpoint.get("best_epoch", 0)

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

        return checkpoint.get("metrics", {})

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> torch.Tensor:
        """
        Get predictions for a dataset.

        Args:
            loader: Data loader.

        Returns:
            Predictions tensor.
        """
        self.model.eval()
        predictions = []

        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            predictions.append(pred.cpu())

        return torch.cat(predictions, dim=0)

