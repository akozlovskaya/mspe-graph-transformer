"""Learning rate scheduler configurations."""

import math
from typing import Optional, Dict, Any
import torch
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


class CosineWithWarmup(LRScheduler):
    """Cosine decay scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize cosine scheduler with warmup.

        Args:
            optimizer: PyTorch optimizer.
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total number of epochs.
            min_lr: Minimum learning rate after decay.
            warmup_start_lr: Starting learning rate for warmup.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

        # Store base LRs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class WarmupScheduler(LRScheduler):
    """Linear warmup scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer.
            warmup_epochs: Number of warmup epochs.
            warmup_start_lr: Starting learning rate.
            last_epoch: The index of last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_epochs: int = 100,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    step_size: int = 30,
    gamma: float = 0.1,
    **kwargs,
) -> LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_type: Type of scheduler: 'cosine', 'step', 'cosine_warmup'.
        total_epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate.
        step_size: Step size for StepLR.
        gamma: Decay factor for StepLR.
        **kwargs: Additional scheduler arguments.

    Returns:
        Learning rate scheduler.
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=min_lr
        )
    elif scheduler_type == "cosine_warmup":
        scheduler = CosineWithWarmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
        )
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "warmup":
        scheduler = WarmupScheduler(
            optimizer, warmup_epochs=warmup_epochs
        )
    elif scheduler_type == "none":
        # No scheduling - constant LR
        scheduler = StepLR(optimizer, step_size=total_epochs + 1, gamma=1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def get_scheduler_from_config(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any]
) -> LRScheduler:
    """
    Create scheduler from configuration dictionary.

    Args:
        optimizer: PyTorch optimizer.
        config: Configuration dictionary with keys:
            - type: Scheduler type
            - total_epochs: Total epochs
            - warmup_epochs: Warmup epochs
            - etc.

    Returns:
        Learning rate scheduler.
    """
    scheduler_type = config.get("type", "cosine_warmup")
    total_epochs = config.get("total_epochs", config.get("epochs", 100))
    warmup_epochs = config.get("warmup_epochs", 10)
    min_lr = config.get("min_lr", 1e-6)
    step_size = config.get("step_size", 30)
    gamma = config.get("gamma", 0.1)

    return get_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        step_size=step_size,
        gamma=gamma,
    )

