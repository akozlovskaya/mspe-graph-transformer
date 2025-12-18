"""Optimizer configurations."""

from typing import Optional, Dict, Any, Iterable
import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.

    Args:
        model: PyTorch model.
        optimizer_type: Type of optimizer: 'adam', 'adamw', 'sgd'.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        betas: Betas for Adam variants.
        eps: Epsilon for Adam variants.
        momentum: Momentum for SGD.
        **kwargs: Additional optimizer arguments.

    Returns:
        PyTorch optimizer.
    """
    # Separate parameters that should not have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to bias and LayerNorm
        if "bias" in name or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=lr, betas=betas, eps=eps, **kwargs
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, eps=eps, **kwargs
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=lr, momentum=momentum, **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def get_optimizer_from_config(
    model: nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration dictionary.

    Args:
        model: PyTorch model.
        config: Configuration dictionary with keys:
            - type: Optimizer type
            - lr: Learning rate
            - weight_decay: Weight decay
            - etc.

    Returns:
        PyTorch optimizer.
    """
    optimizer_type = config.get("type", "adamw")
    lr = config.get("lr", config.get("learning_rate", 1e-4))
    weight_decay = config.get("weight_decay", 0.01)
    betas = config.get("betas", (0.9, 0.999))
    eps = config.get("eps", 1e-8)
    momentum = config.get("momentum", 0.9)

    return get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        betas=tuple(betas),
        eps=eps,
        momentum=momentum,
    )

