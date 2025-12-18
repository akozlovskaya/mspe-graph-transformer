"""Utilities for reproducibility and determinism."""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed.
        deterministic: Whether to enable deterministic CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Enable deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                logger.warning(
                    "Could not enable full deterministic mode. "
                    "Some operations may be non-deterministic."
                )

    # Environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Set random seed to {seed}, deterministic={deterministic}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device.

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.).
                If None, automatically selects CUDA if available.

    Returns:
        torch.device object.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    device = torch.device(device)

    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU")

    return device


def log_system_info():
    """Log system and library information for reproducibility."""
    logger.info("=" * 50)
    logger.info("System Information")
    logger.info("=" * 50)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info("=" * 50)


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def log_model_info(model: torch.nn.Module, name: str = "Model"):
    """Log model information."""
    params = count_parameters(model)
    logger.info(f"{name} parameters:")
    logger.info(f"  Total: {params['total']:,}")
    logger.info(f"  Trainable: {params['trainable']:,}")
    logger.info(f"  Frozen: {params['frozen']:,}")


class ReproducibilityContext:
    """Context manager for reproducible experiments."""

    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Initialize reproducibility context.

        Args:
            seed: Random seed.
            deterministic: Enable deterministic mode.
        """
        self.seed = seed
        self.deterministic = deterministic

        # Save original states
        self._original_random_state = None
        self._original_np_state = None
        self._original_torch_state = None
        self._original_cuda_state = None
        self._original_cudnn_deterministic = None
        self._original_cudnn_benchmark = None

    def __enter__(self):
        """Save states and set seed."""
        # Save current states
        self._original_random_state = random.getstate()
        self._original_np_state = np.random.get_state()
        self._original_torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self._original_cuda_state = torch.cuda.get_rng_state_all()
            self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
            self._original_cudnn_benchmark = torch.backends.cudnn.benchmark

        # Set seed
        set_seed(self.seed, self.deterministic)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original states."""
        random.setstate(self._original_random_state)
        np.random.set_state(self._original_np_state)
        torch.set_rng_state(self._original_torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self._original_cuda_state)
            torch.backends.cudnn.deterministic = self._original_cudnn_deterministic
            torch.backends.cudnn.benchmark = self._original_cudnn_benchmark

        return False

