"""Runtime profiling utilities."""

import time
from typing import Callable, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


@dataclass
class RuntimeStats:
    """Container for runtime statistics."""

    mean: float
    std: float
    min: float
    max: float
    num_runs: int
    unit: str = "ms"

    def __repr__(self):
        return f"{self.mean:.2f} ± {self.std:.2f} {self.unit} (n={self.num_runs})"

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "num_runs": self.num_runs,
            "unit": self.unit,
        }


def _cuda_synchronize():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_time_ms() -> float:
    """Get current time in milliseconds with CUDA sync."""
    _cuda_synchronize()
    return time.perf_counter() * 1000


def benchmark_function(
    fn: Callable,
    *args,
    num_runs: int = 100,
    warmup_runs: int = 10,
    **kwargs,
) -> RuntimeStats:
    """
    Benchmark a function's execution time.

    Args:
        fn: Function to benchmark.
        *args: Positional arguments for function.
        num_runs: Number of measurement runs.
        warmup_runs: Number of warmup runs (excluded from stats).
        **kwargs: Keyword arguments for function.

    Returns:
        RuntimeStats with timing information.
    """
    # Warmup
    for _ in range(warmup_runs):
        fn(*args, **kwargs)

    # Measurement
    times = []
    for _ in range(num_runs):
        start = _get_time_ms()
        fn(*args, **kwargs)
        end = _get_time_ms()
        times.append(end - start)

    times_tensor = torch.tensor(times)

    return RuntimeStats(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
        min=times_tensor.min().item(),
        max=times_tensor.max().item(),
        num_runs=num_runs,
    )


def profile_forward(
    model: nn.Module,
    batch: Data,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> RuntimeStats:
    """
    Profile forward pass runtime.

    Args:
        model: PyTorch model.
        batch: Input batch.
        num_runs: Number of runs.
        warmup_runs: Warmup runs.

    Returns:
        RuntimeStats for forward pass.
    """
    model.eval()

    with torch.no_grad():
        stats = benchmark_function(
            model, batch, num_runs=num_runs, warmup_runs=warmup_runs
        )

    return stats


def profile_backward(
    model: nn.Module,
    batch: Data,
    loss_fn: Optional[nn.Module] = None,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> RuntimeStats:
    """
    Profile backward pass runtime.

    Args:
        model: PyTorch model.
        batch: Input batch.
        loss_fn: Loss function. If None, uses sum of outputs.
        num_runs: Number of runs.
        warmup_runs: Warmup runs.

    Returns:
        RuntimeStats for backward pass.
    """
    model.train()

    def forward_backward():
        model.zero_grad()
        output = model(batch)

        if loss_fn is not None and hasattr(batch, "y"):
            loss = loss_fn(output, batch.y)
        else:
            loss = output.sum()

        loss.backward()

    # Warmup
    for _ in range(warmup_runs):
        forward_backward()

    # Measurement
    times = []
    for _ in range(num_runs):
        start = _get_time_ms()
        forward_backward()
        end = _get_time_ms()
        times.append(end - start)

    times_tensor = torch.tensor(times)

    return RuntimeStats(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
        min=times_tensor.min().item(),
        max=times_tensor.max().item(),
        num_runs=num_runs,
    )


def profile_training_step(
    model: nn.Module,
    batch: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module] = None,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> RuntimeStats:
    """
    Profile complete training step (forward + backward + optimizer step).

    Args:
        model: PyTorch model.
        batch: Input batch.
        optimizer: Optimizer.
        loss_fn: Loss function.
        num_runs: Number of runs.
        warmup_runs: Warmup runs.

    Returns:
        RuntimeStats for training step.
    """
    model.train()

    def training_step():
        optimizer.zero_grad()
        output = model(batch)

        if loss_fn is not None and hasattr(batch, "y"):
            loss = loss_fn(output, batch.y)
        else:
            loss = output.sum()

        loss.backward()
        optimizer.step()

    # Warmup
    for _ in range(warmup_runs):
        training_step()

    # Measurement
    times = []
    for _ in range(num_runs):
        start = _get_time_ms()
        training_step()
        end = _get_time_ms()
        times.append(end - start)

    times_tensor = torch.tensor(times)

    return RuntimeStats(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
        min=times_tensor.min().item(),
        max=times_tensor.max().item(),
        num_runs=num_runs,
    )


class RuntimeProfiler:
    """Comprehensive runtime profiler for Graph Transformers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ):
        """
        Initialize runtime profiler.

        Args:
            model: Model to profile.
            device: Device (auto-detected if None).
            num_runs: Number of measurement runs.
            warmup_runs: Warmup runs.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs

    def profile_forward(self, batch: Data) -> RuntimeStats:
        """Profile forward pass."""
        batch = batch.to(self.device)
        return profile_forward(
            self.model, batch, self.num_runs, self.warmup_runs
        )

    def profile_backward(
        self, batch: Data, loss_fn: Optional[nn.Module] = None
    ) -> RuntimeStats:
        """Profile backward pass."""
        batch = batch.to(self.device)
        return profile_backward(
            self.model, batch, loss_fn, self.num_runs // 2, self.warmup_runs // 2
        )

    def profile_training_step(
        self,
        batch: Data,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[nn.Module] = None,
    ) -> RuntimeStats:
        """Profile training step."""
        batch = batch.to(self.device)
        return profile_training_step(
            self.model, batch, optimizer, loss_fn,
            self.num_runs // 2, self.warmup_runs // 2
        )

    def profile_layer(
        self,
        layer: nn.Module,
        input_tensor: torch.Tensor,
        **kwargs,
    ) -> RuntimeStats:
        """
        Profile a single layer.

        Args:
            layer: Layer to profile.
            input_tensor: Input tensor.
            **kwargs: Additional layer arguments.

        Returns:
            RuntimeStats for the layer.
        """
        layer.eval()
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            return benchmark_function(
                layer, input_tensor, num_runs=self.num_runs,
                warmup_runs=self.warmup_runs, **kwargs
            )

    def profile_all(
        self,
        batch: Data,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, RuntimeStats]:
        """
        Profile all aspects of the model.

        Args:
            batch: Input batch.
            optimizer: Optimizer (optional, for training step).
            loss_fn: Loss function (optional).

        Returns:
            Dictionary of RuntimeStats for each aspect.
        """
        batch = batch.to(self.device)
        results = {}

        # Forward
        results["forward"] = self.profile_forward(batch)

        # Backward
        results["backward"] = self.profile_backward(batch, loss_fn)

        # Training step
        if optimizer is not None:
            results["training_step"] = self.profile_training_step(
                batch, optimizer, loss_fn
            )

        return results

    def profile_components(
        self,
        batch: Data,
    ) -> Dict[str, RuntimeStats]:
        """
        Profile individual model components.

        Args:
            batch: Input batch.

        Returns:
            Dictionary of RuntimeStats per component.
        """
        batch = batch.to(self.device)
        results = {}

        # Try to profile individual components if model has named modules
        self.model.eval()

        with torch.no_grad():
            # Get intermediate outputs
            x = batch.x

            for name, module in self.model.named_children():
                try:
                    # Profile this module
                    stats = benchmark_function(
                        module, x, batch,
                        num_runs=self.num_runs,
                        warmup_runs=self.warmup_runs,
                    )
                    results[name] = stats
                except Exception:
                    # Skip modules that don't accept these inputs
                    pass

        return results


@contextmanager
def timed_region(name: str, results: Dict[str, float]):
    """
    Context manager for timing a code region.

    Args:
        name: Name of the region.
        results: Dictionary to store result.
    """
    _cuda_synchronize()
    start = time.perf_counter()
    yield
    _cuda_synchronize()
    end = time.perf_counter()
    results[name] = (end - start) * 1000  # Convert to ms


def profile_pe_computation(
    pe_transform,
    data: Data,
    num_runs: int = 10,
) -> RuntimeStats:
    """
    Profile positional encoding computation time.

    Args:
        pe_transform: PE transform/function to profile.
        data: Input graph data.
        num_runs: Number of runs.

    Returns:
        RuntimeStats for PE computation.
    """
    # Clear any cached PE
    if hasattr(data, "node_pe"):
        delattr(data, "node_pe")
    if hasattr(data, "edge_pe"):
        delattr(data, "edge_pe")

    times = []
    for _ in range(num_runs):
        # Clear cache
        if hasattr(data, "node_pe"):
            delattr(data, "node_pe")

        start = _get_time_ms()
        pe_transform(data)
        end = _get_time_ms()
        times.append(end - start)

    times_tensor = torch.tensor(times)

    return RuntimeStats(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
        min=times_tensor.min().item(),
        max=times_tensor.max().item(),
        num_runs=num_runs,
    )

