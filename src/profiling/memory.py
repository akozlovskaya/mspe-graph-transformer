"""Memory profiling utilities."""

import gc
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Data


@dataclass
class MemoryStats:
    """Container for memory statistics."""

    peak_mb: float
    allocated_mb: float
    reserved_mb: float
    device: str

    def __repr__(self):
        return f"Peak: {self.peak_mb:.2f} MB, Allocated: {self.allocated_mb:.2f} MB"

    def to_dict(self) -> Dict[str, float]:
        return {
            "peak_mb": self.peak_mb,
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "device": self.device,
        }


def reset_memory_stats():
    """Reset GPU memory statistics."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory(device: torch.device = None) -> float:
    """
    Get peak GPU memory usage in MB.

    Args:
        device: CUDA device. Uses current device if None.

    Returns:
        Peak memory in MB.
    """
    if not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def get_allocated_memory(device: torch.device = None) -> float:
    """Get currently allocated GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()

    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def get_reserved_memory(device: torch.device = None) -> float:
    """Get reserved GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0

    if device is None:
        device = torch.cuda.current_device()

    return torch.cuda.memory_reserved(device) / (1024 ** 2)


def profile_memory_usage(
    fn: Callable,
    *args,
    device: torch.device = None,
    **kwargs,
) -> MemoryStats:
    """
    Profile memory usage of a function.

    Args:
        fn: Function to profile.
        *args: Function arguments.
        device: Device for measurement.
        **kwargs: Function keyword arguments.

    Returns:
        MemoryStats with memory information.
    """
    device_str = "cpu"

    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        device_str = f"cuda:{device}" if isinstance(device, int) else str(device)

        reset_memory_stats()

    # Run function
    result = fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = get_peak_memory(device)
        allocated = get_allocated_memory(device)
        reserved = get_reserved_memory(device)
    else:
        peak = allocated = reserved = 0.0

    # Clean up result to free memory
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return MemoryStats(
        peak_mb=peak,
        allocated_mb=allocated,
        reserved_mb=reserved,
        device=device_str,
    )


def get_memory_breakdown(
    model: nn.Module,
    batch: Data,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Get memory breakdown by component.

    Args:
        model: PyTorch model.
        batch: Input batch.
        device: Device for measurement.

    Returns:
        Dictionary with memory usage per component (in MB).
    """
    breakdown = {}

    if not torch.cuda.is_available():
        return breakdown

    reset_memory_stats()
    batch = batch.to(device)

    # Model parameters
    param_memory = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)
    breakdown["parameters"] = param_memory

    # Gradients (if allocated)
    grad_memory = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    ) / (1024 ** 2)
    breakdown["gradients"] = grad_memory

    # Input data
    input_memory = 0
    if hasattr(batch, "x") and batch.x is not None:
        input_memory += batch.x.numel() * batch.x.element_size()
    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        input_memory += batch.edge_index.numel() * batch.edge_index.element_size()
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        input_memory += batch.edge_attr.numel() * batch.edge_attr.element_size()
    breakdown["input_data"] = input_memory / (1024 ** 2)

    # PE data (if present)
    pe_memory = 0
    if hasattr(batch, "node_pe") and batch.node_pe is not None:
        pe_memory += batch.node_pe.numel() * batch.node_pe.element_size()
    if hasattr(batch, "edge_pe") and batch.edge_pe is not None:
        pe_memory += batch.edge_pe.numel() * batch.edge_pe.element_size()
    breakdown["positional_encodings"] = pe_memory / (1024 ** 2)

    # Forward pass activations
    reset_memory_stats()
    model.eval()
    with torch.no_grad():
        _ = model(batch)
    torch.cuda.synchronize()

    forward_peak = get_peak_memory(device)
    breakdown["forward_activations"] = forward_peak - param_memory - breakdown["input_data"]

    # Backward pass memory
    reset_memory_stats()
    model.train()
    model.zero_grad()
    output = model(batch)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()

    backward_peak = get_peak_memory(device)
    breakdown["backward_memory"] = backward_peak - param_memory

    return breakdown


class MemoryProfiler:
    """Comprehensive memory profiler for Graph Transformers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
    ):
        """
        Initialize memory profiler.

        Args:
            model: Model to profile.
            device: Device for profiling.
        """
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

    def profile_forward(self, batch: Data) -> MemoryStats:
        """
        Profile forward pass memory.

        Args:
            batch: Input batch.

        Returns:
            MemoryStats for forward pass.
        """
        batch = batch.to(self.device)
        self.model.eval()

        def forward():
            with torch.no_grad():
                return self.model(batch)

        return profile_memory_usage(forward, device=self.device)

    def profile_backward(self, batch: Data) -> MemoryStats:
        """
        Profile backward pass memory.

        Args:
            batch: Input batch.

        Returns:
            MemoryStats for backward pass.
        """
        batch = batch.to(self.device)
        self.model.train()

        def forward_backward():
            self.model.zero_grad()
            output = self.model(batch)
            loss = output.sum()
            loss.backward()
            return loss

        return profile_memory_usage(forward_backward, device=self.device)

    def profile_batch_scaling(
        self,
        data_list: list,
        batch_sizes: list = [1, 2, 4, 8, 16, 32],
    ) -> Dict[int, MemoryStats]:
        """
        Profile memory scaling with batch size.

        Args:
            data_list: List of Data objects.
            batch_sizes: Batch sizes to test.

        Returns:
            Dictionary mapping batch size to MemoryStats.
        """
        from torch_geometric.loader import DataLoader

        results = {}

        for bs in batch_sizes:
            if bs > len(data_list):
                continue

            loader = DataLoader(data_list[:bs], batch_size=bs)
            batch = next(iter(loader)).to(self.device)

            stats = self.profile_forward(batch)
            results[bs] = stats

        return results

    def get_parameter_memory(self) -> float:
        """Get memory used by model parameters in MB."""
        return sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 ** 2)

    def get_buffer_memory(self) -> float:
        """Get memory used by model buffers in MB."""
        return sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        ) / (1024 ** 2)

    def profile_all(self, batch: Data) -> Dict[str, Any]:
        """
        Profile all memory aspects.

        Args:
            batch: Input batch.

        Returns:
            Comprehensive memory profile.
        """
        batch = batch.to(self.device)

        results = {
            "parameter_memory_mb": self.get_parameter_memory(),
            "buffer_memory_mb": self.get_buffer_memory(),
            "forward": self.profile_forward(batch).to_dict(),
            "backward": self.profile_backward(batch).to_dict(),
        }

        if torch.cuda.is_available():
            results["breakdown"] = get_memory_breakdown(
                self.model, batch, self.device
            )

        return results


def estimate_attention_memory(
    num_nodes: int,
    num_heads: int,
    hidden_dim: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Estimate memory for attention computation.

    Args:
        num_nodes: Number of nodes.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension.
        batch_size: Batch size.
        dtype: Data type.

    Returns:
        Estimated memory in MB.
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()

    # Q, K, V projections
    qkv_memory = 3 * batch_size * num_nodes * hidden_dim * bytes_per_element

    # Attention weights [batch, heads, N, N]
    attention_memory = batch_size * num_heads * num_nodes * num_nodes * bytes_per_element

    # Output
    output_memory = batch_size * num_nodes * hidden_dim * bytes_per_element

    total = (qkv_memory + attention_memory + output_memory) / (1024 ** 2)

    return total


def estimate_pe_storage(
    num_nodes: int,
    node_pe_dim: int = 0,
    relative_pe_buckets: int = 0,
    sparse_ratio: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Estimate storage for positional encodings.

    Args:
        num_nodes: Number of nodes.
        node_pe_dim: Dimension of node PE.
        relative_pe_buckets: Number of relative PE buckets.
        sparse_ratio: Ratio of non-zero pairs for sparse relative PE.
        dtype: Data type.

    Returns:
        Dictionary with storage estimates in MB.
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    index_bytes = 8  # int64

    storage = {}

    # Node PE: [N, node_pe_dim]
    if node_pe_dim > 0:
        storage["node_pe"] = num_nodes * node_pe_dim * bytes_per_element / (1024 ** 2)

    # Relative PE (dense): [N, N, buckets]
    if relative_pe_buckets > 0:
        dense = num_nodes * num_nodes * relative_pe_buckets * bytes_per_element / (1024 ** 2)
        storage["relative_pe_dense"] = dense

        # Sparse: [2, num_pairs] indices + [num_pairs, buckets] values
        num_pairs = int(num_nodes * num_nodes * sparse_ratio)
        sparse_idx = 2 * num_pairs * index_bytes / (1024 ** 2)
        sparse_val = num_pairs * relative_pe_buckets * bytes_per_element / (1024 ** 2)
        storage["relative_pe_sparse"] = sparse_idx + sparse_val

    return storage

