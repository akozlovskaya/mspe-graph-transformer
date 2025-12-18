"""Profiling utilities."""

import os
import json
import platform
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn


def get_hardware_info() -> Dict[str, Any]:
    """
    Get hardware information for profiling reports.

    Returns:
        Dictionary with hardware information.
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()

        # Get GPU info for each device
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024 ** 3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
    else:
        info["cuda_available"] = False

    return info


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with model information.
    """
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Parameter memory
    param_memory = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)

    # Buffer memory
    buffer_memory = sum(
        b.numel() * b.element_size() for b in model.buffers()
    ) / (1024 ** 2)

    # Layer counts
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "parameter_memory_mb": param_memory,
        "buffer_memory_mb": buffer_memory,
        "layer_counts": layer_counts,
    }


def format_profiling_results(
    results: Dict[str, Any],
    include_hardware: bool = True,
    include_timestamp: bool = True,
) -> Dict[str, Any]:
    """
    Format profiling results for saving/display.

    Args:
        results: Raw profiling results.
        include_hardware: Include hardware information.
        include_timestamp: Include timestamp.

    Returns:
        Formatted results dictionary.
    """
    formatted = {}

    if include_timestamp:
        formatted["timestamp"] = datetime.now().isoformat()

    if include_hardware:
        formatted["hardware"] = get_hardware_info()

    # Convert result objects to dictionaries
    for key, value in results.items():
        if hasattr(value, "to_dict"):
            formatted[key] = value.to_dict()
        elif isinstance(value, dict):
            formatted[key] = {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in value.items()
            }
        else:
            formatted[key] = value

    return formatted


def save_profiling_results(
    results: Dict[str, Any],
    output_path: str,
    format: str = "json",
):
    """
    Save profiling results to file.

    Args:
        results: Results dictionary.
        output_path: Output file path.
        format: Output format ('json' or 'csv').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    formatted = format_profiling_results(results)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2, default=str)
    elif format == "csv":
        import csv

        # Flatten nested dict for CSV
        flat = _flatten_dict(formatted)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in flat.items():
                writer.writerow([key, value])


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def print_profiling_table(
    results: Dict[str, Any],
    title: str = "Profiling Results",
):
    """
    Print profiling results as formatted table.

    Args:
        results: Results dictionary.
        title: Table title.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # Runtime results
    if "runtime" in results:
        print("\nRuntime:")
        print("-" * 40)
        rt = results["runtime"]
        if isinstance(rt, dict):
            for name, stats in rt.items():
                if stats and isinstance(stats, dict):
                    print(f"  {name}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f} ms")
                elif hasattr(stats, "mean"):
                    print(f"  {name}: {stats.mean:.2f} ± {stats.std:.2f} ms")

    # Memory results
    if "memory" in results:
        print("\nMemory:")
        print("-" * 40)
        mem = results["memory"]
        if isinstance(mem, dict):
            for name, stats in mem.items():
                if isinstance(stats, dict):
                    print(f"  {name}: {stats.get('peak_mb', 0):.2f} MB")
                elif hasattr(stats, "peak_mb"):
                    print(f"  {name}: {stats.peak_mb:.2f} MB")

    # FLOPs results
    if "flops" in results:
        print("\nFLOPs:")
        print("-" * 40)
        flops = results["flops"]
        if isinstance(flops, dict):
            total = flops.get("total", flops.get("total_gflops", 0))
            if total > 1e9:
                print(f"  Total: {total / 1e9:.2f} GFLOPs")
            else:
                print(f"  Total: {total / 1e6:.2f} MFLOPs")

            if "breakdown" in flops:
                print("  Breakdown:")
                for name, val in flops["breakdown"].items():
                    if val > 1e9:
                        print(f"    {name}: {val / 1e9:.2f} GFLOPs")
                    else:
                        print(f"    {name}: {val / 1e6:.2f} MFLOPs")

    # Model info
    if "model_info" in results:
        print("\nModel Info:")
        print("-" * 40)
        info = results["model_info"]
        print(f"  Parameters: {info.get('total_parameters', 0):,}")
        print(f"  Trainable: {info.get('trainable_parameters', 0):,}")
        print(f"  Memory: {info.get('parameter_memory_mb', 0):.2f} MB")

    print("\n" + "=" * 60)


class ProfilingContext:
    """Context manager for profiling setup."""

    def __init__(
        self,
        model: nn.Module,
        disable_dropout: bool = True,
        eval_mode: bool = True,
        seed: int = 42,
    ):
        """
        Initialize profiling context.

        Args:
            model: Model to profile.
            disable_dropout: Whether to disable dropout.
            eval_mode: Whether to set model to eval mode.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.disable_dropout = disable_dropout
        self.eval_mode = eval_mode
        self.seed = seed

        self._original_training = model.training
        self._dropout_states = {}

    def __enter__(self):
        """Set up profiling environment."""
        # Set seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Disable dropout
        if self.disable_dropout:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Dropout):
                    self._dropout_states[name] = module.p
                    module.p = 0.0

        # Set eval mode
        if self.eval_mode:
            self.model.eval()

        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original state."""
        # Restore dropout
        if self.disable_dropout:
            for name, module in self.model.named_modules():
                if name in self._dropout_states:
                    module.p = self._dropout_states[name]

        # Restore training mode
        self.model.train(self._original_training)

        return False


def compare_profiling_results(
    results_list: List[Dict[str, Any]],
    names: List[str],
    metric: str = "runtime",
) -> Dict[str, Any]:
    """
    Compare multiple profiling results.

    Args:
        results_list: List of profiling results.
        names: Names for each result set.
        metric: Metric to compare ('runtime', 'memory', 'flops').

    Returns:
        Comparison dictionary.
    """
    comparison = {"names": names, "metric": metric, "values": []}

    for results in results_list:
        if metric in results:
            val = results[metric]
            if isinstance(val, dict):
                # Get primary metric
                if "forward" in val:
                    comparison["values"].append(
                        val["forward"].get("mean", val["forward"].get("peak_mb", 0))
                    )
                elif "total" in val:
                    comparison["values"].append(val["total"])
                else:
                    comparison["values"].append(list(val.values())[0])
            elif hasattr(val, "mean"):
                comparison["values"].append(val.mean)
            else:
                comparison["values"].append(val)
        else:
            comparison["values"].append(None)

    # Find best (minimum for runtime/memory, maximum for flops)
    valid_values = [(i, v) for i, v in enumerate(comparison["values"]) if v is not None]
    if valid_values:
        if metric in ["runtime", "memory"]:
            best_idx = min(valid_values, key=lambda x: x[1])[0]
        else:
            best_idx = max(valid_values, key=lambda x: x[1])[0]
        comparison["best"] = names[best_idx]
        comparison["best_value"] = comparison["values"][best_idx]

    return comparison


def estimate_throughput(
    runtime_ms: float,
    batch_size: int = 1,
    unit: str = "samples/s",
) -> float:
    """
    Estimate throughput from runtime.

    Args:
        runtime_ms: Runtime in milliseconds.
        batch_size: Batch size.
        unit: Output unit.

    Returns:
        Throughput value.
    """
    if runtime_ms <= 0:
        return 0.0

    samples_per_second = (batch_size / runtime_ms) * 1000

    if unit == "samples/s":
        return samples_per_second
    elif unit == "batches/s":
        return samples_per_second / batch_size
    else:
        return samples_per_second

