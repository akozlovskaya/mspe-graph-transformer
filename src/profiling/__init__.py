"""Profiling framework for Graph Transformers."""

from .runtime import (
    RuntimeProfiler,
    profile_forward,
    profile_backward,
    profile_training_step,
    benchmark_function,
)

from .memory import (
    MemoryProfiler,
    profile_memory_usage,
    get_peak_memory,
    reset_memory_stats,
    get_memory_breakdown,
)

from .flops import (
    FLOPsEstimator,
    estimate_attention_flops,
    estimate_mpnn_flops,
    estimate_ffn_flops,
    estimate_pe_flops,
    estimate_model_flops,
)

from .scaling import (
    ScalingExperiment,
    run_node_scaling,
    run_layer_scaling,
    run_pe_scaling,
    generate_scaling_graphs,
)

from .utils import (
    get_hardware_info,
    get_model_info,
    format_profiling_results,
    save_profiling_results,
    print_profiling_table,
    ProfilingContext,
)


__all__ = [
    # Runtime
    "RuntimeProfiler",
    "profile_forward",
    "profile_backward",
    "profile_training_step",
    "benchmark_function",
    # Memory
    "MemoryProfiler",
    "profile_memory_usage",
    "get_peak_memory",
    "reset_memory_stats",
    "get_memory_breakdown",
    # FLOPs
    "FLOPsEstimator",
    "estimate_attention_flops",
    "estimate_mpnn_flops",
    "estimate_ffn_flops",
    "estimate_pe_flops",
    "estimate_model_flops",
    # Scaling
    "ScalingExperiment",
    "run_node_scaling",
    "run_layer_scaling",
    "run_pe_scaling",
    "generate_scaling_graphs",
    # Utils
    "get_hardware_info",
    "get_model_info",
    "format_profiling_results",
    "save_profiling_results",
    "print_profiling_table",
    "ProfilingContext",
]

