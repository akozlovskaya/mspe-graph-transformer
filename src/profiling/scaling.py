"""Scaling experiments for efficiency analysis."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from .runtime import RuntimeProfiler, RuntimeStats, profile_pe_computation
from .memory import MemoryProfiler, MemoryStats, estimate_attention_memory
from .flops import FLOPsEstimator, estimate_model_flops


logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Container for scaling experiment results."""

    parameter_name: str
    parameter_values: List[Any]
    runtime_stats: List[RuntimeStats] = field(default_factory=list)
    memory_stats: List[MemoryStats] = field(default_factory=list)
    flops_estimates: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "runtime": [s.to_dict() if s else None for s in self.runtime_stats],
            "memory": [s.to_dict() if s else None for s in self.memory_stats],
            "flops": self.flops_estimates,
        }


def generate_scaling_graphs(
    num_nodes_list: List[int],
    avg_degree: int = 5,
    feature_dim: int = 16,
    seed: int = 42,
) -> List[Data]:
    """
    Generate synthetic graphs for scaling experiments.

    Args:
        num_nodes_list: List of node counts.
        avg_degree: Average node degree.
        feature_dim: Node feature dimension.
        seed: Random seed.

    Returns:
        List of Data objects.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = []

    for num_nodes in num_nodes_list:
        # Generate random edges
        num_edges = num_nodes * avg_degree

        # Random edges (Erdos-Renyi style)
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))

        # Remove self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]

        # Make undirected
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ])

        # Node features
        x = torch.randn(num_nodes, feature_dim)

        # Dummy target
        y = torch.randn(num_nodes, 1)

        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs


def run_node_scaling(
    model_fn: Callable[[int], nn.Module],
    num_nodes_list: List[int] = [100, 500, 1000, 2000, 5000],
    avg_degree: int = 5,
    feature_dim: int = 16,
    device: torch.device = None,
    num_runs: int = 20,
) -> ScalingResult:
    """
    Run node count scaling experiment.

    Args:
        model_fn: Function that creates model given num_features.
        num_nodes_list: List of node counts to test.
        avg_degree: Average degree for generated graphs.
        feature_dim: Node feature dimension.
        device: Device for profiling.
        num_runs: Number of runs per configuration.

    Returns:
        ScalingResult with runtime and memory for each node count.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = generate_scaling_graphs(num_nodes_list, avg_degree, feature_dim)

    result = ScalingResult(
        parameter_name="num_nodes",
        parameter_values=num_nodes_list,
    )

    for i, (num_nodes, graph) in enumerate(zip(num_nodes_list, graphs)):
        logger.info(f"Testing num_nodes={num_nodes}")

        try:
            model = model_fn(feature_dim).to(device)
            graph = graph.to(device)

            # Runtime
            rt_profiler = RuntimeProfiler(model, device, num_runs=num_runs)
            rt_stats = rt_profiler.profile_forward(graph)
            result.runtime_stats.append(rt_stats)

            # Memory
            mem_profiler = MemoryProfiler(model, device)
            mem_stats = mem_profiler.profile_forward(graph)
            result.memory_stats.append(mem_stats)

            # FLOPs
            flops_est = FLOPsEstimator(model)
            flops = flops_est.estimate(num_nodes, graph.edge_index.size(1) // 2)
            result.flops_estimates.append(flops.total)

            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Failed for num_nodes={num_nodes}: {e}")
            result.runtime_stats.append(None)
            result.memory_stats.append(None)
            result.flops_estimates.append(0)

    return result


def run_layer_scaling(
    model_fn: Callable[[int], nn.Module],
    num_layers_list: List[int] = [2, 4, 6, 8, 12, 16],
    num_nodes: int = 500,
    feature_dim: int = 16,
    device: torch.device = None,
    num_runs: int = 20,
) -> ScalingResult:
    """
    Run layer count scaling experiment.

    Args:
        model_fn: Function that creates model given num_layers.
        num_layers_list: List of layer counts to test.
        num_nodes: Number of nodes in test graph.
        feature_dim: Node feature dimension.
        device: Device for profiling.
        num_runs: Number of runs per configuration.

    Returns:
        ScalingResult with runtime and memory for each layer count.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate single test graph
    graphs = generate_scaling_graphs([num_nodes], feature_dim=feature_dim)
    graph = graphs[0].to(device)

    result = ScalingResult(
        parameter_name="num_layers",
        parameter_values=num_layers_list,
    )

    for num_layers in num_layers_list:
        logger.info(f"Testing num_layers={num_layers}")

        try:
            model = model_fn(num_layers).to(device)

            # Runtime
            rt_profiler = RuntimeProfiler(model, device, num_runs=num_runs)
            rt_stats = rt_profiler.profile_forward(graph)
            result.runtime_stats.append(rt_stats)

            # Memory
            mem_profiler = MemoryProfiler(model, device)
            mem_stats = mem_profiler.profile_forward(graph)
            result.memory_stats.append(mem_stats)

            # FLOPs (estimate scales with layers)
            base_flops = FLOPsEstimator(model).estimate(
                num_nodes, graph.edge_index.size(1) // 2
            ).total
            result.flops_estimates.append(base_flops)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Failed for num_layers={num_layers}: {e}")
            result.runtime_stats.append(None)
            result.memory_stats.append(None)
            result.flops_estimates.append(0)

    return result


def run_pe_scaling(
    model_fn: Callable[[str, int], nn.Module],
    pe_configs: List[Dict[str, Any]],
    num_nodes: int = 500,
    feature_dim: int = 16,
    device: torch.device = None,
    num_runs: int = 20,
) -> Dict[str, ScalingResult]:
    """
    Run PE type scaling experiment.

    Args:
        model_fn: Function that creates model given PE type and dim.
        pe_configs: List of PE configurations to test.
        num_nodes: Number of nodes.
        feature_dim: Node feature dimension.
        device: Device for profiling.
        num_runs: Number of runs.

    Returns:
        Dictionary mapping PE name to ScalingResult.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = generate_scaling_graphs([num_nodes], feature_dim=feature_dim)
    graph = graphs[0].to(device)

    results = {}

    for config in pe_configs:
        pe_name = config.get("name", "unknown")
        logger.info(f"Testing PE: {pe_name}")

        try:
            model = model_fn(config).to(device)

            rt_profiler = RuntimeProfiler(model, device, num_runs=num_runs)
            rt_stats = rt_profiler.profile_forward(graph)

            mem_profiler = MemoryProfiler(model, device)
            mem_stats = mem_profiler.profile_forward(graph)

            results[pe_name] = ScalingResult(
                parameter_name="pe_type",
                parameter_values=[pe_name],
                runtime_stats=[rt_stats],
                memory_stats=[mem_stats],
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Failed for PE={pe_name}: {e}")
            results[pe_name] = ScalingResult(
                parameter_name="pe_type",
                parameter_values=[pe_name],
            )

    return results


class ScalingExperiment:
    """Comprehensive scaling experiment runner."""

    def __init__(
        self,
        model_factory: Callable,
        device: torch.device = None,
        num_runs: int = 20,
        warmup_runs: int = 5,
    ):
        """
        Initialize scaling experiment.

        Args:
            model_factory: Factory function for creating models.
            device: Device for profiling.
            num_runs: Number of runs per measurement.
            warmup_runs: Warmup runs.
        """
        self.model_factory = model_factory
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.results = {}

    def run_node_scaling(
        self,
        num_nodes_list: List[int] = [100, 500, 1000, 2000],
        **kwargs,
    ) -> ScalingResult:
        """Run node scaling experiment."""
        result = run_node_scaling(
            self.model_factory,
            num_nodes_list,
            device=self.device,
            num_runs=self.num_runs,
            **kwargs,
        )
        self.results["node_scaling"] = result
        return result

    def run_layer_scaling(
        self,
        num_layers_list: List[int] = [2, 4, 6, 8, 12],
        **kwargs,
    ) -> ScalingResult:
        """Run layer scaling experiment."""

        def layer_model_fn(num_layers):
            return self.model_factory(num_layers=num_layers)

        result = run_layer_scaling(
            layer_model_fn,
            num_layers_list,
            device=self.device,
            num_runs=self.num_runs,
            **kwargs,
        )
        self.results["layer_scaling"] = result
        return result

    def run_attention_density_scaling(
        self,
        density_list: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
        num_nodes: int = 500,
        feature_dim: int = 16,
    ) -> ScalingResult:
        """
        Run attention density scaling experiment.

        Note: This requires model to support sparse attention.
        """
        result = ScalingResult(
            parameter_name="attention_density",
            parameter_values=density_list,
        )

        graph = generate_scaling_graphs([num_nodes], feature_dim=feature_dim)[0]
        graph = graph.to(self.device)

        for density in density_list:
            logger.info(f"Testing attention_density={density}")

            try:
                model = self.model_factory(attention_density=density).to(self.device)

                rt_profiler = RuntimeProfiler(model, self.device, num_runs=self.num_runs)
                rt_stats = rt_profiler.profile_forward(graph)
                result.runtime_stats.append(rt_stats)

                mem_profiler = MemoryProfiler(model, self.device)
                mem_stats = mem_profiler.profile_forward(graph)
                result.memory_stats.append(mem_stats)

                # Memory estimate
                mem_estimate = estimate_attention_memory(
                    num_nodes, 8, 256, sparse_ratio=density
                )
                result.flops_estimates.append(int(mem_estimate * 1e6))

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Failed for density={density}: {e}")
                result.runtime_stats.append(None)
                result.memory_stats.append(None)
                result.flops_estimates.append(0)

        self.results["attention_density_scaling"] = result
        return result

    def run_pe_precomputation_scaling(
        self,
        pe_transforms: Dict[str, Callable],
        num_nodes_list: List[int] = [100, 500, 1000, 2000],
        feature_dim: int = 16,
    ) -> Dict[str, ScalingResult]:
        """
        Run PE precomputation time scaling.

        Args:
            pe_transforms: Dictionary of PE name to transform function.
            num_nodes_list: Node counts to test.
            feature_dim: Feature dimension.

        Returns:
            Dictionary mapping PE name to ScalingResult.
        """
        graphs = generate_scaling_graphs(num_nodes_list, feature_dim=feature_dim)
        results = {}

        for pe_name, transform in pe_transforms.items():
            logger.info(f"Testing PE precomputation: {pe_name}")

            result = ScalingResult(
                parameter_name="num_nodes",
                parameter_values=num_nodes_list,
            )

            for graph in graphs:
                try:
                    rt_stats = profile_pe_computation(transform, graph, num_runs=10)
                    result.runtime_stats.append(rt_stats)
                except Exception as e:
                    logger.warning(f"PE computation failed: {e}")
                    result.runtime_stats.append(None)

            results[pe_name] = result

        self.results["pe_precomputation"] = results
        return results

    def get_all_results(self) -> Dict[str, Any]:
        """Get all experiment results."""
        return {
            name: (
                result.to_dict() if isinstance(result, ScalingResult)
                else {k: v.to_dict() for k, v in result.items()}
            )
            for name, result in self.results.items()
        }

    def print_summary(self):
        """Print scaling experiment summary."""
        print("\n" + "=" * 60)
        print("Scaling Experiment Summary")
        print("=" * 60)

        for exp_name, result in self.results.items():
            print(f"\n{exp_name}:")
            print("-" * 40)

            if isinstance(result, ScalingResult):
                for i, val in enumerate(result.parameter_values):
                    rt = result.runtime_stats[i] if i < len(result.runtime_stats) else None
                    mem = result.memory_stats[i] if i < len(result.memory_stats) else None

                    rt_str = f"{rt.mean:.2f} ms" if rt else "N/A"
                    mem_str = f"{mem.peak_mb:.1f} MB" if mem else "N/A"

                    print(f"  {result.parameter_name}={val}: {rt_str}, {mem_str}")

            elif isinstance(result, dict):
                for name, sub_result in result.items():
                    if sub_result.runtime_stats:
                        rt = sub_result.runtime_stats[0]
                        print(f"  {name}: {rt.mean:.2f} ms" if rt else f"  {name}: N/A")

