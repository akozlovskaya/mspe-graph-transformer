"""Model profiling script."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataset
from src.models import get_model
from src.profiling import (
    RuntimeProfiler,
    MemoryProfiler,
    FLOPsEstimator,
    ScalingExperiment,
    get_hardware_info,
    get_model_info,
    format_profiling_results,
    save_profiling_results,
    print_profiling_table,
    ProfilingContext,
    generate_scaling_graphs,
)
from src.training.reproducibility import set_seed, get_device


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def profile_runtime(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Profile model runtime."""
    logger.info("Profiling runtime...")

    profiler = RuntimeProfiler(model, device, num_runs=num_runs)

    results = {
        "forward": profiler.profile_forward(batch).to_dict(),
        "backward": profiler.profile_backward(batch).to_dict(),
    }

    # Training step with optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    results["training_step"] = profiler.profile_training_step(
        batch, optimizer
    ).to_dict()

    return results


def profile_memory(
    model: torch.nn.Module,
    batch,
    device: torch.device,
) -> Dict[str, Any]:
    """Profile model memory usage."""
    logger.info("Profiling memory...")

    profiler = MemoryProfiler(model, device)

    return profiler.profile_all(batch)


def profile_flops(
    model: torch.nn.Module,
    batch,
) -> Dict[str, Any]:
    """Estimate model FLOPs."""
    logger.info("Estimating FLOPs...")

    estimator = FLOPsEstimator(model)

    num_nodes = batch.num_nodes
    num_edges = batch.edge_index.size(1) // 2

    estimate = estimator.estimate(num_nodes, num_edges)

    return {
        **estimate.to_dict(),
        "parameters": estimator.count_parameters(),
    }


def profile_scaling(
    model_config: Dict[str, Any],
    device: torch.device,
    num_runs: int = 20,
) -> Dict[str, Any]:
    """Run scaling experiments."""
    logger.info("Running scaling experiments...")

    def model_factory(**kwargs):
        config = {**model_config, **kwargs}
        return get_model(**config)

    experiment = ScalingExperiment(
        model_factory,
        device=device,
        num_runs=num_runs,
    )

    # Node scaling
    logger.info("  Node scaling...")
    node_result = experiment.run_node_scaling(
        num_nodes_list=[100, 500, 1000, 2000],
        feature_dim=model_config.get("num_features", 16),
    )

    # Layer scaling
    logger.info("  Layer scaling...")
    layer_result = experiment.run_layer_scaling(
        num_layers_list=[2, 4, 6, 8, 12],
        feature_dim=model_config.get("num_features", 16),
    )

    return experiment.get_all_results()


def profile_pe_cost(
    batch,
    pe_configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Profile PE computation costs."""
    logger.info("Profiling PE computation costs...")

    from src.pe.node import LapPE, RWSE, HKS
    from src.profiling.runtime import profile_pe_computation

    results = {}

    pe_classes = {
        "lap": LapPE,
        "rwse": RWSE,
        "hks": HKS,
    }

    for config in pe_configs:
        pe_type = config.get("type", "lap")
        pe_dim = config.get("dim", 16)

        if pe_type in pe_classes:
            try:
                pe = pe_classes[pe_type](dim=pe_dim)
                stats = profile_pe_computation(pe.compute, batch, num_runs=10)
                results[pe_type] = {
                    "precomputation_ms": stats.mean,
                    "precomputation_std": stats.std,
                }
            except Exception as e:
                logger.warning(f"PE profiling failed for {pe_type}: {e}")
                results[pe_type] = {"error": str(e)}

    return results


def main():
    """Main profiling entry point."""
    parser = argparse.ArgumentParser(
        description="Profile Graph Transformer models"
    )

    # Model and dataset
    parser.add_argument(
        "--model",
        type=str,
        default="graph_transformer",
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (optional, uses synthetic if not provided)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Data root directory",
    )

    # Model configuration
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=500,
        help="Number of nodes for synthetic graph",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=16,
        help="Feature dimension",
    )

    # Profiling options
    parser.add_argument(
        "--profile",
        nargs="+",
        default=["runtime", "memory", "flops"],
        choices=["runtime", "memory", "flops", "scaling", "pe"],
        help="What to profile",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of runs for timing",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./profiling_results",
        help="Output directory",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Output format",
    )

    # Device and reproducibility
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("Model Profiling")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Profiles: {args.profile}")

    # Create or load data
    if args.dataset:
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = get_dataset(name=args.dataset, root=args.data_root)
        train_data, _, _ = dataset.get_splits()
        loader = DataLoader(train_data[:1], batch_size=1)
        batch = next(iter(loader)).to(device)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    else:
        logger.info(f"Using synthetic graph: {args.num_nodes} nodes")
        graphs = generate_scaling_graphs(
            [args.num_nodes],
            feature_dim=args.feature_dim,
            seed=args.seed,
        )
        batch = graphs[0].to(device)
        num_features = args.feature_dim
        num_classes = 1

    # Create model
    model_config = {
        "name": args.model,
        "num_features": num_features,
        "num_classes": num_classes,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }

    model = get_model(**model_config).to(device)

    # Results container
    results = {
        "config": {
            "model": args.model,
            "dataset": args.dataset or "synthetic",
            "num_nodes": batch.num_nodes,
            "num_edges": batch.edge_index.size(1) // 2,
            **model_config,
        },
        "hardware": get_hardware_info(),
        "model_info": get_model_info(model),
    }

    # Profile with context manager
    with ProfilingContext(model, seed=args.seed):
        # Runtime profiling
        if "runtime" in args.profile:
            results["runtime"] = profile_runtime(
                model, batch, device, args.num_runs
            )

        # Memory profiling
        if "memory" in args.profile:
            results["memory"] = profile_memory(model, batch, device)

        # FLOPs estimation
        if "flops" in args.profile:
            results["flops"] = profile_flops(model, batch)

        # Scaling experiments
        if "scaling" in args.profile:
            results["scaling"] = profile_scaling(
                model_config, device, args.num_runs // 5
            )

        # PE cost profiling
        if "pe" in args.profile:
            pe_configs = [
                {"type": "lap", "dim": 16},
                {"type": "rwse", "dim": 16},
                {"type": "hks", "dim": 16},
            ]
            results["pe_costs"] = profile_pe_cost(batch, pe_configs)

    # Print results
    print_profiling_table(results, title="Profiling Results")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"profile_{args.model}_{args.dataset or 'synthetic'}.{args.output_format}"
    save_profiling_results(results, str(output_file), format=args.output_format)
    logger.info(f"\nResults saved to {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Profiling completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

