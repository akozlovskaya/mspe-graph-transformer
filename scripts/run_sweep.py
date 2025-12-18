"""Script for running experiment sweeps."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix PyTorch 2.6+ safe globals for PyG
torch.serialization.add_safe_globals([Data, Batch])

from src.experiments import (
    SweepConfig,
    GridSweep,
    RandomSweep,
    SeedSweep,
    SweepRunner,
    run_sweep,
    resume_sweep,
    ExperimentConfig,
    get_experiment,
    aggregate_results,
    export_results_table,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_sweep_config(config_path: str) -> SweepConfig:
    """Load sweep configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return SweepConfig(
        name=config_dict["name"],
        base_config=config_dict["base_config"],
        parameters=config_dict.get("parameters", {}),
        sweep_type=config_dict.get("sweep_type", "grid"),
        num_samples=config_dict.get("num_samples", 10),
        seeds=config_dict.get("seeds", [42, 123, 456]),
        tags=config_dict.get("tags", []),
        description=config_dict.get("description", ""),
    )


def create_pe_ablation_sweep() -> SweepConfig:
    """Create PE ablation sweep."""
    base_config = {
        "name": "pe_ablation",
        "dataset": {"name": "zinc", "root": "./data"},
        "model": {"name": "graph_transformer", "hidden_dim": 256, "num_layers": 6},
        "pe": {
            "node": {"type": "combined", "dim": 32},
            "relative": {"type": "spd", "num_buckets": 32},
        },
        "training": {"epochs": 100, "batch_size": 32, "lr": 1e-4},
        "seed": 42,
    }

    return SweepConfig(
        name="pe_ablation",
        base_config=base_config,
        parameters={
            "pe.node.type": ["none", "lap", "rwse", "hks", "combined"],
            "pe.relative.type": ["none", "spd", "diffusion"],
        },
        sweep_type="grid",
        tags=["ablation", "pe"],
        description="Ablation over PE types",
    )


def create_model_depth_sweep() -> SweepConfig:
    """Create model depth sweep."""
    base_config = {
        "name": "depth_sweep",
        "dataset": {"name": "zinc", "root": "./data"},
        "model": {"name": "graph_transformer", "hidden_dim": 256, "num_layers": 6},
        "pe": {
            "node": {"type": "combined", "dim": 32},
            "relative": {"type": "spd", "num_buckets": 32},
        },
        "training": {"epochs": 100, "batch_size": 32, "lr": 1e-4},
        "seed": 42,
    }

    return SweepConfig(
        name="depth_sweep",
        base_config=base_config,
        parameters={
            "model.num_layers": [2, 4, 6, 8, 12, 16],
        },
        sweep_type="grid",
        tags=["ablation", "depth"],
        description="Sweep over model depth",
    )


def create_seed_sweep(base_experiment: str) -> SweepConfig:
    """Create seed sweep for reproducibility study."""
    try:
        base = get_experiment(base_experiment)
        base_config = base.to_dict()
    except KeyError:
        # Default base config
        base_config = {
            "name": f"{base_experiment}_seed",
            "dataset": {"name": "zinc", "root": "./data"},
            "model": {"name": "graph_transformer", "hidden_dim": 256, "num_layers": 6},
            "pe": {
                "node": {"type": "combined", "dim": 32},
                "relative": {"type": "spd", "num_buckets": 32},
            },
            "training": {"epochs": 100, "batch_size": 32, "lr": 1e-4},
            "seed": 42,
        }

    return SweepConfig(
        name=f"{base_experiment}_seeds",
        base_config=base_config,
        parameters={},
        sweep_type="seed",
        seeds=[42, 123, 456, 789, 1024],
        tags=["reproducibility"],
        description=f"Seed sweep for {base_experiment}",
    )


PREDEFINED_SWEEPS = {
    "pe_ablation": create_pe_ablation_sweep,
    "depth_sweep": create_model_depth_sweep,
}


def main():
    """Main entry point for sweep runner."""
    parser = argparse.ArgumentParser(
        description="Run experiment sweeps"
    )

    # Sweep selection
    parser.add_argument(
        "--sweep",
        type=str,
        help="Predefined sweep name or path to sweep config YAML",
    )
    parser.add_argument(
        "--sweep_type",
        type=str,
        default="grid",
        choices=["grid", "random", "seed"],
        help="Sweep type for custom sweeps",
    )

    # Sweep parameters
    parser.add_argument(
        "--base_experiment",
        type=str,
        default=None,
        help="Base experiment name for seed sweep",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        nargs="+",
        help="Parameters to sweep (format: key=val1,val2,val3)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Seeds for seed sweep",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/sweeps",
        help="Output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Sweep name",
    )

    # Execution options
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to sweep directory to resume",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configurations without running",
    )
    parser.add_argument(
        "--export_table",
        type=str,
        default=None,
        help="Export results table to file",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Experiment Sweep Runner")
    logger.info("=" * 60)

    # Resume incomplete sweep
    if args.resume:
        logger.info(f"Resuming sweep from: {args.resume}")
        state = resume_sweep(args.resume)
        logger.info(f"Completed: {len(state.completed)}")
        logger.info(f"Failed: {len(state.failed)}")
        return

    # Get or create sweep config
    if args.sweep:
        if args.sweep in PREDEFINED_SWEEPS:
            # Predefined sweep
            sweep_config = PREDEFINED_SWEEPS[args.sweep]()
            logger.info(f"Using predefined sweep: {args.sweep}")
        elif Path(args.sweep).exists():
            # Load from file
            sweep_config = load_sweep_config(args.sweep)
            logger.info(f"Loaded sweep from: {args.sweep}")
        else:
            logger.error(f"Sweep not found: {args.sweep}")
            logger.info(f"Available predefined sweeps: {list(PREDEFINED_SWEEPS.keys())}")
            return
    elif args.sweep_type == "seed" and args.base_experiment:
        # Seed sweep
        sweep_config = create_seed_sweep(args.base_experiment)
        sweep_config.seeds = args.seeds
    elif args.parameters:
        # Custom grid sweep from CLI
        parameters = {}
        for param in args.parameters:
            key, values = param.split("=")
            parameters[key] = [_parse_value(v) for v in values.split(",")]

        sweep_config = SweepConfig(
            name=args.name or "custom_sweep",
            base_config={
                "name": "custom",
                "dataset": {"name": "zinc", "root": "./data"},
                "model": {"name": "graph_transformer", "hidden_dim": 256, "num_layers": 6},
                "pe": {"node": {"type": "combined", "dim": 32}, "relative": {"type": "spd"}},
                "training": {"epochs": 100, "batch_size": 32, "lr": 1e-4},
                "seed": 42,
            },
            parameters=parameters,
            sweep_type=args.sweep_type,
        )
    else:
        parser.print_help()
        return

    # Override name if provided
    if args.name:
        sweep_config.name = args.name

    # Print sweep info
    logger.info(f"Sweep: {sweep_config.name}")
    logger.info(f"Type: {sweep_config.sweep_type}")
    logger.info(f"Total experiments: {sweep_config.get_num_experiments()}")

    if sweep_config.parameters:
        logger.info("Parameters:")
        for key, values in sweep_config.parameters.items():
            logger.info(f"  {key}: {values}")

    # Dry run - just print configs
    if args.dry_run:
        logger.info("\nDry run - configurations:")
        for i, config in enumerate(sweep_config.generate_configs()):
            logger.info(f"\n{i + 1}. {config.get('name', f'exp_{i}')}")
            for key in ["dataset", "model", "pe"]:
                logger.info(f"   {key}: {config.get(key, {})}")
        return

    # Run sweep
    logger.info("\nStarting sweep execution...")
    runner = SweepRunner(sweep_config, args.output_dir)

    def on_complete(name, results):
        logger.info(f"✓ Completed: {name}")

    def on_failed(name, error):
        logger.error(f"✗ Failed: {name} - {error}")

    state = runner.run(
        on_experiment_complete=on_complete,
        on_experiment_failed=on_failed,
    )

    # Print summary
    summary = runner.get_summary()
    logger.info("\n" + "=" * 60)
    logger.info("Sweep Summary")
    logger.info("=" * 60)
    logger.info(f"Total: {summary['total']}")
    logger.info(f"Completed: {summary['completed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Pending: {summary['pending']}")

    # Export table if requested
    if args.export_table:
        results_list = []
        for exp_name in state.completed:
            results_path = Path(args.output_dir) / sweep_config.name / exp_name / "results.json"
            if results_path.exists():
                import json
                with open(results_path) as f:
                    results_list.append(json.load(f))

        if results_list:
            export_results_table(results_list, args.export_table)
            logger.info(f"Results exported to: {args.export_table}")


def _parse_value(s: str):
    """Parse string to appropriate type."""
    # Try int
    try:
        return int(s)
    except ValueError:
        pass

    # Try float
    try:
        return float(s)
    except ValueError:
        pass

    # Try bool
    if s.lower() in ("true", "false"):
        return s.lower() == "true"

    # Return as string
    return s


if __name__ == "__main__":
    main()

