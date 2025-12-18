"""Script for running single experiments."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix PyTorch 2.6+ safe globals for PyG
from torch_geometric.data import Data, Batch
torch.serialization.add_safe_globals([Data, Batch])

from src.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    get_experiment,
    register_experiment,
)
from src.training.reproducibility import set_seed


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def config_from_hydra(cfg: DictConfig) -> ExperimentConfig:
    """Convert Hydra config to ExperimentConfig."""
    return ExperimentConfig(
        name=cfg.get("experiment_name", "experiment"),
        dataset=OmegaConf.to_container(cfg.get("dataset", {}), resolve=True),
        model=OmegaConf.to_container(cfg.get("model", {}), resolve=True),
        pe=OmegaConf.to_container(cfg.get("pe", {}), resolve=True),
        training=OmegaConf.to_container(cfg.get("training", {}), resolve=True),
        seed=cfg.get("seed", 42),
        tags=list(cfg.get("tags", [])),
        description=cfg.get("description", ""),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main_hydra(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.

    Usage:
        python scripts/run_experiment.py dataset=zinc model=graph_transformer pe=mspe
    """
    logger.info("=" * 60)
    logger.info("Running Experiment")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create config
    config = config_from_hydra(cfg)

    # Setup output directory
    output_dir = cfg.get("output_dir", "./outputs")
    output_dir = Path(output_dir) / config.name

    # Run experiment
    runner = ExperimentRunner(
        config=config,
        output_dir=str(output_dir),
        resume=cfg.get("resume", False),
    )

    results = runner.run_all(
        run_long_range=cfg.get("run_long_range", True),
        run_profiling=cfg.get("run_profiling", False),
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Experiment Summary")
    logger.info("=" * 60)
    logger.info(f"Status: {results['status']}")

    if results.get("evaluation"):
        logger.info(f"Test metrics: {results['evaluation'].get('test_metrics', {})}")

    if results.get("long_range"):
        lr = results["long_range"]
        logger.info(f"Long-range AUC: {lr.get('auc', 'N/A')}")

    logger.info(f"Results saved to: {output_dir}")


def main_cli():
    """Command-line interface without Hydra."""
    parser = argparse.ArgumentParser(
        description="Run a single experiment"
    )

    # Experiment selection
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Registered experiment name (e.g., zinc_mspe)",
    )

    # Config overrides
    parser.add_argument(
        "--dataset",
        type=str,
        default="zinc",
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_transformer",
        help="Model name",
    )
    parser.add_argument(
        "--node_pe",
        type=str,
        default="combined",
        help="Node PE type",
    )
    parser.add_argument(
        "--relative_pe",
        type=str,
        default="spd",
        help="Relative PE type",
    )
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
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )

    # Run options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--no_long_range",
        action="store_true",
        help="Skip long-range evaluation",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiling",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Running Experiment (CLI)")
    logger.info("=" * 60)

    # Get or create config
    if args.experiment:
        try:
            config = get_experiment(args.experiment)
            logger.info(f"Using registered experiment: {args.experiment}")
        except KeyError:
            logger.error(f"Experiment not found: {args.experiment}")
            return
    else:
        # Create config from CLI args
        name = args.name or f"{args.dataset}_{args.model}_{args.node_pe}"

        config = ExperimentConfig(
            name=name,
            dataset={
                "name": args.dataset,
                "root": "./data",
            },
            model={
                "name": args.model,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            },
            pe={
                "node": {"type": args.node_pe, "dim": 32},
                "relative": {"type": args.relative_pe, "num_buckets": 32},
            },
            training={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
            seed=args.seed,
        )

    # Run experiment
    output_dir = Path(args.output_dir) / config.name

    runner = ExperimentRunner(
        config=config,
        output_dir=str(output_dir),
        resume=args.resume,
    )

    results = runner.run_all(
        run_long_range=not args.no_long_range,
        run_profiling=args.profile,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Experiment Complete")
    logger.info("=" * 60)
    logger.info(f"Status: {results['status']}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    # Check if running with Hydra
    if "--help" in sys.argv or any(arg.startswith("--") for arg in sys.argv[1:]):
        main_cli()
    else:
        main_hydra()

