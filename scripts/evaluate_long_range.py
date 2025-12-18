"""Script for long-range dependency evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataset
from src.models import get_model
from src.evaluation import (
    LongRangeEvaluator,
    add_distance_info_to_data,
    compute_model_config_hash,
    save_evaluation_results,
    print_evaluation_summary,
    plot_distance_performance,
    EvaluationLogger,
)
from src.training.reproducibility import set_seed, get_device


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model_config: Model configuration dictionary.
        device: Device to load model to.

    Returns:
        Loaded model in eval mode.
    """
    model = get_model(**model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


def prepare_dataset_with_distances(
    dataset_name: str,
    root: str,
    max_distance: int,
    pe_config: Optional[Dict] = None,
    split: str = "test",
) -> DataLoader:
    """
    Load dataset and add distance information.

    Args:
        dataset_name: Name of dataset.
        root: Data root directory.
        max_distance: Maximum distance to compute.
        pe_config: Optional PE configuration.
        split: Dataset split to use.

    Returns:
        DataLoader with distance-annotated graphs.
    """
    # Load dataset
    dataset = get_dataset(
        name=dataset_name,
        root=root,
        pe_config=pe_config,
    )

    # Get split
    train_data, val_data, test_data = dataset.get_splits()

    if split == "test" and test_data is not None:
        data = test_data
    elif split == "val":
        data = val_data
    else:
        data = train_data

    # Add distance info to each graph
    logger.info(f"Computing distances for {len(data)} graphs...")
    for i, graph in enumerate(data):
        add_distance_info_to_data(
            graph,
            max_distance=max_distance,
            sparse=True,
        )
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(data)} graphs")

    loader = DataLoader(data, batch_size=1, shuffle=False)

    return loader, dataset


def get_distances_for_batch(batch, task_type: str = "node") -> torch.Tensor:
    """
    Extract relevant distances from batch based on task type.

    Args:
        batch: PyG Batch object.
        task_type: Type of task for distance interpretation.

    Returns:
        Distance tensor aligned with predictions.
    """
    if hasattr(batch, "distance_values"):
        # For node tasks, use distance from first node or mean distance
        # This is a simplified version - real implementation depends on task
        if task_type == "node":
            # Return distance to node 0 for each node
            if hasattr(batch, "distance_matrix"):
                return batch.distance_matrix[0]
            else:
                # Use sparse distances
                num_nodes = batch.num_nodes
                distances = torch.full((num_nodes,), -1, dtype=torch.long)

                pairs = batch.distance_pairs
                values = batch.distance_values

                mask = pairs[0] == 0
                distances[pairs[1][mask]] = values[mask]
                distances[0] = 0

                return distances
        else:
            # For graph tasks, return graph diameter or similar
            if hasattr(batch, "diameter"):
                return torch.tensor([batch.diameter])
            else:
                return torch.tensor([batch.distance_values.max().item()])
    else:
        return torch.zeros(batch.num_nodes, dtype=torch.long)


@torch.no_grad()
def run_long_range_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    evaluator: LongRangeEvaluator,
    device: torch.device,
    task_type: str = "node",
    return_intermediate: bool = False,
) -> Dict[str, Any]:
    """
    Run long-range evaluation.

    Args:
        model: PyTorch model.
        loader: DataLoader with distance-annotated graphs.
        evaluator: LongRangeEvaluator instance.
        device: Device.
        task_type: Task type for distance extraction.
        return_intermediate: Whether to collect intermediate states.

    Returns:
        Evaluation results.
    """
    model.eval()
    evaluator.reset()

    all_intermediate = [] if return_intermediate else None

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        # Forward pass
        if return_intermediate and hasattr(model, "forward"):
            try:
                output = model(batch, return_intermediate=True)
                if isinstance(output, tuple):
                    pred, intermediate = output
                    all_intermediate.append(intermediate)
                else:
                    pred = output
            except TypeError:
                pred = model(batch)
        else:
            pred = model(batch)

        # Get targets and distances
        target = batch.y if hasattr(batch, "y") else batch.target
        distances = get_distances_for_batch(batch, task_type)

        # Update evaluator
        evaluator.update(pred, target, distances.to(device))

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Evaluated {batch_idx + 1}/{len(loader)} batches")

    results = evaluator.compute()

    if all_intermediate:
        results["intermediate_states"] = all_intermediate

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Long-range dependency evaluation for Graph Transformers"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., peptides_func, zinc)",
    )

    # Optional arguments
    parser.add_argument(
        "--max_distance",
        type=int,
        default=20,
        help="Maximum distance to evaluate",
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=1,
        help="Size of distance buckets",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification", "binary"],
        help="Task type for metrics",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Data root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./long_range_results",
        help="Output directory for results",
    )
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
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save visualization plots",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to model config JSON (if not in checkpoint)",
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("Long-Range Dependency Evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Max distance: {args.max_distance}")
    logger.info(f"Task type: {args.task_type}")

    # Load model config
    checkpoint_dir = Path(args.checkpoint).parent.parent
    config_path = checkpoint_dir / "config.yaml"

    if args.model_config:
        with open(args.model_config) as f:
            model_config = json.load(f)
    elif config_path.exists():
        import yaml
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        model_config = full_config.get("model", {})
    else:
        logger.warning("No model config found, using defaults")
        model_config = {
            "name": "graph_transformer",
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
        }

    # Prepare dataset
    logger.info("\nPreparing dataset with distance annotations...")
    loader, dataset = prepare_dataset_with_distances(
        dataset_name=args.dataset,
        root=args.data_root,
        max_distance=args.max_distance,
        split=args.split,
    )

    # Update model config with dataset info
    model_config["num_features"] = dataset.num_features
    model_config["num_classes"] = dataset.num_classes

    # Load model
    logger.info("\nLoading model...")
    model = load_model_from_checkpoint(args.checkpoint, model_config, device)

    # Create evaluator
    evaluator = LongRangeEvaluator(
        max_distance=args.max_distance,
        bucket_size=args.bucket_size,
        task_type=args.task_type,
    )

    # Run evaluation
    logger.info("\nRunning evaluation...")
    results = run_long_range_evaluation(
        model=model,
        loader=loader,
        evaluator=evaluator,
        device=device,
        task_type="node" if args.task_type != "graph" else "graph",
    )

    # Add metadata
    results["metadata"] = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "model_config_hash": compute_model_config_hash(model_config),
        "max_distance": args.max_distance,
        "bucket_size": args.bucket_size,
        "task_type": args.task_type,
        "split": args.split,
        "seed": args.seed,
    }

    # Print summary
    print_evaluation_summary(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"long_range_{args.dataset}_{args.split}.json"
    save_evaluation_results(results, str(results_path))
    logger.info(f"\nResults saved to {results_path}")

    # Save plots
    if args.save_plots:
        plot_path = output_dir / f"distance_performance_{args.dataset}.png"
        plot_distance_performance(
            results["metrics_per_bucket"],
            title=f"Performance vs Distance - {args.dataset}",
            output_path=str(plot_path),
        )
        logger.info(f"Plot saved to {plot_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

