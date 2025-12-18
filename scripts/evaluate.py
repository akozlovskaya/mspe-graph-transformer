"""Evaluation script for trained Graph Transformer models."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataset
from src.models import get_model
from src.training.metrics import get_metrics
from src.training.losses import get_loss_fn
from src.training.reproducibility import set_seed, get_device, log_system_info


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: DictConfig,
    dataset,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        cfg: Configuration.
        dataset: Dataset for model configuration.
        device: Device to load model to.

    Returns:
        Loaded model.
    """
    # Create model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["num_features"] = dataset.num_features
    model_cfg["num_classes"] = dataset.num_classes

    pe_cfg = cfg.get("pe", {})
    if pe_cfg.get("node", {}).get("dim"):
        model_cfg["node_pe_dim"] = pe_cfg.node.dim
    if pe_cfg.get("relative", {}).get("num_buckets"):
        model_cfg["relative_pe_dim"] = pe_cfg.relative.num_buckets

    model = get_model(**model_cfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    metrics,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model.
        loader: Data loader.
        loss_fn: Loss function.
        metrics: Metrics object.
        device: Device.

    Returns:
        Dictionary of metrics.
    """
    model.eval()
    metrics.reset()

    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        target = batch.y.to(device)

        pred = model(batch)
        loss = loss_fn(pred, target)

        metrics.update(pred, target)
        total_loss += loss.item()
        num_batches += 1

        all_predictions.append(pred.cpu())
        all_targets.append(target.cpu())

    eval_metrics = metrics.compute()
    eval_metrics["loss"] = total_loss / num_batches

    # Concatenate predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return eval_metrics, all_predictions, all_targets


def save_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_path: str,
):
    """
    Save predictions and targets to file.

    Args:
        predictions: Predictions tensor.
        targets: Targets tensor.
        output_path: Output file path.
    """
    torch.save(
        {
            "predictions": predictions,
            "targets": targets,
        },
        output_path,
    )
    logger.info(f"Saved predictions to {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation entry point.

    Args:
        cfg: Hydra configuration.
    """
    logger.info("=" * 60)
    logger.info("Starting evaluation")
    logger.info("=" * 60)

    # Setup
    seed = cfg.get("seed", 42)
    set_seed(seed, deterministic=True)
    log_system_info()

    device = get_device(cfg.get("device"))

    # Check checkpoint
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Load training config from checkpoint directory if available
    checkpoint_dir = Path(checkpoint_path).parent.parent
    train_config_path = checkpoint_dir / "config.yaml"

    if train_config_path.exists():
        logger.info(f"Loading training config from {train_config_path}")
        train_cfg = OmegaConf.load(train_config_path)
        # Merge with evaluation config
        cfg = OmegaConf.merge(train_cfg, cfg)

    # PE configuration
    pe_config = OmegaConf.to_container(cfg.get("pe", {}), resolve=True)

    # Load dataset
    dataset = get_dataset(
        name=cfg.dataset.name,
        root=cfg.dataset.get("root", "data/"),
        pe_config=pe_config if pe_config else None,
    )

    # Get splits
    train_data, val_data, test_data = dataset.get_splits()

    # Select evaluation split
    eval_split = cfg.get("eval_split", "test")
    if eval_split == "test" and test_data is not None:
        eval_data = test_data
    elif eval_split == "val":
        eval_data = val_data
    elif eval_split == "train":
        eval_data = train_data
    else:
        eval_data = test_data if test_data is not None else val_data

    logger.info(f"Evaluating on {eval_split} split with {len(eval_data)} samples")

    # Create dataloader
    batch_size = cfg.get("batch_size", cfg.training.get("batch_size", 32))
    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Load model
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, cfg, dataset, device
    )

    # Create loss and metrics
    task_type = cfg.dataset.get("task_type", "regression")
    num_classes = dataset.num_classes

    loss_fn = get_loss_fn(
        task_type=task_type,
        num_classes=num_classes,
    )
    metrics = get_metrics(task_type=task_type, num_classes=num_classes)

    # Evaluate
    eval_metrics, predictions, targets = evaluate_model(
        model=model,
        loader=eval_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    for metric_name, value in eval_metrics.items():
        logger.info(f"{metric_name}: {value:.6f}")

    # Save results
    output_dir = Path(cfg.get("output_dir", "eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    results = {
        "checkpoint": checkpoint_path,
        "eval_split": eval_split,
        "num_samples": len(eval_data),
        "metrics": {k: float(v) for k, v in eval_metrics.items()},
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": seed,
    }

    results_path = output_dir / f"eval_results_{eval_split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Save predictions if requested
    if cfg.get("save_predictions", False):
        pred_path = output_dir / f"predictions_{eval_split}.pt"
        save_predictions(predictions, targets, str(pred_path))

    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
