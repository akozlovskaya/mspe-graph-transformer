"""Main training script for Graph Transformers with multi-scale positional encodings."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix PyTorch 2.6+ safe globals for PyG
torch.serialization.add_safe_globals([Data, Batch])

from src.dataset import get_dataset
from src.models import get_model
from src.training.trainer import Trainer
from src.training.optimizer import get_optimizer_from_config
from src.training.scheduler import get_scheduler_from_config
from src.training.losses import get_loss_fn
from src.training.metrics import get_metrics
from src.training.reproducibility import (
    set_seed,
    get_device,
    log_system_info,
    log_model_info,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_logging_dir(cfg: DictConfig) -> Path:
    """Setup logging directory and save config."""
    log_dir = Path(cfg.get("log_dir", "logs")) / cfg.get("experiment_name", "experiment")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = log_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    logger.info(f"Logging directory: {log_dir}")
    return log_dir


def create_dataloaders(
    cfg: DictConfig,
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Args:
        cfg: Configuration with dataset and training settings.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset).
    """
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

    # Create dataloaders
    batch_size = cfg.training.get("batch_size", 32)
    num_workers = cfg.training.get("num_workers", 4)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if test_data is not None else None

    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    if test_data:
        logger.info(f"Test samples: {len(test_data)}")

    return train_loader, val_loader, test_loader, dataset


def create_model(cfg: DictConfig, dataset) -> torch.nn.Module:
    """
    Create model based on configuration.

    Args:
        cfg: Configuration with model settings.
        dataset: Dataset object (for num_features, num_classes).

    Returns:
        PyTorch model.
    """
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    # Add dataset-specific parameters
    model_cfg["num_features"] = dataset.num_features
    model_cfg["num_classes"] = dataset.num_classes

    # Add PE dimensions if available
    pe_cfg = cfg.get("pe", {})
    if pe_cfg.get("node", {}).get("dim"):
        model_cfg["node_pe_dim"] = pe_cfg.node.dim
    if pe_cfg.get("relative", {}).get("num_buckets"):
        model_cfg["relative_pe_dim"] = pe_cfg.relative.num_buckets

    model = get_model(**model_cfg)
    log_model_info(model, name=model_cfg.get("name", "Model"))

    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Log configuration
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup
    log_dir = setup_logging_dir(cfg)
    seed = cfg.get("seed", 42)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    log_system_info()

    device = get_device(cfg.get("device"))

    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(cfg)

    # Create model
    model = create_model(cfg, dataset)

    # Create optimizer
    optimizer = get_optimizer_from_config(model, OmegaConf.to_container(cfg.training.optimizer, resolve=True))

    # Create scheduler
    scheduler_cfg = OmegaConf.to_container(cfg.training.get("scheduler", {}), resolve=True)
    scheduler_cfg["total_epochs"] = cfg.training.epochs
    scheduler = get_scheduler_from_config(optimizer, scheduler_cfg)

    # Create loss function
    task_type = cfg.dataset.get("task_type", "regression")
    num_classes = dataset.num_classes
    loss_fn = get_loss_fn(
        task_type=task_type,
        loss_type=cfg.training.get("loss_type"),
        num_classes=num_classes,
    )

    # Create metrics
    metrics = get_metrics(task_type=task_type, num_classes=num_classes)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
        task_type=task_type,
        num_classes=num_classes,
        grad_clip=cfg.training.get("grad_clip"),
        mixed_precision=cfg.training.get("mixed_precision", False),
        log_every=cfg.training.get("log_every", 50),
        checkpoint_dir=str(log_dir / "checkpoints"),
        experiment_name=cfg.get("experiment_name", "experiment"),
    )

    # Load checkpoint if specified
    if cfg.get("resume_from"):
        trainer.load_checkpoint(cfg.resume_from)
        logger.info(f"Resumed from checkpoint: {cfg.resume_from}")

    # Training
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        metric_for_best=cfg.training.get("metric_for_best", "loss"),
        minimize_metric=cfg.training.get("minimize_metric", True),
        early_stopping=cfg.training.get("early_stopping"),
    )

    # Final evaluation on test set
    if test_loader is not None:
        logger.info("=" * 60)
        logger.info("Final evaluation on test set")
        logger.info("=" * 60)

        # Load best model
        best_checkpoint = log_dir / "checkpoints" / "best.pt"
        if best_checkpoint.exists():
            trainer.load_checkpoint(str(best_checkpoint), load_optimizer=False)

        test_metrics = trainer.eval_epoch(test_loader)
        logger.info(f"Test metrics: {test_metrics}")

        # Save test results
        import json
        test_results = {
            "test_metrics": test_metrics,
            "best_epoch": trainer.best_epoch,
            "best_val_metric": trainer.best_metric,
            "seed": seed,
        }
        with open(log_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
