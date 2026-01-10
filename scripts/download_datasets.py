"""Script to download and prepare datasets.

This script downloads datasets from their original sources and prepares them
for use in experiments. Datasets are automatically downloaded by PyTorch Geometric
on first access, but this script allows pre-downloading them explicitly.

Usage:
    # Download a specific dataset
    python scripts/download_datasets.py dataset=zinc

    # Download multiple datasets
    python scripts/download_datasets.py dataset=zinc dataset=qm9

    # Download with custom root directory
    python scripts/download_datasets.py dataset=peptides_func root=./data
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataset
from src.training.reproducibility import set_seed, log_system_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_dataset(
    dataset_name: str,
    root: str = "./data",
    **kwargs,
) -> bool:
    """
    Download and prepare a dataset.

    Args:
        dataset_name: Name of the dataset to download.
        root: Root directory for dataset storage.
        **kwargs: Additional dataset-specific parameters.

    Returns:
        True if download was successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info("=" * 60)

    try:
        # Disable PE for faster download (we only need to trigger download)
        pe_config = {
            "node": {"enabled": False},
            "relative": {"enabled": False},
        }

        # Get dataset-specific kwargs
        dataset_kwargs = kwargs.copy()
        
        # Add seed for synthetic datasets
        if dataset_name.lower().startswith("synthetic_"):
            dataset_kwargs["seed"] = 42

        logger.info(f"Root directory: {root}")
        logger.info(f"Dataset parameters: {dataset_kwargs}")

        # Initialize dataset (this triggers download if needed)
        logger.info("Initializing dataset (this will download if not already present)...")
        dataset = get_dataset(
            name=dataset_name,
            root=root,
            pe_config=pe_config,
            **dataset_kwargs,
        )

        # Access splits to trigger full download
        logger.info("Loading dataset splits...")
        train_data, val_data, test_data = dataset.get_splits()

        # Print statistics
        logger.info("Dataset statistics:")
        if train_data is not None:
            logger.info(f"  Train: {len(train_data)} graphs")
            if len(train_data) > 0:
                sample = train_data[0]
                logger.info(f"    Sample: {sample.num_nodes} nodes, {sample.edge_index.size(1)} edges")
        else:
            logger.warning("  Train: None")

        if val_data is not None:
            logger.info(f"  Validation: {len(val_data)} graphs")
        else:
            logger.warning("  Validation: None")

        if test_data is not None:
            logger.info(f"  Test: {len(test_data)} graphs")
        else:
            logger.warning("  Test: None")

        logger.info(f"  Node features: {dataset.num_features}")
        logger.info(f"  Classes: {dataset.num_classes}")

        logger.info("=" * 60)
        logger.info(f"Dataset '{dataset_name}' downloaded and ready!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Failed to download dataset '{dataset_name}': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for dataset downloading.

    Args:
        cfg: Hydra configuration.
    """
    logger.info("=" * 60)
    logger.info("Dataset Download Script")
    logger.info("=" * 60)

    # Setup
    seed = cfg.get("seed", 42)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    log_system_info()

    # Get dataset configuration
    dataset_name = cfg.dataset.name
    dataset_root = cfg.dataset.get("root", "./data")

    # Get dataset-specific parameters
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    dataset_kwargs = {k: v for k, v in dataset_cfg.items() if k not in ["name", "root"]}

    # Download dataset
    success = download_dataset(
        dataset_name=dataset_name,
        root=dataset_root,
        **dataset_kwargs,
    )

    if success:
        logger.info("\nNext steps:")
        logger.info("1. Precompute PE for this dataset:")
        logger.info(f"   python scripts/preprocess_pe.py dataset={dataset_name} pe=default")
        logger.info("2. Start training:")
        logger.info(f"   python scripts/train.py dataset={dataset_name}")
    else:
        logger.error("Dataset download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
