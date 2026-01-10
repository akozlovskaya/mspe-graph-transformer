"""Script to precompute positional encodings for datasets.

This script precomputes and caches positional encodings for datasets to speed up
training. It processes all graphs in train/val/test splits and saves them with
precomputed PE attributes.

Usage:
    # Basic usage with default config
    python scripts/preprocess_pe.py dataset=zinc pe=default

    # With custom PE configuration
    python scripts/preprocess_pe.py dataset=zinc pe=mspe pe_cache_dir=./data/pe_cache

    # For synthetic datasets
    python scripts/preprocess_pe.py dataset=synthetic/pairwise_distance pe=light

The preprocessed data is saved to:
    {cache_dir}/{dataset_name}/train.pt
    {cache_dir}/{dataset_name}/val.pt
    {cache_dir}/{dataset_name}/test.pt
    {cache_dir}/{dataset_name}/pe_config.yaml
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch_geometric.data import Data, InMemoryDataset

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


class PreprocessedDataset(InMemoryDataset):
    """In-memory dataset for preprocessed graphs with PE."""

    def __init__(
        self,
        data_list: list,
        root: str,
        name: str = "preprocessed",
        transform=None,
        pre_transform=None,
    ):
        """
        Initialize preprocessed dataset.

        Args:
            data_list: List of Data objects with precomputed PE.
            root: Root directory for dataset.
            name: Dataset name.
            transform: Optional transform to apply.
            pre_transform: Optional pre-transform.
        """
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def preprocess_split(
    dataset,
    split_name: str,
    cache_dir: Path,
    num_workers: int = 0,
) -> InMemoryDataset:
    """
    Preprocess a dataset split and save to cache.

    Args:
        dataset: Dataset split to preprocess.
        split_name: Name of split ("train", "val", "test").
        cache_dir: Directory to save preprocessed data.
        num_workers: Number of workers for processing (not used, kept for compatibility).

    Returns:
        Preprocessed dataset.
    """
    logger.info(f"Preprocessing {split_name} split ({len(dataset)} graphs)...")

    # Process graphs one by one (PE is already computed by transforms)
    preprocessed_data = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
        graph = dataset[i]
        
        # Ensure graph has required attributes
        if not hasattr(graph, "node_pe") or graph.node_pe is None:
            logger.warning(f"Graph {i} in {split_name} has no node_pe, adding zeros")
            num_nodes = graph.num_nodes
            pe_dim = 32  # Default
            graph.node_pe = torch.zeros(num_nodes, pe_dim, dtype=torch.float32)
        
        if not hasattr(graph, "edge_pe") or graph.edge_pe is None:
            logger.warning(f"Graph {i} in {split_name} has no edge_pe, adding zeros")
            num_edges = graph.edge_index.size(1) if hasattr(graph, "edge_index") else 0
            num_buckets = 32  # Default
            graph.edge_pe = torch.zeros(num_edges, num_buckets, dtype=torch.float32)
            if hasattr(graph, "edge_index"):
                graph.edge_pe_index = graph.edge_index
        
        preprocessed_data.append(graph)

    # Save to disk
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_path = cache_dir / f"{split_name}.pt"
    
    # Save as list of Data objects
    torch.save(preprocessed_data, save_path)
    logger.info(f"Saved preprocessed {split_name} ({len(preprocessed_data)} graphs) to {save_path}")

    # Create and return preprocessed dataset
    preprocessed_dataset = PreprocessedDataset(
        preprocessed_data,
        root=str(cache_dir / split_name),
        name=f"{split_name}_preprocessed",
    )

    return preprocessed_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for PE preprocessing.

    Args:
        cfg: Hydra configuration.
    """
    logger.info("=" * 60)
    logger.info("PE Preprocessing Script")
    logger.info("=" * 60)

    # Setup
    seed = cfg.get("seed", 42)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    log_system_info()

    # Get dataset configuration
    dataset_name = cfg.dataset.name
    dataset_root = cfg.dataset.get("root", "./data")
    
    # PE configuration
    pe_config = OmegaConf.to_container(cfg.get("pe", {}), resolve=True)
    if not pe_config:
        logger.warning("No PE configuration provided. Using default PE settings.")
        pe_config = {
            "node": {"enabled": True, "type": "rwse", "dim": 32},
            "relative": {"enabled": True, "type": "spd", "num_buckets": 32},
        }

    # Cache directory (use pe_cache_dir if provided, otherwise default to {root}/pe_cache)
    pe_cache_dir = cfg.get("pe_cache_dir", None)
    if pe_cache_dir is None:
        pe_cache_dir = Path(dataset_root) / "pe_cache"
    else:
        pe_cache_dir = Path(pe_cache_dir)
    cache_dir = pe_cache_dir / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save PE config to cache directory
    config_path = cache_dir / "pe_config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save({"pe": pe_config}, f)
    logger.info(f"PE config saved to {config_path}")

    # Get dataset-specific kwargs
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    dataset_kwargs = {k: v for k, v in dataset_cfg.items() if k not in ["name", "root"]}
    
    # Add seed for synthetic datasets
    if dataset_name.lower().startswith("synthetic_"):
        dataset_kwargs["seed"] = seed

    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"PE config: {pe_config}")

    # Load dataset with PE transforms
    dataset = get_dataset(
        name=dataset_name,
        root=dataset_root,
        pe_config=pe_config,
        **dataset_kwargs,
    )

    # Get splits
    train_data, val_data, test_data = dataset.get_splits()

    # Processing settings
    num_workers = cfg.get("num_workers", 0)

    # Preprocess each split
    if train_data is not None:
        preprocess_split(
            train_data,
            "train",
            cache_dir,
            num_workers=num_workers,
        )
    else:
        logger.warning("Train dataset is None, skipping...")

    if val_data is not None:
        preprocess_split(
            val_data,
            "val",
            cache_dir,
            num_workers=num_workers,
        )
    else:
        logger.warning("Validation dataset is None, skipping...")

    if test_data is not None:
        preprocess_split(
            test_data,
            "test",
            cache_dir,
            num_workers=num_workers,
        )
    else:
        logger.warning("Test dataset is None, skipping...")

    logger.info("=" * 60)
    logger.info("PE preprocessing completed!")
    logger.info(f"Preprocessed data saved to: {cache_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
