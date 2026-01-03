"""Factory function for creating datasets."""

from typing import Optional, Dict, Any, Union
from .base import BaseGraphDataset
from .zinc import ZINCDataset
from .qm9 import QM9Dataset
from .lrgb import LRGBDataset
from .ogb_mol import OGBMolDataset
from .pcqm import PCQM4MDatasetWrapper, PCQMContactDatasetWrapper
from .synthetic import SyntheticDataset


def get_dataset(
    name: str,
    root: str = "./data",
    pe_config: Optional[Dict[str, Any]] = None,
    splits: str = "official",
    **kwargs,
) -> BaseGraphDataset:
    """
    Factory function to create a dataset by name.

    Args:
        name: Dataset name. Supported names:
            - "zinc": ZINC molecular dataset
            - "qm9": QM9 molecular dataset
            - "peptides_func": LRGB Peptides-func
            - "peptides_struct": LRGB Peptides-struct
            - "pascalvoc_sp": LRGB PascalVOC-SP
            - "cifar10_sp": LRGB CIFAR10-SP
            - "ogbg-molhiv": OGB molecular HIV dataset
            - "ogbg-molpcba": OGB molecular PCBA dataset
            - "pcqm4m": PCQM4M dataset
            - "pcqm-contact": PCQM-Contact dataset
            - "synthetic_grid_2d": 2D grid graphs
            - "synthetic_grid_3d": 3D grid graphs
            - "synthetic_ring": Ring graphs
            - "synthetic_tree": Tree graphs
            - "synthetic_random_regular": Random regular graphs
            - "synthetic_barabasi_albert": Barabási–Albert graphs
            - "synthetic_watts_strogatz": Watts–Strogatz graphs
            - "synthetic_erdos_renyi": Erdős–Rényi graphs
            - "synthetic_sbm": Stochastic Block Model graphs
            - "synthetic_random_geometric": Random geometric graphs
        root: Root directory for dataset storage.
        pe_config: Configuration for positional encodings. Format:
            {
                "node": {
                    "enabled": True,
                    "types": ["lap_pe", "rwse", "hks"],
                    "dim": 32,
                    "scales": [1, 2, 4, 8]
                },
                "relative": {
                    "enabled": True,
                    "types": ["spd", "diffusion", "effective_resistance"],
                    "max_distance": 10,
                    "num_buckets": 16
                }
            }
        splits: Split type ("official" or "random").
        **kwargs: Additional arguments passed to dataset constructor.

    Returns:
        Dataset instance with train/val/test splits.

    Examples:
        >>> dataset = get_dataset(
        ...     name="zinc",
        ...     root="./data",
        ...     pe_config={
        ...         "node": {"enabled": True, "types": ["rwse"], "dim": 32},
        ...         "relative": {"enabled": True, "types": ["spd"], "num_buckets": 16}
        ...     }
        ... )
        >>> train_loader = DataLoader(dataset.train, batch_size=32)
    """
    name_lower = name.lower()

    # Define allowed parameters for each dataset type
    # Base parameters accepted by all datasets (via BaseGraphDataset)
    base_params = {"transform", "pre_transform"}
    
    # Dataset-specific allowed parameters
    zinc_allowed = base_params | {"subset"}
    qm9_allowed = base_params | {"target_idx", "normalize_targets"}
    lrgb_allowed = base_params  # Only base params
    ogb_allowed = base_params  # Only base params
    pcqm_allowed = base_params | {"subset"}

    # ZINC
    if name_lower == "zinc":
        # Exclude 'subset' from kwargs since it's passed explicitly
        zinc_kwargs = {k: v for k, v in kwargs.items() 
                      if k in zinc_allowed and k != "subset"}
        return ZINCDataset(
            root=root,
            pe_config=pe_config,
            subset=kwargs.get("subset", True),
            **zinc_kwargs,
        )

    # QM9
    elif name_lower == "qm9":
        # Exclude 'target_idx' and 'normalize_targets' from kwargs since they're passed explicitly
        qm9_kwargs = {k: v for k, v in kwargs.items() 
                     if k in qm9_allowed and k not in ["target_idx", "normalize_targets"]}
        return QM9Dataset(
            root=root,
            pe_config=pe_config,
            target_idx=kwargs.get("target_idx", 0),
            normalize_targets=kwargs.get("normalize_targets", True),
            **qm9_kwargs,
        )

    # LRGB datasets
    lrgb_kwargs = {k: v for k, v in kwargs.items() if k in lrgb_allowed}
    
    if name_lower == "peptides_func":
        return LRGBDataset(
            root=root,
            name="Peptides-func",
            pe_config=pe_config,
            **lrgb_kwargs,
        )
    elif name_lower == "peptides_struct":
        return LRGBDataset(
            root=root,
            name="Peptides-struct",
            pe_config=pe_config,
            **lrgb_kwargs,
        )
    elif name_lower == "pascalvoc_sp":
        return LRGBDataset(
            root=root,
            name="PascalVOC-SP",
            pe_config=pe_config,
            **lrgb_kwargs,
        )
    elif name_lower == "cifar10_sp":
        return LRGBDataset(
            root=root,
            name="CIFAR10-SP",
            pe_config=pe_config,
            **lrgb_kwargs,
        )

    # OGB molecular datasets
    elif name_lower in ["ogbg-molhiv", "molhiv"]:
        ogb_kwargs = {k: v for k, v in kwargs.items() if k in ogb_allowed}
        return OGBMolDataset(
            root=root,
            name="ogbg-molhiv",
            pe_config=pe_config,
            **ogb_kwargs,
        )
    elif name_lower in ["ogbg-molpcba", "molpcba"]:
        ogb_kwargs = {k: v for k, v in kwargs.items() if k in ogb_allowed}
        return OGBMolDataset(
            root=root,
            name="ogbg-molpcba",
            pe_config=pe_config,
            **ogb_kwargs,
        )

    # PCQM datasets
    elif name_lower == "pcqm4m":
        # Exclude 'subset' from kwargs since it's passed explicitly
        pcqm_kwargs = {k: v for k, v in kwargs.items() 
                      if k in pcqm_allowed and k != "subset"}
        return PCQM4MDatasetWrapper(
            root=root,
            pe_config=pe_config,
            subset=kwargs.get("subset", None),
            **pcqm_kwargs,
        )
    elif name_lower == "pcqm-contact":
        # Exclude 'subset' from kwargs since it's passed explicitly
        pcqm_kwargs = {k: v for k, v in kwargs.items() 
                      if k in pcqm_allowed and k != "subset"}
        return PCQMContactDatasetWrapper(
            root=root,
            pe_config=pe_config,
            subset=kwargs.get("subset", None),
            **pcqm_kwargs,
        )

    # Synthetic datasets
    elif name_lower.startswith("synthetic_"):
        graph_type = name_lower.replace("synthetic_", "")
        return SyntheticDataset(
            root=root,
            graph_type=graph_type,
            num_graphs=kwargs.get("num_graphs", 1000),
            graph_params=kwargs.get("graph_params", {}),
            pe_config=pe_config,
            num_classes=kwargs.get("num_classes", 1),
            task_type=kwargs.get("task_type", "regression"),
            task=kwargs.get("task", None),
            task_params=kwargs.get("task_params", {}),
            seed=kwargs.get("seed", 42),
            use_node_features=kwargs.get("use_node_features", True),
            **{k: v for k, v in kwargs.items() if k not in ["num_graphs", "graph_params", "num_classes", "task_type", "task", "task_params", "seed", "use_node_features"]},
        )

    else:
        raise ValueError(
            f"Unknown dataset name: {name}. "
            f"Supported datasets: zinc, qm9, peptides_func, peptides_struct, "
            f"pascalvoc_sp, cifar10_sp, ogbg-molhiv, ogbg-molpcba, "
            f"pcqm4m, pcqm-contact, synthetic_*"
        )


def list_available_datasets() -> Dict[str, list]:
    """
    List all available datasets grouped by category.

    Returns:
        Dictionary with categories and dataset names.
    """
    return {
        "molecular": ["zinc", "qm9"],
        "lrgb": [
            "peptides_func",
            "peptides_struct",
            "pascalvoc_sp",
            "cifar10_sp",
        ],
        "ogb": [
            "ogbg-molhiv",
            "ogbg-molpcba",
            "pcqm4m",
            "pcqm-contact",
        ],
        "synthetic": [
            "synthetic_grid_2d",
            "synthetic_grid_3d",
            "synthetic_ring",
            "synthetic_tree",
            "synthetic_random_regular",
            "synthetic_barabasi_albert",
            "synthetic_watts_strogatz",
            "synthetic_erdos_renyi",
            "synthetic_sbm",
            "synthetic_random_geometric",
        ],
    }

