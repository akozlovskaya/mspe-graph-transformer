"""Transforms for applying positional encodings to graphs."""

from typing import Optional, Dict, Any, List, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ApplyNodePE(BaseTransform):
    """Transform to apply node-wise positional encodings."""

    def __init__(self, node_pe_config: Dict[str, Any], cache_dir: Optional[str] = None):
        """
        Initialize node PE transform.

        Args:
            node_pe_config: Configuration dict with keys:
                - enabled: bool, whether to apply node PE
                - types: List[str], e.g., ["lap_pe", "rwse", "hks"]
                - dim: int, dimension of PE
                - scales: List[float], scales for multi-scale PE
            cache_dir: Directory to cache computed PEs
        """
        self.node_pe_config = node_pe_config
        self.cache_dir = cache_dir
        self.enabled = node_pe_config.get("enabled", True)
        self.types = node_pe_config.get("types", [])
        self.dim = node_pe_config.get("dim", 32)
        self.scales = node_pe_config.get("scales", [1, 2, 4, 8])

    def __call__(self, data: Data) -> Data:
        """
        Apply node PE to graph.

        Args:
            data: PyG Data object

        Returns:
            Data object with node_pe attribute added
        """
        if not self.enabled or not self.types:
            # Add zero PE if disabled
            data.node_pe = torch.zeros(data.num_nodes, self.dim, dtype=torch.float32)
            return data

        node_pe = self._compute_node_pe(data)

        data.node_pe = node_pe
        return data

    def _compute_node_pe(self, data: Data) -> torch.Tensor:
        """
        Compute node positional encodings.

        Args:
            data: PyG Data object

        Returns:
            Tensor of shape [num_nodes, pe_dim]
        """
        from src.pe.node import LapPE, RWSE, HKS, RolePE

        pe_types = self.types
        if not pe_types:
            # Fallback: use zeros
            return torch.zeros(data.num_nodes, self.dim, dtype=torch.float32)

        # Combine different PE types
        all_pe = []
        dim_per_type = self.dim // len(pe_types) if len(pe_types) > 0 else self.dim

        for pe_type in pe_types:
            pe_type_lower = pe_type.lower()
            
            if pe_type_lower in ["lappe", "lap_pe"]:
                pe_computer = LapPE(
                    dim=dim_per_type,
                    k=dim_per_type // 2 if dim_per_type > 1 else 1,
                    scales=[8, 16],
                    sign_invariant=self.node_pe_config.get("sign_invariant", True),
                    cache=False,  # We handle caching at transform level
                )
            elif pe_type_lower == "rwse":
                pe_computer = RWSE(
                    dim=dim_per_type,
                    scales=self.scales,
                    cache=False,
                )
            elif pe_type_lower == "hks":
                pe_computer = HKS(
                    dim=dim_per_type,
                    scales=self.scales,
                    cache=False,
                )
            elif pe_type_lower == "role":
                pe_computer = RolePE(
                    dim=dim_per_type,
                    features=["degree", "clustering", "core"],
                    cache=False,
                )
            else:
                # Unknown type, skip or use zeros
                continue

            try:
                pe = pe_computer.compute(data)
                # Project to target dimension if needed
                if pe.shape[1] != dim_per_type:
                    if pe.shape[1] < dim_per_type:
                        padding = torch.zeros(pe.shape[0], dim_per_type - pe.shape[1], device=pe.device)
                        pe = torch.cat([pe, padding], dim=1)
                    else:
                        pe = pe[:, :dim_per_type]
                all_pe.append(pe)
            except Exception as e:
                # Fallback: use zeros if computation fails
                pe = torch.zeros(data.num_nodes, dim_per_type, dtype=torch.float32)
                all_pe.append(pe)

        if all_pe:
            # Concatenate all PE types
            node_pe = torch.cat(all_pe, dim=1)
            # Truncate or pad to target dimension
            if node_pe.shape[1] > self.dim:
                node_pe = node_pe[:, :self.dim]
            elif node_pe.shape[1] < self.dim:
                padding = torch.zeros(node_pe.shape[0], self.dim - node_pe.shape[1], device=node_pe.device)
                node_pe = torch.cat([node_pe, padding], dim=1)
        else:
            # Fallback: zeros
            node_pe = torch.zeros(data.num_nodes, self.dim, dtype=torch.float32)

        return node_pe


class ApplyRelativePE(BaseTransform):
    """Transform to apply relative pairwise positional encodings."""

    def __init__(
        self,
        relative_pe_config: Dict[str, Any],
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize relative PE transform.

        Args:
            relative_pe_config: Configuration dict with keys:
                - enabled: bool, whether to apply relative PE
                - types: List[str], e.g., ["spd", "diffusion", "effective_resistance"]
                - max_distance: int, maximum distance to consider
                - num_buckets: int, number of distance buckets
            cache_dir: Directory to cache computed PEs
        """
        self.relative_pe_config = relative_pe_config
        self.cache_dir = cache_dir
        self.enabled = relative_pe_config.get("enabled", True)
        self.types = relative_pe_config.get("types", [])
        self.max_distance = relative_pe_config.get("max_distance", 10)
        self.num_buckets = relative_pe_config.get("num_buckets", 16)

    def __call__(self, data: Data) -> Data:
        """
        Apply relative PE to graph.

        Args:
            data: PyG Data object

        Returns:
            Data object with edge_pe attribute added
        """
        if not self.enabled or not self.types:
            # Add zero PE if disabled (on edges)
            num_edges = data.edge_index.size(1) if hasattr(data, "edge_index") else 0
            data.edge_pe_index = data.edge_index if hasattr(data, "edge_index") else None
            data.edge_pe = torch.zeros(num_edges, self.num_buckets, dtype=torch.float32)
            return data

        # Compute relative PE
        edge_index_pe, edge_attr_pe = self._compute_relative_pe(data)

        data.edge_pe_index = edge_index_pe
        data.edge_pe = edge_attr_pe
        return data

    def _compute_relative_pe(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relative positional encodings.

        Args:
            data: PyG Data object

        Returns:
            Tuple of (edge_index_pe, edge_attr_pe):
                - edge_index_pe: [2, num_pairs] with pair indices
                - edge_attr_pe: [num_pairs, num_buckets] with PE values
        """
        from src.pe.relative import (
            SPDBuckets,
            DiffusionPE,
            EffectiveResistancePE,
            BFSDistance,
        )

        pe_types = self.types
        if not pe_types:
            # Fallback: use zeros on edges
            num_edges = data.edge_index.size(1)
            edge_index_pe = data.edge_index
            edge_attr_pe = torch.zeros(num_edges, self.num_buckets, dtype=torch.float32)
            return edge_index_pe, edge_attr_pe

        # Use first PE type (could combine multiple in future)
        pe_type = pe_types[0].lower()

        if pe_type in ["spd", "shortest_path"]:
            pe_computer = SPDBuckets(
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
                use_one_hot=True,
                cache=False,  # We handle caching at transform level
            )
        elif pe_type == "diffusion":
            pe_computer = DiffusionPE(
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
                scales=None,  # Will use default log-spaced
                k_eigenvectors=50,
                cache=False,
            )
        elif pe_type in ["resistance", "effective_resistance"]:
            pe_computer = EffectiveResistancePE(
                num_buckets=1,  # Resistance is scalar
                max_distance=self.max_distance,
                k_eigenvectors=50,
                use_sparse=True,
                cache=False,
            )
            # Need to expand to num_buckets
        elif pe_type == "bfs":
            pe_computer = BFSDistance(
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
                use_one_hot=True,
                cache=False,
            )
        else:
            # Unknown type, use edge-based fallback
            num_edges = data.edge_index.size(1)
            edge_index_pe = data.edge_index
            edge_attr_pe = torch.zeros(num_edges, self.num_buckets, dtype=torch.float32)
            return edge_index_pe, edge_attr_pe

        try:
            edge_index_pe, edge_attr_pe = pe_computer.compute(data)

            # Handle effective resistance (scalar -> expand to num_buckets)
            if pe_type in ["resistance", "effective_resistance"]:
                if edge_attr_pe.shape[1] == 1 and self.num_buckets > 1:
                    # Repeat or pad
                    edge_attr_pe = edge_attr_pe.repeat(1, self.num_buckets)

            # Ensure correct dimension
            if edge_attr_pe.shape[1] != self.num_buckets:
                if edge_attr_pe.shape[1] < self.num_buckets:
                    padding = torch.zeros(
                        edge_attr_pe.shape[0],
                        self.num_buckets - edge_attr_pe.shape[1],
                        device=edge_attr_pe.device,
                    )
                    edge_attr_pe = torch.cat([edge_attr_pe, padding], dim=1)
                else:
                    edge_attr_pe = edge_attr_pe[:, : self.num_buckets]

            return edge_index_pe, edge_attr_pe

        except Exception as e:
            # Fallback: use edge-based zeros
            num_edges = data.edge_index.size(1)
            edge_index_pe = data.edge_index
            edge_attr_pe = torch.zeros(num_edges, self.num_buckets, dtype=torch.float32)
            return edge_index_pe, edge_attr_pe


class NormalizeTargets(BaseTransform):
    """Transform to normalize target values."""

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        """
        Initialize normalization transform.

        Args:
            mean: Mean for normalization (computed from data if None)
            std: Std for normalization (computed from data if None)
        """
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        """
        Normalize target values.

        Args:
            data: PyG Data object with y attribute

        Returns:
            Data object with normalized y
        """
        if hasattr(data, "y") and data.y is not None:
            if self.mean is not None and self.std is not None:
                data.y = (data.y - self.mean) / (self.std + 1e-8)
            # If mean/std not provided, store original y
        return data


class CompositeTransform(BaseTransform):
    """Composite transform applying multiple transforms in sequence."""

    def __init__(
        self,
        pe_config: Optional[Dict[str, Any]] = None,
        normalize_targets: bool = False,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
    ):
        """
        Initialize composite transform.

        Args:
            pe_config: PE configuration dict with 'node' and 'relative' keys
            normalize_targets: Whether to normalize targets
            target_mean: Mean for target normalization
            target_std: Std for target normalization
        """
        self.transforms = []

        pe_config = pe_config or {}

        # Add node PE transform
        node_pe_config = pe_config.get("node", {})
        if node_pe_config.get("enabled", True):
            self.transforms.append(ApplyNodePE(node_pe_config))

        # Add relative PE transform
        relative_pe_config = pe_config.get("relative", {})
        if relative_pe_config.get("enabled", True):
            self.transforms.append(ApplyRelativePE(relative_pe_config))

        # Add target normalization if requested
        if normalize_targets:
            self.transforms.append(NormalizeTargets(target_mean, target_std))

    def __call__(self, data: Data) -> Data:
        """
        Apply all transforms in sequence.

        Args:
            data: PyG Data object

        Returns:
            Transformed Data object
        """
        for transform in self.transforms:
            data = transform(data)
        return data


class CastDataTypes(BaseTransform):
    """Transform to cast data types to specified types."""

    def __init__(
        self,
        node_feat_dtype: torch.dtype = torch.float32,
        edge_attr_dtype: Optional[torch.dtype] = None,
        target_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize dtype casting transform.

        Args:
            node_feat_dtype: Dtype for node features
            edge_attr_dtype: Dtype for edge attributes
            target_dtype: Dtype for targets
        """
        self.node_feat_dtype = node_feat_dtype
        self.edge_attr_dtype = edge_attr_dtype
        self.target_dtype = target_dtype

    def __call__(self, data: Data) -> Data:
        """
        Cast data types.

        Args:
            data: PyG Data object

        Returns:
            Data object with casted types
        """
        if hasattr(data, "x") and data.x is not None:
            data.x = data.x.to(self.node_feat_dtype)

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            if self.edge_attr_dtype is not None:
                data.edge_attr = data.edge_attr.to(self.edge_attr_dtype)

        if hasattr(data, "y") and data.y is not None:
            if self.target_dtype is not None:
                data.y = data.y.to(self.target_dtype)

        return data

