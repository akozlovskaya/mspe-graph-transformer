"""Base class for node-wise positional encodings."""

from abc import ABC, abstractmethod
from typing import Optional, List, Union
import torch
from torch_geometric.data import Data


class BaseNodePE(ABC):
    """Abstract base class for node-wise positional encodings."""

    def __init__(
        self,
        dim: int,
        scales: Optional[List[float]] = None,
        normalization: str = "graph",
        sign_invariant: bool = True,
        cache: bool = True,
    ):
        """
        Initialize base node PE.

        Args:
            dim: Output embedding dimension per node.
            scales: List of scales (e.g., RW steps or diffusion times).
                   If None, will use default scales.
            normalization: Normalization mode: 'graph', 'node', or None.
            sign_invariant: Whether to apply sign-invariant processing
                          (only for spectral PE like LapPE, HKS).
            cache: Whether to cache PE in data.node_pe.
        """
        self.dim = dim
        self.scales = scales or [1.0]
        self.normalization = normalization
        self.sign_invariant = sign_invariant
        self.cache = cache

    @abstractmethod
    def compute(self, data: Data) -> torch.Tensor:
        """
        Compute node positional encodings.

        Args:
            data: PyG Data object with edge_index.

        Returns:
            Tensor of shape [num_nodes, dim] with positional encodings.
        """
        raise NotImplementedError

    def __call__(self, data: Data) -> torch.Tensor:
        """
        Compute PE with caching support.

        Args:
            data: PyG Data object.

        Returns:
            Tensor of shape [num_nodes, dim].
        """
        # Check cache
        if self.cache and hasattr(data, "node_pe") and data.node_pe is not None:
            # Verify shape matches
            if data.node_pe.shape[1] == self.dim:
                return data.node_pe

        # Compute PE
        node_pe = self.compute(data)

        # Apply normalization
        node_pe = self._normalize(node_pe)

        # Project to target dimension if needed
        if node_pe.shape[1] != self.dim:
            node_pe = self._project_to_dim(node_pe)

        # Cache if requested
        if self.cache:
            data.node_pe = node_pe

        return node_pe

    def _normalize(self, pe: torch.Tensor) -> torch.Tensor:
        """
        Normalize positional encodings.

        Args:
            pe: Tensor of shape [num_nodes, pe_dim].

        Returns:
            Normalized tensor.
        """
        if self.normalization is None:
            return pe

        if self.normalization == "graph":
            # Zero mean, unit std per graph
            mean = pe.mean(dim=0, keepdim=True)
            std = pe.std(dim=0, keepdim=True) + 1e-8
            return (pe - mean) / std

        elif self.normalization == "node":
            # Normalize per node (across features)
            mean = pe.mean(dim=1, keepdim=True)
            std = pe.std(dim=1, keepdim=True) + 1e-8
            return (pe - mean) / std

        else:
            raise ValueError(f"Unknown normalization mode: {self.normalization}")

    def _project_to_dim(self, pe: torch.Tensor) -> torch.Tensor:
        """
        Project PE to target dimension.

        Args:
            pe: Tensor of shape [num_nodes, current_dim].

        Returns:
            Tensor of shape [num_nodes, self.dim].
        """
        current_dim = pe.shape[1]
        if current_dim == self.dim:
            return pe
        elif current_dim < self.dim:
            # Pad with zeros
            padding = torch.zeros(pe.shape[0], self.dim - current_dim, device=pe.device)
            return torch.cat([pe, padding], dim=1)
        else:
            # Truncate or use linear projection
            # Simple truncation for now
            return pe[:, : self.dim]

    def _ensure_connected(self, data: Data) -> Data:
        """
        Ensure graph is connected (add self-loops if needed for isolated nodes).

        Args:
            data: PyG Data object.

        Returns:
            Data object (possibly modified).
        """
        # This is a placeholder - actual implementation depends on use case
        # For now, we assume graphs are connected or handle in specific PE classes
        return data

