"""Synthetic graph dataset generators."""

from typing import Optional, Dict, Any, Tuple, List
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx

from .base import BaseGraphDataset, InMemoryGraphDataset
from .transforms import CompositeTransform


class SyntheticDataset(BaseGraphDataset):
    """Synthetic graph dataset with MSPE support."""

    def __init__(
        self,
        root: str,
        graph_type: str,
        num_graphs: int = 1000,
        graph_params: Optional[Dict[str, Any]] = None,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pe_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 1,
        task_type: str = "regression",
    ):
        """
        Initialize synthetic dataset.

        Args:
            root: Root directory (not used for synthetic, but kept for API consistency).
            graph_type: Type of graph to generate:
                - "grid_2d": 2D grid graphs
                - "grid_3d": 3D grid graphs
                - "ring": Ring graphs
                - "tree": Balanced tree graphs
                - "random_regular": Random regular graphs
                - "barabasi_albert": Barabási–Albert graphs
                - "watts_strogatz": Watts–Strogatz small-world graphs
                - "erdos_renyi": Erdős–Rényi random graphs
            num_graphs: Number of graphs to generate.
            graph_params: Parameters for graph generation (depends on graph_type).
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
            num_classes: Number of classes (for classification) or 1 (for regression).
            task_type: "regression" or "classification".
        """
        super().__init__(root, transform, pre_transform, pe_config)
        self.graph_type = graph_type
        self.num_graphs = num_graphs
        self.graph_params = graph_params or {}
        self.task_type = task_type
        self._num_classes = num_classes

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Generate graphs
        self._generate_graphs(transform)

    def _generate_graphs(self, transform: Any):
        """Generate synthetic graphs."""
        graphs = []
        labels = []

        for i in range(self.num_graphs):
            # Generate graph based on type
            g = self._generate_single_graph(i)

            # Convert to PyG Data
            data = from_networkx(g)

            # Add node features if not present
            if not hasattr(data, "x") or data.x is None:
                # Use degree as node features
                degrees = torch.tensor([g.degree(n) for n in g.nodes()], dtype=torch.float32)
                data.x = degrees.unsqueeze(1)

            # Generate target (placeholder - can be customized)
            if self.task_type == "regression":
                # Use graph size as target (example)
                data.y = torch.tensor([[data.num_nodes]], dtype=torch.float32)
            else:
                # Classification: use graph size modulo num_classes
                data.y = torch.tensor([data.num_nodes % self._num_classes], dtype=torch.long)

            # Apply transform
            if transform is not None:
                data = transform(data)

            graphs.append(data)
            labels.append(data.y)

        # Create splits
        train_size = int(0.8 * self.num_graphs)
        val_size = int(0.1 * self.num_graphs)

        train_data = graphs[:train_size]
        val_data = graphs[train_size : train_size + val_size]
        test_data = graphs[train_size + val_size :]

        self.train_dataset = InMemoryGraphDataset(train_data)
        self.val_dataset = InMemoryGraphDataset(val_data)
        self.test_dataset = InMemoryGraphDataset(test_data)

    def _generate_single_graph(self, seed: int) -> nx.Graph:
        """Generate a single graph based on graph_type."""
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        if self.graph_type == "grid_2d":
            m = self.graph_params.get("m", 10)
            n = self.graph_params.get("n", 10)
            return nx.grid_2d_graph(m, n)

        elif self.graph_type == "grid_3d":
            m = self.graph_params.get("m", 5)
            n = self.graph_params.get("n", 5)
            k = self.graph_params.get("k", 5)
            return nx.grid_graph(dim=(m, n, k))

        elif self.graph_type == "ring":
            n = self.graph_params.get("n", 20)
            return nx.cycle_graph(n)

        elif self.graph_type == "tree":
            r = self.graph_params.get("r", 3)  # branching factor
            h = self.graph_params.get("h", 4)  # height
            return nx.balanced_tree(r, h)

        elif self.graph_type == "random_regular":
            n = self.graph_params.get("n", 20)
            d = self.graph_params.get("d", 3)  # degree
            return nx.random_regular_graph(d, n, seed=seed)

        elif self.graph_type == "barabasi_albert":
            n = self.graph_params.get("n", 20)
            m = self.graph_params.get("m", 2)  # edges to attach
            return nx.barabasi_albert_graph(n, m, seed=seed)

        elif self.graph_type == "watts_strogatz":
            n = self.graph_params.get("n", 20)
            k = self.graph_params.get("k", 4)  # each node connected to k neighbors
            p = self.graph_params.get("p", 0.3)  # rewiring probability
            return nx.watts_strogatz_graph(n, k, p, seed=seed)

        elif self.graph_type == "erdos_renyi":
            n = self.graph_params.get("n", 20)
            p = self.graph_params.get("p", 0.3)  # edge probability
            return nx.erdos_renyi_graph(n, p, seed=seed)

        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

    def load(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train, val, and test splits."""
        return self.train_dataset, self.val_dataset, self.test_dataset

    @property
    def num_features(self) -> int:
        """Number of node features."""
        if self._num_features is None:
            sample = self.train_dataset[0]
            if hasattr(sample, "x") and sample.x is not None:
                self._num_features = sample.x.size(1)
            else:
                self._num_features = 1  # Default: degree
        return self._num_features

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self._num_classes

    @property
    def train(self) -> Dataset:
        """Training split."""
        return self.train_dataset

    @property
    def val(self) -> Dataset:
        """Validation split."""
        return self.val_dataset

    @property
    def test(self) -> Dataset:
        """Test split."""
        return self.test_dataset

