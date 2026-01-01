"""Synthetic graph dataset generators with benchmark tasks for PE evaluation."""

from typing import Optional, Dict, Any, Tuple, List
from collections import deque
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx

from .base import BaseGraphDataset, InMemoryGraphDataset
from .transforms import CompositeTransform
from src.evaluation.distance_metrics import compute_shortest_path_distances


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
        task: Optional[str] = None,
        task_params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        use_node_features: bool = True,
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
                - "sbm": Stochastic Block Model
                - "random_geometric": Random geometric graphs
            num_graphs: Number of graphs to generate.
            graph_params: Parameters for graph generation (depends on graph_type).
            transform: Transform to apply to each graph.
            pre_transform: Pre-transform to apply before saving.
            pe_config: Configuration for positional encodings.
            num_classes: Number of classes (for classification) or 1 (for regression).
            task_type: "regression" or "classification".
            task: Task type for synthetic benchmarks:
                - "pairwise_distance_classification": Task A - classify if distance >= threshold
                - "distance_regression": Task B - predict exact shortest-path distance
                - "structural_role": Task C - classify structural role/block ID
                - "local_vs_global": Task D - predict based on local vs global signal
                - "diffusion_source": Task E - identify diffusion source node
                - "pe_capacity": Task F - stress test PE capacity
            task_params: Parameters for task generation.
            seed: Random seed for reproducibility.
            use_node_features: Whether to use node features (False = structure-only).
        """
        super().__init__(root, transform, pre_transform, pe_config)
        self.graph_type = graph_type
        self.num_graphs = num_graphs
        self.graph_params = graph_params or {}
        self.task_type = task_type
        self.task = task
        self.task_params = task_params or {}
        self._num_classes = num_classes
        self.seed = seed
        self.use_node_features = use_node_features

        # Create composite transform with PE
        if transform is None:
            transform = CompositeTransform(pe_config=pe_config)

        # Generate graphs
        self._generate_graphs(transform)

    def _generate_graphs(self, transform: Any):
        """Generate synthetic graphs with task-specific labels."""
        graphs = []
        labels = []

        # Set global seed for reproducibility
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        for i in range(self.num_graphs):
            # Generate graph based on type (use graph index as seed offset)
            graph_seed = self.seed + i
            g = self._generate_single_graph(graph_seed)

            # Convert to PyG Data
            data = from_networkx(g)

            # Add node features if enabled
            if self.use_node_features:
                if not hasattr(data, "x") or data.x is None:
                    # Use degree as node features
                    degrees = torch.tensor([g.degree(n) for n in g.nodes()], dtype=torch.float32)
                    data.x = degrees.unsqueeze(1)
            else:
                # Structure-only: use constant features
                data.x = torch.ones(data.num_nodes, 1, dtype=torch.float32)

            # Generate task-specific targets
            data = self._generate_task_targets(data, g, graph_seed)

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

        elif self.graph_type == "sbm":
            # Stochastic Block Model
            n = self.graph_params.get("n", 50)
            n_blocks = self.graph_params.get("n_blocks", 3)
            block_sizes = self.graph_params.get("block_sizes", None)
            p_in = self.graph_params.get("p_in", 0.3)  # within-block probability
            p_out = self.graph_params.get("p_out", 0.05)  # between-block probability
            
            if block_sizes is None:
                # Equal-sized blocks
                block_sizes = [n // n_blocks] * n_blocks
                # Add remainder to first block
                block_sizes[0] += n - sum(block_sizes)
            
            # Create block membership
            block_membership = []
            for i, size in enumerate(block_sizes):
                block_membership.extend([i] * size)
            
            # Create probability matrix
            p_matrix = np.full((n_blocks, n_blocks), p_out)
            np.fill_diagonal(p_matrix, p_in)
            
            # Generate edges
            G = nx.Graph()
            G.add_nodes_from(range(n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    block_i = block_membership[i]
                    block_j = block_membership[j]
                    if rng.random() < p_matrix[block_i, block_j]:
                        G.add_edge(i, j)
            
            # Store block membership as node attribute
            nx.set_node_attributes(G, {i: block_membership[i] for i in range(n)}, "block")
            return G

        elif self.graph_type == "random_geometric":
            n = self.graph_params.get("n", 50)
            radius = self.graph_params.get("radius", 0.2)
            dim = self.graph_params.get("dim", 2)
            
            # Generate random positions
            pos = rng.random((n, dim))
            
            G = nx.Graph()
            G.add_nodes_from(range(n))
            
            # Add edges based on distance
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < radius:
                        G.add_edge(i, j)
            
            # Store positions as node attributes
            nx.set_node_attributes(G, {i: pos[i] for i in range(n)}, "pos")
            return G

        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

    def _generate_task_targets(self, data: Data, g: nx.Graph, seed: int) -> Data:
        """
        Generate task-specific targets based on task type.

        Args:
            data: PyG Data object
            g: NetworkX graph
            seed: Random seed for task generation

        Returns:
            Data object with task-specific targets and metadata
        """
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        if self.task is None:
            # Default task: graph-level prediction
            if self.task_type == "regression":
                data.y = torch.tensor([[data.num_nodes]], dtype=torch.float32)
            else:
                data.y = torch.tensor([data.num_nodes % self._num_classes], dtype=torch.long)
            return data

        task = self.task.lower()

        if task == "pairwise_distance_classification":
            # Task A: Pairwise Distance Classification
            # Sample node pairs and classify if distance >= threshold
            threshold = self.task_params.get("distance_threshold", 3)
            num_pairs = self.task_params.get("num_pairs", min(100, data.num_nodes * (data.num_nodes - 1) // 2))
            
            # Compute shortest path distances
            distances = compute_shortest_path_distances(
                data.edge_index, data.num_nodes, max_distance=20
            )
            
            # Sample pairs
            all_pairs = []
            for i in range(data.num_nodes):
                for j in range(i + 1, data.num_nodes):
                    if distances[i, j] >= 0:  # Reachable
                        all_pairs.append((i, j, distances[i, j].item()))
            
            if len(all_pairs) > num_pairs:
                selected_pairs = rng.choice(len(all_pairs), num_pairs, replace=False)
                pairs = [all_pairs[i] for i in selected_pairs]
            else:
                pairs = all_pairs
            
            # Create targets: 1 if distance >= threshold, 0 otherwise
            sources = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            targets = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            labels = torch.tensor([1 if p[2] >= threshold else 0 for p in pairs], dtype=torch.long)
            
            # Store as graph-level: use majority vote or average
            data.y = labels.float().mean().unsqueeze(0)  # For binary classification
            data.pair_sources = sources
            data.pair_targets = targets
            data.pair_labels = labels
            self._num_classes = 2
            self.task_type = "classification"

        elif task == "distance_regression":
            # Task B: Distance Regression
            num_pairs = self.task_params.get("num_pairs", min(100, data.num_nodes * (data.num_nodes - 1) // 2))
            
            # Compute shortest path distances
            distances = compute_shortest_path_distances(
                data.edge_index, data.num_nodes, max_distance=20
            )
            
            # Sample pairs
            all_pairs = []
            for i in range(data.num_nodes):
                for j in range(i + 1, data.num_nodes):
                    if distances[i, j] >= 0:  # Reachable
                        all_pairs.append((i, j, distances[i, j].item()))
            
            if len(all_pairs) > num_pairs:
                selected_pairs = rng.choice(len(all_pairs), num_pairs, replace=False)
                pairs = [all_pairs[i] for i in selected_pairs]
            else:
                pairs = all_pairs
            
            # Create targets: exact distances
            sources = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            targets = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            labels = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
            
            # Store as graph-level: use mean distance
            data.y = labels.mean().unsqueeze(0)
            data.pair_sources = sources
            data.pair_targets = targets
            data.pair_labels = labels
            self.task_type = "regression"

        elif task == "structural_role":
            # Task C: Structural Role Classification
            # For SBM: use block ID; for regular graphs: use structural features
            if self.graph_type == "sbm":
                # Use block membership from SBM
                block_ids = [g.nodes[i].get("block", 0) for i in range(data.num_nodes)]
                data.y = torch.tensor(block_ids, dtype=torch.long)  # Node-level
                self._num_classes = len(set(block_ids))
            elif self.graph_type == "random_regular":
                # For regular graphs, use distance-based roles
                # Compute distance to center (node with highest closeness centrality)
                try:
                    centrality = nx.closeness_centrality(g)
                    center = max(centrality, key=centrality.get)
                    distances_to_center = compute_shortest_path_distances(
                        data.edge_index, data.num_nodes, max_distance=20
                    )[center]
                    
                    # Bin distances into roles
                    max_dist = distances_to_center.max().item()
                    num_roles = min(self._num_classes, int(max_dist) + 1)
                    roles = (distances_to_center.clamp(0, num_roles - 1)).long()
                    data.y = roles
                    self._num_classes = num_roles
                except:
                    # Fallback: use degree-based roles
                    degrees = torch.tensor([g.degree(i) for i in range(data.num_nodes)], dtype=torch.long)
                    data.y = degrees
                    self._num_classes = degrees.max().item() + 1
            else:
                # Default: use degree-based roles
                degrees = torch.tensor([g.degree(i) for i in range(data.num_nodes)], dtype=torch.long)
                data.y = degrees
                self._num_classes = degrees.max().item() + 1
            
            self.task_type = "classification"

        elif task == "local_vs_global":
            # Task D: Local vs Global Signal Prediction
            # Label depends on either local degree or global distance
            use_local = self.task_params.get("use_local", True)
            local_threshold = self.task_params.get("local_threshold", 3)
            global_threshold = self.task_params.get("global_threshold", 5)
            
            if use_local:
                # Local signal: based on degree
                degrees = torch.tensor([g.degree(i) for i in range(data.num_nodes)], dtype=torch.float32)
                labels = (degrees >= local_threshold).long()
            else:
                # Global signal: based on distance to a landmark node
                landmark = rng.integers(0, data.num_nodes)
                distances = compute_shortest_path_distances(
                    data.edge_index, data.num_nodes, max_distance=20
                )[landmark]
                labels = (distances >= global_threshold).long()
            
            data.y = labels.float().mean().unsqueeze(0)  # Graph-level
            data.node_labels = labels  # Also store node-level
            self._num_classes = 2
            self.task_type = "classification"

        elif task == "diffusion_source":
            # Task E: Diffusion Source Identification
            # Simulate diffusion from a random source and predict source node
            source = rng.integers(0, data.num_nodes)
            diffusion_steps = self.task_params.get("diffusion_steps", 5)
            diffusion_rate = self.task_params.get("diffusion_rate", 0.5)
            
            # Simulate diffusion
            infected = {source: 0}  # node -> time step
            queue = deque([(source, 0)])
            
            while queue:
                node, step = queue.popleft()
                if step >= diffusion_steps:
                    continue
                
                neighbors = list(g.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in infected:
                        if rng.random() < diffusion_rate:
                            infected[neighbor] = step + 1
                            queue.append((neighbor, step + 1))
            
            # Create node states: 1 if infected, 0 otherwise
            node_states = torch.zeros(data.num_nodes, dtype=torch.float32)
            for node in infected:
                node_states[node] = 1.0
            
            # Target: source node ID (as one-hot or index)
            data.x = node_states.unsqueeze(1)  # Override node features with diffusion states
            data.y = torch.tensor([source], dtype=torch.long)  # Graph-level: source node
            data.source_node = source
            self._num_classes = data.num_nodes
            self.task_type = "classification"

        elif task == "pe_capacity":
            # Task F: PE Capacity Stress Test
            # Same as Task A or C, but used for analysis
            base_task = self.task_params.get("base_task", "pairwise_distance_classification")
            self.task_params["base_task"] = base_task
            # Recursively call with base task
            data = self._generate_task_targets(data, g, seed)
            data.task_f = True  # Mark as capacity test
            return data

        else:
            # Unknown task, use default
            if self.task_type == "regression":
                data.y = torch.tensor([[data.num_nodes]], dtype=torch.float32)
            else:
                data.y = torch.tensor([data.num_nodes % self._num_classes], dtype=torch.long)

        return data

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

