# Dataset Module

Module for loading and processing graph datasets with support for Multi-Scale Structural Positional Encodings (MSPE).

## Key Features

- **Unified API** for all datasets
- **Automatic data loading and preparation**
- **Built-in PE support** via transforms
- **Compatibility with PyTorch Geometric**
- **Support for multiple datasets**: ZINC, QM9, LRGB, OGB, PCQM, synthetic graphs

## Supported Datasets

### Molecular
- **ZINC**: Molecular dataset for regression
- **QM9**: Quantum-mechanical molecular properties

### LRGB
- **Peptides-func**: Functional properties of peptides
- **Peptides-struct**: Structural properties of peptides
- **PascalVOC-SP**: Superpixel graphs from PascalVOC
- **CIFAR10-SP**: Superpixel graphs from CIFAR10

### OGB
- **ogbg-molhiv**: Binary classification of HIV activity
- **ogbg-molpcba**: Multi-task classification on PCBA
- **PCQM4M**: Quantum molecular properties (4M graphs)
- **PCQM-Contact**: Edge prediction task

### Synthetic
- **synthetic_grid_2d**: 2D grids
- **synthetic_grid_3d**: 3D grids
- **synthetic_ring**: Ring graphs
- **synthetic_tree**: Balanced trees
- **synthetic_random_regular**: Random regular graphs
- **synthetic_barabasi_albert**: Barabási–Albert graphs
- **synthetic_watts_strogatz**: Watts–Strogatz graphs
- **synthetic_erdos_renyi**: Erdős–Rényi graphs

## Quick Start

### Basic Usage

```python
from src.dataset import get_dataset
from torch_geometric.data import DataLoader

# Load dataset with PE
dataset = get_dataset(
    name="zinc",
    root="./data",
    pe_config={
        "node": {
            "enabled": True,
            "types": ["rwse"],
            "dim": 32,
            "scales": [1, 2, 4, 8]
        },
        "relative": {
            "enabled": True,
            "types": ["spd"],
            "max_distance": 10,
            "num_buckets": 16
        }
    }
)

# Create DataLoader
train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)

# Use in training loop
for batch in train_loader:
    print(f"Batch size: {batch.batch.max().item() + 1}")
    print(f"Node features: {batch.x.shape}")
    print(f"Node PE: {batch.node_pe.shape}")
    print(f"Edge PE: {batch.edge_pe.shape}")
    print(f"Targets: {batch.y.shape}")
```

### Loading LRGB Dataset

```python
dataset = get_dataset(
    name="peptides_func",
    root="./data",
    pe_config={
        "node": {"enabled": True, "types": ["lap_pe", "rwse"], "dim": 32},
        "relative": {"enabled": True, "types": ["spd", "diffusion"], "num_buckets": 32}
    }
)
```

### Synthetic Graphs

```python
# Generate ring graphs
dataset = get_dataset(
    name="synthetic_ring",
    root="./data",
    num_graphs=1000,
    graph_params={"n": 20},  # 20 nodes per graph
    pe_config={"node": {"enabled": True}, "relative": {"enabled": True}}
)

# Generate 2D grids
dataset = get_dataset(
    name="synthetic_grid_2d",
    num_graphs=500,
    graph_params={"m": 10, "n": 10},  # 10x10 grid
    pe_config={"node": {"enabled": True}, "relative": {"enabled": True}}
)
```

## PE Configuration

### Node-wise PE

```python
node_pe_config = {
    "enabled": True,
    "types": ["lap_pe", "rwse", "hks"],  # PE types
    "dim": 32,                            # PE dimension
    "scales": [1, 2, 4, 8]                # Scales for multi-scale
}
```

### Relative PE

```python
relative_pe_config = {
    "enabled": True,
    "types": ["spd", "diffusion", "effective_resistance"],
    "max_distance": 10,                   # Maximum distance
    "num_buckets": 16                     # Number of buckets
}
```

## Data Structure

Each graph in the dataset has the following structure:

```python
Data(
    x=torch.Tensor,           # Node features [num_nodes, num_features]
    edge_index=torch.Tensor,   # Edge indices [2, num_edges]
    edge_attr=torch.Tensor,   # Edge attributes [num_edges, edge_dim] (optional)
    node_pe=torch.Tensor,      # Node positional encodings [num_nodes, pe_dim]
    edge_pe=torch.Tensor,     # Relative positional encodings [num_edges, num_buckets]
    y=torch.Tensor,           # Target [num_targets] or [1]
    pos=torch.Tensor,         # Node positions (if available) [num_nodes, 2/3]
)
```

## Utilities

### Computing Dataset Statistics

```python
from src.dataset.utils import compute_dataset_stats

stats = compute_dataset_stats(dataset.train)
print(f"Average nodes: {stats['avg_num_nodes']}")
print(f"Average edges: {stats['avg_num_edges']}")
print(f"Target mean: {stats.get('target_mean', 'N/A')}")
```

### Creating Random Splits

```python
from src.dataset.utils import create_random_split

train_idx, val_idx, test_idx = create_random_split(
    dataset, train_ratio=0.8, val_ratio=0.1, seed=42
)
```

### Normalizing Targets

```python
from src.dataset.utils import normalize_targets

mean, std = normalize_targets(dataset.train)
```

## Testing

Run tests:

```bash
pytest tests/test_dataset_loading.py -v
```

## API Reference

### `get_dataset(name, root, pe_config, **kwargs)`

Factory function for creating datasets.

**Parameters:**
- `name` (str): Dataset name
- `root` (str): Root directory for data storage
- `pe_config` (dict): Positional encoding configuration
- `**kwargs`: Additional parameters for specific datasets

**Returns:**
- `BaseGraphDataset`: Dataset instance with `train`, `val`, `test` attributes

### `BaseGraphDataset`

Base class for all datasets.

**Methods:**
- `load()`: Loads train/val/test splits
- `get_splits(splits="official")`: Returns tuple with splits

**Properties:**
- `num_features`: Number of node features
- `num_classes`: Number of classes (1 for regression)

### Transforms

- `ApplyNodePE`: Applies node-wise PE
- `ApplyRelativePE`: Applies relative PE
- `CompositeTransform`: Composite transform for applying multiple transforms
- `NormalizeTargets`: Normalizes targets
- `CastDataTypes`: Casts data types

## Usage Examples

### Complete Training Example

```python
from src.dataset import get_dataset
from torch_geometric.data import DataLoader
import torch

# Load dataset
dataset = get_dataset(
    name="peptides_func",
    root="./data",
    pe_config={
        "node": {"enabled": True, "types": ["rwse"], "dim": 32},
        "relative": {"enabled": True, "types": ["spd"], "num_buckets": 16}
    }
)

# Create DataLoaders
train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.val, batch_size=32, shuffle=False)

# Training
for epoch in range(10):
    for batch in train_loader:
        # batch.x - node features
        # batch.node_pe - node positional encodings
        # batch.edge_pe - relative positional encodings
        # batch.y - targets
        # ... your training code ...
        pass
```

## Additional Information

- All datasets automatically apply PE via transforms
- PE is computed once during loading and cached in memory
- Support for both classification and regression
- Compatible with PyTorch Geometric DataLoader
- Graceful fallback if PE is disabled (zero PE)

## Contributing

When adding new datasets:
1. Create a class inheriting from `BaseGraphDataset`
2. Implement methods `load()`, `num_features`, `num_classes`
3. Add support in `factory.py`
4. Add tests in `test_dataset_loading.py`
