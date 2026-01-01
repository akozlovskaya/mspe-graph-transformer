# Synthetic Graph Benchmarks

This module provides synthetic graph datasets for evaluating positional encodings (PEs) under controlled conditions.

## Graph Generators

The following graph types are supported:

- **Erdős–Rényi** (`erdos_renyi`): Random graphs with edge probability p
- **Watts–Strogatz** (`watts_strogatz`): Small-world graphs
- **Barabási–Albert** (`barabasi_albert`): Scale-free networks
- **Stochastic Block Model** (`sbm`): Community-structured graphs
- **Random Geometric** (`random_geometric`): Spatial graphs based on distance
- **Regular** (`random_regular`): Regular graphs
- **Grid** (`grid_2d`, `grid_3d`): Grid graphs
- **Ring** (`ring`): Cycle graphs
- **Tree** (`tree`): Balanced trees

## Benchmark Tasks

### Task A: Pairwise Distance Classification
Classify whether the shortest-path distance between two nodes is >= threshold.

**Usage:**
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="pairwise_distance_classification",
    task_params={"distance_threshold": 3, "num_pairs": 100},
    graph_params={"n": 50, "p": 0.3},
)
```

### Task B: Distance Regression
Predict the exact shortest-path distance between node pairs.

**Usage:**
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="distance_regression",
    task_params={"num_pairs": 100},
    graph_params={"n": 50, "p": 0.3},
)
```

### Task C: Structural Role Classification
Classify nodes by their structural role (e.g., block ID in SBM).

**Usage:**
```python
dataset = get_dataset(
    name="synthetic_sbm",
    task="structural_role",
    graph_params={"n": 100, "n_blocks": 3, "p_in": 0.3, "p_out": 0.05},
    use_node_features=False,  # Structure-only
)
```

### Task D: Local vs Global Signal
Predict based on local (degree) or global (distance) signals.

**Usage:**
```python
dataset = get_dataset(
    name="synthetic_watts_strogatz",
    task="local_vs_global",
    task_params={"use_local": True, "local_threshold": 3},
    graph_params={"n": 50, "k": 4, "p": 0.3},
)
```

### Task E: Diffusion Source Identification
Identify the source node of a diffusion process.

**Usage:**
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="diffusion_source",
    task_params={"diffusion_steps": 5, "diffusion_rate": 0.5},
    graph_params={"n": 30, "p": 0.3},
)
```

### Task F: PE Capacity Stress Test
Test PE capacity by varying dimensionality.

**Usage:**
```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    task="pe_capacity",
    task_params={"base_task": "pairwise_distance_classification"},
    graph_params={"n": 50, "p": 0.3},
)
```

## Configuration Files

Pre-configured datasets are available in `configs/dataset/synthetic/`:

- `pairwise_distance.yaml` - Task A
- `distance_regression.yaml` - Task B
- `structural_role.yaml` - Task C
- `local_vs_global.yaml` - Task D
- `diffusion_source.yaml` - Task E
- `pe_capacity.yaml` - Task F

## Running Experiments

### Using Hydra configs:

```bash
# Task A with different PEs
python scripts/train.py dataset=synthetic/pairwise_distance pe=mspe

# Task C (structural role)
python scripts/train.py dataset=synthetic/structural_role pe.node.type=lap

# PE capacity test
python scripts/run_sweep.py --sweep configs/experiments/synthetic/pe_capacity.yaml
```

### Cross-Graph Generalization:

Train on one graph type, test on another:

```bash
python scripts/run_experiment.py experiment=synthetic/cross_graph_generalization
```

### Size Extrapolation:

Train on small graphs, test on large:

```bash
python scripts/run_experiment.py experiment=synthetic/size_extrapolation
```

## Reproducibility

All randomness is controlled via the `seed` parameter:

```python
dataset = get_dataset(
    name="synthetic_erdos_renyi",
    seed=42,  # Ensures deterministic generation
    ...
)
```

Same config + seed → identical results.

## Integration

Synthetic datasets integrate seamlessly with the existing pipeline:

- Use same training scripts (`scripts/train.py`)
- Support all PE types (LapPE, RWSE, HKS, SPD, etc.)
- Work with existing evaluation tools
- Compatible with experiment sweeps

## Notes

- For structure-only tasks, set `use_node_features=False`
- Graph-level tasks use `data.y` for targets
- Node-level tasks (e.g., structural role) use `data.y` with shape `[num_nodes]`
- Pairwise tasks store additional metadata in `data.pair_sources`, `data.pair_targets`, `data.pair_labels`

