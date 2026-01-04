# Long-Range Dependency Evaluation Framework

Framework for evaluating the ability of Graph Transformers to model long-range dependencies in graphs.

## Structure

```
src/evaluation/
├── __init__.py           # Public API
├── distance_metrics.py   # Graph distance computation
├── stratification.py     # Distance-based stratification
├── long_range.py         # Long-range evaluation metrics
├── probes.py             # Synthetic probing tasks
├── utils.py              # Utilities and visualization
└── README.md             # Documentation
```

## Key Concepts

### Distance-based Evaluation

Long-range evaluation is based on measuring prediction quality as a function of graph distance between informative nodes and target nodes.

### Key Metrics

- **Metrics per bucket**: Accuracy/MAE for each distance range
- **Relative performance drop**: Quality degradation relative to short-range
- **AUC (distance-performance)**: Area under the quality-distance curve
- **Effective Receptive Field (ERF)**: Maximum distance at which the model works

## Quick Start

### Basic Usage

```python
from src.evaluation import (
    LongRangeEvaluator,
    add_distance_info_to_data,
    print_evaluation_summary,
)

# Add distance information to data
for data in dataset:
    add_distance_info_to_data(data, max_distance=20)

# Create evaluator
evaluator = LongRangeEvaluator(
    max_distance=20,
    bucket_size=2,
    task_type="regression",
)

# Evaluation
for batch in loader:
    pred = model(batch)
    distances = get_distances(batch)  # Distances for stratification
    evaluator.update(pred, batch.y, distances)

# Results
results = evaluator.compute()
print_evaluation_summary(results)
```

### Running Script

```bash
python scripts/evaluate_long_range.py \
    --dataset peptides_func \
    --checkpoint path/to/model.pt \
    --max_distance 20 \
    --save_plots
```

## Distance Computation

### Shortest-Path Distances (BFS)

```python
from src.evaluation import compute_shortest_path_distances

# Full distance matrix [N x N]
distances = compute_shortest_path_distances(
    edge_index, num_nodes, max_distance=20
)

# Sparse format (only pairs within max_distance)
pair_indices, pair_distances = compute_shortest_path_distances_sparse(
    edge_index, num_nodes, max_distance=10
)
```

### Landmark-based Approximation

For large graphs, use landmark-based approximation:

```python
from src.evaluation import compute_landmark_distances

landmark_indices, landmark_distances = compute_landmark_distances(
    edge_index, num_nodes,
    num_landmarks=10,
    selection="degree",  # or "random"
)
```

### Adding Distances to Data

```python
from src.evaluation import add_distance_info_to_data

data = add_distance_info_to_data(
    data,
    max_distance=20,
    sparse=True,           # Sparse storage for large graphs
    use_landmarks=False,   # Landmark approximation
)
```

## Stratification

### Creating Buckets

```python
from src.evaluation import create_distance_buckets, stratify_by_distance

# Create buckets: [(0,1), (2,3), (4,5), ...]
buckets = create_distance_buckets(max_distance=10, bucket_size=2)

# Stratify predictions
stratified = stratify_by_distance(
    predictions, targets, distances, buckets
)
```

### DistanceStratifier (for accumulating batches)

```python
from src.evaluation import DistanceStratifier

stratifier = DistanceStratifier(max_distance=20, bucket_size=2)

for batch in loader:
    stratifier.update(predictions, targets, distances)

aggregated = stratifier.compute()
```

## Long-Range Metrics

### Metrics per Bucket

```python
from src.evaluation import compute_metrics_per_bucket

metrics = compute_metrics_per_bucket(
    stratified,
    task_type="regression",  # or "classification", "binary"
)
# Returns: {(0,1): {"mae": 0.5, "mse": 0.3, ...}, ...}
```

### Relative Performance Drop

```python
from src.evaluation import compute_relative_performance_drop

drops = compute_relative_performance_drop(
    metrics_per_bucket,
    higher_is_better=True,  # False for MAE
)
# Returns: {(0,1): 0.0, (2,3): 0.15, ...}
```

### Area Under Distance Curve

```python
from src.evaluation import compute_area_under_distance_curve

auc = compute_area_under_distance_curve(
    metrics_per_bucket,
    max_distance=20,
    normalize=True,
)
```

### Effective Receptive Field

```python
from src.evaluation import find_effective_receptive_field

erf = find_effective_receptive_field(
    metrics_per_bucket,
    threshold=0.5,  # 50% of baseline
)
```

## Probing Tasks

### Path Parity Probe

Tests the ability to compute XOR along a path:

```python
from src.evaluation import PathParityProbe

probe = PathParityProbe(path_length=10)
data = probe.generate_path_graph(seed=42)
batch = probe.generate_batch(batch_size=32)
```

### Node Counting Probe

Tests receptive field:

```python
from src.evaluation import NodeCountingProbe

probe = NodeCountingProbe(max_hops=5)
counts, distances = probe.generate_task(data)
# counts[i, k] = number of nodes in k-hop neighborhood of node i
```

### Synthetic Long-Range Task

Task requiring information from a given distance:

```python
from src.evaluation import SyntheticLongRangeTask

task = SyntheticLongRangeTask(
    graph_size=100,
    signal_distance=10,
    noise_level=0.1,
)
data = task.generate(seed=42)
```

## Visualization

```python
from src.evaluation import (
    plot_distance_performance,
    plot_layer_wise_analysis,
    plot_pe_comparison,
)

# Performance vs Distance
plot_distance_performance(
    metrics_per_bucket,
    title="Model Performance vs Distance",
    output_path="distance_performance.png",
)

# Layer-wise analysis
plot_layer_wise_analysis(
    layer_results,
    metric_name="auc",
    output_path="layer_analysis.png",
)
```

## Layer-wise Analysis

For analyzing how information propagates through layers:

```python
from src.evaluation import evaluate_layer_wise_long_range

# Model must support return_intermediate=True
results = evaluate_layer_wise_long_range(
    hidden_states,     # List[Tensor] per layer
    targets,
    distances,
    linear_probes,     # Optional: probes for each layer
    max_distance=20,
)

# results[layer_idx] = {"auc": ..., "erf": ..., ...}
```

## Comparing PE Configurations

```python
from src.evaluation import compare_pe_configurations

results_dict = {
    "LapPE": evaluator_lap.compute(),
    "RWSE": evaluator_rwse.compute(),
    "No PE": evaluator_none.compute(),
}

comparison = compare_pe_configurations(results_dict, metric_name="auc")
print(f"Best config: {comparison['best']}")
print(f"Ranking: {comparison['ranking']}")
```

## Output Format

Results are saved in JSON:

```json
{
  "metadata": {
    "dataset": "peptides_func",
    "model_config_hash": "a1b2c3d4",
    "max_distance": 20
  },
  "metrics_per_bucket": {
    "[0, 1]": {"mae": 0.12, "count": 1000},
    "[2, 3]": {"mae": 0.15, "count": 800}
  },
  "relative_drops": {
    "[0, 1]": 0.0,
    "[2, 3]": 0.25
  },
  "auc": 0.73,
  "effective_receptive_field": 8
}
```

## Determinism

All computations are deterministic with a fixed seed:

```python
from src.training import set_seed

set_seed(42)
# All operations will be reproducible
```

## Constraints

- Distances are computed once and cached in Data objects
- O(N²) storage is avoided for large graphs (sparse format)
- Gradients are disabled during evaluation
