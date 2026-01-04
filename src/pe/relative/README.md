# Relative (Pairwise) Positional Encodings

Module for computing relative (pairwise) structural positional encodings for graphs.

## Available PEs

### 1. **SPDBuckets** - Shortest-Path Distance Buckets
Computes shortest distances between all node pairs and discretizes them into buckets.

```python
from src.pe.relative import SPDBuckets

pe = SPDBuckets(
    num_buckets=16,
    max_distance=10,
    use_one_hot=True,
    symmetric=True
)

edge_index_pe, edge_attr_pe = pe(data)
# edge_index_pe: [2, num_pairs] - all pairs (i,j)
# edge_attr_pe: [num_pairs, 16] - one-hot encodings of distances
```

### 2. **BFSDistance** - Truncated BFS Distance
Lightweight version of SPD, stores only pairs within k hops.

```python
from src.pe.relative import BFSDistance

pe = BFSDistance(
    num_buckets=8,
    max_distance=5,  # k
    use_one_hot=True
)

edge_index_pe, edge_attr_pe = pe(data)
# Only pairs with d(i,j) <= max_distance
```

### 3. **DiffusionPE** - Heat Kernel Pairwise Encoding
Uses heat diffusion on the graph to compute pairwise values.

```python
from src.pe.relative import DiffusionPE

pe = DiffusionPE(
    num_buckets=4,
    max_distance=10,  # Not used
    scales=[0.1, 1.0, 5.0, 10.0],  # Diffusion times
    k_eigenvectors=50
)

edge_index_pe, edge_attr_pe = pe(data)
# edge_attr_pe: [num_pairs, 4] - values for each scale
```

### 4. **EffectiveResistancePE** - Effective Resistance
Approximates effective resistance between node pairs.

```python
from src.pe.relative import EffectiveResistancePE

pe = EffectiveResistancePE(
    num_buckets=1,  # Scalar per pair
    max_distance=10,  # Not used
    k_eigenvectors=50,
    use_sparse=True
)

edge_index_pe, edge_attr_pe = pe(data)
# edge_attr_pe: [num_pairs, 1] - resistance values
```

### 5. **LandmarkSPD** - Landmark-based SPD Approximation
Approximates SPD using distances to landmark nodes.

```python
from src.pe.relative import LandmarkSPD

pe = LandmarkSPD(
    num_buckets=8,
    max_distance=5,
    num_landmarks=10,
    landmark_method="random",  # or "degree"
    approximation_method="min_diff"  # or "max_diff", "mean_diff"
)

edge_index_pe, edge_attr_pe = pe(data)
```

## Integration with Attention

Usage for building attention bias:

```python
from src.pe.relative import SPDBuckets, build_attention_bias

# Compute PE
pe = SPDBuckets(num_buckets=16, max_distance=10)
edge_index_pe, edge_attr_pe = pe(data)

# Build attention bias
bias = build_attention_bias(
    edge_index_pe,
    edge_attr_pe,
    num_nodes=data.num_nodes,
    num_heads=8,
    mode="dense",  # or "sparse"
    gating=True
)

# bias shape: [8, num_nodes, num_nodes] for multi-head attention
```

## Data Structure

Each PE returns:

- `edge_index_pe`: Tensor [2, num_pairs] - indices of node pairs (i, j)
- `edge_attr_pe`: Tensor [num_pairs, num_buckets] - PE values for each pair

## Parameters

All PE classes support:

- `num_buckets`: Number of buckets or channels
- `max_distance`: Maximum distance (for SPD/BFS)
- `normalization`: `"graph"`, `"pair"`, or `None`
- `symmetric`: Ensure PE(i,j) == PE(j,i) symmetry
- `cache`: Whether to cache PE in `data`

## Mathematical Formulas

### SPD
Shortest distance between nodes:
```
d(i,j) = shortest_path_length(i, j)
```

### Diffusion
Heat kernel on graph:
```
K_t(i,j) = Σ_k exp(-λ_k * t) * φ_k(i) * φ_k(j)
```

### Effective Resistance
Resistance between nodes:
```
R(i,j) = L^+_{ii} + L^+_{jj} - 2*L^+_{ij}
```
where L^+ is the pseudoinverse of the Laplacian.

### Landmark SPD
Approximation via landmarks:
```
d(i,j) ≈ min_ℓ |d(i,ℓ) - d(j,ℓ)|
```

## Testing

```bash
pytest tests/test_relative_pe.py -v
```

## Usage in Graph Transformers

```python
# In attention layer
def forward(self, x, edge_index_pe, edge_attr_pe):
    # Build bias
    bias = build_attention_bias(
        edge_index_pe, edge_attr_pe,
        num_nodes=x.size(0),
        num_heads=self.num_heads
    )
    
    # Apply in attention
    attn = self.attention(x, bias=bias)
    return attn
```

## Sparse vs Dense

- **Dense**: Stores all pairs (O(N²)) - for small graphs
- **Sparse**: Stores only relevant pairs - for large graphs

Use sparse versions (`SPDBucketsSparse`, `LandmarkSPDSparse`) for scalability.
