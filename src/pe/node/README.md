# Node-wise Positional Encodings

Module for computing node-wise structural positional encodings for graphs.

## Available PEs

### 1. **LapPE** - Laplacian Positional Encoding
Uses eigenvectors of the normalized graph Laplacian.

```python
from src.pe.node import LapPE

pe = LapPE(
    dim=32,
    k=16,  # Number of eigenvectors
    sign_invariant=True,
    sign_invariance_method="abs"  # or "flip", "square"
)

node_pe = pe(data)  # [num_nodes, 32]
```

### 2. **RWSE** - Random-Walk Structural Encoding
Computes return probabilities of random walks.

```python
from src.pe.node import RWSE

pe = RWSE(
    dim=32,
    scales=[1, 2, 4, 8, 16],  # RW steps
    normalization="graph"
)

node_pe = pe(data)  # [num_nodes, 32]
```

### 3. **HKS** - Heat Kernel Signatures
Uses heat diffusion on the graph.

```python
from src.pe.node import HKS

pe = HKS(
    dim=32,
    scales=[0.1, 1.0, 10.0],  # Diffusion times
    k_eigenvectors=50,
    normalization="graph"
)

node_pe = pe(data)  # [num_nodes, 32]
```

### 4. **RolePE** - Role-based Positional Encoding
Structural node features (degree, clustering, k-core).

```python
from src.pe.node import RolePE

pe = RolePE(
    dim=8,
    features=["degree", "clustering", "core"],
    normalization="graph"
)

node_pe = pe(data)  # [num_nodes, 8]
```

## Common Parameters

All PE classes support:

- `dim`: Output embedding dimension
- `normalization`: `"graph"`, `"node"`, or `None`
- `cache`: Whether to cache PE in `data.node_pe`

## Usage Example

```python
from src.pe.node import RWSE
from torch_geometric.data import Data

# Create graph
data = Data(
    edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
    num_nodes=3
)

# Create PE
pe = RWSE(dim=16, scales=[1, 2, 4, 8])

# Compute PE
node_pe = pe(data)
print(node_pe.shape)  # [3, 16]
```

## Testing

```bash
pytest tests/test_node_pe.py -v
```

## Mathematical Formulas

### LapPE
Uses top-k eigenvectors of normalized Laplacian:
```
L = I - D^{-1/2} A D^{-1/2}
```

### RWSE
Return probability in random walk:
```
RWSE_t(i) = P^t(i, i)
```
where P = D^{-1} A is the transition matrix.

### HKS
Heat kernel signature:
```
HKS_t(i) = Σ_k exp(-λ_k * t) * φ_k(i)^2
```
where λ_k, φ_k are eigenvalues and eigenvectors of the Laplacian.

## Sign-Invariance

For spectral PEs (LapPE, HKS), sign-invariance methods are available:

- `"abs"`: Takes absolute value
- `"flip"`: Concatenates [φ, -φ]
- `"square"`: Squares values (default for HKS)
