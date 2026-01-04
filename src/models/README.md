# Graph Transformer Models

Module implementing Graph Transformer architecture in GraphGPS style with support for multi-scale positional encodings.

## Main Components

### 1. **GraphTransformer** - Main Model

```python
from src.models import GraphTransformer

model = GraphTransformer(
    node_dim=9,            # Node feature dimension
    hidden_dim=128,        # Hidden dimension
    num_layers=12,         # Number of GPS layers
    num_heads=8,           # Number of attention heads
    out_dim=1,             # Output dimension
    mpnn_type="gin",       # MPNN type: 'gin', 'gat', 'gcn'
    node_pe_dim=16,        # Node PE dimension (0 = no PE)
    use_relative_pe=True,  # Use relative PE
    dropout=0.1,
    task="graph",          # 'graph' or 'node'
)

# Forward pass
out = model(data)  # data.x, data.edge_index, data.node_pe, data.edge_pe
```

### 2. **GPSLayer** - Main Block

```python
from src.models import GPSLayer

layer = GPSLayer(
    hidden_dim=128,
    num_heads=8,
    mpnn_type="gin",
    dropout=0.1,
    gate_type="scalar",  # 'scalar', 'vector', 'mlp'
    use_local=True,      # Use local MPNN
    use_global=True,     # Use global attention
)

out = layer(x, edge_index, attention_bias=bias)
```

### 3. **MultiHeadAttention** - Attention with relative PE

```python
from src.models import MultiHeadAttention

attn = MultiHeadAttention(
    hidden_dim=128,
    num_heads=8,
    dropout=0.1,
)

# With attention bias from relative PE
out = attn(x, attention_bias=bias, batch=batch)
```

### 4. **MPNNBlock** - Local message passing

```python
from src.models import MPNNBlock

# GIN
mpnn = MPNNBlock(hidden_dim=128, mpnn_type="gin")

# GAT
mpnn = MPNNBlock(hidden_dim=128, mpnn_type="gat", num_heads=4)

# GCN
mpnn = MPNNBlock(hidden_dim=128, mpnn_type="gcn")

out = mpnn(x, edge_index)
```

## Architecture

```
Input: data.x [N, node_dim], data.node_pe [N, pe_dim], data.edge_pe [P, pe_dim]
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Node PE Integration       │
                    │   [x, pe] → Linear → hidden   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        GPS Layer × L          │
                    │ ┌───────────┐ ┌───────────┐   │
                    │ │ Local     │ │ Global    │   │
                    │ │ MPNN      │ │ Attention │   │
                    │ └─────┬─────┘ └─────┬─────┘   │
                    │       │   Gate Mixing │       │
                    │       └──────┬───────┘       │
                    │              │                │
                    │       ┌──────▼──────┐        │
                    │       │     FFN     │        │
                    │       └─────────────┘        │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Readout + Pred Head       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                              Output [B, out_dim]
```

## Input Data

The model expects PyG Data or Batch with fields:

- `x`: Node features [N, node_dim]
- `edge_index`: Edge indices [2, E]
- `node_pe`: Node positional encodings [N, pe_dim] (optional)
- `edge_pe_index`: Relative PE indices [2, P] (optional)
- `edge_pe`: Relative PE values [P, pe_dim] (optional)
- `batch`: Batch assignment [N] (for Batch)

## Features

### Pre-LN Normalization
Uses Pre-LN (LayerNorm before each sub-block) for stability in deep models.

### Gating Mechanism
Learnable gate for mixing local (MPNN) and global (attention) features:
```
h = gate * h_global + (1 - gate) * h_local
```

### Stochastic Depth
Drop path for regularization in deep models (linearly increasing rate).

### Relative PE Integration
Attention bias from relative PE:
```
attn = softmax(QK^T / √d + bias)
```

## Testing

```bash
pytest tests/test_graph_transformer.py -v
```

## Examples

### Graph Classification

```python
from src.models import GraphTransformer
from torch_geometric.data import DataLoader

model = GraphTransformer(
    node_dim=dataset.num_features,
    hidden_dim=128,
    num_layers=12,
    num_heads=8,
    out_dim=dataset.num_classes,
    mpnn_type="gin",
    node_pe_dim=16,
)

for batch in train_loader:
    out = model(batch)  # [B, num_classes]
    loss = criterion(out, batch.y)
```

### Node Classification

```python
model = GraphTransformer(
    node_dim=dataset.num_features,
    hidden_dim=128,
    num_layers=8,
    num_heads=8,
    out_dim=dataset.num_classes,
    task="node",
)

out = model(data)  # [N, num_classes]
```

### Getting Node Embeddings

```python
embeddings = model.get_node_embeddings(data)  # [N, hidden_dim]
```

## References

- [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454) (Rampášek et al., 2022)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., 2018)
