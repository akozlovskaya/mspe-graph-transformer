# Graph Transformer Models

Модуль реализующий Graph Transformer архитектуру в стиле GraphGPS с поддержкой multi-scale positional encodings.

## 📦 Основные компоненты

### 1. **GraphTransformer** - Основная модель

```python
from src.models import GraphTransformer

model = GraphTransformer(
    node_dim=9,            # Размерность node features
    hidden_dim=128,        # Скрытое измерение
    num_layers=12,         # Количество GPS слоёв
    num_heads=8,           # Количество attention heads
    out_dim=1,             # Выходное измерение
    mpnn_type="gin",       # Тип MPNN: 'gin', 'gat', 'gcn'
    node_pe_dim=16,        # Размерность node PE (0 = без PE)
    use_relative_pe=True,  # Использовать relative PE
    dropout=0.1,
    task="graph",          # 'graph' или 'node'
)

# Forward pass
out = model(data)  # data.x, data.edge_index, data.node_pe, data.edge_pe
```

### 2. **GPSLayer** - Основной блок

```python
from src.models import GPSLayer

layer = GPSLayer(
    hidden_dim=128,
    num_heads=8,
    mpnn_type="gin",
    dropout=0.1,
    gate_type="scalar",  # 'scalar', 'vector', 'mlp'
    use_local=True,      # Использовать local MPNN
    use_global=True,     # Использовать global attention
)

out = layer(x, edge_index, attention_bias=bias)
```

### 3. **MultiHeadAttention** - Attention с relative PE

```python
from src.models import MultiHeadAttention

attn = MultiHeadAttention(
    hidden_dim=128,
    num_heads=8,
    dropout=0.1,
)

# С attention bias от relative PE
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

## 🏗️ Архитектура

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

## 📊 Входные данные

Модель ожидает PyG Data или Batch с полями:

- `x`: Node features [N, node_dim]
- `edge_index`: Edge indices [2, E]
- `node_pe`: Node positional encodings [N, pe_dim] (опционально)
- `edge_pe_index`: Relative PE indices [2, P] (опционально)
- `edge_pe`: Relative PE values [P, pe_dim] (опционально)
- `batch`: Batch assignment [N] (для Batch)

## 🔧 Особенности

### Pre-LN Normalization
Используется Pre-LN (LayerNorm перед каждым sub-block) для стабильности глубоких моделей.

### Gating Mechanism
Learnable gate для смешивания local (MPNN) и global (attention) features:
```
h = gate * h_global + (1 - gate) * h_local
```

### Stochastic Depth
Drop path для регуляризации глубоких моделей (linearly increasing rate).

### Relative PE Integration
Attention bias из relative PE:
```
attn = softmax(QK^T / √d + bias)
```

## 🧪 Тестирование

```bash
pytest tests/test_graph_transformer.py -v
```

## 📚 Примеры

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

## 📖 Ссылки

- [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454) (Rampášek et al., 2022)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., 2018)

