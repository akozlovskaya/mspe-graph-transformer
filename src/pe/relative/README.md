# Relative (Pairwise) Positional Encodings

Модуль для вычисления relative (pairwise) структурных позиционных кодировок для графов.

## 📦 Доступные PE

### 1. **SPDBuckets** - Shortest-Path Distance Buckets
Вычисляет кратчайшие расстояния между всеми парами узлов и дискретизирует их в buckets.

```python
from src.pe.relative import SPDBuckets

pe = SPDBuckets(
    num_buckets=16,
    max_distance=10,
    use_one_hot=True,
    symmetric=True
)

edge_index_pe, edge_attr_pe = pe(data)
# edge_index_pe: [2, num_pairs] - все пары (i,j)
# edge_attr_pe: [num_pairs, 16] - one-hot encodings расстояний
```

### 2. **BFSDistance** - Truncated BFS Distance
Легковесная версия SPD, хранит только пары в пределах k hops.

```python
from src.pe.relative import BFSDistance

pe = BFSDistance(
    num_buckets=8,
    max_distance=5,  # k
    use_one_hot=True
)

edge_index_pe, edge_attr_pe = pe(data)
# Только пары с d(i,j) <= max_distance
```

### 3. **DiffusionPE** - Heat Kernel Pairwise Encoding
Использует диффузию тепла на графе для вычисления pairwise значений.

```python
from src.pe.relative import DiffusionPE

pe = DiffusionPE(
    num_buckets=4,
    max_distance=10,  # Не используется
    scales=[0.1, 1.0, 5.0, 10.0],  # Времена диффузии
    k_eigenvectors=50
)

edge_index_pe, edge_attr_pe = pe(data)
# edge_attr_pe: [num_pairs, 4] - значения для каждого scale
```

### 4. **EffectiveResistancePE** - Effective Resistance
Аппроксимирует effective resistance между парами узлов.

```python
from src.pe.relative import EffectiveResistancePE

pe = EffectiveResistancePE(
    num_buckets=1,  # Scalar per pair
    max_distance=10,  # Не используется
    k_eigenvectors=50,
    use_sparse=True
)

edge_index_pe, edge_attr_pe = pe(data)
# edge_attr_pe: [num_pairs, 1] - resistance values
```

### 5. **LandmarkSPD** - Landmark-based SPD Approximation
Аппроксимирует SPD используя расстояния до landmark узлов.

```python
from src.pe.relative import LandmarkSPD

pe = LandmarkSPD(
    num_buckets=8,
    max_distance=5,
    num_landmarks=10,
    landmark_method="random",  # или "degree"
    approximation_method="min_diff"  # или "max_diff", "mean_diff"
)

edge_index_pe, edge_attr_pe = pe(data)
```

## 🔧 Интеграция в Attention

Использование для построения attention bias:

```python
from src.pe.relative import SPDBuckets, build_attention_bias

# Вычислить PE
pe = SPDBuckets(num_buckets=16, max_distance=10)
edge_index_pe, edge_attr_pe = pe(data)

# Построить attention bias
bias = build_attention_bias(
    edge_index_pe,
    edge_attr_pe,
    num_nodes=data.num_nodes,
    num_heads=8,
    mode="dense",  # или "sparse"
    gating=True
)

# bias shape: [8, num_nodes, num_nodes] для multi-head attention
```

## 📊 Структура данных

Каждый PE возвращает:

- `edge_index_pe`: Tensor [2, num_pairs] - индексы пар узлов (i, j)
- `edge_attr_pe`: Tensor [num_pairs, num_buckets] - значения PE для каждой пары

## ⚙️ Параметры

Все PE классы поддерживают:

- `num_buckets`: Количество buckets или каналов
- `max_distance`: Максимальное расстояние (для SPD/BFS)
- `normalization`: `"graph"`, `"pair"`, или `None`
- `symmetric`: Гарантировать симметричность PE(i,j) == PE(j,i)
- `cache`: Кэшировать ли PE в `data`

## 📚 Математические формулы

### SPD
Кратчайшее расстояние между узлами:
```
d(i,j) = shortest_path_length(i, j)
```

### Diffusion
Heat kernel на графе:
```
K_t(i,j) = Σ_k exp(-λ_k * t) * φ_k(i) * φ_k(j)
```

### Effective Resistance
Сопротивление между узлами:
```
R(i,j) = L^+_{ii} + L^+_{jj} - 2*L^+_{ij}
```
где L^+ - псевдообратная матрица Лапласиана.

### Landmark SPD
Аппроксимация через landmarks:
```
d(i,j) ≈ min_ℓ |d(i,ℓ) - d(j,ℓ)|
```

## 🧪 Тестирование

```bash
pytest tests/test_relative_pe.py -v
```

## 💡 Использование в Graph Transformers

```python
# В attention слое
def forward(self, x, edge_index_pe, edge_attr_pe):
    # Построить bias
    bias = build_attention_bias(
        edge_index_pe, edge_attr_pe,
        num_nodes=x.size(0),
        num_heads=self.num_heads
    )
    
    # Применить в attention
    attn = self.attention(x, bias=bias)
    return attn
```

## 🔍 Sparse vs Dense

- **Dense**: Хранит все пары (O(N²)) - для малых графов
- **Sparse**: Хранит только релевантные пары - для больших графов

Используйте sparse версии (`SPDBucketsSparse`, `LandmarkSPDSparse`) для масштабируемости.

