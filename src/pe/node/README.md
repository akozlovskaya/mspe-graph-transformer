# Node-wise Positional Encodings

Модуль для вычисления node-wise структурных позиционных кодировок для графов.

## 📦 Доступные PE

### 1. **LapPE** - Laplacian Positional Encoding
Использует собственные векторы нормализованного лапласиана графа.

```python
from src.pe.node import LapPE

pe = LapPE(
    dim=32,
    k=16,  # Количество собственных векторов
    sign_invariant=True,
    sign_invariance_method="abs"  # или "flip", "square"
)

node_pe = pe(data)  # [num_nodes, 32]
```

### 2. **RWSE** - Random-Walk Structural Encoding
Вычисляет вероятности возврата случайного блуждания.

```python
from src.pe.node import RWSE

pe = RWSE(
    dim=32,
    scales=[1, 2, 4, 8, 16],  # Шаги RW
    normalization="graph"
)

node_pe = pe(data)  # [num_nodes, 32]
```

### 3. **HKS** - Heat Kernel Signatures
Использует диффузию тепла на графе.

```python
from src.pe.node import HKS

pe = HKS(
    dim=32,
    scales=[0.1, 1.0, 10.0],  # Времена диффузии
    k_eigenvectors=50,
    normalization="graph"
)

node_pe = pe(data)  # [num_nodes, 32]
```

### 4. **RolePE** - Role-based Positional Encoding
Структурные признаки узлов (степень, кластеризация, k-core).

```python
from src.pe.node import RolePE

pe = RolePE(
    dim=8,
    features=["degree", "clustering", "core"],
    normalization="graph"
)

node_pe = pe(data)  # [num_nodes, 8]
```

## 🔧 Общие параметры

Все PE классы поддерживают:

- `dim`: Размерность выходного embedding
- `normalization`: `"graph"`, `"node"`, или `None`
- `cache`: Кэшировать ли PE в `data.node_pe`

## 📊 Пример использования

```python
from src.pe.node import RWSE
from torch_geometric.data import Data

# Создать граф
data = Data(
    edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
    num_nodes=3
)

# Создать PE
pe = RWSE(dim=16, scales=[1, 2, 4, 8])

# Вычислить PE
node_pe = pe(data)
print(node_pe.shape)  # [3, 16]
```

## 🧪 Тестирование

```bash
pytest tests/test_node_pe.py -v
```

## 📚 Математические формулы

### LapPE
Использует top-k собственные векторы нормализованного лапласиана:
```
L = I - D^{-1/2} A D^{-1/2}
```

### RWSE
Вероятность возврата в случайном блуждании:
```
RWSE_t(i) = P^t(i, i)
```
где P = D^{-1} A - матрица переходов.

### HKS
Heat kernel signature:
```
HKS_t(i) = Σ_k exp(-λ_k * t) * φ_k(i)^2
```
где λ_k, φ_k - собственные значения и векторы лапласиана.

## 🔍 Sign-Invariance

Для спектральных PE (LapPE, HKS) доступны методы sign-invariance:

- `"abs"`: Берет модуль значений
- `"flip"`: Конкатенирует [φ, -φ]
- `"square"`: Возводит в квадрат (по умолчанию для HKS)

