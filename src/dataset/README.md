# Dataset Module

Модуль для загрузки и обработки графовых датасетов с поддержкой Multi-Scale Structural Positional Encodings (MSPE).

## 🎯 Основные возможности

- **Унифицированный API** для всех датасетов
- **Автоматическая загрузка и подготовка** данных
- **Встроенная поддержка PE** через transforms
- **Совместимость с PyTorch Geometric**
- **Поддержка множества датасетов**: ZINC, QM9, LRGB, OGB, PCQM, синтетические графы

## 📦 Поддерживаемые датасеты

### Molecular
- **ZINC**: Молекулярный датасет для регрессии
- **QM9**: Квантово-механические свойства молекул

### LRGB
- **Peptides-func**: Функциональные свойства пептидов
- **Peptides-struct**: Структурные свойства пептидов
- **PascalVOC-SP**: Superpixel графы из PascalVOC
- **CIFAR10-SP**: Superpixel графы из CIFAR10

### OGB
- **ogbg-molhiv**: Бинарная классификация HIV активности
- **ogbg-molpcba**: Мульти-задачная классификация PCBA
- **PCQM4M**: Квантовые свойства молекул (4M графов)
- **PCQM-Contact**: Edge prediction задача

### Synthetic
- **synthetic_grid_2d**: 2D сетки
- **synthetic_grid_3d**: 3D сетки
- **synthetic_ring**: Кольцевые графы
- **synthetic_tree**: Сбалансированные деревья
- **synthetic_random_regular**: Случайные регулярные графы
- **synthetic_barabasi_albert**: Графы Барабаши-Альберта
- **synthetic_watts_strogatz**: Графы Уоттса-Строгаца
- **synthetic_erdos_renyi**: Графы Эрдёша-Реньи

## 🚀 Быстрый старт

### Базовое использование

```python
from src.dataset import get_dataset
from torch_geometric.data import DataLoader

# Загрузка датасета с PE
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

# Создание DataLoader
train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)

# Использование в цикле обучения
for batch in train_loader:
    print(f"Batch size: {batch.batch.max().item() + 1}")
    print(f"Node features: {batch.x.shape}")
    print(f"Node PE: {batch.node_pe.shape}")
    print(f"Edge PE: {batch.edge_pe.shape}")
    print(f"Targets: {batch.y.shape}")
```

### Загрузка LRGB датасета

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

### Синтетические графы

```python
# Генерация кольцевых графов
dataset = get_dataset(
    name="synthetic_ring",
    root="./data",
    num_graphs=1000,
    graph_params={"n": 20},  # 20 узлов в каждом графе
    pe_config={"node": {"enabled": True}, "relative": {"enabled": True}}
)

# Генерация 2D сеток
dataset = get_dataset(
    name="synthetic_grid_2d",
    num_graphs=500,
    graph_params={"m": 10, "n": 10},  # 10x10 сетка
    pe_config={"node": {"enabled": True}, "relative": {"enabled": True}}
)
```

## 🔧 Конфигурация PE

### Node-wise PE

```python
node_pe_config = {
    "enabled": True,
    "types": ["lap_pe", "rwse", "hks"],  # Типы PE
    "dim": 32,                            # Размерность PE
    "scales": [1, 2, 4, 8]                # Масштабы для multi-scale
}
```

### Relative PE

```python
relative_pe_config = {
    "enabled": True,
    "types": ["spd", "diffusion", "effective_resistance"],
    "max_distance": 10,                   # Максимальное расстояние
    "num_buckets": 16                     # Количество бакетов
}
```

## 📊 Структура данных

Каждый граф в датасете имеет следующую структуру:

```python
Data(
    x=torch.Tensor,           # Node features [num_nodes, num_features]
    edge_index=torch.Tensor,   # Edge indices [2, num_edges]
    edge_attr=torch.Tensor,   # Edge attributes [num_edges, edge_dim] (опционально)
    node_pe=torch.Tensor,      # Node positional encodings [num_nodes, pe_dim]
    edge_pe=torch.Tensor,     # Relative positional encodings [num_edges, num_buckets]
    y=torch.Tensor,           # Target [num_targets] или [1]
    pos=torch.Tensor,         # Node positions (если доступны) [num_nodes, 2/3]
)
```

## 🛠️ Утилиты

### Вычисление статистики датасета

```python
from src.dataset.utils import compute_dataset_stats

stats = compute_dataset_stats(dataset.train)
print(f"Average nodes: {stats['avg_num_nodes']}")
print(f"Average edges: {stats['avg_num_edges']}")
print(f"Target mean: {stats.get('target_mean', 'N/A')}")
```

### Создание случайных split'ов

```python
from src.dataset.utils import create_random_split

train_idx, val_idx, test_idx = create_random_split(
    dataset, train_ratio=0.8, val_ratio=0.1, seed=42
)
```

### Нормализация таргетов

```python
from src.dataset.utils import normalize_targets

mean, std = normalize_targets(dataset.train)
```

## 🧪 Тестирование

Запуск тестов:

```bash
pytest tests/test_dataset_loading.py -v
```

## 📝 API Reference

### `get_dataset(name, root, pe_config, **kwargs)`

Фабричная функция для создания датасета.

**Параметры:**
- `name` (str): Имя датасета
- `root` (str): Корневая директория для хранения данных
- `pe_config` (dict): Конфигурация позиционных кодировок
- `**kwargs`: Дополнительные параметры для конкретного датасета

**Возвращает:**
- `BaseGraphDataset`: Экземпляр датасета с атрибутами `train`, `val`, `test`

### `BaseGraphDataset`

Базовый класс для всех датасетов.

**Методы:**
- `load()`: Загружает train/val/test splits
- `get_splits(splits="official")`: Возвращает словарь с splits

**Свойства:**
- `num_features`: Количество признаков узлов
- `num_classes`: Количество классов (1 для регрессии)

### Transforms

- `ApplyNodePE`: Применяет node-wise PE
- `ApplyRelativePE`: Применяет relative PE
- `CompositeTransform`: Композитный transform для применения нескольких transforms
- `NormalizeTargets`: Нормализует таргеты
- `CastDataTypes`: Приводит типы данных

## 🔍 Примеры использования

### Полный пример обучения

```python
from src.dataset import get_dataset
from torch_geometric.data import DataLoader
import torch

# Загрузка датасета
dataset = get_dataset(
    name="peptides_func",
    root="./data",
    pe_config={
        "node": {"enabled": True, "types": ["rwse"], "dim": 32},
        "relative": {"enabled": True, "types": ["spd"], "num_buckets": 16}
    }
)

# Создание DataLoader'ов
train_loader = DataLoader(dataset.train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset.val, batch_size=32, shuffle=False)

# Обучение
for epoch in range(10):
    for batch in train_loader:
        # batch.x - node features
        # batch.node_pe - node positional encodings
        # batch.edge_pe - relative positional encodings
        # batch.y - targets
        # ... ваш код обучения ...
        pass
```

## 📚 Дополнительная информация

- Все датасеты автоматически применяют PE через transforms
- PE вычисляются один раз при загрузке и кешируются в памяти
- Поддержка как классификации, так и регрессии
- Совместимость с PyTorch Geometric DataLoader
- Graceful fallback если PE отключены (нулевые PE)

## 🤝 Вклад

При добавлении новых датасетов:
1. Создайте класс, наследующий `BaseGraphDataset`
2. Реализуйте методы `load()`, `num_features`, `num_classes`
3. Добавьте поддержку в `factory.py`
4. Добавьте тесты в `test_dataset_loading.py`

