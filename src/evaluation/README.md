# Long-Range Dependency Evaluation Framework

Фреймворк для оценки способности Graph Transformers моделировать long-range зависимости в графах.

## Структура

```
src/evaluation/
├── __init__.py           # Публичный API
├── distance_metrics.py   # Вычисление расстояний в графе
├── stratification.py     # Стратификация по расстоянию
├── long_range.py         # Метрики long-range оценки
├── probes.py             # Синтетические probing задачи
├── utils.py              # Утилиты и визуализация
└── README.md             # Документация
```

## Основные концепции

### Distance-based Evaluation

Long-range оценка основана на измерении качества предсказаний как функции графового расстояния между информативными узлами и целевыми узлами.

### Ключевые метрики

- **Metrics per bucket**: Точность/MAE для каждого диапазона расстояний
- **Relative performance drop**: Падение качества относительно short-range
- **AUC (distance-performance)**: Площадь под кривой качество-расстояние
- **Effective Receptive Field (ERF)**: Максимальное расстояние, на котором модель работает

## Быстрый старт

### Базовое использование

```python
from src.evaluation import (
    LongRangeEvaluator,
    add_distance_info_to_data,
    print_evaluation_summary,
)

# Добавить информацию о расстояниях к данным
for data in dataset:
    add_distance_info_to_data(data, max_distance=20)

# Создать evaluator
evaluator = LongRangeEvaluator(
    max_distance=20,
    bucket_size=2,
    task_type="regression",
)

# Оценка
for batch in loader:
    pred = model(batch)
    distances = get_distances(batch)  # Расстояния для стратификации
    evaluator.update(pred, batch.y, distances)

# Результаты
results = evaluator.compute()
print_evaluation_summary(results)
```

### Запуск скрипта

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

# Полная матрица расстояний [N x N]
distances = compute_shortest_path_distances(
    edge_index, num_nodes, max_distance=20
)

# Sparse формат (только пары в пределах max_distance)
pair_indices, pair_distances = compute_shortest_path_distances_sparse(
    edge_index, num_nodes, max_distance=10
)
```

### Landmark-based Approximation

Для больших графов используйте аппроксимацию через landmarks:

```python
from src.evaluation import compute_landmark_distances

landmark_indices, landmark_distances = compute_landmark_distances(
    edge_index, num_nodes,
    num_landmarks=10,
    selection="degree",  # или "random"
)
```

### Добавление расстояний к Data

```python
from src.evaluation import add_distance_info_to_data

data = add_distance_info_to_data(
    data,
    max_distance=20,
    sparse=True,           # Sparse storage для больших графов
    use_landmarks=False,   # Landmark approximation
)
```

## Stratification

### Создание buckets

```python
from src.evaluation import create_distance_buckets, stratify_by_distance

# Создать buckets: [(0,1), (2,3), (4,5), ...]
buckets = create_distance_buckets(max_distance=10, bucket_size=2)

# Стратифицировать предсказания
stratified = stratify_by_distance(
    predictions, targets, distances, buckets
)
```

### DistanceStratifier (для накопления batch'ей)

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
    task_type="regression",  # или "classification", "binary"
)
# Возвращает: {(0,1): {"mae": 0.5, "mse": 0.3, ...}, ...}
```

### Relative Performance Drop

```python
from src.evaluation import compute_relative_performance_drop

drops = compute_relative_performance_drop(
    metrics_per_bucket,
    higher_is_better=True,  # False для MAE
)
# Возвращает: {(0,1): 0.0, (2,3): 0.15, ...}
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
    threshold=0.5,  # 50% от baseline
)
```

## Probing Tasks

### Path Parity Probe

Проверяет способность вычислять XOR вдоль пути:

```python
from src.evaluation import PathParityProbe

probe = PathParityProbe(path_length=10)
data = probe.generate_path_graph(seed=42)
batch = probe.generate_batch(batch_size=32)
```

### Node Counting Probe

Проверяет receptive field:

```python
from src.evaluation import NodeCountingProbe

probe = NodeCountingProbe(max_hops=5)
counts, distances = probe.generate_task(data)
# counts[i, k] = количество узлов в k-окрестности узла i
```

### Synthetic Long-Range Task

Задача, требующая информации с заданного расстояния:

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

Для анализа как информация распространяется по слоям:

```python
from src.evaluation import evaluate_layer_wise_long_range

# Модель должна поддерживать return_intermediate=True
results = evaluate_layer_wise_long_range(
    hidden_states,     # List[Tensor] per layer
    targets,
    distances,
    linear_probes,     # Optional: probes for each layer
    max_distance=20,
)

# results[layer_idx] = {"auc": ..., "erf": ..., ...}
```

## Сравнение PE Конфигураций

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

Результаты сохраняются в JSON:

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

## Детерминизм

Все вычисления детерминистичны при фиксированном seed:

```python
from src.training import set_seed

set_seed(42)
# Все операции будут воспроизводимы
```

## Constraints

- Расстояния вычисляются один раз и кэшируются в Data объектах
- Избегается O(N²) хранение для больших графов (sparse формат)
- Gradients отключены во время оценки

