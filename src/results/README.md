# Results Processing Module

Модуль для обработки результатов экспериментов и генерации таблиц и графиков для диссертации.

## Структура

```
src/results/
    __init__.py      # Публичный API
    loader.py        # Загрузка результатов
    aggregate.py     # Агрегация по seeds
    tables.py        # Генерация таблиц
    plots.py         # Генерация графиков
    formatting.py    # Форматирование и стили
    utils.py         # Утилиты
```

## Быстрый старт

### Загрузка результатов

```python
from src.results import ResultLoader, load_all_experiments

# Загрузить все эксперименты
loader = ResultLoader("./outputs")
results = loader.load_all()

# Или напрямую
results = load_all_experiments("./outputs")

# Фильтрация
from src.results import filter_results
zinc_results = filter_results(results, dataset="zinc", complete=True)
```

### Агрегация

```python
from src.results import aggregate_by_seed

# Агрегация по seeds
aggregated = aggregate_by_seed(
    results,
    group_keys=["dataset", "model", "node_pe", "relative_pe"]
)

for agg in aggregated:
    print(f"{agg.group_values}: MAE = {agg.format_metric('mae')}")
```

### Генерация таблиц

```python
from src.results import make_performance_table, export_table

# Создать таблицу
table = make_performance_table(results, metrics=["mae", "loss"])

# Экспорт в LaTeX
export_table(table, "results/tables/performance.tex", format="latex")

# Экспорт в CSV
export_table(table, "results/tables/performance.csv", format="csv")
```

### Генерация графиков

```python
from src.results import PlotGenerator, save_figure

gen = PlotGenerator()

# Performance vs Distance
fig = gen.performance_vs_distance(results, metric="mae", max_distance=20)
save_figure(fig, "results/figures/perf_vs_dist", formats=["pdf", "png"])

# Ablation heatmap
fig = gen.ablation_heatmap(results, row_key="node_pe", col_key="relative_pe")
save_figure(fig, "results/figures/ablation_heatmap")
```

## CLI

### Генерация таблиц

```bash
# Основная таблица производительности
python scripts/make_tables.py --tables performance --output_format latex csv

# Аблационная таблица
python scripts/make_tables.py --tables ablation --ablation_key node_pe

# Все таблицы для конкретного датасета
python scripts/make_tables.py --tables all --dataset zinc --output_dir results/tables
```

### Генерация графиков

```bash
# Performance vs distance
python scripts/make_plots.py --plots performance_vs_distance --metric mae

# Все графики
python scripts/make_plots.py --plots all --output_dir results/figures

# С фильтрацией
python scripts/make_plots.py --plots runtime_vs_accuracy --dataset peptides_func
```

## Типы таблиц

### Performance Table
Основная таблица с метриками по всем экспериментам:
- Группировка по Dataset, Model, Node PE, Relative PE
- Агрегация по seeds (mean ± std)
- Выбор метрик

### Ablation Table
Таблица аблационного исследования:
- Строки: значения базовой группы
- Столбцы: значения ablation key × метрики

### Long-Range Table
Таблица long-range производительности:
- Метрики по distance buckets
- LR-AUC (area under curve)

### Efficiency Table
Таблица efficiency vs accuracy:
- Метрики качества
- Время, память, параметры

## Типы графиков

### Performance vs Distance
- X: Graph distance
- Y: Metric (MAE, accuracy)
- Группировка по PE type

### Performance vs Size
- X: Parameters / Runtime / Memory
- Y: Accuracy metric
- Scatter plot

### Runtime vs Accuracy
- Trade-off curve
- Pareto frontier indication

### Depth vs Effective Distance
- X: Number of layers
- Y: Effective receptive field
- По PE types

### Ablation Heatmap
- Rows: Node PE types
- Columns: Relative PE types
- Color: Metric value

## Форматирование

### Именование метрик
```python
METRIC_FORMATS = {
    "mae": MetricFormat("MAE", precision=4, lower_is_better=True),
    "accuracy": MetricFormat("Accuracy", precision=2, multiply=100, unit="%"),
    ...
}
```

### Именование PE
```python
PE_NAMES = {
    "lap": "LapPE",
    "rwse": "RWSE",
    "combined": "MSPE",
    "spd": "SPD",
    ...
}
```

### Стили графиков
```python
style = PlotStyle(
    figure_size=(8, 6),
    font_size=11,
    colors=["#0077BB", "#EE7733", ...],  # Colorblind-friendly
)
```

## LaTeX совместимость

Все LaTeX таблицы:
- Компилируются с `article` или `report` class
- Используют `booktabs` для линий
- Готовы к включению через `\input{}`

```latex
\documentclass{article}
\usepackage{booktabs}
\begin{document}
\input{tables/performance.tex}
\end{document}
```

## Воспроизводимость

- Все выходные данные детерминированы
- ID экспериментов встроены в результаты
- Конфигурации сохраняются verbatim
- Временные метки включены

## Структура выходов

```
results/
    tables/
        performance.tex
        performance.csv
        ablation_node_pe.tex
        long_range.tex
        efficiency.tex
    figures/
        performance_vs_distance.pdf
        performance_vs_distance.png
        runtime_vs_accuracy.pdf
        ablation_heatmap.pdf
```

