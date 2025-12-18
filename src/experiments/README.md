# Experiment Orchestration Module

Модуль для управления экспериментами, аблациями и sweep'ами.

## Структура

```
src/experiments/
    __init__.py      # Публичный API
    registry.py      # Реестр экспериментов
    runner.py        # Запуск экспериментов
    sweeps.py        # Управление sweep'ами
    logging.py       # Логирование
    utils.py         # Утилиты
```

## Быстрый старт

### Запуск одного эксперимента

```python
from src.experiments import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    name="zinc_mspe",
    dataset={"name": "zinc", "root": "./data"},
    model={"name": "graph_transformer", "num_layers": 6},
    pe={
        "node": {"type": "combined", "dim": 32},
        "relative": {"type": "spd", "num_buckets": 32}
    },
    training={"epochs": 100, "batch_size": 32, "lr": 1e-4},
    seed=42,
)

runner = ExperimentRunner(config, output_dir="./outputs/zinc_mspe")
results = runner.run_all()
```

### Использование реестра экспериментов

```python
from src.experiments import register_experiment, get_experiment

# Регистрация
register_experiment(
    "my_experiment",
    dataset={"name": "zinc"},
    model={"name": "graph_transformer"},
    tags=["baseline"],
)

# Получение
config = get_experiment("my_experiment")
```

### Создание аблации

```python
from src.experiments import ExperimentRegistry

registry = ExperimentRegistry()
registry.register("base", dataset={"name": "zinc"}, model={"num_layers": 6})

# Аблация с изменением глубины
ablation = registry.create_ablation(
    "base",
    "base_depth_12",
    **{"model.num_layers": 12}
)
```

## Sweep'ы

### Grid Sweep

```python
from src.experiments import GridSweep, ExperimentConfig

base = ExperimentConfig(...)

sweep = GridSweep(
    base,
    parameters={
        "pe.node.type": ["none", "lap", "rwse", "combined"],
        "pe.relative.type": ["none", "spd", "diffusion"],
    }
)

for config in sweep.generate():
    runner = ExperimentRunner(config)
    runner.run_all()
```

### Seed Sweep

```python
from src.experiments import SeedSweep

sweep = SeedSweep(base_config, seeds=[42, 123, 456, 789, 1024])

for config in sweep.generate():
    runner = ExperimentRunner(config)
    runner.run_all()
```

## CLI

### Запуск эксперимента

```bash
# С Hydra
python scripts/run_experiment.py dataset=zinc model=graph_transformer pe=mspe

# С CLI аргументами
python scripts/run_experiment.py --dataset zinc --model graph_transformer --epochs 100
```

### Запуск sweep'а

```bash
# Предустановленный sweep
python scripts/run_sweep.py --sweep pe_ablation

# Из файла конфигурации
python scripts/run_sweep.py --sweep configs/experiments/model_depth.yaml

# Seed sweep
python scripts/run_sweep.py --sweep_type seed --base_experiment zinc_mspe --seeds 42 123 456
```

## Структура выходных данных

```
outputs/
    experiment_name/
        experiment_id/
            config.yaml          # Конфигурация
            checkpoints/         # Чекпоинты модели
                best.pt
                last.pt
            logs/                # Логи
                experiment_*.log
            metrics.json         # История метрик
            results.json         # Итоговые результаты
            long_range.json      # Long-range метрики
            artifacts/           # Дополнительные артефакты
```

## Агрегация результатов

```python
from src.experiments import aggregate_results, results_to_dataframe

# Загрузка результатов
results_list = [load_experiment_results(path) for path in experiment_paths]

# Агрегация (mean, std)
aggregated = aggregate_results(results_list)
print(f"MAE: {aggregated['aggregated_metrics']['mae']['mean']:.4f} ± "
      f"{aggregated['aggregated_metrics']['mae']['std']:.4f}")

# Экспорт в таблицу
df = results_to_dataframe(results_list)
df.to_csv("results_table.csv")
```

## Предустановленные конфигурации

### Эксперименты

- `zinc_baseline` - Baseline на ZINC
- `zinc_mspe` - MSPE на ZINC
- `peptides_func_baseline` - Baseline на Peptides-func
- `peptides_func_mspe` - MSPE на Peptides-func

### Sweep'ы

- `pe_ablation` - Аблация по типам PE
- `depth_sweep` - Аблация по глубине модели
- `pe_scale` - Аблация по размерности PE
- `seed_sweep` - Воспроизводимость с разными seed'ами

## Требования

- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Geometric >= 2.0
- Hydra >= 1.0 (опционально)
- PyYAML

