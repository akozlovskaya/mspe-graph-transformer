# Experiment Orchestration Module

Module for managing experiments, ablations, and sweeps.

## Structure

```
src/experiments/
    __init__.py      # Public API
    registry.py      # Experiment registry
    runner.py        # Experiment execution
    sweeps.py        # Sweep management
    logging.py       # Logging
    utils.py         # Utilities
```

## Quick Start

### Running a Single Experiment

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

### Using Experiment Registry

```python
from src.experiments import register_experiment, get_experiment

# Registration
register_experiment(
    "my_experiment",
    dataset={"name": "zinc"},
    model={"name": "graph_transformer"},
    tags=["baseline"],
)

# Retrieval
config = get_experiment("my_experiment")
```

### Creating Ablations

```python
from src.experiments import ExperimentRegistry

registry = ExperimentRegistry()
registry.register("base", dataset={"name": "zinc"}, model={"num_layers": 6})

# Ablation with depth change
ablation = registry.create_ablation(
    "base",
    "base_depth_12",
    **{"model.num_layers": 12}
)
```

## Sweeps

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

### Running an Experiment

```bash
# With Hydra
python scripts/run_experiment.py dataset=zinc model=graph_transformer pe=mspe

# With CLI arguments
python scripts/run_experiment.py --dataset zinc --model graph_transformer --epochs 100
```

### Running a Sweep

```bash
# Predefined sweep
python scripts/run_sweep.py --sweep pe_ablation

# From configuration file
python scripts/run_sweep.py --sweep configs/experiments/model_depth.yaml

# Seed sweep
python scripts/run_sweep.py --sweep_type seed --base_experiment zinc_mspe --seeds 42 123 456
```

## Output Structure

```
outputs/
    experiment_name/
        experiment_id/
            config.yaml          # Configuration
            checkpoints/         # Model checkpoints
                best.pt
                last.pt
            logs/                # Logs
                experiment_*.log
            metrics.json         # Metric history
            results.json         # Final results
            long_range.json      # Long-range metrics
            artifacts/           # Additional artifacts
```

## Result Aggregation

```python
from src.experiments import aggregate_results, results_to_dataframe

# Load results
results_list = [load_experiment_results(path) for path in experiment_paths]

# Aggregation (mean, std)
aggregated = aggregate_results(results_list)
print(f"MAE: {aggregated['aggregated_metrics']['mae']['mean']:.4f} ± "
      f"{aggregated['aggregated_metrics']['mae']['std']:.4f}")

# Export to table
df = results_to_dataframe(results_list)
df.to_csv("results_table.csv")
```

## Predefined Configurations

### Experiments

- `zinc_baseline` - Baseline on ZINC
- `zinc_mspe` - MSPE on ZINC
- `peptides_func_baseline` - Baseline on Peptides-func
- `peptides_func_mspe` - MSPE on Peptides-func

### Sweeps

- `pe_ablation` - Ablation over PE types
- `depth_sweep` - Ablation over model depth
- `pe_scale` - Ablation over PE dimension
- `seed_sweep` - Reproducibility with different seeds

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Geometric >= 2.0
- Hydra >= 1.0 (optional)
- PyYAML
