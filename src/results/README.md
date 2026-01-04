# Results Processing Module

Module for processing experiment results and generating tables and figures for thesis.

## Structure

```
src/results/
    __init__.py      # Public API
    loader.py        # Result loading
    aggregate.py     # Aggregation over seeds
    tables.py        # Table generation
    plots.py         # Plot generation
    formatting.py    # Formatting and styles
    utils.py         # Utilities
```

## Quick Start

### Loading Results

```python
from src.results import ResultLoader, load_all_experiments

# Load all experiments
loader = ResultLoader("./outputs")
results = loader.load_all()

# Or directly
results = load_all_experiments("./outputs")

# Filtering
from src.results import filter_results
zinc_results = filter_results(results, dataset="zinc", complete=True)
```

### Aggregation

```python
from src.results import aggregate_by_seed

# Aggregation over seeds
aggregated = aggregate_by_seed(
    results,
    group_keys=["dataset", "model", "node_pe", "relative_pe"]
)

for agg in aggregated:
    print(f"{agg.group_values}: MAE = {agg.format_metric('mae')}")
```

### Table Generation

```python
from src.results import make_performance_table, export_table

# Create table
table = make_performance_table(results, metrics=["mae", "loss"])

# Export to LaTeX
export_table(table, "results/tables/performance.tex", format="latex")

# Export to CSV
export_table(table, "results/tables/performance.csv", format="csv")
```

### Plot Generation

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

### Table Generation

```bash
# Main performance table
python scripts/make_tables.py --tables performance --output_format latex csv

# Ablation table
python scripts/make_tables.py --tables ablation --ablation_key node_pe

# All tables for specific dataset
python scripts/make_tables.py --tables all --dataset zinc --output_dir results/tables
```

### Plot Generation

```bash
# Performance vs distance
python scripts/make_plots.py --plots performance_vs_distance --metric mae

# All plots
python scripts/make_plots.py --plots all --output_dir results/figures

# With filtering
python scripts/make_plots.py --plots runtime_vs_accuracy --dataset peptides_func
```

## Table Types

### Performance Table
Main table with metrics for all experiments:
- Grouping by Dataset, Model, Node PE, Relative PE
- Aggregation over seeds (mean ± std)
- Metric selection

### Ablation Table
Ablation study table:
- Rows: base group values
- Columns: ablation key × metrics

### Long-Range Table
Long-range performance table:
- Metrics per distance buckets
- LR-AUC (area under curve)

### Efficiency Table
Efficiency vs accuracy table:
- Quality metrics
- Time, memory, parameters

## Plot Types

### Performance vs Distance
- X: Graph distance
- Y: Metric (MAE, accuracy)
- Grouping by PE type

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
- By PE types

### Ablation Heatmap
- Rows: Node PE types
- Columns: Relative PE types
- Color: Metric value

## Formatting

### Metric Naming
```python
METRIC_FORMATS = {
    "mae": MetricFormat("MAE", precision=4, lower_is_better=True),
    "accuracy": MetricFormat("Accuracy", precision=2, multiply=100, unit="%"),
    ...
}
```

### PE Naming
```python
PE_NAMES = {
    "lap": "LapPE",
    "rwse": "RWSE",
    "combined": "MSPE",
    "spd": "SPD",
    ...
}
```

### Plot Styles
```python
style = PlotStyle(
    figure_size=(8, 6),
    font_size=11,
    colors=["#0077BB", "#EE7733", ...],  # Colorblind-friendly
)
```

## LaTeX Compatibility

All LaTeX tables:
- Compile with `article` or `report` class
- Use `booktabs` for lines
- Ready to include via `\input{}`

```latex
\documentclass{article}
\usepackage{booktabs}
\begin{document}
\input{tables/performance.tex}
\end{document}
```

## Reproducibility

- All outputs are deterministic
- Experiment IDs are embedded in results
- Configurations are saved verbatim
- Timestamps are included

## Output Structure

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
