"""Experiment utilities."""

import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

import numpy as np


def generate_experiment_id(
    config: Dict[str, Any],
    include_timestamp: bool = True,
) -> str:
    """
    Generate unique experiment ID.

    Args:
        config: Configuration dictionary.
        include_timestamp: Whether to include timestamp.

    Returns:
        Unique experiment ID.
    """
    # Hash config
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{config_hash}_{timestamp}"

    return config_hash


def get_output_dir(
    config: Union[Dict[str, Any], Any],
    base_dir: str = "./outputs",
) -> Path:
    """
    Get output directory for experiment.

    Args:
        config: Experiment configuration.
        base_dir: Base output directory.

    Returns:
        Path to output directory.
    """
    if hasattr(config, "to_dict"):
        config = config.to_dict()

    name = config.get("name", "experiment")
    exp_id = generate_experiment_id(config)

    output_dir = Path(base_dir) / name / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_experiment_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "results.json",
):
    """
    Save experiment results.

    Args:
        results: Results dictionary.
        output_dir: Output directory.
        filename: Results filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / filename

    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        return obj

    with open(results_path, "w") as f:
        json.dump(convert(results), f, indent=2, default=str)


def load_experiment_results(
    output_dir: Union[str, Path],
    filename: str = "results.json",
) -> Dict[str, Any]:
    """
    Load experiment results.

    Args:
        output_dir: Output directory.
        filename: Results filename.

    Returns:
        Results dictionary.
    """
    results_path = Path(output_dir) / filename

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path) as f:
        return json.load(f)


def aggregate_results(
    results_list: List[Dict[str, Any]],
    metrics_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate results from multiple experiments.

    Args:
        results_list: List of experiment results.
        metrics_keys: Specific metric keys to aggregate.

    Returns:
        Aggregated results with mean and std.
    """
    if not results_list:
        return {}

    aggregated = {
        "num_experiments": len(results_list),
        "experiments": [],
        "aggregated_metrics": {},
    }

    # Collect metrics
    metrics_values = {}

    for results in results_list:
        exp_summary = {
            "name": results.get("experiment_name", "unknown"),
            "status": results.get("status", "unknown"),
        }

        # Extract evaluation metrics
        eval_metrics = results.get("evaluation", {}).get("test_metrics", {})
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                if key not in metrics_values:
                    metrics_values[key] = []
                metrics_values[key].append(value)
                exp_summary[key] = value

        aggregated["experiments"].append(exp_summary)

    # Compute aggregated statistics
    for key, values in metrics_values.items():
        if metrics_keys is None or key in metrics_keys:
            values_array = np.array(values)
            aggregated["aggregated_metrics"][key] = {
                "mean": float(values_array.mean()),
                "std": float(values_array.std()),
                "min": float(values_array.min()),
                "max": float(values_array.max()),
                "n": len(values),
            }

    return aggregated


def results_to_dataframe(
    results_list: List[Dict[str, Any]],
) -> "pd.DataFrame":
    """
    Convert results to pandas DataFrame.

    Args:
        results_list: List of experiment results.

    Returns:
        DataFrame with results.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for results_to_dataframe")

    rows = []

    for results in results_list:
        row = {
            "experiment_name": results.get("experiment_name", "unknown"),
            "experiment_id": results.get("experiment_id", "unknown"),
            "status": results.get("status", "unknown"),
            "seed": results.get("config", {}).get("seed", None),
        }

        # Config
        config = results.get("config", {})
        row["dataset"] = config.get("dataset", {}).get("name", "unknown")
        row["model"] = config.get("model", {}).get("name", "unknown")
        row["num_layers"] = config.get("model", {}).get("num_layers", None)
        row["hidden_dim"] = config.get("model", {}).get("hidden_dim", None)

        # PE config
        pe_config = config.get("pe", {})
        row["node_pe_type"] = pe_config.get("node", {}).get("type", "none")
        row["node_pe_dim"] = pe_config.get("node", {}).get("dim", 0)
        row["relative_pe_type"] = pe_config.get("relative", {}).get("type", "none")

        # Evaluation metrics
        eval_metrics = results.get("evaluation", {}).get("test_metrics", {})
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                row[f"test_{key}"] = value

        # Training info
        training = results.get("training", {})
        row["best_epoch"] = training.get("best_epoch")
        row["best_metric"] = training.get("best_metric")

        # Long-range
        long_range = results.get("long_range", {})
        row["lr_auc"] = long_range.get("auc")
        row["lr_erf"] = long_range.get("effective_receptive_field")

        rows.append(row)

    return pd.DataFrame(rows)


def compare_experiments(
    results_list: List[Dict[str, Any]],
    metric: str = "test_loss",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """
    Compare experiments and find best.

    Args:
        results_list: List of experiment results.
        metric: Metric to compare.
        lower_is_better: Whether lower metric is better.

    Returns:
        Comparison results.
    """
    experiments = []

    for results in results_list:
        name = results.get("experiment_name", "unknown")
        eval_metrics = results.get("evaluation", {}).get("test_metrics", {})
        metric_value = eval_metrics.get(metric.replace("test_", ""))

        if metric_value is not None:
            experiments.append({
                "name": name,
                "value": metric_value,
                "config": results.get("config", {}),
            })

    if not experiments:
        return {"error": "No valid experiments found"}

    # Sort
    experiments.sort(key=lambda x: x["value"], reverse=not lower_is_better)

    return {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "best": experiments[0],
        "worst": experiments[-1],
        "ranking": [e["name"] for e in experiments],
        "all": experiments,
    }


def export_results_table(
    results_list: List[Dict[str, Any]],
    output_path: str,
    format: str = "csv",
    columns: Optional[List[str]] = None,
):
    """
    Export results as table.

    Args:
        results_list: List of experiment results.
        output_path: Output file path.
        format: Output format ('csv', 'latex', 'markdown').
        columns: Columns to include.
    """
    df = results_to_dataframe(results_list)

    if columns:
        df = df[[c for c in columns if c in df.columns]]

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "latex":
        df.to_latex(output_path, index=False)
    elif format == "markdown":
        with open(output_path, "w") as f:
            f.write(df.to_markdown(index=False))


def create_ablation_table(
    results_list: List[Dict[str, Any]],
    ablation_param: str,
    metrics: List[str] = ["loss", "mae"],
) -> Dict[str, Any]:
    """
    Create ablation study table.

    Args:
        results_list: List of experiment results.
        ablation_param: Parameter being ablated.
        metrics: Metrics to include.

    Returns:
        Ablation table data.
    """
    rows = []

    for results in results_list:
        config = results.get("config", {})

        # Get ablation parameter value
        param_value = _get_nested_value(config, ablation_param)

        row = {
            "parameter": ablation_param,
            "value": param_value,
        }

        # Get metrics
        eval_metrics = results.get("evaluation", {}).get("test_metrics", {})
        for metric in metrics:
            row[metric] = eval_metrics.get(metric)

        rows.append(row)

    # Group by parameter value and compute stats
    grouped = {}
    for row in rows:
        value = row["value"]
        if value not in grouped:
            grouped[value] = {metric: [] for metric in metrics}
        for metric in metrics:
            if row[metric] is not None:
                grouped[value][metric].append(row[metric])

    # Compute statistics
    table = []
    for value, metric_values in grouped.items():
        entry = {"value": value}
        for metric, values in metric_values.items():
            if values:
                arr = np.array(values)
                entry[f"{metric}_mean"] = float(arr.mean())
                entry[f"{metric}_std"] = float(arr.std())
        table.append(entry)

    return {
        "ablation_param": ablation_param,
        "metrics": metrics,
        "table": table,
    }


def _get_nested_value(d: Dict, key: str, default=None):
    """Get nested dict value using dot notation."""
    keys = key.split(".")
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

