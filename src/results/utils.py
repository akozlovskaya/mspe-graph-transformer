"""Utility functions for result processing."""

import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

import numpy as np


logger = logging.getLogger(__name__)


def ensure_output_dir(path: Union[str, Path]) -> Path:
    """
    Ensure output directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_id(config: Dict[str, Any]) -> str:
    """
    Generate unique experiment ID from config.

    Args:
        config: Experiment configuration.

    Returns:
        Short hash string.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """
    Compute mean safely.

    Args:
        values: List of values.
        default: Default if empty.

    Returns:
        Mean value.
    """
    if not values:
        return default
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return default
    return float(np.mean(arr))


def safe_std(values: List[float], default: float = 0.0) -> float:
    """
    Compute std safely.

    Args:
        values: List of values.
        default: Default if empty.

    Returns:
        Standard deviation.
    """
    if not values or len(values) < 2:
        return default
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) < 2:
        return default
    return float(np.std(arr))


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Nested dictionary.
        parent_key: Parent key prefix.
        sep: Separator.

    Returns:
        Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(
    d: Dict[str, Any],
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Unflatten dictionary with dot notation keys.

    Args:
        d: Flattened dictionary.
        sep: Separator.

    Returns:
        Nested dictionary.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(n: Union[int, float], precision: int = 2) -> str:
    """
    Format large number with K/M/B suffix.

    Args:
        n: Number to format.
        precision: Decimal precision.

    Returns:
        Formatted string.
    """
    if n >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif n >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif n >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    return str(int(n))


def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file.

    Args:
        data: Data to save.
        path: Output path.
        indent: JSON indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=convert)


def load_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load JSON file safely.

    Args:
        path: File path.

    Returns:
        Loaded data or None.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def create_results_index(
    results_dir: Union[str, Path],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create index of all experiment results.

    Args:
        results_dir: Results directory.
        output_path: Optional output path for index.

    Returns:
        Index dictionary.
    """
    results_dir = Path(results_dir)

    index = {
        "created": datetime.now().isoformat(),
        "base_dir": str(results_dir),
        "experiments": [],
    }

    for results_path in results_dir.glob("**/results.json"):
        try:
            data = load_json(results_path)
            if data:
                index["experiments"].append({
                    "path": str(results_path.parent.relative_to(results_dir)),
                    "name": data.get("experiment_name", "unknown"),
                    "id": data.get("experiment_id", ""),
                    "status": data.get("status", "unknown"),
                    "dataset": data.get("config", {}).get("dataset", {}).get("name", ""),
                    "model": data.get("config", {}).get("model", {}).get("name", ""),
                })
        except Exception as e:
            logger.warning(f"Failed to index {results_path}: {e}")

    if output_path:
        save_json(index, output_path)

    return index


def merge_results(
    results_list: List[Dict[str, Any]],
    keys: List[str] = None,
) -> Dict[str, Any]:
    """
    Merge multiple result dictionaries.

    Args:
        results_list: List of result dictionaries.
        keys: Keys to merge (None for all).

    Returns:
        Merged dictionary.
    """
    if not results_list:
        return {}

    merged = {}

    for results in results_list:
        for key, value in results.items():
            if keys is not None and key not in keys:
                continue

            if key not in merged:
                merged[key] = []
            merged[key].append(value)

    return merged


def compare_configs(
    config1: Dict[str, Any],
    config2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two configurations.

    Args:
        config1: First config.
        config2: Second config.

    Returns:
        Dictionary of differences.
    """
    flat1 = flatten_dict(config1)
    flat2 = flatten_dict(config2)

    all_keys = set(flat1.keys()) | set(flat2.keys())

    differences = {}
    for key in all_keys:
        val1 = flat1.get(key)
        val2 = flat2.get(key)
        if val1 != val2:
            differences[key] = {"config1": val1, "config2": val2}

    return differences


def generate_report_id() -> str:
    """Generate unique report ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"report_{timestamp}"

