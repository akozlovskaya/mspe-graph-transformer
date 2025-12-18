"""Reproducibility utilities for deterministic experiments."""

import os
import random
import hashlib
import json
import subprocess
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic CUDA operations.
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic operations
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # PyTorch 1.8+
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    logger.info(f"Set global seed to {seed}, deterministic={deterministic}")


def get_git_info() -> Dict[str, str]:
    """
    Get current git repository information.

    Returns:
        Dictionary with commit hash, branch, and dirty status.
    """
    info = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Dirty status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0

    except Exception as e:
        logger.warning(f"Failed to get git info: {e}")

    return info


def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information for reproducibility.

    Returns:
        Dictionary with environment details.
    """
    info = {
        "python_version": None,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "numpy_version": None,
    }

    try:
        import sys
        info["python_version"] = sys.version

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)

        info["numpy_version"] = np.__version__

    except Exception as e:
        logger.warning(f"Failed to get environment info: {e}")

    return info


def create_reproducibility_info(
    config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """
    Create complete reproducibility information.

    Args:
        config: Experiment configuration.
        seed: Random seed used.

    Returns:
        Full reproducibility dictionary.
    """
    return {
        "seed": seed,
        "git": get_git_info(),
        "environment": get_environment_info(),
        "config_hash": hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "timestamp": datetime.now().isoformat(),
    }


def save_reproducibility_info(
    info: Dict[str, Any],
    output_dir: Path,
    filename: str = "reproducibility.json",
) -> Path:
    """
    Save reproducibility information to file.

    Args:
        info: Reproducibility information.
        output_dir: Output directory.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)

    return path


def verify_reproducibility(
    output_dir1: Path,
    output_dir2: Path,
    metrics_file: str = "results.json",
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Verify that two experiment outputs are identical.

    Args:
        output_dir1: First output directory.
        output_dir2: Second output directory.
        metrics_file: Metrics file to compare.
        tolerance: Numerical tolerance.

    Returns:
        Verification results.
    """
    result = {
        "identical": True,
        "differences": [],
        "warnings": [],
    }

    # Load results
    path1 = Path(output_dir1) / metrics_file
    path2 = Path(output_dir2) / metrics_file

    if not path1.exists():
        result["identical"] = False
        result["differences"].append(f"Missing: {path1}")
        return result

    if not path2.exists():
        result["identical"] = False
        result["differences"].append(f"Missing: {path2}")
        return result

    with open(path1) as f:
        data1 = json.load(f)
    with open(path2) as f:
        data2 = json.load(f)

    # Compare metrics
    def compare_values(v1, v2, path=""):
        if isinstance(v1, dict) and isinstance(v2, dict):
            for key in set(v1.keys()) | set(v2.keys()):
                if key not in v1:
                    result["differences"].append(f"Missing in run1: {path}.{key}")
                    result["identical"] = False
                elif key not in v2:
                    result["differences"].append(f"Missing in run2: {path}.{key}")
                    result["identical"] = False
                else:
                    compare_values(v1[key], v2[key], f"{path}.{key}")
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(v1 - v2) > tolerance:
                result["differences"].append(
                    f"Mismatch at {path}: {v1} vs {v2} (diff={abs(v1-v2)})"
                )
                result["identical"] = False
        elif v1 != v2:
            # Skip timestamps and paths
            if "time" not in path.lower() and "path" not in path.lower():
                result["differences"].append(f"Mismatch at {path}: {v1} vs {v2}")
                result["identical"] = False

    compare_values(data1, data2)

    return result


@contextmanager
def reproducibility_context(seed: int = 42, deterministic: bool = True):
    """
    Context manager for reproducible execution.

    Args:
        seed: Random seed.
        deterministic: Whether to use deterministic operations.
    """
    # Save current state
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Set seed
    set_global_seed(seed, deterministic)

    try:
        yield
    finally:
        # Restore state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class ReproducibilityChecker:
    """Checker for experiment reproducibility."""

    def __init__(self, output_dir: Path):
        """
        Initialize checker.

        Args:
            output_dir: Output directory for test runs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_twice(
        self,
        run_fn: callable,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run experiment twice and compare.

        Args:
            run_fn: Function to run (should return results dict).
            seed: Random seed.
            **kwargs: Arguments for run_fn.

        Returns:
            Comparison results.
        """
        # Run 1
        run1_dir = self.output_dir / "run1"
        set_global_seed(seed)
        results1 = run_fn(output_dir=run1_dir, **kwargs)

        # Save results
        with open(run1_dir / "results.json", "w") as f:
            json.dump(results1, f, default=str)

        # Run 2
        run2_dir = self.output_dir / "run2"
        set_global_seed(seed)
        results2 = run_fn(output_dir=run2_dir, **kwargs)

        # Save results
        with open(run2_dir / "results.json", "w") as f:
            json.dump(results2, f, default=str)

        # Compare
        comparison = verify_reproducibility(run1_dir, run2_dir)

        return {
            "run1": results1,
            "run2": results2,
            "comparison": comparison,
        }

