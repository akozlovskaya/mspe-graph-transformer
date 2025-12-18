"""Experiment traceability utilities."""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from .reproducibility import get_git_info


logger = logging.getLogger(__name__)


class ExperimentTracer:
    """Tracer for linking experiments to code versions and outputs."""

    def __init__(self, base_dir: str = "./outputs"):
        """
        Initialize tracer.

        Args:
            base_dir: Base directory for experiment outputs.
        """
        self.base_dir = Path(base_dir)

    def create_trace(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Create experiment trace.

        Args:
            experiment_name: Name of experiment.
            config: Experiment configuration.
            output_dir: Output directory.

        Returns:
            Trace dictionary.
        """
        git_info = get_git_info()

        trace = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "git_dirty": git_info["dirty"],
            "output_dir": str(output_dir) if output_dir else None,
            "config": config,
        }

        return trace

    def save_trace(
        self,
        trace: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """
        Save trace to output directory.

        Args:
            trace: Trace dictionary.
            output_dir: Output directory.

        Returns:
            Path to trace file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "trace.json"
        with open(path, "w") as f:
            json.dump(trace, f, indent=2, default=str)

        return path

    def verify_trace(
        self,
        output_dir: Path,
        expected_commit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify that output corresponds to current code.

        Args:
            output_dir: Output directory.
            expected_commit: Expected git commit (current if None).

        Returns:
            Verification result.
        """
        trace_path = Path(output_dir) / "trace.json"

        if not trace_path.exists():
            return {
                "valid": False,
                "error": "No trace file found",
            }

        with open(trace_path) as f:
            trace = json.load(f)

        current_git = get_git_info()
        expected = expected_commit or current_git["commit"]

        result = {
            "valid": True,
            "trace_commit": trace.get("git_commit"),
            "current_commit": current_git["commit"],
            "matches": trace.get("git_commit") == expected,
            "warnings": [],
        }

        if not result["matches"]:
            result["warnings"].append(
                f"Commit mismatch: trace={trace.get('git_commit')[:8]}, "
                f"current={current_git['commit'][:8]}"
            )

        if trace.get("git_dirty"):
            result["warnings"].append("Experiment was run with uncommitted changes")

        if current_git["dirty"]:
            result["warnings"].append("Current repository has uncommitted changes")

        return result

    def find_experiments(
        self,
        commit: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find experiments matching criteria.

        Args:
            commit: Git commit to match.
            experiment_name: Experiment name to match.

        Returns:
            List of matching experiment traces.
        """
        matches = []

        for trace_path in self.base_dir.glob("**/trace.json"):
            try:
                with open(trace_path) as f:
                    trace = json.load(f)

                if commit and not trace.get("git_commit", "").startswith(commit):
                    continue

                if experiment_name and trace.get("experiment_name") != experiment_name:
                    continue

                trace["_trace_path"] = str(trace_path)
                trace["_output_dir"] = str(trace_path.parent)
                matches.append(trace)

            except Exception as e:
                logger.warning(f"Failed to load trace {trace_path}: {e}")

        return matches


def attach_trace_to_experiment(
    experiment_name: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Attach trace information to experiment output.

    Args:
        experiment_name: Experiment name.
        config: Configuration.
        output_dir: Output directory.

    Returns:
        Trace dictionary.
    """
    tracer = ExperimentTracer()
    trace = tracer.create_trace(experiment_name, config, output_dir)
    tracer.save_trace(trace, output_dir)
    return trace


def verify_experiment_trace(output_dir: Path) -> Dict[str, Any]:
    """
    Verify experiment trace against current code.

    Args:
        output_dir: Experiment output directory.

    Returns:
        Verification result.
    """
    tracer = ExperimentTracer()
    return tracer.verify_trace(output_dir)

