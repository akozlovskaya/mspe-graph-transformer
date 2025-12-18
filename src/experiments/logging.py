"""Experiment logging utilities."""

import logging
import json
import sys
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime


def setup_experiment_logging(
    output_dir: Union[str, Path],
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Setup logging for experiment.

    Args:
        output_dir: Output directory for log files.
        log_level: Logging level.
        log_to_file: Whether to log to file.
        log_to_console: Whether to log to console.

    Returns:
        Configured logger.
    """
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"experiment_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class ExperimentLogger:
    """Logger for experiment tracking."""

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize experiment logger.

        Args:
            output_dir: Output directory for logs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._config_logged = False
        self._metrics_history = []

    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary.
        """
        config_path = self.output_dir / "config.yaml"

        # Save as YAML
        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON
            config_path = self.output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        self._config_logged = True

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Log metrics.

        Args:
            metrics: Metrics dictionary.
            step: Optional step/epoch number.
            prefix: Prefix for metric names.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "prefix": prefix,
            "metrics": metrics,
        }
        self._metrics_history.append(entry)

        # Save to file
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics_history, f, indent=2, default=str)

    def log_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = "json",
    ):
        """
        Log artifact (file).

        Args:
            name: Artifact name.
            data: Artifact data.
            artifact_type: Type of artifact ('json', 'pt', 'txt').
        """
        artifacts_dir = self.output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if artifact_type == "json":
            path = artifacts_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif artifact_type == "pt":
            import torch
            path = artifacts_dir / f"{name}.pt"
            torch.save(data, path)
        elif artifact_type == "txt":
            path = artifacts_dir / f"{name}.txt"
            with open(path, "w") as f:
                f.write(str(data))

    def log_text(self, name: str, text: str):
        """Log text content."""
        self.log_artifact(name, text, artifact_type="txt")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: Optional[float] = None,
    ):
        """
        Log epoch results.

        Args:
            epoch: Epoch number.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            lr: Learning rate.
        """
        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
        }
        self.log_metrics(entry, step=epoch, prefix="epoch")

    def log_final_results(self, results: Dict[str, Any]):
        """
        Log final experiment results.

        Args:
            results: Final results dictionary.
        """
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def get_metrics_history(self) -> list:
        """Get all logged metrics."""
        return self._metrics_history


def log_config(config: Dict[str, Any], output_dir: Union[str, Path]):
    """
    Log configuration to file.

    Args:
        config: Configuration dictionary.
        output_dir: Output directory.
    """
    logger = ExperimentLogger(output_dir)
    logger.log_config(config)


def log_metrics(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    step: Optional[int] = None,
):
    """
    Log metrics to file.

    Args:
        metrics: Metrics dictionary.
        output_dir: Output directory.
        step: Optional step number.
    """
    logger = ExperimentLogger(output_dir)
    logger.log_metrics(metrics, step)


def log_artifact(
    name: str,
    data: Any,
    output_dir: Union[str, Path],
    artifact_type: str = "json",
):
    """
    Log artifact to file.

    Args:
        name: Artifact name.
        data: Artifact data.
        output_dir: Output directory.
        artifact_type: Artifact type.
    """
    logger = ExperimentLogger(output_dir)
    logger.log_artifact(name, data, artifact_type)


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Log directory.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            self.writer = None
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        """Close writer."""
        if self.enabled:
            self.writer.close()


class WandbLogger:
    """Weights & Biases logging wrapper."""

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name.
            name: Run name.
            config: Configuration to log.
        """
        try:
            import wandb
            self.run = wandb.init(project=project, name=name, config=config)
            self.enabled = True
        except ImportError:
            self.run = None
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            import wandb
            wandb.log(metrics, step=step)

    def log_artifact(self, name: str, path: str, artifact_type: str = "model"):
        """Log artifact."""
        if self.enabled:
            import wandb
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)

    def finish(self):
        """Finish run."""
        if self.enabled:
            import wandb
            wandb.finish()

