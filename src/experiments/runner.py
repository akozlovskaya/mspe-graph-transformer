"""Experiment runner for orchestrating training and evaluation."""

import logging
import traceback
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader

from .registry import ExperimentConfig
from .logging import ExperimentLogger, setup_experiment_logging
from .utils import generate_experiment_id, get_output_dir, save_experiment_results


logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for executing experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[str] = None,
        resume: bool = False,
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration.
            output_dir: Output directory. Auto-generated if None.
            resume: Whether to resume from checkpoint.
        """
        self.config = config
        self.resume = resume

        # Setup output directory
        if output_dir is None:
            output_dir = get_output_dir(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = ExperimentLogger(self.output_dir)
        setup_experiment_logging(self.output_dir)

        # Save config
        self.logger.log_config(config.to_dict())

        # Initialize components
        self.device = self._get_device()
        self.model = None
        self.dataset = None
        self.trainer = None

        # Results storage
        self.results = {
            "experiment_id": config.get_id(),
            "experiment_name": config.name,
            "config": config.to_dict(),
            "status": "initialized",
            "start_time": None,
            "end_time": None,
            "training": {},
            "evaluation": {},
            "long_range": {},
            "profiling": {},
            "errors": [],
        }

    def _get_device(self) -> torch.device:
        """Get device for experiment."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_seed(self):
        """Set random seed for reproducibility."""
        from src.training.reproducibility import set_seed
        set_seed(self.config.seed)
        logger.info(f"Set seed to {self.config.seed}")

    def _load_dataset(self):
        """Load dataset based on config."""
        from src.dataset import get_dataset

        dataset_cfg = self.config.dataset
        pe_cfg = self.config.pe

        logger.info(f"Loading dataset: {dataset_cfg['name']}")

        self.dataset = get_dataset(
            name=dataset_cfg["name"],
            root=dataset_cfg.get("root", "./data"),
            pe_config=pe_cfg,
        )

        return self.dataset

    def _create_model(self):
        """Create model based on config."""
        from src.models import get_model

        model_cfg = self.config.model
        pe_cfg = self.config.pe

        logger.info(f"Creating model: {model_cfg['name']}")

        self.model = get_model(
            name=model_cfg["name"],
            num_features=self.dataset.num_features,
            num_classes=self.dataset.num_classes,
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            dropout=model_cfg.get("dropout", 0.1),
            node_pe_dim=pe_cfg.get("node", {}).get("dim", 0),
            relative_pe_dim=pe_cfg.get("relative", {}).get("num_buckets", 0),
        )

        self.model = self.model.to(self.device)

        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        return self.model

    def _create_trainer(self):
        """Create trainer based on config."""
        from src.training import (
            Trainer, get_optimizer, get_scheduler, get_loss_fn, get_metrics
        )

        training_cfg = self.config.training
        dataset_cfg = self.config.dataset

        # Get optimizer config (can be string or dict)
        opt_cfg = training_cfg.get("optimizer", {})
        if isinstance(opt_cfg, dict):
            optimizer_type = opt_cfg.get("type", "adamw")
            lr = opt_cfg.get("lr", training_cfg.get("lr", 1e-4))
            weight_decay = opt_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.01))
        else:
            optimizer_type = opt_cfg
            lr = training_cfg.get("lr", 1e-4)
            weight_decay = training_cfg.get("weight_decay", 0.01)

        # Optimizer
        optimizer = get_optimizer(
            self.model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Get scheduler config (can be string or dict)
        sched_cfg = training_cfg.get("scheduler", {})
        if isinstance(sched_cfg, dict):
            scheduler_type = sched_cfg.get("type", "cosine_warmup")
            warmup_epochs = sched_cfg.get("warmup_epochs", training_cfg.get("warmup_epochs", 10))
        else:
            scheduler_type = sched_cfg
            warmup_epochs = training_cfg.get("warmup_epochs", 10)

        # Scheduler
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_epochs=training_cfg.get("epochs", 100),
            warmup_epochs=warmup_epochs,
        )

        # Loss and metrics
        task_type = dataset_cfg.get("task_type", "regression")
        loss_fn = get_loss_fn(task_type, num_classes=self.dataset.num_classes)
        metrics = get_metrics(task_type, num_classes=self.dataset.num_classes)

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            device=self.device,
            task_type=task_type,
            grad_clip=training_cfg.get("grad_clip", 1.0),
            mixed_precision=training_cfg.get("mixed_precision", False),
            checkpoint_dir=str(self.output_dir / "checkpoints"),
        )

        return self.trainer

    def _get_data_splits(self):
        """Get train, val, test splits from dataset."""
        splits = self.dataset.get_splits()
        if isinstance(splits, dict):
            train_data = splits.get("train")
            val_data = splits.get("val")
            test_data = splits.get("test")
        else:
            train_data, val_data, test_data = splits
        return train_data, val_data, test_data

    def setup(self):
        """Setup all experiment components."""
        logger.info("=" * 60)
        logger.info(f"Setting up experiment: {self.config.name}")
        logger.info("=" * 60)

        self._set_seed()
        self._load_dataset()
        self._create_model()
        self._create_trainer()

        # Resume from checkpoint if requested
        if self.resume:
            checkpoint_path = self.output_dir / "checkpoints" / "last.pt"
            if checkpoint_path.exists():
                self.trainer.load_checkpoint(str(checkpoint_path))
                logger.info(f"Resumed from checkpoint: {checkpoint_path}")

        self.results["status"] = "setup_complete"

    def run_training(self) -> Dict[str, Any]:
        """
        Run training.

        Returns:
            Training results dictionary.
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

        self.results["start_time"] = datetime.now().isoformat()
        self.results["status"] = "training"

        try:
            # Get data splits
            train_data, val_data, test_data = self._get_data_splits()

            training_cfg = self.config.training
            batch_size = training_cfg.get("batch_size", 32)

            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, num_workers=4
            )
            val_loader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # Training
            history = self.trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=training_cfg.get("epochs", 100),
                metric_for_best=training_cfg.get("metric_for_best", "loss"),
                minimize_metric=training_cfg.get("minimize_metric", True),
                early_stopping=training_cfg.get("early_stopping"),
            )

            self.results["training"] = {
                "history": history,
                "best_epoch": self.trainer.best_epoch,
                "best_metric": self.trainer.best_metric,
            }

            self.results["status"] = "training_complete"
            logger.info("Training completed successfully")

        except Exception as e:
            self.results["status"] = "training_failed"
            self.results["errors"].append({
                "phase": "training",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            logger.error(f"Training failed: {e}")
            raise

        return self.results["training"]

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on test set.

        Returns:
            Evaluation results dictionary.
        """
        logger.info("=" * 60)
        logger.info("Running evaluation")
        logger.info("=" * 60)

        try:
            # Load best checkpoint
            best_checkpoint = self.output_dir / "checkpoints" / "best.pt"
            if best_checkpoint.exists():
                self.trainer.load_checkpoint(str(best_checkpoint), load_optimizer=False)

            # Get test data
            train_data, val_data, test_data = self._get_data_splits()

            if test_data is None:
                logger.warning("No test data available, using validation set")
                test_data = val_data

            test_loader = DataLoader(
                test_data,
                batch_size=self.config.training.get("batch_size", 32),
                shuffle=False,
            )

            # Evaluate
            test_metrics = self.trainer.eval_epoch(test_loader)

            self.results["evaluation"] = {
                "test_metrics": test_metrics,
                "num_samples": len(test_data),
            }

            logger.info(f"Test metrics: {test_metrics}")

        except Exception as e:
            self.results["status"] = "evaluation_failed"
            self.results["errors"].append({
                "phase": "evaluation",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            logger.error(f"Evaluation failed: {e}")
            raise

        return self.results["evaluation"]

    def run_long_range(self, max_distance: int = 20) -> Dict[str, Any]:
        """
        Run long-range evaluation.

        Args:
            max_distance: Maximum distance for evaluation.

        Returns:
            Long-range evaluation results.
        """
        logger.info("=" * 60)
        logger.info("Running long-range evaluation")
        logger.info("=" * 60)

        try:
            # Check if task is suitable for long-range evaluation
            # Long-range evaluation is designed for node-level or edge-level tasks
            model_task = self.config.model.get("task", "graph")
            if model_task == "graph":
                logger.info(
                    "Skipping long-range evaluation: not applicable for graph-level tasks. "
                    "Long-range evaluation is designed for node-level or edge-level predictions."
                )
                self.results["long_range"] = {
                    "status": "skipped",
                    "reason": "graph-level task",
                }
                return self.results["long_range"]

            from src.evaluation import LongRangeEvaluator, add_distance_info_to_data

            # Load best checkpoint
            best_checkpoint = self.output_dir / "checkpoints" / "best.pt"
            if best_checkpoint.exists():
                self.trainer.load_checkpoint(str(best_checkpoint), load_optimizer=False)

            # Get test data
            train_data, val_data, test_data = self._get_data_splits()
            if test_data is None:
                test_data = val_data

            # Add distance info
            for data in test_data:
                add_distance_info_to_data(data, max_distance=max_distance, sparse=True)

            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

            # Evaluate
            task_type = self.config.dataset.get("task_type", "regression")
            evaluator = LongRangeEvaluator(
                max_distance=max_distance,
                task_type=task_type,
            )

            self.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    pred = self.model(batch)

                    # Get distances
                    if hasattr(batch, "distance_values"):
                        distances = batch.distance_values[:batch.num_nodes]
                    else:
                        distances = torch.zeros(batch.num_nodes, dtype=torch.long)

                    evaluator.update(pred, batch.y, distances)

            lr_results = evaluator.compute()
            self.results["long_range"] = lr_results

            logger.info(f"Long-range AUC: {lr_results.get('auc', 'N/A')}")

        except Exception as e:
            self.results["errors"].append({
                "phase": "long_range",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            logger.error(f"Long-range evaluation failed: {e}")

        return self.results["long_range"]

    def run_profiling(self) -> Dict[str, Any]:
        """
        Run efficiency profiling.

        Returns:
            Profiling results.
        """
        logger.info("=" * 60)
        logger.info("Running profiling")
        logger.info("=" * 60)

        try:
            from src.profiling import (
                RuntimeProfiler, MemoryProfiler, FLOPsEstimator, ProfilingContext
            )

            # Get sample batch
            train_data, val_data, test_data = self._get_data_splits()
            sample_batch = train_data[0].to(self.device)

            with ProfilingContext(self.model, seed=self.config.seed):
                # Runtime
                rt_profiler = RuntimeProfiler(self.model, self.device, num_runs=50)
                rt_forward = rt_profiler.profile_forward(sample_batch)

                # Memory
                mem_profiler = MemoryProfiler(self.model, self.device)
                mem_stats = mem_profiler.profile_forward(sample_batch)

                # FLOPs
                flops_est = FLOPsEstimator(self.model)
                flops = flops_est.estimate(
                    sample_batch.num_nodes,
                    sample_batch.edge_index.size(1) // 2
                )

            self.results["profiling"] = {
                "runtime_forward_ms": rt_forward.to_dict(),
                "memory": mem_stats.to_dict(),
                "flops": flops.to_dict(),
                "parameters": flops_est.count_parameters(),
            }

            logger.info(f"Forward: {rt_forward.mean:.2f} ms")
            logger.info(f"Memory: {mem_stats.peak_mb:.2f} MB")

        except Exception as e:
            self.results["errors"].append({
                "phase": "profiling",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            logger.error(f"Profiling failed: {e}")

        return self.results["profiling"]

    def run_all(
        self,
        run_long_range: bool = True,
        run_profiling: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete experiment pipeline.

        Args:
            run_long_range: Whether to run long-range evaluation.
            run_profiling: Whether to run profiling.

        Returns:
            Complete results dictionary.
        """
        try:
            self.setup()
            self.run_training()
            self.run_evaluation()

            if run_long_range:
                self.run_long_range()

            if run_profiling:
                self.run_profiling()

            self.results["status"] = "completed"
            self.results["end_time"] = datetime.now().isoformat()

        except Exception as e:
            self.results["status"] = "failed"
            self.results["end_time"] = datetime.now().isoformat()
            logger.error(f"Experiment failed: {e}")

        finally:
            # Save results
            save_experiment_results(self.results, self.output_dir)
            self.logger.log_metrics(self.results)

        return self.results

    def get_results(self) -> Dict[str, Any]:
        """Get current results."""
        return self.results


def run_experiment(
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run experiment.

    Args:
        config: Experiment configuration.
        output_dir: Output directory.
        **kwargs: Additional runner arguments.

    Returns:
        Experiment results.
    """
    runner = ExperimentRunner(config, output_dir)
    return runner.run_all(**kwargs)


def run_training_only(
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run training only."""
    runner = ExperimentRunner(config, output_dir)
    runner.setup()
    return runner.run_training()


def run_evaluation_only(
    config: ExperimentConfig,
    checkpoint_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation only from checkpoint."""
    runner = ExperimentRunner(config, output_dir)
    runner.setup()
    runner.trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
    return runner.run_evaluation()

