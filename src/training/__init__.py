"""Training and evaluation pipeline for Graph Transformers."""

from .losses import (
    RegressionLoss,
    ClassificationLoss,
    EdgePredictionLoss,
    get_loss_fn,
)
from .metrics import (
    MetricAccumulator,
    RegressionMetrics,
    ClassificationMetrics,
    EdgeMetrics,
    get_metrics,
)
from .optimizer import get_optimizer, get_optimizer_from_config
from .scheduler import (
    CosineWithWarmup,
    WarmupScheduler,
    get_scheduler,
    get_scheduler_from_config,
)
from .trainer import Trainer
from .loops import (
    train_step,
    eval_step,
    run_training_loop,
    run_eval_loop,
    run_inference,
    EarlyStopping,
    GradientAccumulator,
)
from .reproducibility import (
    set_seed,
    get_device,
    log_system_info,
    log_model_info,
    count_parameters,
    ReproducibilityContext,
)


__all__ = [
    # Losses
    "RegressionLoss",
    "ClassificationLoss",
    "EdgePredictionLoss",
    "get_loss_fn",
    # Metrics
    "MetricAccumulator",
    "RegressionMetrics",
    "ClassificationMetrics",
    "EdgeMetrics",
    "get_metrics",
    # Optimizer
    "get_optimizer",
    "get_optimizer_from_config",
    # Scheduler
    "CosineWithWarmup",
    "WarmupScheduler",
    "get_scheduler",
    "get_scheduler_from_config",
    # Trainer
    "Trainer",
    # Loops
    "train_step",
    "eval_step",
    "run_training_loop",
    "run_eval_loop",
    "run_inference",
    "EarlyStopping",
    "GradientAccumulator",
    # Reproducibility
    "set_seed",
    "get_device",
    "log_system_info",
    "log_model_info",
    "count_parameters",
    "ReproducibilityContext",
]
