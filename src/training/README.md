# Training Pipeline

Module for training and evaluating Graph Transformers with multi-scale positional encodings.

## Structure

```
src/training/
├── __init__.py         # Public API of the module
├── trainer.py          # Main Trainer class
├── optimizer.py        # Optimizer configuration
├── scheduler.py        # Learning rate schedulers
├── losses.py           # Loss functions
├── metrics.py          # Evaluation metrics
├── loops.py            # Training/eval loops
├── reproducibility.py  # Reproducibility utilities
└── README.md           # Documentation
```

## Quick Start

### Basic Training

```python
from src.training import Trainer, get_optimizer, get_scheduler, get_loss_fn, get_metrics, set_seed
from src.dataset import get_dataset
from src.models import get_model
from torch_geometric.loader import DataLoader

# Set seed for reproducibility
set_seed(42)

# Load dataset
dataset = get_dataset(name="zinc", root="data/")
train_data, val_data, test_data = dataset.get_splits()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Create model
model = get_model(
    name="graph_transformer",
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    hidden_dim=256,
    num_layers=6,
)

# Optimizer and scheduler
optimizer = get_optimizer(model, optimizer_type="adamw", lr=1e-4)
scheduler = get_scheduler(optimizer, scheduler_type="cosine_warmup", total_epochs=100)

# Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    task_type="regression",
    checkpoint_dir="checkpoints/",
)

# Training
history = trainer.fit(train_loader, val_loader, epochs=100)
```

### Using Scripts

```bash
# Training
python scripts/train.py dataset=zinc model=graph_transformer pe=default

# Evaluation
python scripts/evaluate.py checkpoint=logs/experiment/checkpoints/best.pt
```

## Trainer API

### Initialization

```python
trainer = Trainer(
    model=model,                    # PyTorch model
    optimizer=optimizer,            # Optimizer
    scheduler=scheduler,            # LR scheduler (optional)
    loss_fn=loss_fn,               # Loss function (optional, auto-inferred)
    metrics=metrics,               # Metrics (optional, auto-inferred)
    device="cuda",                 # Device
    task_type="regression",        # Task type
    grad_clip=1.0,                 # Gradient clipping
    mixed_precision=False,         # AMP training
    log_every=50,                  # Logging frequency
    checkpoint_dir="checkpoints/", # Checkpoint directory
)
```

### Methods

```python
# One training epoch
train_metrics = trainer.train_epoch(train_loader)

# One validation epoch
val_metrics = trainer.eval_epoch(val_loader)

# Full training
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    metric_for_best="loss",        # Metric for best model
    minimize_metric=True,          # Minimize metric
    early_stopping=20,             # Early stopping patience
)

# Checkpoints
trainer.save_checkpoint("model.pt")
trainer.load_checkpoint("model.pt")

# Predictions
predictions = trainer.predict(test_loader)
```

## Loss Functions

```python
from src.training import get_loss_fn

# Regression
loss_fn = get_loss_fn("regression", loss_type="mse")  # or "mae", "huber"

# Classification
loss_fn = get_loss_fn("classification", num_classes=10, loss_type="cross_entropy")
loss_fn = get_loss_fn("classification", num_classes=2, loss_type="bce_logits")

# Edge prediction
loss_fn = get_loss_fn("edge_prediction")
```

## Metrics

```python
from src.training import get_metrics

# Regression metrics: mse, mae, rmse
metrics = get_metrics("regression")

# Classification metrics: accuracy, f1, roc_auc, ap
metrics = get_metrics("classification", num_classes=2)

# Edge metrics: roc_auc, ap, precision@k
metrics = get_metrics("edge_prediction")
```

## Optimizer and Scheduler

### Optimizers

```python
from src.training import get_optimizer

optimizer = get_optimizer(
    model,
    optimizer_type="adamw",  # "adam", "adamw", "sgd"
    lr=1e-4,
    weight_decay=0.01,
)
```

### Schedulers

```python
from src.training import get_scheduler

# Cosine decay with warmup
scheduler = get_scheduler(
    optimizer,
    scheduler_type="cosine_warmup",
    total_epochs=100,
    warmup_epochs=10,
    min_lr=1e-6,
)

# Step LR
scheduler = get_scheduler(
    optimizer,
    scheduler_type="step",
    step_size=30,
    gamma=0.1,
)
```

## Reproducibility

```python
from src.training import set_seed, ReproducibilityContext

# Set seed
set_seed(42, deterministic=True)

# Reproducibility context
with ReproducibilityContext(seed=123):
    # Code inside will be reproducible
    result = train_model(...)
```

## Checkpointing

Checkpoint format:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": int,
    "global_step": int,
    "best_metric": float,
    "best_epoch": int,
    "metrics": dict,
}
```

## Logging Hooks

Support for TensorBoard and Weights & Biases:

```python
# TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment")

def tensorboard_hook(metrics, step, prefix):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)

trainer.add_log_hook(tensorboard_hook)

# Weights & Biases
import wandb

wandb.init(project="mspe")

def wandb_hook(metrics, step, prefix):
    wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)

trainer.add_log_hook(wandb_hook)
```

## Configuration via Hydra

```yaml
# configs/config.yaml
training:
  epochs: 100
  batch_size: 32
  grad_clip: 1.0
  mixed_precision: false
  
  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 0.01
    
  scheduler:
    type: cosine_warmup
    warmup_epochs: 10
    min_lr: 1e-6
```

## Useful Utilities

```python
from src.training import (
    EarlyStopping,
    GradientAccumulator,
    get_device,
    log_model_info,
)

# Early stopping
early_stop = EarlyStopping(patience=10, mode="min")
if early_stop(val_loss):
    break

# Gradient accumulation (for large batch sizes)
accumulator = GradientAccumulator(accumulation_steps=4)

# Device selection
device = get_device()  # Automatically selects CUDA if available

# Model info
log_model_info(model)  # Logs number of parameters
```
