"""Tests for training loop components."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses import (
    RegressionLoss,
    ClassificationLoss,
    EdgePredictionLoss,
    get_loss_fn,
)
from src.training.metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    EdgeMetrics,
    get_metrics,
)
from src.training.optimizer import get_optimizer, get_optimizer_from_config
from src.training.scheduler import get_scheduler, CosineWithWarmup
from src.training.trainer import Trainer
from src.training.loops import (
    train_step,
    eval_step,
    run_eval_loop,
    EarlyStopping,
)
from src.training.reproducibility import set_seed, ReproducibilityContext


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleGNN(nn.Module):
        def __init__(self, in_dim=16, hidden_dim=32, out_dim=1):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, out_dim)
            self.relu = nn.ReLU()

        def forward(self, batch):
            x = batch.x
            x = self.relu(self.lin1(x))
            x = self.lin2(x)
            # Simple graph-level readout: mean pooling
            if hasattr(batch, 'batch'):
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, batch.batch)
            else:
                x = x.mean(dim=0, keepdim=True)
            return x

    return SimpleGNN()


@pytest.fixture
def sample_batch():
    """Create a sample batch of graphs."""
    graphs = []
    for i in range(4):
        num_nodes = 10 + i * 2
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.randn(1)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return Batch.from_data_list(graphs)


@pytest.fixture
def classification_batch():
    """Create a sample batch for classification."""
    graphs = []
    for i in range(4):
        num_nodes = 10 + i * 2
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.randint(0, 2, (1,))
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return Batch.from_data_list(graphs)


@pytest.fixture
def toy_dataset():
    """Create a toy dataset for training loop tests."""
    graphs = []
    for i in range(20):
        num_nodes = 10
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        y = x.mean()  # Target is mean of features
        graphs.append(Data(x=x, edge_index=edge_index, y=y.unsqueeze(0)))
    return graphs


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestLossFunctions:
    """Tests for loss functions."""

    def test_regression_mse(self, sample_batch):
        """Test MSE regression loss."""
        loss_fn = RegressionLoss(loss_type="mse")
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_regression_mae(self):
        """Test MAE regression loss."""
        loss_fn = RegressionLoss(loss_type="mae")
        pred = torch.tensor([[1.0], [2.0]])
        target = torch.tensor([[0.0], [0.0]])
        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.tensor(1.5))

    def test_classification_ce(self, classification_batch):
        """Test cross-entropy classification loss."""
        loss_fn = ClassificationLoss(loss_type="cross_entropy", num_classes=2)
        pred = torch.randn(4, 2)
        target = torch.randint(0, 2, (4,))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_classification_bce(self):
        """Test BCE classification loss."""
        loss_fn = ClassificationLoss(loss_type="bce_logits")
        pred = torch.randn(4, 1)
        target = torch.randint(0, 2, (4, 1)).float()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_edge_prediction_loss(self):
        """Test edge prediction loss."""
        loss_fn = EdgePredictionLoss()
        pred = torch.randn(100)
        target = torch.randint(0, 2, (100,)).float()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_get_loss_fn_factory(self):
        """Test loss function factory."""
        reg_loss = get_loss_fn("regression")
        assert isinstance(reg_loss, RegressionLoss)

        cls_loss = get_loss_fn("classification", num_classes=5)
        assert isinstance(cls_loss, ClassificationLoss)

        edge_loss = get_loss_fn("edge_prediction")
        assert isinstance(edge_loss, EdgePredictionLoss)


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Tests for metrics."""

    def test_regression_metrics(self):
        """Test regression metrics computation."""
        metrics = RegressionMetrics()

        # Simulate batches
        for _ in range(3):
            pred = torch.randn(8, 1)
            target = pred + torch.randn(8, 1) * 0.1  # Add noise
            metrics.update(pred, target)

        result = metrics.compute()
        assert "mse" in result
        assert "mae" in result
        assert "rmse" in result
        assert all(v >= 0 for v in result.values())

    def test_classification_metrics_binary(self):
        """Test binary classification metrics."""
        metrics = ClassificationMetrics(num_classes=2, task="binary")

        for _ in range(3):
            pred = torch.randn(8, 1)
            target = torch.randint(0, 2, (8,))
            metrics.update(pred, target)

        result = metrics.compute()
        assert "accuracy" in result
        assert "roc_auc" in result
        assert 0 <= result["accuracy"] <= 1

    def test_classification_metrics_multiclass(self):
        """Test multiclass classification metrics."""
        metrics = ClassificationMetrics(num_classes=5, task="multiclass")

        for _ in range(3):
            pred = torch.randn(8, 5)
            target = torch.randint(0, 5, (8,))
            metrics.update(pred, target)

        result = metrics.compute()
        assert "accuracy" in result
        assert "f1_macro" in result

    def test_edge_metrics(self):
        """Test edge prediction metrics."""
        metrics = EdgeMetrics()

        for _ in range(3):
            pred = torch.randn(100)
            target = torch.randint(0, 2, (100,))
            metrics.update(pred, target)

        result = metrics.compute()
        assert "roc_auc" in result
        assert "ap" in result
        assert "precision@100" in result

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = RegressionMetrics()

        pred = torch.randn(8, 1)
        target = torch.randn(8, 1)
        metrics.update(pred, target)

        metrics.reset()
        assert len(metrics.accumulator.predictions) == 0


# ============================================================================
# Optimizer Tests
# ============================================================================

class TestOptimizer:
    """Tests for optimizer configuration."""

    def test_adamw_optimizer(self, simple_model):
        """Test AdamW optimizer creation."""
        optimizer = get_optimizer(simple_model, optimizer_type="adamw", lr=1e-3)
        assert isinstance(optimizer, torch.optim.AdamW)
        # Check param groups (decay and no_decay)
        assert len(optimizer.param_groups) == 2

    def test_adam_optimizer(self, simple_model):
        """Test Adam optimizer creation."""
        optimizer = get_optimizer(simple_model, optimizer_type="adam", lr=1e-3)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_sgd_optimizer(self, simple_model):
        """Test SGD optimizer creation."""
        optimizer = get_optimizer(simple_model, optimizer_type="sgd", lr=1e-2, momentum=0.9)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_optimizer_from_config(self, simple_model):
        """Test optimizer creation from config dict."""
        config = {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        }
        optimizer = get_optimizer_from_config(simple_model, config)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4


# ============================================================================
# Scheduler Tests
# ============================================================================

class TestScheduler:
    """Tests for learning rate schedulers."""

    def test_cosine_warmup_scheduler(self, simple_model):
        """Test cosine scheduler with warmup."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        scheduler = CosineWithWarmup(
            optimizer,
            warmup_epochs=5,
            total_epochs=100,
            min_lr=1e-6,
        )

        lrs = []
        for epoch in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Check warmup (LR should increase)
        assert lrs[1] > lrs[0]
        assert lrs[4] > lrs[1]

        # Check decay (LR should decrease after warmup)
        assert lrs[50] < lrs[10]
        assert lrs[99] >= 1e-6

    def test_step_scheduler(self, simple_model):
        """Test step LR scheduler."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="step",
            step_size=10,
            gamma=0.1,
        )

        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            scheduler.step()

        # LR should decrease by factor of gamma
        assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.1)


# ============================================================================
# Training Loop Tests
# ============================================================================

class TestTrainingLoop:
    """Tests for training loop functionality."""

    def test_single_train_step(self, simple_model, sample_batch):
        """Test single training step executes without error."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        loss_fn = RegressionLoss(loss_type="mse")
        device = torch.device("cpu")

        result = train_step(
            model=simple_model,
            batch=sample_batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        assert "loss" in result
        assert result["loss"] >= 0

    def test_train_step_with_grad_clip(self, simple_model, sample_batch):
        """Test training step with gradient clipping."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        loss_fn = RegressionLoss(loss_type="mse")
        device = torch.device("cpu")

        result = train_step(
            model=simple_model,
            batch=sample_batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=1.0,
        )

        assert "grad_norm" in result
        assert result["grad_norm"] <= 1.0 + 1e-5  # Allow small numerical error

    def test_eval_step_no_grad(self, simple_model, sample_batch):
        """Test evaluation step doesn't compute gradients."""
        loss_fn = RegressionLoss(loss_type="mse")
        device = torch.device("cpu")

        result = eval_step(
            model=simple_model,
            batch=sample_batch,
            loss_fn=loss_fn,
            device=device,
        )

        assert "loss" in result
        assert "pred" in result
        assert "target" in result
        assert not result["pred"].requires_grad

    def test_loss_decreases(self, simple_model, toy_dataset):
        """Test that loss decreases during training."""
        set_seed(42)

        optimizer = get_optimizer(simple_model, lr=1e-2)
        loss_fn = RegressionLoss(loss_type="mse")
        device = torch.device("cpu")

        loader = DataLoader(toy_dataset, batch_size=4, shuffle=True)

        # Initial loss
        initial_loss = 0
        for batch in loader:
            result = eval_step(simple_model, batch, loss_fn, device)
            initial_loss += result["loss"]
        initial_loss /= len(loader)

        # Train for a few epochs
        for _ in range(10):
            for batch in loader:
                train_step(simple_model, batch, optimizer, loss_fn, device)

        # Final loss
        final_loss = 0
        for batch in loader:
            result = eval_step(simple_model, batch, loss_fn, device)
            final_loss += result["loss"]
        final_loss /= len(loader)

        # Loss should decrease
        assert final_loss < initial_loss


# ============================================================================
# Trainer Tests
# ============================================================================

class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_init(self, simple_model):
        """Test Trainer initialization."""
        optimizer = get_optimizer(simple_model, lr=1e-3)

        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            task_type="regression",
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.metrics is not None

    def test_trainer_train_epoch(self, simple_model, toy_dataset):
        """Test Trainer train_epoch method."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            task_type="regression",
            log_every=100,  # Reduce logging
        )

        loader = DataLoader(toy_dataset, batch_size=4, shuffle=True)
        metrics = trainer.train_epoch(loader)

        assert "loss" in metrics
        assert "time" in metrics
        assert metrics["loss"] >= 0

    def test_trainer_eval_epoch(self, simple_model, toy_dataset):
        """Test Trainer eval_epoch method."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            task_type="regression",
        )

        loader = DataLoader(toy_dataset, batch_size=4, shuffle=False)
        metrics = trainer.eval_epoch(loader)

        assert "loss" in metrics
        assert "mse" in metrics
        assert "mae" in metrics

    def test_trainer_checkpoint(self, simple_model, tmp_path):
        """Test checkpoint save and load."""
        optimizer = get_optimizer(simple_model, lr=1e-3)
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            task_type="regression",
            checkpoint_dir=str(tmp_path),
        )

        # Save checkpoint
        trainer.current_epoch = 5
        trainer.best_metric = 0.1
        trainer.save_checkpoint("test.pt", {"loss": 0.5})

        # Verify file exists
        assert (tmp_path / "test.pt").exists()

        # Load checkpoint
        loaded_metrics = trainer.load_checkpoint(str(tmp_path / "test.pt"))
        assert trainer.current_epoch == 5
        assert trainer.best_metric == 0.1
        assert loaded_metrics["loss"] == 0.5


# ============================================================================
# Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_seed_determinism(self, simple_model, toy_dataset):
        """Test that training is deterministic with fixed seed."""
        loader = DataLoader(toy_dataset, batch_size=4, shuffle=True)
        loss_fn = RegressionLoss(loss_type="mse")
        device = torch.device("cpu")

        def train_once(seed):
            set_seed(seed)
            model = type(simple_model)()  # Create new instance
            optimizer = get_optimizer(model, lr=1e-3)

            losses = []
            for batch in loader:
                result = train_step(model, batch, optimizer, loss_fn, device)
                losses.append(result["loss"])

            return losses

        losses1 = train_once(42)
        losses2 = train_once(42)

        # Same seed should give same results
        assert losses1 == pytest.approx(losses2, rel=1e-5)

    def test_reproducibility_context(self):
        """Test ReproducibilityContext manager."""
        # Generate random values outside context
        initial = torch.randn(5)

        with ReproducibilityContext(seed=123):
            inside1 = torch.randn(5)

        with ReproducibilityContext(seed=123):
            inside2 = torch.randn(5)

        # Values inside context with same seed should be equal
        assert torch.allclose(inside1, inside2)


# ============================================================================
# Early Stopping Tests
# ============================================================================

class TestEarlyStopping:
    """Tests for early stopping utility."""

    def test_early_stopping_triggers(self):
        """Test early stopping triggers after patience."""
        early_stop = EarlyStopping(patience=3, mode="min")

        scores = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]  # No improvement after 3
        for i, score in enumerate(scores):
            should_stop = early_stop(score)
            if i < 5:
                assert not should_stop
            else:
                assert should_stop

    def test_early_stopping_no_trigger(self):
        """Test early stopping doesn't trigger with improvements."""
        early_stop = EarlyStopping(patience=3, mode="min")

        scores = [1.0, 0.9, 0.8, 0.7, 0.6]  # Continuous improvement
        for score in scores:
            should_stop = early_stop(score)
            assert not should_stop

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        early_stop = EarlyStopping(patience=2, mode="max")

        scores = [0.5, 0.6, 0.65, 0.64, 0.63]  # Improvement then plateau
        results = [early_stop(s) for s in scores]

        assert results[-1] == True  # Should trigger at the end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

