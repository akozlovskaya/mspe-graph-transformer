"""Experiment registry for managing experiment configurations."""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    pe: Dict[str, Any]
    training: Dict[str, Any]
    seed: int = 42
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dataset": self.dataset,
            "model": self.model,
            "pe": self.pe,
            "training": self.training,
            "seed": self.seed,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**d)

    def get_id(self) -> str:
        """Generate unique experiment ID based on config."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def with_overrides(self, **overrides) -> "ExperimentConfig":
        """Create copy with overridden values."""
        config_dict = self.to_dict()

        for key, value in overrides.items():
            if "." in key:
                # Nested override: "model.num_layers" -> model["num_layers"]
                parts = key.split(".")
                target = config_dict
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            else:
                config_dict[key] = value

        return ExperimentConfig.from_dict(config_dict)


class ExperimentRegistry:
    """Registry for experiment configurations."""

    def __init__(self):
        """Initialize registry."""
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._validators: List[Callable[[ExperimentConfig], bool]] = []
        self._defaults = self._get_default_configs()

    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations."""
        return {
            "dataset": {
                "name": "zinc",
                "root": "./data",
                "task_type": "regression",
            },
            "model": {
                "name": "graph_transformer",
                "hidden_dim": 256,
                "num_layers": 6,
                "num_heads": 8,
                "dropout": 0.1,
                "ffn_dim": 512,
            },
            "pe": {
                "node": {
                    "type": "combined",
                    "dim": 32,
                    "lap": {"k": 16, "sign_invariant": "flip"},
                    "rwse": {"scales": [1, 2, 4, 8, 16]},
                },
                "relative": {
                    "type": "spd",
                    "num_buckets": 32,
                    "max_distance": 10,
                },
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "lr": 1e-4,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "early_stopping": 20,
            },
        }

    def register(
        self,
        name: str,
        config: Optional[ExperimentConfig] = None,
        **kwargs,
    ) -> ExperimentConfig:
        """
        Register an experiment.

        Args:
            name: Experiment name.
            config: Full ExperimentConfig, or None to build from kwargs.
            **kwargs: Configuration fields if config is None.

        Returns:
            Registered ExperimentConfig.
        """
        if config is None:
            # Merge with defaults
            dataset = {**self._defaults["dataset"], **kwargs.get("dataset", {})}
            model = {**self._defaults["model"], **kwargs.get("model", {})}
            pe = deepcopy(self._defaults["pe"])
            if "pe" in kwargs:
                for key, value in kwargs["pe"].items():
                    if isinstance(value, dict) and key in pe:
                        pe[key].update(value)
                    else:
                        pe[key] = value
            training = {**self._defaults["training"], **kwargs.get("training", {})}

            config = ExperimentConfig(
                name=name,
                dataset=dataset,
                model=model,
                pe=pe,
                training=training,
                seed=kwargs.get("seed", 42),
                tags=kwargs.get("tags", []),
                description=kwargs.get("description", ""),
            )

        # Validate
        if not self.validate(config):
            raise ValueError(f"Invalid experiment config: {name}")

        self._experiments[name] = config
        return config

    def get(self, name: str) -> ExperimentConfig:
        """Get experiment by name."""
        if name not in self._experiments:
            raise KeyError(f"Experiment not found: {name}")
        return deepcopy(self._experiments[name])

    def list(self, tags: Optional[List[str]] = None) -> List[str]:
        """List all registered experiments."""
        if tags is None:
            return list(self._experiments.keys())

        return [
            name
            for name, config in self._experiments.items()
            if any(tag in config.tags for tag in tags)
        ]

    def add_validator(self, validator: Callable[[ExperimentConfig], bool]):
        """Add configuration validator."""
        self._validators.append(validator)

    def validate(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration."""
        # Basic validation
        if not config.name:
            return False

        if not config.dataset.get("name"):
            return False

        if not config.model.get("name"):
            return False

        # Custom validators
        for validator in self._validators:
            if not validator(config):
                return False

        return True

    def create_ablation(
        self,
        base_name: str,
        ablation_name: str,
        **overrides,
    ) -> ExperimentConfig:
        """
        Create ablation experiment from base.

        Args:
            base_name: Name of base experiment.
            ablation_name: Name for ablation experiment.
            **overrides: Configuration overrides.

        Returns:
            New ExperimentConfig.
        """
        base = self.get(base_name)
        ablation = base.with_overrides(**overrides)
        ablation = ExperimentConfig(
            name=ablation_name,
            dataset=ablation.dataset,
            model=ablation.model,
            pe=ablation.pe,
            training=ablation.training,
            seed=ablation.seed,
            tags=ablation.tags + ["ablation"],
            description=f"Ablation of {base_name}",
        )
        return self.register(ablation_name, ablation)


# Global registry instance
_registry = ExperimentRegistry()


def register_experiment(name: str, **kwargs) -> ExperimentConfig:
    """Register experiment in global registry."""
    return _registry.register(name, **kwargs)


def get_experiment(name: str) -> ExperimentConfig:
    """Get experiment from global registry."""
    return _registry.get(name)


def list_experiments(tags: Optional[List[str]] = None) -> List[str]:
    """List experiments in global registry."""
    return _registry.list(tags)


def validate_config(config: ExperimentConfig) -> bool:
    """Validate config using global registry."""
    return _registry.validate(config)


# ============================================================================
# Pre-defined Experiments
# ============================================================================

def register_default_experiments():
    """Register default experiment configurations."""

    # Base experiments
    register_experiment(
        "zinc_baseline",
        dataset={"name": "zinc"},
        model={"name": "graph_transformer", "num_layers": 6},
        pe={"node": {"type": "lap", "dim": 16}, "relative": {"type": "none"}},
        tags=["baseline", "zinc"],
        description="ZINC baseline with LapPE only",
    )

    register_experiment(
        "zinc_mspe",
        dataset={"name": "zinc"},
        model={"name": "graph_transformer", "num_layers": 6},
        pe={
            "node": {"type": "combined", "dim": 32},
            "relative": {"type": "spd", "num_buckets": 32},
        },
        tags=["mspe", "zinc"],
        description="ZINC with multi-scale PE",
    )

    register_experiment(
        "peptides_func_baseline",
        dataset={"name": "peptides_func", "task_type": "multilabel"},
        model={"name": "graph_transformer", "num_layers": 8},
        pe={"node": {"type": "rwse", "dim": 16}, "relative": {"type": "none"}},
        tags=["baseline", "lrgb"],
        description="Peptides-func baseline with RWSE",
    )

    register_experiment(
        "peptides_func_mspe",
        dataset={"name": "peptides_func", "task_type": "multilabel"},
        model={"name": "graph_transformer", "num_layers": 8},
        pe={
            "node": {"type": "combined", "dim": 64},
            "relative": {"type": "spd", "num_buckets": 32},
        },
        tags=["mspe", "lrgb"],
        description="Peptides-func with multi-scale PE",
    )


# Register defaults on import
register_default_experiments()

