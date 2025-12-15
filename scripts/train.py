"""Main training script."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Train a model based on configuration."""
    # TODO: Implement training loop
    print(f"Training model: {cfg.model.name}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"PE: {cfg.pe.type}")
    pass


if __name__ == "__main__":
    main()

