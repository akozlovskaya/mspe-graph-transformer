"""Evaluation script."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    # TODO: Implement evaluation logic
    print(f"Evaluating checkpoint: {cfg.checkpoint}")
    pass


if __name__ == "__main__":
    main()

