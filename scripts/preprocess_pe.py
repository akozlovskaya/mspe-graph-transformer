"""Script to precompute positional encodings for datasets."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="pe")
def main(cfg: DictConfig) -> None:
    """Precompute positional encodings based on configuration."""
    # TODO: Implement PE precomputation logic
    print(f"Precomputing positional encodings: {cfg.type}")
    pass


if __name__ == "__main__":
    main()

