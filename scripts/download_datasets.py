"""Script to download and prepare datasets."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="dataset")
def main(cfg: DictConfig) -> None:
    """Download and prepare datasets based on configuration."""
    # TODO: Implement dataset downloading logic
    print(f"Downloading dataset: {cfg.name}")
    pass


if __name__ == "__main__":
    main()

