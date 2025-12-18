#!/usr/bin/env python
"""
Determinism Verification Script

Runs a small experiment twice and compares outputs to verify determinism.
"""

import argparse
import logging
import sys
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import (
    set_global_seed,
    verify_reproducibility,
    create_reproducibility_info,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_mini_experiment(output_dir: Path, seed: int = 42):
    """
    Run a minimal experiment.

    Args:
        output_dir: Output directory.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    set_global_seed(seed, deterministic=True)

    # Create reproducible data
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.randn(100, 16)
    y = torch.randn(100, 1)

    # Model
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training
    losses = []
    for epoch in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Final predictions
    model.eval()
    with torch.no_grad():
        final_pred = model(x[:10])

    # Results
    results = {
        "seed": seed,
        "final_loss": losses[-1],
        "losses": losses,
        "final_predictions": final_pred.tolist(),
        "model_params_sum": sum(p.sum().item() for p in model.parameters()),
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify experiment determinism"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (temp if not specified)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DETERMINISM VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Tolerance: {args.tolerance}")

    # Setup output
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="determinism_test_"))

    run1_dir = base_dir / "run1"
    run2_dir = base_dir / "run2"

    # Run 1
    logger.info("\nRun 1...")
    results1 = run_mini_experiment(run1_dir, args.seed)
    logger.info(f"  Final loss: {results1['final_loss']:.8f}")

    # Run 2
    logger.info("\nRun 2...")
    results2 = run_mini_experiment(run2_dir, args.seed)
    logger.info(f"  Final loss: {results2['final_loss']:.8f}")

    # Compare
    logger.info("\nComparing results...")

    # Losses
    loss_diff = abs(results1["final_loss"] - results2["final_loss"])
    logger.info(f"  Loss difference: {loss_diff:.2e}")

    # Predictions
    pred1 = np.array(results1["final_predictions"])
    pred2 = np.array(results2["final_predictions"])
    pred_diff = np.abs(pred1 - pred2).max()
    logger.info(f"  Max prediction difference: {pred_diff:.2e}")

    # Parameters
    param_diff = abs(results1["model_params_sum"] - results2["model_params_sum"])
    logger.info(f"  Parameter sum difference: {param_diff:.2e}")

    # Verify
    verification = verify_reproducibility(
        run1_dir, run2_dir, tolerance=args.tolerance
    )

    # Summary
    logger.info("\n" + "=" * 60)

    if verification["identical"]:
        logger.info("✓ DETERMINISM VERIFIED")
        logger.info("  Both runs produced identical results.")
    else:
        logger.error("✗ DETERMINISM FAILED")
        logger.error("  Differences found:")
        for diff in verification["differences"]:
            logger.error(f"    - {diff}")

    logger.info(f"\nOutputs saved to: {base_dir}")
    logger.info("=" * 60)

    return 0 if verification["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())

