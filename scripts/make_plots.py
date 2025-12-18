#!/usr/bin/env python
"""Script for generating thesis-ready plots from experiment results."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results import (
    ResultLoader,
    filter_results,
    PlotGenerator,
    save_figure,
    plot_performance_vs_distance,
    plot_performance_vs_size,
    plot_runtime_vs_accuracy,
    plot_memory_vs_accuracy,
    plot_depth_vs_distance,
)
from src.results.utils import ensure_output_dir
from src.results.formatting import PlotStyle


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


AVAILABLE_PLOTS = [
    "performance_vs_distance",
    "performance_vs_size",
    "runtime_vs_accuracy",
    "memory_vs_accuracy",
    "depth_vs_distance",
    "ablation_heatmap",
]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready plots from experiment results"
    )

    # Input options
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./outputs",
        help="Directory containing experiment outputs",
    )

    # Filtering options
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Filter by dataset(s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="Filter by model(s)",
    )
    parser.add_argument(
        "--node_pe",
        type=str,
        nargs="+",
        help="Filter by node PE type(s)",
    )
    parser.add_argument(
        "--complete_only",
        action="store_true",
        help="Only include completed experiments",
    )

    # Plot selection
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=["performance_vs_distance"],
        choices=AVAILABLE_PLOTS + ["all"],
        help="Plots to generate",
    )

    # Plot options
    parser.add_argument(
        "--metric",
        type=str,
        default="mae",
        help="Primary metric for plots",
    )
    parser.add_argument(
        "--max_distance",
        type=int,
        default=20,
        help="Maximum distance for distance plots",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="node_pe",
        help="Key to group curves by",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output format(s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames",
    )

    # Style options
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8, 6],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=11,
        help="Base font size",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Plot Generation")
    logger.info("=" * 60)

    # Load results
    loader = ResultLoader(args.results_dir)
    results = loader.load_all()
    logger.info(f"Loaded {len(results)} experiments")

    if not results:
        logger.warning("No experiments found!")
        return

    # Apply filters
    if args.dataset:
        results = [r for r in results if r.dataset in args.dataset]
    if args.model:
        results = [r for r in results if r.model in args.model]
    if args.node_pe:
        results = [r for r in results if r.node_pe_type in args.node_pe]
    if args.complete_only:
        results = [r for r in results if r.is_complete()]

    logger.info(f"After filtering: {len(results)} experiments")

    if not results:
        logger.warning("No experiments after filtering!")
        return

    # Setup output
    output_dir = ensure_output_dir(args.output_dir)

    # Create style
    style = PlotStyle(
        figure_size=tuple(args.figsize),
        font_size=args.font_size,
        dpi=args.dpi,
    )
    generator = PlotGenerator(style)

    # Determine which plots to generate
    plots_to_generate = args.plots
    if "all" in plots_to_generate:
        plots_to_generate = AVAILABLE_PLOTS

    # Generate plots
    for plot_type in plots_to_generate:
        logger.info(f"Generating {plot_type} plot...")

        try:
            if plot_type == "performance_vs_distance":
                fig = generator.performance_vs_distance(
                    results,
                    metric=args.metric,
                    max_distance=args.max_distance,
                    group_by=args.group_by,
                )
            elif plot_type == "performance_vs_size":
                fig = generator.performance_vs_size(
                    results,
                    metric=args.metric,
                    size_metric="parameters",
                    group_by="model",
                )
            elif plot_type == "runtime_vs_accuracy":
                fig = generator.runtime_vs_accuracy(
                    results,
                    accuracy_metric=args.metric,
                )
            elif plot_type == "memory_vs_accuracy":
                fig = generator.memory_vs_accuracy(
                    results,
                    accuracy_metric=args.metric,
                )
            elif plot_type == "depth_vs_distance":
                fig = generator.depth_vs_distance(
                    results,
                    metric=args.metric,
                )
            elif plot_type == "ablation_heatmap":
                fig = generator.ablation_heatmap(
                    results,
                    row_key="node_pe",
                    col_key="relative_pe",
                    metric=args.metric,
                )
            else:
                continue

            # Save figure
            filename = f"{args.prefix}{plot_type}" if args.prefix else plot_type
            output_path = output_dir / filename
            saved = save_figure(fig, str(output_path), formats=args.formats, dpi=args.dpi)
            logger.info(f"Saved: {saved}")

        except Exception as e:
            logger.error(f"Failed to generate {plot_type} plot: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("=" * 60)
    logger.info(f"Figures saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

