#!/usr/bin/env python
"""Script for generating thesis-ready tables from experiment results."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results import (
    ResultLoader,
    filter_results,
    TableGenerator,
    export_table,
    make_performance_table,
    make_ablation_table,
    make_long_range_table,
    make_efficiency_table,
)
from src.results.utils import ensure_output_dir, save_json


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready tables from experiment results"
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

    # Table selection
    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        default=["performance"],
        choices=["performance", "ablation", "long_range", "efficiency", "all"],
        help="Tables to generate",
    )

    # Ablation options
    parser.add_argument(
        "--ablation_key",
        type=str,
        default="node_pe",
        help="Key for ablation table",
    )

    # Grouping options
    parser.add_argument(
        "--group_by",
        type=str,
        nargs="+",
        default=["dataset", "model", "node_pe", "relative_pe"],
        help="Keys to group by",
    )

    # Metric options
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mae", "loss"],
        help="Metrics to include",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/tables",
        help="Output directory for tables",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        nargs="+",
        default=["latex", "csv"],
        choices=["latex", "csv", "markdown"],
        help="Output format(s)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Table Generation")
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
    generator = TableGenerator()

    # Determine which tables to generate
    tables_to_generate = args.tables
    if "all" in tables_to_generate:
        tables_to_generate = ["performance", "ablation", "long_range", "efficiency"]

    # Generate tables
    for table_type in tables_to_generate:
        logger.info(f"Generating {table_type} table...")

        try:
            if table_type == "performance":
                table = generator.performance_table(
                    results,
                    metrics=args.metrics,
                    group_by=args.group_by,
                )
            elif table_type == "ablation":
                table = generator.ablation_table(
                    results,
                    ablation_key=args.ablation_key,
                    metrics=args.metrics,
                )
            elif table_type == "long_range":
                table = generator.long_range_table(
                    results,
                    metric=args.metrics[0] if args.metrics else "mae",
                )
            elif table_type == "efficiency":
                table = generator.efficiency_table(
                    results,
                    metrics=args.metrics,
                )
            else:
                continue

            # Export in each format
            for fmt in args.output_format:
                filename = f"{args.prefix}{table_type}" if args.prefix else table_type
                if fmt == "latex":
                    output_path = output_dir / f"{filename}.tex"
                elif fmt == "csv":
                    output_path = output_dir / f"{filename}.csv"
                elif fmt == "markdown":
                    output_path = output_dir / f"{filename}.md"

                export_table(table, str(output_path), format=fmt)

            # Also save raw data
            save_json(table, output_dir / f"{filename}_data.json")

        except Exception as e:
            logger.error(f"Failed to generate {table_type} table: {e}")
            continue

    logger.info("=" * 60)
    logger.info(f"Tables saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

