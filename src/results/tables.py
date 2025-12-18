"""Table generation for thesis-ready outputs."""

import csv
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from io import StringIO

from .loader import ExperimentResult
from .aggregate import AggregatedResult, ResultAggregator, aggregate_by_seed
from .formatting import TableFormatter, METRIC_FORMATS, PE_NAMES, DATASET_NAMES


logger = logging.getLogger(__name__)


class TableGenerator:
    """Generator for thesis-ready tables."""

    def __init__(
        self,
        formatter: Optional[TableFormatter] = None,
    ):
        """
        Initialize table generator.

        Args:
            formatter: Table formatter instance.
        """
        self.formatter = formatter or TableFormatter()

    def performance_table(
        self,
        results: Union[List[ExperimentResult], List[AggregatedResult]],
        metrics: List[str] = None,
        group_by: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate main performance table.

        Args:
            results: Experiment results or aggregated results.
            metrics: Metrics to include.
            group_by: Grouping keys.

        Returns:
            Table data structure.
        """
        # Aggregate if needed
        if results and isinstance(results[0], ExperimentResult):
            if group_by is None:
                group_by = ["dataset", "model", "node_pe", "relative_pe"]
            aggregated = aggregate_by_seed(results, group_by)
        else:
            aggregated = results

        if metrics is None:
            metrics = ["mae", "loss", "roc_auc", "ap"]

        # Build table
        rows = []
        for agg in aggregated:
            row = {
                "Dataset": DATASET_NAMES.get(agg.group_values.get("dataset", ""), ""),
                "Model": agg.group_values.get("model", ""),
                "Node PE": PE_NAMES.get(agg.group_values.get("node_pe", ""), ""),
                "Rel. PE": PE_NAMES.get(agg.group_values.get("relative_pe", ""), ""),
                "Seeds": agg.n_experiments,
            }

            for metric in metrics:
                if metric in agg.metrics:
                    row[self.formatter.metric_name(metric)] = agg.format_metric(
                        metric, precision=self.formatter.get_precision(metric)
                    )
                else:
                    row[self.formatter.metric_name(metric)] = "—"

            rows.append(row)

        # Get columns
        base_cols = ["Dataset", "Model", "Node PE", "Rel. PE", "Seeds"]
        metric_cols = [self.formatter.metric_name(m) for m in metrics]

        return {
            "columns": base_cols + metric_cols,
            "rows": rows,
            "caption": "Model performance comparison",
            "label": "tab:performance",
        }

    def ablation_table(
        self,
        results: List[ExperimentResult],
        ablation_key: str,
        metrics: List[str] = None,
        base_group: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate PE ablation table.

        Args:
            results: Experiment results.
            ablation_key: Key being ablated (e.g., "node_pe").
            metrics: Metrics to include.
            base_group: Base grouping keys.

        Returns:
            Table data structure.
        """
        if base_group is None:
            base_group = ["dataset"]
        if metrics is None:
            metrics = ["mae", "loss"]

        # Group by ablation key
        group_keys = base_group + [ablation_key]
        aggregated = aggregate_by_seed(results, group_keys)

        # Organize by base group then ablation value
        table_data = {}
        ablation_values = set()

        for agg in aggregated:
            base_key = tuple(agg.group_values.get(k, "") for k in base_group)
            abl_value = agg.group_values.get(ablation_key, "")
            ablation_values.add(abl_value)

            if base_key not in table_data:
                table_data[base_key] = {}
            table_data[base_key][abl_value] = agg

        ablation_values = sorted(ablation_values)

        # Build rows
        rows = []
        for base_key, abl_data in sorted(table_data.items()):
            row = {k: v for k, v in zip(base_group, base_key)}

            for abl_value in ablation_values:
                agg = abl_data.get(abl_value)
                if agg:
                    for metric in metrics:
                        col_name = f"{PE_NAMES.get(abl_value, abl_value)} {self.formatter.metric_name(metric)}"
                        row[col_name] = agg.format_metric(
                            metric, precision=self.formatter.get_precision(metric)
                        )
                else:
                    for metric in metrics:
                        col_name = f"{PE_NAMES.get(abl_value, abl_value)} {self.formatter.metric_name(metric)}"
                        row[col_name] = "—"

            rows.append(row)

        # Columns
        base_cols = [c.capitalize() for c in base_group]
        metric_cols = []
        for abl_value in ablation_values:
            for metric in metrics:
                metric_cols.append(f"{PE_NAMES.get(abl_value, abl_value)} {self.formatter.metric_name(metric)}")

        return {
            "columns": base_cols + metric_cols,
            "rows": rows,
            "caption": f"Ablation over {ablation_key.replace('_', ' ')}",
            "label": f"tab:ablation_{ablation_key}",
        }

    def long_range_table(
        self,
        results: List[ExperimentResult],
        distances: List[int] = None,
        metric: str = "mae",
    ) -> Dict[str, Any]:
        """
        Generate long-range performance table.

        Args:
            results: Experiment results.
            distances: Distance buckets to include.
            metric: Metric to report per distance.

        Returns:
            Table data structure.
        """
        if distances is None:
            distances = [1, 2, 3, 5, 10, 15, 20]

        aggregated = aggregate_by_seed(results, ["dataset", "node_pe", "relative_pe"])

        rows = []
        for agg in aggregated:
            row = {
                "Dataset": DATASET_NAMES.get(agg.group_values.get("dataset", ""), ""),
                "Node PE": PE_NAMES.get(agg.group_values.get("node_pe", ""), ""),
                "Rel. PE": PE_NAMES.get(agg.group_values.get("relative_pe", ""), ""),
            }

            # Add LR summary
            if agg.long_range:
                auc = agg.long_range.get("auc", {})
                if auc:
                    row["LR-AUC"] = f"{auc.get('mean', 0):.3f}"
                else:
                    row["LR-AUC"] = "—"
            else:
                row["LR-AUC"] = "—"

            # Add per-distance metrics
            for dist in distances:
                dist_key = str(dist) if isinstance(list(agg.distance_metrics.keys())[0] if agg.distance_metrics else 0, str) else dist
                if dist_key in agg.distance_metrics:
                    dist_data = agg.distance_metrics[dist_key]
                    if metric in dist_data:
                        val = dist_data[metric].get("mean", 0)
                        row[f"d={dist}"] = f"{val:.3f}"
                    else:
                        row[f"d={dist}"] = "—"
                else:
                    row[f"d={dist}"] = "—"

            rows.append(row)

        # Columns
        base_cols = ["Dataset", "Node PE", "Rel. PE", "LR-AUC"]
        dist_cols = [f"d={d}" for d in distances]

        return {
            "columns": base_cols + dist_cols,
            "rows": rows,
            "caption": f"Long-range performance ({metric.upper()}) by distance",
            "label": "tab:long_range",
        }

    def efficiency_table(
        self,
        results: List[ExperimentResult],
        metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate efficiency vs accuracy trade-off table.

        Args:
            results: Experiment results.
            metrics: Accuracy metrics to include.

        Returns:
            Table data structure.
        """
        if metrics is None:
            metrics = ["mae"]

        aggregated = aggregate_by_seed(
            results, ["dataset", "model", "node_pe", "relative_pe"]
        )

        rows = []
        for agg in aggregated:
            row = {
                "Model": agg.group_values.get("model", ""),
                "Node PE": PE_NAMES.get(agg.group_values.get("node_pe", ""), ""),
                "Rel. PE": PE_NAMES.get(agg.group_values.get("relative_pe", ""), ""),
            }

            # Add metrics
            for metric in metrics:
                row[self.formatter.metric_name(metric)] = agg.format_metric(
                    metric, precision=self.formatter.get_precision(metric)
                )

            # Add efficiency metrics
            if agg.parameters:
                row["Params (M)"] = f"{agg.parameters / 1e6:.2f}"
            else:
                row["Params (M)"] = "—"

            if agg.runtime_ms:
                row["Time (ms)"] = f"{agg.runtime_ms['mean']:.1f}"
            else:
                row["Time (ms)"] = "—"

            if agg.memory_mb:
                row["Memory (MB)"] = f"{agg.memory_mb['mean']:.0f}"
            else:
                row["Memory (MB)"] = "—"

            rows.append(row)

        # Sort by primary metric
        if metrics and rows:
            primary = self.formatter.metric_name(metrics[0])
            try:
                rows.sort(key=lambda r: float(r[primary].split("±")[0]) if r[primary] != "—" else float("inf"))
            except (ValueError, KeyError):
                pass

        # Columns
        base_cols = ["Model", "Node PE", "Rel. PE"]
        metric_cols = [self.formatter.metric_name(m) for m in metrics]
        eff_cols = ["Params (M)", "Time (ms)", "Memory (MB)"]

        return {
            "columns": base_cols + metric_cols + eff_cols,
            "rows": rows,
            "caption": "Efficiency vs accuracy trade-off",
            "label": "tab:efficiency",
        }


def make_performance_table(
    results: List[ExperimentResult],
    **kwargs,
) -> Dict[str, Any]:
    """Generate main performance table."""
    gen = TableGenerator()
    return gen.performance_table(results, **kwargs)


def make_ablation_table(
    results: List[ExperimentResult],
    ablation_key: str,
    **kwargs,
) -> Dict[str, Any]:
    """Generate ablation table."""
    gen = TableGenerator()
    return gen.ablation_table(results, ablation_key, **kwargs)


def make_long_range_table(
    results: List[ExperimentResult],
    **kwargs,
) -> Dict[str, Any]:
    """Generate long-range performance table."""
    gen = TableGenerator()
    return gen.long_range_table(results, **kwargs)


def make_efficiency_table(
    results: List[ExperimentResult],
    **kwargs,
) -> Dict[str, Any]:
    """Generate efficiency table."""
    gen = TableGenerator()
    return gen.efficiency_table(results, **kwargs)


def export_table(
    table: Dict[str, Any],
    output_path: str,
    format: str = "latex",
) -> str:
    """
    Export table to file.

    Args:
        table: Table data from generator.
        output_path: Output file path.
        format: Output format ('latex', 'csv', 'markdown').

    Returns:
        Formatted table string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "latex":
        content = _table_to_latex(table)
    elif format == "csv":
        content = _table_to_csv(table)
    elif format == "markdown":
        content = _table_to_markdown(table)
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Exported table to {output_path}")
    return content


def _table_to_latex(table: Dict[str, Any]) -> str:
    """Convert table to LaTeX format."""
    columns = table["columns"]
    rows = table["rows"]
    caption = table.get("caption", "")
    label = table.get("label", "tab:table")

    # Column specification
    col_spec = "l" * len(columns)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header
    header = " & ".join(_latex_escape(c) for c in columns) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Rows
    for row in rows:
        row_vals = [_latex_escape(str(row.get(c, "—"))) for c in columns]
        lines.append(" & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def _table_to_csv(table: Dict[str, Any]) -> str:
    """Convert table to CSV format."""
    output = StringIO()
    columns = table["columns"]
    rows = table["rows"]

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return output.getvalue()


def _table_to_markdown(table: Dict[str, Any]) -> str:
    """Convert table to Markdown format."""
    columns = table["columns"]
    rows = table["rows"]

    lines = []

    # Header
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    # Rows
    for row in rows:
        row_vals = [str(row.get(c, "—")) for c in columns]
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "$": "\\$",
        "{": "\\{",
        "}": "\\}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text

