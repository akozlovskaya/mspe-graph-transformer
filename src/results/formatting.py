"""Formatting and styling definitions for thesis outputs."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# Naming Conventions
# ============================================================================

PE_NAMES: Dict[str, str] = {
    # Node PE
    "none": "None",
    "lap": "LapPE",
    "rwse": "RWSE",
    "hks": "HKS",
    "role": "Role",
    "combined": "MSPE",
    # Relative PE
    "spd": "SPD",
    "bfs": "BFS",
    "diffusion": "Diff.",
    "resistance": "Resist.",
    "landmark": "Landmark",
}

DATASET_NAMES: Dict[str, str] = {
    "zinc": "ZINC",
    "qm9": "QM9",
    "peptides_func": "Peptides-func",
    "peptides_struct": "Peptides-struct",
    "pascalvoc_sp": "PascalVOC-SP",
    "cifar10_sp": "CIFAR10-SP",
    "ogbg_molhiv": "ogbg-molhiv",
    "ogbg_molpcba": "ogbg-molpcba",
    "pcqm4m": "PCQM4M",
    "pcqm_contact": "PCQM-Contact",
}

MODEL_NAMES: Dict[str, str] = {
    "graph_transformer": "GT",
    "gps": "GPS",
    "san": "SAN",
    "graphormer": "Graphormer",
    "gin": "GIN",
    "gcn": "GCN",
    "gat": "GAT",
    "mpnn": "MPNN",
}

METRIC_NAMES: Dict[str, str] = {
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "loss": "Loss",
    "accuracy": "Acc.",
    "roc_auc": "ROC-AUC",
    "ap": "AP",
    "precision_at_k": "P@k",
    "f1": "F1",
}


# ============================================================================
# Metric Formatting
# ============================================================================

@dataclass
class MetricFormat:
    """Format specification for a metric."""

    name: str
    precision: int = 4
    multiply: float = 1.0
    unit: str = ""
    lower_is_better: bool = True


METRIC_FORMATS: Dict[str, MetricFormat] = {
    "mae": MetricFormat("MAE", precision=4, lower_is_better=True),
    "mse": MetricFormat("MSE", precision=4, lower_is_better=True),
    "rmse": MetricFormat("RMSE", precision=4, lower_is_better=True),
    "loss": MetricFormat("Loss", precision=4, lower_is_better=True),
    "accuracy": MetricFormat("Accuracy", precision=2, multiply=100, unit="%", lower_is_better=False),
    "roc_auc": MetricFormat("ROC-AUC", precision=3, lower_is_better=False),
    "ap": MetricFormat("AP", precision=3, lower_is_better=False),
    "precision": MetricFormat("Precision", precision=3, lower_is_better=False),
    "recall": MetricFormat("Recall", precision=3, lower_is_better=False),
    "f1": MetricFormat("F1", precision=3, lower_is_better=False),
}


# ============================================================================
# Table Formatter
# ============================================================================

class TableFormatter:
    """Formatter for thesis tables."""

    def __init__(
        self,
        metric_formats: Optional[Dict[str, MetricFormat]] = None,
        pe_names: Optional[Dict[str, str]] = None,
        dataset_names: Optional[Dict[str, str]] = None,
        model_names: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize table formatter.

        Args:
            metric_formats: Custom metric formats.
            pe_names: Custom PE names.
            dataset_names: Custom dataset names.
            model_names: Custom model names.
        """
        self.metric_formats = metric_formats or METRIC_FORMATS
        self.pe_names = pe_names or PE_NAMES
        self.dataset_names = dataset_names or DATASET_NAMES
        self.model_names = model_names or MODEL_NAMES

    def metric_name(self, metric: str) -> str:
        """Get display name for metric."""
        if metric in self.metric_formats:
            return self.metric_formats[metric].name
        return METRIC_NAMES.get(metric, metric.upper())

    def get_precision(self, metric: str) -> int:
        """Get precision for metric."""
        if metric in self.metric_formats:
            return self.metric_formats[metric].precision
        return 4

    def format_value(
        self,
        value: float,
        metric: str,
        include_unit: bool = True,
    ) -> str:
        """
        Format a metric value.

        Args:
            value: Raw value.
            metric: Metric name.
            include_unit: Whether to include unit.

        Returns:
            Formatted string.
        """
        if metric in self.metric_formats:
            fmt = self.metric_formats[metric]
            scaled = value * fmt.multiply
            formatted = f"{scaled:.{fmt.precision}f}"
            if include_unit and fmt.unit:
                formatted += fmt.unit
            return formatted

        return f"{value:.4f}"

    def format_mean_std(
        self,
        mean: float,
        std: float,
        metric: str,
    ) -> str:
        """
        Format mean ± std.

        Args:
            mean: Mean value.
            std: Standard deviation.
            metric: Metric name.

        Returns:
            Formatted string.
        """
        precision = self.get_precision(metric)

        if metric in self.metric_formats:
            fmt = self.metric_formats[metric]
            mean_scaled = mean * fmt.multiply
            std_scaled = std * fmt.multiply

            if std_scaled > 0:
                return f"{mean_scaled:.{precision}f} ± {std_scaled:.{precision}f}"
            return f"{mean_scaled:.{precision}f}"

        if std > 0:
            return f"{mean:.{precision}f} ± {std:.{precision}f}"
        return f"{mean:.{precision}f}"

    def pe_name(self, pe_type: str) -> str:
        """Get display name for PE type."""
        return self.pe_names.get(pe_type, pe_type)

    def dataset_name(self, dataset: str) -> str:
        """Get display name for dataset."""
        return self.dataset_names.get(dataset, dataset)

    def model_name(self, model: str) -> str:
        """Get display name for model."""
        return self.model_names.get(model, model)

    def bold_best(
        self,
        values: List[str],
        metric: str,
        format: str = "latex",
    ) -> List[str]:
        """
        Bold the best value in a list.

        Args:
            values: List of formatted values.
            metric: Metric name (to determine if lower is better).
            format: Output format ('latex' or 'markdown').

        Returns:
            List with best value bolded.
        """
        # Parse values
        parsed = []
        for v in values:
            if v == "—" or v == "-":
                parsed.append(float("inf"))
            else:
                try:
                    # Extract first number
                    num_str = v.split("±")[0].strip().replace("%", "")
                    parsed.append(float(num_str))
                except ValueError:
                    parsed.append(float("inf"))

        # Find best
        lower_better = True
        if metric in self.metric_formats:
            lower_better = self.metric_formats[metric].lower_is_better

        if lower_better:
            best_idx = parsed.index(min(parsed))
        else:
            # Flip inf to -inf for max
            parsed_max = [float("-inf") if p == float("inf") else p for p in parsed]
            best_idx = parsed_max.index(max(parsed_max))

        # Bold
        result = list(values)
        if format == "latex":
            result[best_idx] = f"\\textbf{{{result[best_idx]}}}"
        elif format == "markdown":
            result[best_idx] = f"**{result[best_idx]}**"

        return result


# ============================================================================
# Plot Style
# ============================================================================

@dataclass
class PlotStyle:
    """Style configuration for plots."""

    # Figure settings
    figure_size: Tuple[float, float] = (8, 6)
    dpi: int = 300

    # Font settings
    font_family: str = "serif"
    font_size: int = 11
    label_size: int = 12
    title_size: int = 14
    legend_size: int = 10
    tick_size: int = 10

    # Colors (colorblind-friendly palette)
    colors: List[str] = field(default_factory=lambda: [
        "#0077BB",  # Blue
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red
        "#33BBEE",  # Cyan
        "#EE3377",  # Magenta
        "#BBBBBB",  # Grey
        "#000000",  # Black
    ])

    # Markers
    markers: List[str] = field(default_factory=lambda: [
        "o", "s", "^", "D", "v", "p", "h", "*",
    ])

    # Line styles
    line_styles: List[str] = field(default_factory=lambda: [
        "-", "--", "-.", ":",
    ])

    def get_color(self, index: int) -> str:
        """Get color by index (cycles)."""
        return self.colors[index % len(self.colors)]

    def get_marker(self, index: int) -> str:
        """Get marker by index (cycles)."""
        return self.markers[index % len(self.markers)]

    def get_linestyle(self, index: int) -> str:
        """Get line style by index (cycles)."""
        return self.line_styles[index % len(self.line_styles)]


# ============================================================================
# LaTeX Utilities
# ============================================================================

def latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "$": "\\$",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def latex_bold(text: str) -> str:
    """Make text bold in LaTeX."""
    return f"\\textbf{{{text}}}"


def latex_italic(text: str) -> str:
    """Make text italic in LaTeX."""
    return f"\\textit{{{text}}}"


def latex_math(text: str) -> str:
    """Wrap text in math mode."""
    return f"${text}$"

