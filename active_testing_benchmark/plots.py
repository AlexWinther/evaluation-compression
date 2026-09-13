"""Visualize offline active-testing experiment results."""

# ruff: noqa: B008 -- Typer declares CLI options as argument defaults.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False, help=__doc__)

REQUIRED_COLUMNS = {
    "budget",
    "run",
    "model",
    "metric",
    "estimate",
    "full_value",
    "absolute_error",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "roc_auc": "ROC AUC",
    "calibration_error": "Calibration error",
}


def load_results(experiment_dir: Path) -> tuple[pd.DataFrame, dict, Path]:
    """Load and validate an active-testing result directory or results CSV."""
    results_path = experiment_dir / "results.csv" if experiment_dir.is_dir() else experiment_dir
    if not results_path.is_file():
        raise ValueError(f"Results file does not exist: {results_path}")

    results = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(f"Results file is missing columns: {', '.join(sorted(missing))}")
    if results.empty:
        raise ValueError("Results file contains no rows")

    for column in ("budget", "run", "estimate", "full_value", "absolute_error"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    if results["budget"].isna().any() or (results["budget"] <= 0).any():
        raise ValueError("Budget values must be positive numbers")
    if results[["model", "metric"]].isna().any().any():
        raise ValueError("Model and metric values must be present")

    root = results_path.parent
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return results, config, root


def model_labels(config: dict, models: list[str]) -> dict[str, str]:
    """Use recorded model sources as concise, unique legend labels."""
    configured = config.get("models", {})
    labels = {}
    for model in models:
        source = configured.get(model, {}).get("source", model)
        labels[model] = str(source).removeprefix("models:/")
    if len(set(labels.values())) != len(labels):
        return {model: f"{label} ({model})" for model, label in labels.items()}
    return labels


def _x_values(frame: pd.DataFrame, population: int | None, x_axis: str) -> np.ndarray:
    budgets = frame["budget"].to_numpy(dtype=float)
    if x_axis == "percent":
        if population is None or population <= 0:
            raise ValueError("Percentage x-axis requires a positive population in config.json")
        return budgets / population * 100
    return budgets


def _plot_metric(
    estimate_axis: Axes,
    error_axis: Axes,
    frame: pd.DataFrame,
    labels: dict[str, str],
    population: int | None,
    x_axis: str,
) -> None:
    for model, model_frame in frame.groupby("model", sort=False):
        color = estimate_axis._get_lines.get_next_color()
        finite = model_frame.dropna(subset=["estimate"])
        estimate_axis.scatter(
            _x_values(finite, population, x_axis),
            finite["estimate"],
            color=color,
            alpha=0.28,
            edgecolors="none",
            s=22,
        )

        summary = (
            model_frame.groupby("budget", as_index=False)
            .agg(
                estimate=("estimate", "mean"),
                absolute_error=("absolute_error", "mean"),
            )
            .sort_values("budget")
        )
        x = _x_values(summary, population, x_axis)
        estimate_axis.plot(x, summary["estimate"], color=color, marker="o", label=labels[model])
        error_axis.plot(
            x,
            summary["absolute_error"],
            color=color,
            marker="o",
            label=labels[model],
        )

        full_values = model_frame["full_value"].dropna().unique()
        if len(full_values) > 1:
            raise ValueError(f"Full-test value changes within {model}/{frame['metric'].iloc[0]}")
        if len(full_values) == 1:
            estimate_axis.axhline(full_values[0], color=color, linestyle="--", alpha=0.75)

    estimate_axis.grid(alpha=0.2)
    error_axis.grid(alpha=0.2)
    error_axis.set_ylim(bottom=0)


def plot_results(
    results: pd.DataFrame,
    config: dict,
    output_path: Path,
    metrics: list[str] | None = None,
    x_axis: str = "percent",
    dpi: int = 180,
) -> Path:
    """Create an overview of estimates and absolute errors for each metric."""
    available = list(dict.fromkeys(results["metric"].astype(str)))
    selected = metrics or available
    unknown = [metric for metric in selected if metric not in available]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {', '.join(unknown)}. Available: {', '.join(available)}"
        )
    if x_axis not in {"percent", "count"}:
        raise ValueError("x_axis must be 'percent' or 'count'")

    models = list(dict.fromkeys(results["model"].astype(str)))
    labels = model_labels(config, models)
    population_value = config.get("population")
    population = int(population_value) if population_value is not None else None

    figure, axes = plt.subplots(
        len(selected),
        2,
        figsize=(13, max(3.2 * len(selected), 4.2)),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )
    figure.suptitle(
        f"Active-testing estimates ({config.get('reducer', 'unknown')} selection, "
        f"{config.get('runs', int(results['run'].nunique()))} runs)",
        fontsize=14,
    )
    for row, metric in enumerate(selected):
        metric_frame = results.loc[results["metric"] == metric]
        _plot_metric(axes[row, 0], axes[row, 1], metric_frame, labels, population, x_axis)
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        axes[row, 0].set_ylabel(label)
        axes[row, 1].set_ylabel(f"Absolute {label.lower()} error")

    axes[0, 0].set_title("Subset estimate (dots are individual runs)")
    axes[0, 1].set_title("Mean absolute error")
    x_label = (
        "Test subset size (% of full test set)" if x_axis == "percent" else "Test subset size"
    )
    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, legend_labels, loc="outside upper right", title="Model")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


@app.command()
def main(
    experiment_dir: Path = typer.Argument(
        ...,
        help="Experiment directory containing results.csv, or the CSV itself.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output image path (default: <experiment>/active-testing-results.png).",
    ),
    metric: list[str] | None = typer.Option(
        None,
        "--metric",
        help="Metric to plot; repeat to select multiple metrics.",
    ),
    x_axis: str = typer.Option("percent", help="Use 'percent' or 'count' for subset size."),
    dpi: int = typer.Option(180, min=72, max=600),
) -> None:
    """Plot subset estimates and errors from one active-testing experiment."""
    try:
        results, config, root = load_results(experiment_dir)
        destination = output_path or root / "active-testing-results.png"
        plot_results(results, config, destination, metric, x_axis, dpi)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Figure: {destination}")


if __name__ == "__main__":
    app()
