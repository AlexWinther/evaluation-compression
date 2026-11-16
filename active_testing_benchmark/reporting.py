"""Turn one active-testing benchmark run into the tables and figures of the report.

The input is a run saved by `experiments.at_benchmarking`, that is, a curves CSV and
its metadata JSON with the same name. Optionally, the surrogate diagnostics from
`experiments.pool_diagnostics` and a feature-experiment CSV add the diagnostics table.

Outputs are prefixed by the disease, so DR and glaucoma runs can live side by side:

    report/figures/results/<prefix>_rmse_curves.pdf   RMSE against label budget
    report/figures/tables/<prefix>_reference.tex      full-pool reference risk per group
    report/figures/tables/<prefix>_rmse.tex           RMSE at selected budgets
    report/figures/tables/<prefix>_mse_ratio.tex      uniform MSE / method MSE with CIs
    report/figures/tables/<prefix>_allocation.tex     labels per group at one budget
    report/figures/tables/<prefix>_diagnostics.tex    surrogate diagnostics
    reports/active-testing/analysis/<prefix>_*.csv    the same numbers at full precision

The LaTeX files contain only the `tabular` environment. Captions and labels stay in
the report, so the text around each table can differ between diseases.

Errors are measured against the exact risk of the pool that each repetition evaluated,
which exhaustive labeling of that pool gives. Newer runs store it per row in the
`reference_risk` columns, which also covers runs that draw a new pool in every
repetition. Older runs used one frozen pool, whose risk is read from the metadata.
The reference table always describes the whole evaluation split. The MSE ratio uses a
paired bootstrap over repetitions, because all methods share the same repetition seeds
and pools.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from active_testing_benchmark.config import PROJ_ROOT

# Benchmark class names, their display names, and a fixed plotting order.
METHOD_NAMES = {
    "UniformSampleMean": "Uniform",
    "StratifiedSampleMeanProportional": "Stratified (prop.)",
    "StratifiedSampleMeanBalanced": "Stratified (bal.)",
    "KossenActiveTesting": "Kossen",
    "ActiveSurrogateEstimator": r"\acrshort{ase}",
    "PredictionPoweredActiveTesting": r"\acrshort{ppat}",
}
BASELINE = "UniformSampleMean"
# Plot labels cannot use LaTeX acronym commands.
PLOT_NAMES = {**METHOD_NAMES, "ActiveSurrogateEstimator": "ASE"}
PLOT_NAMES["PredictionPoweredActiveTesting"] = "PPAT"
# Categorical colors in a fixed order, with line styles so that identity never
# depends on color alone. Uniform is the reference and drawn in near-black.
PLOT_STYLES = {
    "UniformSampleMean": ("#0b0b0b", "-"),
    "StratifiedSampleMeanProportional": ("#2a78d6", "--"),
    "StratifiedSampleMeanBalanced": ("#1baf7a", ":"),
    "KossenActiveTesting": ("#eb6834", "-"),
    "ActiveSurrogateEstimator": ("#6250d6", "-."),
    "PredictionPoweredActiveTesting": ("#008300", "--"),
}
OVERALL = "overall"


@dataclass(frozen=True)
class BenchmarkRun:
    """Curves and metadata of one benchmark run, restricted to one objective and design."""

    curves: pd.DataFrame
    metadata: dict
    reference: dict[str, float]
    group_sizes: dict[str, int]

    @property
    def disease(self) -> str:
        return self.metadata["configuration"]["disease"]

    @property
    def methods(self) -> list[str]:
        present = set(self.curves["method"])
        return [method for method in METHOD_NAMES if method in present]


def run_paths(run: Path) -> tuple[Path, Path]:
    """Return the curves CSV and metadata JSON for a run given by either file or their stem."""
    stem = run.with_suffix("") if run.suffix in {".csv", ".json"} else run
    return stem.with_name(stem.name + ".csv"), stem.with_name(stem.name + ".json")


def load_run(
    run: Path,
    objective: str = "unweighted_cross_entropy",
    protected_attribute: str | None = None,
) -> BenchmarkRun:
    """Load a run and keep the rows for one risk objective and sampling design.

    Older runs have no `risk_objective` or `protected_attribute` column. In newer
    runs, `protected_attribute` is the stratification attribute of the stratified
    methods and `shared` for the others, which is one design. Only when a method
    appears with several attributes must one be chosen with `protected_attribute`.
    The `shared` rows are then kept alongside the chosen attribute.
    """
    curves_path, metadata_path = run_paths(run)
    curves = pd.read_csv(curves_path)
    metadata = json.loads(metadata_path.read_text())
    if "risk_objective" in curves:
        curves = curves[curves["risk_objective"] == objective]
        if curves.empty:
            raise ValueError(f"No rows with risk objective {objective!r} in {curves_path}")
    if "protected_attribute" in curves:
        if protected_attribute is not None:
            keep = curves["protected_attribute"].isin([protected_attribute, "shared"])
            curves = curves[keep]
        designs_per_method = curves.groupby("method")["protected_attribute"].nunique()
        if (designs_per_method > 1).any():
            designs = sorted(set(curves["protected_attribute"]) - {"shared"})
            raise ValueError(f"Choose one protected attribute from {designs}")

    reference_metrics = metadata["reference_metrics"]
    reference = {OVERALL: reference_metrics["loss"]}
    group_sizes = {OVERALL: metadata["evaluation_size"]}
    for attribute, groups in reference_metrics.get("subgroup_metrics", {}).items():
        for group, values in groups.items():
            reference[f"{attribute}={group}"] = values["loss"]
            group_sizes[f"{attribute}={group}"] = values["n"]
    return BenchmarkRun(curves.reset_index(drop=True), metadata, reference, group_sizes)


def risk_column(target: str) -> str:
    """Column of the curves CSV that holds the estimate for `target`."""
    return "risk" if target == OVERALL else f"risk {target}"


def estimation_errors(run: BenchmarkRun, method: str, target: str, budget: int) -> np.ndarray:
    """Signed errors of one method at one budget, ordered by repetition.

    Each error uses the reference risk stored with its row, that is, the risk of the
    pool that the repetition evaluated. Runs without these columns fall back to the
    frozen full-pool reference of the metadata. Missing estimates (a subgroup without
    labels yet, or absent from a resampled pool) stay as NaN so that the repetitions of
    different methods remain aligned for the paired bootstrap.
    """
    rows = run.curves[(run.curves["method"] == method) & (run.curves["n_labels"] == budget)]
    rows = rows.sort_values("repetition")
    reference_column = f"reference_{risk_column(target)}"
    reference = rows[reference_column] if reference_column in rows else run.reference[target]
    return (rows[risk_column(target)] - reference).to_numpy(dtype=float)


def error_summary(run: BenchmarkRun, targets: list[str]) -> pd.DataFrame:
    """RMSE, bias, and median absolute error per target, method, and budget."""
    rows = []
    for target in targets:
        for method in run.methods:
            for budget in sorted(run.curves["n_labels"].unique()):
                errors = estimation_errors(run, method, target, budget)
                observed = errors[~np.isnan(errors)]
                rows.append(
                    {
                        "target": target,
                        "method": method,
                        "budget": int(budget),
                        "rmse": np.sqrt(np.mean(observed**2)) if len(observed) else np.nan,
                        "bias": observed.mean() if len(observed) else np.nan,
                        "median_absolute_error": (
                            np.median(np.abs(observed)) if len(observed) else np.nan
                        ),
                        "missing": int(np.isnan(errors).sum()),
                        "repetitions": len(errors),
                    }
                )
    return pd.DataFrame(rows)


def mse_ratios(
    run: BenchmarkRun,
    targets: list[str],
    budgets: list[int],
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return MSE(uniform) / MSE(method) with 95% paired bootstrap intervals.

    A ratio above one means that the method needs fewer labels than uniform
    sampling for the same error. Repetitions where either method has no estimate
    are dropped from that comparison.
    """
    generator = np.random.default_rng(seed)
    rows = []
    for target in targets:
        for method in run.methods:
            if method == BASELINE:
                continue
            for budget in budgets:
                uniform = estimation_errors(run, BASELINE, target, budget)
                other = estimation_errors(run, method, target, budget)
                keep = ~np.isnan(uniform) & ~np.isnan(other)
                uniform, other = uniform[keep] ** 2, other[keep] ** 2
                if not len(uniform):
                    continue
                resamples = generator.integers(0, len(uniform), size=(n_bootstrap, len(uniform)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    bootstrap = uniform[resamples].mean(1) / other[resamples].mean(1)
                    ratio = uniform.mean() / other.mean()
                # A resample whose errors are all zero gives 0/0, which has no ratio.
                finite = bootstrap[np.isfinite(bootstrap)]
                low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan,) * 2
                rows.append(
                    {
                        "target": target,
                        "method": method,
                        "budget": budget,
                        "ratio": ratio,
                        "low": low,
                        "high": high,
                        "pairs": int(keep.sum()),
                    }
                )
    return pd.DataFrame(rows)


def label_allocation(run: BenchmarkRun, budget: int, groups: list[str]) -> pd.DataFrame:
    """Mean number of labels each method has acquired per group at `budget`."""
    rows = run.curves[run.curves["n_labels"] == budget]
    table = rows.groupby("method")[[f"n_labels_{group}" for group in groups]].mean()
    table.columns = groups
    return table.loc[[method for method in run.methods if method in table.index]]


def default_groups(run: BenchmarkRun, attributes: list[str]) -> list[str]:
    """All groups of the given attributes, largest first within each attribute."""
    groups = []
    for attribute in attributes:
        members = [key for key in run.group_sizes if key.startswith(f"{attribute}=")]
        groups += sorted(members, key=lambda key: -run.group_sizes[key])
    return groups


def group_label(target: str) -> str:
    """Readable name of a target, for example `race=black` becomes `Black`."""
    return "Overall" if target == OVERALL else target.split("=", 1)[1].capitalize()


# LaTeX tables ---------------------------------------------------------------------


def _format(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    text = f"{value:.{digits}f}"
    return f"${text}$" if value < 0 else text


def _tabular(column_spec: str, header: list[str], body: list[str], before: str = "") -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule"]
    lines += [before] if before else []
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += body + [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def reference_table(run: BenchmarkRun, attributes: list[str]) -> str:
    """Full-pool size, risk, and AUC overall and for each group of `attributes`."""
    metrics = run.metadata["reference_metrics"]
    overall = (
        f"All & & {run.group_sizes[OVERALL]} & {_format(metrics['loss'])} & "
        f"{_format(metrics.get('auc'))} \\\\"
    )
    body = [overall]
    for attribute in attributes:
        body.append(r"\midrule")
        groups = default_groups(run, [attribute])
        for index, group in enumerate(groups):
            values = metrics["subgroup_metrics"][attribute][group.split("=", 1)[1]]
            name = attribute.replace("_", " ").capitalize() if index == 0 else ""
            body.append(
                f"{name} & {group_label(group)} & {values['n']} & {_format(values['loss'])} & "
                f"{_format(values.get('auc'))} \\\\"
            )
    return _tabular("llrrr", ["Attribute", "Group", "$n$", "Risk", "AUC"], body)


def rmse_table(summary: pd.DataFrame, targets: list[str], budgets: list[int]) -> str:
    """RMSE per method for several targets and budgets, best value per column in bold."""
    methods = [method for method in METHOD_NAMES if method in set(summary["method"])]
    values = summary.set_index(["target", "method", "budget"])["rmse"]
    columns = [(target, budget) for target in targets for budget in budgets]
    best = {
        (target, budget): summary[(summary["target"] == target) & (summary["budget"] == budget)][
            "rmse"
        ].min()
        for target, budget in columns
    }
    body = []
    for method in methods:
        cells = []
        for target, budget in columns:
            value = values.loc[target, method, budget]
            text = _format(value)
            cells.append(
                rf"\textbf{{{text}}}" if np.isclose(value, best[target, budget]) else text
            )
        body.append(f"{METHOD_NAMES[method]} & " + " & ".join(cells) + r" \\")
    width = len(budgets)
    groups = " & ".join(
        rf"\multicolumn{{{width}}}{{c}}{{{group_label(target)}}}" for target in targets
    )
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + i * width}-{1 + (i + 1) * width}}}" for i in range(len(targets))
    )
    header = ["Method"] + [str(budget) for _ in targets for budget in budgets]
    return _tabular("l" + "r" * len(columns), header, body, before=f" & {groups} \\\\\n{rules}")


def mse_ratio_table(ratios: pd.DataFrame, target: str, budgets: list[int]) -> str:
    """MSE ratio against uniform sampling with its interval, one row per method."""
    rows = ratios[ratios["target"] == target].set_index(["method", "budget"])
    body = []
    for method, name in METHOD_NAMES.items():
        if method == BASELINE or method not in rows.index.get_level_values(0):
            continue
        cells = []
        for budget in budgets:
            if (method, budget) not in rows.index:
                cells.append("--")
                continue
            row = rows.loc[method, budget]
            cells.append(f"{row['ratio']:.2f} [{row['low']:.2f}, {row['high']:.2f}]")
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    header = ["Method"] + [f"$M={budget}$" for budget in budgets]
    return _tabular("l" + "c" * len(budgets), header, body)


def allocation_table(allocation: pd.DataFrame) -> str:
    """Mean labels per group, one row per method."""
    header = ["Method"] + [group_label(group) for group in allocation.columns]
    body = [
        f"{METHOD_NAMES[method]} & " + " & ".join(f"{value:.1f}" for value in row) + r" \\"
        for method, row in allocation.iterrows()
    ]
    return _tabular("l" + "r" * len(allocation.columns), header, body)


# Diagnostics rows whose value depends on the surrogate, with their display format.
SURROGATE_DIAGNOSTICS = {
    "kossen_score_spearman": ("Kossen score Spearman", "{:.2f}"),
    "kossen_error_auc": ("Kossen error AUC", "{:.2f}"),
    "kossen_variance_gain": ("Kossen variance gain", "{:.2f}"),
    "ase_relative_bias": (r"\acrshort{ase} relative bias", "percent"),
    "ppat_proxy_correlation": (r"\acrshort{ppat} proxy correlation", "{:.2f}"),
}
# Rows that depend only on the classifier and the data, not on the surrogate.
MODEL_DIAGNOSTICS = {
    "target_variance_gain": ("Model as surrogate gain", "{:.2f}"),
    "confidence_oracle_variance_gain": ("Confidence-only gain", "{:.2f}"),
    "oracle_variance_gain": ("Oracle variance gain", "{:.0f}"),
}
EXTRACTOR_NAMES = {
    "pooled_pixels": "Pixels",
    "spatial_texture": "Texture",
    "imagenet_resnet18": "ImageNet",
}


def _diagnostic(value: float | None, style: str) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if style == "percent":
        return rf"{value * 100:+.0f}\%"
    text = style.format(value)
    return f"${text}$" if value < 0 else text


def diagnostics_table(
    holdout: dict[str, dict] | None = None,
    pool: dict | None = None,
    surrogate_auc: dict[str, float] | None = None,
) -> str:
    """Surrogate diagnostics with one column per holdout surrogate and one for the pool.

    `holdout` maps a column name (for example a feature extractor) to the output of
    `surrogate_diagnostics` on held-out development data. `pool` is the same on the
    evaluation pool. `surrogate_auc` optionally gives the surrogate's own AUC per
    holdout column. The model-only rows are equal for every holdout column and are
    therefore shown once across them.
    """
    holdout = holdout or {}
    columns = list(holdout) + (["Ensemble"] if pool is not None else [])
    width = len(holdout)
    spans = []
    if width:
        spans.append(rf"\multicolumn{{{width}}}{{c}}{{Development holdout}}")
    if pool is not None:
        spans.append("Pool")
    rules = []
    if width:
        rules.append(rf"\cmidrule(lr){{2-{1 + width}}}")
    if pool is not None:
        rules.append(rf"\cmidrule(lr){{{2 + width}-{2 + width}}}")
    before = " & " + " & ".join(spans) + r" \\" + "\n" + " ".join(rules)

    body = []
    if surrogate_auc:
        cells = [_diagnostic(surrogate_auc.get(name), "{:.2f}") for name in holdout]
        cells += ["--"] if pool is not None else []
        body.append("Surrogate AUC & " + " & ".join(cells) + r" \\")
    for key, (label, style) in SURROGATE_DIAGNOSTICS.items():
        cells = [_diagnostic(values.get(key), style) for values in holdout.values()]
        cells += [_diagnostic(pool.get(key), style)] if pool is not None else []
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    body.append(r"\midrule")
    first = next(iter(holdout.values()), None)
    for key, (label, style) in MODEL_DIAGNOSTICS.items():
        cells = []
        if width:
            cells.append(rf"\multicolumn{{{width}}}{{c}}{{{_diagnostic(first.get(key), style)}}}")
        cells += [_diagnostic(pool.get(key), style)] if pool is not None else []
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    return _tabular("l" + "r" * len(columns), ["Diagnostic"] + columns, body, before=before)


# Figure ------------------------------------------------------------------------------


def plot_rmse_curves(
    summary: pd.DataFrame,
    targets: list[str],
    group_sizes: dict[str, int],
    path: Path,
    max_budget: int | None = None,
    columns: int = 3,
) -> None:
    """Small multiples of RMSE against label budget, one panel per target.

    The y-axis is logarithmic so that the error decay at large budgets stays
    visible next to the large errors at small budgets.
    """
    rows = int(np.ceil(len(targets) / columns))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(
        rows, columns, figsize=(7.0, 2.1 * rows + 0.4), sharex=True, squeeze=False
    )
    data = summary if max_budget is None else summary[summary["budget"] <= max_budget]
    for axis, target in zip(axes.flat, targets):
        for method in METHOD_NAMES:
            curve = data[(data["target"] == target) & (data["method"] == method)]
            if curve.empty:
                continue
            color, style = PLOT_STYLES[method]
            axis.plot(
                curve["budget"],
                curve["rmse"],
                style,
                color=color,
                lw=1.5,
                label=PLOT_NAMES[method],
            )
        axis.set_title(f"{group_label(target)} (n={group_sizes[target]})", fontsize=8)
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
        axis.grid(True, which="major", color="#e5e5e5", lw=0.5)
        axis.tick_params(colors="#52514e")
    for axis in axes.flat[len(targets) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Labels acquired $M$")
    for axis in axes[:, 0]:
        axis.set_ylabel("RMSE of risk estimate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


# Entry point -------------------------------------------------------------------------


def _feature_experiment_columns(path: Path) -> tuple[dict[str, dict], dict[str, float]]:
    """Holdout diagnostics and surrogate AUC per extractor from a feature-experiment CSV."""
    table = pd.read_csv(path)
    keys = list(SURROGATE_DIAGNOSTICS) + list(MODEL_DIAGNOSTICS)
    holdout, auc = {}, {}
    for _, row in table.iterrows():
        name = EXTRACTOR_NAMES.get(row["extractor"], row["extractor"])
        holdout[name] = {key: row[key] for key in keys if key in row}
        auc[name] = row["auc"]
    return holdout, auc


def write_report(
    run_path: Path,
    figure_dir: Path,
    table_dir: Path,
    data_dir: Path,
    prefix: str | None = None,
    attributes: tuple[str, ...] = ("gender", "race"),
    table_targets: tuple[str, ...] | None = None,
    budgets: tuple[int, ...] = (50, 100, 200, 500),
    rmse_budgets: tuple[int, ...] = (50, 100, 500),
    allocation_budget: int = 100,
    max_budget: int = 500,
    diagnostics_path: Path | None = None,
    feature_experiment_path: Path | None = None,
    objective: str = "unweighted_cross_entropy",
    protected_attribute: str | None = None,
) -> list[Path]:
    """Write every table and figure for one run and return the written paths.

    `table_targets` selects the targets of the RMSE table. By default these are the
    overall risk, the largest group of the first attribute, and the smallest group
    of the last attribute, which shows the range of subgroup sizes.
    """
    run = load_run(run_path, objective, protected_attribute)
    prefix = prefix or run.disease
    groups = default_groups(run, list(attributes))
    targets = [OVERALL] + groups
    if table_targets is None:
        first = default_groups(run, [attributes[0]])
        last = default_groups(run, [attributes[-1]])
        table_targets = (OVERALL, first[0], last[-1])

    summary = error_summary(run, targets)
    ratios = mse_ratios(run, [OVERALL, *groups], list(budgets))
    allocation = label_allocation(run, allocation_budget, groups)

    written = []
    for directory in (figure_dir, table_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("error_summary", summary), ("mse_ratio", ratios)):
        frame.to_csv(data_dir / f"{prefix}_{name}.csv", index=False)
        written.append(data_dir / f"{prefix}_{name}.csv")
    allocation.to_csv(data_dir / f"{prefix}_label_allocation.csv")
    written.append(data_dir / f"{prefix}_label_allocation.csv")

    tables = {
        "reference": reference_table(run, list(attributes)),
        "rmse": rmse_table(summary, list(table_targets), list(rmse_budgets)),
        "mse_ratio": mse_ratio_table(ratios, OVERALL, list(budgets)),
        "allocation": allocation_table(allocation),
    }

    # Holdout diagnostics come from the feature experiment, from the diagnostics file,
    # or from the run's own metadata. The pool column needs the diagnostics file.
    diagnostics_path = diagnostics_path or default_diagnostics_path(run_path)
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    holdout, auc = {}, {}
    if feature_experiment_path is not None:
        holdout, auc = _feature_experiment_columns(feature_experiment_path)
    ensemble = diagnostics.get("holdout") or run.metadata.get("surrogate_diagnostics")
    if ensemble:
        holdout["Ensemble"] = ensemble
    if holdout or diagnostics.get("pool"):
        tables["diagnostics"] = diagnostics_table(holdout, diagnostics.get("pool"), auc)

    for name, text in tables.items():
        path = table_dir / f"{prefix}_{name}.tex"
        path.write_text(text)
        written.append(path)
    figure_path = figure_dir / f"{prefix}_rmse_curves.pdf"
    plot_rmse_curves(summary, targets, run.group_sizes, figure_path, max_budget)
    written.append(figure_path)
    return written


def default_diagnostics_path(run_path: Path) -> Path:
    """Where `experiments.pool_diagnostics` saves the diagnostics for a run."""
    curves_path, _ = run_paths(run_path)
    return curves_path.with_name(curves_path.stem + "-diagnostics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="Benchmark run: its CSV, JSON, or their stem")
    parser.add_argument("--prefix", help="Output file prefix (default: the run's disease)")
    parser.add_argument("--attributes", nargs="+", default=["gender", "race"])
    parser.add_argument(
        "--table-targets",
        nargs="+",
        help="Targets of the RMSE table, for example overall gender=female race=black",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument("--rmse-budgets", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--allocation-budget", type=int, default=100)
    parser.add_argument("--max-budget", type=int, default=500, help="Last budget in the figure")
    parser.add_argument("--diagnostics", type=Path, help="Output of experiments.pool_diagnostics")
    parser.add_argument("--feature-experiment", type=Path, help="Feature-experiment CSV")
    parser.add_argument("--objective", default="unweighted_cross_entropy")
    parser.add_argument("--protected-attribute", help="Design to keep if the run has several")
    parser.add_argument(
        "--figure-dir", type=Path, default=PROJ_ROOT / "report" / "figures" / "results"
    )
    parser.add_argument(
        "--table-dir", type=Path, default=PROJ_ROOT / "report" / "figures" / "tables"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJ_ROOT / "reports" / "active-testing" / "analysis"
    )
    args = parser.parse_args()
    written = write_report(
        args.run,
        args.figure_dir,
        args.table_dir,
        args.data_dir,
        prefix=args.prefix,
        attributes=tuple(args.attributes),
        table_targets=tuple(args.table_targets) if args.table_targets else None,
        budgets=tuple(args.budgets),
        rmse_budgets=tuple(args.rmse_budgets),
        allocation_budget=args.allocation_budget,
        max_budget=args.max_budget,
        diagnostics_path=args.diagnostics,
        feature_experiment_path=args.feature_experiment,
        objective=args.objective,
        protected_attribute=args.protected_attribute,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
