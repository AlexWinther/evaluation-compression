import json

import numpy as np
import pandas as pd
import pytest

from active_testing_benchmark import reporting


def write_run(tmp_path, rows, metadata_extra=None, name="run"):
    """Save a tiny benchmark run with two methods, two repetitions, and gender groups."""
    metadata = {
        "configuration": {"disease": "dr"},
        "evaluation_size": 10,
        "reference_metrics": {
            "loss": 0.5,
            "auc": 0.8,
            "subgroup_metrics": {
                "gender": {
                    "female": {"n": 6, "loss": 0.4, "auc": 0.7},
                    "male": {"n": 4, "loss": 0.65, "auc": None},
                }
            },
        },
        **(metadata_extra or {}),
    }
    pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / f"{name}.json").write_text(json.dumps(metadata))
    return tmp_path / name


def curve_row(method, budget, repetition, risk, female, male, female_labels=None):
    return {
        "n_labels": budget,
        "risk": risk,
        "method": method,
        "protected_attribute": "shared",
        "risk_objective": "unweighted_cross_entropy",
        "repetition": repetition,
        "risk gender=female": female,
        "risk gender=male": male,
        "n_labels_gender=female": budget // 2 if female_labels is None else female_labels,
        "n_labels_gender=male": budget - (budget // 2 if female_labels is None else female_labels),
    }


@pytest.fixture
def run_path(tmp_path):
    rows = []
    for budget in (2, 4):
        # Uniform misses the overall risk by 0.2 and -0.2, Kossen by 0.1 and -0.1.
        rows += [
            curve_row("UniformSampleMean", budget, 1, 0.7, 0.4, np.nan, female_labels=budget),
            curve_row("UniformSampleMean", budget, 2, 0.3, 0.4, 0.65),
            curve_row("KossenActiveTesting", budget, 1, 0.6, 0.5, 0.65),
            curve_row("KossenActiveTesting", budget, 2, 0.4, 0.3, 0.65),
        ]
    return write_run(tmp_path, rows)


def test_load_run_accepts_either_file_or_the_stem(run_path):
    for path in (run_path, run_path.with_suffix(".csv"), run_path.with_suffix(".json")):
        run = reporting.load_run(path)
        assert run.disease == "dr"
        assert run.methods == ["UniformSampleMean", "KossenActiveTesting"]
        assert run.reference == {"overall": 0.5, "gender=female": 0.4, "gender=male": 0.65}
        assert run.group_sizes["gender=male"] == 4


def test_load_run_treats_stratified_and_shared_methods_as_one_design(tmp_path):
    rows = [
        curve_row("UniformSampleMean", 2, 1, 0.7, 0.4, 0.6),
        {**curve_row("StratifiedSampleMeanBalanced", 2, 1, 0.6, 0.4, 0.6),
         "protected_attribute": "gender"},
    ]
    run = reporting.load_run(write_run(tmp_path, rows))
    assert run.methods == ["UniformSampleMean", "StratifiedSampleMeanBalanced"]


def test_load_run_keeps_one_objective_and_requires_one_design(tmp_path):
    rows = [
        curve_row("UniformSampleMean", 2, 1, 0.7, 0.4, 0.6),
        {**curve_row("UniformSampleMean", 2, 1, 9.0, 9.0, 9.0), "risk_objective": "error_rate"},
    ]
    run = reporting.load_run(write_run(tmp_path, rows))
    assert run.curves["risk"].tolist() == [0.7]

    # The stratified method was run once per stratification attribute.
    stratified = curve_row("StratifiedSampleMeanBalanced", 2, 1, 0.2, 0.4, 0.6)
    rows += [
        {**stratified, "protected_attribute": "gender"},
        {**stratified, "risk": 0.1, "protected_attribute": "race"},
    ]
    path = write_run(tmp_path, rows, name="designs")
    with pytest.raises(ValueError, match="Choose one protected attribute"):
        reporting.load_run(path)
    assert reporting.load_run(path, protected_attribute="race").curves["risk"].tolist() == [
        0.7,
        0.1,
    ]


def test_error_summary_measures_rmse_and_bias_and_counts_missing_estimates(run_path):
    run = reporting.load_run(run_path)
    summary = reporting.error_summary(run, ["overall", "gender=male"]).set_index(
        ["target", "method", "budget"]
    )
    uniform = summary.loc["overall", "UniformSampleMean", 2]
    assert uniform["rmse"] == pytest.approx(0.2)
    assert uniform["bias"] == pytest.approx(0.0)
    assert summary.loc["overall", "KossenActiveTesting", 4]["rmse"] == pytest.approx(0.1)
    # The male estimate of the first uniform repetition is missing.
    male = summary.loc["gender=male", "UniformSampleMean", 2]
    assert male["missing"] == 1
    assert male["rmse"] == pytest.approx(0.0)


def test_errors_use_the_reference_of_each_resampled_pool(tmp_path):
    # Each repetition evaluated its own pool, whose risk differs from the metadata's.
    rows = [
        {**curve_row("UniformSampleMean", 2, 1, 0.7, 0.4, 0.6), "reference_risk": 0.6,
         "reference_risk gender=female": 0.5},
        {**curve_row("UniformSampleMean", 2, 2, 0.3, 0.4, 0.6), "reference_risk": 0.2,
         "reference_risk gender=female": np.nan},
    ]
    run = reporting.load_run(write_run(tmp_path, rows))
    np.testing.assert_allclose(
        reporting.estimation_errors(run, "UniformSampleMean", "overall", 2), [0.1, 0.1]
    )
    female = reporting.estimation_errors(run, "UniformSampleMean", "gender=female", 2)
    assert female[0] == pytest.approx(-0.1)
    # A group absent from a pool has no reference and stays missing.
    assert np.isnan(female[1])
    # Without the column, the frozen-pool reference from the metadata is used.
    np.testing.assert_allclose(
        reporting.estimation_errors(run, "UniformSampleMean", "gender=male", 2), [-0.05, -0.05]
    )


def test_mse_ratio_favors_the_method_with_smaller_errors_and_pairs_repetitions(run_path):
    run = reporting.load_run(run_path)
    ratios = reporting.mse_ratios(run, ["overall", "gender=male"], [2], n_bootstrap=200)
    overall = ratios[ratios["target"] == "overall"].iloc[0]
    assert overall["method"] == "KossenActiveTesting"
    assert overall["ratio"] == pytest.approx(4.0)
    assert overall["low"] <= overall["ratio"] <= overall["high"]
    # Only the second repetition has a male estimate for both methods.
    assert ratios[ratios["target"] == "gender=male"].iloc[0]["pairs"] == 1


def test_label_allocation_averages_labels_per_group(run_path):
    run = reporting.load_run(run_path)
    allocation = reporting.label_allocation(run, 4, ["gender=female", "gender=male"])
    assert allocation.loc["UniformSampleMean"].tolist() == [3.0, 1.0]
    assert allocation.loc["KossenActiveTesting"].tolist() == [2.0, 2.0]


def test_rmse_table_marks_the_lowest_error_per_column(run_path):
    run = reporting.load_run(run_path)
    summary = reporting.error_summary(run, ["overall"])
    table = reporting.rmse_table(summary, ["overall"], [2, 4])
    assert table.startswith(r"\begin{tabular}{lrr}")
    assert r"Kossen & \textbf{0.100} & \textbf{0.100} \\" in table
    assert r"Uniform & 0.200 & 0.200 \\" in table


def test_diagnostics_table_shows_model_rows_once_across_holdout_columns():
    holdout = {
        "Pixels": {
            "kossen_score_spearman": -0.15,
            "ase_relative_bias": 1.38,
            "oracle_variance_gain": 369.3,
            "kossen_error_auc": None,
        },
        "Ensemble": {
            "kossen_score_spearman": 0.1,
            "ase_relative_bias": 0.89,
            "oracle_variance_gain": 369.3,
        },
    }
    pool = {"kossen_score_spearman": 0.06, "ase_relative_bias": 0.75, "oracle_variance_gain": 425}
    table = reporting.diagnostics_table(holdout, pool, {"Pixels": 0.635})
    assert r"Kossen score Spearman & $-0.15$ & 0.10 & 0.06 \\" in table
    assert r"\acrshort{ase} relative bias & +138\% & +89\% & +75\% \\" in table
    assert r"Oracle variance gain & \multicolumn{2}{c}{369} & 425 \\" in table
    assert r"Surrogate AUC & 0.64 & -- & -- \\" in table
    assert r"Kossen error AUC & -- & -- & -- \\" in table


def test_write_report_writes_every_table_and_the_figure(tmp_path, run_path):
    diagnostics = {"kossen_score_spearman": 0.1, "oracle_variance_gain": 10.0}
    run_path = write_run(
        run_path.parent,
        pd.read_csv(run_path.with_suffix(".csv")).to_dict("records"),
        {"surrogate_diagnostics": diagnostics},
        name="with_diagnostics",
    )
    written = reporting.write_report(
        run_path,
        tmp_path / "figures",
        tmp_path / "tables",
        tmp_path / "data",
        attributes=("gender",),
        budgets=(2,),
        rmse_budgets=(2, 4),
        allocation_budget=2,
        max_budget=4,
    )
    names = {path.name for path in written}
    for table in ("reference", "rmse", "mse_ratio", "allocation", "diagnostics"):
        assert f"dr_{table}.tex" in names
    assert (tmp_path / "figures" / "dr_rmse_curves.pdf").stat().st_size > 0
    reference = (tmp_path / "tables" / "dr_reference.tex").read_text()
    # Groups are listed largest first, and a missing AUC is shown as a dash.
    assert reference.index("Female") < reference.index("Male")
    assert r"& Male & 4 & 0.650 & -- \\" in reference
    # The default RMSE table targets are overall, the largest, and the smallest group.
    assert r"{Overall}" in (tmp_path / "tables" / "dr_rmse.tex").read_text()
    assert r"{Male}" in (tmp_path / "tables" / "dr_rmse.tex").read_text()


def test_write_report_skips_the_diagnostics_table_without_diagnostics(tmp_path, run_path):
    written = reporting.write_report(
        run_path,
        tmp_path / "f",
        tmp_path / "t",
        tmp_path / "d",
        attributes=("gender",),
        budgets=(2,),
        rmse_budgets=(2,),
        allocation_budget=2,
    )
    assert "dr_diagnostics.tex" not in {path.name for path in written}
