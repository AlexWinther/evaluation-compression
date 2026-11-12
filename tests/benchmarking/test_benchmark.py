from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from active_testing_benchmark import benchmark
from active_testing_benchmark.active_testing import (
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
)
from active_testing_benchmark.active_testing.baseline import (
    StratifiedSampleMean,
    UniformSampleMean,
)
from active_testing_benchmark.active_testing.interface import Problem
from active_testing_benchmark.active_testing.runner import (
    EvaluationPool,
    prepare_evaluation_pool,
    run_active_evaluation,
)
from active_testing_benchmark.active_testing.sota import (
    ActiveSurrogateEstimator,
)
from active_testing_benchmark.active_testing.surrogate import RandomForestModule
from active_testing_benchmark.benchmark import build_method_configs


class TinyDataset:
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 4

    def __getitem__(self, index):
        self.reads += 1
        return {
            "image": torch.tensor([2.0, 0.0]),
            "label": index % 2,
            "subgroups": {
                "gender": ["female", "female", "male", "male"][index],
                "age_group": ["<50", "80+", "<50", "80+"][index],
                "race": ["asian", "white", "asian", "white"][index],
                "ethnicity": ["hispanic", "non-hispanic", "hispanic", "non-hispanic"][index],
                "language": ["english", "spanish", "english", "spanish"][index],
                "maritalstatus": ["single", "married", "single", "married"][index],
            },
        }


def test_attribute_views_share_tensors_without_implicit_stratification():
    data = TinyDataset()
    pool = prepare_evaluation_pool(data, "cpu", ["gender", "age_group"])
    assert pool.stratification_groups is None
    for attribute, expected in [
        ("gender", ["gender=female"] * 2 + ["gender=male"] * 2),
        ("age_group", ["age_group=<50", "age_group=80+"] * 2),
    ]:
        view = prepare_evaluation_pool(pool, "cpu", attribute)
        assert view.X_pool is pool.X_pool
        assert view.y_pool is pool.y_pool
        assert view.stratification_groups is None
        assert view.attribute_values[attribute] == expected
    assert data.reads == 4


@pytest.mark.parametrize("method", [UniformSampleMean, StratifiedSampleMean])
def test_benchmark_shares_only_attribute_independent_methods(monkeypatch, method):
    data = TinyDataset()
    monkeypatch.setattr(
        benchmark, "build_method_configs", lambda *a, **k: [(method, {"batch_size": 2})]
    )
    curves = benchmark.run_benchmark(
        model=nn.Identity(),
        dataset=data,
        auxiliary=SimpleNamespace(),
        num_classes=2,
        device="cpu",
        seed=42,
        repetitions=2,
        budget=4,
        protected_columns=("gender", "age_group"),
    )
    assert data.reads == 4
    shared = method is UniformSampleMean
    assert set(curves.protected_attribute) == ({"shared"} if shared else {"gender", "age_group"})
    assert len(curves) == (4 if shared else 8)
    assert {c for c in curves if c.startswith("risk ")} == {
        "risk gender=female",
        "risk gender=male",
        "risk age_group=<50",
        "risk age_group=80+",
    }
    for attribute in ("gender", "age_group"):
        frame = curves if shared else curves[curves.protected_attribute == attribute]
        final = frame[frame.n_labels == 4]
        for group in ("female", "male") if attribute == "gender" else ("<50", "80+"):
            assert final[f"n_labels_{attribute}={group}"].tolist() == [2, 2]
            assert final[f"risk {attribute}={group}"].notna().all()


def test_stratification_attribute_is_separate_from_reporting_attributes():
    class InspectStratifiedSampleMean(StratifiedSampleMean):
        def initialize(self, problem):
            self.problem = problem
            return super().initialize(problem)

    method = InspectStratifiedSampleMean(seed=42, batch_size=4)
    history = run_active_evaluation(
        method,
        TinyDataset(),
        nn.Identity(),
        "cpu",
        nn.CrossEntropyLoss(),
        ["gender", "age_group"],
        stratification_attribute="gender",
        budget=4,
    )

    expected_gender_groups = ["gender=female"] * 2 + ["gender=male"] * 2
    assert method.problem.stratification_groups == expected_gender_groups
    record = history[0]
    selected_gender_groups = [
        expected_gender_groups[index] for index in record["selection"].indices
    ]
    assert record["selection"].stratification_groups == selected_gender_groups
    assert record["stratification_groups"] == selected_gender_groups
    assert set(record["estimate"].subgroup_risks) == {
        "gender=female",
        "gender=male",
        "age_group=<50",
        "age_group=80+",
    }
    assert all(risk is not None for risk in record["estimate"].subgroup_risks.values())


def test_shared_run_has_no_implicit_stratification_and_keeps_unobserved_marginals_missing():
    class InspectUniformSampleMean(UniformSampleMean):
        def initialize(self, problem):
            self.problem = problem
            return super().initialize(problem)

    method = InspectUniformSampleMean(seed=42, batch_size=1)
    history = run_active_evaluation(
        method,
        TinyDataset(),
        nn.Identity(),
        "cpu",
        nn.CrossEntropyLoss(),
        ["gender", "age_group"],
        budget=1,
    )
    record = history[0]

    assert method.problem.stratification_groups is None
    assert record["selection"].stratification_groups is None
    assert record["stratification_groups"] is None
    estimate = record["estimate"]
    assert set(estimate.subgroup_risks) == {
        "gender=female",
        "gender=male",
        "age_group=<50",
        "age_group=80+",
    }
    for group, count in estimate.subgroup_n_labels.items():
        if count == 0:
            assert estimate.subgroup_risks[group] is None
        else:
            assert estimate.subgroup_risks[group] == pytest.approx(record["losses"][0])


def test_uniform_marginal_estimates_use_observations_without_joint_coverage():
    history = run_active_evaluation(
        UniformSampleMean(seed=42, batch_size=1),
        TinyDataset(),
        nn.Identity(),
        "cpu",
        nn.CrossEntropyLoss(),
        "gender",
        budget=1,
    )
    record = history[0]
    estimate = record["estimate"]
    assert record["stratification_groups"] is None
    observed_group = ["gender=female", "gender=female", "gender=male", "gender=male"][
        record["selection"].indices[0]
    ]
    assert estimate.subgroup_risks[observed_group] == pytest.approx(record["losses"][0])
    assert estimate.subgroup_n_labels[observed_group] == 1
    unobserved_group = "gender=male" if observed_group == "gender=female" else "gender=female"
    assert estimate.subgroup_n_labels[unobserved_group] == 0
    assert estimate.subgroup_risks[unobserved_group] is None
    assert all("," not in group for group in estimate.subgroup_risks)


@pytest.mark.parametrize(
    "columns,expected_runs",
    [
        (("gender", "age_group"), 8),
        (("gender",), 6),
        (("gender", "race", "ethnicity", "language", "maritalstatus", "age_group"), 16),
    ],
)
def test_all_methods_schedule_shared_and_stratified_trajectories(
    monkeypatch, columns, expected_runs
):
    calls = []
    original = benchmark.run_active_evaluation

    def track(**kwargs):
        calls.append(
            (
                type(kwargs["method"]).__name__,
                tuple(kwargs["protected_column"]),
                kwargs["stratification_attribute"],
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(benchmark, "run_active_evaluation", track)
    curves = benchmark.run_benchmark(
        model=nn.Identity(),
        dataset=TinyDataset(),
        auxiliary=SimpleNamespace(surrogates=(nn.Identity(), nn.Identity()), proxy=nn.Identity()),
        num_classes=2,
        device="cpu",
        seed=42,
        repetitions=1,
        budget=4,
        protected_columns=columns,
    )
    assert len(calls) == expected_runs
    assert len(curves) == expected_runs
    for method, used_columns, stratification_attribute in calls:
        assert used_columns == columns
        if method == "StratifiedSampleMean":
            assert stratification_attribute in columns
        else:
            assert stratification_attribute is None
    shared = curves[curves.protected_attribute == ("shared" if len(columns) > 1 else "gender")]
    assert len(shared) == (4 if len(columns) > 1 else 6)
    for column in columns:
        risk_columns = [c for c in curves if c.startswith(f"risk {column}=")]
        count_columns = [c for c in curves if c.startswith(f"n_labels_{column}=")]
        assert len(risk_columns) == 2
        assert curves[risk_columns].notna().all().all()
        assert curves[count_columns].eq(2).all().all()
        assert (shared[count_columns].sum(axis=1) == 4).all()
        assert (curves[count_columns].sum(axis=1) == 4).all()
    assert not any(c.startswith("risk age=") for c in curves)


def test_uniform_shared_prefixes_match_independent_attribute_runs():
    def run(columns):
        return run_active_evaluation(
            UniformSampleMean(seed=42),
            TinyDataset(),
            nn.Identity(),
            "cpu",
            nn.CrossEntropyLoss(),
            columns,
        )

    shared = run(["gender", "age_group"])
    for column in ["gender", "age_group"]:
        independent = run(column)
        for actual, expected in zip(shared, independent):
            assert actual["selection"].indices == expected["selection"].indices
            assert actual["estimate"].risk == expected["estimate"].risk
            for group, risk in expected["estimate"].subgroup_risks.items():
                assert actual["estimate"].subgroup_risks[group] == risk
                assert (
                    actual["estimate"].subgroup_n_labels[group]
                    == expected["estimate"].subgroup_n_labels[group]
                )


def test_selected_stratification_still_reports_all_attributes():
    curves = benchmark.run_benchmark(
        model=nn.Identity(),
        dataset=TinyDataset(),
        auxiliary=SimpleNamespace(surrogates=(nn.Identity(), nn.Identity()), proxy=nn.Identity()),
        num_classes=2,
        device="cpu",
        seed=42,
        repetitions=1,
        budget=4,
        protected_columns=("gender", "age_group"),
        stratification_attribute="gender",
    )
    assert len(curves) == 6
    assert set(curves.protected_attribute) == {"shared", "gender"}
    for attribute in ("gender", "age_group"):
        risks = [c for c in curves if c.startswith(f"risk {attribute}=")]
        assert len(risks) == 2
        assert curves[risks].notna().all().all()


def fitted_surrogates():
    development = [
        {"image": torch.tensor([0.0, 0.0]), "label": 0},
        {"image": torch.tensor([1.0, 1.0]), "label": 0},
    ]
    return tuple(
        RandomForestModule(2, nn.Identity(), n_estimators=5, random_state=seed).fit(
            DataLoader(development, batch_size=2)
        )
        for seed in (1, 2)
    )


def test_benchmark_ase_updates_are_isolated_between_objectives_and_repetitions(monkeypatch):
    auxiliary = SimpleNamespace(surrogates=fitted_surrogates(), proxy=nn.Identity())
    before = [deepcopy(model.forest.predict_proba([[3.0, 3.0]])) for model in auxiliary.surrogates]
    configs = benchmark.build_method_configs(auxiliary, 2, ase_update_every=3)
    ase_config = next(item for item in configs if item[0] is ActiveSurrogateEstimator)
    assert ase_config[1]["update_every"] == 3
    assert callable(ase_config[1]["surrogate_update"])
    pool = EvaluationPool(torch.ones(3, 2), torch.ones(3).long(), ["A"] * 3, ("group",))
    seen = []

    def fake_run(*, method, **kwargs):
        for model, expected in zip(method.surrogate, before):
            np.testing.assert_array_equal(model.forest.predict_proba([[3.0, 3.0]]), expected)
        seen.append(method.surrogate)
        method.surrogate_update(method.surrogate, torch.ones(3, 2) * 3, torch.ones(3).long())
        return []

    monkeypatch.setattr(benchmark, "run_active_evaluation", fake_run)
    for repetition in range(2):
        benchmark.run_repetition(
            seed=0,
            device="cpu",
            dataset=pool,
            model=nn.Identity(),
            method_configs=[ase_config],
            repetition=repetition,
            budget=3,
            protected_columns=("group",),
            include_accuracy=True,
        )
    assert len(seen) == 4
    assert len({id(models[0]) for models in seen}) == 4
    for model, expected in zip(auxiliary.surrogates, before):
        np.testing.assert_array_equal(model.forest.predict_proba([[3.0, 3.0]]), expected)


class FixedLogits(nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("logits", torch.tensor(probabilities, dtype=torch.float64).log())

    def forward(self, X):
        return self.logits.expand(len(X), -1)


def test_benchmark_kossen_and_ppat_share_the_same_predictive_mixture():
    auxiliary = SimpleNamespace(
        surrogates=(FixedLogits([0.9, 0.1]), FixedLogits([0.2, 0.8])), proxy=nn.Identity()
    )
    configs = dict(build_method_configs(auxiliary, num_classes=2))
    kossen = KossenActiveTesting(**configs[KossenActiveTesting])
    ppat = PredictionPoweredActiveTesting(**configs[PredictionPoweredActiveTesting])
    X = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    problem = Problem(3, 3, X, ["A"] * 3, nn.Identity(), nn.CrossEntropyLoss())
    kossen_state = kossen.initialize(problem)
    ppat_state = ppat.initialize(problem)
    expected_mixture = np.array([[0.55, 0.45]] * 3)
    kossen_probs = kossen.surrogate(X).softmax(dim=-1).numpy()
    ppat_probs = ppat_state["surrogate_members"].mean(axis=0)
    np.testing.assert_allclose(kossen_probs, expected_mixture)
    np.testing.assert_allclose(kossen_probs, ppat_probs)
    expected_scores = (ppat_probs * ppat_state["hypothetical_losses"]).sum(axis=1)
    np.testing.assert_allclose(kossen._acquisition_scores(kossen_state), expected_scores)
