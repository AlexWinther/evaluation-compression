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
    KOSSEN_RETRAINING_SCHEDULE,
    ActiveSurrogateEstimator,
)
from active_testing_benchmark.active_testing.surrogate import LabelOracle, RandomForestModule
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


def method_kwargs(configs, method_class):
    return next(kwargs for cls, kwargs in configs if cls is method_class)


def test_methods_share_the_floor_lambda_and_forest_retraining_schedule():
    auxiliary = SimpleNamespace(
        surrogates=fitted_surrogates(), proxy=nn.Identity(), surrogate_kind="forest"
    )
    configs = build_method_configs(auxiliary, 2)
    kossen = method_kwargs(configs, KossenActiveTesting)
    ase = method_kwargs(configs, ActiveSurrogateEstimator)
    ppat = method_kwargs(configs, PredictionPoweredActiveTesting)
    assert kossen["exploration"] == ppat["exploration"] == 0.2
    assert ppat["lambda_value"] == 0.5
    assert ase["update_schedule"] == KOSSEN_RETRAINING_SCHEDULE
    assert "update_every" not in ase
    assert ase["surrogate_update"] is benchmark.update_random_forest_surrogates
    # One shared surrogate: AT averages the members that ASE and PPAT use.
    assert list(kossen["surrogate"].members) == ase["surrogate"] == ppat["surrogate"]


def test_deep_ensemble_surrogate_is_never_refitted_or_copied(monkeypatch):
    members = (FixedLogits([0.7, 0.3]), FixedLogits([0.6, 0.4]))
    auxiliary = SimpleNamespace(
        surrogates=members, proxy=nn.Identity(), surrogate_kind="resnet_ensemble"
    )
    configs = build_method_configs(auxiliary, 2, ase_update_every=10)
    ase_config = next(item for item in configs if item[0] is ActiveSurrogateEstimator)
    assert "surrogate_update" not in ase_config[1]
    seen = []

    def fake_run(*, method, **kwargs):
        seen.append(method.surrogate)
        return []

    monkeypatch.setattr(benchmark, "run_active_evaluation", fake_run)
    pool = EvaluationPool(torch.ones(3, 2), torch.ones(3).long(), ["A"] * 3, ("group",))
    benchmark.run_repetition(
        seed=0,
        device="cpu",
        dataset=pool,
        model=nn.Identity(),
        method_configs=[ase_config],
        budget=3,
        protected_columns=("group",),
    )
    assert all(member is original for member, original in zip(seen[0], members))


@pytest.mark.parametrize("frequency,valid", [(1, True), (0, False), (-2, False)])
def test_ase_update_frequency_accepts_every_positive_value(frequency, valid):
    auxiliary = SimpleNamespace(surrogates=fitted_surrogates(), proxy=nn.Identity())
    if not valid:
        with pytest.raises(ValueError, match="positive"):
            build_method_configs(auxiliary, 2, ase_update_every=frequency)
        return
    ase = method_kwargs(build_method_configs(auxiliary, 2, ase_update_every=frequency),
                        ActiveSurrogateEstimator)
    assert ase["update_every"] == 1


class LabelledDataset:
    """Twelve two-feature images, a quarter of them positive, with a gender per image."""

    def __init__(self, size=12):
        self.size = size
        self.reads = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads += 1
        return {
            "image": torch.tensor([float(index % 3), float(index % 2)]),
            "label": int(index % 4 == 0),
            "subgroups": {"gender": "female" if index % 2 else "male"},
        }


def test_class_stratified_positions_keep_class_shares_and_depend_only_on_the_seed():
    labels = np.array([0] * 70 + [1] * 20 + [2] * 10)
    positions = benchmark.class_stratified_positions(labels, 30, benchmark.pool_rng(7, 0))
    assert len(set(positions)) == 30
    assert list(positions) == sorted(positions)
    assert np.bincount(labels[positions]).tolist() == [21, 6, 3]
    again = benchmark.class_stratified_positions(labels, 30, benchmark.pool_rng(7, 0))
    other = benchmark.class_stratified_positions(labels, 30, benchmark.pool_rng(7, 1))
    np.testing.assert_array_equal(positions, again)
    assert not np.array_equal(positions, other)
    # Largest remainders fill the places that rounding down leaves open.
    shares = benchmark.class_stratified_positions(labels, 11, benchmark.pool_rng(7, 0))
    assert np.bincount(labels[shares]).tolist() == [8, 2, 1]
    with pytest.raises(ValueError, match="pool size"):
        benchmark.class_stratified_positions(labels, 101, benchmark.pool_rng(7, 0))


def test_subset_pool_indexes_inputs_labels_and_memberships_together():
    from active_testing_benchmark.active_testing.runner import subset_evaluation_pool

    pool = prepare_evaluation_pool(LabelledDataset(), "cpu", ["gender"])
    subset = subset_evaluation_pool(pool, [8, 1, 4])
    assert subset.X_pool.tolist() == pool.X_pool[[8, 1, 4]].tolist()
    assert subset.y_pool.tolist() == [1, 0, 1]
    assert subset.attribute_values["gender"] == ["gender=male", "gender=female", "gender=male"]
    assert subset.protected_columns == ("gender",)
    with pytest.raises(ValueError, match="unique"):
        subset_evaluation_pool(pool, [1, 1])
    with pytest.raises(ValueError, match="within the pool"):
        subset_evaluation_pool(pool, [12])


def test_cached_predictions_match_the_modules_and_refit_forests_in_isolation():
    from active_testing_benchmark.active_testing.prediction_cache import cache_pool_predictions
    from active_testing_benchmark.active_testing.surrogate import (
        update_random_forest_surrogates,
    )

    torch.manual_seed(0)
    pool = prepare_evaluation_pool(LabelledDataset(), "cpu", ["gender"])
    model = nn.Linear(2, 2)
    extractor = nn.Linear(2, 3)
    development = [
        {"image": torch.tensor([float(i % 3), float(i % 2)]), "label": i % 2} for i in range(8)
    ]
    forests = tuple(
        RandomForestModule(2, extractor, n_estimators=5, random_state=seed).fit(
            DataLoader(development, batch_size=4)
        )
        for seed in (1, 2)
    )
    auxiliary = SimpleNamespace(surrogates=(*forests, FixedLogits([0.3, 0.7])), proxy=model)

    index_pool, cached_model, cached_auxiliary = cache_pool_predictions(pool, model, auxiliary)

    assert index_pool.X_pool.tolist() == list(range(12))
    assert index_pool.y_pool is pool.y_pool
    indices = torch.tensor([5, 0, 11])
    images = pool.X_pool[indices]
    with torch.inference_mode():
        torch.testing.assert_close(cached_model(indices), model(images))
        torch.testing.assert_close(cached_auxiliary.proxy(indices), model(images))
        for cached, original in zip(cached_auxiliary.surrogates, auxiliary.surrogates):
            torch.testing.assert_close(cached(indices), original(images))
    # The forests share one feature lookup and keep their fitted state.
    cached_forests = cached_auxiliary.surrogates[:2]
    assert cached_forests[0].feature_extractor is cached_forests[1].feature_extractor
    assert all(forest.feature_extractor is extractor for forest in forests)

    # Refitting through the cache equals refitting from images, and leaves the originals.
    reference = deepcopy(forests)
    labels = torch.tensor([1, 0, 1])
    update_random_forest_surrogates(cached_forests, indices, labels)
    update_random_forest_surrogates(reference, images, labels)
    with torch.inference_mode():
        for cached, refitted in zip(cached_forests, reference):
            torch.testing.assert_close(cached(indices), refitted(images))
    for forest in forests:
        assert len(forest.forest.estimators_[0].tree_.value) > 0
        assert forest._development_features.shape == (8, 3)


def run_small_benchmark(dataset, **kwargs):
    defaults = dict(
        model=nn.Identity(),
        dataset=dataset,
        auxiliary=SimpleNamespace(
            surrogates=(FixedLogits([0.7, 0.3]), FixedLogits([0.4, 0.6])),
            proxy=FixedLogits([0.5, 0.5]),
            surrogate_kind="resnet_ensemble",
        ),
        num_classes=2,
        device="cpu",
        seed=3,
        repetitions=3,
        budget=4,
        protected_columns=("gender",),
        stratification_attribute="gender",
    )
    return benchmark.run_benchmark(**{**defaults, **kwargs})


def pool_losses(dataset):
    pool = prepare_evaluation_pool(dataset, "cpu", ["gender"])
    losses = nn.functional.cross_entropy(pool.X_pool, pool.y_pool, reduction="none")
    # Per-item losses in double precision, as the runner sums `loss.item()` values.
    return pool, losses.numpy().astype(np.float64)


def test_frozen_pool_rows_store_the_full_pool_reference():
    dataset = LabelledDataset()
    curves = run_small_benchmark(dataset)
    # The images are read once, although three repetitions evaluate them.
    assert dataset.reads == 12
    pool, losses = pool_losses(dataset)
    assert set(curves["pool_size"]) == {12}
    np.testing.assert_allclose(curves["reference_risk"], losses.mean())
    female = np.array(pool.attribute_values["gender"]) == "gender=female"
    np.testing.assert_allclose(curves["reference_risk gender=female"], losses[female].mean())
    assert set(curves["pool_n_gender=female"]) == {6}
    # At the full budget every label-based estimate recovers the reference. ASE with a
    # fixed surrogate predicts from the surrogate alone, as in "ASE (const.)".
    full = run_small_benchmark(LabelledDataset(), budget=12, repetitions=1)
    final = full[(full.n_labels == 12) & (full.method != "ActiveSurrogateEstimator")]
    assert len(final) == 5
    np.testing.assert_allclose(final["risk"], final["reference_risk"])


def test_a_pool_of_the_whole_dataset_reproduces_the_frozen_pool():
    frozen = run_small_benchmark(LabelledDataset())
    whole = run_small_benchmark(LabelledDataset(), pool_size=12)
    pd_testing = pytest.importorskip("pandas.testing")
    pd_testing.assert_frame_equal(frozen, whole)


def test_resampled_pools_are_stratified_shared_by_methods_and_scored_against_themselves():
    dataset = LabelledDataset()
    curves = run_small_benchmark(dataset, pool_size=8, repetitions=4)
    _, losses = pool_losses(dataset)
    assert set(curves["pool_size"]) == {8}
    references = curves.groupby("repetition")["reference_risk"].agg(["min", "max"])
    # All methods and designs of one repetition share its pool.
    np.testing.assert_allclose(references["min"], references["max"])
    # Different repetitions draw different pools from the 12 images.
    assert references["min"].nunique() > 1
    labels = np.array([int(i % 4 == 0) for i in range(12)])
    for repetition in range(4):
        positions = benchmark.class_stratified_positions(
            labels, 8, benchmark.pool_rng(3, repetition)
        )
        assert np.bincount(labels[positions]).tolist() == [6, 2]
        rows = curves[curves.repetition == repetition + 1]
        np.testing.assert_allclose(rows["reference_risk"], losses[positions].mean())
    # The budget of 4 stops before the pool of 8 is exhausted.
    assert curves["n_labels"].max() == 4


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"pool_size": 13}, "pool_size"),
        ({"pool_size": 0}, "pool_size"),
        ({"pool_size": 3, "budget": 4}, "budget"),
        ({"budget": 13}, "budget"),
    ],
)
def test_benchmark_rejects_pools_and_budgets_outside_the_dataset(arguments, message):
    with pytest.raises(ValueError, match=message):
        run_small_benchmark(LabelledDataset(), **arguments)


def test_cached_oracles_are_built_from_the_pool_labels_and_model_outputs():
    from active_testing_benchmark.active_testing.prediction_cache import cache_pool_predictions

    torch.manual_seed(0)
    pool = prepare_evaluation_pool(LabelledDataset(), "cpu", ["gender"])
    model = nn.Linear(2, 2)
    auxiliary = SimpleNamespace(
        surrogates=(LabelOracle(2, "oracle"), LabelOracle(2, "model_noisy_oracle", 1.0)),
        proxy=LabelOracle(2, "oracle", hard_labels=True),
    )
    index_pool, cached_model, cached = cache_pool_predictions(pool, model, auxiliary)
    indices = index_pool.X_pool
    exact, model_copy = cached.surrogates
    assert exact(indices).argmax(dim=1).tolist() == pool.y_pool.tolist()
    assert cached.proxy(indices).argmax(dim=1).tolist() == pool.y_pool.tolist()
    torch.testing.assert_close(
        model_copy(indices).softmax(dim=1), cached_model(indices).softmax(dim=1), rtol=1e-4, atol=0
    )
    # The originals stay oracles, so another pool gets its own labels.
    assert isinstance(auxiliary.surrogates[0], LabelOracle)


def test_oracle_auxiliaries_remove_the_estimation_error_of_ase_and_ppat():
    dataset = LabelledDataset(24)
    oracle = SimpleNamespace(
        surrogates=(LabelOracle(2, "oracle"),),
        proxy=LabelOracle(2, "oracle", hard_labels=True),
        surrogate_kind="oracle",
    )
    curves = run_small_benchmark(dataset, auxiliary=oracle, budget=8, repetitions=5)
    error = (curves["risk"] - curves["reference_risk"]).abs()
    # ASE predicts every loss from the surrogate, so an exact oracle gives the exact
    # risk before any label. It is never refitted.
    ase = curves.method == "ActiveSurrogateEstimator"
    np.testing.assert_allclose(error[ase], 0.0, atol=1e-5)
    # The oracle proxy loss equals the true loss. With lambda = 1/2 the control variate
    # removes half of every residual, so PPAT beats LURE with the same proposal floor.
    final = curves[curves.n_labels == 8]
    mse = ((final["risk"] - final["reference_risk"]) ** 2).groupby(final["method"]).mean()
    assert mse["PredictionPoweredActiveTesting"] < mse["UniformSampleMean"]
    assert mse["KossenActiveTesting"] < mse["UniformSampleMean"]


def test_acquisition_batch_size_sets_every_methods_step_and_rejects_zero():
    curves = run_small_benchmark(LabelledDataset(24), budget=12, acquisition_batch_size=3)
    for _, method_curves in curves.groupby(["method", "repetition"]):
        assert method_curves["n_labels"].tolist() == [3, 6, 9, 12]
    default = run_small_benchmark(LabelledDataset(24), budget=12, repetitions=1)
    # The default step of 25 labels exceeds this budget, so one step acquires it all.
    assert set(default["n_labels"]) == {12}
    auxiliary = SimpleNamespace(surrogates=(FixedLogits([0.5, 0.5]),), proxy=nn.Identity())
    configs = build_method_configs(auxiliary, 2, 7)
    assert {kwargs["batch_size"] for _, kwargs in configs} == {7}
    with pytest.raises(ValueError, match="acquisition_batch_size"):
        run_small_benchmark(LabelledDataset(24), acquisition_batch_size=0)
