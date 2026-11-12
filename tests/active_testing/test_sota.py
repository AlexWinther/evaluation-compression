import math
from statistics import NormalDist

from loguru import logger
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from active_testing_benchmark.active_testing.interface import Problem, SelectionPlan
from active_testing_benchmark.active_testing.runner import EvaluationPool, run_active_evaluation
from active_testing_benchmark.active_testing.sota import (
    ActiveSurrogateEstimator,
    PredictionPoweredActiveTesting,
)
from active_testing_benchmark.active_testing.surrogate import RandomForestModule


class FixedClassifier(nn.Module):
    def forward(self, X):
        value = X[:, :1]
        return torch.cat((0.7 * value, -0.2 * value), dim=1)


class FixedSurrogate(nn.Module):
    def forward(self, X):
        value = X[:, :1]
        return torch.cat((value, torch.ones_like(value) * 0.3), dim=1)


def _make_method(method_name, seed):
    if method_name == "ase":
        return ActiveSurrogateEstimator(
            FixedSurrogate(),
            num_classes=2,
            acquisition="expected_loss",
            seed=seed,
            batch_size=1,
        )
    return PredictionPoweredActiveTesting(
        num_classes=2,
        proxy_losses=[0.2, 0.7, 0.4, 1.1, 0.5, 0.9],
        lambda_value=0.4,
        exploration=0.2,
        seed=seed,
        batch_size=1,
    )


def _make_problem(memberships=None):
    X_pool = torch.arange(1, 7, dtype=torch.float32).reshape(-1, 1)
    labels = torch.tensor([0, 1, 1, 0, 1, 0])
    protected = ["first" if index % 2 == 0 else "second" for index in range(len(X_pool))]
    problem = Problem(
        N=len(X_pool),
        budget=len(X_pool),
        X_pool=X_pool,
        stratification_groups=protected,
        model=FixedClassifier(),
        loss_fn=nn.CrossEntropyLoss(),
        subgroup_indices=memberships,
    )
    target_logits = problem.model(X_pool)
    losses = F.cross_entropy(target_logits, labels, reduction="none").numpy()
    return problem, labels, losses


def _run(method_name, problem, labels, losses, seed):
    method = _make_method(method_name, seed)
    state = method.initialize(problem)
    estimates = [method.estimate(state, [])]
    selections = []
    while state["remaining_budget"]:
        plan = method.acquire(state, [])
        selections.extend(plan.indices)
        indices = plan.indices
        state = method.observe(state, plan, labels[indices], losses[indices])
        estimates.append(method.estimate(state, []))
    return selections, estimates


@pytest.mark.parametrize("method_name", ["ase", "ppat"])
def test_sota_estimators_support_overlapping_marginal_subgroups(method_name):
    memberships = {
        "left": (0, 1, 2, 3),
        "right": (2, 3, 4, 5),
        "sparse": (1, 4),
    }
    multi_problem, labels, losses = _make_problem(memberships)
    seed = 37

    multi_selections, multi_estimates = _run(method_name, multi_problem, labels, losses, seed)

    single_runs = {}
    for subgroup, indices in memberships.items():
        member_set = set(indices)
        single_problem, _, _ = _make_problem()
        single_problem.stratification_groups = [
            "member" if index in member_set else "outside" for index in range(single_problem.N)
        ]
        single_runs[subgroup] = _run(method_name, single_problem, labels, losses, seed)

    for subgroup, (single_selections, single_estimates) in single_runs.items():
        assert multi_selections == single_selections
        for multi_estimate, single_estimate in zip(multi_estimates, single_estimates):
            assert multi_estimate.risk == pytest.approx(single_estimate.risk, nan_ok=True)
            assert multi_estimate.std_error == pytest.approx(
                single_estimate.std_error, nan_ok=True
            )
            assert multi_estimate.subgroup_risks[subgroup] == pytest.approx(
                single_estimate.subgroup_risks["member"], nan_ok=True
            )
            assert (
                multi_estimate.subgroup_n_labels[subgroup]
                == (single_estimate.subgroup_n_labels["member"])
            )

    final = multi_estimates[-1]
    assert final.subgroup_n_labels == {name: len(indices) for name, indices in memberships.items()}
    if method_name == "ppat":
        for subgroup, indices in memberships.items():
            assert final.subgroup_risks[subgroup] == pytest.approx(losses[list(indices)].mean())
    else:
        # ASE reports the full-pool surrogate expected loss, even after all labels are seen.
        method = _make_method(method_name, seed)
        state = method.initialize(multi_problem)
        expected_losses = (
            state["surrogate_members"].mean(axis=0) * state["hypothetical_losses"]
        ).sum(axis=1)
        for subgroup, indices in memberships.items():
            assert final.subgroup_risks[subgroup] == pytest.approx(
                expected_losses[list(indices)].mean()
            )


def make_method_and_state(counts, pool_size=200, batch_size=25):
    def update(surrogate, X, y):
        counts.append(len(y))

    method = ActiveSurrogateEstimator(
        nn.Identity(),
        num_classes=2,
        acquisition="expected_loss",
        surrogate_update=update,
        update_every=50,
        batch_size=batch_size,
    )
    state = method.initialize(
        Problem(
            N=pool_size,
            budget=pool_size,
            X_pool=torch.ones(pool_size, 2),
            stratification_groups=None,
            model=nn.Identity(),
            loss_fn=nn.CrossEntropyLoss(),
        )
    )
    return method, state


@pytest.mark.parametrize("batch_size", [25, 30, 75])
def test_acquisition_refits_at_absolute_label_thresholds(batch_size):
    counts = []
    method, state = make_method_and_state(counts, batch_size=batch_size)
    while state["remaining_budget"]:
        selection = method.acquire(state, [])
        method.observe(
            state,
            selection,
            torch.zeros(len(selection.indices)).long(),
            [math.log(2)] * len(selection.indices),
        )
    assert counts == [50, 100, 150, 200]


def test_crossing_a_threshold_keeps_next_update_anchored_to_frequency():
    counts = []
    method, state = make_method_and_state(counts)
    # Observe externally supplied batches, including one that jumps past label 100.
    for start, end in [(0, 25), (25, 50), (50, 125), (125, 150), (150, 175), (175, 200)]:
        method.observe(
            state,
            SelectionPlan(list(range(start, end))),
            torch.zeros(end - start).long(),
            [math.log(2)] * (end - start),
        )
    assert counts == [50, 125, 150, 200]


def test_refit_logs_actual_count_and_scheduled_threshold():
    counts, messages = [], []
    method, state = make_method_and_state(counts)
    sink = logger.add(lambda message: messages.append(str(message)), level="TRACE")
    try:
        method.observe(
            state, SelectionPlan(list(range(125))), torch.zeros(125).long(), [math.log(2)] * 125
        )
    finally:
        logger.remove(sink)
    assert counts == [125]  # One fit with all new labels suffices when several thresholds pass.
    assert any(
        "125 acquired labels" in message and "threshold: 50" in message for message in messages
    )
    method.observe(
        state, SelectionPlan(list(range(125, 150))), torch.zeros(25).long(), [math.log(2)] * 25
    )
    assert counts == [125, 150]


@pytest.mark.parametrize("horizon", [2, 5, 9, 10])
def test_ppat_confidence_interval_matches_author_formula(horizon):
    N = 10
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]] * 5)
    labels = torch.zeros(N, dtype=torch.long)
    method = PredictionPoweredActiveTesting(
        num_classes=2, proxy_losses=np.linspace(0.2, 1.2, N), lambda_value=0.5
    )
    state = method.initialize(
        Problem(N, N, logits, ["A"] * N, nn.Identity(), nn.CrossEntropyLoss())
    )
    # A feasible sequence of conditional probabilities from a nonuniform proposal.
    scores = np.arange(1, N + 1, dtype=float)
    q = np.array([scores[i] / scores[i:].sum() for i in range(horizon)])
    losses = nn.functional.cross_entropy(logits[:horizon], labels[:horizon], reduction="none")
    method.observe(
        state,
        SelectionPlan(list(range(horizon)), selection_probabilities=q.tolist()),
        labels[:horizon],
        losses.tolist(),
    )
    estimate = method.estimate(state, [])
    if horizon == N:
        assert estimate.std_error == 0
        assert estimate.confidence_interval == (estimate.risk, estimate.risk)
        assert estimate.risk == pytest.approx(losses.mean().item())
        return

    residuals = losses.numpy() - 0.5 * (np.linspace(0.2, 1.2, N)[:horizon] - 0.7)
    m = np.arange(1, horizon + 1)
    # Authors' compute_ppat_prefix_cis: b_m = N(N-k)/[(N-m)(N-m+1)].
    b = N * (N - horizon) / ((N - m) * (N - m + 1))
    A = (residuals / q + np.r_[0.0, np.cumsum(residuals)[:-1]]) / N
    risk = np.mean(b * A)
    se = np.sqrt(np.mean(b**2 * (A - risk) ** 2) / horizon)
    assert estimate.risk == pytest.approx(risk)
    assert estimate.std_error == pytest.approx(se)
    z = NormalDist().inv_cdf(0.975)
    assert estimate.confidence_interval == pytest.approx((risk - z * se, risk + z * se))


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


def test_ase_refits_at_label_frequency_and_refreshes_estimates():
    from active_testing_benchmark.active_testing.surrogate import update_random_forest_surrogates

    counts = []

    def update(models, X, y):
        counts.append(len(y))
        update_random_forest_surrogates(models, X, y)

    X = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 2
    target = nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.zero_()
        target.bias.copy_(torch.tensor([2.0, 0.0]))
    pool = EvaluationPool(X, torch.ones(6, dtype=torch.long), ["A"] * 6, ("group",))
    method = ActiveSurrogateEstimator(
        fitted_surrogates(), num_classes=2, surrogate_update=update, update_every=3, batch_size=2
    )
    history = run_active_evaluation(method, pool, target, "cpu", nn.CrossEntropyLoss(), "group")
    assert [row["estimate"].n_labels for row in history] == [2, 3, 5, 6]
    assert counts == [3, 6]
    assert history[1]["estimate"].risk > history[0]["estimate"].risk
    assert history[2]["estimate"].risk == history[1]["estimate"].risk
