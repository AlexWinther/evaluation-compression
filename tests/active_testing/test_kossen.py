import numpy as np
import pytest
import torch
from torch import nn

from active_testing_benchmark.active_testing.interface import Problem
from active_testing_benchmark.active_testing.kossen import KossenActiveTesting

N = 8


X_POOL = torch.tensor(
    [
        [2.0, 0.0],
        [0.0, 2.0],
        [1.5, 0.0],
        [0.0, 1.5],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ]
)


LABELS = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])


ATTRIBUTE_A = ["A=x"] * 4 + ["A=y"] * 4


ATTRIBUTE_B = ["B=p", "B=p", "B=q", "B=q", "B=p", "B=p", "B=q", "B=q"]


MARGINAL_GROUPS = {
    "A=x": (0, 1, 2, 3),
    "A=y": (4, 5, 6, 7),
    "B=p": (0, 1, 4, 5),
    "B=q": (2, 3, 6, 7),
}


def run_prefixes(stratification_groups, *, subgroup_indices=None, seed=17):
    model = nn.Identity()
    problem = Problem(
        N=N,
        budget=N,
        X_pool=X_POOL,
        stratification_groups=stratification_groups,
        model=model,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        subgroup_indices=subgroup_indices,
    )
    method = KossenActiveTesting(num_classes=2, seed=seed, batch_size=1)
    state = method.initialize(problem)
    selected_order = []
    estimates = [method.estimate(state, [])]

    while state["remaining"]:
        selection = method.acquire(state, [])
        indices = selection.indices
        labels = LABELS[indices]
        losses = nn.functional.cross_entropy(X_POOL[indices], labels, reduction="none")
        method.observe(state, selection, labels, losses)
        selected_order.extend(indices)
        estimates.append(method.estimate(state, []))

    return selected_order, estimates


def test_overlapping_marginals_match_independent_runs_at_every_prefix():
    combined_order, combined = run_prefixes(ATTRIBUTE_A, subgroup_indices=MARGINAL_GROUPS)
    a_order, a_estimates = run_prefixes(ATTRIBUTE_A)
    b_order, b_estimates = run_prefixes(ATTRIBUTE_B)

    assert combined_order == a_order == b_order
    for prefix, (combined_estimate, a_estimate, b_estimate) in enumerate(
        zip(combined, a_estimates, b_estimates)
    ):
        assert combined_estimate.n_labels == prefix
        if prefix == 0:
            assert np.isnan(combined_estimate.risk)
            assert np.isnan(a_estimate.risk)
            assert np.isnan(b_estimate.risk)
        else:
            assert combined_estimate.risk == pytest.approx(a_estimate.risk)
            assert combined_estimate.risk == pytest.approx(b_estimate.risk)
        for group in MARGINAL_GROUPS:
            reference = a_estimate if group.startswith("A=") else b_estimate
            if reference.subgroup_risks[group] is None:
                assert combined_estimate.subgroup_risks[group] is None
            else:
                assert combined_estimate.subgroup_risks[group] == pytest.approx(
                    reference.subgroup_risks[group]
                )
            assert (
                combined_estimate.subgroup_n_labels[group] == (reference.subgroup_n_labels[group])
            )


def test_full_census_recovers_true_loss_for_every_overlapping_subgroup():
    _, estimates = run_prefixes(ATTRIBUTE_A, subgroup_indices=MARGINAL_GROUPS)
    estimate = estimates[-1]
    losses = nn.functional.cross_entropy(X_POOL, LABELS, reduction="none").numpy()

    assert estimate.risk == pytest.approx(float(losses.mean()))
    for group, indices in MARGINAL_GROUPS.items():
        assert estimate.subgroup_risks[group] == pytest.approx(
            float(np.mean(losses[list(indices)]))
        )
        assert estimate.subgroup_n_labels[group] == len(indices)


def test_unseen_marginals_are_none_and_counts_include_each_membership():
    problem = Problem(
        N=N,
        budget=N,
        X_pool=X_POOL,
        stratification_groups=ATTRIBUTE_A,
        model=nn.Identity(),
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        subgroup_indices=MARGINAL_GROUPS,
    )
    method = KossenActiveTesting(num_classes=2, seed=17)
    state = method.initialize(problem)
    selection = method.acquire(state, [])
    selected = selection.indices[0]
    labels = LABELS[[selected]]
    losses = nn.functional.cross_entropy(X_POOL[[selected]], labels, reduction="none")
    method.observe(state, selection, labels, losses)
    estimate = method.estimate(state, [])

    for group, indices in MARGINAL_GROUPS.items():
        expected_count = int(selected in indices)
        assert estimate.subgroup_n_labels[group] == expected_count
        assert state["subgroup_counts"][group] == expected_count
        if expected_count:
            assert estimate.subgroup_risks[group] is not None
        else:
            assert estimate.subgroup_risks[group] is None
    assert sum(estimate.subgroup_n_labels.values()) == 2
