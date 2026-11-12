import math

import numpy as np
import pytest
import torch
from torch import nn

from active_testing_benchmark.active_testing.baseline import StratifiedSampleMean
from active_testing_benchmark.active_testing.interface import Problem

STRATA = ["A=small"] * 2 + ["A=large"] * 6


LOSSES = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


GROUPS = {
    "A=small": (0, 1),
    "A=large": (2, 3, 4, 5, 6, 7),
    "B=cross": (0, 2, 3, 4),
    "B=other": (1, 5, 6, 7),
    "sparse": (0, 7),
}


def run_prefixes(allocation, groups, batch_size=1):
    problem = Problem(
        len(STRATA),
        len(STRATA),
        torch.zeros((len(STRATA), 1)),
        STRATA,
        nn.Identity(),
        nn.MSELoss(),
        subgroup_indices=groups,
    )
    method = StratifiedSampleMean(allocation=allocation, seed=11, batch_size=batch_size)
    state = method.initialize(problem)
    selections = []
    estimates = [method.estimate(state, [])]
    while state["remaining_budget"]:
        plan = method.acquire(state, [])
        selections.extend(plan.indices)
        method.observe(state, plan, [], LOSSES[plan.indices])
        estimates.append(method.estimate(state, []))
    return selections, estimates


@pytest.mark.parametrize("allocation", ["proportional", "balanced"])
def test_marginal_estimates_use_overall_total_weights_and_known_group_sizes(allocation):
    selections, estimates = run_prefixes(allocation, GROUPS)
    saw_non_naive = False
    for n, estimate in enumerate(estimates):
        selected = selections[:n]
        stratum_counts = {g: sum(STRATA[i] == g for i in selected) for g in set(STRATA)}
        for group, members in GROUPS.items():
            observed = [i for i in selected if i in members]
            assert estimate.subgroup_n_labels[group] == len(observed)
            if not observed:
                assert estimate.subgroup_risks[group] is None
                continue
            assert math.isfinite(estimate.subgroup_risks[group])
            # Sampling strata retain their direct sample mean.
            if group in {"A=small", "A=large"}:
                expected = np.mean(LOSSES[observed])
            else:
                if allocation == "proportional":
                    weights = [len(STRATA) / n for i in observed]
                elif n < len(stratum_counts):
                    weights = [len(stratum_counts) * STRATA.count(STRATA[i]) / n for i in observed]
                else:
                    weights = [
                        STRATA.count(STRATA[i]) / stratum_counts[STRATA[i]] for i in observed
                    ]
                expected = sum(weight * LOSSES[i] for weight, i in zip(weights, observed)) / len(
                    members
                )
                if group == "B=cross" and not np.isclose(expected, np.mean(LOSSES[observed])):
                    saw_non_naive = True
            assert estimate.subgroup_risks[group] == pytest.approx(expected)
    assert saw_non_naive

    for group, members in GROUPS.items():
        assert estimates[-1].subgroup_risks[group] == pytest.approx(np.mean(LOSSES[list(members)]))


@pytest.mark.parametrize("allocation", ["proportional", "balanced"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_reporting_memberships_do_not_change_sampling_or_overall_estimates(allocation, batch_size):
    multi_order, multi = run_prefixes(allocation, GROUPS, batch_size)
    single_order, single = run_prefixes(allocation, None, batch_size)
    assert multi_order == single_order
    for multiple, partition in zip(multi, single):
        assert multiple.risk == pytest.approx(partition.risk, nan_ok=True)
        for group in ("A=small", "A=large"):
            assert multiple.subgroup_risks[group] == partition.subgroup_risks[group]
            assert multiple.subgroup_n_labels[group] == partition.subgroup_n_labels[group]
