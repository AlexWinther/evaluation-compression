import heapq
from bisect import bisect_left
from collections import defaultdict
from typing import Hashable, Iterable

import numpy as np
from loguru import logger

from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Estimate,
    SelectionPlan,
)


class StratifiedSampleMean(ActiveEvaluationMethod):
    """Stratified sampling by protected groups, with prefix risk estimates.

    allocation="proportional": population-proportional group order with a
        uniformly random rotation. The ordinary sample mean is unbiased
        at every nonempty prefix.

    allocation="balanced": randomized round-robin group order until some
        groups run out. Use inverse inclusion probability for first K-1
        observations and population-weighted stratum means thereafter.

    Within every stratum, indices are sampled uniformly without replacement.
    """

    def __init__(
        self,
        protected_attributes: Iterable[Hashable] | None = None,
        allocation: str = "proportional",
        seed: int = 0,
        batch_size: int | None = 1,
    ) -> None:
        if allocation not in {"proportional", "balanced"}:
            raise ValueError("allocation must be 'proportional' or 'balanced'")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")
        self.protected_attributes = (
            None if protected_attributes is None else list(protected_attributes)
        )
        self.allocation = allocation
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

    def _proportional_slots(self, sizes):
        """Generate a randomized order of subgroup slots sampling proportionally to their size."""
        rank = self.rng.permutation(len(sizes))
        heap = []
        for r, (g, size) in zip(rank, sizes.items()):
            heapq.heappush(heap, (0.5 / size, int(r), g, 1))

        slots = []
        while heap:
            _, r, g, used = heapq.heappop(heap)
            slots.append(g)
            if used < sizes[g]:
                heapq.heappush(heap, ((used + 0.5) / sizes[g], r, g, used + 1))

        # Offset the slots by a random amount to ensure uniformity across starting points
        offset = int(self.rng.integers(len(slots)))
        final_slots = slots[offset:] + slots[:offset]

        logger.debug(f"Proportional slots: {final_slots}, offset: {offset}")

        return final_slots

    def _balanced_slots(self, sizes):
        """Generate a randomized order of subgroup slots sampling in a balanced
           round-robin manner with uniformity within each cycle."""
        slots, group_draw_prob, cycle_ends = [], [], []
        active = list(sizes.keys())
        cycle_number = 0

        while active:
            order = self.rng.permutation(len(active))
            for j, position in enumerate(order):
                slots.append(active[int(position)])
                group_draw_prob.append(1.0 / (len(active) - j))
            cycle_ends.append(len(slots))
            cycle_number += 1

            # Remove groups from active when they have been fully selected
            active = [g for g in active if sizes[g] > cycle_number]

        logger.debug(
            f"Balanced slots: {slots}, group_draw_prob: {group_draw_prob}, cycle_ends: {cycle_ends}"
        )
        return slots, group_draw_prob, cycle_ends

    def initialize(self, problem: dict) -> dict:
        N = int(problem["N"])
        budget = int(problem["budget"])
        if not 0 < N or not 0 <= budget <= N:
            raise ValueError("Require N > 0 and 0 <= budget <= N")
        raw_groups = (
            self.protected_attributes
            if self.protected_attributes is not None
            else problem["protected_attribute"]
        )
        groups = list(raw_groups)
        if len(groups) != N:
            raise ValueError("Protected attributes must match X_pool order and length")

        by_group = defaultdict(list)
        for i, group in enumerate(groups):
            # Group IDs should be hashable (e.g. int, str, or tuple).
            by_group[group].append(i)
        sizes = {g: len(indices) for g, indices in by_group.items()}
        orders = {
            g: [int(i) for i in self.rng.permutation(indices)]
            for g, indices in by_group.items()
        }

        if self.allocation == "balanced":
            slots, group_draw_prob, cycle_ends = self._balanced_slots(sizes)
        else:
            slots = self._proportional_slots(sizes)
            group_draw_prob, cycle_ends = None, None

        return {
            "N": N,
            "remaining": set(range(problem["N"])),
            "remaining_budget": problem["budget"],
            "observed_losses": [],
            "selections": set(),
            "group_by_index": groups,
            "sizes": sizes,
            "orders": orders,
            "slots": slots,
            "group_draw_prob": group_draw_prob,
            "cycle_ends": cycle_ends,
            "cursor": 0,
            "counts": {g: 0 for g in sizes},
            "loss_sums": {g: 0.0 for g in sizes},
            "loss_total": 0.0,
        }

    def _balanced_inclusion(self, state, group, n):
        cycle_index = bisect_left(state["cycle_ends"], n)
        cycle_start_idx = 0 if cycle_index == 0 else state["cycle_ends"][cycle_index - 1]
        selections_made_in_cycle = n - cycle_start_idx
        current_active_groups = state["cycle_ends"][cycle_index] - cycle_start_idx
        current_group_count = min(state["sizes"][group], cycle_index)
        if state["sizes"][group] > cycle_index:
            expected_group_count = current_group_count + selections_made_in_cycle / current_active_groups
        return expected_group_count / state["sizes"][group]

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        selection_count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            state["N"] - state["cursor"],
        )
        if selection_count < 1:
            raise ValueError("No remaining budget or unlabeled items")

        start_idx = state["cursor"]
        batch_group_counts = defaultdict(int)
        indices, selection_probs = [], []
        selected_groups = []

        for t in range(start_idx, start_idx + selection_count):
            group = state["slots"][t]
            used = state["counts"][group] + batch_group_counts[group]
            indices.append(state["orders"][group][used])
            selected_groups.append(group)
            if self.allocation == "balanced":
                p_group = state["group_draw_prob"][t]
                selection_probs.append(p_group / (state["sizes"][group] - used))
            batch_group_counts[group] += 1

        n_after = start_idx + selection_count
        if self.allocation == "proportional":
            inclusion_probs = [
                (n_after - offset) / state["N"] for offset in range(selection_count)
            ]
        else:
            inclusion_probs = [
                self._balanced_inclusion(state, state["group_by_index"][i], offset)
                for offset, i in enumerate(indices, start_idx + 1)
            ]

        return SelectionPlan(
            indices=indices,
            selection_probabilities=(
                selection_probs if self.allocation == "balanced" else None
            ),
            inclusion_probabilities=inclusion_probs,
            protected_attribute_group=selected_groups,
            metadata={
                "design": f"stratified_{self.allocation}_without_replacement",
                "inclusion_horizon": n_after,
            },
        )

    def observe(self, state: dict, selection: SelectionPlan, labels, losses) -> dict:
        if len(selection.indices) != len(losses):
            raise ValueError("One loss is required per selected index")
        for i, loss in zip(selection.indices, losses):
            group = state["group_by_index"][i]
            state["counts"][group] += 1
            state["loss_sums"][group] += float(loss)
            state["loss_total"] += float(loss)
        state["selections"].update(selection.indices)
        state["cursor"] += len(selection.indices)
        state["remaining_budget"] -= len(selection.indices)
        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        n, N = state["cursor"], state["N"]
        if n == 0:
            return Estimate(risk=float("nan"), n_labels=0)
        if self.allocation == "proportional":
            risk = state["loss_total"] / n
        elif n < len(state["sizes"]):
            # Early design-unbiased Horvitz--Thompson estimate: first
            # cycle selects n of K strata uniformly without replacement.
            K = len(state["sizes"])
            risk = (
                K
                / (n * N)
                * sum(
                    size * state["loss_sums"][g] for g, size in state["sizes"].items()
                )
            )
        else:
            # All groups represented after the first complete cycle.
            risk = sum(
                (size / N) * (state["loss_sums"][g] / state["counts"][g])
                for g, size in state["sizes"].items()
            )
        return Estimate(risk=float(risk), n_labels=n)

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        n, N = state["cursor"], state["N"]

        # Estimate mean loss within each protected group.
        subgroup_risks = {
            g: (
                state["loss_sums"][g] / state["counts"][g]
                if state["counts"][g] > 0
                else None
            )
            for g in state["sizes"]
        }

        subgroup_n_labels = dict(state["counts"])

        # Existing overall risk estimator.
        if n == 0:
            risk = float("nan")

        elif self.allocation == "proportional":
            risk = state["loss_total"] / n

        elif n < len(state["sizes"]):
            K = len(state["sizes"])

            risk = K / (n * N) * sum(
                size * state["loss_sums"][g]
                for g, size in state["sizes"].items()
            )

        else:
            risk = sum(
                (size / N)
                * (state["loss_sums"][g] / state["counts"][g])
                for g, size in state["sizes"].items()
            )

        return Estimate(
            risk=float(risk),
            n_labels=n,
            subgroup_risks=subgroup_risks,
            subgroup_n_labels=subgroup_n_labels,
        )


class UniformSampleMean(ActiveEvaluationMethod):
    """Uniform sampling without replacement and the mean of observed losses."""

    def __init__(self, seed: int = 0, batch_size: int | None = 1) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.batch_size = batch_size

    def initialize(self, problem: dict) -> dict:
        groups = list(problem["protected_attribute"])

        state = {
            "N": int(problem["N"]),
            "remaining": set(range(problem["N"])),
            "remaining_budget": int(problem["budget"]),
            "observed_losses": [],
            "selections": set(),
            "group_by_index": groups,
            "subgroup_counts": {g: 0 for g in groups},
            "subgroup_loss_sums": {g: 0.0 for g in groups}
        }

        return state

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        remaining = sorted(state["remaining"])
        count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            len(remaining),
        )
        if count < 1:
            raise ValueError("No remaining budget or unlabeled items")

        positions = self.rng.choice(len(remaining), size=count, replace=False)
        indices = [int(remaining[int(position)]) for position in positions]
        return SelectionPlan(
            indices=indices,
            selection_probabilities=[
                1.0 / (len(remaining) - offset) for offset in range(count)
            ],
            inclusion_probabilities=[
                (len(state["selections"]) + offset) / state["N"]
                for offset in range(1, count + 1)
            ],
            protected_attribute_group=[state["group_by_index"][i] for i in indices],
            metadata={"design": "uniform_without_replacement"},
        )

    def observe(
        self, state: dict, selection: SelectionPlan, labels: list, losses: list
    ) -> dict:
        state["remaining"].difference_update(selection.indices)
        state["remaining_budget"] -= len(selection.indices)
        state["observed_losses"].extend(losses)
        state["selections"].update(selection.indices)

        for index, loss in zip(selection.indices, losses):
            group = state["group_by_index"][index]
            state["subgroup_counts"][group] += 1
            state["subgroup_loss_sums"][group] += float(loss)

        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        losses = state["observed_losses"]
        counts = state["subgroup_counts"]

        subgroup_risks = {
            g: (
                state["subgroup_loss_sums"][g] / count
                if count > 0
                else None
            )
            for g, count in counts.items()
        }

        return Estimate(
            risk=float(np.mean(losses)) if losses else float("nan"),
            n_labels=len(losses),
            subgroup_risks=subgroup_risks,
            subgroup_n_labels=dict(counts),
        )