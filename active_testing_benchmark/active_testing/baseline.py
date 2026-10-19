from collections import defaultdict
import heapq

from loguru import logger
import numpy as np

from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Estimate,
    Problem,
    SelectionPlan,
)


class StratifiedSampleMean(ActiveEvaluationMethod):
    """Stratified sampling by protected groups, with intermediate risk estimates.

    allocation="proportional": population-proportional group order with a
        uniformly random rotation. The ordinary sample mean is unbiased
        at every nonempty prefix.

    allocation="balanced": randomized round-robin group order until some
        groups run out. Use a group-size-weighted estimate for the first K-1
        observations and population-weighted stratum means thereafter.

    Within every stratum, indices are sampled uniformly without replacement.
    """

    def __init__(
        self,
        allocation: str = "proportional",
        seed: int = 0,
        batch_size: int | None = 1,
    ) -> None:
        if allocation not in {"proportional", "balanced"}:
            raise ValueError("allocation must be 'proportional' or 'balanced'")

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive or None")

        self.allocation = allocation
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

    def _proportional_slots(self, sizes: dict[str, int]) -> list[str]:
        """Generate the full list of subgroup slots sampling proportionally to their group size."""
        # give ranks to each group to break ties
        ranks = self.rng.permutation(len(sizes))

        heap = []
        for rank, (group, size) in zip(ranks, sizes.items()):
            # add groups to heap with priority inverse to their size
            priority = 0.5 / size
            heapq.heappush(heap, (priority, int(rank), group, 1))

        slots = []
        while heap:
            # pick next group to sample from (minheap so lower priority is picked first)
            _, rank, group, used = heapq.heappop(heap)
            slots.append(group)

            # If group has been sampled less than group size then add again with updated priority
            if used < sizes[group]:
                priority = (used + 0.5) / sizes[group]
                heapq.heappush(heap, (priority, rank, group, used + 1))

        # Offset the slots by a random amount to ensure uniformity across starting points
        # Ensures not picking largest group first every time
        offset = int(self.rng.integers(len(slots)))
        final_slots = slots[offset:] + slots[:offset]

        logger.debug(f"Proportional slots: {final_slots}, offset: {offset}")

        return final_slots

    def _balanced_slots(self, sizes):
        """Generate a randomized order of subgroup slots sampling to visit
        still-active groups once per cycle, while shufling group order within
        each cycle."""
        slots = []

        # At the start all groups are "active" meaning they have remaining samples
        active = list(sizes.keys())
        cycle_number = 0

        while active:
            # Shufle the group order in each cycle
            order = self.rng.permutation(len(active))

            for position in order:
                slots.append(active[int(position)])

            cycle_number += 1

            # Update still active groups
            active = [g for g in active if sizes[g] > cycle_number]

        logger.debug(f"Balanced slots: {slots}")

        return slots

    def initialize(self, problem: Problem) -> dict:
        N = int(problem.N)
        budget = problem.budget
        if not 0 < N or not 0 <= budget <= N:
            raise ValueError("Require N > 0 and 0 <= budget <= N")

        groups = problem.protected_attribute
        if groups is None or len(groups) != N:
            raise ValueError("Protected attributes must match X_pool order and length")

        # match sample idx to groups
        by_group = defaultdict(list)
        for i, group in enumerate(groups):
            by_group[group].append(i)

        sizes = {g: len(indices) for g, indices in by_group.items()}

        # random ordering of idx in each group to simulate random sampling without replacement
        selection_orders = {
            g: [int(i) for i in self.rng.permutation(indices)] for g, indices in by_group.items()
        }

        if self.allocation == "balanced":
            slots = self._balanced_slots(sizes)
        else:
            slots = self._proportional_slots(sizes)

        return {
            "N": N,
            "remaining": set(range(problem.N)),
            "remaining_budget": problem.budget,
            "observed_losses": [],
            "selections": set(),
            "group_by_index": groups,
            "sizes": sizes,
            "selection_orders": selection_orders,
            "slots": slots,
            "current_pos": 0,
            "subgroup_counts": {g: 0 for g in sizes},
            "subgroup_loss_sums": {g: 0.0 for g in sizes},
            "loss_total": 0.0,
        }

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        """Selection of next sample indicies to obtain labels for"""
        selection_count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            state["N"] - state["current_pos"],
        )
        if selection_count < 1:
            raise ValueError("No remaining budget or unlabeled items")

        start_idx = state["current_pos"]
        batch_group_counts = defaultdict(int)
        selected_indices = []
        selected_groups = []

        for t in range(start_idx, start_idx + selection_count):
            group = state["slots"][t]
            group_sample_idx = state["subgroup_counts"][group] + batch_group_counts[group]

            selected_indices.append(state["selection_orders"][group][group_sample_idx])
            selected_groups.append(group)
            batch_group_counts[group] += 1

        n_after = start_idx + selection_count

        return SelectionPlan(
            indices=selected_indices,
            selection_probabilities=None,
            inclusion_probabilities=None,
            protected_attribute_group=selected_groups,
            metadata={
                "design": f"stratified_{self.allocation}_without_replacement",
                "inclusion_horizon": n_after,
            },
        )

    def observe(self, state: dict, selection: SelectionPlan, new_labels, new_losses) -> dict:
        if len(selection.indices) != len(new_losses):
            raise ValueError("One loss is required per selected index")

        # Update the loss totals with new observations
        for i, loss in zip(selection.indices, new_losses):
            group = state["group_by_index"][i]
            state["subgroup_counts"][group] += 1
            state["subgroup_loss_sums"][group] += float(loss)
            state["loss_total"] += float(loss)

        state["selections"].update(selection.indices)
        state["current_pos"] += len(selection.indices)
        state["remaining_budget"] -= len(selection.indices)
        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        n, N = state["current_pos"], state["N"]

        # Estimate mean loss within each protected group.
        subgroup_risks = {
            g: (
                state["subgroup_loss_sums"][g] / state["subgroup_counts"][g]
                if state["subgroup_counts"][g] > 0
                else None
            )
            for g in state["sizes"]
        }

        subgroup_n_labels = dict(state["subgroup_counts"])

        # Existing overall risk estimator.
        if n == 0:
            risk = float("nan")

        elif self.allocation == "proportional":
            risk = state["loss_total"] / n

        # Horvitz-Thompson estimator when not all subgroups have been seen
        elif n < len(state["sizes"]):
            K = len(state["sizes"])

            risk = (
                K
                / (n * N)
                * sum(size * state["subgroup_loss_sums"][g] for g, size in state["sizes"].items())
            )

        else:
            risk = sum(
                (size / N) * (state["subgroup_loss_sums"][g] / state["subgroup_counts"][g])
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

    def initialize(self, problem: Problem) -> dict:
        groups = problem.protected_attribute

        if groups is None or len(groups) != problem.N:
            raise ValueError("Protected attributes must match X_pool order and length")

        state = {
            "N": int(problem.N),
            "remaining": set(range(problem.N)),
            "remaining_budget": int(problem.budget),
            "observed_losses": [],
            "selections": set(),
            "group_by_index": groups,
            "subgroup_counts": {g: 0 for g in groups},
            "subgroup_loss_sums": {g: 0.0 for g in groups},
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
            selection_probabilities=None,
            inclusion_probabilities=None,
            protected_attribute_group=[state["group_by_index"][i] for i in indices],
            metadata={"design": "uniform_without_replacement"},
        )

    def observe(
        self, state: dict, selection: SelectionPlan, new_labels: list, new_losses: list
    ) -> dict:
        # Update remaining to exclude the newly selected indices
        state["remaining"].difference_update(selection.indices)
        state["remaining_budget"] -= len(selection.indices)
        state["observed_losses"].extend(new_losses)
        state["selections"].update(selection.indices)

        for index, loss in zip(selection.indices, new_losses):
            group = state["group_by_index"][index]
            state["subgroup_counts"][group] += 1
            state["subgroup_loss_sums"][group] += float(loss)

        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        losses = state["observed_losses"]
        counts = state["subgroup_counts"]

        overall_risk = float(np.mean(losses)) if losses else float("nan")

        subgroup_risks = {
            g: (state["subgroup_loss_sums"][g] / count if count > 0 else None)
            for g, count in counts.items()
        }

        return Estimate(
            risk=overall_risk,
            n_labels=len(losses),
            subgroup_risks=subgroup_risks,
            subgroup_n_labels=dict(counts),
        )
