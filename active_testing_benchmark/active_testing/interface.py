from collections.abc import Hashable
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class SelectionPlan:
    """Items requested and probabilities used by the methods' sampling design.

    Selection probabilities describe ordered conditional draws. Inclusion
    probabilities describe the marginal chance that each selected item is included.
    """

    indices: list[int]
    selection_probabilities: list[float] | None = None
    inclusion_probabilities: list[float] | None = None
    protected_attribute_group: list[Hashable] | None = None
    metadata: dict | None = None


@dataclass
class Estimate:
    """Current finite-pool risk and subgroup risk estimates."""

    # The estimated mean loss over the entire finite evaluation pool
    risk: float
    # Label count that have been acquired before producing the estimate
    n_labels: int
    # Estimated population mean loss for every known protected group
    subgroup_risks: dict[Hashable, float | None] = field(default_factory=dict)
    # Observed label counts by group
    subgroup_n_labels: dict[Hashable, int] = field(default_factory=dict)
    # Optionally attach an uncertainty estimate; most methods here leave this unset.
    std_error: float | None = None
    # Optionally attach the current risk CI, produced by PPAT
    confidence_interval: tuple[float, float] | None = None


@dataclass
class Problem:
    """The fixed problem definition for a single active evaluation run.

    This is the information that methods need to initialize their state.
    """

    N: int
    budget: int
    X_pool: torch.Tensor
    protected_attribute: list[Hashable]
    model: nn.Module
    loss_fn: nn.Module


class ActiveEvaluationMethod:
    """Interface implemented by methods that choose labels and estimate risk.

    All methods implement the same four-stage loop: initialize once then repeatedly
    acquire -> observe -> estimate.
    """

    # Prepare method-specific state from the unlabeled inputs, fixed target model, loss and budget
    def initialize(self, problem: Problem) -> dict:
        raise NotImplementedError

    # Selection of currently unlabeled pool items to acquire, based on the current state and previous history
    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        raise NotImplementedError

    # Incorporate ONLY the new labels/losses for the chosen indices
    # Here sampling state or learned surrogate can be updated
    def observe(
        self, state: dict, selection: SelectionPlan, new_labels: list, new_losses: list
    ) -> dict:
        raise NotImplementedError

    # Produce a risk estimate from the information acquired so far.
    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        raise NotImplementedError
