from dataclasses import dataclass, field
from typing import Hashable


@dataclass
class SelectionPlan:
    """Items requested and probabilities under the method's sampling design.

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

    risk: float
    n_labels: int
    subgroup_risks: dict[Hashable, float | None] = field(
        default_factory=dict
    )
    subgroup_n_labels: dict[Hashable, int] = field(
        default_factory=dict
    )
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None


class ActiveEvaluationMethod:
    """Interface implemented by methods that choose labels and estimate risk."""

    def initialize(self, problem):
        raise NotImplementedError

    def acquire(self, state, history) -> SelectionPlan:
        raise NotImplementedError

    def observe(self, state, selection, labels, losses):
        raise NotImplementedError

    def estimate(self, state, history) -> Estimate:
        raise NotImplementedError