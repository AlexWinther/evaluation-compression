from dataclasses import dataclass, field

from loguru import logger
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Problem,
)


@dataclass(frozen=True)
class EvaluationPool:
    """Materialized inputs and runner-only labels shared across evaluation runs.

    Methods must treat the shared input tensors as read-only. Labels are never
    included in the Problem passed to methods.
    """

    X_pool: torch.Tensor
    y_pool: torch.Tensor
    protected_attributes: list[str]
    protected_columns: tuple[str, ...]
    subgroup_memberships: dict[str, dict[str, int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X_pool)


def prepare_evaluation_pool(
    dataset: Dataset | EvaluationPool,
    device: torch.device,
    protected_column: list[str] | str,
) -> EvaluationPool:
    """Load and transfer a fixed evaluation pool once, or reuse a prepared pool."""
    protected_column = (
        [protected_column] if isinstance(protected_column, str) else list(protected_column)
    )
    if isinstance(dataset, EvaluationPool):
        if tuple(protected_column) != dataset.protected_columns:
            raise ValueError("Prepared pool uses different protected columns")
        return dataset
    if len(dataset) == 0:
        raise ValueError("Evaluation pool must be nonempty")

    logger.debug(f"Loading {len(dataset):,} samples from the dataset for active evaluation")

    X_pool = []
    y_pool = []
    protected_attributes = []
    subgroup_memberships = {}

    logger.debug(f"Using protected column(s): {protected_column}")

    for index in range(len(dataset)):
        sample = dataset[index]

        X_pool.append(sample["image"])
        y_pool.append(sample["label"] if "label" in sample else sample["target"])

        subgroups = sample.get("subgroups", sample.get("protected", {}))
        protected_attribute = ", ".join(str(subgroups[col]) for col in protected_column)
        if len(protected_column) > 1:
            for col in protected_column:
                members = subgroup_memberships.setdefault(f"{col}={subgroups[col]}", {})
                members[protected_attribute] = members.get(protected_attribute, 0) + 1

        protected_attributes.append(protected_attribute)

    logger.debug(f"protected_attributes: {protected_attributes[:10]} (showing first 10)")

    logger.debug(f"image shapes: {X_pool[0].shape}, dtype: {X_pool[0].dtype}")
    logger.debug(f"y_pool type: {type(y_pool[0])}")

    X_pool = torch.stack(X_pool).to(device, non_blocking=True)
    logger.debug(f"X_pool shape: {X_pool.shape}, dtype: {X_pool.dtype}")

    y_pool = torch.tensor(y_pool).to(device, non_blocking=True).long()
    logger.debug(f"y_pool shape: {y_pool.shape}, dtype: {y_pool.dtype}")

    logger.debug(
        f"Loaded {len(X_pool):,} samples for active evaluation with {len(y_pool.unique()):,} unique classes"
    )

    return EvaluationPool(
        X_pool, y_pool, protected_attributes, tuple(protected_column), subgroup_memberships
    )


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    dataset: Dataset | EvaluationPool,
    model: nn.Module,
    device: torch.device,
    loss_fn: nn.Module,
    protected_column: list[str] | str,
    budget: int | None = None,
    progress_position: int = 0,
) -> list[dict]:
    """Run finite-pool active evaluation, returning risk estimates after each batch."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    logger.info("Model set to eval mode for active evaluation")

    pool = prepare_evaluation_pool(dataset, device, protected_column)
    X_pool = pool.X_pool
    y_pool = pool.y_pool
    # Each run gets its own group list; image tensors are shared read-only.
    protected_attributes = list(pool.protected_attributes)

    label_budget = len(X_pool) if budget is None else int(budget)
    if not 0 <= label_budget <= len(X_pool):
        raise ValueError("budget must be between 0 and the pool size")

    # Note: no labels are given to the method at initialization
    problem = Problem(
        N=len(X_pool),
        budget=label_budget,
        X_pool=X_pool,
        protected_attribute=protected_attributes,
        model=model,
        loss_fn=loss_fn,
    )

    # Create the inital state
    state = method.initialize(problem)

    history = []

    pbar = tqdm(
        total=problem.budget,
        desc="Active evaluation budget",
        unit="label",
        position=progress_position,
        leave=False,
    )

    # Loop until the acquired labels reach the budget
    while len(state["selections"]) < problem.budget:
        # Ask the method to pick its next samples to reveal using its current state and the known history
        selection = method.acquire(state, history)
        indices = selection.indices
        logger.trace(f"Acquired selection: {indices}")

        if (
            not indices
            or len(indices) != len(set(indices))
            or any(
                index < 0 or index >= len(X_pool) or index in state["selections"]
                for index in indices
            )
        ):
            raise ValueError(
                "A selection must contain unique and not previously acquired"
                "indices from the remaining pool"
            )

        # Reveal ground-truth labels for the selected indices.
        labels = y_pool[indices]

        with torch.inference_mode():
            y_pred = model(X_pool[indices])

            # Compute one loss per selected item
            losses = [
                problem.loss_fn(y_pred[i : i + 1], labels[i : i + 1]).item()
                for i in range(len(indices))
            ]

        # The selected samples are observed by the model, including the per sample losses and labels
        # The state is updated, optionally including a learned surrogate model
        state = method.observe(state, selection, labels, losses)

        estimate = method.estimate(state, history)
        # Marginal subgroup risks are population-weighted means of the joint group estimates.
        # Preserve missing estimates rather than treating unobserved groups as zero risk.
        for group, members in pool.subgroup_memberships.items():
            risks = [estimate.subgroup_risks[joint] for joint in members]
            estimate.subgroup_risks[group] = (
                sum(size * risk for size, risk in zip(members.values(), risks))
                / sum(members.values())
                if all(risk is not None for risk in risks)
                else None
            )
            estimate.subgroup_n_labels[group] = sum(
                estimate.subgroup_n_labels[joint] for joint in members
            )

        # Add the current information about the current selection batch
        # including the current risk estimates to history
        history.append(
            {
                "round": len(history) + 1,
                "total selections": len(state["selections"]),
                "selection": selection,
                "labels": labels.to("cpu").tolist(),
                "protected_attribute_group": [state["group_by_index"][i] for i in indices],
                "losses": losses,
                "estimate": estimate,
            }
        )
        pbar.update(len(indices))

    pbar.close()

    # Return the full history to enable post-hoc analysis of the risk estimates
    return history
