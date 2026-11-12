from dataclasses import dataclass, field, replace

from loguru import logger
import torch
from torch import nn
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.baseline import StratifiedSampleMean
from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Problem,
)
from active_testing_benchmark.modeling.data import FairVisionData


@dataclass(frozen=True)
class EvaluationPool:
    """Materialized inputs and runner-only labels shared across evaluation runs.

    Methods must treat the shared input tensors as read-only. Labels are never
    included in the Problem passed to methods.
    Attribute views share tensors. Reporting memberships may span multiple partitions.
    """

    X_pool: torch.Tensor
    y_pool: torch.Tensor
    stratification_groups: list[str] | None
    protected_columns: tuple[str, ...]
    attribute_values: dict[str, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X_pool)


def prepare_evaluation_pool(
    dataset: FairVisionData | EvaluationPool,
    device: torch.device,
    protected_column: list[str] | str,
) -> EvaluationPool:
    """Load and transfer a fixed evaluation pool once, or reuse a prepared pool."""
    protected_column = (
        [protected_column] if isinstance(protected_column, str) else list(protected_column)
    )
    if not protected_column or len(set(protected_column)) != len(protected_column):
        raise ValueError("Provide nonempty, unique protected columns")
    if "age" in protected_column:
        raise ValueError("Use age_group instead of raw age")
    if isinstance(dataset, EvaluationPool):
        if tuple(protected_column) == dataset.protected_columns:
            return dataset
        if any(column not in dataset.attribute_values for column in protected_column):
            raise ValueError("Prepared pool uses different protected columns")
        return replace(
            dataset,
            protected_columns=tuple(protected_column),
            stratification_groups=None,
        )
    if len(dataset) == 0:
        raise ValueError("Evaluation pool must be nonempty")

    logger.debug(f"Loading {len(dataset):,} samples from the dataset for active evaluation")

    X_pool = []
    y_pool = []
    attribute_values = {column: [] for column in protected_column}

    logger.debug(f"Reporting attributes: {protected_column}")

    for index in range(len(dataset)):
        sample = dataset[index]

        X_pool.append(sample["image"])
        y_pool.append(sample["label"] if "label" in sample else sample["target"])

        subgroups = sample.get("subgroups", sample.get("protected", {}))
        for column in protected_column:
            attribute_values[column].append(f"{column}={subgroups[column]}")

    stratification_groups = None
    for column, values in attribute_values.items():
        logger.debug(f"Reporting groups for {column}: {sorted(set(values))}")

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
        X_pool, y_pool, stratification_groups, tuple(protected_column), attribute_values
    )


def _reporting_memberships(pool: EvaluationPool) -> dict[str, tuple[int, ...]]:
    """Build marginal reporting groups for the requested attributes only."""
    groups: dict[str, list[int]] = {}
    for column in pool.protected_columns:
        values = pool.attribute_values.get(column, pool.stratification_groups)
        if values is None:
            raise ValueError(f"Missing reporting memberships for {column}")
        for index, group in enumerate(values):
            groups.setdefault(group, []).append(index)
    return {group: tuple(indices) for group, indices in groups.items()}


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    dataset: FairVisionData | EvaluationPool,
    model: nn.Module,
    device: torch.device,
    loss_fn: nn.Module,
    protected_column: list[str] | str,
    budget: int | None = None,
    progress_position: int = 0,
    stratification_attribute: str | None = None,
) -> list[dict]:
    """Run finite-pool active evaluation, returning risk estimates after each batch."""
    columns = [protected_column] if isinstance(protected_column, str) else protected_column
    if isinstance(method, StratifiedSampleMean) and stratification_attribute is None:
        if len(columns) != 1:
            raise ValueError("Select one protected attribute for stratification")
        stratification_attribute = columns[0]
    if not isinstance(method, StratifiedSampleMean) and stratification_attribute is not None:
        raise ValueError("Only stratified methods use a stratification attribute")
    if stratification_attribute is not None and stratification_attribute not in columns:
        raise ValueError("The stratification attribute must be included in reporting columns")
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    logger.info("Model set to eval mode for active evaluation")

    pool = prepare_evaluation_pool(dataset, device, protected_column)
    X_pool = pool.X_pool
    y_pool = pool.y_pool
    # Reporting metadata covers all selected attributes. Only stratified methods
    # receive a sampling partition
    stratification_groups = None
    if stratification_attribute is not None:
        values = pool.attribute_values.get(stratification_attribute, pool.stratification_groups)
        if values is None:
            raise ValueError("Missing stratification memberships")
        stratification_groups = list(values)
    logger.debug(f"Stratification attribute: {stratification_attribute}")

    label_budget = len(X_pool) if budget is None else int(budget)
    if not 0 <= label_budget <= len(X_pool):
        raise ValueError("budget must be between 0 and the pool size")

    # Note: no labels are given to the method at initialization
    problem = Problem(
        N=len(X_pool),
        budget=label_budget,
        X_pool=X_pool,
        stratification_groups=stratification_groups,
        model=model,
        loss_fn=loss_fn,
        subgroup_indices=_reporting_memberships(pool),
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
        # Add the current information about the current selection batch
        # including the current risk estimates to history
        history.append(
            {
                "round": len(history) + 1,
                "total selections": len(state["selections"]),
                "selection": selection,
                "labels": labels.to("cpu").tolist(),
                "stratification_groups": selection.stratification_groups,
                "losses": losses,
                "estimate": estimate,
            }
        )
        pbar.update(len(indices))

    pbar.close()

    # Return the full history to enable post-hoc analysis of the risk estimates
    return history
