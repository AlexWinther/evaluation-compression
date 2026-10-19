from loguru import logger
import torch
from torch import nn
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.interface import (
    ActiveEvaluationMethod,
    Problem,
)
from active_testing_benchmark.modeling.data import FairVisionDataset


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    dataset: FairVisionDataset,
    model: nn.Module,
    device: torch.device,
    loss_fn: nn.Module,
    protected_column: list[str] | str,
    budget: int | None = None,
) -> list[dict]:
    """Run finite-pool active evaluation, returning risk estimates after each batch."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    logger.info("Model set to eval mode for active evaluation")

    logger.debug(f"Loading {len(dataset):,} samples from the dataset for active evaluation")

    X_pool = []
    y_pool = []
    protected_attributes = []

    logger.debug(f"Using protected column(s): {protected_column}")

    if isinstance(protected_column, str):
        protected_column = [protected_column]

    for index in range(len(dataset)):
        sample = dataset[index]

        X_pool.append(sample["image"])
        y_pool.append(sample["target"])

        protected_attribute = ", ".join(
            str(sample["protected"].get(col)) for col in protected_column or []
        )

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

    pbar = tqdm(total=problem.budget, desc="Active evaluation progress", unit="label")

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
                "estimate": method.estimate(state, history),
            }
        )
        pbar.update(len(indices))

    # Return the full history to enable post-hoc analysis of the risk estimates
    return history
