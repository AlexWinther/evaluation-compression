import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.interface import ActiveEvaluationMethod
from active_testing_benchmark.modeling.data import FairVisionDataset


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    dataset: FairVisionDataset,
    model: nn.Module,
    device: torch.device,
    loss_fn: nn.Module = nn.CrossEntropyLoss(reduction="sum"),
    protected_column: list[str] | str | None = None,
    budget: int | None = None,
) -> list[dict]:
    """Run finite-pool evaluation, returning prefix estimates after each batch."""
    model.eval()
    logger.info("Model set to eval mode for active evaluation")

    logger.debug(
        f"Loading {len(dataset):,} samples from the dataset for active evaluation"
    )
    X_pool = []
    y_pool = []

    if protected_column is not None:
        if isinstance(protected_column, str):
            protected_column = [protected_column]

    logger.debug(f"Using protected column(s): {protected_column}")

    protected_attributes = []

    for index in range(len(dataset)):
        sample = dataset[index]

        X_pool.append(sample["image"])
        y_pool.append(sample["target"])

        protected_attribute = ", ".join(
            str(sample["protected"].get(col)) for col in protected_column or []
        ) if protected_column else None

        if protected_attribute is not None:
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

    problem = {
        "N": len(X_pool),
        "X_pool": X_pool,
        "protected_attribute": protected_attributes,
        "model": model,
        "loss_fn": loss_fn,
        "budget": label_budget,
    }

    state = method.initialize(problem)
    history = []

    pbar = tqdm(
        total=problem["budget"], desc="Active evaluation progress", unit="label"
    )

    while len(state["selections"]) < problem["budget"]:
        selection = method.acquire(state, history)
        indices = selection.indices
        # logger.debug(f"Acquired selection: {indices}")

        if not indices or len(indices) != len(set(indices)):
            raise ValueError("A selection must contain unique indices")
        if any(not isinstance(index, (int, np.integer)) for index in indices):
            raise ValueError("Selection indices must be integers")
        if len(state["selections"]) + len(indices) > problem["budget"]:
            raise ValueError("Selection exceeds the remaining label budget")
        if any(
            index < 0 or index >= len(X_pool) or index in state["selections"]
            for index in indices
        ):
            raise ValueError("Selection indices must be remaining pool indices")

        labels = y_pool[indices]
        with torch.inference_mode():
            y_pred = model(X_pool[indices])

            losses = [
                problem["loss_fn"](y_pred[i : i + 1], labels[i : i + 1]).item()
                for i in range(len(indices))
            ]

        state = method.observe(state, selection, labels, losses)
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

    return history