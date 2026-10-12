"""Active finite-pool risk estimation and its MLflow-backed image-classification runner."""

# ruff: noqa: B008 -- Typer declares CLI options as argument defaults.

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from active_testing_benchmark.modeling.data import (
    FairVisionDataset,
)


@dataclass
class SelectionPlan:
    """Items requested and probabilities under the method's sampling design.

    Selection probabilities describe ordered conditional draws. Inclusion
    probabilities describe the marginal chance that each selected item is included.
    """

    indices: list[int]
    selection_probabilities: list[float] | None = None
    inclusion_probabilities: list[float] | None = None
    metadata: dict | None = None


@dataclass
class Estimate:
    """A method's current estimate of finite-pool risk."""

    risk: float
    n_labels: int


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


class UniformSampleMean(ActiveEvaluationMethod):
    """Uniform sampling without replacement and the mean of observed losses."""

    def __init__(self, seed: int = 0, batch_size: int | None = 1) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.batch_size = batch_size

    def initialize(self, problem: dict) -> dict:
        return {
            "N": problem["N"],
            "remaining": set(range(problem["N"])),
            "remaining_budget": problem["budget"],
            "observed_losses": [],
            "selections": set(),
        }

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
            selection_probabilities=[1.0 / (len(remaining) - offset) for offset in range(count)],
            inclusion_probabilities=[(len(state['selections']) + offset) / state['N'] for offset in range(1, count + 1)],
            metadata={"design": "uniform_without_replacement"},
        )

    def observe(self, state: dict, selection: SelectionPlan, labels: list, losses: list) -> dict:
        state["remaining"].difference_update(selection.indices)
        state["remaining_budget"] -= len(selection.indices)
        state["observed_losses"].extend(losses)
        state["selections"].update(selection.indices)

        return state

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        losses = state["observed_losses"]
        return Estimate(risk=float(np.mean(losses)), n_labels=len(losses))


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    dataset: FairVisionDataset,
    model: nn.Module,
    device: torch.device,
    loss_fn: nn.Module = nn.CrossEntropyLoss(reduction="sum"),
) -> Estimate:
    """Run the label-acquisition loop against a finite-pool label oracle."""
    model.eval()

    X_pool = []
    y_pool = []
    logger.info(f"Loading {len(dataset):,} samples from the dataset for active evaluation")
    for index in range(len(dataset)):
        sample = dataset[index]

        X_pool.append(sample["image"])
        y_pool.append(sample["target"])

    logger.debug(f"image shapes: {X_pool[0].shape}, dtype: {X_pool[0].dtype}")
    logger.debug(f"y_pool type: {type(y_pool[0])}")

    X_pool = torch.stack(X_pool).to(device, non_blocking=True)
    logger.debug(f"X_pool shape: {X_pool.shape}, dtype: {X_pool.dtype}")

    y_pool = torch.tensor(y_pool).to(device, non_blocking=True).long()
    logger.debug(f"y_pool shape: {y_pool.shape}, dtype: {y_pool.dtype}")

    logger.info(f"Loaded {len(X_pool):,} samples for active evaluation with {len(y_pool.unique()):,} unique classes")

    problem = {
        "N": len(X_pool),
        "X_pool": X_pool,
        "model": model,
        "loss_fn": loss_fn,
        "budget": len(X_pool),
    }

    state = method.initialize(problem)
    history = []

    pbar = tqdm(total=problem["budget"], desc="Active evaluation progress", unit="label")

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
        if any(index < 0 or index >= len(X_pool) or index in state["selections"] for index in indices):
            raise ValueError("Selection indices must be remaining pool indices")

        labels = y_pool[indices]
        y_pred = model(X_pool[indices])

        losses = [problem["loss_fn"](y_pred[i], labels[i]).item() for i in range(len(indices))]

        state = method.observe(state, selection, labels, losses)
        history.append(
            {"round": len(history) + 1, "total selections": len(state["selections"]), "selection": selection, "labels": labels, "losses": losses, "estimate": method.estimate(state, history)}
        )
        pbar.update(len(indices))

    return history
