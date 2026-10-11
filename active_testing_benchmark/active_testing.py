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
            inclusion_probabilities=[count / len(remaining)] * count,
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


def resolve_budget(value: str, population: int) -> int:
    """Integer syntax means count; decimal syntax means fraction, rounded up."""
    value = value.strip()
    if population < 1:
        raise ValueError("Test population must be positive")
    if re.fullmatch(r"[0-9]+", value):
        budget = int(value)
    else:
        fraction = float(value)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Fractional test sizes must be in (0, 1]")
        budget = math.ceil(fraction * population)

    if not 1 <= budget <= population:
        raise ValueError(f"Test size must resolve to 1..{population} rows")
    return budget


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    dataset: FairVisionDataset,
    model: nn.Module,
    device: torch.device,
) -> Estimate:
    """Run the label-acquisition loop against a finite-pool label oracle."""
    model.eval()

    X_pool = np.array([])
    y_pool = np.array([])
    for index in range(len(dataset)):
        sample = dataset[index]
        X_pool = np.append(X_pool, sample["image"])
        y_pool = np.append(y_pool, sample["target"])

    problem = {
        "N": len(X_pool),
        "X_pool": X_pool,
        "model": model,
        "loss_fn": nn.CrossEntropyLoss(),
        "budget": len(X_pool),
    }

    state = method.initialize(problem)
    history = []

    while len(state["selections"]) < problem["budget"]:
        selection = method.acquire(state, history)
        indices = selection.indices
        logger.info(f"Acquired selection: {indices}")

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

        losses = [nn.CrossEntropyLoss(y_pred[i], labels[i]) for i in range(len(indices))]

        history.append(
            {"round": len(history), "selection": selection, "labels": labels, "losses": losses, "estimate": method.estimate(state, history)}
        )
        state = method.observe(state, selection, labels, losses)

    return history[-1]["estimate"] if history else method.estimate(state, history)
