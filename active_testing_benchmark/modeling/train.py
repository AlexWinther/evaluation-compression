"""Training routines for the image classifier and auxiliary models used for AT."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from itertools import islice
import math
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.surrogate import RandomForestModule
from active_testing_benchmark.modeling.data import (
    FairVisionDataset,
    get_class_counts,
    get_dataset_targets,
    inverse_frequency_weights,
)
from active_testing_benchmark.modeling.evaluation import evaluate
from active_testing_benchmark.modeling.models import create_model
from active_testing_benchmark.modeling.preparation import PreparedData


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    image_size: int = 224
    weighted_sampling: bool = False
    weighted_loss: bool = True
    sampling_power: float = 1.0
    loss_power: float = 1.0
    selection_objective: str = "class_balanced_cross_entropy"
    pretrained: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.num_workers < 0 or self.epochs < 1:
            raise ValueError(
                "batch_size and epochs must be positive; num_workers cannot be negative"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
        if not math.isfinite(self.learning_rate) or not math.isfinite(self.weight_decay):
            raise ValueError("learning_rate and weight_decay must be finite")
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if not math.isfinite(self.sampling_power) or not 0.0 <= self.sampling_power <= 1.0:
            raise ValueError("sampling_power must be finite and in [0, 1]")
        if not math.isfinite(self.loss_power) or not 0.0 <= self.loss_power <= 1.0:
            raise ValueError("loss_power must be finite and in [0, 1]")
        if self.selection_objective not in {
            "class_balanced_cross_entropy",
            "unweighted_cross_entropy",
        }:
            raise ValueError(
                "selection_objective must be class_balanced_cross_entropy or unweighted_cross_entropy"
            )
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 4294967295")


@dataclass
class TrainingResult:
    model: nn.Module
    class_counts: torch.Tensor
    risk_weights: torch.Tensor
    selected_epochs: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AuxiliaryModels:
    surrogates: tuple[RandomForestModule, RandomForestModule]
    proxy: RandomForestModule
    seeds: tuple[int, int, int]


def set_seed(seed: int, device: torch.device) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiment starts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit device or automatically select the best available one."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_train_loader_and_loss(
    dataset: FairVisionDataset | Subset,
    config: TrainingConfig,
    device: torch.device,
):
    counts = get_class_counts(dataset)
    sampler = None
    if config.weighted_sampling:
        per_class_weights = inverse_frequency_weights(counts, config.sampling_power)
        targets = torch.as_tensor(get_dataset_targets(dataset), dtype=torch.long)
        sample_weights = per_class_weights[targets].double().tolist()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    weights = (
        inverse_frequency_weights(counts, config.loss_power).to(device)
        if config.weighted_loss
        else None
    )
    return loader, nn.CrossEntropyLoss(weight=weights), counts, weights


def make_class_balanced_weights(class_counts: torch.Tensor) -> torch.Tensor:
    """Return inverse-prevalence weights whose weighted risk is macro-averaged CE."""
    counts = class_counts.to(dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() < 2 or not torch.isfinite(counts).all():
        raise ValueError("Class counts must be a finite one-dimensional tensor for 2+ classes")
    if torch.any(counts <= 0):
        raise ValueError("Every class needs at least one training example")
    return inverse_frequency_weights(class_counts, power=1.0, normalization="sample_mean")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    progress_position: int = 0,
    progress_total: int | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    denominator_total = 0.0
    sample_count = 0

    for batch in tqdm(
        loader,
        total=progress_total,
        desc="Training batches",
        unit="batch",
        position=progress_position,
        leave=False,
    ):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = len(targets)
        sample_count += batch_size
        valid_targets = (
            targets[targets != criterion.ignore_index]
            if isinstance(criterion, nn.CrossEntropyLoss)
            else targets
        )
        denominator = (
            valid_targets.numel()
            if not isinstance(criterion, nn.CrossEntropyLoss) or criterion.weight is None
            else criterion.weight[valid_targets].sum().item()
        )
        batch_loss = loss.item()
        if getattr(criterion, "reduction", "mean") != "sum":
            batch_loss *= denominator
        running_loss += batch_loss
        denominator_total += denominator

    if sample_count == 0:
        raise ValueError("Cannot train on an empty data loader")
    if denominator_total == 0:
        raise ValueError("Cannot aggregate training loss with a zero criterion denominator")
    return running_loss / denominator_total


def _train_classifier(
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
    *,
    fast: bool,
    model_factory: Callable | None = None,
) -> TrainingResult:
    model_factory = model_factory or create_model
    if len(data.pilot_fit) == 0 or len(data.pilot_validation) == 0 or len(data.development) == 0:
        raise ValueError("Pilot fit, pilot validation, and development datasets must be non-empty")
    if len(data.classes) < 2:
        raise ValueError("Training requires at least two classes")

    set_seed(config.seed, device)
    pilot_model = model_factory(out_dim=len(data.classes), use_pretrained=config.pretrained)
    pilot_model = pilot_model.to(device)

    pilot_loader, pilot_criterion, pilot_counts, _ = make_train_loader_and_loss(
        data.pilot_fit, config, device
    )
    pilot_weights = make_class_balanced_weights(pilot_counts).to(device)
    validation_loader = DataLoader(
        data.pilot_validation,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    pilot_epochs = 1 if fast else config.epochs
    best_score = float("inf")
    best_epoch = 1
    history = []
    pilot_batches = []
    optimizer = AdamW(
        pilot_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pilot_epochs, eta_min=1e-6
    )

    epoch_progress = tqdm(range(pilot_epochs), desc="Pilot epochs", unit="epoch", position=0)
    for epoch in epoch_progress:
        epoch_loader = islice(pilot_loader, 10) if fast else pilot_loader
        training_loss = train_one_epoch(
            pilot_model,
            epoch_loader,
            pilot_criterion,
            optimizer,
            device,
            progress_position=1,
            progress_total=min(len(pilot_loader), 10) if fast else len(pilot_loader),
        )
        # The limit is known without consuming the loader a second time.
        pilot_batches.append(min(len(pilot_loader), 10) if fast else len(pilot_loader))
        metrics = evaluate(
            pilot_model,
            validation_loader,
            device,
            protected_column="gender",
            class_weights=pilot_weights,
        )
        row = {
            "epoch": epoch + 1,
            "training_loss": float(training_loss),
            "unweighted_cross_entropy": float(metrics["loss"]),
            "class_balanced_cross_entropy": float(metrics["weighted_loss"]),
        }
        history.append(row)
        score = row[config.selection_objective]
        if not math.isfinite(score):
            raise ValueError(
                f"Pilot selection objective {config.selection_objective!r} was non-finite "
                f"at epoch {epoch + 1}: {score}"
            )
        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
        epoch_progress.set_postfix(loss=training_loss, validation=score)
        scheduler.step()
    epoch_progress.close()

    # Release the pilot and its optimizer state before allocating the refit model.
    del pilot_model, optimizer, scheduler
    # Use a distinct random stream so the refit starts from fresh parameters.
    refit_seed = (config.seed + 1) % 2**32
    set_seed(refit_seed, device)
    model = model_factory(out_dim=len(data.classes), use_pretrained=config.pretrained)
    model.to(device)
    loader, criterion, counts, _ = make_train_loader_and_loss(data.development, config, device)
    risk_weights = make_class_balanced_weights(counts)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1 if fast else best_epoch, eta_min=1e-6
    )
    refit_epochs = 1 if fast else best_epoch
    refit_batches = []
    refit_history = []
    epoch_progress = tqdm(range(refit_epochs), desc="Refit epochs", unit="epoch", position=0)
    for _epoch in epoch_progress:
        epoch_loader = islice(loader, 10) if fast else loader
        training_loss = train_one_epoch(
            model,
            epoch_loader,
            criterion,
            optimizer,
            device,
            progress_position=1,
            progress_total=min(len(loader), 10) if fast else len(loader),
        )
        refit_history.append(float(training_loss))
        refit_batches.append(min(len(loader), 10) if fast else len(loader))
        epoch_progress.set_postfix(loss=training_loss)
        scheduler.step()
    epoch_progress.close()

    metadata = {
        "mode": "fast" if fast else "full",
        "config": asdict(config),
        "pilot": {
            "fit_size": len(data.pilot_fit),
            "validation_size": len(data.pilot_validation),
            "epochs": pilot_epochs,
            "batches_per_epoch": pilot_batches,
            "history": history,
            "selected_epoch": best_epoch,
            "selection_objective": config.selection_objective,
            "selection_score": float(best_score),
            "class_counts": pilot_counts.tolist(),
            "risk_weights": pilot_weights.cpu().tolist(),
        },
        "refit": {
            "size": len(data.development),
            "epochs": refit_epochs,
            "batches_per_epoch": refit_batches,
            "training_loss": refit_history,
            "class_counts": counts.tolist(),
            "risk_weights": risk_weights.tolist(),
            "seed": refit_seed,
        },
    }
    return TrainingResult(model, counts, risk_weights, best_epoch, metadata)


def train_classifier(
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
    *,
    model_factory: Callable | None = None,
) -> TrainingResult:
    """Select a duration on the pilot partition, then refit on all development data."""
    return _train_classifier(data, config, device, fast=False, model_factory=model_factory)


def train_classifier_fast(
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
    *,
    model_factory: Callable | None = None,
) -> TrainingResult:
    """Run a one-epoch pilot/refit loop capped at ten batches per training phase."""
    return _train_classifier(data, config, device, fast=True, model_factory=model_factory)


def fit_auxiliary_models(
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
) -> AuxiliaryModels:
    """Fit the two active-testing surrogates and proxy on all development images."""
    children = np.random.SeedSequence(config.seed).spawn(3)
    seeds = tuple(int(child.generate_state(1)[0]) for child in children)
    loader = DataLoader(
        data.development_deterministic,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    forests = tuple(
        RandomForestModule(num_classes=len(data.classes), random_state=seed).fit(
            loader, device=device
        )
        for seed in seeds
    )
    return AuxiliaryModels((forests[0], forests[1]), forests[2], seeds)
