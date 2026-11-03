"""Train one image classifier and log the run to MLflow."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

from loguru import logger
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm
import typer

from active_testing_benchmark.modeling.data import (
    FairVisionDataset,
    get_class_counts,
    get_dataset_targets,
    inverse_frequency_weights,
)

# from active_testing_benchmark.modeling.provenance import (
#     FAIRVISION_SOURCE,
#     build_dataset_provenance,
#     git_provenance,
# )

app = typer.Typer(add_completion=False, help=__doc__)


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


@dataclass(frozen=True)
class TrainConfig:
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


def make_train_loader_and_loss(
    dataset: FairVisionDataset | Subset, config: TrainConfig, device: torch.device
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

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )

    loss_weights = (
        inverse_frequency_weights(counts, config.loss_power).to(device)
        if config.weighted_loss
        else None
    )
    train_criterion = nn.CrossEntropyLoss(weight=loss_weights)
    return train_loader, train_criterion, counts, loss_weights


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
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    progress_position: int = 0,
):
    model.train()
    logger.info("Model set to train mode")
    running_loss = 0.0
    loss_denominator_total = 0.0
    sample_count = 0

    for batch in tqdm(
        loader,
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
        loss_denominator = batch_size
        if isinstance(criterion, nn.CrossEntropyLoss):
            valid_targets = targets[targets != criterion.ignore_index]
            if criterion.weight is None:
                loss_denominator = valid_targets.numel()
            else:
                loss_denominator = criterion.weight[valid_targets].sum().item()

        batch_loss = loss.item()
        if getattr(criterion, "reduction", "mean") != "sum":
            batch_loss *= loss_denominator
        running_loss += batch_loss
        loss_denominator_total += loss_denominator

    if sample_count == 0:
        raise ValueError("Cannot train on an empty data loader")
    if loss_denominator_total == 0:
        raise ValueError("Cannot aggregate training loss with a zero criterion denominator")

    return running_loss / loss_denominator_total


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    protected_column: str | list[str] | None = None,
    class_weights: torch.Tensor | None = None,
):
    model.eval()
    logger.info("Model set to eval mode")
    if isinstance(protected_column, str):
        protected_column = [protected_column]

    criterion = nn.CrossEntropyLoss(reduction="none")
    weighted_criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="none")
        if class_weights is not None
        else None
    )

    predictions = []
    subgroups = defaultdict(lambda: defaultdict(list))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()

        logits = model(images)
        probabilities = logits.softmax(dim=1)
        predicted = logits.argmax(dim=1)
        losses = criterion(logits, targets)
        weighted_losses = (
            weighted_criterion(logits, targets) if weighted_criterion is not None else None
        )

        # Move batch outputs to CPU once
        targets = targets.cpu().tolist()
        predicted = predicted.cpu().tolist()
        probabilities = probabilities.cpu().tolist()
        losses = losses.cpu().tolist()
        if weighted_losses is not None:
            weighted_losses = weighted_losses.cpu().tolist()

        for i, image_id in enumerate(batch["id"]):
            protected = {
                name: (values[i].item() if isinstance(values[i], torch.Tensor) else values[i])
                for name, values in batch["protected"].items()
                if protected_column is None or name in protected_column
            }

            record = {
                "id": image_id,
                "target": targets[i],
                "prediction": predicted[i],
                "probabilities": probabilities[i],
                "loss": losses[i],
                "protected": protected,
            }
            if weighted_losses is not None:
                record["weighted_loss"] = weighted_losses[i]
            predictions.append(record)

            # Group each image by every protected attribute
            for attribute, value in protected.items():
                subgroups[attribute][str(value)].append(record)

    def compute_metrics(records) -> dict:
        y_true = np.asarray([r["target"] for r in records])
        probabilities = np.asarray([r["probabilities"] for r in records])
        n_classes = probabilities.shape[1]
        observed_classes = np.unique(y_true)

        if n_classes == 2 and len(observed_classes) == 2:
            auc = float(roc_auc_score(y_true, probabilities[:, 1]))
        elif np.array_equal(observed_classes, np.arange(n_classes)):
            auc = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
        else:
            auc = None

        metrics = {
            "n": len(records),
            "loss": sum(r["loss"] for r in records) / len(records),
            "accuracy": sum(r["target"] == r["prediction"] for r in records) / len(records),
            "roc_auc": auc,
        }
        metrics["weighted_loss"] = (
            sum(r["weighted_loss"] for r in records) / len(records)
            if weighted_criterion is not None
            else None
        )
        return metrics

    # Overall metrics
    overall = compute_metrics(predictions)

    # Metrics for each protected attribute and subgroup
    subgroup_metrics = {
        attribute: {group: compute_metrics(records) for group, records in groups.items()}
        for attribute, groups in subgroups.items()
    }

    # Ground-truth risks for comparison with active testing
    subgroup_risks = {
        attribute: {group: metrics["loss"] for group, metrics in groups.items()}
        for attribute, groups in subgroup_metrics.items()
    }
    subgroup_weighted_risks = {
        attribute: {group: metrics["weighted_loss"] for group, metrics in groups.items()}
        for attribute, groups in subgroup_metrics.items()
    }

    return {
        "loss": overall["loss"],
        "weighted_loss": overall["weighted_loss"],
        "accuracy": overall["accuracy"],
        "roc_auc": overall["roc_auc"],
        "subgroup_risks": subgroup_risks,
        "subgroup_weighted_risks": subgroup_weighted_risks,
        "subgroup_metrics": subgroup_metrics,
        "predictions": predictions,
    }
