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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import typer

from active_testing_benchmark.modeling.data import (
    FairVisionDataset,
    get_class_counts,
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
    epochs: int = 5
    learning_rate: float = 1e-4
    seed: int = 42
    image_size: int = 224
    weighted_sampling: bool = False
    weighted_loss: bool = True
    sampling_power: float = 1.0
    loss_power: float = 1.0


def make_train_loader_and_loss(
    dataset: FairVisionDataset, config: TrainConfig, device: torch.device
):
    counts = get_class_counts(dataset)
    sampler = None
    if config.weighted_sampling:
        per_class_weights = inverse_frequency_weights(counts, config.sampling_power)
        targets = torch.as_tensor(dataset.targets, dtype=torch.long)
        sample_weights = per_class_weights[targets].double()
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    debug: int | None = None,
):
    model.train()
    logger.info("Model set to train mode")
    running_loss = 0.0

    for i, batch in enumerate(tqdm(loader)):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if debug and i > debug:
            break

    return running_loss / len(loader)  # Mean minibatch loss


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    protected_column: str | list[str] | None = None,
):
    model.eval()
    logger.info("Model set to eval mode")
    if isinstance(protected_column, str):
        protected_column = [protected_column]

    criterion = nn.CrossEntropyLoss(reduction="none")

    predictions = []
    subgroups = defaultdict(lambda: defaultdict(list))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()

        logits = model(images)
        probabilities = logits.softmax(dim=1)
        predicted = logits.argmax(dim=1)
        losses = criterion(logits, targets)

        # Move batch outputs to CPU once
        targets = targets.cpu().tolist()
        predicted = predicted.cpu().tolist()
        probabilities = probabilities.cpu().tolist()
        losses = losses.cpu().tolist()

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
            predictions.append(record)

            # Group each image by every protected attribute
            for attribute, value in protected.items():
                subgroups[attribute][str(value)].append(record)

    def compute_metrics(records):
        if not records:
            return None

        y_true = [r["target"] for r in records]
        y_score = [r["probabilities"][1] for r in records]

        # Binary ROC-AUC is undefined if only one class is present
        auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None

        return {
            "n": len(records),
            "loss": sum(r["loss"] for r in records) / len(records),
            "accuracy": sum(r["target"] == r["prediction"] for r in records) / len(records),
            "roc_auc": auc,
        }

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

    return {
        "loss": overall["loss"],
        "accuracy": overall["accuracy"],
        "roc_auc": overall["roc_auc"],
        "subgroup_risks": subgroup_risks,
        "subgroup_metrics": subgroup_metrics,
        "predictions": predictions,
    }
