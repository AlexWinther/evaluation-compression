"""Train one image classifier and log the run to MLflow."""

from __future__ import annotations

import random
from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import typer
from loguru import logger
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

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
from active_testing_benchmark.modeling.registry import (
    configure_mlflow,
    fairvision_model_name,
    register_trained_model,
)

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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logger.info("Model set to eval mode")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    n = 0
    predictions = []
    for batch in loader:
        with torch.inference_mode():
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True).long()
            logits = model(images)
            probabilities = logits.softmax(dim=1)
            predicted = logits.argmax(dim=1)

            total_loss += criterion(logits, targets).item()
        correct += (predicted == targets).sum().item()
        n += targets.numel()
        for i, image_id in enumerate(batch["id"]):
            predictions.append(
                {
                    "id": image_id,
                    "target": targets[i].item(),
                    "prediction": predicted[i].item(),
                    "probabilities": probabilities[i].cpu().tolist(),
                    "protected": {
                        name: values[i] for name, values in batch["protected"].items()
                    },
                }
            )

    # auc
    auc = roc_auc_score(
        [p["target"] for p in predictions],
        [p["probabilities"][1] for p in predictions],
    )
    
    return {"loss": total_loss / n, "accuracy": correct / n, "roc_auc": auc, "predictions": predictions}
