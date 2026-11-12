from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass

from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm.auto import tqdm

from active_testing_benchmark.active_testing.surrogate import RandomForestModule
from active_testing_benchmark.config import RAW_DATA_DIR
from active_testing_benchmark.modeling.data import FairVisionData, prepare_data
from active_testing_benchmark.modeling.models import create_model


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 20
    batch_size: int = 32
    model_type: str = "resnet"
    initial_lr: float = 1e-4
    weight_decay: float = 1e-3
    image_size: int = 224

    validation_fraction: float = 0.2
    random_state: int = 42
    seed: int = 42
    num_workers: int = 0


@dataclass(init=False)
class TrainingSetup:
    model: nn.Module
    loss_fn: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | None

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        )
        self.scheduler = scheduler


def split_dataset(
    dataset: FairVisionData,
    labels: np.ndarray,
    *,
    strategy: str = "holdout",
    validation_fraction: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[tuple[Subset[FairVisionData], Subset[FairVisionData]]]:
    if strategy == "holdout":
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
    elif strategy == "kfold":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    for train_idx, val_idx in splitter.split(np.arange(len(dataset)), labels):
        yield (
            Subset(dataset, train_idx.tolist()),
            Subset(dataset, val_idx.tolist()),
        )


def train_model(
    training_setup: TrainingSetup,
    training_config: TrainingConfig,
    training_dataset: FairVisionData | Subset[FairVisionData],
    val_dataset: FairVisionData | Subset[FairVisionData],
    device: torch.device | str = "mps",
):
    """Return the best model and history, based on best validation checkpoint."""
    model = training_setup.model.to(device)

    logger.info(
        f"Starting training of {model.__module__} with {training_config.max_epochs} max epochs"
    )

    train_loader = DataLoader(
        training_dataset, batch_size=training_config.batch_size, shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

    logger.info(
        f"Split into training batches ({len(train_loader)}) "
        f"and validation batches ({len(val_loader)})"
    )

    optimizer = training_setup.optimizer
    logger.info(optimizer)

    criterion = training_setup.loss_fn.to(device)
    scheduler = training_setup.scheduler
    history = []
    best_auc = -float("inf")
    best_state = None

    for epoch in tqdm(
        range(1, training_config.max_epochs + 1), desc="Epochs", leave=False, position=0
    ):
        model.train()
        epoch_loss = 0
        denominator_total = 0

        tl_pbar = tqdm(train_loader, desc="Batches", leave=False, position=1)
        for batch in tl_pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].long().to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            denominator = targets.numel()
            if getattr(criterion, "reduction", "mean") != "sum":
                batch_loss *= denominator
            epoch_loss += batch_loss
            denominator_total += denominator

            tl_pbar.set_postfix(Loss=epoch_loss / denominator_total)

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        labels, probabilities = [], []
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["label"].long().to(device, non_blocking=True)
                logits = model(images)
                batch_loss = criterion(logits, targets).item()
                if getattr(criterion, "reduction", "mean") != "sum":
                    batch_loss *= targets.numel()
                validation_loss += batch_loss
                validation_count += targets.numel()
                labels.append(targets.cpu().numpy())
                probabilities.append(logits.softmax(dim=1).cpu().numpy())

        y_true = np.concatenate(labels)
        y_score = np.concatenate(probabilities)
        y_pred = y_score.argmax(axis=1)
        num_classes = y_score.shape[1]
        auc = None
        if np.array_equal(np.unique(y_true), np.arange(num_classes)):
            auc = (
                roc_auc_score(y_true, y_score[:, 1])
                if num_classes == 2
                else roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            )
        validation_loss /= validation_count
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / denominator_total,
                "validation_loss": validation_loss,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    average="binary" if num_classes == 2 else "macro",
                    labels=np.arange(num_classes),
                    zero_division=0.0,
                ),
                "roc_auc": auc,
            }
        )
        logger.info(f"Epoch {epoch}: {history[-1]}")
        if scheduler is not None:
            scheduler.step()
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise ValueError("Training completed without a valid best checkpoint")
    model.load_state_dict(best_state)

    return model, history


@dataclass
class AuxiliaryModels:
    surrogates: tuple[RandomForestModule, RandomForestModule]
    proxy: RandomForestModule
    seeds: tuple[int, int, int]


def fit_auxiliary_models(
    dataset: FairVisionData | Subset[FairVisionData],
    classes: list[str],
    config: TrainingConfig,
    device: torch.device | str,
):
    """Fit the two active-testing surrogates and proxy on all development images."""
    children = np.random.SeedSequence(config.random_state).spawn(3)
    seeds = tuple(int(child.generate_state(1)[0]) for child in children)
    if len(seeds) != 3:
        raise ValueError("Expected only 3 generated seeds")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda" if isinstance(device, torch.device) else device == "cuda",
    )
    forests = tuple(
        RandomForestModule(num_classes=len(classes), random_state=seed).fit(loader, device=device)
        for seed in seeds
    )
    return AuxiliaryModels((forests[0], forests[1]), forests[2], seeds)


def main():
    training_config = TrainingConfig(
        max_epochs=10,
        batch_size=32,
        model_type="resnet",
        initial_lr=1e-4,
        weight_decay=1e-3,
        image_size=224,
    )

    data = prepare_data(
        disease="DR",
        train_split="train",
        transform=transforms.v2.Compose(
            [
                transforms.v2.Resize((training_config.image_size, training_config.image_size)),
                transforms.v2.ToDtype(torch.float32, scale=True),
                transforms.v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
        test_split="val",
        class_to_idx=None,
        fairvision_data_root=RAW_DATA_DIR / "fairvision",
    )

    model = create_model(
        model_type=training_config.model_type, out_dim=len(data.classes), use_pretrained=True
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=training_config.initial_lr,
        weight_decay=training_config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, training_config.max_epochs)

    training_setup = TrainingSetup(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    train_model(training_setup, training_config, data.train, val_dataset=data.test)


if __name__ == "__main__":
    main()
