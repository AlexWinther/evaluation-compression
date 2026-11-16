from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

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

from active_testing_benchmark.active_testing.surrogate import (
    ORACLE_KINDS,
    FrozenFeatureProbe,
    LabelOracle,
    MeanProbabilityEnsemble,
    RandomForestModule,
    TemperatureScaled,
    fit_temperature,
    predict_logits,
)
from active_testing_benchmark.config import RAW_DATA_DIR
from active_testing_benchmark.modeling.data import FairVisionData, prepare_data
from active_testing_benchmark.modeling.models import FrozenImageNetFeatures, create_model
from active_testing_benchmark.modeling.registry import log_metrics


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
    # "roc_auc" keeps the epoch with the highest validation AUC (best ranking).
    # "validation_loss" keeps the lowest validation cross-entropy, which also rewards
    # calibration. Active testing estimates the cross-entropy, so an overconfident
    # checkpoint changes the quantity being estimated, not only its ranking quality.
    checkpoint_metric: str = "roc_auc"
    # Stop after this many epochs without improvement of `checkpoint_metric`.
    # None trains for all `max_epochs`.
    early_stopping_patience: int | None = None
    # "cosine" anneals the learning rate to zero over `max_epochs`. "constant" keeps
    # `initial_lr` throughout, as FairVision trains its CNNs (Luo et al., 2024, Sec. 5,
    # with a step schedule that only decays after epoch 30).
    lr_schedule: str = "cosine"
    seed: int = 42
    num_workers: int = 0


CHECKPOINT_METRICS = ("roc_auc", "validation_loss")
LR_SCHEDULES = ("cosine", "constant")
# Values of fields added to TrainingConfig after checkpoints were first saved. A stored
# configuration without the field was trained with this value.
HISTORICAL_TRAINING_DEFAULTS = {
    "checkpoint_metric": "roc_auc",
    "early_stopping_patience": None,
    "lr_schedule": "cosine",
}


def stored_training_config(stored: dict) -> TrainingConfig:
    """Rebuild the TrainingConfig a checkpoint was trained with.

    Fields that did not exist when the checkpoint was saved take the value they
    implicitly had then (`HISTORICAL_TRAINING_DEFAULTS`).
    """
    known = {name for name in TrainingConfig.__dataclass_fields__}
    return TrainingConfig(
        **{
            **HISTORICAL_TRAINING_DEFAULTS,
            **{name: value for name, value in stored.items() if name in known},
        }
    )


def create_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LRScheduler | None:
    """Return the learning-rate schedule named by `config.lr_schedule`.

    None means a constant learning rate, since `train_model` then never steps one.
    """
    if config.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    if config.lr_schedule == "constant":
        return None
    raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")


def _is_better(metric: str, value: float | None, best: float | None) -> bool:
    """Return whether `value` improves on `best`; AUC is maximised and loss minimised."""
    if value is None:
        return False
    if best is None:
        return True
    return value > best if metric == "roc_auc" else value < best


def best_history_row(history: list[dict], metric: str = "roc_auc") -> dict:
    """Return the history row of the selected checkpoint for `metric`.

    The first epoch wins ties, as in `train_model`, which only replaces the best
    checkpoint on a strict improvement.
    """
    if metric not in CHECKPOINT_METRICS:
        raise ValueError(f"checkpoint_metric must be one of {CHECKPOINT_METRICS}")
    best = None
    for row in history:
        if _is_better(metric, row[metric], best[metric] if best else None):
            best = row
    if best is None:
        raise ValueError(f"No epoch has a valid {metric}")
    return best


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


def _epoch_metrics(labels, probabilities, loss: float) -> dict[str, float | None]:
    """Compute classification metrics once over all predictions from an epoch."""
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
    return {
        "loss": loss,
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


def train_model(
    training_setup: TrainingSetup,
    training_config: TrainingConfig,
    training_dataset: FairVisionData | Subset[FairVisionData],
    val_dataset: FairVisionData | Subset[FairVisionData],
    device: torch.device | str = "mps",
    *,
    metric_prefix: str = "training",
):
    """Return the best model and history, based on best validation checkpoint.

    Epoch metrics are logged under `<metric_prefix>/train` and `<metric_prefix>/validation`,
    so that surrogate members trained in the same MLflow run do not overwrite the
    history of the model under test.
    """
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
    metric = training_config.checkpoint_metric
    if metric not in CHECKPOINT_METRICS:
        raise ValueError(f"checkpoint_metric must be one of {CHECKPOINT_METRICS}")
    patience = training_config.early_stopping_patience
    if patience is not None and patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")
    history = []
    best_value = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in tqdm(
        range(1, training_config.max_epochs + 1), desc="Epochs", leave=False, position=0
    ):
        model.train()
        epoch_loss = 0
        denominator_total = 0
        train_labels, train_probabilities = [], []

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
            train_labels.append(targets.cpu().numpy())
            train_probabilities.append(logits.detach().softmax(dim=1).cpu().numpy())

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

        train_metrics = _epoch_metrics(
            train_labels, train_probabilities, epoch_loss / denominator_total
        )
        validation_metrics = _epoch_metrics(
            labels, probabilities, validation_loss / validation_count
        )
        auc = validation_metrics["roc_auc"]
        # Keep the validation history keys used by existing checkpoint readers.
        history.append(
            {
                "epoch": epoch,
                "training_loss": train_metrics["loss"],
                "training_accuracy": train_metrics["accuracy"],
                "training_f1": train_metrics["f1"],
                "training_roc_auc": train_metrics["roc_auc"],
                "validation_loss": validation_metrics["loss"],
                "accuracy": validation_metrics["accuracy"],
                "f1": validation_metrics["f1"],
                "roc_auc": auc,
            }
        )
        logger.info(f"Epoch {epoch}: {history[-1]}")
        log_metrics(train_metrics, prefix=f"{metric_prefix}/train", step=epoch)
        log_metrics(validation_metrics, prefix=f"{metric_prefix}/validation", step=epoch)
        if scheduler is not None:
            scheduler.step()
        if _is_better(metric, history[-1][metric], best_value):
            best_value = history[-1][metric]
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience is not None and epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping after epoch {epoch}: no {metric} improvement "
                f"for {patience} epochs"
            )
            break

    if best_state is None:
        raise ValueError("Training completed without a valid best checkpoint")
    model.load_state_dict(best_state)

    return model, history


SURROGATE_KINDS = ("forest", "resnet_ensemble")
PROXY_KINDS = ("resnet_ensemble", "imagenet_probe")
ENSEMBLE_SIZE = 5
# Members stop after this many epochs without a lower validation loss (validation-loss early stopping for the deep-ensemble surrogate).
ENSEMBLE_EARLY_STOPPING_PATIENCE = 3


@dataclass
class AuxiliaryModels:
    """The surrogate ensemble and proxy shared by all active-testing methods of a run.

    `surrogates` are the calibrated ensemble members. AT averages them, ASE and PPAT use
    them as members. `proxy` is the fixed PPAT proxy labeller. `seeds` are the member
    seeds of the surrogate ensemble.
    """

    surrogates: tuple[nn.Module, ...]
    proxy: nn.Module
    seeds: tuple[int, ...]
    surrogate_kind: str = "forest"
    proxy_kind: str = "resnet_ensemble"


def member_seeds(random_state: int, kind: str, count: int = ENSEMBLE_SIZE) -> tuple[int, ...]:
    """Derive independent member seeds for one ensemble kind from the experiment seed.

    Each kind has its own branch of the seed sequence, so the forests and the ResNets
    of one experiment never share a seed, and adding a kind changes no existing seed.
    """
    branch = SURROGATE_KINDS.index(kind)
    children = np.random.SeedSequence(random_state).spawn(len(SURROGATE_KINDS))[branch]
    return tuple(int(child.generate_state(1)[0]) for child in children.spawn(count))


def oracle_seeds(random_state: int) -> tuple[int, int]:
    """Derive the noise seeds of an oracle surrogate and an oracle proxy.

    The two differ, so a noisy oracle surrogate and a noisy oracle proxy make their
    mistakes on different items. Their branch of the seed sequence comes after the
    branches of `member_seeds`, so no fitted member shares a seed with an oracle.
    """
    branch = np.random.SeedSequence(random_state).spawn(len(SURROGATE_KINDS) + 1)[-1]
    surrogate_seed, proxy_seed = branch.generate_state(2)
    return int(surrogate_seed), int(proxy_seed)


def _loader(dataset, config: TrainingConfig, device: torch.device | str) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=str(device).startswith("cuda"),
    )


def fit_forest_ensemble(
    train: FairVisionData | Subset[FairVisionData],
    holdout: FairVisionData | Subset[FairVisionData],
    classes: list[str],
    config: TrainingConfig,
    device: torch.device | str,
) -> tuple[tuple[RandomForestModule, ...], tuple[int, ...]]:
    """Fit five bootstrapped, temperature-calibrated forests on frozen ImageNet features.

    This is the cheap surrogate condition. Each member draws its own bootstrap of the
    development split, so the members disagree where the finite training data leaves
    the forest uncertain (see `RandomForestModule`). The members share one frozen
    feature extractor to avoid holding five copies of the backbone.
    """
    seeds = member_seeds(config.random_state, "forest")
    extractor = FrozenImageNetFeatures()
    forests = tuple(
        RandomForestModule(
            num_classes=len(classes),
            feature_extractor=extractor,
            random_state=seed,
            bootstrap_development=True,
        )
        .fit(_loader(train, config, device), device=device)
        .calibrate(_loader(holdout, config, device), device=device)
        for seed in seeds
    )
    for index, forest in enumerate(forests):
        logger.info(f"Forest member {index}: holdout temperature {forest.temperature:.3f}")
    return forests, seeds


def train_resnet_ensemble(
    train: FairVisionData | Subset[FairVisionData],
    holdout: FairVisionData | Subset[FairVisionData],
    classes: list[str],
    config: TrainingConfig,
    device: torch.device | str,
    cache_dir: Path | None = None,
) -> tuple[tuple[TemperatureScaled, ...], tuple[int, ...]]:
    """Train or load the expensive surrogate: five fine-tuned, calibrated ResNet-18s.

    This follows the deep-ensemble surrogates of Kossen et al. (2021, Sec. 5.3) and
    Kossen et al. (2022, Sec. 7.3). Members use the same architecture, optimiser,
    schedule and data as the model under test and differ only in their seed. The model
    under test is not a member. Each member keeps its lowest validation loss checkpoint,
    stops after `ENSEMBLE_EARLY_STOPPING_PATIENCE` epochs without improvement, and is then
    temperature-scaled on the same holdout (Kossen et al., 2022, app. D.3). The members
    share one training split, since bagging hurt deep ensembles in Lakshminarayanan et
    al. (2017). Their diversity comes from the seed alone.
    The ensemble is trained once and never updated during acquisition.

    Members are cached in `cache_dir` and reused when their seed, classes and training
    configuration match, so that repeated benchmarks and diagnostics share one ensemble.
    """
    seeds = member_seeds(config.random_state, "resnet_ensemble")
    # The members keep their own recipe whatever schedule trained the model under test
    # (agreed 16.11.2026: the surrogates need not follow FairVision).
    member_config = replace(
        config,
        checkpoint_metric="validation_loss",
        early_stopping_patience=ENSEMBLE_EARLY_STOPPING_PATIENCE,
        lr_schedule="cosine",
    )
    members = []
    for index, seed in enumerate(seeds):
        path = None if cache_dir is None else Path(cache_dir) / f"member-{index}.pt"
        expected = asdict(replace(member_config, random_state=seed))
        if path is not None and path.exists():
            stored = torch.load(path, map_location="cpu", weights_only=True)
            stored_config = {**HISTORICAL_TRAINING_DEFAULTS, **stored["config"]}
            if stored_config != expected or stored["classes"] != list(classes):
                raise ValueError(
                    f"Cached ensemble member {path} was trained with a different "
                    "configuration. Remove the cache directory to retrain it."
                )
            model = create_model(config.model_type, out_dim=len(classes), use_pretrained=False)
            model.load_state_dict(stored["model_state_dict"])
            members.append(TemperatureScaled(model, stored["temperature"]).to(device).eval())
            logger.info(f"Loaded ensemble member {index} from {path}")
            continue

        torch.manual_seed(seed)
        model = create_model(config.model_type, out_dim=len(classes), use_pretrained=True)
        model = model.to(device)
        optimizer = AdamW(
            model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay
        )
        setup = TrainingSetup(
            model, optimizer=optimizer, scheduler=create_scheduler(optimizer, member_config)
        )
        torch.manual_seed(seed)
        model, history = train_model(
            setup,
            replace(member_config, random_state=seed),
            train,
            holdout,
            device=device,
            metric_prefix=f"surrogate_ensemble/member_{index}",
        )
        logits, labels = predict_logits(model, holdout, device, config.batch_size)
        temperature = fit_temperature(logits, labels)
        logger.info(f"Ensemble member {index}: holdout temperature {temperature:.3f}")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "temperature": temperature,
                    "config": expected,
                    "classes": list(classes),
                    "history": history,
                },
                path,
            )
        members.append(TemperatureScaled(model, temperature).to(device).eval())
    return tuple(members), seeds


def fit_auxiliary_models(
    train: FairVisionData | Subset[FairVisionData],
    holdout: FairVisionData | Subset[FairVisionData],
    classes: list[str],
    config: TrainingConfig,
    device: torch.device | str,
    *,
    surrogate_kind: str = "forest",
    proxy_kind: str = "resnet_ensemble",
    ensemble_dir: Path | None = None,
    oracle_noise: float = 0.0,
) -> AuxiliaryModels:
    """Fit the surrogate ensemble and the PPAT proxy for one experiment.

    Both are fitted on the development split `train`. `holdout` is the split that
    selected the checkpoint of the model under test. It early-stops and calibrates the
    surrogates, and is never the evaluation pool.

    The proxy is chosen independently of the surrogate, so that both surrogate
    conditions are compared with the same proxy (as in PPAT's surrogate ablation,
    Ashouritaklimi et al., 2026, app. C.2.2). The `resnet_ensemble` proxy predicts the
    argmax of the deep ensemble's mean probabilities. `imagenet_probe` is a cheap
    logistic-regression probe on frozen ImageNet features, mainly for development runs.

    The kinds in `ORACLE_KINDS` are control conditions that read the true labels (see
    `LabelOracle`). They need no fitting, and `oracle_noise` sets their noise level. An
    oracle surrogate is a single member, and `seeds` then holds its noise seed.
    """
    if surrogate_kind not in SURROGATE_KINDS + ORACLE_KINDS:
        raise ValueError(f"surrogate_kind must be one of {SURROGATE_KINDS + ORACLE_KINDS}")
    if proxy_kind not in PROXY_KINDS + ORACLE_KINDS:
        raise ValueError(f"proxy_kind must be one of {PROXY_KINDS + ORACLE_KINDS}")
    surrogate_oracle_seed, proxy_oracle_seed = oracle_seeds(config.random_state)

    ensemble = None
    if "resnet_ensemble" in (surrogate_kind, proxy_kind):
        ensemble = train_resnet_ensemble(train, holdout, classes, config, device, ensemble_dir)

    if surrogate_kind == "forest":
        surrogates, seeds = fit_forest_ensemble(train, holdout, classes, config, device)
    elif surrogate_kind in ORACLE_KINDS:
        surrogates = (
            LabelOracle(len(classes), surrogate_kind, oracle_noise, surrogate_oracle_seed),
        )
        seeds = (surrogate_oracle_seed,)
    else:
        surrogates, seeds = ensemble

    if proxy_kind == "resnet_ensemble":
        proxy = MeanProbabilityEnsemble(ensemble[0])
    elif proxy_kind in ORACLE_KINDS:
        # PPAT uses the proxy's hard label, so the noisy grader returns its label.
        proxy = LabelOracle(
            len(classes), proxy_kind, oracle_noise, proxy_oracle_seed, hard_labels=True
        )
    else:
        proxy = FrozenFeatureProbe(len(classes), FrozenImageNetFeatures()).fit(
            _loader(train, config, device), device=device
        )
    return AuxiliaryModels(tuple(surrogates), proxy, seeds, surrogate_kind, proxy_kind)


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
