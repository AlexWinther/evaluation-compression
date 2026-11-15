"""Shared evaluation, checkpoint, and metric logging helpers for experiments."""

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
from time import perf_counter

from loguru import logger
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from active_testing_benchmark.modeling.models import create_model
from active_testing_benchmark.modeling.registry import log_metrics
from active_testing_benchmark.modeling.train import (
    TrainingSetup,
    train_model,
)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device | str
) -> dict[str, object]:
    """Evaluate overall and per-subgroup loss, accuracy, F1, and AUC."""
    model.eval()
    labels, probabilities, losses = [], [], []
    subgroups = defaultdict(list)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        targets = batch["label"].long().to(device)
        logits = model(batch["image"].to(device))
        losses.append(nn.functional.cross_entropy(logits, targets, reduction="none").cpu().numpy())
        labels.append(targets.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
        for name, values in batch.get("subgroups", {}).items():
            subgroups[name].extend(values)

    y_true = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    losses = np.concatenate(losses)

    def metrics(targets, probabilities, sample_losses):
        predictions = probabilities.argmax(axis=1)
        num_classes = probabilities.shape[1]
        auc = None
        if np.array_equal(np.unique(targets), np.arange(num_classes)):
            auc = (
                roc_auc_score(targets, probabilities[:, 1])
                if num_classes == 2
                else roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")
            )
        return {
            "loss": float(sample_losses.mean()),
            "accuracy": accuracy_score(targets, predictions),
            "f1": f1_score(
                targets,
                predictions,
                average="binary" if num_classes == 2 else "macro",
                labels=np.arange(num_classes),
                zero_division=0.0,
            ),
            "auc": float(auc) if auc is not None else None,
        }

    result = metrics(y_true, scores, losses)
    subgroup_metrics = {}
    for name, values in subgroups.items():
        values = np.asarray(values)
        subgroup_metrics[name] = {}
        for value in np.unique(values):
            mask = values == value
            subgroup_metrics[name][str(value)] = {
                "n": int(mask.sum()),
                **metrics(y_true[mask], scores[mask], losses[mask]),
            }
    result["subgroup_metrics"] = subgroup_metrics
    return result


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a checkpoint saved by `_train_or_load` onto the CPU."""
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def model_from_checkpoint(checkpoint: dict, device: torch.device | str) -> nn.Module:
    """Rebuild the trained classifier stored in a checkpoint, in evaluation mode."""
    model = create_model(
        checkpoint["config"]["model_type"],
        out_dim=len(checkpoint["classes"]),
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def reused_checkpoint_seed(checkpoint_path: Path, requested_seed: int | None) -> int:
    """Return the seed that trained a reused checkpoint, rejecting a conflicting seed.

    The seed also fixes the 80/20 development split, the auxiliary-model seeds and the
    benchmark repetitions. A reused model must therefore run with its training seed,
    or the holdout can contain images the model was trained on.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint to reuse at {checkpoint_path}")
    training_seed = int(load_checkpoint(checkpoint_path)["config"]["random_state"])
    if requested_seed is not None and requested_seed != training_seed:
        raise ValueError(
            f"--seed {requested_seed} does not match the seed {training_seed} that trained "
            f"the reused checkpoint {checkpoint_path}. Omit --seed or pass --seed {training_seed}."
        )
    return training_seed


def _train_or_load(config, classes, train, validation, device, checkpoint_path, reuse):
    if reuse:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["classes"] != classes or any(
            checkpoint["config"][name] != getattr(config, name)
            for name in ("model_type", "image_size")
        ):
            raise ValueError(
                f"Checkpoint does not match the model, classes or image size: {checkpoint_path}"
            )
        model = model_from_checkpoint(checkpoint, device)
        logger.info(
            f"Loaded {checkpoint_path} (training seed: {checkpoint['config']['random_state']})"
        )
        _log_training(checkpoint, checkpoint_path, reused=True)
        return model, checkpoint

    if mlflow.active_run() is not None:
        mlflow.set_tag("stage", "training")
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    model = create_model(config.model_type, out_dim=len(classes), use_pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)
    setup = TrainingSetup(
        model,
        optimizer=optimizer,
        scheduler=CosineAnnealingLR(optimizer, T_max=config.max_epochs),
    )
    torch.manual_seed(config.random_state)
    start = perf_counter()
    model, history = train_model(setup, config, train, val_dataset=validation, device=device)
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "config": asdict(config),
        "classes": list(classes),
        "history": json.loads(json.dumps(history)),
        "training_seconds": perf_counter() - start,
        "training_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved best checkpoint to {checkpoint_path}")
    _log_training(checkpoint, checkpoint_path, reused=False)

    return model, checkpoint


def _log_training(checkpoint, checkpoint_path, *, reused):
    if mlflow.active_run() is None:
        return
    mlflow.log_params({f"training.{name}": value for name, value in checkpoint["config"].items()})
    mlflow.log_param("reused_checkpoint", reused)
    if checkpoint.get("training_run_id"):
        mlflow.set_tag("training.source_run_id", checkpoint["training_run_id"])
    mlflow.log_dict(
        {name: value for name, value in checkpoint.items() if name != "model_state_dict"},
        "training.json",
    )
    mlflow.log_artifact(str(checkpoint_path), "checkpoints")
    best = max(
        (row for row in checkpoint["history"] if row["roc_auc"] is not None),
        key=lambda row: row["roc_auc"],
    )
    log_metrics({"best_epoch": best["epoch"], "best_validation_auc": best["roc_auc"]})


def _log_evaluation(metrics):
    if mlflow.active_run() is None:
        return
    mlflow.set_tag("stage", "evaluation")
    mlflow.log_dict(metrics, "reference_metrics.json")
    log_metrics(
        {name: value for name, value in metrics.items() if name != "subgroup_metrics"},
        prefix="evaluation",
    )
    for attribute, groups in metrics["subgroup_metrics"].items():
        for group, values in groups.items():
            log_metrics(values, prefix=f"evaluation/{attribute}/{group}")
