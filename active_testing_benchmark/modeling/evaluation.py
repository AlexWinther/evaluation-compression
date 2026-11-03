"""Evaluation metrics for trained image classifiers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    protected_column: str | list[str] | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict:
    """Evaluate per-example CE, accuracy, AUC, and protected-group risks."""
    model.eval()
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
        probabilities = logits.softmax(dim=1).cpu().tolist()
        predicted = logits.argmax(dim=1).cpu().tolist()
        targets_list = targets.cpu().tolist()
        losses = criterion(logits, targets).cpu().tolist()
        weighted_losses = (
            weighted_criterion(logits, targets).cpu().tolist()
            if weighted_criterion is not None
            else None
        )
        for i, image_id in enumerate(batch["id"]):
            protected = {
                name: values[i].item() if isinstance(values[i], torch.Tensor) else values[i]
                for name, values in batch["protected"].items()
                if protected_column is None or name in protected_column
            }
            record = {
                "id": image_id,
                "target": targets_list[i],
                "prediction": predicted[i],
                "probabilities": probabilities[i],
                "loss": losses[i],
                "protected": protected,
            }
            if weighted_losses is not None:
                record["weighted_loss"] = weighted_losses[i]
            predictions.append(record)
            for attribute, value in protected.items():
                subgroups[attribute][str(value)].append(record)

    if not predictions:
        raise ValueError("Cannot evaluate an empty data loader")

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
        return {
            "n": len(records),
            "loss": sum(r["loss"] for r in records) / len(records),
            "accuracy": sum(r["target"] == r["prediction"] for r in records) / len(records),
            "roc_auc": auc,
            "weighted_loss": (
                sum(r["weighted_loss"] for r in records) / len(records)
                if weighted_criterion is not None
                else None
            ),
        }

    overall = compute_metrics(predictions)
    subgroup_metrics = {
        attribute: {group: compute_metrics(records) for group, records in groups.items()}
        for attribute, groups in subgroups.items()
    }
    return {
        "loss": overall["loss"],
        "weighted_loss": overall["weighted_loss"],
        "accuracy": overall["accuracy"],
        "roc_auc": overall["roc_auc"],
        "subgroup_risks": {
            attribute: {group: metrics["loss"] for group, metrics in groups.items()}
            for attribute, groups in subgroup_metrics.items()
        },
        "subgroup_weighted_risks": {
            attribute: {group: metrics["weighted_loss"] for group, metrics in groups.items()}
            for attribute, groups in subgroup_metrics.items()
        },
        "subgroup_metrics": subgroup_metrics,
        "predictions": predictions,
    }
