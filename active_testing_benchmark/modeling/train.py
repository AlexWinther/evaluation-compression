"""Train one image classifier and log the run to local MLflow."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import typer
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from active_testing_benchmark.config import (
    MLFLOW_DB_PATH,
    MODELS_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
)
from active_testing_benchmark.modeling.data import (
    ImageClassificationDataset,
    class_mapping,
)
from active_testing_benchmark.modeling.models import (
    create_model_and_transform,
    torchvision_transform,
)
from active_testing_benchmark.modeling.provenance import (
    FAIRVISION_SOURCE,
    build_dataset_provenance,
    git_provenance,
)

app = typer.Typer(add_completion=False, help=__doc__)
MODEL_NAMES = ["cnn", "resnet18", "densenet121", "vit", "clip"]
IMBALANCE_STRATEGIES = ["none", "weighted-sampling", "class-weighted-loss"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiment starts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float, list[int], list[float], list[int]]:
    """Train when an optimizer is supplied; otherwise evaluate."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = correct = total = 0
    labels: list[int] = []
    positive_probabilities: list[float] = []
    predictions: list[int] = []
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * targets.size(0)
            correct += (batch_predictions == targets).sum().item()
            total += targets.size(0)
            labels.extend(targets.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            if outputs.shape[1] == 2:
                positive_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
    return total_loss / total, correct / total, labels, positive_probabilities, predictions


def classification_metrics(
    labels: list[int],
    positive_probabilities: list[float],
    predictions: list[int],
    num_classes: int,
) -> dict[str, float]:
    """Compute imbalance-aware classification metrics when they are defined."""
    metrics = {"macro_f1": f1_score(labels, predictions, average="macro", zero_division=0.0)}
    if num_classes == 2 and len(set(labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(labels, positive_probabilities)
    return metrics


def inverse_frequency_class_weights(
    dataset: ImageClassificationDataset, num_classes: int
) -> torch.Tensor:
    """Return inverse-frequency class weights for labels present in ``dataset``."""
    labels = torch.tensor([label for _, label in dataset.records], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).to(dtype=torch.float)
    weights = torch.zeros(num_classes, dtype=torch.float)
    present = counts > 0
    weights[present] = len(dataset) / (num_classes * counts[present])
    return weights


def make_loader(
    dataset: ImageClassificationDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    imbalance_strategy: str = "none",
) -> DataLoader:
    if imbalance_strategy == "weighted-sampling":
        class_weights = inverse_frequency_class_weights(
            dataset, num_classes=max(label for _, label in dataset.records) + 1
        )
        sample_weights = torch.tensor(
            [class_weights[label] for _, label in dataset.records], dtype=torch.float
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True
    )


@app.command()
def main(
    model: str = typer.Option("resnet18", help="cnn, resnet18, densenet121, vit, or clip."),
    batch_size: int = typer.Option(32, min=1),
    epochs: int = typer.Option(5, min=1),
    learning_rate: float = typer.Option(1e-4, min=0.0),
    seed: int = typer.Option(42),
    num_workers: int = typer.Option(0, min=0),
    image_size: int = typer.Option(224, min=32),
    metadata_path: Path = typer.Option(  # noqa: B008 - Typer declares CLI options as defaults.
        RAW_DATA_DIR / "fairvision/dr/metadata.csv"
    ),
    image_root: Path | None = typer.Option(None),  # noqa: B008 - Typer CLI option default.
    image_column: str = typer.Option("filename"),
    target_column: str = typer.Option("dr"),
    split_column: str = typer.Option("use"),
    output_dir: Path = typer.Option(MODELS_DIR),  # noqa: B008 - Typer CLI option default.
    experiment_name: str = typer.Option("image-classification"),
    dataset_source: str = typer.Option(FAIRVISION_SOURCE),
    pretrained: bool = typer.Option(True, "--pretrained/--no-pretrained"),
    imbalance_strategy: str = typer.Option(
        "none", help="none, weighted-sampling, or class-weighted-loss."
    ),
) -> None:
    """Run a minimal train/validation/test image-classification experiment."""
    if model not in MODEL_NAMES:
        raise typer.BadParameter(f"Choose one of: {', '.join(MODEL_NAMES)}")
    if model == "vit" and image_size != 224:
        raise typer.BadParameter("torchvision vit_b_16 currently requires --image-size 224")
    if imbalance_strategy not in IMBALANCE_STRATEGIES:
        raise typer.BadParameter(f"Choose one of: {', '.join(IMBALANCE_STRATEGIES)}")
    metadata_path = metadata_path.resolve()
    image_root = (image_root or metadata_path.parent).resolve()
    if not metadata_path.is_file():
        raise typer.BadParameter(f"Metadata file does not exist: {metadata_path}")

    set_seed(seed)
    device = selected_device()
    class_to_index = class_mapping(metadata_path, target_column)
    classifier, train_transform = create_model_and_transform(
        model, len(class_to_index), pretrained, image_size
    )
    validation_transform = (
        train_transform if model == "clip" else torchvision_transform(image_size, training=False)
    )
    datasets = {
        split: ImageClassificationDataset(
            metadata_path,
            image_root,
            image_column,
            target_column,
            split_column,
            split,
            class_to_index,
            train_transform if split == "training" else validation_transform,
        )
        for split in ("training", "validation", "test")
    }
    loaders = {
        split: make_loader(
            dataset,
            batch_size,
            num_workers,
            shuffle=split == "training",
            imbalance_strategy=imbalance_strategy if split == "training" else "none",
        )
        for split, dataset in datasets.items()
    }
    classifier.to(device)
    optimizer = AdamW((p for p in classifier.parameters() if p.requires_grad), lr=learning_rate)
    class_weights = (
        inverse_frequency_class_weights(datasets["training"], len(class_to_index))
        if imbalance_strategy == "class-weighted-loss"
        else None
    )
    training_loss_function = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    evaluation_loss_function = nn.CrossEntropyLoss()
    validation_loss_function = (
        training_loss_function
        if imbalance_strategy == "class-weighted-loss"
        else evaluation_loss_function
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model}_best.pt"
    config = {
        "model": model,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "num_workers": num_workers,
        "image_size": image_size,
        "metadata_path": str(metadata_path),
        "image_root": str(image_root),
        "image_column": image_column,
        "target_column": target_column,
        "split_column": split_column,
        "pretrained": pretrained,
        "imbalance_strategy": imbalance_strategy,
        "dataset_source": dataset_source,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
    }
    config_path = output_dir / f"{model}_config.json"
    mapping_path = output_dir / f"{model}_classes.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    mapping_path.write_text(json.dumps(class_to_index, indent=2) + "\n")

    print(
        f"Model: {model}\nDevice: {device}\nImbalance strategy: {imbalance_strategy}"
        f"\nTrain samples: {len(datasets['training'])}"
    )
    print(
        f"Validation samples: {len(datasets['validation'])}\nTest samples: {len(datasets['test'])}"
    )
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri))
    mlflow.set_experiment(experiment_name)
    provenance = build_dataset_provenance(
        metadata_path,
        image_root,
        image_column,
        target_column,
        split_column,
        class_to_index,
        dataset_source,
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    with mlflow.start_run(run_name=f"{model}-seed-{seed}-{timestamp}"):
        mlflow.set_tags(
            {
                **git_provenance(PROJ_ROOT),
                "dataset.source": dataset_source,
                "dataset.metadata.sha256": provenance.metadata_digest,
                "dataset.manifest.sha256": provenance.manifest_digest,
                "model.family": model,
                "training.seed": str(seed),
                "runtime.python": sys.version.split()[0],
                "runtime.mlflow": mlflow.__version__,
                "runtime.torch": torch.__version__,
            }
        )
        for split, dataset_input in provenance.datasets.items():
            mlflow.log_input(dataset_input, context=split)
        mlflow.log_params(
            {
                **config,
                "optimizer": "AdamW",
                "device": str(device),
                "num_classes": len(class_to_index),
                "train_samples": len(datasets["training"]),
                "validation_samples": len(datasets["validation"]),
                "test_samples": len(datasets["test"]),
            }
        )
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(mapping_path))
        mlflow.log_artifact(str(PROJ_ROOT / "uv.lock"))
        mlflow.log_dict(
            {
                "dataset_source": dataset_source,
                "metadata_sha256": provenance.metadata_digest,
                "manifest_sha256": provenance.manifest_digest,
                "split_counts": provenance.split_counts,
                "image_bytes_logged": False,
                "metadata_rows_logged": False,
                "model_input_example": "synthetic zero tensor",
            },
            "provenance.json",
        )
        best_validation_loss = float("inf")
        best_validation_accuracy = 0.0
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy, train_labels, train_probabilities, train_predictions = (
                run_epoch(
                    classifier, loaders["training"], training_loss_function, device, optimizer
                )
            )
            (
                validation_loss,
                validation_accuracy,
                validation_labels,
                validation_probabilities,
                validation_predictions,
            ) = run_epoch(classifier, loaders["validation"], validation_loss_function, device)
            train_metrics = classification_metrics(
                train_labels, train_probabilities, train_predictions, len(class_to_index)
            )
            validation_metrics = classification_metrics(
                validation_labels,
                validation_probabilities,
                validation_predictions,
                len(class_to_index),
            )
            print(f"\nEpoch {epoch}/{epochs}")
            print(
                f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f}"
                f" | Train macro-F1: {train_metrics['macro_f1']:.4f}"
                + _roc_auc_report(train_metrics, "Train")
            )
            print(
                f"Val loss:   {validation_loss:.4f} | Val accuracy:   {validation_accuracy:.4f}"
                f" | Val macro-F1:   {validation_metrics['macro_f1']:.4f}"
                + _roc_auc_report(validation_metrics, "Val")
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    **{
                        f"train_{name}": value
                        for name, value in train_metrics.items()
                        if name == "roc_auc"
                    },
                    **{
                        f"validation_{name}": value
                        for name, value in validation_metrics.items()
                        if name == "roc_auc"
                    },
                },
                step=epoch,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss, best_validation_accuracy = (
                    validation_loss,
                    validation_accuracy,
                )
                torch.save(
                    {
                        "model_state_dict": classifier.state_dict(),
                        "class_to_index": class_to_index,
                        "config": config,
                    },
                    checkpoint_path,
                )
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        classifier.load_state_dict(state["model_state_dict"])
        test_loss, test_accuracy, test_labels, test_probabilities, test_predictions = run_epoch(
            classifier, loaders["test"], evaluation_loss_function, device
        )
        final_metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            **{
                f"test_{name}": value
                for name, value in classification_metrics(
                    test_labels, test_probabilities, test_predictions, len(class_to_index)
                ).items()
            },
        }
        mlflow.log_metrics(final_metrics)
        matrix_path = output_dir / f"{model}_test_confusion_matrix.csv"
        np.savetxt(
            matrix_path, confusion_matrix(test_labels, test_predictions), delimiter=",", fmt="%d"
        )
        for artifact in (checkpoint_path, matrix_path):
            mlflow.log_artifact(str(artifact))
        input_example = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
        classifier.eval()
        with torch.no_grad():
            output_example = classifier(input_example.to(device)).cpu().numpy()
        mlflow.pytorch.log_model(
            classifier,
            name="model",
            signature=infer_signature(input_example.numpy(), output_example),
            input_example=input_example.numpy(),
            code_paths=[str(PROJ_ROOT / "active_testing_benchmark")],
            extra_files=[str(config_path), str(mapping_path)],
            serialization_format="pickle",
        )
    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")
    if "test_roc_auc" in final_metrics:
        print(f"Test ROC-AUC: {final_metrics['test_roc_auc']:.4f}")


def _roc_auc_report(metrics: dict[str, float], split: str) -> str:
    """Format ROC-AUC only for binary splits containing both classes."""
    return f" | {split} ROC-AUC: {metrics['roc_auc']:.4f}" if "roc_auc" in metrics else ""


if __name__ == "__main__":
    app()
