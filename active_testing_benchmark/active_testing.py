"""Offline subset estimation benchmark for the project's MLflow PyTorch models."""

# ruff: noqa: B008 -- Typer declares CLI options as argument defaults.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Protocol

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import typer

from active_testing_benchmark.config import RAW_DATA_DIR, REPORTS_DIR
from active_testing_benchmark.modeling.data import ImageClassificationDataset
from active_testing_benchmark.modeling.models import torchvision_transform
from active_testing_benchmark.modeling.provenance import FAIRVISION_SOURCE
from active_testing_benchmark.modeling.registry import configure_mlflow, latest_model_version

app = typer.Typer(add_completion=False, help=__doc__)


class Reducer(Protocol):
    """Return ordered unique candidate IDs; revealed labels comes only from earlier rounds.

    Prediction arrays, when supplied, align with candidate_ids. Callers own the
    label-reveal loop; the offline scorer must never supply unrevealed labels.
    """

    def __call__(
        self,
        candidate_ids: np.ndarray,
        budget: int,
        seed: int,
        *,
        model_predictions: Mapping[str, np.ndarray] | None = None,
        revealed_labels: Mapping[int, int] | None = None,
    ) -> np.ndarray: ...


class Metric(Protocol):
    def __call__(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        metadata: pd.DataFrame | None = None,
    ) -> dict[str, float]: ...


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


def random_reducer(
    candidate_ids: np.ndarray,
    budget: int,
    seed: int,
    *,
    model_predictions: Mapping[str, np.ndarray] | None = None,
    revealed_labels: Mapping[int, int] | None = None,
) -> np.ndarray:
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique")
    if not 1 <= budget <= len(candidate_ids):
        raise ValueError("Budget must be less than the candidate population")
    return np.random.default_rng(seed).choice(candidate_ids, budget, replace=False)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Fixed-class macro metrics and top-label ECE with 10 equal-width bins."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(probabilities.shape[1]),
        average="macro",
        zero_division="warn",
    )
    confidence = probabilities.max(axis=1).astype(np.float64)
    bins = np.minimum((confidence * 10).astype(int), 9)
    correct = predictions == labels
    ece = sum(
        np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
        for b in range(10)
        if np.any(bins == b)
    )
    auc = float("nan")
    if probabilities.shape[1] == 2 and len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probabilities[:, 1]))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "roc_auc": auc,
        "calibration_error": float(ece),
    }


@dataclass
class Predictions:
    row_ids: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


def evaluate_subsets(
    models: Mapping[str, Predictions],
    budgets: list[int],
    runs: int,
    seed: int,
    reducer: Reducer = random_reducer,
    metric: Metric = classification_metrics,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score shared, independent selections without disclosing labels to the reducer."""
    if not models or runs < 1 or seed < 0:
        raise ValueError("Supply models, positive runs, and a nonnegative seed")
    candidates = next(iter(models.values())).row_ids
    for data in models.values():
        if not np.array_equal(data.row_ids, candidates):
            raise ValueError("Models must be evaluated on identical ordered metadata rows")
    full = {
        name: metric(data.labels, data.predictions, data.probabilities)
        for name, data in models.items()
    }
    full_rows = [
        {"model": name, "metric": key, "value": value}
        for name, values in full.items()
        for key, value in values.items()
    ]
    rows, manifests = [], {}
    positions = {row_id: index for index, row_id in enumerate(candidates)}
    for size_index, budget in enumerate(tqdm(budgets, desc="Subset evaluation", unit="size")):
        for repetition in tqdm(range(runs), desc=f"Budget {budget}", unit="run", leave=False):
            child_seed = int(
                np.random.SeedSequence([seed, size_index, repetition]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            selected = reducer(candidates.copy(), budget, child_seed)
            if (
                len(selected) != budget
                or len(np.unique(selected)) != budget
                or not np.isin(selected, candidates).all()
            ):
                raise ValueError("Reducer must return budget unique candidate IDs")
            subset_id = f"size-{size_index}-run-{repetition}"
            common = {
                "subset_id": subset_id,
                "budget": budget,
                "run": repetition,
                "seed": child_seed,
            }
            manifests[subset_id] = pd.DataFrame(
                {
                    **common,
                    "row_id": selected,
                    "selection_order": np.arange(budget),
                }
            )
            indices = np.array([positions[row_id] for row_id in selected])
            for name, data in models.items():
                estimates = metric(
                    data.labels[indices], data.predictions[indices], data.probabilities[indices]
                )
                for key, estimate in estimates.items():
                    reference = full[name][key]
                    rows.append(
                        dict(
                            **common,
                            model=name,
                            metric=key,
                            estimate=estimate,
                            full_value=reference,
                            error=estimate - reference,
                            absolute_error=abs(estimate - reference),
                        )
                    )
    return pd.DataFrame(rows), pd.DataFrame(full_rows), manifests


def normalize_model_source(source: str) -> str:
    """Bare registry names use MLflow's native latest-version selector."""
    source = source.strip()
    if source.startswith("runs:/"):
        if not re.fullmatch(r"runs:/[^/]+/model", source):
            raise ValueError("Run URI must be runs:/<run-id>/model")
        return source
    if source.startswith("run:"):
        run_id = source[4:]
        if not run_id or "/" in run_id:
            raise ValueError("Expected run:<run-id>")
        return f"runs:/{run_id}/model"
    name = source.removeprefix("models:/")
    if name.endswith("@latest"):
        name = name.removesuffix("@latest")
        if re.fullmatch(r"[^/@:\s]+", name):
            return f"models:/{name}/latest"
    if re.fullmatch(r"[^/@:\s]+/(?:latest|[1-9][0-9]*)", name):
        return f"models:/{name}"
    if not re.fullmatch(r"[^/@:\s]+(?:@[^/@:\s]+)?", name):
        raise ValueError("Expected registry-name[@alias] or run:<run-id>")
    return f"models:/{name}" + ("/latest" if "@" not in name else "")


def infer_model(
    source: str,
    metadata_path: Path,
    image_root: Path,
    image_column: str,
    target_column: str,
    split_column: str,
    image_size: int | None,
    batch_size: int,
    num_workers: int,
) -> tuple[Predictions, dict]:
    # Resolve the movable alias once, recording the immutable version used.
    resolved = source
    if source.startswith("models:/"):
        model_reference = source[len("models:/") :]
        if "@" in model_reference:
            name, alias = model_reference.split("@")
            version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
            resolved = f"models:/{name}/{version.version}"
        elif model_reference.endswith("/latest"):
            name = model_reference.removesuffix("/latest")
            version = latest_model_version(name)
            resolved = f"models:/{name}/{version.version}"
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=resolved))
    configs = list(local.glob("extra_files/*_config.json"))
    mappings = list(local.glob("extra_files/*_classes.json"))
    if len(configs) != 1 or len(mappings) != 1:
        raise ValueError(f"{source}: expected training config and classes in model extra_files")
    config = json.loads(configs[0].read_text())
    mapping = json.loads(mappings[0].read_text())
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class mapping must contain contiguous indices starting at zero")
    size = image_size or int(config["image_size"])
    if config["model"] == "clip":
        from open_clip.transform import image_transform

        transform = image_transform(size, is_train=False)
    else:
        transform = torchvision_transform(size, training=False)
    dataset = ImageClassificationDataset(
        metadata_path,
        image_root,
        image_column,
        target_column,
        split_column,
        "test",
        mapping,
        transform,
    )
    classifier = mlflow.pytorch.load_model(str(local), map_location="cpu").eval()
    batches, labels = [], []
    with (
        torch.inference_mode(),
        tqdm(total=len(dataset), desc="Full-test inference", unit="image") as progress,
    ):
        for images, targets in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ):
            logits = classifier(images)
            if logits.ndim != 2 or logits.shape[1] != len(mapping):
                raise ValueError("Model output must be N x classes logits matching its mapping")
            probabilities = logits.softmax(dim=1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("Model returned non-finite probabilities")
            batches.append(probabilities)
            labels.append(targets.numpy())
            progress.update(len(targets))
    probabilities = np.concatenate(batches)
    return Predictions(
        dataset.row_ids, np.concatenate(labels), probabilities.argmax(axis=1), probabilities
    ), {
        "source": source,
        "resolved_uri": resolved,
        "image_size": size,
        "class_to_index": mapping,
        "training_config": config,
    }


@app.command()
def main(
    model: list[str] = typer.Option(..., help="Repeat registry-name[@alias] or run:<run-id>."),
    test_size: list[str] = typer.Option(..., help="Repeat counts (10) or fractions (0.1)."),
    runs: int = typer.Option(5, min=1),
    seed: int = typer.Option(42, min=0),
    reducer: str = typer.Option("random"),
    metadata_path: Path = typer.Option(RAW_DATA_DIR / "fairvision/dr/metadata.csv"),
    image_root: Path | None = typer.Option(None),
    image_column: str = typer.Option("filename"),
    target_column: str = typer.Option("dr"),
    split_column: str = typer.Option("use"),
    dataset_source: str = typer.Option(FAIRVISION_SOURCE),
    image_size: int | None = typer.Option(None, min=32),
    batch_size: int = typer.Option(32, min=1),
    num_workers: int = typer.Option(0, min=0),
    output_dir: Path = typer.Option(REPORTS_DIR / "active-testing"),
    experiment_name: str = typer.Option("active-testing"),
) -> None:
    """Infer once per model, score subsets, and record one MLflow experiment run."""
    try:
        if reducer != "random":
            raise ValueError("Only the random reducer is currently supported")
        sources = list(dict.fromkeys(normalize_model_source(source) for source in model))
        metadata_path = metadata_path.resolve()
        image_root = (image_root or metadata_path.parent).resolve()
        frame = pd.read_csv(metadata_path)
        population = int((frame[split_column].astype(str).str.lower() == "test").sum())
        budgets = [resolve_budget(value, population) for value in test_size]
    except (ValueError, KeyError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    configure_mlflow()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{reducer}-seed-{seed}") as run:
        destination = output_dir / run.info.run_id
        destination.mkdir(parents=True, exist_ok=False)
        models, provenance = {}, {}
        for index, source in enumerate(sources):
            name = f"model-{index}"
            typer.echo(f"Model {index + 1}/{len(sources)}: loading and inferring {source}")
            models[name], provenance[name] = infer_model(
                source,
                metadata_path,
                image_root,
                image_column,
                target_column,
                split_column,
                image_size,
                batch_size,
                num_workers,
            )
        results, full, manifests = evaluate_subsets(models, budgets, runs, seed)
        config = {
            "reducer": reducer,
            "runs": runs,
            "seed": seed,
            "test_sizes": test_size,
            "budgets": budgets,
            "population": population,
            "models": provenance,
            "metadata_path": str(metadata_path),
            "image_root": str(image_root),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "image_column": image_column,
            "target_column": target_column,
            "split_column": split_column,
            "dataset_source": dataset_source,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "image_size": image_size,
            "calibration_bins": 10,
            "row_id": "zero-based CSV row index",
        }
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        results.to_csv(destination / "results.csv", index=False)
        full.to_csv(destination / "full_metrics.csv", index=False)
        manifest_dir = destination / "subsets"
        manifest_dir.mkdir()
        for name, manifest in manifests.items():
            manifest.to_csv(manifest_dir / f"{name}.csv", index=False)
        mlflow.log_params(
            {"reducer": reducer, "runs": runs, "seed": seed, "population": population}
        )
        for row in full.itertuples():
            if math.isfinite(row.value):
                mlflow.log_metric(f"full/{row.model}/{row.metric}", row.value)
        typer.echo("Uploading configuration, metrics, and subset manifests to MLflow...")
        mlflow.log_artifacts(str(destination))
    typer.echo(f"Results: {destination}")


if __name__ == "__main__":
    app()
