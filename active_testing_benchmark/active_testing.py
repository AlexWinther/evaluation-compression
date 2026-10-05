"""Active finite-pool risk estimation and its MLflow-backed image-classification runner."""

# ruff: noqa: B008 -- Typer declares CLI options as argument defaults.

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import typer

from active_testing_benchmark.config import RAW_DATA_DIR, REPORTS_DIR
from active_testing_benchmark.modeling.data import ImageClassificationDataset
from active_testing_benchmark.modeling.models import torchvision_transform
from active_testing_benchmark.modeling.provenance import FAIRVISION_SOURCE
from active_testing_benchmark.modeling.registry import (
    configure_mlflow,
    latest_model_version,
    normalize_model_source,
)

app = typer.Typer(add_completion=False, help=__doc__)


@dataclass
class SelectionPlan:
    """Items requested and probabilities under the method's sampling design.

    Selection probabilities describe ordered conditional draws. Inclusion
    probabilities describe the marginal chance that each selected item is included.
    """

    indices: list[int]
    selection_probabilities: list[float] | None = None
    inclusion_probabilities: list[float] | None = None
    metadata: dict | None = None


@dataclass
class Estimate:
    """A method's current estimate of finite-pool risk."""

    risk: float
    n_labels: int


class ActiveEvaluationMethod:
    """Interface implemented by methods that choose labels and estimate risk."""

    def initialize(self, problem):
        raise NotImplementedError

    def acquire(self, state, history) -> SelectionPlan:
        raise NotImplementedError

    def observe(self, state, selection, labels, losses):
        raise NotImplementedError

    def estimate(self, state, history) -> Estimate:
        raise NotImplementedError


class UniformSampleMean(ActiveEvaluationMethod):
    """Uniform sampling without replacement and the mean of observed losses."""

    def __init__(self, seed: int = 0, batch_size: int | None = None) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.batch_size = batch_size

    def initialize(self, problem: dict) -> dict:
        return {
            "remaining": set(range(problem["N"])),
            "remaining_budget": problem["budget"],
            "observed_losses": [],
        }

    def acquire(self, state: dict, history: list[dict]) -> SelectionPlan:
        remaining = sorted(state["remaining"])
        count = min(
            self.batch_size or state["remaining_budget"],
            state["remaining_budget"],
            len(remaining),
        )
        if count < 1:
            raise ValueError("No remaining budget or unlabeled items")

        positions = self.rng.choice(len(remaining), size=count, replace=False)
        indices = [int(remaining[int(position)]) for position in positions]
        return SelectionPlan(
            indices=indices,
            selection_probabilities=[1.0 / (len(remaining) - offset) for offset in range(count)],
            inclusion_probabilities=[count / len(remaining)] * count,
            metadata={"design": "uniform_without_replacement"},
        )

    def observe(self, state: dict, selection: SelectionPlan, labels: list, losses: list) -> None:
        state["remaining"].difference_update(selection.indices)
        state["remaining_budget"] -= len(selection.indices)
        state["observed_losses"].extend(losses)

    def estimate(self, state: dict, history: list[dict]) -> Estimate:
        losses = state["observed_losses"]
        return Estimate(risk=float(np.mean(losses)), n_labels=len(losses))


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


def run_active_evaluation(
    method: ActiveEvaluationMethod,
    X_pool,
    y_pool,
    model,
    loss_fn,
    budget: int,
    history_out: list[dict] | None = None,
) -> Estimate:
    """Run the label-acquisition loop against a finite-pool label oracle."""
    if len(X_pool) != len(y_pool):
        raise ValueError("X_pool and y_pool must have the same length")
    if not 1 <= budget <= len(X_pool):
        raise ValueError("budget must be between 1 and the pool size")

    problem = {
        "N": len(X_pool),
        "X_pool": X_pool,
        "model": model,
        "loss_fn": loss_fn,
        "budget": budget,
    }
    state = method.initialize(problem)
    history = []
    acquired = set()

    while len(acquired) < budget:
        selection = method.acquire(state, history)
        indices = selection.indices
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("A selection must contain unique indices")
        if any(not isinstance(index, (int, np.integer)) for index in indices):
            raise ValueError("Selection indices must be integers")
        if len(acquired) + len(indices) > budget:
            raise ValueError("Selection exceeds the remaining label budget")
        if any(index < 0 or index >= len(X_pool) or index in acquired for index in indices):
            raise ValueError("Selection indices must be remaining pool indices")

        labels = [y_pool[index] for index in indices]
        losses = [
            float(loss_fn(model(X_pool[index]), label)) for index, label in zip(indices, labels)
        ]
        for probabilities in (
            selection.selection_probabilities,
            selection.inclusion_probabilities,
        ):
            if probabilities is not None and (
                len(probabilities) != len(indices)
                or not np.isfinite(probabilities).all()
                or any(probability < 0 or probability > 1 for probability in probabilities)
            ):
                raise ValueError(
                    "Selection probabilities must align with selected indices in [0, 1]"
                )
        history.append(
            {"round": len(history), "selection": selection, "labels": labels, "losses": losses}
        )
        method.observe(state, selection, labels, losses)
        acquired.update(indices)

    if history_out is not None:
        history_out.extend(history)
    return method.estimate(state, history)


@dataclass
class Predictions:
    row_ids: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


def classification_error(prediction, target) -> float:
    """Return zero-one loss for a class-probability vector and integer target."""
    return float(int(np.argmax(prediction)) != int(target))


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
    method: str = typer.Option("uniform-sample-mean"),
    metadata_path: Path = typer.Option(RAW_DATA_DIR / "fairvision/dr/metadata.csv"),
    image_root: Path | None = typer.Option(None),
    image_column: str = typer.Option("filename"),
    target_column: str = typer.Option("dr"),
    split_column: str = typer.Option("use"),
    dataset_source: str = typer.Option(FAIRVISION_SOURCE),
    image_size: int | None = typer.Option(None, min=32),
    batch_size: int = typer.Option(32, min=1),
    acquisition_batch_size: int | None = typer.Option(None, min=1),
    num_workers: int = typer.Option(0, min=0),
    output_dir: Path = typer.Option(REPORTS_DIR / "active-testing"),
    experiment_name: str = typer.Option("active-testing"),
) -> None:
    """Estimate finite-pool classification error using an active label budget."""
    try:
        if method != "uniform-sample-mean":
            raise ValueError("Only the uniform-sample-mean method is currently supported")
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
    with mlflow.start_run(run_name=f"{method}-seed-{seed}") as run:
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
        result_rows, manifest_rows = [], []
        for size_index, budget in enumerate(budgets):
            for repetition in range(runs):
                child_seed = int(
                    np.random.SeedSequence([seed, size_index, repetition]).generate_state(
                        1, dtype=np.uint64
                    )[0]
                )
                for name, data in models.items():
                    sampled_history = []
                    model_predictions = data.probabilities
                    prediction_model = lambda index, outputs=model_predictions: outputs[int(index)]
                    estimate = run_active_evaluation(
                        UniformSampleMean(child_seed, acquisition_batch_size),
                        np.arange(len(data.labels)),
                        data.labels,
                        prediction_model,
                        classification_error,
                        budget,
                        history_out=sampled_history,
                    )
                    full_risk = float(np.mean(data.predictions != data.labels))
                    result_rows.append(
                        {
                            "method": method,
                            "model": name,
                            "metric": "classification_error",
                            "budget": budget,
                            "run": repetition,
                            "seed": child_seed,
                            "estimate": estimate.risk,
                            "full_value": full_risk,
                            "error": estimate.risk - full_risk,
                            "absolute_error": abs(estimate.risk - full_risk),
                            "n_labels": estimate.n_labels,
                        }
                    )
                    selection_order = 0
                    for batch in sampled_history:
                        selection = batch["selection"]
                        for offset, pool_index in enumerate(selection.indices):
                            manifest_rows.append(
                                {
                                    "method": method,
                                    "model": name,
                                    "budget": budget,
                                    "run": repetition,
                                    "seed": child_seed,
                                    "row_id": int(data.row_ids[pool_index]),
                                    "selection_round": batch["round"],
                                    "selection_order": selection_order + offset,
                                    "selection_probability": (
                                        selection.selection_probabilities[offset]
                                        if selection.selection_probabilities is not None
                                        else None
                                    ),
                                    "inclusion_probability": (
                                        selection.inclusion_probabilities[offset]
                                        if selection.inclusion_probabilities is not None
                                        else None
                                    ),
                                    "label_revealed": True,
                                }
                            )
                        selection_order += len(selection.indices)
        results = pd.DataFrame(result_rows)
        manifests = pd.DataFrame(manifest_rows)
        config = {
            "method": method,
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
            "row_id": "zero-based CSV row index",
            "loss": "zero-one classification error",
            "acquisition_batch_size": acquisition_batch_size or "full_budget",
        }
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        results.to_csv(destination / "results.csv", index=False)
        manifests.to_csv(destination / "selection_manifest.csv", index=False)
        mlflow.log_params(
            {
                "method": method,
                "runs": runs,
                "seed": seed,
                "population": population,
                "acquisition_batch_size": acquisition_batch_size or "full_budget",
            }
        )
        for row in results.itertuples():
            mlflow.log_metric(f"risk/{row.model}/full", row.full_value)
            mlflow.log_metric(f"risk/{row.model}/estimate/budget-{row.budget}", row.estimate)
        typer.echo("Uploading configuration, risk estimates, and selection manifest to MLflow...")
        mlflow.log_artifacts(str(destination))
    typer.echo(f"Results: {destination}")


if __name__ == "__main__":
    app()
