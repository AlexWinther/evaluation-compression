"""Configure the remote MLflow registry and manage registered model versions."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def fairvision_model_name(disease_type: str, model_type: str) -> str:
    """Return the canonical registered-model name for a FairVision classifier."""
    components = []
    for label, value in (("disease type", disease_type), ("model type", model_type)):
        component = value.strip().lower().replace("_", "-")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", component):
            raise ValueError(f"Invalid {label}: {value!r}")
        components.append(component)
    return f"fairvision-{components[0]}-{components[1]}"


def configure_mlflow() -> str:
    """Configure MLflow from explicit environment variables."""
    load_dotenv()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is required. Set it to the remote MLflow server "
            "(for example, https://mlflow.alwn.dev)."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri))
    return tracking_uri


def latest_model_version(registry_name: str, client: MlflowClient | None = None):
    """Return the highest numbered version of a registered model."""
    if "'" in registry_name:
        raise ValueError("Registered model names containing quotes are not supported")
    versions = (client or MlflowClient()).search_model_versions(
        filter_string=f"name = '{registry_name}'",
        max_results=1,
        order_by=["version_number DESC"],
    )
    if not versions:
        raise ValueError(f"Registered model has no versions: {registry_name}")
    return versions[0]


def model_version_for_run(registry_name: str, run_id: str, client: MlflowClient | None = None):
    """Return the newest registered-model version produced by ``run_id``."""
    if "'" in registry_name:
        raise ValueError("Registered model names containing quotes are not supported")
    versions = (client or MlflowClient()).search_model_versions(
        filter_string=f"name = '{registry_name}'",
        max_results=10_000,
        order_by=["version_number DESC"],
    )
    matching_versions = [version for version in versions if version.run_id == run_id]
    if not matching_versions:
        raise ValueError(f"No version of {registry_name} was produced by run {run_id}")
    return matching_versions[0]


def register_baseline(
    registry_name: str, run_id: str | None = None, alias: str = "baseline"
) -> str:
    """Promote the latest version, or a version selected by training run, to an alias."""
    configure_mlflow()
    client = MlflowClient()
    model_version = (
        model_version_for_run(registry_name, run_id, client)
        if run_id
        else latest_model_version(registry_name, client)
    )
    source_run_id = model_version.run_id
    if not source_run_id:
        raise ValueError(
            f"Model version {registry_name}/{model_version.version} has no source run"
        )
    run = mlflow.get_run(source_run_id)
    tags = {
        "review.status": "accepted",
        "source.run_id": source_run_id,
        "source.experiment_id": run.info.experiment_id,
    }
    for metric_name in ("test_accuracy", "test_macro_f1", "test_roc_auc", "best_validation_loss"):
        if metric_name in run.data.metrics:
            tags[f"metric.{metric_name}"] = str(run.data.metrics[metric_name])
    for key, value in tags.items():
        client.set_model_version_tag(registry_name, model_version.version, key, value)
    client.set_registered_model_alias(registry_name, alias, model_version.version)
    return model_version.version


def register_trained_model(
    model_uri: str,
    registry_name: str,
    run_id: str,
    experiment_id: str,
    disease_type: str,
    model_type: str,
    metrics: dict[str, float],
) -> str:
    """Register and tag a completed training model as its newest version."""
    tags = {
        "disease.type": disease_type,
        "model.type": model_type,
        "source.run_id": run_id,
        "source.experiment_id": experiment_id,
        **{f"metric.{name}": str(value) for name, value in metrics.items()},
    }
    model_version = mlflow.register_model(model_uri, registry_name, tags=tags)
    return model_version.version


@app.command()
def main(
    registry_name: str = typer.Option(..., help="Registered model to promote."),
    run_id: str | None = typer.Option(
        None, help="Optional older training run to promote instead of the latest version."
    ),
    alias: str = typer.Option("baseline", help="Movable alias for the selected model version."),
) -> None:
    """Promote a reviewed model's latest registered version to an alias."""
    version = register_baseline(registry_name, run_id, alias)
    typer.echo(f"Registered {registry_name} version {version}; @{alias} now resolves to it.")


if __name__ == "__main__":
    app()
