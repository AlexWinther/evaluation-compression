"""Manually register a reviewed MLflow training run as a research baseline."""

from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient
import typer

from active_testing_benchmark.config import MLFLOW_DB_PATH

app = typer.Typer(add_completion=False, help=__doc__)


def configure_mlflow() -> str:
    """Configure the local store unless callers provide a remote tracking URI."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri))
    return tracking_uri


def register_baseline(run_id: str, registry_name: str, alias: str = "baseline") -> str:
    """Register a reviewed run's canonical model and move its named alias to it."""
    configure_mlflow()
    run = mlflow.get_run(run_id)
    model_version = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
    client = MlflowClient()
    tags = {
        "review.status": "accepted",
        "source.run_id": run_id,
        "source.experiment_id": run.info.experiment_id,
    }
    for metric_name in ("test_accuracy", "test_macro_f1", "test_roc_auc", "best_validation_loss"):
        if metric_name in run.data.metrics:
            tags[f"metric.{metric_name}"] = str(run.data.metrics[metric_name])
    for key, value in tags.items():
        client.set_model_version_tag(registry_name, model_version.version, key, value)
    client.set_registered_model_alias(registry_name, alias, model_version.version)
    return model_version.version


@app.command()
def main(
    run_id: str = typer.Option(
        ..., help="Reviewed training run ID containing the `model` artifact."
    ),
    registry_name: str = typer.Option(..., help="Destination registered model name."),
    alias: str = typer.Option("baseline", help="Movable alias for the selected model version."),
) -> None:
    """Create one reviewed registry version; training never invokes this automatically."""
    version = register_baseline(run_id, registry_name, alias)
    typer.echo(f"Registered {registry_name} version {version}; @{alias} now resolves to it.")


if __name__ == "__main__":
    app()
