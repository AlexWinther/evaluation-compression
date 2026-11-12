"""Explicit MLflow run setup and metric logging for the three experiment types."""

from contextlib import contextmanager
from dataclasses import asdict
import math
import os
import re

from dotenv import load_dotenv
from loguru import logger
import mlflow

from active_testing_benchmark.modeling.provenance import log_code_provenance

EXPERIMENT_NAMES = {
    "model_selection": "FairVision/model_selection",
    "AT_dev": "FairVision/AT_dev",
    "AT_final": "FairVision/AT_final",
    "feature_extractors": "FairVision/feature_extractors",
}


def configure_mlflow() -> str:
    """Use the tracking server and credentials configured in the environment or .env."""
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI in .env, or use --no-mlflow for a local run")
    mlflow.set_tracking_uri(uri)
    return uri


@contextmanager
def experiment_run(kind, config, disease, device, *, enabled=True):
    """Keep run ownership here so exceptions mark the run failed and always close it."""
    if not enabled:
        yield None
        return
    configure_mlflow()
    experiment = mlflow.set_experiment(EXPERIMENT_NAMES[kind])
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"{disease}-{config.model_type}"
    ) as run:
        mlflow.log_params({**asdict(config), "disease": disease, "device": str(device)})
        mlflow.set_tag("stage", "preparing")
        log_code_provenance()
        logger.info(f"MLflow experiment: {EXPERIMENT_NAMES[kind]}; run: {run.info.run_id}")
        yield run
        mlflow.set_tag("stage", "complete")


def log_metrics(metrics, *, prefix="", step=None):
    """Log finite scalar metrics only when an experiment has explicitly opened a run."""
    if mlflow.active_run() is None:
        return
    values = {
        re.sub(r"[^\w./ -]", "_", f"{prefix}/{name}" if prefix else name): float(value)
        for name, value in metrics.items()
        if value is not None and math.isfinite(value)
    }
    if values:
        mlflow.log_metrics(values, step=step)
