from loguru import logger
import mlflow.pytorch
import typer

from active_testing_benchmark.active_testing import normalize_model_source
from active_testing_benchmark.modeling.registry import configure_mlflow

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option(
        "fairvision-dr-cnn", help="Registered model name, optionally with an alias."
    ),
):
    """Load a model from the configured MLflow server for use by inference code."""
    configure_mlflow()
    model_uri = normalize_model_source(model)
    logger.info(f"Loading model from {model_uri}")
    mlflow.pytorch.load_model(model_uri, map_location="cpu")
    logger.success("Model loaded from MLflow.")


if __name__ == "__main__":
    app()
