from mlflow import MlflowClient
import pytest

from active_testing_benchmark.modeling import registry


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """Keep tracking tests isolated from the configured remote server and its artifacts."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    for name in (
        "MLFLOW_REGISTRY_URI",
        "MLFLOW_EXPERIMENT_ID",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    assert registry.configure_mlflow() == tracking_uri
    client = MlflowClient()
    for name in registry.EXPERIMENT_NAMES.values():
        artifact_root = tmp_path / "artifacts" / name.replace("/", "_")
        artifact_root.mkdir(parents=True)
        client.create_experiment(name, artifact_location=str(artifact_root))
    return client
