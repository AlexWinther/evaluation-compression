from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
from PIL import Image

from active_testing_benchmark.modeling.provenance import build_dataset_provenance
from active_testing_benchmark.modeling.registry import latest_model_version
from active_testing_benchmark.modeling.train import main as train


def test_dataset_provenance_is_stable_and_logs_split_digests(tmp_path: Path) -> None:
    metadata_path, image_root = _write_dataset(tmp_path)
    mapping = {"disease": 0, "healthy": 1}

    first = build_dataset_provenance(
        metadata_path, image_root, "image", "label", "split", mapping, "https://example.test/data"
    )
    second = build_dataset_provenance(
        metadata_path, image_root, "image", "label", "split", mapping, "https://example.test/data"
    )

    assert first.manifest_digest == second.manifest_digest
    assert first.metadata_digest == second.metadata_digest
    assert first.split_counts == {"training": 2, "validation": 2, "test": 2}
    assert set(first.datasets) == {"training", "validation", "test"}
    assert first.datasets["training"].digest == second.datasets["training"].digest
    assert first.datasets["training"].source.url == "https://example.test/data"


def test_dataset_provenance_changes_when_a_selected_label_changes(tmp_path: Path) -> None:
    metadata_path, image_root = _write_dataset(tmp_path)
    mapping = {"disease": 0, "healthy": 1}
    original = build_dataset_provenance(
        metadata_path, image_root, "image", "label", "split", mapping
    )

    dataframe = pd.read_csv(metadata_path)
    dataframe.loc[0, "label"] = "disease"
    dataframe.to_csv(metadata_path, index=False)
    changed = build_dataset_provenance(
        metadata_path, image_root, "image", "label", "split", mapping
    )

    assert changed.manifest_digest != original.manifest_digest
    assert changed.metadata_digest != original.metadata_digest


def test_training_logs_dataset_inputs_and_reloadable_model(tmp_path: Path, monkeypatch) -> None:
    metadata_path, image_root = _write_dataset(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "tiny-mlflow-training"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=str(tmp_path / "artifacts")
    )

    train(
        model="cnn",
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=7,
        num_workers=0,
        image_size=32,
        device_preference="cpu",
        metadata_path=metadata_path,
        image_root=image_root,
        image_column="image",
        target_column="label",
        disease_type="label",
        split_column="split",
        validate_images=True,
        experiment_name=experiment_name,
        pretrained=False,
        imbalance_strategy="none",
        dataset_source="https://example.test/fairvision",
    )

    run = mlflow.search_runs([experiment_id]).iloc[0]
    run_id = run["run_id"]
    logged_run = mlflow.get_run(run_id)
    assert {
        next(tag.value for tag in item.tags if tag.key == "mlflow.data.context")
        for item in logged_run.inputs.dataset_inputs
    } == {
        "training",
        "validation",
        "test",
    }
    assert logged_run.data.tags["dataset.manifest.sha256"]
    loaded_model = mlflow.pytorch.load_model("models:/fairvision-label-cnn/latest")
    assert loaded_model is not None
    selected = latest_model_version("fairvision-label-cnn")
    assert selected.run_id == run_id
    assert selected.tags["disease.type"] == "label"
    assert selected.tags["model.type"] == "cnn"


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for split in ("training", "validation", "test"):
        for index, label in enumerate(("healthy", "disease")):
            relative_path = Path(split) / f"image_{index}.jpg"
            path = tmp_path / relative_path
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)
            rows.append({"image": str(relative_path), "label": label, "split": split})
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path, tmp_path
