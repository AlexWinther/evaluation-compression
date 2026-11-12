from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import Subset

from active_testing_benchmark.modeling import provenance, registry


@dataclass(frozen=True)
class TinyConfig:
    model_type: str = "tiny"
    max_epochs: int = 1


def _run(client, run_id):
    return client.get_run(run_id)


def test_tracking_requires_an_explicit_server(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(registry, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        registry.configure_mlflow()


def test_nested_failure_marks_both_runs_failed_and_closes_them(local_mlflow):
    import mlflow

    with (
        pytest.raises(RuntimeError, match="child failed"),
        registry.experiment_run(
            "model_selection", TinyConfig(), "dr", "cpu", role="parent"
        ) as parent,
        registry.experiment_run(
            "model_selection", TinyConfig(), "dr", "cpu", nested=True
        ) as child,
    ):
        raise RuntimeError("child failed")
    assert local_mlflow.get_run(parent.info.run_id).info.status == "FAILED"
    child_run = local_mlflow.get_run(child.info.run_id)
    assert child_run.info.status == "FAILED"
    assert child_run.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert mlflow.active_run() is None


def test_experiment_runs_are_separate_and_record_terminal_statuses(local_mlflow):
    client = local_mlflow
    config = TinyConfig()
    run_ids = {}

    for kind, expected_name in registry.EXPERIMENT_NAMES.items():
        with registry.experiment_run(kind, config, "amd", "cpu") as active:
            run_ids[kind] = active.info.run_id
            assert (
                active.info.experiment_id
                == client.get_experiment_by_name(expected_name).experiment_id
            )

    assert {
        client.get_experiment(_run(client, run_id).info.experiment_id).name
        for run_id in run_ids.values()
    } == set(registry.EXPERIMENT_NAMES.values())
    assert all(_run(client, run_id).info.status == "FINISHED" for run_id in run_ids.values())

    failed_run_id = None
    with (
        pytest.raises(RuntimeError, match="training failed"),
        registry.experiment_run("AT_final", config, "amd", "cpu") as active,
    ):
        failed_run_id = active.info.run_id
        raise RuntimeError("training failed")

    assert failed_run_id is not None
    assert _run(client, failed_run_id).info.status == "FAILED"


def test_disabled_tracking_and_outside_metric_calls_create_no_runs(local_mlflow):
    client = local_mlflow
    with registry.experiment_run(
        "model_selection", TinyConfig(), "amd", "cpu", enabled=False
    ) as active:
        assert active is None
        registry.log_metrics({"inside_disabled_run": 1.0})

    registry.log_metrics({"outside_run": 2.0})
    experiment_ids = [experiment.experiment_id for experiment in client.search_experiments()]
    assert client.search_runs(experiment_ids) == []


def test_metrics_are_filtered_sanitized_prefixed_and_step_scoped(local_mlflow):
    client = local_mlflow
    with registry.experiment_run("AT_dev", TinyConfig(), "amd", "cpu") as active:
        run_id = active.info.run_id
        registry.log_metrics(
            {
                "validation<loss": 0.25,
                "roc+auc": 0.8,
                "threshold=raw": 0.4,
                "missing": None,
                "nan_value": float("nan"),
                "positive_infinity": float("inf"),
                "negative_infinity": float("-inf"),
            },
            prefix="training",
            step=3,
        )

    histories = {
        name: client.get_metric_history(run_id, name)
        for name in ("training/validation_loss", "training/roc_auc", "training/threshold_raw")
    }
    assert set(histories) == {
        "training/validation_loss",
        "training/roc_auc",
        "training/threshold_raw",
    }
    assert all(len(history) == 1 and history[0].step == 3 for history in histories.values())
    assert client.get_run(run_id).data.metrics == {
        "training/validation_loss": 0.25,
        "training/roc_auc": 0.8,
        "training/threshold_raw": 0.4,
    }


class MetadataOnlyDataset:
    """A tiny metadata view that fails if provenance tries to fetch an item/image."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.sample_ids = ["sample-a", "sample-b", "sample-c"]
        self.labels = np.asarray([0, 1, 0])
        self.splits = {"training"}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        raise AssertionError("dataset provenance must not decode or fetch images")


def test_dataset_provenance_hashes_nested_subset_order_and_metadata_without_reading_images(
    local_mlflow, tmp_path
):
    client = local_mlflow
    root_dir = tmp_path / "dataset"
    root_dir.mkdir()
    metadata_path = root_dir / "metadata.csv"
    metadata_path.write_text("sample,label\nsample-a,0\nsample-b,1\nsample-c,0\n")
    dataset = MetadataOnlyDataset(root_dir)
    nested = Subset(Subset(dataset, [2, 0, 1]), [1, 0])
    reordered = Subset(Subset(dataset, [2, 0, 1]), [0, 1])

    with registry.experiment_run("AT_dev", TinyConfig(), "amd", "cpu") as active:
        run_id = active.info.run_id

        provenance.log_dataset_provenance(training=nested)
        first = json.loads(
            Path(
                client.download_artifacts(run_id, "datasets.json", str(tmp_path / "first"))
            ).read_text()
        )["training"]

        provenance.log_dataset_provenance(training=reordered)
        second = json.loads(
            Path(
                client.download_artifacts(run_id, "datasets.json", str(tmp_path / "second"))
            ).read_text()
        )["training"]

        metadata_path.write_text("sample,label\nsample-a,0\nsample-b,1\nsample-c,0\n# changed\n")
        provenance.log_dataset_provenance(training=nested)
        third = json.loads(
            Path(
                client.download_artifacts(run_id, "datasets.json", str(tmp_path / "third"))
            ).read_text()
        )["training"]

    assert first["size"] == 2
    assert first["splits"] == ["training"]
    assert first["sample_digest"] != second["sample_digest"]
    assert first["sample_digest"] == third["sample_digest"]
    assert (
        first["metadata_sha256"]
        == hashlib.sha256(b"sample,label\nsample-a,0\nsample-b,1\nsample-c,0\n").hexdigest()
    )
    assert first["metadata_sha256"] != third["metadata_sha256"]
