import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from typer.testing import CliRunner

from active_testing_benchmark import active_testing as benchmark


@pytest.mark.parametrize(
    "value,expected", [("1", 1), ("10", 10), ("0.21", 3), ("1.0", 10), ("1e-1", 1)]
)
def test_budgets(value, expected):
    assert benchmark.resolve_budget(value, 10) == expected


@pytest.mark.parametrize("value", ["0", "11", "-1", "1.1", "nan", "inf", "no", "0.0"])
def test_invalid_budgets(value):
    with pytest.raises(ValueError):
        benchmark.resolve_budget(value, 10)


def test_random_selection():
    candidates = np.arange(100) * 3
    selected = benchmark.random_reducer(candidates, 20, 42)
    np.testing.assert_array_equal(selected, benchmark.random_reducer(candidates, 20, 42))
    assert len(np.unique(selected)) == 20
    assert np.isin(selected, candidates).all()
    assert not np.array_equal(selected, benchmark.random_reducer(candidates, 20, 43))


def synthetic_predictions():
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    return benchmark.Predictions(
        np.array([2, 5, 8, 11]), np.array([0, 0, 1, 1]), probabilities.argmax(1), probabilities
    )


def test_metrics_and_errors():
    data = synthetic_predictions()
    values = benchmark.classification_metrics(data.labels, data.predictions, data.probabilities)
    assert values == pytest.approx(
        dict(
            accuracy=0.5,
            macro_f1=0.5,
            macro_precision=0.5,
            macro_recall=0.5,
            roc_auc=0.75,
            calibration_error=0.4,
        )
    )
    results, full, manifests = benchmark.evaluate_subsets({"a": data, "b": data}, [2, 4], 3, 4)
    assert len(results) == 2 * 2 * 3 * 6
    assert len(full) == 12
    assert len(manifests) == 6
    assert results[results.budget == 4].absolute_error.max() == 0
    np.testing.assert_allclose(results.error, results.estimate - results.full_value)
    for _, group in results.groupby(["subset_id", "metric"]):
        np.testing.assert_allclose(group.estimate.iloc[0], group.estimate.iloc[1])
    assert manifests["size-0-run-0"].row_id.isin(data.row_ids).all()
    one_class = benchmark.classification_metrics(
        np.array([0]), np.array([0]), np.array([[0.9, 0.1]])
    )
    assert np.isnan(one_class["roc_auc"])
    assert one_class["macro_f1"] == 0.5  # Fixed class universe even in tiny subsets.


def test_repetitions_are_independent_and_reproducible():
    probabilities = np.tile([0.7, 0.3], (100, 1))
    data = benchmark.Predictions(np.arange(100), np.zeros(100), np.zeros(100), probabilities)
    first = benchmark.evaluate_subsets({"a": data}, [10, 20], 3, 42)[2]
    second = benchmark.evaluate_subsets({"a": data}, [10, 20], 3, 42)[2]
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    assert len({int(frame.seed.iloc[0]) for frame in first.values()}) == 6
    assert not np.array_equal(first["size-0-run-0"].row_id, first["size-0-run-1"].row_id)
    assert not np.array_equal(first["size-0-run-0"].row_id, first["size-1-run-0"].row_id[:10])


def test_callable_contract_and_label_isolation():
    def iterative(candidate_ids, budget, seed, *, model_predictions=None, revealed_labels=None):
        available = np.array([row for row in candidate_ids if row not in (revealed_labels or {})])
        scores = dict(zip(candidate_ids, model_predictions["uncertainty"]))
        return np.array(sorted(available, key=scores.get, reverse=True)[:budget])

    reducer: benchmark.Reducer = iterative
    candidates = np.array([10, 20, 30])
    kwargs = dict(model_predictions={"uncertainty": np.array([0.2, 0.9, 0.5])})
    np.testing.assert_array_equal(reducer(candidates, 1, 7, **kwargs), [20])
    np.testing.assert_array_equal(
        reducer(candidates, 1, 8, revealed_labels={20: 1}, **kwargs), [30]
    )

    def spy(candidate_ids, budget, seed, *, model_predictions=None, revealed_labels=None):
        assert model_predictions is None
        assert revealed_labels is None
        return candidate_ids[:budget]

    benchmark.evaluate_subsets({"a": synthetic_predictions()}, [2], 1, 42, reducer=spy)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("classifier", "models:/classifier@baseline"),
        ("classifier@reviewed", "models:/classifier@reviewed"),
        ("models:/classifier@baseline", "models:/classifier@baseline"),
        ("run:abc", "runs:/abc/model"),
        ("runs:/abc/model", "runs:/abc/model"),
    ],
)
def test_model_sources(source, expected):
    assert benchmark.normalize_model_source(source) == expected


@pytest.mark.parametrize("source", ["", "run:", "runs:/abc", "name@", "name/path"])
def test_invalid_model_sources(source):
    with pytest.raises(ValueError):
        benchmark.normalize_model_source(source)


def test_cli_local_mlflow(tmp_path: Path, monkeypatch):
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking)
    previous_tracking, previous_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    try:
        benchmark.configure_mlflow()
        client = mlflow.MlflowClient()
        experiment_id = client.create_experiment(
            "fixture", artifact_location=str(tmp_path / "artifacts")
        )
        config = tmp_path / "cnn_config.json"
        config.write_text(json.dumps({"model": "cnn", "image_size": 32}))
        mapping = tmp_path / "cnn_classes.json"
        mapping.write_text(json.dumps({"healthy": 0, "disease": 1}))
        classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)
        )
        with mlflow.start_run(experiment_id=experiment_id) as training:
            mlflow.pytorch.log_model(
                classifier,
                name="model",
                extra_files=[str(config), str(mapping)],
                pip_requirements=["torch"],
                serialization_format="pickle",
            )
        version = mlflow.register_model(f"runs:/{training.info.run_id}/model", "tiny")
        client.set_registered_model_alias("tiny", "baseline", version.version)
        rows = []
        for index in range(7):
            path = tmp_path / f"image-{index}.jpg"
            Image.new("RGB", (16, 16), (index * 30, 20, 40)).save(path)
            rows.append(
                dict(
                    image=path.name,
                    label="disease" if index % 2 else "healthy",
                    split="training" if index == 0 else "test",
                )
            )
        metadata = tmp_path / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
        inference_calls = []
        original = benchmark.infer_model

        def infer(*args, **kwargs):
            inference_calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(benchmark, "infer_model", infer)
        client.create_experiment(
            "benchmark", artifact_location=str(tmp_path / "benchmark-artifacts")
        )
        result = CliRunner().invoke(
            benchmark.app,
            [
                "--model",
                "tiny",
                "--model",
                f"run:{training.info.run_id}",
                "--test-size",
                "0.5",
                "--test-size",
                "6",
                "--runs",
                "2",
                "--metadata-path",
                str(metadata),
                "--image-column",
                "image",
                "--target-column",
                "label",
                "--split-column",
                "split",
                "--output-dir",
                str(tmp_path / "output"),
                "--experiment-name",
                "benchmark",
            ],
        )
        assert result.exit_code == 0, f"{result.output}\n{result.exception}"
        assert len(inference_calls) == 2
        (destination,) = (tmp_path / "output").iterdir()
        results = pd.read_csv(destination / "results.csv")
        assert len(results) == 48
        assert results.loc[results.budget == 6, "absolute_error"].max() == pytest.approx(
            0, abs=1e-12
        )
        saved_config = json.loads((destination / "config.json").read_text())
        assert (
            saved_config["models"]["model-0"]["resolved_uri"] == f"models:/tiny/{version.version}"
        )
        for path in (destination / "subsets").iterdir():
            manifest = pd.read_csv(path)
            assert manifest.row_id.between(1, 6).all()
            assert manifest.row_id.is_unique
            assert list(manifest.selection_order) == list(range(len(manifest)))
        artifacts = {item.path for item in client.list_artifacts(destination.name)}
        assert artifacts == {"config.json", "results.csv", "full_metrics.csv", "subsets"}
        assert len(client.list_artifacts(destination.name, "subsets")) == 4
        assert client.get_run(destination.name).info.status == "FINISHED"
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)
