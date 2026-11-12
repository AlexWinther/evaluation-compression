from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace
from zipfile import ZipFile

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from active_testing_benchmark import benchmark
from active_testing_benchmark.active_testing.surrogate import (
    RandomForestModule,
    update_random_forest_surrogates,
)
from active_testing_benchmark.experiments import at_benchmarking as experiment
from active_testing_benchmark.experiments import model_selection, utils
from active_testing_benchmark.modeling.registry import EXPERIMENT_NAMES
from active_testing_benchmark.modeling.train import TrainingConfig


class TinyData(Dataset):
    disease = "dr"

    def __init__(self, root_dir, split, count):
        self.classes = ["no", "yes"]
        self.root_dir = root_dir
        self.splits = {split}
        self.labels = np.arange(count) % 2
        self.sample_ids = [f"{split}-{index}" for index in range(count)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = int(self.labels[index])
        return {
            "image": nn.functional.one_hot(torch.tensor(label), 2).float(),
            "label": label,
            "subgroups": {"age_group": "<50" if index % 4 < 2 else "80+"},
        }


@pytest.fixture
def tiny_pipeline(tmp_path, monkeypatch):
    root = tmp_path / "fairvision" / "dr"
    root.mkdir(parents=True)
    (root / "metadata.csv").write_text("synthetic metadata\n")
    data = {
        "train": TinyData(root, "training", 40),
        "validation": TinyData(root, "validation", 16),
        "test": TinyData(root, "test", 16),
    }
    monkeypatch.setattr(experiment, "FairVisionData", lambda **kwargs: data[kwargs["split"]])
    monkeypatch.setattr(
        model_selection,
        "prepare_data",
        lambda **kwargs: SimpleNamespace(
            train=data["train"], test=data["validation"], classes=data["train"].classes
        ),
    )
    model_calls = []

    def create(model_type, out_dim, use_pretrained):
        model_calls.append((model_type, use_pretrained))
        return nn.Linear(2, out_dim)

    monkeypatch.setattr(utils, "create_model", create)
    monkeypatch.setattr(
        experiment,
        "fit_auxiliary_models",
        lambda *args: SimpleNamespace(
            surrogates=(nn.Identity(), nn.Linear(2, 2)), proxy=nn.Identity(), seeds=(11, 12, 13)
        ),
    )
    return {
        "training_config": TrainingConfig(max_epochs=2, batch_size=8, random_state=42),
        "device": "cpu",
        "output_dir": tmp_path / "results",
        "checkpoint_dir": tmp_path / "checkpoints",
    }, model_calls


def runs_for(client, kind):
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAMES[kind]).experiment_id
    return [
        run
        for run in client.search_runs([experiment_id])
        if run.data.tags.get("run_role") != "parent"
    ]


@pytest.mark.parametrize("kind", ["model_selection", "AT_dev", "AT_final"])
def test_complete_experiments_log_live_metrics_checkpoints_and_provenance(
    local_mlflow, tiny_pipeline, monkeypatch, tmp_path, kind
):
    arguments, _ = tiny_pipeline
    if kind == "model_selection":
        model_selection.run_model_selection_experiment(**arguments)
    elif kind == "AT_dev":
        experiment.run_dev_experiment(**arguments, repetitions=2)
    else:
        monkeypatch.setattr("builtins.input", lambda prompt: "FINAL")
        experiment.run_final_experiment(**arguments, repetitions=2)
    runs = runs_for(local_mlflow, kind)
    if kind == "model_selection":
        parents = local_mlflow.search_runs(
            [runs[0].info.experiment_id], filter_string="tags.run_role = 'parent'"
        )
        assert len(parents) == 1
        assert parents[0].info.status == "FINISHED"
        assert parents[0].data.params["random_state"] == "42"
        assert "comparison.csv" in {
            artifact.path for artifact in local_mlflow.list_artifacts(parents[0].info.run_id)
        }
        assert all(run.data.tags["mlflow.parentRunId"] == parents[0].info.run_id for run in runs)
    assert len(runs) == (2 if kind == "model_selection" else 1)
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.data.tags["stage"] == "complete"
        assert run.data.params["random_state"] == "42"
        assert run.data.tags["training.source_run_id"] == run.info.run_id
        history = local_mlflow.get_metric_history(run.info.run_id, "training/train/loss")
        assert [metric.step for metric in history] == [1, 2]
        assert "evaluation/auc" in run.data.metrics
        assert "evaluation/age_group/_50/auc" in run.data.metrics
        checkpoint_path = local_mlflow.download_artifacts(
            run.info.run_id, f"checkpoints/{run.data.params['model_type']}.pt", str(tmp_path)
        )
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert checkpoint["training_run_id"] == run.info.run_id
        assert checkpoint["config"]["random_state"] == 42
        assert len(checkpoint["history"]) == 2
        provenance_path = local_mlflow.download_artifacts(
            run.info.run_id, "provenance/source.zip", str(tmp_path)
        )
        with ZipFile(provenance_path) as archive:
            assert "active_testing_benchmark/experiments/at_benchmarking.py" in archive.namelist()
            assert all(name.endswith(".py") for name in archive.namelist())
        assert "datasets.json" in {
            item.path for item in local_mlflow.list_artifacts(run.info.run_id)
        }
        if kind != "model_selection":
            progress = local_mlflow.get_metric_history(
                run.info.run_id, "AT/age_group/UniformSampleMean/unweighted_cross_entropy/risk"
            )
            assert [metric.step for metric in progress] == [1, 2]
            assert len(local_mlflow.list_artifacts(run.info.run_id, "results")) == 2
            assert run.data.params["repetitions"] == "2"
            metadata = json.loads(next(arguments["output_dir"].glob("*.json")).read_text())
            assert metadata["mlflow_run_id"] == run.info.run_id
    assert mlflow.active_run() is None


def test_reuse_keeps_original_training_lineage_and_skips_training(
    local_mlflow, tiny_pipeline, monkeypatch
):
    arguments, model_calls = tiny_pipeline
    model_selection.run_model_selection_experiment(**arguments)
    original_ids = {run.info.run_id for run in runs_for(local_mlflow, "model_selection")}
    model_calls.clear()
    monkeypatch.setattr(
        utils, "train_model", lambda *args, **kwargs: pytest.fail("Retrained reused model")
    )
    arguments["training_config"] = replace(arguments["training_config"], random_state=7)
    model_selection.run_model_selection_experiment(**arguments, reuse_trained_model=True)
    assert model_calls == [("resnet", False), ("efficientnet", False)]
    reused = [
        run
        for run in runs_for(local_mlflow, "model_selection")
        if run.info.run_id not in original_ids
    ]
    assert len(reused) == 2
    reused_parent_ids = {run.data.tags["mlflow.parentRunId"] for run in reused}
    original_parent_ids = {
        local_mlflow.get_run(run_id).data.tags["mlflow.parentRunId"] for run_id in original_ids
    }
    assert len(reused_parent_ids) == len(original_parent_ids) == 1
    assert reused_parent_ids.isdisjoint(original_parent_ids)
    for run in reused:
        assert run.data.params["random_state"] == "7"
        assert run.data.params["training.random_state"] == "42"
        assert run.data.params["reused_checkpoint"] == "True"
        assert run.data.tags["training.source_run_id"] in original_ids
        assert "training/train/loss" not in run.data.metrics
        assert "evaluation/auc" in run.data.metrics


def test_local_only_pipeline_creates_outputs_without_mlflow_runs(local_mlflow, tiny_pipeline):
    arguments, _ = tiny_pipeline
    model_selection.run_model_selection_experiment(**arguments, track=False)
    experiment.run_dev_experiment(**arguments, repetitions=1, track=False)
    assert (arguments["output_dir"] / "comparison.csv").exists()
    assert list(arguments["checkpoint_dir"].rglob("*.pt"))
    assert not local_mlflow.search_runs(
        [item.experiment_id for item in local_mlflow.search_experiments()]
    )


def test_final_cancellation_never_creates_run_or_reads_dataset(local_mlflow, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    monkeypatch.setattr(
        experiment, "FairVisionData", lambda **kwargs: pytest.fail("Read final dataset")
    )
    assert "CANCELED" in experiment.run_final_experiment()
    assert not runs_for(local_mlflow, "AT_final")


def test_pipeline_exception_marks_run_failed(local_mlflow, tiny_pipeline, monkeypatch):
    arguments, _ = tiny_pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(utils, "train_model", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        experiment.run_dev_experiment(**arguments, repetitions=1)
    (run,) = runs_for(local_mlflow, "AT_dev")
    assert run.info.status == "FAILED"
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    "module,arguments,runner_name",
    [
        (experiment, [], "run_dev_experiment"),
        (model_selection, [], "run_model_selection_experiment"),
        (experiment, ["--FINAL"], "run_final_experiment"),
    ],
)
def test_cli_forwards_local_only_flag(monkeypatch, module, arguments, runner_name):
    calls = []
    monkeypatch.setattr("sys.argv", ["experiment", "--no-mlflow", "--seed", "0", *arguments])

    def run(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(module, runner_name, run)
    module.main()
    assert calls[0]["track"] is False
    assert calls[0]["training_config"].random_state == 0


@pytest.mark.parametrize("final", [False, True])
def test_pipeline_refits_ase_and_records_custom_frequency(
    local_mlflow, tiny_pipeline, monkeypatch, final
):
    arguments, _ = tiny_pipeline

    def fit_auxiliary(dataset, classes, config, device):
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        models = tuple(
            RandomForestModule(2, nn.Identity(), n_estimators=5, random_state=seed).fit(loader)
            for seed in (11, 12, 13)
        )
        return SimpleNamespace(surrogates=models[:2], proxy=models[2], seeds=(11, 12, 13))

    updates = []

    def update(models, X, y):
        updates.append(len(y))
        update_random_forest_surrogates(models, X, y)

    monkeypatch.setattr(experiment, "fit_auxiliary_models", fit_auxiliary)
    monkeypatch.setattr(benchmark, "update_random_forest_surrogates", update)
    monkeypatch.setattr("builtins.input", lambda prompt: "FINAL")
    runner = experiment.run_final_experiment if final else experiment.run_dev_experiment
    runner(**arguments, repetitions=1, ase_update_every=7)
    assert updates == [7, 14]
    (run,) = runs_for(local_mlflow, "AT_final" if final else "AT_dev")
    assert run.data.params["ase_update_every"] == "7"
    metadata = json.loads(next(arguments["output_dir"].glob("*.json")).read_text())
    assert metadata["configuration"]["ase_update_every"] == 7
    curves = pd.read_csv(next(arguments["output_dir"].glob("*.csv")))
    ase = curves[curves["method"] == "ActiveSurrogateEstimator"]
    assert ase["n_labels"].tolist() == [7, 14, 16]


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("protected_attribute", [None, "race"])
def test_cli_forwards_protected_attribute(monkeypatch, final, protected_attribute):
    arguments = ["experiment", "--no-mlflow"]
    runner_name = "run_final_experiment" if final else "run_dev_experiment"
    if final:
        arguments.append("--FINAL")
    if protected_attribute is not None:
        arguments.extend(["--protected-attribute", protected_attribute])
    monkeypatch.setattr("sys.argv", arguments)
    calls = []
    monkeypatch.setattr(experiment, runner_name, lambda **kwargs: calls.append(kwargs) or {})

    experiment.main()

    assert calls[0]["protected_attribute"] == protected_attribute


def test_cli_rejects_raw_age_as_protected_attribute(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["experiment", "--protected-attribute", "age"])

    with pytest.raises(SystemExit) as error:
        experiment.main()

    assert error.value.code == 2
    assert "invalid choice: 'age'" in capsys.readouterr().err


class TinyDataset:
    disease = "dr"
    classes = ("no", "yes")

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return 4


@pytest.mark.parametrize(
    "protected_attribute,expected_columns,should_reject",
    [(None, ("gender", "race"), False), ("race", ("gender", "race"), False), ("age", None, True)],
)
def test_run_experiment_selects_columns_for_benchmark_and_metadata(
    monkeypatch, tmp_path, protected_attribute, expected_columns, should_reject
):
    dataset = TinyDataset(tmp_path / "fairvision" / "dr")
    checkpoint = {
        "config": {"random_state": 4},
        "history": [{"epoch": 1, "roc_auc": 0.75}],
        "training_seconds": 1.0,
    }
    metrics = {
        "subgroup_metrics": {
            "age": {"20": {"n": 2}},
            "gender": {"female": {"n": 2}},
            "race": {"unknown": {"n": 4}},
        }
    }
    benchmark_arguments = {}
    monkeypatch.setattr(experiment, "experiment_run", lambda *args, **kwargs: nullcontext(None))
    monkeypatch.setattr(experiment, "_train_or_load", lambda *args: (object(), checkpoint))
    monkeypatch.setattr(experiment, "evaluate_model", lambda *args: metrics)
    monkeypatch.setattr(
        experiment,
        "fit_auxiliary_models",
        lambda *args: SimpleNamespace(seeds=(1, 2, 3)),
    )

    def run_benchmark(**kwargs):
        benchmark_arguments.update(kwargs)
        return pd.DataFrame({"protected_attribute": kwargs["protected_columns"]})

    monkeypatch.setattr(experiment, "run_benchmark", run_benchmark)

    def run():
        return experiment._run_experiment(
            dataset,
            dataset,
            dataset,
            TrainingConfig(random_state=4),
            "cpu",
            tmp_path / "checkpoint.pt",
            output_dir=tmp_path / "results",
            track=False,
            protected_attribute=protected_attribute,
        )

    if should_reject:
        with pytest.raises(ValueError, match="not available"):
            run()
        assert not benchmark_arguments
        return

    run()

    assert benchmark_arguments["protected_columns"] == expected_columns
    assert benchmark_arguments["stratification_attribute"] == protected_attribute
    metadata_path = next((tmp_path / "results").glob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["configuration"]["protected_columns"] == list(expected_columns)
    assert metadata["configuration"]["stratification_attribute"] == protected_attribute


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("frequency", [None, 7])
def test_cli_forwards_ase_update_frequency(monkeypatch, final, frequency):
    args = ["experiment", "--no-mlflow"]
    if final:
        args += ["--FINAL"]
    if frequency is not None:
        args += ["--ase-update-every", str(frequency)]
    monkeypatch.setattr("sys.argv", args)
    calls = []
    runner = "run_final_experiment" if final else "run_dev_experiment"
    monkeypatch.setattr(experiment, runner, lambda **kwargs: calls.append(kwargs) or {})
    experiment.main()
    assert calls[0]["ase_update_every"] == (100 if frequency is None else frequency)


@pytest.mark.parametrize("frequency", [0, -1])
def test_cli_rejects_nonpositive_ase_update_frequency(monkeypatch, capsys, frequency):
    monkeypatch.setattr("sys.argv", ["experiment", "--ase-update-every", str(frequency)])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2
    assert "--ase-update-every must be positive" in capsys.readouterr().err
