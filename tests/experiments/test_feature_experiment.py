from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from active_testing_benchmark.experiments import feature_experiment
from active_testing_benchmark.modeling.models import create_model


class SyntheticData(Dataset):
    def __init__(self, **kwargs):
        self.labels = np.array([0, 1] * 10)
        self.classes = ["zero", "one"]
        self.images = torch.randn(20, 3, 32, 32, generator=torch.Generator().manual_seed(1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": self.labels[index]}


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_comparison_logs_matching_splits_and_metrics(monkeypatch, tmp_path, seed):
    generated_seeds = []

    def generate_seed(bits):
        generated_seeds.append(bits)
        return 123

    monkeypatch.setattr(feature_experiment.secrets, "randbits", generate_seed)
    monkeypatch.setattr(feature_experiment, "FairVisionData", SyntheticData)
    monkeypatch.setattr(
        feature_experiment,
        "FrozenImageNetFeatures",
        lambda: torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()),
    )
    partitions = []
    monkeypatch.setattr(
        feature_experiment,
        "log_dataset_provenance",
        lambda **kwargs: partitions.append(tuple(tuple(d.indices) for d in kwargs.values())),
    )
    monkeypatch.setattr(
        "active_testing_benchmark.modeling.registry.log_code_provenance", lambda: None
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    results = feature_experiment.run_feature_experiment(
        seed=seed, n_estimators=3, batch_size=4, device="cpu"
    )
    names = {"pooled_pixels", "spatial_texture", "imagenet_resnet18"}
    assert set(results["extractor"]) == names
    assert results["train_size"].eq(16).all()
    assert results["validation_size"].eq(4).all()
    assert partitions[0] == partitions[1] == partitions[2]
    assert not set(partitions[0][0]) & set(partitions[0][1])
    experiment = mlflow.get_experiment_by_name("FairVision/feature_extractors")
    runs = mlflow.search_runs([experiment.experiment_id])
    parents = runs[runs["tags.mlflow.parentRunId"].isna()]
    assert len(parents) == 1
    parent = parents.iloc[0]
    assert parent["status"] == "FINISHED"
    assert parent["params.random_state"] == str(123 if seed is None else seed)
    assert "comparison.csv" in {
        artifact.path for artifact in mlflow.MlflowClient().list_artifacts(parent["run_id"])
    }
    runs = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert runs["tags.mlflow.parentRunId"].eq(parent["run_id"]).all()
    assert set(results["mlflow_run_id"]) == set(runs["run_id"])
    assert set(results["mlflow_parent_run_id"]) == {parent["run_id"]}
    assert len(runs) == 3
    assert set(runs["params.model_type"]) == names
    backbone_run = runs[runs["params.model_type"] == "imagenet_resnet18"].iloc[0]
    assert backbone_run["params.pretrained_weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    assert generated_seeds == ([32] if seed is None else [])
    assert runs["params.random_state"].eq(str(123 if seed is None else seed)).all()
    assert runs["metrics.evaluation/accuracy"].between(0, 1).all()
    assert runs["metrics.evaluation/loss"].notna().all()
    assert mlflow.active_run() is None


def test_handcrafted_selection_does_not_initialize_backbone(monkeypatch):
    monkeypatch.setattr(feature_experiment, "FairVisionData", SyntheticData)
    monkeypatch.setattr(
        feature_experiment,
        "FrozenImageNetFeatures",
        lambda: pytest.fail("Unselected backbone must not be initialized"),
    )
    results = feature_experiment.run_feature_experiment(
        n_estimators=3, device="cpu", track=False, extractors=("spatial_texture",)
    )
    assert results["extractor"].tolist() == ["spatial_texture"]


@pytest.fixture
def target_checkpoint(tmp_path):
    """A small saved classifier checkpoint trained, nominally, with seed 5."""
    model = create_model("resnet", out_dim=2, use_pretrained=False)
    path = tmp_path / "resnet.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"model_type": "resnet", "image_size": 224, "random_state": 5},
            "classes": ["zero", "one"],
        },
        path,
    )
    return path


def test_target_checkpoint_adds_diagnostics_and_uses_its_seed(monkeypatch, target_checkpoint):
    monkeypatch.setattr(feature_experiment, "FairVisionData", SyntheticData)
    splits = []
    split_dataset = feature_experiment.split_dataset

    def record_split(*args, **kwargs):
        splits.append(kwargs["random_state"])
        return split_dataset(*args, **kwargs)

    monkeypatch.setattr(feature_experiment, "split_dataset", record_split)
    results = feature_experiment.run_feature_experiment(
        n_estimators=3,
        batch_size=4,
        device="cpu",
        track=False,
        extractors=("pooled_pixels", "spatial_texture"),
        target_checkpoint=target_checkpoint,
    )
    assert splits == [5]
    assert results["kossen_variance_gain"].gt(0).all()
    assert results["oracle_variance_gain"].ge(results["kossen_variance_gain"]).all()
    assert {
        "kossen_score_spearman",
        "kossen_error_auc",
        "ase_relative_bias",
        "ppat_proxy_correlation",
    } <= set(
        results.columns
    )


def test_target_checkpoint_rejects_a_different_seed(monkeypatch, target_checkpoint, capsys):
    monkeypatch.setattr(feature_experiment, "FairVisionData", SyntheticData)
    with pytest.raises(ValueError, match="does not match the seed 5"):
        feature_experiment.run_feature_experiment(
            seed=6, device="cpu", track=False, target_checkpoint=target_checkpoint
        )
    arguments = ["feature", "--seed", "6", "--target-checkpoint", str(target_checkpoint)]
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(
        feature_experiment, "run_feature_experiment", lambda **kwargs: pytest.fail("Ran")
    )
    with pytest.raises(SystemExit) as error:
        feature_experiment.main()
    assert error.value.code == 2
    assert "does not match" in capsys.readouterr().err


def test_main_saves_the_comparison_when_given_an_output_path(monkeypatch, tmp_path):
    results = pd.DataFrame({"extractor": ["pooled_pixels"], "auc": [0.6]})
    monkeypatch.setattr(feature_experiment, "run_feature_experiment", lambda **kwargs: results)
    output = tmp_path / "comparison" / "features.csv"
    monkeypatch.setattr("sys.argv", ["feature", "--no-mlflow", "--output", str(output)])
    feature_experiment.main()
    pd.testing.assert_frame_equal(pd.read_csv(output), results)
