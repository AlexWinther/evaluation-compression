import torch
from torch import nn

from active_testing_benchmark.active_testing import (
    ActiveSurrogateEstimator,
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
    StratifiedSampleMean,
    UniformSampleMean,
)
from active_testing_benchmark.active_testing.interface import Estimate
from active_testing_benchmark import experiment


class FakeDataset:
    classes = ["class 0", "class 1"]

    def __len__(self) -> int:
        return 2


class FakeClassifier(nn.Module):
    def __init__(self, num_classes: int, random_state: int = 0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.random_state = random_state
        self.linear = nn.Linear(1, num_classes)
        self.fit_called = False

    def fit(self, loader) -> "FakeClassifier":
        self.fit_called = True
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.reshape(len(inputs), -1)[:, :1])


def test_experiment_builds_and_runs_all_current_methods(monkeypatch) -> None:
    datasets = []
    active_methods = []
    plot_targets = []
    model_class_counts = []

    def make_dataset(*, disease, split, transform):
        datasets.append((disease, split, transform))
        return FakeDataset()

    def make_model(*, num_classes, pretrained, device):
        model_class_counts.append(num_classes)
        return FakeClassifier(num_classes)

    def run_active(method, **kwargs):
        active_methods.append(method)
        estimate = Estimate(
            risk=0.25,
            n_labels=1,
            subgroup_risks={"female": 0.2, "male": 0.3},
            subgroup_n_labels={"female": 1, "male": 0},
        )
        return [{"estimate": estimate}]

    monkeypatch.setattr(experiment, "configure_mlflow", lambda: None)
    monkeypatch.setattr(experiment, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(experiment, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "FairVisionDataset", make_dataset)
    monkeypatch.setattr(experiment, "create_image_transform", lambda **kwargs: kwargs)
    monkeypatch.setattr(experiment, "create_model", make_model)
    monkeypatch.setattr(
        experiment,
        "make_train_loader_and_loss",
        lambda **kwargs: (object(), nn.CrossEntropyLoss(), None, None),
    )
    monkeypatch.setattr(experiment, "DataLoader", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(experiment, "train_one_epoch", lambda **kwargs: 0.5)
    monkeypatch.setattr(experiment, "evaluate", lambda **kwargs: {"predictions": []})
    monkeypatch.setattr(experiment, "RandomForestModule", FakeClassifier)
    monkeypatch.setattr(experiment, "run_active_evaluation", run_active)
    monkeypatch.setattr(
        experiment.sns,
        "lineplot",
        lambda *, data, x, y, hue, ax: plot_targets.append((y, set(data["method"]))),
    )
    monkeypatch.setattr(experiment.plt, "show", lambda: None)

    result = experiment.run_experiment(dev_splits="train", eval_splits="test", disease="dr")

    expected_types = [
        UniformSampleMean,
        StratifiedSampleMean,
        StratifiedSampleMean,
        KossenActiveTesting,
        ActiveSurrogateEstimator,
        PredictionPoweredActiveTesting,
    ]
    assert result == "Experiment run successfully"
    assert len(datasets) == 2
    assert [split for _, split, _ in datasets] == ["train", "test"]
    assert model_class_counts == [2]
    assert len(active_methods) == 2 * len(expected_types)

    for repetition in range(2):
        start = repetition * len(expected_types)
        methods = active_methods[start : start + len(expected_types)]
        assert [type(method) for method in methods] == expected_types
        assert methods[1].allocation == "balanced"
        assert methods[2].allocation == "proportional"
        assert [method.num_classes for method in methods[3:]] == [2, 2, 2]

    assert {target for target, _ in plot_targets} == {"risk", "risk female", "risk male"}
    plotted_methods = set.union(*(methods for _, methods in plot_targets))
    assert "StratifiedSampleMeanBalanced" in plotted_methods
    assert "StratifiedSampleMeanProportional" in plotted_methods
