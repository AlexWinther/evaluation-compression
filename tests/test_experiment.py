from types import SimpleNamespace

import torch
from torch import nn

from active_testing_benchmark import experiment
from active_testing_benchmark.active_testing import (
    ActiveSurrogateEstimator,
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
    StratifiedSampleMean,
    UniformSampleMean,
)
from active_testing_benchmark.active_testing.interface import Estimate


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


def test_execute_active_evaluation_runs_all_current_methods(monkeypatch) -> None:
    active_methods = []
    active_calls = []

    def run_active(method, **kwargs):
        active_methods.append(method)
        active_calls.append(kwargs)
        estimate = Estimate(
            risk=0.25,
            n_labels=1,
            subgroup_risks={"female": 0.2, "male": 0.3},
            subgroup_n_labels={"female": 1, "male": 0},
        )
        return [{"estimate": estimate}]

    monkeypatch.setattr(experiment, "run_active_evaluation", run_active)

    dataset = FakeDataset()
    model = FakeClassifier(num_classes=2)
    surrogate_1 = FakeClassifier(num_classes=2)
    surrogate_2 = FakeClassifier(num_classes=2)
    proxy_model = FakeClassifier(num_classes=2)
    method_configs = [
        (UniformSampleMean, {"batch_size": 10}),
        (StratifiedSampleMean, {"allocation": "balanced", "batch_size": 10}),
        (StratifiedSampleMean, {"allocation": "proportional", "batch_size": 10}),
        (
            KossenActiveTesting,
            {"surrogate": surrogate_1, "num_classes": 2, "batch_size": 10},
        ),
        (
            ActiveSurrogateEstimator,
            {
                "surrogate": [surrogate_1, surrogate_2],
                "num_classes": 2,
                "acquisition": "xwed",
                "loss_type": "cross_entropy",
                "surrogate_output": "logits",
                "batch_size": 10,
            },
        ),
        (
            PredictionPoweredActiveTesting,
            {
                "num_classes": 2,
                "proxy": proxy_model,
                "surrogate": [surrogate_1, surrogate_2],
                "lambda_value": 0.5,
                "exploration": 0.05,
                "loss_type": "cross_entropy",
                "batch_size": 10,
            },
        ),
    ]
    histories = {}
    repetition_index = 7

    result = experiment.execute_active_evaluation(
        config=SimpleNamespace(seed=42),
        device=torch.device("cpu"),
        test_dataset=dataset,
        model=model,
        method_configs=method_configs,
        histories=histories,
        repitition=repetition_index,
    )

    expected_types = [
        UniformSampleMean,
        StratifiedSampleMean,
        StratifiedSampleMean,
        KossenActiveTesting,
        ActiveSurrogateEstimator,
        PredictionPoweredActiveTesting,
    ]
    expected_names = [
        "UniformSampleMean",
        "StratifiedSampleMeanBalanced",
        "StratifiedSampleMeanProportional",
        "KossenActiveTesting",
        "ActiveSurrogateEstimator",
        "PredictionPoweredActiveTesting",
    ]

    assert [type(method) for method in active_methods] == expected_types
    assert active_methods[1].allocation == "balanced"
    assert active_methods[2].allocation == "proportional"
    assert [method.num_classes for method in active_methods[3:]] == [2, 2, 2]
    assert len(active_calls) == len(expected_types)
    assert all(call["model"] is model for call in active_calls)
    assert all(call["dataset"] is dataset for call in active_calls)
    assert all(call["device"] == torch.device("cpu") for call in active_calls)
    assert all(call["protected_column"] == ["gender"] for call in active_calls)
    assert all(call["budget"] == 990 for call in active_calls)
    assert all(isinstance(call["loss_fn"], nn.CrossEntropyLoss) for call in active_calls)
    assert all(call["loss_fn"].reduction == "sum" for call in active_calls)

    assert len(result) == len(expected_types)
    assert [frame["method"].iloc[0] for frame in result] == expected_names
    assert set(histories) == set(expected_names)
    for frame in result:
        assert frame["repetition"].tolist() == [repetition_index + 1]
        assert frame["seed"].tolist() == [42 + repetition_index]
        assert frame["risk"].tolist() == [0.25]
        assert frame["risk female"].tolist() == [0.2]
        assert frame["risk male"].tolist() == [0.3]
