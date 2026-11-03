import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from active_testing_benchmark import benchmark, experiment
from active_testing_benchmark.active_testing import (
    KossenActiveTesting,
    UniformSampleMean,
    run_active_evaluation,
)
from active_testing_benchmark.active_testing.interface import Estimate, Problem
from active_testing_benchmark.active_testing.sota import _compute_hypothetical_class_losses
from active_testing_benchmark.modeling.data import inverse_frequency_weights
from active_testing_benchmark.modeling.train import make_class_balanced_weights


class TinyPool(Dataset):
    def __init__(self) -> None:
        self.logits = torch.tensor([[1.2, -0.4], [-0.2, 0.9]], dtype=torch.float32)
        self.targets = [0, 1]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.logits[index],
            "target": self.targets[index],
            "id": f"sample-{index}",
            "protected": {"gender": "all"},
        }


def test_class_balanced_weights_equalize_class_contributions() -> None:
    counts = torch.tensor([90, 10])

    weights = make_class_balanced_weights(counts)

    assert weights.tolist() == pytest.approx([5 / 9, 5.0])
    torch.testing.assert_close(
        weights,
        inverse_frequency_weights(counts, power=1.0, normalization="sample_mean"),
    )
    assert torch.dot(counts.float(), weights).item() / counts.sum().item() == pytest.approx(1.0)
    assert (counts[0] * weights[0]).item() == pytest.approx(
        (counts[1] * weights[1]).item()
    )


def test_class_balanced_weights_require_examples_from_every_class() -> None:
    with pytest.raises(ValueError, match="at least one training example"):
        make_class_balanced_weights(torch.tensor([10, 0]))


def test_execute_active_evaluation_adds_class_balanced_objective(monkeypatch) -> None:
    calls = []

    def run_active(method, **kwargs):
        calls.append(kwargs)
        return [{"estimate": Estimate(risk=0.5, n_labels=1)}]

    monkeypatch.setattr(benchmark, "run_active_evaluation", run_active)
    balanced_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([5 / 9, 5.0]),
        reduction="sum",
    )

    results = benchmark.run_repetition(
        seed=17,
        device=torch.device("cpu"),
        dataset=TinyPool(),
        model=nn.Identity(),
        method_configs=[(UniformSampleMean, {"batch_size": 1})],
        repetition=0,
        class_weights=balanced_loss.weight,
    )

    assert [call["loss_fn"].reduction for call in calls] == ["sum", "sum"]
    assert calls[0]["loss_fn"].weight is None
    torch.testing.assert_close(calls[1]["loss_fn"].weight, balanced_loss.weight)
    assert [frame["risk_objective"].iloc[0] for frame in results] == [
        "unweighted_cross_entropy",
        "class_balanced_cross_entropy",
    ]


@pytest.mark.parametrize(
    ("arguments", "enabled"),
    [([], True), (["--no-class-balanced-at"], False)],
)
def test_cli_controls_class_balanced_at(monkeypatch, arguments, enabled: bool) -> None:
    received = {}
    monkeypatch.setattr(sys, "argv", ["experiment.py", *arguments])
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda config: received.update(class_balanced_at=config.class_balanced_at),
    )

    experiment.main()

    assert received["class_balanced_at"] is enabled


def test_active_runner_estimates_mean_of_weighted_per_example_losses() -> None:
    dataset = TinyPool()
    weights = make_class_balanced_weights(torch.tensor([90, 10]))
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="sum")

    history = run_active_evaluation(
        method=UniformSampleMean(seed=2, batch_size=len(dataset)),
        model=nn.Identity(),
        dataset=dataset,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
        protected_column="gender",
        budget=len(dataset),
    )

    expected_losses = nn.functional.cross_entropy(
        dataset.logits,
        torch.tensor(dataset.targets),
        weight=weights,
        reduction="none",
    )
    assert history[-1]["estimate"].risk == pytest.approx(expected_losses.mean().item())


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.2])
def test_kossen_acquisition_matches_runner_weighted_loss(
    reduction: str, label_smoothing: float
) -> None:
    logits = torch.tensor([[1.2, -0.4], [-0.2, 0.9]], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([0.5, 3.0]),
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    problem = Problem(
        N=len(logits),
        budget=1,
        X_pool=logits,
        protected_attribute=["group-a", "group-b"],
        model=nn.Identity(),
        loss_fn=loss_fn,
    )
    method = KossenActiveTesting(num_classes=2)
    state = method.initialize(problem)

    actual_scores = method._acquisition_scores(state)
    class_losses = _compute_hypothetical_class_losses(
        logits.numpy(), loss_fn, loss_type="cross_entropy"
    )
    model_probabilities = torch.softmax(logits, dim=1).numpy()
    expected_scores = np.sum(model_probabilities * class_losses, axis=1)

    np.testing.assert_allclose(actual_scores, expected_scores)
