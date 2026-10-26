import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from active_testing_benchmark.active_testing import (
    ActiveSurrogateEstimator,
    KossenActiveTesting,
    PredictionPoweredActiveTesting,
    StratifiedSampleMean,
    UniformSampleMean,
    run_active_evaluation,
)
from active_testing_benchmark.active_testing.interface import Problem
from active_testing_benchmark.active_testing.sota import _predict_surrogate_members


class TinyEvaluationDataset(Dataset):
    def __init__(self) -> None:
        self.features = torch.tensor(
            [
                [2.0, -1.0],
                [-1.0, 2.0],
                [1.5, -0.5],
                [-0.5, 1.5],
                [1.0, 0.2],
                [0.1, 1.0],
            ]
        )
        self.targets = [0, 1, 0, 1, 0, 1]
        self.groups = ["female", "male", "female", "male", "female", "male"]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.features[index],
            "target": self.targets[index],
            "id": f"image-{index}",
            "protected": {"gender": self.groups[index]},
        }


class ShiftedClassifier(nn.Module):
    def __init__(self, reverse_classes: bool = False) -> None:
        super().__init__()
        self.reverse_classes = reverse_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.flip(1) if self.reverse_classes else inputs


def _problem(dataset: TinyEvaluationDataset, model: nn.Module, budget: int) -> Problem:
    return Problem(
        N=len(dataset),
        budget=budget,
        X_pool=dataset.features,
        protected_attribute=dataset.groups,
        model=model,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    )


def _methods() -> list:
    surrogate_members = [ShiftedClassifier(), ShiftedClassifier(reverse_classes=True)]
    return [
        UniformSampleMean(seed=7, batch_size=2),
        StratifiedSampleMean(allocation="balanced", seed=7, batch_size=2),
        StratifiedSampleMean(allocation="proportional", seed=7, batch_size=2),
        KossenActiveTesting(
            surrogate=ShiftedClassifier(reverse_classes=True),
            num_classes=2,
            seed=7,
            batch_size=2,
        ),
        ActiveSurrogateEstimator(
            surrogate=surrogate_members,
            num_classes=2,
            acquisition="xwed",
            seed=7,
            batch_size=2,
        ),
        PredictionPoweredActiveTesting(
            num_classes=2,
            proxy_labels=[0, 1, 0, 1, 0, 1],
            surrogate=surrogate_members,
            lambda_value=0.0,
            seed=7,
            batch_size=2,
        ),
    ]


@pytest.mark.parametrize(
    "method_index", range(6), ids=["uniform", "balanced", "proportional", "kossen", "ase", "ppat"]
)
def test_active_methods_run_through_the_shared_runner(method_index: int) -> None:
    dataset = TinyEvaluationDataset()
    method = _methods()[method_index]

    history = run_active_evaluation(
        method=method,
        model=nn.Identity(),
        dataset=dataset,
        device=torch.device("cpu"),
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        protected_column="gender",
        budget=4,
    )

    assert [record["total selections"] for record in history] == [2, 4]
    assert all(record["estimate"].n_labels == record["total selections"] for record in history)
    selected = [index for record in history for index in record["selection"].indices]
    assert len(selected) == len(set(selected)) == 4
    assert all(np.isfinite(record["estimate"].risk) for record in history)

    if isinstance(method, (KossenActiveTesting, PredictionPoweredActiveTesting)):
        assert all(record["selection"].selection_probabilities is not None for record in history)
    else:
        assert all(record["selection"].selection_probabilities is None for record in history)


def test_kossen_reports_unobserved_subgroups_as_missing() -> None:
    dataset = TinyEvaluationDataset()
    method = KossenActiveTesting(num_classes=2, seed=3, batch_size=1)

    history = run_active_evaluation(
        method=method,
        model=nn.Identity(),
        dataset=dataset,
        device=torch.device("cpu"),
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        protected_column="gender",
        budget=1,
    )

    estimate = history[0]["estimate"]
    unobserved_groups = [
        group for group, count in estimate.subgroup_n_labels.items() if count == 0
    ]
    assert unobserved_groups
    assert all(estimate.subgroup_risks[group] is None for group in unobserved_groups)


def test_kossen_full_pool_estimate_is_the_empirical_mean() -> None:
    dataset = TinyEvaluationDataset()
    history = run_active_evaluation(
        method=KossenActiveTesting(num_classes=2, seed=3, batch_size=len(dataset)),
        model=nn.Identity(),
        dataset=dataset,
        device=torch.device("cpu"),
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        protected_column="gender",
        budget=len(dataset),
    )

    final_record = history[-1]
    assert final_record["estimate"].risk == pytest.approx(np.mean(final_record["losses"]))


def test_kossen_rejects_target_output_with_wrong_class_count() -> None:
    dataset = TinyEvaluationDataset()
    method = KossenActiveTesting(num_classes=3)

    with pytest.raises(ValueError, match=r"\[batch, 3\]"):
        method.initialize(_problem(dataset, nn.Identity(), budget=2))


def test_surrogate_draws_are_normalized_over_classes() -> None:
    dataset = TinyEvaluationDataset()
    members = _predict_surrogate_members(
        [ShiftedClassifier(), ShiftedClassifier(reverse_classes=True)],
        dataset.features,
        batch_size=2,
        output_kind="logits",
        num_classes=2,
    )

    assert members.shape == (2, len(dataset), 2)
    np.testing.assert_allclose(members.sum(axis=2), 1.0)
