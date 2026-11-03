from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from active_testing_benchmark.modeling.train import (
    TrainingConfig,
    fit_auxiliary_models,
    train_classifier,
    train_classifier_fast,
)


class TinyDataset(Dataset):
    classes: ClassVar[list[str]] = ["zero", "one"]

    def __init__(self, targets=None):
        self.targets = targets or [0, 0, 1, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "image": torch.tensor([float(index), 1.0]),
            "target": self.targets[index],
            "id": str(index),
            "protected": {"gender": "all"},
        }


@dataclass
class Data:
    development: Dataset
    pilot_fit: Dataset
    pilot_validation: Dataset
    development_deterministic: Dataset
    classes: list[str]


def data():
    dataset = TinyDataset()
    return Data(dataset, dataset, dataset, dataset, dataset.classes)


def custom_data(pilot_targets, validation_targets, development_targets):
    development = TinyDataset(development_targets)
    pilot_fit = TinyDataset(pilot_targets)
    pilot_validation = TinyDataset(validation_targets)
    return Data(
        development,
        pilot_fit,
        pilot_validation,
        TinyDataset(development_targets),
        development.classes,
    )


class TinyModel(nn.Module):
    def __init__(self, num_classes, pretrained=False, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(2, num_classes)
        self.to(device)

    def forward(self, x):
        return self.linear(x)


def config(**kwargs):
    values = {"batch_size": 2, "num_workers": 0, "epochs": 3}
    values.update(kwargs)
    return TrainingConfig(**values)


def test_full_trainer_selects_earliest_minimum_and_refits_for_selected_epochs(monkeypatch):
    from active_testing_benchmark.modeling import train

    created = []

    def model_factory(**kwargs):
        model = TinyModel(**kwargs)
        created.append(model)
        return model

    losses = iter([0.4, 0.2, 0.2])

    def fake_evaluate(*args, **kwargs):
        score = next(losses)
        return {"loss": score, "weighted_loss": score, "accuracy": 0.5}

    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    result = train_classifier(data(), config(), torch.device("cpu"), model_factory=model_factory)

    assert result.selected_epochs == 2
    assert result.metadata["pilot"]["epochs"] == 3
    assert result.metadata["refit"]["epochs"] == 2
    assert len(result.metadata["refit"]["training_loss"]) == 2
    assert all(loss >= 0 for loss in result.metadata["refit"]["training_loss"])
    assert len(created) == 2
    assert not torch.equal(created[0].linear.weight, created[1].linear.weight)
    assert result.class_counts.tolist() == [2, 2]


def test_selection_objective_is_configurable(monkeypatch):
    from active_testing_benchmark.modeling import train

    scores = iter([
        {"loss": 0.1, "weighted_loss": 0.8},
        {"loss": 0.2, "weighted_loss": 0.4},
        {"loss": 0.3, "weighted_loss": 0.5},
    ])
    monkeypatch.setattr(train, "evaluate", lambda *args, **kwargs: next(scores))
    result = train_classifier(
        data(), config(selection_objective="unweighted_cross_entropy"), torch.device("cpu"),
        model_factory=TinyModel,
    )
    assert result.selected_epochs == 1


def test_fast_training_caps_both_phases_and_updates_model(monkeypatch):
    from active_testing_benchmark.modeling import train

    counts = {"train": 0}
    initial_parameters = []
    created_models = []
    original = train.train_one_epoch

    def model_factory(**kwargs):
        model = TinyModel(**kwargs)
        initial_parameters.append(model.linear.weight.detach().clone())
        created_models.append(model)
        return model

    def count_batches(model, loader, *args, **kwargs):
        batches = list(loader)
        counts["train"] += len(batches)
        return original(model, batches, *args, **kwargs)

    monkeypatch.setattr(train, "train_one_epoch", count_batches)
    long_data = custom_data(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    )
    result = train_classifier_fast(
        long_data, config(batch_size=1), torch.device("cpu"), model_factory=model_factory
    )
    assert result.selected_epochs == 1
    assert counts["train"] == 20
    assert result.metadata["mode"] == "fast"
    assert result.metadata["pilot"]["batches_per_epoch"] == [10]
    assert result.metadata["refit"]["batches_per_epoch"] == [10]
    assert not torch.equal(initial_parameters[1], result.model.linear.weight.detach())


def test_training_config_rejects_unknown_selection_objective():
    with pytest.raises(ValueError, match="selection_objective"):
        TrainingConfig(selection_objective="accuracy")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_size", 0),
        ("learning_rate", float("inf")),
        ("weight_decay", float("nan")),
        ("sampling_power", float("nan")),
        ("loss_power", 1.1),
    ],
)
def test_training_config_rejects_invalid_numeric_settings(field, value):
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})


def test_pilot_and_refit_weights_use_their_own_partition_counts(monkeypatch):
    from active_testing_benchmark.modeling import train

    seen_weights = []

    def fake_evaluate(*args, **kwargs):
        seen_weights.append(kwargs["class_weights"].cpu())
        return {"loss": 0.4, "weighted_loss": 0.2}

    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    prepared = custom_data(
        pilot_targets=[0, 0, 0, 1],
        validation_targets=[0, 0, 1, 1],
        development_targets=[0] * 8 + [1] * 2,
    )
    result = train_classifier(prepared, config(epochs=1), torch.device("cpu"), model_factory=TinyModel)

    assert result.class_counts.tolist() == [8, 2]
    torch.testing.assert_close(seen_weights[0], torch.tensor([2 / 3, 2.0]))
    torch.testing.assert_close(result.risk_weights, torch.tensor([0.625, 2.5]))


def test_nonfinite_selection_score_fails_before_refit(monkeypatch):
    from active_testing_benchmark.modeling import train

    monkeypatch.setattr(
        train,
        "evaluate",
        lambda *args, **kwargs: {"loss": float("nan"), "weighted_loss": 0.4},
    )
    calls = []

    def model_factory(**kwargs):
        calls.append(1)
        return TinyModel(**kwargs)

    with pytest.raises(ValueError, match="non-finite"):
        train_classifier(
            data(),
            config(epochs=1, selection_objective="unweighted_cross_entropy"),
            torch.device("cpu"),
            model_factory=model_factory,
        )
    assert len(calls) == 1


def test_auxiliary_models_fit_on_full_development_deterministic_data(monkeypatch):
    from active_testing_benchmark.modeling import train

    seen = []

    class FakeForest:
        def __init__(self, **kwargs):
            pass

        def fit(self, loader, device="cpu"):
            seen.append(sum(len(batch["target"]) for batch in loader))
            return self

    monkeypatch.setattr(train, "RandomForestModule", FakeForest)
    auxiliary = fit_auxiliary_models(data(), config(), torch.device("cpu"))
    assert seen == [4, 4, 4]
    assert len(auxiliary.surrogates) == 2
    assert len(auxiliary.seeds) == 3
