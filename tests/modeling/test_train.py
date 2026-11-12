import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from active_testing_benchmark.modeling import train
from active_testing_benchmark.modeling.registry import experiment_run


class FixedLogits(Dataset):
    def __init__(self, classes, *, reverse=False, single_class=False):
        self.labels = np.zeros(12, dtype=int) if single_class else np.arange(12) % classes
        predictions = (self.labels + 1) % classes if reverse else self.labels
        self.images = nn.functional.one_hot(torch.tensor(predictions), classes).float() * 4

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": int(self.labels[index])}


@pytest.mark.parametrize("classes", [2, 4])
def test_train_and_validation_metrics_are_separate_and_epoch_scoped(local_mlflow, classes):
    training_data = FixedLogits(classes)
    validation_data = FixedLogits(classes, reverse=True)
    model = nn.Linear(classes, classes, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(classes))
    setup = train.TrainingSetup(model, optimizer=torch.optim.SGD(model.parameters(), lr=0))
    config = train.TrainingConfig(max_epochs=2, batch_size=5)
    forwards = []
    model.register_forward_pre_hook(
        lambda module, args: forwards.append((module.training, torch.is_grad_enabled()))
    )
    with experiment_run("AT_dev", config, "dr", "cpu") as run:
        _, history = train.train_model(setup, config, training_data, validation_data, "cpu")

    # Metrics are aggregated over whole epochs, with no extra training-set inference pass.
    assert forwards == ([(True, True)] * 3 + [(False, False)] * 3) * 2
    for split, data in (("train", training_data), ("validation", validation_data)):
        scores = data.images.softmax(dim=1).numpy()
        labels = data.labels
        expected = {
            "loss": nn.functional.cross_entropy(data.images, torch.tensor(labels)).item(),
            "accuracy": train.accuracy_score(labels, scores.argmax(axis=1)),
            "f1": train.f1_score(
                labels,
                scores.argmax(axis=1),
                average="binary" if classes == 2 else "macro",
                zero_division=0,
            ),
            "roc_auc": (
                train.roc_auc_score(labels, scores[:, 1])
                if classes == 2
                else train.roc_auc_score(labels, scores, multi_class="ovr", average="macro")
            ),
        }
        for metric, value in expected.items():
            logged = local_mlflow.get_metric_history(run.info.run_id, f"training/{split}/{metric}")
            assert [entry.step for entry in logged] == [1, 2]
            assert all(entry.value == pytest.approx(value) for entry in logged)
    for row in history:
        assert row["training_accuracy"] == row["training_roc_auc"] == 1
        assert row["accuracy"] == row["f1"] == 0
        assert row["training_loss"] < row["validation_loss"]
    assert "training/accuracy" not in local_mlflow.get_run(run.info.run_id).data.metrics


def test_missing_training_class_keeps_validation_auc_and_checkpoint_valid(local_mlflow):
    model = nn.Linear(2, 2)
    setup = train.TrainingSetup(model)
    config = train.TrainingConfig(max_epochs=1)
    with experiment_run("AT_dev", config, "dr", "cpu") as run:
        _, history = train.train_model(
            setup, config, FixedLogits(2, single_class=True), FixedLogits(2), "cpu"
        )
    assert history[0]["training_roc_auc"] is None
    assert history[0]["roc_auc"] is not None
    assert not local_mlflow.get_metric_history(run.info.run_id, "training/train/roc_auc")
    assert local_mlflow.get_metric_history(run.info.run_id, "training/validation/roc_auc")


def test_checkpoint_selection_still_uses_validation_auc(monkeypatch):
    aucs = iter([0.9, 0.5, 0.8, 0.7, 0.95, 0.6])
    monkeypatch.setattr(train, "roc_auc_score", lambda *args, **kwargs: next(aucs))
    model = nn.Linear(2, 2)
    snapshots = []

    def capture_validation_state(module, args, output):
        if not module.training:
            snapshots.append({name: value.clone() for name, value in module.state_dict().items()})

    model.register_forward_hook(capture_validation_state)
    result, history = train.train_model(
        train.TrainingSetup(model),
        train.TrainingConfig(max_epochs=3),
        FixedLogits(2),
        FixedLogits(2),
        "cpu",
    )
    assert [row["roc_auc"] for row in history] == [0.5, 0.7, 0.6]
    for name, value in result.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][name])
