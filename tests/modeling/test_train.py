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


def _fixed_validation_metrics(monkeypatch, validation_losses):
    """Make every epoch report the given validation losses, with a constant AUC."""
    losses = iter(validation_losses)
    calls = iter(range(1_000))

    def epoch_metrics(labels, probabilities, loss):
        # train_model computes training metrics first and validation metrics second.
        is_validation = next(calls) % 2 == 1
        value = next(losses) if is_validation else loss
        return {"loss": value, "accuracy": 0.5, "f1": 0.5, "roc_auc": 0.7}

    monkeypatch.setattr(train, "_epoch_metrics", epoch_metrics)


def test_loss_selection_with_early_stopping_keeps_the_lowest_validation_loss(monkeypatch):
    _fixed_validation_metrics(monkeypatch, [0.5, 0.4, 0.6, 0.7, 0.8, 0.3])
    model = nn.Linear(2, 2)
    snapshots = []

    def capture_validation_state(module, args, output):
        if not module.training:
            snapshots.append({name: value.clone() for name, value in module.state_dict().items()})

    model.register_forward_hook(capture_validation_state)
    result, history = train.train_model(
        train.TrainingSetup(model),
        train.TrainingConfig(
            max_epochs=6, checkpoint_metric="validation_loss", early_stopping_patience=3
        ),
        FixedLogits(2),
        FixedLogits(2),
        "cpu",
    )

    # Epochs 3 to 5 do not improve on epoch 2, so training stops before epoch 6.
    assert [row["validation_loss"] for row in history] == [0.5, 0.4, 0.6, 0.7, 0.8]
    assert train.best_history_row(history, "validation_loss")["epoch"] == 2
    for name, value in result.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][name])


def test_best_history_row_maximises_auc_minimises_loss_and_keeps_the_first_tie():
    history = [
        {"epoch": 1, "roc_auc": 0.8, "validation_loss": 0.6},
        {"epoch": 2, "roc_auc": None, "validation_loss": 0.4},
        {"epoch": 3, "roc_auc": 0.8, "validation_loss": 0.4},
    ]
    assert train.best_history_row(history, "roc_auc")["epoch"] == 1
    assert train.best_history_row(history, "validation_loss")["epoch"] == 2
    with pytest.raises(ValueError, match="checkpoint_metric"):
        train.best_history_row(history, "accuracy")


def test_training_rejects_an_unknown_metric_or_non_positive_patience():
    for config in (
        train.TrainingConfig(max_epochs=1, checkpoint_metric="accuracy"),
        train.TrainingConfig(max_epochs=1, early_stopping_patience=0),
    ):
        with pytest.raises(ValueError):
            train.train_model(
                train.TrainingSetup(nn.Linear(2, 2)), config, FixedLogits(2), FixedLogits(2), "cpu"
            )


def test_constant_schedule_keeps_the_learning_rate_and_cosine_anneals_it():
    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3)
    assert train.create_scheduler(optimizer, train.TrainingConfig(lr_schedule="constant")) is None
    cosine = train.create_scheduler(optimizer, train.TrainingConfig(max_epochs=4))
    for _ in range(4):
        optimizer.step()
        cosine.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="lr_schedule"):
        train.create_scheduler(optimizer, train.TrainingConfig(lr_schedule="step"))


def test_stored_config_fills_fields_that_old_checkpoints_lack():
    old = {"max_epochs": 10, "batch_size": 32, "random_state": 5, "removed_field": 1}
    config = train.stored_training_config(old)
    assert (config.max_epochs, config.batch_size, config.random_state) == (10, 32, 5)
    assert config.checkpoint_metric == "roc_auc"
    assert config.early_stopping_patience is None
    assert config.lr_schedule == "cosine"
    new = train.stored_training_config({**old, "lr_schedule": "constant"})
    assert new.lr_schedule == "constant"
