"""Acceptance tests for the optional interpretable AT objective."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from active_testing_benchmark import benchmark, experiment
from active_testing_benchmark.active_testing.losses import ZeroOneLoss


class TinyAccuracyPool(Dataset):
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    targets = [0, 1, 0, 1]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "image": self.logits[index],
            "target": self.targets[index],
            "id": str(index),
            "protected": {"gender": "f" if index < 2 else "m"},
        }


def test_zero_one_loss_uses_argmax_and_is_per_example():
    loss = ZeroOneLoss()
    assert loss(torch.tensor([[1.0, 2.0]]), torch.tensor([1])).item() == 0
    assert loss(torch.tensor([[1.0, 2.0]]), torch.tensor([0])).item() == 1
    np.testing.assert_allclose(
        loss(TinyAccuracyPool.logits, torch.tensor(TinyAccuracyPool.targets)).numpy(), [0, 0, 1, 1]
    )


def test_all_six_methods_support_accuracy_and_full_pool_references():
    auxiliary = SimpleNamespace(surrogates=(nn.Identity(), nn.Identity()), proxy=nn.Identity())
    frames = benchmark.run_repetition(
        seed=2,
        device=torch.device("cpu"),
        dataset=TinyAccuracyPool(),
        model=nn.Identity(),
        method_configs=benchmark.build_method_configs(auxiliary, num_classes=2),
        budget=4,
        include_accuracy=True,
    )
    accuracy_frames = [frame for frame in frames if frame.risk_objective.iloc[0] == "error_rate"]
    assert len(accuracy_frames) == 6
    for frame in accuracy_frames:
        assert np.isfinite(frame.risk).all()
        if frame.method.iloc[0] != "ActiveSurrogateEstimator":
            row = frame.iloc[-1]
            assert row.risk == pytest.approx(0.5)
            assert row["risk f"] == pytest.approx(0.0)
            assert row["risk m"] == pytest.approx(1.0)


def test_accuracy_cli_is_opt_in_and_recorded(monkeypatch):
    configs = []
    monkeypatch.setattr(experiment, "run_experiment", configs.append)
    experiment.main([])
    experiment.main(["--include-accuracy"])
    assert [config.include_accuracy for config in configs] == [False, True]


def test_benchmark_artifact_and_reference_sidecar_integrate_with_analysis():
    import pandas as pd
    from torch.utils.data import DataLoader

    from active_testing_benchmark.modeling.evaluation import evaluate
    from active_testing_benchmark.postprocessing import analyze_results

    auxiliary = SimpleNamespace(surrogates=(nn.Identity(), nn.Identity()), proxy=nn.Identity())
    dataset = TinyAccuracyPool()
    weights = torch.tensor([0.5, 2.0])
    references = evaluate(
        nn.Identity(),
        DataLoader(dataset, batch_size=4),
        torch.device("cpu"),
        protected_column="gender",
        class_weights=weights,
    )
    method_configs = benchmark.build_method_configs(auxiliary, num_classes=2)
    for _, kwargs in method_configs:
        kwargs["batch_size"] = 1
    frames = []
    for repetition in range(2):
        frames.extend(
            benchmark.run_repetition(
                seed=2 + repetition,
                device=torch.device("cpu"),
                dataset=dataset,
                model=nn.Identity(),
                method_configs=method_configs,
                budget=4,
                repetition=repetition,
                class_weights=weights,
                include_accuracy=True,
            )
        )
    result = analyze_results(
        pd.concat(frames, ignore_index=True),
        {
            "configuration": {"repetitions": 2},
            "reference_metrics": references,
        },
        n_bootstrap=20,
    )
    assert set(result.curves.risk_objective) == {
        "unweighted_cross_entropy",
        "class_balanced_cross_entropy",
        "error_rate",
    }
    assert set(result.curves.target) == {
        "overall",
        "subgroup:gender:f",
        "subgroup:gender:m",
        "disparity:gender",
    }
    final = result.curves.query("n_labels == 4 and method != 'ActiveSurrogateEstimator'")
    np.testing.assert_allclose(final.error, 0, atol=1e-6)
