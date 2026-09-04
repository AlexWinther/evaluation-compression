from types import SimpleNamespace

from active_testing_benchmark.modeling import registry


def test_register_baseline_tags_version_and_moves_alias(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="7"),
        data=SimpleNamespace(metrics={"test_accuracy": 0.9, "test_macro_f1": 0.8}),
    )
    client = SimpleNamespace(
        set_model_version_tag=lambda *args: calls.append(("tag", *args)),
        set_registered_model_alias=lambda *args: calls.append(("alias", *args)),
    )

    monkeypatch.setattr(registry, "configure_mlflow", lambda: "sqlite:///test.db")
    monkeypatch.setattr(registry.mlflow, "get_run", lambda run_id: run)
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        lambda model_uri, name: SimpleNamespace(version="3"),
    )
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)

    version = registry.register_baseline("run-123", "fairvision-dr-image-classifier")

    assert version == "3"
    assert ("tag", "fairvision-dr-image-classifier", "3", "review.status", "accepted") in calls
    assert ("tag", "fairvision-dr-image-classifier", "3", "source.run_id", "run-123") in calls
    assert ("tag", "fairvision-dr-image-classifier", "3", "metric.test_accuracy", "0.9") in calls
    assert ("alias", "fairvision-dr-image-classifier", "baseline", "3") in calls
