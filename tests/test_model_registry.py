from types import SimpleNamespace

from active_testing_benchmark.modeling import registry


def test_register_baseline_tags_version_and_moves_alias(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="7"),
        data=SimpleNamespace(metrics={"test_accuracy": 0.9, "test_macro_f1": 0.8}),
    )
    client = SimpleNamespace(
        search_model_versions=lambda **kwargs: [SimpleNamespace(version="3", run_id="run-123")],
        set_model_version_tag=lambda *args: calls.append(("tag", *args)),
        set_registered_model_alias=lambda *args: calls.append(("alias", *args)),
    )

    monkeypatch.setattr(registry, "configure_mlflow", lambda: "sqlite:///test.db")
    monkeypatch.setattr(registry.mlflow, "get_run", lambda run_id: run)
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)

    version = registry.register_baseline("fairvision-dr-image-classifier")

    assert version == "3"
    assert ("tag", "fairvision-dr-image-classifier", "3", "review.status", "accepted") in calls
    assert ("tag", "fairvision-dr-image-classifier", "3", "source.run_id", "run-123") in calls
    assert ("tag", "fairvision-dr-image-classifier", "3", "metric.test_accuracy", "0.9") in calls
    assert ("alias", "fairvision-dr-image-classifier", "baseline", "3") in calls


def test_register_baseline_can_select_older_run(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    versions = [
        SimpleNamespace(version="4", run_id="new-run"),
        SimpleNamespace(version="2", run_id="old-run"),
    ]
    run = SimpleNamespace(
        info=SimpleNamespace(experiment_id="7"), data=SimpleNamespace(metrics={})
    )
    client = SimpleNamespace(
        search_model_versions=lambda **kwargs: versions,
        set_model_version_tag=lambda *args: calls.append(("tag", *args)),
        set_registered_model_alias=lambda *args: calls.append(("alias", *args)),
    )
    monkeypatch.setattr(registry, "configure_mlflow", lambda: "sqlite:///test.db")
    monkeypatch.setattr(registry.mlflow, "get_run", lambda run_id: run)
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)

    version = registry.register_baseline("fairvision-dr-cnn", run_id="old-run")

    assert version == "2"
    assert ("alias", "fairvision-dr-cnn", "baseline", "2") in calls
    assert ("tag", "fairvision-dr-cnn", "2", "source.run_id", "old-run") in calls


def test_fairvision_model_name_is_canonical() -> None:
    assert registry.fairvision_model_name("DR", "resnet_18") == "fairvision-dr-resnet-18"


def test_register_trained_model_tags_new_version(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        registry.mlflow,
        "register_model",
        lambda uri, name, tags: (
            calls.append(("register", uri, name, tags)) or SimpleNamespace(version="4")
        ),
    )
    version = registry.register_trained_model(
        "models:/m-123",
        "fairvision-dr-cnn",
        "run-123",
        "experiment-1",
        "dr",
        "cnn",
        {"test_accuracy": 0.9},
    )

    assert version == "4"
    registration = calls[0]
    assert registration[:3] == ("register", "models:/m-123", "fairvision-dr-cnn")
    assert registration[3]["disease.type"] == "dr"
    assert registration[3]["metric.test_accuracy"] == "0.9"


def test_configure_mlflow_requires_tracking_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.setattr(registry, "load_dotenv", lambda: None)

    try:
        registry.configure_mlflow()
    except RuntimeError as error:
        assert "MLFLOW_TRACKING_URI is required" in str(error)
    else:
        raise AssertionError("configure_mlflow() should reject missing MLFLOW_TRACKING_URI")


def test_configure_mlflow_uses_tracking_uri_for_registry_by_default(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.setattr(
        registry.mlflow,
        "set_tracking_uri",
        lambda uri: calls.append(("tracking", uri)),
    )
    monkeypatch.setattr(
        registry.mlflow,
        "set_registry_uri",
        lambda uri: calls.append(("registry", uri)),
    )

    result = registry.configure_mlflow()

    assert result == "https://mlflow.example.test"
    assert calls == [
        ("tracking", "https://mlflow.example.test"),
        ("registry", "https://mlflow.example.test"),
    ]


def test_configure_mlflow_honors_registry_override(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "sqlite:///registry.db")
    monkeypatch.setattr(
        registry.mlflow,
        "set_tracking_uri",
        lambda uri: calls.append(("tracking", uri)),
    )
    monkeypatch.setattr(
        registry.mlflow,
        "set_registry_uri",
        lambda uri: calls.append(("registry", uri)),
    )

    registry.configure_mlflow()

    assert calls == [
        ("tracking", "https://mlflow.example.test"),
        ("registry", "sqlite:///registry.db"),
    ]
