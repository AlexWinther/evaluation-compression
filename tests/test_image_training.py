from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from active_testing_benchmark.modeling.data import ImageClassificationDataset, class_mapping
from active_testing_benchmark.modeling.models import create_model
from active_testing_benchmark.modeling.train import (
    classification_metrics,
    inverse_frequency_class_weights,
    make_loader,
    run_epoch,
)


@pytest.fixture()
def tiny_image_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for split in ("training", "validation", "test"):
        for index, label in enumerate(("healthy", "disease")):
            relative_path = Path(split) / f"image_{index}.jpg"
            path = tmp_path / relative_path
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (16, 16), color=(index * 100, 20, 40)).save(path)
            rows.append({"image": str(relative_path), "label": label, "split": split})
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path, tmp_path


def test_dataset_loads_a_batch(tiny_image_dataset: tuple[Path, Path]) -> None:
    metadata_path, image_root = tiny_image_dataset
    mapping = class_mapping(metadata_path, "label")
    dataset = ImageClassificationDataset(
        metadata_path,
        image_root,
        "image",
        "label",
        "split",
        "training",
        mapping,
        transform=lambda image: torch.zeros(3, 32, 32),
    )
    images, labels = next(iter(DataLoader(dataset, batch_size=2)))
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)


@pytest.mark.parametrize("name", ["cnn", "resnet18", "densenet121", "vit", "clip"])
def test_models_produce_class_logits(name: str) -> None:
    model = create_model(name, num_classes=3, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 3)


def test_short_training_epoch_runs(tiny_image_dataset: tuple[Path, Path]) -> None:
    metadata_path, image_root = tiny_image_dataset
    mapping = class_mapping(metadata_path, "label")
    dataset = ImageClassificationDataset(
        metadata_path,
        image_root,
        "image",
        "label",
        "split",
        "training",
        mapping,
        transform=lambda image: torch.zeros(3, 32, 32),
    )
    model = create_model("cnn", num_classes=2, pretrained=False)
    loss, accuracy, *_ = run_epoch(
        model,
        DataLoader(dataset, batch_size=2),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        AdamW(model.parameters(), lr=1e-3),
    )
    assert loss >= 0
    assert 0 <= accuracy <= 1


def test_none_imbalance_strategy_keeps_shuffled_loader(
    tiny_image_dataset: tuple[Path, Path],
) -> None:
    metadata_path, image_root = tiny_image_dataset
    dataset = ImageClassificationDataset(
        metadata_path,
        image_root,
        "image",
        "label",
        "split",
        "training",
        class_mapping(metadata_path, "label"),
        transform=lambda image: torch.zeros(3, 32, 32),
    )

    loader = make_loader(dataset, batch_size=2, workers=0, shuffle=True)

    assert isinstance(loader.sampler, RandomSampler)


def test_weighted_sampling_uses_inverse_frequency_sample_weights(tmp_path: Path) -> None:
    metadata_path, image_root = _imbalanced_image_dataset(tmp_path)
    dataset = ImageClassificationDataset(
        metadata_path,
        image_root,
        "image",
        "label",
        "split",
        "training",
        class_mapping(metadata_path, "label"),
        transform=lambda image: torch.zeros(3, 32, 32),
    )

    loader = make_loader(
        dataset, batch_size=2, workers=0, shuffle=True, imbalance_strategy="weighted-sampling"
    )

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.replacement is True
    assert loader.sampler.weights.tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 2.0])


def test_class_weighted_loss_weights_minority_class_more(tmp_path: Path) -> None:
    metadata_path, image_root = _imbalanced_image_dataset(tmp_path)
    dataset = ImageClassificationDataset(
        metadata_path,
        image_root,
        "image",
        "label",
        "split",
        "training",
        class_mapping(metadata_path, "label"),
        transform=lambda image: torch.zeros(3, 32, 32),
    )

    weights = inverse_frequency_class_weights(dataset, num_classes=2)

    assert weights.tolist() == pytest.approx([2.0, 2 / 3])


def test_classification_metrics_include_macro_f1_and_binary_roc_auc() -> None:
    metrics = classification_metrics(
        labels=[0, 0, 1, 1],
        positive_probabilities=[0.1, 0.4, 0.6, 0.9],
        predictions=[0, 1, 1, 1],
        num_classes=2,
    )

    assert metrics["macro_f1"] == pytest.approx(0.7333333333333334)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_omit_undefined_roc_auc() -> None:
    metrics = classification_metrics(
        labels=[0, 0], positive_probabilities=[0.1, 0.2], predictions=[0, 0], num_classes=2
    )

    assert metrics == {"macro_f1": 1.0}


def _imbalanced_image_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for index, label in enumerate(("healthy", "healthy", "healthy", "disease")):
        relative_path = Path("training") / f"image_{index}.jpg"
        path = tmp_path / relative_path
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (16, 16)).save(path)
        rows.append({"image": str(relative_path), "label": label, "split": "training"})
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path, tmp_path
