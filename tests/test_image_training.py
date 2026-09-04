from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from active_testing_benchmark.modeling.data import ImageClassificationDataset, class_mapping
from active_testing_benchmark.modeling.models import create_model
from active_testing_benchmark.modeling.train import run_epoch


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
        metadata_path, image_root, "image", "label", "split", "training", mapping,
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
        metadata_path, image_root, "image", "label", "split", "training", mapping,
        transform=lambda image: torch.zeros(3, 32, 32),
    )
    model = create_model("cnn", num_classes=2, pretrained=False)
    loss, accuracy, *_ = run_epoch(
        model, DataLoader(dataset, batch_size=2), nn.CrossEntropyLoss(), torch.device("cpu"),
        AdamW(model.parameters(), lr=1e-3),
    )
    assert loss >= 0
    assert 0 <= accuracy <= 1
