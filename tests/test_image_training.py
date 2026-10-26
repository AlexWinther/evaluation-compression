from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from active_testing_benchmark.modeling.data import (
    FairVisionDataset,
    get_class_counts,
    inverse_frequency_weights,
)
from active_testing_benchmark.modeling.models import create_image_transform, create_model
from active_testing_benchmark.modeling.train import (
    TrainConfig,
    evaluate,
    make_train_loader_and_loss,
    train_one_epoch,
)


@pytest.fixture
def fairvision_tree(tmp_path: Path) -> Path:
    return _write_fairvision_tree(tmp_path)


def test_torch_dataset_decodes_images_and_maps_labels(fairvision_tree: Path) -> None:
    dataset = FairVisionDataset(
        disease="DR",
        split="train",
        transform=lambda image: image.to(torch.float32) / 255,
        protected_columns=("gender", "age"),
        fairvision_data_root=fairvision_tree,
    )

    assert len(dataset) == 4
    assert dataset.classes == ["non-vision threatening dr", "vision threatening dr"]
    assert get_class_counts(dataset).tolist() == [3, 1]

    sample = dataset[0]
    assert sample["id"] == "000"
    assert sample["target"] == 0
    assert sample["image"].shape == (3, 12, 16)
    assert sample["image"].dtype == torch.float32
    assert sample["protected"]["gender"] == "female"
    assert sample["protected"]["age"] == "Unknown"


def test_dataset_aliases_splits_and_combines_requested_splits(fairvision_tree: Path) -> None:
    dataset = FairVisionDataset(
        split=("train", "val"),
        protected_columns=("gender",),
        fairvision_data_root=fairvision_tree,
    )

    assert len(dataset) == 6
    assert {sample["path"].parent.name for sample in dataset.samples} == {
        "training",
        "validation",
    }


@pytest.mark.parametrize(
    "disease,split,error",
    [
        ("unknown", "train", ValueError),
        ("dr", (), ValueError),
        ("dr", ("train", "training"), ValueError),
        ("dr", "holdout", ValueError),
        ("dr", "test", FileNotFoundError),
    ],
)
def test_dataset_rejects_invalid_disease_or_split(
    tmp_path: Path, disease: str, split: tuple[str, ...] | str, error: type[Exception]
) -> None:
    root = _write_fairvision_tree(tmp_path, include_test=False)
    with pytest.raises(error):
        FairVisionDataset(
            disease=disease,
            split=split,
            protected_columns=("gender",),
            fairvision_data_root=root,
        )


def test_dataset_rejects_missing_metadata_columns(fairvision_tree: Path) -> None:
    metadata_path = fairvision_tree / "dr" / "metadata.csv"
    metadata = pd.read_csv(metadata_path).drop(columns="gender")
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="Missing metadata columns"):
        FairVisionDataset(
            split="train",
            protected_columns=("gender",),
            fairvision_data_root=fairvision_tree,
        )


def test_class_counts_reject_a_missing_training_class() -> None:
    dataset = SimpleNamespace(targets=[0, 0], classes=["class 0", "class 1"])

    with pytest.raises(ValueError, match="No training samples for classes"):
        get_class_counts(dataset)


def test_inverse_frequency_weights_are_mean_normalized() -> None:
    weights = inverse_frequency_weights(torch.tensor([3, 1]))

    assert weights.tolist() == pytest.approx([0.5, 1.5])
    with pytest.raises(ValueError, match="power must be in \\[0, 1\\]"):
        inverse_frequency_weights(torch.tensor([3, 1]), power=1.1)


def test_weighted_loader_and_loss_use_training_class_counts(fairvision_tree: Path) -> None:
    dataset = FairVisionDataset(
        split="train",
        transform=create_image_transform(image_size=8, training=False),
        protected_columns=("gender",),
        fairvision_data_root=fairvision_tree,
    )
    config = TrainConfig(
        batch_size=2,
        num_workers=0,
        weighted_sampling=True,
        weighted_loss=True,
    )

    loader, criterion, counts, loss_weights = make_train_loader_and_loss(
        dataset, config, torch.device("cpu")
    )

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.replacement
    assert loader.sampler.weights.tolist() == pytest.approx([0.5, 0.5, 0.5, 1.5])
    assert counts.tolist() == [3, 1]
    assert loss_weights.tolist() == pytest.approx([0.5, 1.5])
    assert criterion.weight.tolist() == pytest.approx([0.5, 1.5])


def test_torch_training_and_evaluation_use_dataset_records(fairvision_tree: Path) -> None:
    transform = create_image_transform(image_size=8, training=False)
    train_dataset = FairVisionDataset(
        split="train",
        transform=transform,
        protected_columns=("gender",),
        fairvision_data_root=fairvision_tree,
    )
    test_dataset = FairVisionDataset(
        split="test",
        transform=transform,
        protected_columns=("gender",),
        fairvision_data_root=fairvision_tree,
    )
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    optimizer = AdamW(model.parameters(), lr=1e-3)

    loss = train_one_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
    )
    results = evaluate(
        model,
        DataLoader(test_dataset, batch_size=2),
        torch.device("cpu"),
        protected_column="gender",
    )

    assert loss >= 0
    assert results["loss"] >= 0
    assert 0 <= results["accuracy"] <= 1
    assert results["roc_auc"] is None or 0 <= results["roc_auc"] <= 1
    assert set(results["subgroup_metrics"]["gender"]) == {"female", "male"}
    assert len(results["predictions"]) == 2
    assert set(results["predictions"][0]) == {
        "id",
        "target",
        "prediction",
        "probabilities",
        "loss",
        "protected",
    }


def test_model_factory_uses_requested_class_count() -> None:
    model = create_model(num_classes=3, pretrained=False, device="cpu")
    model.eval()

    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 32, 32))

    assert logits.shape == (1, 3)


def _write_fairvision_tree(
    tmp_path: Path, *, include_test: bool = True
) -> Path:
    fairvision_root = tmp_path / "fairvision"
    disease_root = fairvision_root / "dr"
    rows = []
    split_labels = {
        "training": [
            "non-vision threatening dr",
            "non-vision threatening dr",
            "non-vision threatening dr",
            "vision threatening dr",
        ],
        "validation": ["non-vision threatening dr", "vision threatening dr"],
        "test": ["non-vision threatening dr", "vision threatening dr"],
    }
    if not include_test:
        split_labels.pop("test")

    for split_index, (split, labels) in enumerate(split_labels.items()):
        split_dir = disease_root / split
        split_dir.mkdir(parents=True)
        for index, label in enumerate(labels):
            image_id = f"{split_index}{index:02d}"
            Image.new("RGB", (16, 12), color=(index * 40, 60, 100)).save(
                split_dir / f"slo_fundus_{image_id}.jpg"
            )
            rows.append(
                {
                    "filename": f"data_{image_id}.npz",
                    "dr": label,
                    "gender": "female" if index % 2 == 0 else "male",
                    "age": None if index == 0 else 60 + index,
                    "race": "Unknown",
                    "ethnicity": "Unknown",
                    "language": "Unknown",
                    "maritalstatus": "Unknown",
                }
            )

    pd.DataFrame(rows).to_csv(disease_root / "metadata.csv", index=False)
    return fairvision_root
