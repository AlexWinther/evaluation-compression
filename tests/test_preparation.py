from dataclasses import dataclass
from pathlib import Path

import pytest

from active_testing_benchmark.modeling import preparation
from active_testing_benchmark.modeling.data import get_class_counts, get_dataset_targets


@dataclass
class TinyDataset:
    split: tuple[str, ...]
    transform: str
    samples: list[dict]
    targets: list[int]
    classes: list[str]

    def __len__(self):
        return len(self.samples)


def dataset_factory(*, split, transform, **kwargs):
    splits = (split,) if isinstance(split, str) else tuple(split)
    labels = [0] * 15 + [1] * 5 if "train" in splits else [0, 1]
    samples = [
        {"id": f"{name}-{i}", "target": label}
        for name in splits
        for i, label in enumerate(labels)
    ]
    targets = [sample["target"] for sample in samples]
    return TinyDataset(splits, transform, samples, targets, ["a", "b"])


def test_prepare_data_creates_seeded_stratified_pilot_and_transform_views(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(preparation, "FairVisionDataset", dataset_factory)
    monkeypatch.setattr(
        preparation,
        "create_image_transform",
        lambda *, image_size, training: "training" if training else "evaluation",
    )

    first = preparation.prepare_data(fairvision_data_root=tmp_path, seed=12)
    second = preparation.prepare_data(fairvision_data_root=tmp_path, seed=12)

    assert first.development.transform == "training"
    assert first.development_deterministic.transform == "evaluation"
    assert first.evaluation.transform == "evaluation"
    assert first.classes == ["a", "b"]
    assert len(first.pilot_validation) == 4
    assert get_class_counts(first.pilot_fit).tolist() == [12, 4]
    assert get_class_counts(first.pilot_validation).tolist() == [3, 1]
    fit_ids = {first.development.samples[i]["id"] for i in first.pilot_fit.indices}
    val_ids = {
        first.development_deterministic.samples[i]["id"]
        for i in first.pilot_validation.indices
    }
    assert fit_ids.isdisjoint(val_ids)
    assert first.pilot_fit.indices == second.pilot_fit.indices
    assert first.pilot_validation.indices == second.pilot_validation.indices


def test_prepare_data_rejects_development_evaluation_overlap(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        preparation,
        "create_image_transform",
        lambda *, image_size, training: "training" if training else "evaluation",
    )
    def overlapping_factory(*, split, transform, **kwargs):
        name = "shared" if split == "val" else "training"
        samples = [{"id": "shared", "target": 0}, {"id": f"{name}-1", "target": 1}]
        return TinyDataset((name,), transform, samples, [0, 1], ["a", "b"])

    monkeypatch.setattr(preparation, "FairVisionDataset", overlapping_factory)

    with pytest.raises(ValueError, match="overlap"):
        preparation.prepare_data(fairvision_data_root=tmp_path)


def test_prepare_data_rejects_pilot_partitions_missing_a_class(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        preparation,
        "create_image_transform",
        lambda *, image_size, training: "training" if training else "evaluation",
    )
    def too_small_factory(*, split, transform, **kwargs):
        samples = [{"id": f"{split}-0", "target": 0}, {"id": f"{split}-1", "target": 1}]
        return TinyDataset((split,), transform, samples, [0, 1], ["a", "b"])

    monkeypatch.setattr(preparation, "FairVisionDataset", too_small_factory)

    with pytest.raises(ValueError, match="pilot"):
        preparation.prepare_data(fairvision_data_root=tmp_path)


def test_dataset_helpers_support_generic_and_nested_subsets() -> None:
    from torch.utils.data import Subset

    dataset = TinyDataset(
        ("train",), "training", [], [0, 1, 1, 0], ["a", "b"]
    )
    nested = Subset(Subset(dataset, [3, 2, 1]), [0, 2])

    assert get_dataset_targets(nested) == [0, 1]
    assert get_class_counts(nested).tolist() == [1, 1]
