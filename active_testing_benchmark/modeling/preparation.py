"""Build deterministic development, pilot, and evaluation data views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from active_testing_benchmark.config import RAW_DATA_DIR
from active_testing_benchmark.modeling.data import (
    FairVisionDataset,
    get_class_counts,
    get_dataset_targets,
)
from active_testing_benchmark.modeling.models import create_image_transform


@dataclass(frozen=True)
class PreparedData:
    """Dataset views used by training and evaluation stages."""

    development: Dataset
    development_deterministic: Dataset
    pilot_fit: Subset
    pilot_validation: Subset
    evaluation: Dataset
    classes: list[str]


def _as_splits(splits: str | tuple[str, ...], name: str) -> tuple[str, ...]:
    values = (splits,) if isinstance(splits, str) else tuple(splits)
    if not values:
        raise ValueError(f"{name} must contain at least one split")
    return values


def _ids(dataset: Dataset) -> list[str]:
    source = dataset
    if isinstance(source, Subset):
        ids = _ids(source.dataset)
        return [ids[index] for index in source.indices]
    if hasattr(source, "samples"):
        return [str(sample["id"]) for sample in source.samples]
    if hasattr(source, "ids"):
        return [str(value) for value in source.ids]
    raise TypeError("Dataset must expose sample IDs through `samples` or `ids`")


def _require_all_classes(dataset: Dataset, classes: list[str], partition: str) -> None:
    try:
        get_class_counts(dataset)
    except ValueError as exc:
        raise ValueError(f"The {partition} partition must contain every class: {exc}") from exc
    if len(get_dataset_targets(dataset)) == 0:
        raise ValueError(f"The {partition} partition cannot be empty")


def prepare_data(
    *,
    disease: str = "dr",
    dev_splits: str | tuple[str, ...] = "train",
    eval_splits: str | tuple[str, ...] = "val",
    image_size: int = 224,
    seed: int = 42,
    fairvision_data_root: str | Path = RAW_DATA_DIR / "fairvision",
    protected_columns: tuple[str, ...] = ("gender",),
    dataset_factory: Callable | None = None,
    transform_factory: Callable | None = None,
) -> PreparedData:
    """Prepare independent transforms and a seeded stratified 20% pilot holdout."""
    dataset_factory = dataset_factory or FairVisionDataset
    transform_factory = transform_factory or create_image_transform
    dev_split_values = _as_splits(dev_splits, "dev_splits")
    eval_split_values = _as_splits(eval_splits, "eval_splits")
    development = dataset_factory(
        disease=disease,
        split=dev_split_values,
        transform=transform_factory(image_size=image_size, training=True),
        protected_columns=protected_columns,
        fairvision_data_root=fairvision_data_root,
    )
    development_deterministic = dataset_factory(
        disease=disease,
        split=dev_split_values,
        transform=transform_factory(image_size=image_size, training=False),
        protected_columns=protected_columns,
        fairvision_data_root=fairvision_data_root,
    )
    evaluation = dataset_factory(
        disease=disease,
        split=eval_split_values,
        transform=transform_factory(image_size=image_size, training=False),
        protected_columns=protected_columns,
        fairvision_data_root=fairvision_data_root,
    )

    classes = list(development.classes)
    if list(development_deterministic.classes) != classes or list(evaluation.classes) != classes:
        raise ValueError("Development and evaluation datasets must use the same ordered classes")
    development_ids = _ids(development)
    development_deterministic_ids = _ids(development_deterministic)
    evaluation_ids = _ids(evaluation)
    if development_ids != development_deterministic_ids:
        raise ValueError("Augmented and deterministic development views have different sample IDs")
    development_targets = get_dataset_targets(development)
    if development_targets != get_dataset_targets(development_deterministic):
        raise ValueError("Augmented and deterministic development views have different labels")
    overlap = set(development_ids).intersection(evaluation_ids)
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"Development and evaluation datasets overlap for {len(overlap)} sample IDs "
            f"(examples: {examples})"
        )

    get_class_counts(development)
    if len(development) == 0:
        raise ValueError("Development dataset cannot be empty")
    indices = np.arange(len(development))
    try:
        fit_indices, validation_indices = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed,
            stratify=development_targets,
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot create a stratified 20% pilot split; each class needs enough "
            "development examples for both pilot partitions"
        ) from exc

    pilot_fit = Subset(development, fit_indices.tolist())
    pilot_validation = Subset(development_deterministic, validation_indices.tolist())
    _require_all_classes(pilot_fit, classes, "pilot fit")
    _require_all_classes(pilot_validation, classes, "pilot validation")
    return PreparedData(
        development=development,
        development_deterministic=development_deterministic,
        pilot_fit=pilot_fit,
        pilot_validation=pilot_validation,
        evaluation=evaluation,
        classes=classes,
    )
