from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import decode_image

from active_testing_benchmark.config import CLASS_TO_IDX, RAW_DATA_DIR

SUBGROUP_COLUMNS = ("age", "gender", "race", "ethnicity", "language", "maritalstatus")
POSSIBLE_SPLIT_TYPES = {"train", "training", "val", "validation", "test"}
POSSIBLE_DISEASES = {"dr", "amd", "glaucoma"}


def _text(value: object) -> str:
    """Helper function to convert Null-like values to 'unknown'"""
    text = str(value).strip()
    return "unknown" if not text or text.lower() in {"nan", "none", "null"} else text


def _age_group(age: str) -> str:
    """Helper function to bin age into groups"""
    try:
        value = float(age)
    except ValueError:
        return "unknown"
    if not math.isfinite(value) or value < 0:
        return "unknown"
    if value < 50:
        return "<50"
    if value < 65:
        return "50-64"
    if value < 80:
        return "65-79"
    return "80+"


class FairVisionData(Dataset):
    def __init__(
        self,
        disease: str = "DR",
        split: str | set[str] = "train",
        transform: Callable[[Tensor], Tensor] | None = None,
        class_to_idx: dict[str, int] | None = None,
        fairvision_data_root: Path = RAW_DATA_DIR / "fairvision",
    ) -> None:
        if disease.lower() not in POSSIBLE_DISEASES:
            raise ValueError(
                f"Disease must be one of [{', '.join(POSSIBLE_DISEASES)}], got: {disease}"
            )

        splits = {split} if isinstance(split, str) else split

        if not splits or not splits.issubset(POSSIBLE_SPLIT_TYPES):
            raise ValueError(
                f"Split must only contain [{', '.join(POSSIBLE_SPLIT_TYPES)}], got: {', '.join(splits)}"
            )

        self.disease = disease
        self.root_dir = (fairvision_data_root / disease.lower()).resolve()
        self.splits = set(
            map(lambda s: {"train": "training", "val": "validation"}.get(s, s), splits)
        )

        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX[disease.lower()]

        if not self.class_to_idx or set(self.class_to_idx.values()) != set(
            range(len(self.class_to_idx))
        ):
            raise ValueError("class_to_idx must map labels bijectively to 0, ..., C-1.")

        self.classes = [
            label for label, _ in sorted(self.class_to_idx.items(), key=lambda pair: pair[1])
        ]

        metadata = pd.read_csv(self.root_dir / "metadata.csv", dtype=str, keep_default_na=False)
        if "use" not in metadata.columns:
            raise ValueError("metadata must contain 'use' column specifying the data split")

        self.records = (
            metadata[metadata["use"].isin(list(self.splits))].copy().reset_index(drop=True)
        )

        label_column = disease.lower()
        required = {"filename", label_column}
        missing = required.difference(self.records.columns)
        if missing:
            raise ValueError(f"metadata missing columns: {sorted(missing)}")
        if self.records.empty:
            raise ValueError(f"no metadata rows for data with splits: {self.splits}")
        for name in SUBGROUP_COLUMNS:
            if name not in self.records:
                self.records[name] = ""

        filename_pattern = re.compile(r"data_(\d+)\.npz\Z")
        ids, paths, labels = [], [], []
        self.subgroups = []
        for split_ in self.splits:
            split_dir = (self.root_dir / split_).resolve()
            split_records = self.records.loc[self.records["use"] == split_]
            for _, record in split_records.iterrows():
                filename = str(record["filename"])
                match = filename_pattern.fullmatch(filename)
                if match is None:
                    raise ValueError(f"invalid metadata filename: {filename}")

                sample_id = match.group(1)
                label = record[label_column].strip()
                if not label or label not in self.class_to_idx:
                    raise ValueError(f"unknown or missing label {label} for {filename}")
                candidates = [
                    split_dir / f"slo_fundus_{sample_id}.jpg",
                    split_dir / f"slo_{sample_id}.jpg",
                ]

                valid = []
                for candidate in candidates:
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(split_dir):
                        raise ValueError(f"image path escapes split directory: {candidate}")
                    if resolved.is_file():
                        valid.append(resolved)
                if not valid:
                    raise FileNotFoundError(f"no JPEG for {filename} in {split_dir}")
                if len(valid) > 1:
                    raise ValueError(f"multiple JPEGs for {filename} in {split_dir}")
                ids.append(sample_id)
                paths.append(valid[0])
                labels.append(float(self.class_to_idx[label]))
                subgroups = {name: _text(record[name]) for name in SUBGROUP_COLUMNS}
                subgroups["age_group"] = _age_group(subgroups["age"])
                self.subgroups.append(subgroups)

        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate sample IDs in {split}")

        self.sample_ids = ids
        self.image_paths = paths
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image = decode_image(self.image_paths[idx], mode="RGB")
        image = self.transform(image) if self.transform else image

        return {
            "image": image,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "sample_id": self.sample_ids[idx],
            "subgroups": self.subgroups[idx],
        }


@dataclass
class PreparedData:
    """Dataset views used by training and evaluation stages."""

    train: FairVisionData
    test: FairVisionData
    classes: list[str]


def prepare_data(
    disease: str = "DR",
    train_split: str = "train",
    transform: Callable[[Tensor], Tensor] | None = None,
    test_split: str = "val",
    class_to_idx: dict[str, int] | None = None,
    fairvision_data_root: Path = RAW_DATA_DIR / "fairvision",
):
    if train_split not in POSSIBLE_SPLIT_TYPES or test_split not in POSSIBLE_SPLIT_TYPES:
        raise ValueError(
            f"Train/test split must be in [{', '.join(POSSIBLE_SPLIT_TYPES)}], got: {train_split} and {test_split}"
        )

    if disease.lower() not in POSSIBLE_DISEASES:
        raise ValueError(
            f"Disease must be one of [{', '.join(POSSIBLE_DISEASES)}], got: {disease}"
        )

    class_to_idx = class_to_idx or CLASS_TO_IDX[disease.lower()]
    classes = [label for label, _ in sorted(class_to_idx.items(), key=lambda pair: pair[1])]

    return PreparedData(
        train=FairVisionData(
            disease=disease,
            split=train_split,
            transform=transform,
            class_to_idx=class_to_idx,
            fairvision_data_root=fairvision_data_root,
        ),
        test=FairVisionData(
            disease=disease,
            split=test_split,
            transform=transform,
            class_to_idx=class_to_idx,
            fairvision_data_root=fairvision_data_root,
        ),
        classes=classes,
    )


if __name__ == "__main__":
    data = FairVisionData()
