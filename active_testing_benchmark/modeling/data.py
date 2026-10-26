"""Torch dataset implementation for FairVision dataset for image classification."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import re

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import decode_image

from active_testing_benchmark.config import (
    CLASS_TO_IDX,
    FAIRVISION_PROTECTED_ATTRIBUTES,
    RAW_DATA_DIR,
)


class FairVisionDataset(Dataset):
    def __init__(
        self,
        disease: str = "dr",
        split: str | Iterable[str] = "train",
        transform: Callable | None = None,
        protected_columns: tuple[str, ...] = FAIRVISION_PROTECTED_ATTRIBUTES,
        fairvision_data_root: str | Path = RAW_DATA_DIR / "fairvision",
    ):
        if disease.lower() not in {"dr", "glaucoma", "amd"}:
            raise ValueError(f"Unsupported disease: {disease}")

        self.root = Path(fairvision_data_root) / disease.lower()
        self.transform = transform
        self.class_to_idx = dict(CLASS_TO_IDX[disease.lower()])

        if not self.class_to_idx or set(self.class_to_idx.values()) != set(
            range(len(self.class_to_idx))
        ):
            raise ValueError("class_to_idx must map labels bijectively to 0, ..., C-1.")

        self.classes = [
            label for label, _ in sorted(self.class_to_idx.items(), key=lambda pair: pair[1])
        ]

        splits = (split,) if isinstance(split, str) else tuple(split)

        # Make "train" -> "training" and "val" -> "validation"
        splits = tuple(
            "training" if s == "train" else "validation" if s == "val" else s for s in splits
        )

        if (
            not splits
            or len(splits) != len(set(splits))
            or not set(splits) <= {"training", "validation", "test"}
        ):
            raise ValueError(
                "split must contain only unique values among training (train), validation (val), test."
            )

        metadata_id_column = "filename"
        metadata = pd.read_csv(
            self.root / "metadata.csv",
            dtype={metadata_id_column: "string", disease.lower(): "string"},
        )

        # Extract change the id from metadata filename coloumn
        metadata[metadata_id_column] = metadata[metadata_id_column].apply(
            lambda x: re.sub(r"^data_(\d+)\.npz$", r"\1", x)
        )

        required = {metadata_id_column, disease.lower(), *protected_columns}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")
        if metadata[metadata_id_column].isna().any():
            raise ValueError("Metadata contains missing image IDs.")
        metadata = metadata.set_index(metadata_id_column)
        if not metadata.index.is_unique:
            raise ValueError("Metadata contains duplicates in index coloumn")

        self.samples = []
        self.targets = []
        seen_ids = set()

        for name in splits:
            image_dir = self.root / name
            if not image_dir.is_dir():
                raise FileNotFoundError(image_dir)
            paths = sorted(image_dir.glob("*.jpg"))
            if not paths:
                raise ValueError(f"No JPEG images found in {image_dir}")

            for path in paths:
                # Extract the image ID from the path stem (filename without extension and without the "slo_fundus_" prefix)
                image_id = path.stem.replace("slo_fundus_", "")
                if image_id in seen_ids:
                    raise ValueError(f"Duplicate image ID across selected splits: {image_id}")
                seen_ids.add(image_id)
                if image_id not in metadata.index:
                    raise ValueError(f"Missing metadata for image: {image_id}")

                row = metadata.loc[image_id]
                label = row[disease.lower()]
                if pd.isna(label) or str(label) not in self.class_to_idx:
                    raise ValueError(f"Missing or unknown class for {image_id}: {label}")

                target = self.class_to_idx[str(label)]
                protected = {
                    col: "Unknown" if pd.isna(row[col]) else str(row[col])
                    for col in protected_columns
                }
                self.samples.append(
                    {
                        "path": path,
                        "id": image_id,
                        "target": target,
                        "protected": protected,
                    }
                )
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]

        image = decode_image(record["path"], mode="RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": record["target"],
            "id": record["id"],
            "protected": record["protected"],
        }


def get_class_counts(dataset: FairVisionDataset) -> torch.Tensor:
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=len(dataset.classes))
    missing = torch.where(counts == 0)[0].tolist()
    if missing:
        raise ValueError(
            f"No training samples for classes: {[dataset.classes[i] for i in missing]}"
        )
    return counts


def inverse_frequency_weights(counts: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be in [0, 1].")
    weights = counts.to(dtype=torch.float64).pow(-power)
    return (weights / weights.mean()).float()
