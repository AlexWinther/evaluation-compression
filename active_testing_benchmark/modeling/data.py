"""Small dataset helpers for image-classification experiments."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset[tuple[Tensor, int]]):
    """Load one metadata split without copying or transforming the source images."""

    def __init__(
        self,
        metadata_path: Path,
        image_root: Path,
        image_column: str,
        target_column: str,
        split_column: str,
        split: str,
        class_to_index: dict[str, int],
        transform: object | None = None,
        validate_paths: bool = True,
    ) -> None:
        dataframe = pd.read_csv(metadata_path)
        required = {image_column, target_column, split_column}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Metadata is missing required columns: {', '.join(sorted(missing))}")

        split_dataframe = dataframe[
            dataframe[split_column].astype(str).str.lower() == split.lower()
        ]
        if split_dataframe.empty:
            available = ", ".join(sorted(dataframe[split_column].dropna().astype(str).unique()))
            raise ValueError(
                f"No '{split}' rows in {split_column!r}; available values: {available}"
            )

        self.records: list[tuple[Path, int]] = []
        self.row_ids = split_dataframe.index.to_numpy(copy=True)
        for _, row in split_dataframe.iterrows():
            image_path = resolve_image_path(
                image_root, str(row[image_column]), split, check_exists=validate_paths
            )
            if validate_paths and not image_path.is_file():
                raise FileNotFoundError(
                    f"Image for metadata value {row[image_column]!r} was not found at {image_path}. "
                    "Pass --image-root or --image-column for a different layout."
                )
            label = str(row[target_column])
            self.records.append((image_path, class_to_index[label]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self.records[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def resolve_image_path(
    image_root: Path, image_value: str, split: str, check_exists: bool = True
) -> Path:
    """Resolve direct relative paths and FairVision ``data_*.npz`` identifiers.

    FairVision metadata names OCT files (``data_00001.npz``), while this project
    downloads corresponding SLO JPEGs (``training/slo_fundus_00001.jpg``).
    """
    supplied_path = Path(image_value)
    match = re.fullmatch(r"data_(\d+)\.npz", supplied_path.name)
    if match:
        fairvision_path = image_root / split / f"slo_fundus_{match.group(1)}.jpg"
        if not check_exists or fairvision_path.is_file():
            return fairvision_path
    candidates = [
        supplied_path if supplied_path.is_absolute() else image_root / supplied_path,
        image_root / split / supplied_path.name,
    ]
    if not check_exists:
        if supplied_path.is_absolute() or supplied_path.parent != Path("."):
            return candidates[0]
        return candidates[1]
    match = re.search(r"(\d+)$", supplied_path.stem)
    if match:
        candidates.append(image_root / split / f"slo_fundus_{match.group(1)}.jpg")
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])


def class_mapping(metadata_path: Path, target_column: str) -> dict[str, int]:
    """Build a stable class-to-index mapping from all metadata rows."""
    dataframe = pd.read_csv(metadata_path)
    if target_column not in dataframe.columns:
        raise ValueError(f"Metadata has no target column {target_column!r}")
    labels = sorted(dataframe[target_column].dropna().astype(str).unique())
    if len(labels) < 2:
        raise ValueError(f"Expected at least two labels in {target_column!r}, found {labels}")
    return {label: index for index, label in enumerate(labels)}
