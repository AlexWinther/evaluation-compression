"""Dataset and runtime provenance helpers for MLflow training runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import mlflow.data
import pandas as pd

from active_testing_benchmark.modeling.data import resolve_image_path

FAIRVISION_SOURCE = "https://huggingface.co/datasets/harvardairobotics/FairVision"
SPLITS = ("training", "validation", "test")


@dataclass(frozen=True)
class DatasetProvenance:
    """Sanitized dataset lineage suitable for MLflow Tracking."""

    datasets: dict[str, object]
    manifest_digest: str
    metadata_digest: str
    split_counts: dict[str, int]


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_provenance(
    metadata_path: Path,
    image_root: Path,
    image_column: str,
    target_column: str,
    split_column: str,
    class_to_index: dict[str, int],
    source: str = FAIRVISION_SOURCE,
) -> DatasetProvenance:
    """Build MLflow dataset metadata without logging patient rows or image bytes."""
    dataframe = pd.read_csv(metadata_path)
    required = {image_column, target_column, split_column}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(sorted(missing))}")

    records_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
    for _, row in dataframe.iterrows():
        split = str(row[split_column]).lower()
        if split not in records_by_split:
            continue
        label = str(row[target_column])
        image_path = resolve_image_path(
            image_root, str(row[image_column]), split, check_exists=False
        )
        try:
            image_reference = str(image_path.relative_to(image_root))
        except ValueError:
            image_reference = image_path.name
        records_by_split[split].append(
            {
                "class_index": class_to_index[label],
                "image": image_reference,
                "split": split,
                "target": label,
            }
        )

    for records in records_by_split.values():
        records.sort(key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":")))

    manifest = [record for split in SPLITS for record in records_by_split[split]]
    manifest_digest = _digest_json(manifest)
    datasets = {
        split: mlflow.data.from_pandas(
            pd.DataFrame(records),
            source=source,
            targets="target",
            name=f"fairvision-image-classification-{split}",
            digest=_dataset_digest(records),
        )
        for split, records in records_by_split.items()
    }
    return DatasetProvenance(
        datasets=datasets,
        manifest_digest=manifest_digest,
        metadata_digest=file_digest(metadata_path),
        split_counts={split: len(records) for split, records in records_by_split.items()},
    )


def git_provenance(project_root: Path) -> dict[str, str]:
    """Return concise Git state, including a useful fallback outside a repository."""
    commit = _git(project_root, "rev-parse", "HEAD") or "unknown"
    status = _git(project_root, "status", "--porcelain")
    return {"code.git.commit": commit, "code.git.is_dirty": str(bool(status)).lower()}


def _digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _dataset_digest(value: object) -> str:
    """Fit the full content hash into MLflow's 36-character dataset digest field."""
    return _digest_json(value)[:36]


def _git(project_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
