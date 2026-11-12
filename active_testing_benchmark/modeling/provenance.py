"""Record the exact source, environment, and dataset partitions used by an MLflow run."""

import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import mlflow
from torch.utils.data import Subset

from active_testing_benchmark.config import PROJ_ROOT


def log_code_provenance():
    """Capture working source as well as Git identity, including uncommitted changes."""
    for tag, arguments in (
        ("code.git.commit", ["rev-parse", "HEAD"]),
        ("code.git.is_dirty", ["status", "--porcelain"]),
    ):
        result = subprocess.run(
            ["git", *arguments], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
        )
        value = result.stdout.strip() if result.returncode == 0 else "unknown"
        if tag.endswith("is_dirty") and value != "unknown":
            value = str(bool(value)).lower()
        mlflow.set_tag(tag, value)
    mlflow.set_tag("python.version", platform.python_version())
    mlflow.set_tag("platform", platform.platform())
    with TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "source.zip"
        with ZipFile(archive_path, "w") as archive:
            for path in sorted((PROJ_ROOT / "active_testing_benchmark").rglob("*.py")):
                archive.write(path, path.relative_to(PROJ_ROOT))
        mlflow.log_artifact(str(archive_path), "provenance")
    for name in ("pyproject.toml", "uv.lock"):
        mlflow.log_artifact(str(PROJ_ROOT / name), "provenance")
    packages = sorted(
        f"{package.metadata['Name']}=={package.version}" for package in distributions()
    )
    mlflow.log_text("\n".join(packages) + "\n", "provenance/installed-packages.txt")


def log_dataset_provenance(**datasets):
    """Hash metadata and ordered sample IDs/labels without decoding or uploading images."""
    summaries = {}
    for name, dataset in datasets.items():
        indices = list(range(len(dataset)))
        source = dataset
        while isinstance(source, Subset):
            indices = [int(source.indices[index]) for index in indices]
            source = source.dataset
        samples = [(source.sample_ids[index], int(source.labels[index])) for index in indices]
        summaries[name] = {
            "size": len(dataset),
            "splits": sorted(source.splits),
            "sample_digest": hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
            "metadata_sha256": hashlib.sha256(
                (source.root_dir / "metadata.csv").read_bytes()
            ).hexdigest(),
        }
    mlflow.log_dict(summaries, "datasets.json")
