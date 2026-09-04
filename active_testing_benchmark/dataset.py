"""Download FairVision SLO fundus images and metadata.

FairVision is available under CC BY-NC-ND 4.0 for non-commercial research.
See https://huggingface.co/datasets/harvardairobotics/FairVision; it must not be
used for clinical decisions or patient care.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil
from zipfile import ZipFile, ZipInfo

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from remotezip import RemoteZip
from tqdm import tqdm
import typer

from active_testing_benchmark.config import RAW_DATA_DIR

REPO_ID = "harvardairobotics/FairVision"
DISEASES = {
    "amd": {"upstream_dir": "AMD", "metadata": "data_summary_amd.csv"},
    "dr": {"upstream_dir": "DR", "metadata": "data_summary_dr.csv"},
    "glaucoma": {
        "upstream_dir": "Glaucoma",
        "metadata": "data_summary_glaucoma.csv",
    },
}
SPLITS = {"Training": "training", "Validation": "validation", "Test": "test"}

app = typer.Typer(add_completion=False, help=__doc__)


def selected_diseases(disease: str) -> list[str]:
    """Validate a disease selection and expand ``all`` explicitly."""
    if disease == "all":
        return list(DISEASES)
    if disease not in DISEASES:
        choices = ", ".join([*DISEASES, "all"])
        raise typer.BadParameter(f"must be one of: {choices}")
    return [disease]


def is_slo_image_member(member_name: str) -> bool:
    """Return whether an upstream ZIP member is a standalone SLO JPEG."""
    path = Path(member_name)
    return (
        len(path.parts) == 2
        and path.parts[0] in SPLITS
        and path.name.startswith("slo_")
        and path.suffix.lower() == ".jpg"
    )


def extract_slo_images(
    archive: ZipFile | RemoteZip,
    destination: Path,
    members: Iterable[ZipInfo] | None = None,
    show_progress: bool = False,
    progress_description: str = "SLO images",
) -> dict[str, int]:
    """Copy only complete standalone SLO JPEGs into split directories."""
    counts = {split: 0 for split in SPLITS.values()}
    archive_members = archive.infolist() if members is None else members
    image_members = [member for member in archive_members if is_slo_image_member(member.filename)]
    progress = tqdm(
        image_members,
        desc=progress_description,
        unit="image",
        disable=not show_progress,
    )

    for member in progress:
        split = SPLITS[Path(member.filename).parts[0]]
        output_path = destination / split / Path(member.filename).name
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists() or output_path.stat().st_size != member.file_size:
            temporary_path.unlink(missing_ok=True)
            with archive.open(member) as source, temporary_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary_path.replace(output_path)
        counts[split] += 1

    return counts


def download_metadata(disease: str, destination: Path) -> Path:
    """Download an unchanged disease CSV through Hugging Face's local cache."""
    info = DISEASES[disease]
    upstream_path = f"{info['upstream_dir']}/ReadMe/{info['metadata']}"
    cached_path = Path(
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=upstream_path)
    )
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "metadata.csv"
    shutil.copy2(cached_path, metadata_path)
    return metadata_path


def download_slo_images(disease: str, destination: Path) -> dict[str, int]:
    """Range-read and extract SLO JPEG members, never the OCT-containing NPZs."""
    upstream_dir = DISEASES[disease]["upstream_dir"]
    archive_path = f"{upstream_dir}/Dataset/dataset.zip"
    archive_url = hf_hub_url(repo_id=REPO_ID, repo_type="dataset", filename=archive_path)
    download_url = get_hf_file_metadata(url=archive_url).location

    # RemoteZip first obtains the ZIP central directory, then requests only the
    # selected JPEG byte ranges. It does not download dataset.zip as a whole.
    with RemoteZip(download_url) as archive:
        members = [member for member in archive.infolist() if is_slo_image_member(member.filename)]
        if not members:
            raise RuntimeError(
                f"No standalone SLO JPEGs found in {archive_path}; the upstream layout may have changed."
            )
        return extract_slo_images(
            archive,
            destination,
            members,
            show_progress=True,
            progress_description=f"{disease.title()} SLO",
        )


def summarize(disease: str, destination: Path, metadata_path: Path) -> None:
    """Print lightweight checks on the locally written disease directory."""
    counts = {split: len(list((destination / split).glob("*.jpg"))) for split in SPLITS.values()}
    npz_count = len(list(destination.rglob("*.npz")))

    typer.echo(f"\nFairVision / {disease.title()}")
    for split in SPLITS.values():
        typer.echo(f"{split.title() + ':':<12} {counts[split]} SLO images")
    typer.echo(f"{'Metadata:':<12} {metadata_path.name if metadata_path.exists() else 'missing'}")
    typer.echo(f"{'NPZ files:':<12} {npz_count}")


@app.command()
def main(
    disease: str = typer.Option(..., "--disease", help="amd, dr, glaucoma, or explicitly all."),
    metadata_only: bool = typer.Option(
        False, "--metadata-only", help="Download only metadata CSVs."
    ),
    output_dir: Path = typer.Option(  # noqa: B008 - Typer declares CLI options as defaults.
        RAW_DATA_DIR / "fairvision", "--output-dir", help="Directory containing disease folders."
    ),
) -> None:
    """Download FairVision metadata and standalone high-resolution SLO JPEGs."""
    for disease_name in selected_diseases(disease):
        destination = output_dir / disease_name
        metadata_path = download_metadata(disease_name, destination)
        typer.echo(f"Downloaded metadata to {metadata_path}")

        if metadata_only:
            summarize(disease_name, destination, metadata_path)
            continue

        typer.echo(
            "Retrieving standalone SLO JPEGs by ZIP byte range; "
            "the full archive and NPZ/OCT data are not downloaded."
        )
        extracted = download_slo_images(disease_name, destination)
        typer.echo(
            "Archive members processed: " + ", ".join(f"{k}={v}" for k, v in extracted.items())
        )
        summarize(disease_name, destination, metadata_path)


if __name__ == "__main__":
    app()
