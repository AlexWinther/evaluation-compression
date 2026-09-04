from pathlib import Path
from zipfile import ZipFile

from active_testing_benchmark.dataset import (
    extract_slo_images,
    is_slo_image_member,
    selected_diseases,
)


def test_extract_slo_images_ignores_npz_and_macos_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "fairvision.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Training/slo_fundus_00001.jpg", b"training image")
        archive.writestr("Validation/slo_fundus_00002.jpg", b"validation image")
        archive.writestr("Test/slo_fundus_00003.jpg", b"test image")
        archive.writestr("Training/data_00001.npz", b"oct data")
        archive.writestr("__MACOSX/Training/._slo_fundus_00001.jpg", b"metadata")

    destination = tmp_path / "output"
    with ZipFile(archive_path) as archive:
        counts = extract_slo_images(archive, destination)

    assert counts == {"training": 1, "validation": 1, "test": 1}
    assert (destination / "training" / "slo_fundus_00001.jpg").read_bytes() == b"training image"
    assert not list(destination.rglob("*.npz"))


def test_extract_slo_images_replaces_an_incomplete_image(tmp_path: Path) -> None:
    archive_path = tmp_path / "fairvision.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Training/slo_fundus_00001.jpg", b"complete image")

    destination = tmp_path / "output"
    incomplete_path = destination / "training" / "slo_fundus_00001.jpg"
    incomplete_path.parent.mkdir(parents=True)
    incomplete_path.write_bytes(b"partial")
    incomplete_path.with_suffix(".jpg.part").write_bytes(b"leftover temporary file")

    with ZipFile(archive_path) as archive:
        extract_slo_images(archive, destination)

    assert incomplete_path.read_bytes() == b"complete image"
    assert not incomplete_path.with_suffix(".jpg.part").exists()


def test_upstream_member_filter_and_selection() -> None:
    assert is_slo_image_member("Training/slo_fundus_00001.jpg")
    assert not is_slo_image_member("Training/data_00001.npz")
    assert not is_slo_image_member("__MACOSX/Training/._slo_fundus_00001.jpg")
    assert selected_diseases("all") == ["amd", "dr", "glaucoma"]
