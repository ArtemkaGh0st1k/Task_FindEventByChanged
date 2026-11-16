"""Архив весов: запись, замена одной модели, распаковка только при необходимости."""

from __future__ import annotations

import pytest

from gtm_detector.ml_freq.models.archive import WeightsArchive


@pytest.fixture
def archive(tmp_path):
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "gboost.joblib").write_bytes(b"boost" * 1000)
    (directory / "transformer.pt").write_bytes(b"net" * 1000)
    return WeightsArchive(directory / "weights.zip", directory)


def test_pack_ignores_other_files(archive):
    (archive.directory / "notes.txt").write_text("не веса", encoding="utf-8")
    assert archive.pack() == ["gboost.joblib", "transformer.pt"]
    assert sorted(archive.members()) == ["gboost.joblib", "transformer.pt"]


def test_pack_without_weights_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        WeightsArchive(tmp_path / "weights.zip", tmp_path).pack()


def test_locate_extracts_missing_copy(archive):
    # Как после git clone: в репозитории только архив.
    archive.pack()
    (archive.directory / "gboost.joblib").unlink()
    path = archive.locate("gboost")
    assert path.read_bytes() == b"boost" * 1000
    assert not list(archive.directory.glob("*.tmp"))


def test_locate_replaces_stale_copy_of_same_size(archive):
    # Как после git pull: рабочая копия старая, размер совпал случайно.
    archive.pack()
    path = archive.directory / "transformer.pt"
    path.write_bytes(b"old" * 1000)
    assert archive.locate("transformer").read_bytes() == b"net" * 1000


def test_locate_keeps_matching_copy_untouched(archive):
    archive.pack()
    path = archive.directory / "gboost.joblib"
    before = path.stat().st_mtime_ns
    archive.locate("gboost")
    assert path.stat().st_mtime_ns == before


def test_add_replaces_one_model_and_keeps_others(archive):
    # Как после train одной модели на свежем клоне: других рабочих копий нет.
    archive.pack()
    (archive.directory / "transformer.pt").unlink()
    new = archive.directory / "gboost.joblib"
    new.write_bytes(b"retrained")
    archive.add([new])
    assert sorted(archive.members()) == ["gboost.joblib", "transformer.pt"]
    new.unlink()
    assert archive.locate("gboost").read_bytes() == b"retrained"
    assert archive.locate("transformer").read_bytes() == b"net" * 1000


def test_locate_uses_copy_missing_from_archive(archive):
    assert archive.locate("gboost").read_bytes() == b"boost" * 1000


def test_locate_without_weights_fails(archive):
    with pytest.raises(FileNotFoundError, match="logreg"):
        archive.locate("logreg")
