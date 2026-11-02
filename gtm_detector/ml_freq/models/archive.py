"""Zip-архив весов: в git лежит он один, распакованные файлы — только рабочие копии.

Веса деревьев не помещаются в лимит GitHub на файл (100 МБ), а сжимаются примерно втрое.
Архив — главный источник весов: :meth:`WeightsArchive.locate` распаковывает модель, если
рабочей копии нет или она отличается, например после ``git pull`` с новым архивом.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

SUFFIXES = (".pt", ".joblib")
COMPRESS_LEVEL = 9  # максимум: архив пересобирается редко, а размер в git важнее времени
CHUNK = 1 << 20


class WeightsArchive:
    """Архив ``path`` и папка ``directory`` с рабочими копиями весов."""

    def __init__(self, path: Path, directory: Path) -> None:
        self.path = Path(path)
        self.directory = Path(directory)

    def members(self) -> list[str]:
        if not self.path.exists():
            return []
        with zipfile.ZipFile(self.path) as archive:
            return archive.namelist()

    def add(self, files: Iterable[Path]) -> None:
        """Записать ``files`` в архив; остальные модели в нём остаются как были.

        Архив собирается заново во временном файле: прерванный запуск не испортит прежний.
        """
        files = [Path(file) for file in files]
        names = {file.name for file in files}
        temporary = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=COMPRESS_LEVEL
            ) as target:
                if self.path.exists():
                    with zipfile.ZipFile(self.path) as source:
                        for member in source.namelist():
                            if member not in names:
                                with source.open(member) as src, target.open(member, "w") as dst:
                                    shutil.copyfileobj(src, dst, CHUNK)
                for file in files:
                    target.write(file, file.name)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("Архив весов: %s, %d МБ", self.path, self.path.stat().st_size >> 20)

    def pack(self) -> list[str]:
        """Записать в архив все рабочие копии весов из папки."""
        files = sorted(p for p in self.directory.iterdir() if p.suffix in SUFFIXES)
        if not files:
            raise FileNotFoundError(f"Нет весов в {self.directory}: сначала выполните train")
        self.add(files)
        return [file.name for file in files]

    def locate(self, name: str) -> Path:
        """Путь к весам модели ``name``: при необходимости распаковывает их из архива."""
        members = self.members()
        for suffix in SUFFIXES:
            file = self.directory / f"{name}{suffix}"
            if file.name in members:
                self._extract(file)
                return file
            if file.exists():
                return file  # в архив ещё не добавлена: например, скопирована вручную
        raise FileNotFoundError(
            f"Нет весов модели {name} в {self.path} и {self.directory}: сначала выполните train"
        )

    def _extract(self, file: Path) -> None:
        with zipfile.ZipFile(self.path) as archive:
            info = archive.getinfo(file.name)
            if file.exists() and file.stat().st_size == info.file_size:
                if _crc32(file) == info.CRC:
                    return
            logger.info("Распаковка весов %s из %s", file.name, self.path)
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = file.with_name(file.name + ".tmp")
            try:
                with archive.open(info) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                temporary.replace(file)
            finally:
                temporary.unlink(missing_ok=True)


def _crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
    return crc
