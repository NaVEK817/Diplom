"""Фоновый загрузчик данных."""

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional
import zipfile

from PyQt5.QtCore import QThread, pyqtSignal

from data.archive_extractor import ArchiveExtractor
from data.mongodb_service import MongoDBService
from data.track_parser import TrackParser


class LocalFileProvider:
    """Поставщик локальных файлов с тем же контрактом чтения, что и ArchiveExtractor."""

    extract_dir = Path(".")

    def read_file_content(self, file_path: Path) -> Optional[List[str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [line.rstrip("\n\r") for line in file.readlines()]
        except OSError as error:
            print(f"Ошибка чтения файла {file_path}: {error}")
            return None

    def extract_range(self, *args, **kwargs) -> List[Path]:
        return []

    def extract_all_lt6(self) -> List[Path]:
        return []


class LoadStrategy(ABC):
    def __init__(self, worker: "DataLoadWorker"):
        self.worker = worker

    @abstractmethod
    def load(self) -> list:
        ...

    def is_cancelled(self) -> bool:
        return self.worker._is_cancelled

    def emit_progress(self, value: int, message: str) -> None:
        self.worker.progress.emit(value, message)


class RangeArchiveLoadStrategy(LoadStrategy):
    def load(self) -> list:
        worker = self.worker
        self.emit_progress(5, f"Загрузка файлов {worker.start_num}-{worker.end_num}...")

        extractor = ArchiveExtractor(worker.source, worker.extract_dir)
        if not extractor.check_archive_exists():
            worker.error.emit(f"Архив не найден: {worker.source}")
            return []

        total_files = worker.end_num - worker.start_num + 1
        parser = TrackParser(extractor, verbose=False, fill_missing=True)
        all_tracks = []

        for index, number in enumerate(range(worker.start_num, worker.end_num + 1)):
            if self.is_cancelled():
                return []

            filename = f"2006010{number}.lt6"
            progress = 10 + int((index / max(1, total_files)) * 85)
            self.emit_progress(progress, f"Обработка {filename}...")

            file_path = extractor.extract_file(filename)
            if file_path and file_path.exists():
                all_tracks.extend(parser.parse_track_file(file_path))
                self.emit_progress(progress, f"Загружено траекторий: {len(all_tracks)}")

        return all_tracks


class LocalFilesLoadStrategy(LoadStrategy):
    def load(self) -> list:
        file_list = self.worker.file_list or []
        self.emit_progress(5, f"Загрузка из {len(file_list)} файлов...")

        parser = TrackParser(LocalFileProvider(), verbose=False, fill_missing=True)
        all_tracks = []

        for index, file_path in enumerate(file_list):
            if self.is_cancelled():
                return []

            file_path = Path(file_path)
            progress = 5 + int((index / max(1, len(file_list))) * 90)
            self.emit_progress(progress, f"Разбор файла {file_path.name}...")
            all_tracks.extend(parser.parse_track_file(file_path))

        return all_tracks


class SingleArchiveLoadStrategy(LoadStrategy):
    def load(self) -> list:
        worker = self.worker
        self.emit_progress(5, "Подготовка к извлечению...")

        extractor = ArchiveExtractor(worker.source, worker.extract_dir)
        if not extractor.check_archive_exists():
            worker.error.emit(f"Архив не найден: {worker.source}")
            return []

        extracted_files = self._extract_files(worker.source, worker.file_list or [])
        return self._parse_files(extractor, extracted_files)

    def _extract_files(self, archive_path: str, filenames: Iterable[str]) -> List[Path]:
        extracted_files = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for filename in filenames:
                if self.is_cancelled():
                    return []

                target_path = Path(self.worker.extract_dir) / filename
                if not target_path.exists() or target_path.stat().st_size == 0:
                    try:
                        archive.extract(filename, self.worker.extract_dir)
                        self.emit_progress(10, f"Извлечён файл: {filename}")
                    except KeyError:
                        continue
                extracted_files.append(target_path)

        return extracted_files

    def _parse_files(self, extractor: ArchiveExtractor, file_paths: List[Path]) -> list:
        parser = TrackParser(extractor, verbose=False, fill_missing=True)
        all_tracks = []

        for index, file_path in enumerate(file_paths):
            if self.is_cancelled():
                return []

            progress = 20 + int((index / max(1, len(file_paths))) * 75)
            self.emit_progress(progress, f"Разбор файла {file_path.name}...")

            if file_path.exists():
                all_tracks.extend(parser.parse_track_file(file_path))

        return all_tracks


class MultipleArchivesLoadStrategy(LoadStrategy):
    def load(self) -> list:
        source = self.worker.source or []
        files_info = self.worker.file_list or []
        self.emit_progress(5, f"Загрузка из {len(source)} архивов...")

        files_by_archive = defaultdict(list)
        for archive_path, filename in files_info:
            files_by_archive[archive_path].append(filename)

        total_files = len(files_info)
        processed = 0
        all_tracks = []

        for archive_path, filenames in files_by_archive.items():
            if self.is_cancelled():
                return []

            self.emit_progress(10, f"Обработка архива {Path(archive_path).name}...")
            extractor = ArchiveExtractor(archive_path, self.worker.extract_dir)
            if not extractor.check_archive_exists():
                continue

            extracted_files = self._extract_archive_files(archive_path, filenames)
            parser = TrackParser(extractor, verbose=False, fill_missing=True)

            for file_path in extracted_files:
                if self.is_cancelled():
                    return []

                processed += 1
                progress = 20 + int((processed / max(1, total_files)) * 75)
                self.emit_progress(progress, f"Разбор файла {file_path.name}...")

                if file_path.exists():
                    all_tracks.extend(parser.parse_track_file(file_path))

        return all_tracks

    def _extract_archive_files(self, archive_path: str, filenames: Iterable[str]) -> List[Path]:
        extracted_files = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for filename in filenames:
                if self.is_cancelled():
                    return []

                target_path = Path(self.worker.extract_dir) / filename
                if not target_path.exists() or target_path.stat().st_size == 0:
                    try:
                        archive.extract(filename, self.worker.extract_dir)
                    except KeyError:
                        continue
                extracted_files.append(target_path)

        return extracted_files


class MongoDBLoadStrategy(LoadStrategy):
    def load(self) -> list:
        self.emit_progress(5, "Загрузка траекторий из MongoDB...")
        source = self.worker.source or {}
        service = MongoDBService(
            uri=source.get("uri", "mongodb://localhost:27017"),
            db_name=source.get("db_name", "admin"),
        )
        tracks = service.load_tracks(limit=source.get("limit"))
        if self.is_cancelled():
            return []
        self.emit_progress(100, f"Из MongoDB загружено траекторий: {len(tracks)}")
        return tracks


class DataLoadWorker(QThread):
    """Фоновый поток загрузки треков из архивов или локальных файлов."""

    finished = pyqtSignal(list)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    STRATEGIES: dict[str, Callable[["DataLoadWorker"], LoadStrategy]] = {
        "archive": SingleArchiveLoadStrategy,
        "files": LocalFilesLoadStrategy,
        "multiple": MultipleArchivesLoadStrategy,
        "range": RangeArchiveLoadStrategy,
        "mongodb": MongoDBLoadStrategy,
    }

    def __init__(self, source, extract_dir, file_list, mode="archive", start_num=1, end_num=10):
        super().__init__()
        self.source = source
        self.extract_dir = extract_dir
        self.file_list = file_list
        self.mode = mode
        self.start_num = start_num
        self.end_num = end_num
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            strategy_factory = self.STRATEGIES.get(self.mode)
            if strategy_factory is None:
                self.error.emit(f"Неподдерживаемый режим загрузки: {self.mode}")
                return

            all_tracks = strategy_factory(self).load()
            if self._is_cancelled:
                return

            self.progress.emit(100, "Загрузка завершена")
            self.finished.emit(all_tracks)

        except Exception as error:
            import traceback

            traceback.print_exc()
            self.error.emit(str(error))
