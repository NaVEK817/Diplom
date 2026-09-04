"""Главное окно приложения с меню и сохранением настроек"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QSplitter,
                             QMessageBox, QFileDialog, QInputDialog, QLabel,
                             QPushButton, QHBoxLayout, QDialog, QListWidget,
                             QListWidgetItem, QDialogButtonBox, QTableWidgetItem,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QTabWidget,
                             QFormLayout, QComboBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import numpy as np

from gui.control_panel import ControlPanel
from gui.results_panel import ResultsPanel
from gui.cluster_visualization_panel import ClusterVisualizationPanel
from gui.track_filter_dialog import TrackFilterDialog
from gui.log_panel import LogPanel
from workers.data_loader import DataLoadWorker
from workers.algorithm_runner import AlgorithmRunnerWorker
from workers.analysis_worker import FullAnalysisWorker
from data.mongodb_service import MongoDBService


class MainWindow(QMainWindow):
    """Главное окно интегрированного анализа"""
    
    SETTINGS_FILE = "tracon_settings.json"
    RESULTS_HISTORY_FILE = "analysis_history.json"
    DEFAULT_DATA_FILE = Path("data/default_dataset.json")
    CONSTRAINTS_FILE = Path("data/airspace_constraints.json")
    REPORTS_DIR = Path("reports")
    MONGODB_URI = "mongodb://localhost:27017"
    MONGODB_DB = "admin"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интегрированная система анализа траекторий ВС")
        self.setGeometry(100, 100, 1400, 800)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        
        self.tracks = []
        self.tracks_dict = {}
        self.results = {}
        self.data_loaded = False
        self.loading_default_data = False
        self.workers = []
        self.process_cancelled = False
        self.last_run_id = None
        self.db_service = self.init_database()
        
        self.settings = self.load_settings()
        self.extracted_files_dir = self.settings.get('extracted_dir', None)
        self.results_history = self.load_results_history()
        
        self.init_menu()
        self.init_ui()
        self.apply_settings()
        QTimer.singleShot(0, self.load_default_dataset)
    
    # ==================== ЗАГРУЗКА/СОХРАНЕНИЕ НАСТРОЕК ====================
    
    def init_database(self):
        try:
            return MongoDBService(self.MONGODB_URI, self.MONGODB_DB)
        except Exception as error:
            print(f"MongoDB connection error: {error}")
            return None

    def load_settings(self) -> dict:
        if self.db_service is not None:
            try:
                document = self.db_service.settings.find_one({"_id": "main"})
                if document and isinstance(document.get("settings"), dict):
                    return document["settings"]
            except Exception as e:
                print(f"MongoDB settings load error: {e}")

        settings_path = Path(self.SETTINGS_FILE)
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки настроек: {e}")
        return {
            'extracted_dir': None,
            'last_archive': None,
            'last_algorithm': 'RANSAC',
            'auto_load_history': True,
            'debug_mode': False,
            'max_algorithm_tracks': 300,
            'visual_max_clusters': 10,
            'max_history_items': 20,
            'ransac_threshold': 0.05,
            'ransac_distance_threshold': 0.05,
            'ransac_max_iterations': 1000,
            'ransac_line_distance_threshold': 0.5,
            'ransac_max_line_iterations': 300,
            'ransac_min_cluster_tracks': 2,
            'mlesac_outlier_ratio': 0.3,
            'mlesac_outlier_prob': 0.1,
            'mean_shift_bandwidth': 0.0,
            'mean_shift_max_iterations': 300,
            'mean_shift_min_cluster_size': 3,
            'mean_shift_bin_seeding': True,
            'scc_n_clusters': 0,
            'scc_gamma': 1.0,
            'scc_affinity_threshold': 0.5,
            'scc_max_exact_tracks': 700,
            'lscc_n_clusters': 0,
            'lscc_n_scales': 3,
            'lscc_affinity_metric': 'hybrid',
            'lscc_use_normalized_laplacian': True,
            'anomaly_threshold': 95.0,
            'cpm_iterations': 20
        }
    
    def save_settings(self):
        if self.db_service is not None:
            try:
                self.db_service.save_settings(self.settings)
                return
            except Exception as e:
                print(f"MongoDB settings save error: {e}")

        try:
            with open(self.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")
    
    def load_results_history(self) -> list:
        if self.db_service is not None:
            try:
                return self.db_service.load_history(20)
            except Exception as e:
                print(f"MongoDB history load error: {e}")

        history_path = Path(self.RESULTS_HISTORY_FILE)
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return []
    
    def save_result_to_history(self, result_data: dict):
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_tracks': result_data.get('statistics', {}).get('total_tracks', 0),
            'total_clusters': result_data.get('statistics', {}).get('total_clusters', 0),
            'anomalies_detected': result_data.get('statistics', {}).get('anomaly_stats', {}).get('total_anomalies_detected', 0),
            'algorithm': result_data.get('statistics', {}).get('algorithm', 'неизвестен'),
            'execution_time': result_data.get('statistics', {}).get('execution_time', 0)
        }
        self.results_history.insert(0, result_entry)
        max_items = self.settings.get('max_history_items', 20)
        self.results_history = self.results_history[:max_items]
        if self.db_service is not None:
            try:
                self.results_history = self.db_service.save_history_entry(result_entry, max_items)
                return
            except Exception as e:
                print(f"MongoDB history save error: {e}")

        try:
            with open(self.RESULTS_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения истории: {e}")
    
    def apply_settings(self):
        if hasattr(self, "cluster_visualization"):
            self.cluster_visualization.set_max_display_clusters(self.settings.get("visual_max_clusters", 10))

    def is_debug_mode(self):
        return bool(self.settings.get('debug_mode', False))

    def load_default_dataset(self):
        """Автоматически загружает данные из файла конфигурации при запуске."""
        if self.data_loaded:
            return

        if self.db_service is not None:
            try:
                if self.db_service.tracks.estimated_document_count() > 0:
                    self.loading_default_data = True
                    self.load_data_from_mongodb()
                    return
            except Exception as error:
                self.log_panel.log_message(f"MongoDB автозагрузка недоступна: {error}", "yellow")

        config_path = self.DEFAULT_DATA_FILE
        if not config_path.exists():
            self.log_panel.log_message(f"Файл автозагрузки не найден: {config_path}", "yellow")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
        except Exception as error:
            self.log_panel.log_message(f"Ошибка чтения файла автозагрузки: {error}", "red")
            return

        base_dir = config_path.parent.parent
        mode = config.get("mode", "files")

        if mode == "files":
            files = []
            for file_name in config.get("files", []):
                file_path = Path(file_name)
                if not file_path.is_absolute():
                    file_path = base_dir / file_path
                if file_path.exists():
                    files.append(str(file_path))

            if not files:
                self.log_panel.log_message("В файле автозагрузки нет доступных .lt6 файлов", "yellow")
                return

            self.loading_default_data = True
            self.load_data_from_files(files)
            return

        if mode == "archive":
            archive_path = Path(config.get("archive", ""))
            extract_dir = Path(config.get("extract_dir", "resourses/data"))
            if not archive_path.is_absolute():
                archive_path = base_dir / archive_path
            if not extract_dir.is_absolute():
                extract_dir = base_dir / extract_dir
            selected_files = config.get("files", [])

            if archive_path.exists() and selected_files:
                self.loading_default_data = True
                self.load_data(str(archive_path), str(extract_dir), selected_files, mode='archive')
            else:
                self.log_panel.log_message("В файле автозагрузки указан недоступный архив или пустой список файлов", "yellow")
    
    # ==================== МЕНЮ ====================
    
    def init_menu(self):
        menubar = self.menuBar()
    
        
        file_menu = menubar.addMenu("Файл")
        load_menu = file_menu.addMenu("Загрузить данные")
        
        load_menu.addAction("Из ZIP архива (выбор файлов)").triggered.connect(self.load_from_archive_dialog)
        load_menu.addAction("Из папки с .lt6 файлами").triggered.connect(self.load_from_folder_dialog)
        load_menu.addAction("Из ранее извлеченных файлов").triggered.connect(self.load_from_extracted_dialog)
        load_menu.addAction("Из нескольких архивов").triggered.connect(self.load_from_multiple_archives_dialog)
        
        load_menu.addAction("Из MongoDB").triggered.connect(self.load_from_mongodb)

        file_menu.addSeparator()
        history_menu = file_menu.addMenu("История анализов")
        history_menu.addAction("Показать историю").triggered.connect(self.show_results_history)
        history_menu.addSeparator()
        history_menu.addAction("Очистить историю").triggered.connect(self.clear_results_history)
        
        file_menu.addSeparator()
        export_action = file_menu.addAction("Сформировать выходные документы")
        export_action.triggered.connect(self.export_results)
        export_action.setEnabled(False)
        self.export_action = export_action
        
        file_menu.addSeparator()
        file_menu.addAction("Выход").triggered.connect(self.close)
        
        # Меню Анализ
        analysis_menu = menubar.addMenu("Анализ")
        analysis_menu.addAction("Полный анализ").triggered.connect(self.run_full_analysis)
        analysis_menu.addSeparator()
        analysis_menu.addAction("Только секторизация").triggered.connect(self.run_all_algorithms)
        analysis_menu.addAction("Детектировать аномалии").triggered.connect(self.detect_anomalies_for_cluster)
        analysis_menu.addSeparator()
        analysis_menu.addAction("Выбрать траектории ВС").triggered.connect(self.show_track_filter_dialog)
        analysis_menu.addAction("Сбросить выбор траекторий").triggered.connect(self.clear_track_selection)
        
        # Меню Настройки
        settings_menu = menubar.addMenu("Настройки")
        settings_menu.addAction("Указать папку с файлами").triggered.connect(self.set_extracted_files_directory)
        settings_menu.addAction("Настройка алгоритмов").triggered.connect(self.show_algorithm_settings)
        settings_menu.addAction("Настройка отображения").triggered.connect(self.show_display_settings)
        
        auto_load_action = settings_menu.addAction("Автозагрузка истории")
        auto_load_action.setCheckable(True)
        auto_load_action.setChecked(self.settings.get('auto_load_history', True))
        auto_load_action.triggered.connect(lambda checked: self._toggle_auto_load(checked))

        debug_action = settings_menu.addAction("Режим отладки")
        debug_action.setCheckable(True)
        debug_action.setChecked(self.is_debug_mode())
        debug_action.triggered.connect(self._toggle_debug_mode)
        
        # Меню Справка
        help_menu = menubar.addMenu("Справка")
        help_menu.addAction("О программе").triggered.connect(self.show_about)
        help_menu.addAction("Помощь").triggered.connect(self.show_help)
    
    def _toggle_auto_load(self, checked):
        self.settings['auto_load_history'] = checked
        self.save_settings()

    def _toggle_debug_mode(self, checked):
        self.settings['debug_mode'] = checked
        self.save_settings()
        self.log_panel.log_message(
            "Режим отладки включен" if checked else "Режим отладки выключен",
            "yellow" if checked else "green"
        )
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.control_panel = ControlPanel()
        self.control_panel.analyze_requested.connect(self.run_all_algorithms)
        self.control_panel.cancel_requested.connect(self.cancel_current_process)
        main_layout.addWidget(self.control_panel)
        
        splitter = QSplitter(Qt.Horizontal)
        self.cluster_visualization = ClusterVisualizationPanel()
        self.cluster_visualization.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_panel = ResultsPanel()
        self.log_panel = LogPanel()

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.results_panel)
        right_splitter.addWidget(self.log_panel)
        right_splitter.setSizes([360, 240])

        splitter.addWidget(self.cluster_visualization)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1050, 350])
        main_layout.addWidget(splitter)
        
        self.statusBar().showMessage("Готов к работе")
        self.results_panel.algorithm_selected.connect(self.show_algorithm_details)
    
    # ==================== ДИАЛОГИ ЗАГРУЗКИ ====================
    
    def set_extracted_files_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку с .lt6 файлами")
        if dir_path:
            self.extracted_files_dir = dir_path
            self.settings['extracted_dir'] = dir_path
            self.save_settings()
            self.log_panel.log_message(f"Папка установлена: {dir_path}", "green")

    def select_lt6_files_dialog(self, title: str, files: list) -> list:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(520, 420)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Выберите конкретные .lt6 файлы для загрузки:"))

        files_list = QListWidget()
        files_list.setSelectionMode(QListWidget.ExtendedSelection)
        for file_name in sorted(files):
            item = QListWidgetItem(str(file_name))
            files_list.addItem(item)
        layout.addWidget(files_list)

        buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Выбрать все")
        clear_btn = QPushButton("Снять выбор")
        buttons_layout.addWidget(select_all_btn)
        buttons_layout.addWidget(clear_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        select_all_btn.clicked.connect(files_list.selectAll)
        clear_btn.clicked.connect(files_list.clearSelection)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(dialog.accept)
        dialog_buttons.rejected.connect(dialog.reject)
        layout.addWidget(dialog_buttons)

        if dialog.exec_() != QDialog.Accepted:
            return []

        return [item.text() for item in files_list.selectedItems()]
    
    def load_from_extracted_dialog(self):
        if not self.extracted_files_dir or not Path(self.extracted_files_dir).exists():
            reply = QMessageBox.question(self, "Папка не указана", 
                                        "Указать папку с файлами?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.set_extracted_files_directory()
                if not self.extracted_files_dir:
                    return
            else:
                return
        
        folder = Path(self.extracted_files_dir)
        lt6_files = list(folder.glob("*.lt6"))
        if not lt6_files:
            QMessageBox.warning(self, "Нет файлов", f"В папке нет .lt6 файлов")
            return
        
        selected_files = self.select_lt6_files_dialog(
            "Выбор ранее извлеченных файлов",
            [str(f) for f in lt6_files]
        )
        if not selected_files:
            return
        self.load_data_from_files(selected_files)
    
    def load_from_archive_dialog(self):
        archive_path, _ = QFileDialog.getOpenFileName(self, "Выберите ZIP архив", "", "ZIP архивы (*.zip)")
        if not archive_path:
            return
        
        extract_dir = QFileDialog.getExistingDirectory(self, "Папка для извлечения")
        if not extract_dir:
            return
        
        self.extracted_files_dir = extract_dir
        self.settings['extracted_dir'] = extract_dir
        self.save_settings()
        
        import zipfile
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                available_files = [f for f in zf.namelist() if f.endswith('.lt6')]
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        
        if not available_files:
            QMessageBox.warning(self, "Нет файлов", "В архиве нет .lt6 файлов")
            return
        
        selected_files = self.select_lt6_files_dialog("Выбор файлов из архива", available_files)
        if not selected_files:
            return
        self.load_data(archive_path, extract_dir, selected_files, mode='archive')
    
    def load_from_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Папка с .lt6 файлами")
        if not folder_path:
            return
        
        self.extracted_files_dir = folder_path
        self.settings['extracted_dir'] = folder_path
        self.save_settings()
        
        lt6_files = list(Path(folder_path).glob("*.lt6"))
        if not lt6_files:
            QMessageBox.warning(self, "Нет файлов", "В папке нет .lt6 файлов")
            return
        
        selected_files = self.select_lt6_files_dialog(
            "Выбор файлов из папки",
            [str(f) for f in lt6_files]
        )
        if not selected_files:
            return
        self.load_data_from_files(selected_files)
    
    def load_from_multiple_archives_dialog(self):
        archives, _ = QFileDialog.getOpenFileNames(self, "Выберите ZIP архивы", "", "ZIP архивы (*.zip)")
        if not archives:
            return
        
        extract_dir = QFileDialog.getExistingDirectory(self, "Папка для извлечения")
        if not extract_dir:
            return
        
        self.extracted_files_dir = extract_dir
        self.settings['extracted_dir'] = extract_dir
        self.save_settings()
        
        all_files = []
        import zipfile
        for archive_path in archives:
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    for f in zf.namelist():
                        if f.endswith('.lt6'):
                            all_files.append((archive_path, f))
            except Exception:
                continue
        
        if not all_files:
            QMessageBox.warning(self, "Нет файлов", "В архивах нет .lt6 файлов")
            return
        
        labels = [
            f"{index + 1}. {Path(archive_path).name} :: {filename}"
            for index, (archive_path, filename) in enumerate(all_files)
        ]
        selected_labels = set(self.select_lt6_files_dialog("Выбор файлов из архивов", labels))
        if not selected_labels:
            return

        selected_files = [
            file_info for file_info, label in zip(all_files, labels)
            if label in selected_labels
        ]

        self.load_data_from_multiple_archives(archives, extract_dir, selected_files)
    
    # ==================== ЗАГРУЗКА ДАННЫХ ====================
    
    def load_from_mongodb(self):
        self.load_data_from_mongodb()

    def load_data_from_mongodb(self):
        self.process_cancelled = False
        self.control_panel.set_loading_state(True)

        self.load_worker = DataLoadWorker(
            {"uri": self.MONGODB_URI, "db_name": self.MONGODB_DB},
            None,
            None,
            mode='mongodb'
        )
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_data_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.start()
        self.workers.append(self.load_worker)

    def load_data(self, archive_path, extract_dir, selected_files, mode='archive'):
        self.process_cancelled = False
        self.control_panel.set_loading_state(True)
        
        self.load_worker = DataLoadWorker(archive_path, extract_dir, selected_files, mode)
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_data_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.start()
        self.workers.append(self.load_worker)
    
    def load_data_from_files(self, file_paths):
        self.process_cancelled = False
        self.control_panel.set_loading_state(True)
        
        self.load_worker = DataLoadWorker(None, None, file_paths, mode='files')
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_data_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.start()
        self.workers.append(self.load_worker)
    
    def load_data_from_multiple_archives(self, archives, extract_dir, files_info):
        self.process_cancelled = False
        self.control_panel.set_loading_state(True)
        
        self.load_worker = DataLoadWorker(archives, extract_dir, files_info, mode='multiple')
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_data_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.start()
        self.workers.append(self.load_worker)
    
    def on_load_progress(self, value, message):
        self.control_panel.update_progress(value)
    
    def on_data_loaded(self, tracks):
        if self.process_cancelled:
            self.process_cancelled = False
            return
        self.tracks = tracks
        self.tracks_dict = {getattr(t, "mongo_id", None) or f"{t.track_num}_{t.eventid}": t for t in tracks}
        if self.db_service is not None and tracks:
            try:
                saved_count = self.db_service.save_tracks(tracks)
                self.log_panel.log_message(f"MongoDB: сохранено/обновлено траекторий: {saved_count}", "green")
            except Exception as error:
                self.log_panel.log_message(f"MongoDB: не удалось сохранить траектории: {error}", "yellow")
        
        self.data_loaded = True
        self.control_panel.set_data_loaded(True)
        self.control_panel.set_loading_state(False)
        self.export_action.setEnabled(True)
        self.results_panel.clear_results()
        self.cluster_visualization.clear()
        self.results = {}
        
        if self.loading_default_data:
            self.loading_default_data = False
        else:
            QMessageBox.information(self, "Загрузка завершена", f"Загружено {len(tracks)} траекторий")
    
    def on_load_error(self, error_msg):
        if self.process_cancelled:
            self.process_cancelled = False
            self.control_panel.set_cancelled_state(self.data_loaded)
            return
        self.log_panel.log_message(f"ОШИБКА: {error_msg}", "red")
        self.control_panel.set_loading_state(False)
        self.loading_default_data = False
        self.process_cancelled = False
        QMessageBox.critical(self, "Ошибка загрузки", error_msg)

    def cancel_current_process(self):
        cancelled = False
        for attr_name in ("load_worker", "analysis_worker", "algo_worker"):
            worker = getattr(self, attr_name, None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                cancelled = True

        if not cancelled:
            return

        self.loading_default_data = False
        self.process_cancelled = True
        self.control_panel.set_cancelled_state(self.data_loaded)
        self.log_panel.log_message("Текущий процесс отменяется. Если алгоритм выполняет тяжелую операцию, остановка произойдет на ближайшей контрольной точке.", "yellow")
    def load_from_archive_with_range_dialog(self):
        """Загрузка выбранных файлов из ZIP архива."""
        archive_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите ZIP архив с данными", "", "ZIP архивы (*.zip)"
        )
        if not archive_path:
            return

        extract_dir = QFileDialog.getExistingDirectory(
            self, "Выберите папку для извлечения файлов"
        )
        if not extract_dir:
            return

        import zipfile
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                available_files = [f for f in zf.namelist() if f.endswith('.lt6')]
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        if not available_files:
            QMessageBox.warning(self, "Нет файлов", "В архиве нет .lt6 файлов")
            return

        selected_files = self.select_lt6_files_dialog("Выбор файлов из архива", available_files)
        if not selected_files:
            return

        self.extracted_files_dir = extract_dir
        self.settings['extracted_dir'] = extract_dir
        self.settings['last_archive'] = archive_path
        self.save_settings()

        self.load_data(archive_path, extract_dir, selected_files, mode='archive')


    def load_data_range(self, archive_path: str, extract_dir: str, num_files: int):
        """Загрузка диапазона файлов"""
        self.process_cancelled = False
        self.control_panel.set_loading_state(True)
        
        self.load_worker = DataLoadWorker(
            source=archive_path,
            extract_dir=extract_dir,
            file_list=None,
            mode='range',
            start_num=1,
            end_num=num_files
        )
        self.load_worker.progress.connect(self.on_load_progress)
        self.load_worker.finished.connect(self.on_data_loaded)
        self.load_worker.error.connect(self.on_load_error)
        self.load_worker.start()
        
        self.workers.append(self.load_worker)

    # ==================== ГЛАВНАЯ ФУНКЦИЯ - ПОЛНЫЙ АНАЛИЗ ====================
    
    def run_full_analysis(self):
        """Запуск полного трехэтапного анализа"""
        if not self.data_loaded:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        algorithm = "mlesac"
        
        use_cpm, ok = QInputDialog.getItem(
            self, "Моделирование центроида",
            "Использовать CPM?",
            ["Да", "Нет"], 0, False
        )
        use_cpm = (use_cpm == "Да")
        
        self.log_panel.log_message("="*60, "purple")
        self.log_panel.log_message("ЗАПУСК ПОЛНОГО АНАЛИЗА (4 ЭТАПА)", "purple")
        self.log_panel.log_message("  Этап 1: Совместная кластеризация", "blue")
        self.log_panel.log_message("  Этап 2: Построение управляемых 3D-секторов", "blue")
        self.log_panel.log_message(f"  Этап 3: Моделирование центроидов ({'CPM' if use_cpm else 'Среднее'})", "blue")
        self.log_panel.log_message(f"  Этап 4: Детектирование аномалий", "blue")
        self.log_panel.log_message("="*60, "purple")
        
        self.control_panel.set_analyzing_state(True)
        self.process_cancelled = False
        self.results_panel.clear_results()
        self.cluster_visualization.clear()
        self.results = {}
        
        self.analysis_worker = FullAnalysisWorker(
            self.tracks_dict, 
            algorithm=algorithm,
            use_cpm=use_cpm,
            anomaly_threshold=self.settings.get('anomaly_threshold', 95.0),
            restricted_zones=self.load_airspace_constraints(),
            settings=self.settings
        )
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.stage_completed.connect(self.on_stage_completed)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()
        self.workers.append(self.analysis_worker)

    def load_airspace_constraints(self):
        if not self.CONSTRAINTS_FILE.exists():
            return []
        try:
            with open(self.CONSTRAINTS_FILE, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data.get('restricted_zones', [])
        except Exception as error:
            self.log_panel.log_message(f"Не удалось загрузить ограничения ОрВД: {error}", "yellow")
            return []
    
    def on_analysis_progress(self, value, message):
        self.control_panel.update_progress(value)
        self.log_panel.log_message(message, "blue")
    
    def on_stage_completed(self, stage, result):
        if stage == 'clustering':
            self.log_panel.log_message(f"Этап 1: {result['num_clusters']} пучков", "green")
        elif stage == 'sectors':
            self.log_panel.log_message(
                f"Этап 2: {result['num_sectors']} секторов, "
                f"входов {result['total_entry_fixes']}, выходов {result['total_exit_fixes']}",
                "green"
            )
        elif stage == 'workload':
            self.log_panel.log_message(
                f"Нагрузка: Wmax={result['max_workload_index']:.1f}, "
                f"конфликтных событий={result['total_conflicts']}, пик={result['max_peak_load']} ВС",
                "green"
            )
        elif stage == 'runways':
            self.log_panel.log_message(
                f"ВПП: {result['runways_count']} направлений, "
                f"нарушений интервала={result['spacing_violations']}, "
                f"перегруженных={result['overloaded_count']}, планов переноса={result['reassignment_plans']}",
                "green"
            )
        elif stage == 'centroids':
            self.log_panel.log_message(f"Этап 3: {result['num_centroids']} центроидов", "green")
        elif stage == 'procedure_design':
            self.log_panel.log_message(
                f"PBN/RNP: замечаний={result['pbn_issues']}, "
                f"GeoJSON={result['geojson_features']}, ARINC-строк={result['arinc_rows']}, "
                f"качество={result['avg_quality']:.1f}",
                "green"
            )
    
    def on_analysis_finished(self, results):
        if self.process_cancelled:
            self.process_cancelled = False
            self.control_panel.set_cancelled_state(self.data_loaded)
            return
        self.results = results
        clusters = results.get('clusters', {})
        display_clusters = self._filter_output_clusters(clusters)
        sectors = results.get('sectors', {})
        anomalies = results.get('anomalies', {})
        risks = results.get('risks', {})
        runway_optimization = results.get('runway_optimization', {})
        stats = results.get('statistics', {})
        
        self.save_result_to_history(results)
        self.last_run_id = None
        if self.db_service is not None:
            try:
                self.last_run_id = self.db_service.save_full_analysis(results, self.tracks_dict, self.settings)
                self.log_panel.log_message(f"MongoDB: результат анализа сохранён, run_id={self.last_run_id}", "green")
            except Exception as error:
                self.log_panel.log_message(f"MongoDB: не удалось сохранить результат анализа: {error}", "yellow")
        self.results_panel.display_full_results(display_clusters, anomalies, risks, runway_optimization)
        self.cluster_visualization.set_clusters(
            self.tracks_dict,
            display_clusters,
            "Кластеры и управляемые секторы"
        )
        
        self.log_panel.log_message("="*60, "purple")
        self.log_panel.log_message("РЕЗУЛЬТАТЫ АНАЛИЗА", "purple")
        self.log_panel.log_message(f"Всего траекторий: {stats.get('total_tracks', 0)}", "white")
        self.log_panel.log_message(f"Найдено пучков: {stats.get('total_clusters', 0)}", "cyan")
        self.log_panel.log_message(f"Построено 3D-секторов: {stats.get('total_sectors', len(sectors))}", "cyan")
        self.log_panel.log_message(
            f"Макс. индекс нагрузки W: {stats.get('max_workload_index', 0):.1f}; "
            f"конфликтных событий: {stats.get('total_conflict_events', 0)}",
            "yellow"
        )
        if sectors:
            total_entry = sum(len(sector.get('entry_fixes', [])) for sector in sectors.values())
            total_exit = sum(len(sector.get('exit_fixes', [])) for sector in sectors.values())
            self.log_panel.log_message(f"Виртуальные точки: входов {total_entry}, выходов {total_exit}", "white")
        runway_stats = stats.get('runway_optimization_stats', {})
        if runway_stats:
            overloaded = ", ".join(runway_stats.get('overloaded_runways', [])) or "нет"
            self.log_panel.log_message(
                f"ВПП: прибытий {runway_stats.get('arrivals_count', 0)}, "
                f"нарушений интервала {runway_stats.get('spacing_violations', 0)}, "
                f"перегруженные: {overloaded}",
                "yellow"
            )
        self.log_panel.log_message(
            f"PBN/RNP замечаний: {stats.get('pbn_issues_count', 0)}; "
            f"ARINC-строк: {stats.get('procedure_export_rows', 0)}; "
            f"качество процедуры: {stats.get('avg_procedure_quality', 0):.1f}",
            "cyan"
        )
        what_if_delta = stats.get('what_if_delta', {})
        if what_if_delta:
            self.log_panel.log_message(
                f"What-if: изменение нагрузки {what_if_delta.get('max_workload_index', 0):.1f}, "
                f"конфликты {what_if_delta.get('total_conflicts', 0)}",
                "white"
            )
        
        anomaly_stats = stats.get('anomaly_stats', {})
        if anomaly_stats:
            self.log_panel.log_message(f"Аномалий: {anomaly_stats.get('total_anomalies_detected', 0)} ({anomaly_stats.get('anomaly_percentage', 0):.1f}%)", "yellow")
        rule_risk_stats = stats.get('rule_risk_stats', {})
        if rule_risk_stats:
            self.log_panel.log_message(
                f"Рисков по жестким критериям: {rule_risk_stats.get('total_rule_risks', 0)} "
                f"на {rule_risk_stats.get('tracks_with_rule_risks', 0)} траекториях",
                "yellow"
            )
        
        self.log_panel.log_message(f"Время: {stats.get('execution_time', 0):.1f} сек", "white")
        report_paths = self.generate_analysis_documents(results)
        if report_paths:
            self.log_panel.log_message(f"Выходные документы сохранены: {Path(report_paths[0]).parent}", "green")
        self.log_panel.log_message("="*60, "purple")
        
        self.control_panel.set_analyzing_state(False)
        self.export_action.setEnabled(True)
        
        QMessageBox.information(self, "Анализ завершен", 
                               f"Найдено пучков: {stats.get('total_clusters', 0)}\n"
                               f"Аномалий: {anomaly_stats.get('total_anomalies_detected', 0)}\n"
                               f"Жестких рисков: {rule_risk_stats.get('total_rule_risks', 0)}\n"
                               f"Время: {stats.get('execution_time', 0):.1f} сек")
    
    def on_analysis_error(self, error_msg):
        if self.process_cancelled:
            self.process_cancelled = False
            self.control_panel.set_cancelled_state(self.data_loaded)
            return
        self.log_panel.log_message(f"ОШИБКА: {error_msg}", "red")
        self.control_panel.set_analyzing_state(False)
        self.process_cancelled = False
        QMessageBox.critical(self, "Ошибка анализа", error_msg)
    
    # ==================== КЛАСТЕРИЗАЦИЯ ====================
    
    def run_all_algorithms(self):
        if not self.data_loaded:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        self.log_panel.log_message("ЗАПУСК КЛАСТЕРИЗАЦИИ", "purple")
        self.control_panel.set_analyzing_state(True)
        self.process_cancelled = False
        self.results_panel.clear_results()
        self.cluster_visualization.clear()
        self.results = {}
        
        self.algo_worker = AlgorithmRunnerWorker(
            self.tracks_dict,
            debug_enabled=self.is_debug_mode(),
            max_tracks=self.settings.get('max_algorithm_tracks', 300),
            settings=self.settings
        )
        self.algo_worker.progress.connect(self.on_algo_progress)
        self.algo_worker.debug.connect(self.on_algorithm_debug)
        self.algo_worker.algorithm_done.connect(self.on_algorithm_done)
        self.algo_worker.finished.connect(self.on_all_algorithms_finished)
        self.algo_worker.error.connect(self.on_algo_error)
        self.algo_worker.start()
        self.workers.append(self.algo_worker)
    
    def on_algo_progress(self, algorithm, progress):
        if self.is_debug_mode() or progress == 100:
            self.log_panel.log_message(f"{algorithm}: {progress}%", "blue")

    def on_algorithm_debug(self, message):
        self.log_panel.log_message(f"ОТЛАДКА: {message}", "gray")
    
    def on_algorithm_done(self, algorithm, result):
        if self.process_cancelled:
            return
        if 'error' in result:
            self.log_panel.log_message(f"{algorithm}: ОШИБКА", "red")
            return
        self.results[algorithm] = result
        self.results_panel.add_algorithm_result(algorithm, result)
        self.cluster_visualization.set_clusters(
            self.tracks_dict,
            result.get('clusters', {}),
            f"Кластеры алгоритма {algorithm}"
        )
        self.log_panel.log_message(f"{algorithm}: {result['num_clusters']} кластеров, качество={result.get('avg_quality', 0):.1%}", "green")
    
    def on_all_algorithms_finished(self, results):
        if self.process_cancelled:
            self.process_cancelled = False
            self.control_panel.set_cancelled_state(self.data_loaded)
            return
        self.log_panel.log_message("КЛАСТЕРИЗАЦИЯ ЗАВЕРШЕНА", "purple")
        self.control_panel.set_analyzing_state(False)
        self._show_summary()
    
    def on_algo_error(self, error_msg):
        if self.process_cancelled:
            return
        self.log_panel.log_message(f"ОШИБКА: {error_msg}", "red")
        self.control_panel.set_analyzing_state(False)
        self.process_cancelled = False
    
    # ==================== ДЕТЕКТИРОВАНИЕ АНОМАЛИЙ ====================
    
    def detect_anomalies_for_cluster(self):
        if not self.results or 'clusters' not in self.results:
            QMessageBox.warning(self, "Нет результатов", "Сначала выполните анализ")
            return
        
        clusters = self.results.get('clusters', {})
        if not clusters:
            QMessageBox.warning(self, "Нет кластеров", "Не найдено кластеров")
            return
        
        cluster_ids = list(clusters.keys())
        cluster_items = [f"Кластер {cid} ({clusters[cid]['size']} траекторий)" for cid in cluster_ids]
        
        cluster_choice, ok = QInputDialog.getItem(self, "Выбор кластера", "Выберите кластер:", cluster_items, 0, False)
        if not ok:
            return
        
        cluster_idx = cluster_items.index(cluster_choice)
        cluster_id = cluster_ids[cluster_idx]
        
        self.log_panel.log_message(f"Анализ {cluster_choice}...", "blue")
        
        track_ids = clusters[cluster_id]['tracks']
        tracks_in_cluster = [self.tracks_dict[tid] for tid in track_ids if tid in self.tracks_dict]
        
        if not tracks_in_cluster:
            self.log_panel.log_message("Нет треков", "red")
            return
        
        if 'centroid' in clusters[cluster_id]:
            centroid = np.array(clusters[cluster_id]['centroid'])
        else:
            from algorithms.centroid_modeling import CentroidModeler
            modeler = CentroidModeler(use_cpm=False, verbose=False)
            centroid_result = modeler.fit(tracks_in_cluster, use_cpm=False, use_cosine_optimization=True)
            centroid = centroid_result['centroid']
        
        from algorithms.anomaly_detection import AnomalyDetector, RiskAssessor
        from algorithms.pans_ops_risk import PansOpsRiskDetector
        
        detector = AnomalyDetector(use_cosine_metric=True, anomaly_threshold=self.settings.get('anomaly_threshold', 95.0))
        anomaly_results = detector.detect_anomalies(tracks_in_cluster, track_ids, centroid)
        rule_risks = PansOpsRiskDetector().evaluate_tracks(tracks_in_cluster, track_ids)
        
        self.log_panel.log_message("="*50, "purple")
        for r in anomaly_results:
            risk_label = ResultsPanel.RISK_LABELS.get(str(r.risk_level).lower(), str(r.risk_level))
            rule_count = rule_risks.get(r.track_id, {}).get('risk_count', 0)
            if r.is_anomaly:
                self.log_panel.log_message(f"{r.track_id}: скор={r.anomaly_score:.2%}, риск={risk_label}, жестких={rule_count}", "yellow")
            else:
                self.log_panel.log_message(f"{r.track_id}: скор={r.anomaly_score:.2%}, жестких={rule_count}", "green")
        
        risk_assessor = RiskAssessor()
        risk_result = risk_assessor.assess_overall_risk(anomaly_results)
        self.log_panel.log_message(f"ОБЩИЙ РИСК: {risk_result.get('overall_risk', 'неизвестен')}", 
                                  "red" if risk_result.get('risk_score', 0) > 0.7 else "yellow")
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def export_results(self):
        if not self.results:
            QMessageBox.warning(self, "Нет результатов", "Сначала выполните анализ")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для выходных документов")
        if not output_dir:
            return

        report_paths = self.generate_analysis_documents(self.results, output_dir)
        if report_paths:
            QMessageBox.information(
                self,
                "Документы сформированы",
                "Сформированы выходные документы:\n" + "\n".join(str(path) for path in report_paths)
            )
            self.log_panel.log_message(f"Выходные документы сохранены: {output_dir}", "green")
    
    def _serialize_results(self):
        serialized = {}
        for algo, result in self.results.items():
            if 'error' in result:
                serialized[algo] = {'error': result['error']}
            else:
                serialized[algo] = {
                    'num_clusters': result.get('num_clusters', 0),
                    'total_tracks': result.get('total_tracks', 0),
                    'avg_quality': result.get('avg_quality', 0)
                }
        return serialized

    def generate_analysis_documents(self, results: dict, output_dir=None):
        if not results:
            return []

        target_dir = Path(output_dir) if output_dir else self.REPORTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"analysis_report_{timestamp}"
        docx_path = target_dir / f"{base_name}.docx"
        text_path = target_dir / f"{base_name}.txt"
        csv_path = target_dir / f"{base_name}_clusters.csv"
        json_path = target_dir / f"{base_name}.json"
        cluster_rows = self._cluster_report_rows(results)
        report_paths = []

        try:
            from reports.document_builder import build_word_report
            build_word_report(results, cluster_rows, docx_path)
            report_paths.append(docx_path)
        except Exception as error:
            self.log_panel.log_message(f"Word-отчет не сформирован: {error}", "yellow")

        with open(text_path, "w", encoding="utf-8") as file:
            file.write(self._build_analysis_report_text(results))

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([
                "Пучок",
                "Количество траекторий",
                "ВПП",
                "Аномалии",
                "Риски по критериям",
                "Индекс нагрузки W",
                "Уровень риска",
            ])
            for row in cluster_rows:
                csv_row = list(row)
                csv_row[5] = f'="{csv_row[5]}"'
                writer.writerow(csv_row)

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "report_type": "analysis_output_documents",
                    "results": self._build_report_payload(results),
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        report_paths.extend([text_path, csv_path, json_path])
        if self.db_service is not None:
            try:
                report_id = self.db_service.save_report(self.last_run_id, report_paths, results)
                self.log_panel.log_message(f"MongoDB: отчёт сохранён, report_id={report_id}", "green")
            except Exception as error:
                self.log_panel.log_message(f"MongoDB: не удалось сохранить отчёт: {error}", "yellow")
        return report_paths

    def _build_analysis_report_text(self, results: dict) -> str:
        if "statistics" not in results:
            return self._build_algorithm_report_text(results)

        stats = results.get("statistics", {})
        clusters = results.get("clusters", {})
        sectors = results.get("sectors", {})
        anomalies = results.get("anomalies", {})
        runway_stats = stats.get("runway_optimization_stats", {})
        anomaly_stats = stats.get("anomaly_stats", {})
        rule_stats = stats.get("rule_risk_stats", {})
        visible_cluster_ids = set(self._filter_output_clusters(clusters).keys())
        has_cluster_filter = bool(clusters)

        lines = [
            "ОТЧЕТ ПО АНАЛИЗУ ТРАЕКТОРИЙ ВС",
            f"Дата формирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. Сводные показатели",
            f"Всего траекторий: {stats.get('total_tracks', 0)}",
            f"Найдено пучков: {stats.get('total_clusters', len(clusters))}",
            f"Построено 3D-секторов: {stats.get('total_sectors', len(sectors))}",
            f"Центроидов: {stats.get('total_centroids', 0)}",
            f"Время выполнения: {stats.get('execution_time', 0):.1f} сек",
            "",
            "2. Нагрузка и конфликтность",
            f"Максимальный индекс нагрузки W: {stats.get('max_workload_index', 0):.1f}",
            f"Конфликтных событий: {stats.get('total_conflict_events', 0)}",
            "",
            "3. Аномалии и риски",
            f"Аномалий: {anomaly_stats.get('total_anomalies_detected', 0)}",
            f"Доля аномалий: {anomaly_stats.get('anomaly_percentage', 0):.1f}%",
            f"Рисков по жестким критериям: {rule_stats.get('total_rule_risks', 0)}",
            f"Траекторий с рисками: {rule_stats.get('tracks_with_rule_risks', 0)}",
            "",
            "4. ВПП и процедурные данные",
            f"Прибытий по ВПП: {runway_stats.get('arrivals_count', 0)}",
            f"Нарушений интервала: {runway_stats.get('spacing_violations', 0)}",
            f"Перегруженные ВПП: {', '.join(runway_stats.get('overloaded_runways', [])) or 'нет'}",
            f"PBN/RNP замечаний: {stats.get('pbn_issues_count', 0)}",
            f"ARINC-строк: {stats.get('procedure_export_rows', 0)}",
            f"Средняя оценка качества процедуры: {stats.get('avg_procedure_quality', 0):.1f}",
            "",
            "5. Пучки траекторий",
        ]

        for row in self._cluster_report_rows(results):
            lines.append(
                f"Пучок {row[0]}: траекторий {row[1]}, ВПП {row[2]}, "
                f"аномалий {row[3]}, рисков {row[4]}, W={row[5]}, риск {row[6]}"
            )

        lines.extend(["", "6. Ключевые аномалии"])
        for cluster_id, cluster_data in anomalies.items():
            if has_cluster_filter and cluster_id not in visible_cluster_ids:
                continue
            results_list = cluster_data.get("results", [])
            if not results_list:
                continue
            top_items = sorted(results_list, key=lambda item: item.get("anomaly_score", 0), reverse=True)[:5]
            lines.append(f"Пучок {cluster_id}:")
            for item in top_items:
                lines.append(
                    f"  {item.get('track_id', 'н/д')}: "
                    f"оценка {item.get('anomaly_score', 0):.2%}, "
                    f"риск {item.get('risk_level', 'н/д')}, "
                    f"причина: {item.get('risk_description', 'н/д')}"
                )

        return "\n".join(lines) + "\n"

    def _build_algorithm_report_text(self, results: dict) -> str:
        lines = [
            "ОТЧЕТ ПО СРАВНЕНИЮ АЛГОРИТМОВ КЛАСТЕРИЗАЦИИ",
            f"Дата формирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for algorithm, result in results.items():
            if isinstance(result, dict) and "error" in result:
                lines.append(f"{algorithm}: ошибка - {result['error']}")
            elif isinstance(result, dict):
                lines.append(
                    f"{algorithm}: пучков {result.get('num_clusters', 0)}, "
                    f"траекторий {result.get('total_tracks', 0)}, "
                    f"качество {result.get('avg_quality', 0):.1%}"
                )
        return "\n".join(lines) + "\n"

    def _cluster_report_rows(self, results: dict):
        clusters = results.get("clusters", {})
        anomalies = results.get("anomalies", {})
        workload = results.get("workload", {})
        rows = []
        for cluster_id, cluster in clusters.items():
            if self._is_single_without_runway(cluster):
                continue
            anomaly_data = anomalies.get(cluster_id, {})
            anomaly_stats = anomaly_data.get("statistics", {})
            rule_stats = anomaly_data.get("rule_statistics", {})
            workload_data = cluster.get("workload", {}) or workload.get(cluster_id, {})
            rows.append([
                cluster_id,
                cluster.get("size", len(cluster.get("tracks", []))),
                self._cluster_runway_for_report(cluster),
                anomaly_stats.get("anomalies_count", cluster.get("anomalies_count", 0)),
                rule_stats.get("total_rule_risks", cluster.get("rule_risks_count", 0)),
                f"{workload_data.get('workload_index', 0):.1f}",
                cluster.get("risk_level", "н/д"),
            ])
        return rows

    def _filter_output_clusters(self, clusters: dict) -> dict:
        return {
            cluster_id: cluster
            for cluster_id, cluster in (clusters or {}).items()
            if not self._is_single_without_runway(cluster)
        }

    def _is_single_without_runway(self, cluster: dict) -> bool:
        size = cluster.get("size", len(cluster.get("tracks", [])))
        return size <= 1 or self._cluster_runway_for_report(cluster) == "н/д"

    def _build_report_payload(self, results: dict) -> dict:
        if "statistics" not in results:
            return self._make_json_safe(self._serialize_results())

        anomalies = results.get("anomalies", {})
        source_clusters = results.get("clusters", {})
        visible_cluster_ids = set(self._filter_output_clusters(source_clusters).keys())
        has_cluster_filter = bool(source_clusters)
        top_anomalies = []
        for cluster_id, cluster_data in anomalies.items():
            if has_cluster_filter and cluster_id not in visible_cluster_ids:
                continue
            for item in cluster_data.get("results", [])[:10]:
                top_anomalies.append({
                    "cluster_id": cluster_id,
                    "track_id": item.get("track_id"),
                    "anomaly_score": item.get("anomaly_score", 0),
                    "risk_level": item.get("risk_level"),
                    "rule_risk_count": item.get("rule_risk_count", 0),
                    "risk_description": item.get("risk_description", ""),
                })

        return self._make_json_safe({
            "statistics": results.get("statistics", {}),
            "clusters": [
                {
                    "cluster_id": row[0],
                    "tracks_count": row[1],
                    "runway": row[2],
                    "anomalies": row[3],
                    "rule_risks": row[4],
                    "workload_index": row[5],
                    "risk_level": row[6],
                }
                for row in self._cluster_report_rows(results)
            ],
            "top_anomalies": top_anomalies[:50],
        })

    def _cluster_runway_for_report(self, cluster: dict) -> str:
        counts = {}
        for track_id in cluster.get("tracks", []):
            track = self.tracks_dict.get(track_id)
            if track is None:
                continue
            runway = str(getattr(track, "runwayname", "") or "").strip()
            if runway and runway.lower() not in {"nan", "none", "null"}:
                counts[runway] = counts.get(runway, 0) + 1
        return max(counts, key=counts.get) if counts else "н/д"

    def _make_json_safe(self, value):
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._make_json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def show_track_filter_dialog(self):
        if not self.data_loaded or not self.tracks_dict:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return

        dialog = TrackFilterDialog(self.tracks_dict, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        selected_ids = dialog.selected_track_ids
        self.cluster_visualization.set_selected_tracks(selected_ids)

        if not self.cluster_visualization.clusters:
            self.cluster_visualization.set_clusters(
                self.tracks_dict,
                {
                    "выбор": {
                        "tracks": selected_ids,
                        "size": len(selected_ids),
                        "algorithm": "Фильтр"
                    }
                },
                "Выбранные траектории ВС"
            )
            self.cluster_visualization.set_selected_tracks(selected_ids)

        self.statusBar().showMessage(f"Выбрано траекторий ВС: {len(selected_ids)}")

    def clear_track_selection(self):
        self.cluster_visualization.set_selected_tracks([])
        self.statusBar().showMessage("Выбор траекторий сброшен")
    
    def show_algorithm_details(self, algorithm):
        if algorithm not in self.results:
            return
        result = self.results[algorithm]
        if 'error' in result:
            self.results_panel.set_details_text(f"Ошибка: {result['error']}")
        else:
            self.results_panel.show_algorithm_details(algorithm, result)
            self.cluster_visualization.set_clusters(
                self.tracks_dict,
                result.get('clusters', {}),
                f"Кластеры алгоритма {algorithm}"
            )
    
    def show_results_history(self):
        if not self.results_history:
            QMessageBox.information(self, "История", "История пуста")
            return
        text = "История анализов:\n\n"
        for i, entry in enumerate(self.results_history[:10]):
            text += f"{i+1}. {entry['date']} - {entry.get('summary', '')}\n"
        QMessageBox.information(self, "История", text)
    
    def clear_results_history(self):
        self.results_history = []
        if self.db_service is not None:
            try:
                self.db_service.clear_history()
                self.log_panel.log_message("История очищена в MongoDB", "green")
                return
            except Exception as e:
                self.log_panel.log_message(f"MongoDB: ошибка очистки истории: {e}", "yellow")
        try:
            with open(self.RESULTS_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
            self.log_panel.log_message("История очищена", "green")
        except Exception as e:
            self.log_panel.log_message(f"Ошибка: {e}", "red")
    
    def show_algorithm_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки алгоритмов анализа")
        dialog.setModal(True)
        dialog.resize(560, 520)
        layout = QVBoxLayout(dialog)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        controls = {}

        def spin(name, value, min_value, max_value, step=1):
            widget = QSpinBox()
            widget.setRange(min_value, max_value)
            widget.setSingleStep(step)
            widget.setValue(int(value))
            controls[name] = widget
            return widget

        def dspin(name, value, min_value, max_value, step=0.01, decimals=3):
            widget = QDoubleSpinBox()
            widget.setRange(min_value, max_value)
            widget.setSingleStep(step)
            widget.setDecimals(decimals)
            widget.setValue(float(value))
            controls[name] = widget
            return widget

        def checkbox(name, value):
            widget = QCheckBox()
            widget.setChecked(bool(value))
            controls[name] = widget
            return widget

        def combo(name, value, items):
            widget = QComboBox()
            widget.addItems(items)
            if value in items:
                widget.setCurrentText(value)
            controls[name] = widget
            return widget

        general_tab = QWidget()
        general_form = QFormLayout(general_tab)
        general_form.addRow(
            "Максимум траекторий для сравнения:",
            spin("max_algorithm_tracks", self.settings.get("max_algorithm_tracks", 300), 100, 10000, 100)
        )
        general_form.addRow(
            "Порог аномалий, процентиль:",
            dspin("anomaly_threshold", self.settings.get("anomaly_threshold", 95.0), 50.0, 99.9, 0.5, 1)
        )
        general_form.addRow(
            "Итераций CPM:",
            spin("cpm_iterations", self.settings.get("cpm_iterations", 20), 5, 100, 5)
        )
        tabs.addTab(general_tab, "Общие")

        ransac_tab = QWidget()
        ransac_form = QFormLayout(ransac_tab)
        ransac_form.addRow(
            "Порог направления:",
            dspin("ransac_distance_threshold", self.settings.get("ransac_distance_threshold", self.settings.get("ransac_threshold", 0.05)), 0.01, 0.5, 0.01, 3)
        )
        ransac_form.addRow(
            "Итераций RANSAC:",
            spin("ransac_max_iterations", self.settings.get("ransac_max_iterations", 1000), 50, 10000, 50)
        )
        ransac_form.addRow(
            "Порог расстояния до линии:",
            dspin("ransac_line_distance_threshold", self.settings.get("ransac_line_distance_threshold", 0.5), 0.05, 5.0, 0.05, 3)
        )
        ransac_form.addRow(
            "Итераций подбора линии:",
            spin("ransac_max_line_iterations", self.settings.get("ransac_max_line_iterations", 300), 20, 5000, 20)
        )
        ransac_form.addRow(
            "Минимум ВС в пучке:",
            spin("ransac_min_cluster_tracks", self.settings.get("ransac_min_cluster_tracks", 2), 1, 100, 1)
        )
        ransac_form.addRow(
            "MLESAC доля выбросов:",
            dspin("mlesac_outlier_ratio", self.settings.get("mlesac_outlier_ratio", 0.3), 0.0, 0.9, 0.05, 2)
        )
        ransac_form.addRow(
            "MLESAC вероятность выброса:",
            dspin("mlesac_outlier_prob", self.settings.get("mlesac_outlier_prob", 0.1), 0.01, 1.0, 0.01, 2)
        )
        tabs.addTab(ransac_tab, "RANSAC/MLESAC")

        mean_shift_tab = QWidget()
        mean_shift_form = QFormLayout(mean_shift_tab)
        mean_shift_form.addRow(
            "Bandwidth (0 = авто):",
            dspin("mean_shift_bandwidth", self.settings.get("mean_shift_bandwidth", 0.0), 0.0, 5.0, 0.05, 3)
        )
        mean_shift_form.addRow(
            "Итераций Mean Shift:",
            spin("mean_shift_max_iterations", self.settings.get("mean_shift_max_iterations", 300), 50, 3000, 50)
        )
        mean_shift_form.addRow(
            "Минимум ВС в пучке:",
            spin("mean_shift_min_cluster_size", self.settings.get("mean_shift_min_cluster_size", 3), 1, 100, 1)
        )
        mean_shift_form.addRow(
            "Ускоренный старт по корзинам:",
            checkbox("mean_shift_bin_seeding", self.settings.get("mean_shift_bin_seeding", True))
        )
        tabs.addTab(mean_shift_tab, "Mean Shift")

        spectral_tab = QWidget()
        spectral_form = QFormLayout(spectral_tab)
        spectral_form.addRow(
            "SCC число кластеров (0 = авто):",
            spin("scc_n_clusters", self.settings.get("scc_n_clusters", 0), 0, 100, 1)
        )
        spectral_form.addRow(
            "SCC gamma:",
            dspin("scc_gamma", self.settings.get("scc_gamma", 1.0), 0.01, 10.0, 0.1, 2)
        )
        spectral_form.addRow(
            "SCC порог сходства:",
            dspin("scc_affinity_threshold", self.settings.get("scc_affinity_threshold", 0.5), 0.0, 1.0, 0.05, 2)
        )
        spectral_form.addRow(
            "SCC точный расчет до N ВС:",
            spin("scc_max_exact_tracks", self.settings.get("scc_max_exact_tracks", 700), 100, 5000, 100)
        )
        spectral_form.addRow(
            "LSCC число кластеров (0 = авто):",
            spin("lscc_n_clusters", self.settings.get("lscc_n_clusters", 0), 0, 100, 1)
        )
        spectral_form.addRow(
            "LSCC масштабов:",
            spin("lscc_n_scales", self.settings.get("lscc_n_scales", 3), 1, 10, 1)
        )
        spectral_form.addRow(
            "LSCC метрика:",
            combo("lscc_affinity_metric", self.settings.get("lscc_affinity_metric", "hybrid"), ["hybrid", "curvature", "spatial"])
        )
        spectral_form.addRow(
            "Нормализованный лапласиан:",
            checkbox("lscc_use_normalized_laplacian", self.settings.get("lscc_use_normalized_laplacian", True))
        )
        tabs.addTab(spectral_tab, "SCC/LSCC")

        btn = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn.accepted.connect(dialog.accept)
        btn.rejected.connect(dialog.reject)
        layout.addWidget(btn)
        
        if dialog.exec_():
            for name, widget in controls.items():
                if isinstance(widget, QDoubleSpinBox):
                    self.settings[name] = widget.value()
                elif isinstance(widget, QSpinBox):
                    self.settings[name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    self.settings[name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    self.settings[name] = widget.currentText()
            self.settings['ransac_threshold'] = self.settings.get('ransac_distance_threshold', 0.05)
            self.save_settings()
            self.log_panel.log_message("Настройки сохранены", "green")
    
    def show_display_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки отображения")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Максимум кластеров на схеме:"))
        visual_max_clusters = QSpinBox()
        visual_max_clusters.setRange(1, 50)
        visual_max_clusters.setValue(self.settings.get("visual_max_clusters", 10))
        layout.addWidget(visual_max_clusters)

        self.show_grid_cb = QCheckBox("Показывать сетку")
        self.show_grid_cb.setChecked(True)
        layout.addWidget(self.show_grid_cb)

        btn = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn.accepted.connect(dialog.accept)
        btn.rejected.connect(dialog.reject)
        layout.addWidget(btn)
        if dialog.exec_():
            self.settings["visual_max_clusters"] = visual_max_clusters.value()
            self.save_settings()
            self.apply_settings()
            self.log_panel.log_message("Настройки отображения сохранены", "green")
    
    def show_about(self):
        QMessageBox.about(self, "О программе", "<h3>Анализ траекторий ВС</h3><p>Версия 3.0</p>")
    
    def show_help(self):
        QMessageBox.information(self, "Помощь", "1. Загрузите данные\n2. Выберите 'Полный анализ'\n3. Просмотрите результаты")
    
    def _show_summary(self):
        if not self.results:
            return
        summary = "\n" + "="*50 + "\n"
        for algo, result in self.results.items():
            if 'error' not in result:
                summary += f"{algo}: {result.get('num_clusters', 0)} кластеров\n"
        self.log_panel.log_message(summary, "purple")
    
    def closeEvent(self, event):
        self.save_settings()
        for worker in self.workers:
            if worker.isRunning():
                worker.cancel()
                worker.wait(1000)
        event.accept()
