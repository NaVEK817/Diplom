from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QProgressBar, QLabel
from PyQt5.QtCore import pyqtSignal


class ControlPanel(QGroupBox):
    """Панель управления с выбором количества файлов и кнопкой запуска анализа"""
    
    analyze_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__("Управление анализом")
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Информационная метка
        self.info_label = QLabel("Данные не загружены")
        self.info_label.setStyleSheet("color: #e74c3c;")
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMaximumWidth(200)
        layout.addWidget(self.progress_bar)
        
        # Кнопка запуска
        self.analyze_btn = QPushButton("Запустить анализ кластеров")
        self.analyze_btn.setStyleSheet("background-color: #2980b9; color: white; padding: 5px 10px;")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_requested.emit)
        layout.addWidget(self.analyze_btn)

        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.setStyleSheet("background-color: #c0392b; color: white; padding: 5px 10px;")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.cancel_btn)
    
    def update_progress(self, value: int):
        """Обновление прогресс-бара"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
    
    def set_loading_state(self, loading: bool):
        """Установка состояния загрузки"""
        if loading:
            self.progress_bar.setVisible(True)
            self.cancel_btn.setVisible(True)
            self.info_label.setText("Загрузка данных...")
            self.info_label.setStyleSheet("color: #f39c12;")
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            self.cancel_btn.setVisible(False)
    
    def set_data_loaded(self, loaded: bool):
        """Установка состояния наличия данных"""
        self.analyze_btn.setEnabled(loaded)
        if loaded:
            self.info_label.setText("Данные загружены")
            self.info_label.setStyleSheet("color: #27ae60;")
        else:
            self.info_label.setText("Данные не загружены")
            self.info_label.setStyleSheet("color: #e74c3c;")
    
    def set_analyzing_state(self, analyzing: bool):
        """Установка состояния анализа"""
        self.analyze_btn.setEnabled(not analyzing)
        self.analyze_btn.setText("Анализ..." if analyzing else "Запустить анализ кластеров")
        self.cancel_btn.setVisible(analyzing)
        if analyzing:
            self.progress_bar.setVisible(True)
            self.info_label.setText("Выполняется анализ...")
            self.info_label.setStyleSheet("color: #3498db;")
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            self.info_label.setText("Анализ завершен")
            self.info_label.setStyleSheet("color: #27ae60;")

    def set_cancelled_state(self, data_loaded: bool):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.cancel_btn.setVisible(False)
        self.analyze_btn.setEnabled(data_loaded)
        self.analyze_btn.setText("Запустить анализ кластеров")
        self.info_label.setText("Процесс отменен")
        self.info_label.setStyleSheet("color: #c0392b;")
