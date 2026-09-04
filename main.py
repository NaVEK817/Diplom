#!/usr/bin/env python3
"""
Интегрированная система анализа траекторий TRACON
Точка входа в приложение
"""

import os
import sys

# === БЛОК ИНИЦИАЛИЗАЦИИ QT ===
try:
    import PyQt5
    pyqt_path = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(pyqt_path, 'Qt5', 'plugins', 'platforms')
    if not os.path.exists(plugin_path):
        plugin_path = os.path.join(pyqt_path, 'Qt', 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Главная функция запуска приложения"""
    app = QApplication(sys.argv)
    app.setApplicationName("Анализ траекторий TRACON")
    app.setOrganizationName("Авиационная аналитика")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
