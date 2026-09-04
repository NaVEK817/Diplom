"""Панель лога выполнения"""

from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTextEdit,
                             QPushButton, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor


class LogPanel(QWidget):
    """Панель с логом выполнения"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок с кнопкой очистки
        header_layout = QHBoxLayout()
        title = QLabel("Лог выполнения")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        btn_clear = QPushButton("Очистить лог")
        btn_clear.clicked.connect(self.clear_log)
        header_layout.addWidget(btn_clear)
        
        layout.addLayout(header_layout)
        
        # Текстовое поле для лога
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_text)
    
    def log_message(self, message: str, color: str = "black"):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colored_msg = f'<span style="color:{color};">[{timestamp}] {message}</span>'
        self.log_text.append(colored_msg)
        
        # Прокрутка вниз
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
    
    def clear_log(self):
        """Очистка лога"""
        self.log_text.clear()