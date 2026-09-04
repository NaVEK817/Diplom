"""Панель отображения результатов кластеризации"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget,
                             QTableWidgetItem, QTextEdit, QHeaderView,
                             QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont


class ResultsPanel(QWidget):
    """Панель с таблицей результатов и деталями"""

    RISK_LABELS = {
        'low': 'Низкий',
        'medium': 'Средний',
        'high': 'Высокий',
        'unknown': 'Неизвестен',
        'низкий': 'Низкий',
        'средний': 'Средний',
        'высокий': 'Высокий',
        'неизвестен': 'Неизвестен',
    }

    RISK_COLORS = {
        'low': QColor(0, 128, 0),
        'medium': QColor(255, 165, 0),
        'high': QColor(255, 0, 0),
        'низкий': QColor(0, 128, 0),
        'средний': QColor(255, 165, 0),
        'высокий': QColor(255, 0, 0),
    }
    
    algorithm_selected = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Результаты кластеризации")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Таблица результатов
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self._set_algorithm_headers()
        self._configure_results_table()
        self.results_table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.results_table)
        
        # Детальная информация
        layout.addWidget(QLabel("Детальная информация по алгоритму:"))
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        layout.addWidget(self.details_text)

    def _configure_results_table(self):
        self.results_table.setWordWrap(True)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.verticalHeader().setDefaultSectionSize(26)
        header = self.results_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setMinimumHeight(44)
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def _set_algorithm_headers(self):
        self.results_table.setHorizontalHeaderLabels([
            "Алго-\nритм", "Клас-\nтеров", "Траек-\nторий",
            "Ср.\nразмер", "Каче-\nство", "Время\n(с)"
        ])

    def _set_full_analysis_headers(self):
        self.results_table.setHorizontalHeaderLabels([
            "Пучок", "Траек-\nторий", "Ано-\nмалий",
            "Риск/\nжестк.", "Нагрузка\nW", "Пик\nВС"
        ])
    
    def clear_results(self):
        """Очистка таблицы результатов"""
        self.results_table.setRowCount(0)
        self.details_text.clear()
    
    def add_algorithm_result(self, algorithm: str, result: dict):
        """Добавление результата алгоритма в таблицу"""
        row = self.results_table.rowCount()
        if row == 0:
            self._set_algorithm_headers()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(algorithm))
        self.results_table.setItem(row, 1, QTableWidgetItem(str(result['num_clusters'])))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(result['total_tracks'])))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{result.get('avg_size', 0):.1f}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{result.get('avg_quality', 0):.1%}"))
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{result.get('time', 0):.1f}"))
    
    def show_algorithm_details(self, algorithm: str, result: dict):
        """Отображение детальной информации по алгоритму"""
        clusters = result['clusters']
        
        text = f"=== {algorithm} ===\n\n"
        text += f"Всего кластеров: {len(clusters)}\n"
        text += f"Всего траекторий в кластерах: {result['total_tracks']}\n\n"
        
        text += "Кластеры:\n"
        text += "-" * 40 + "\n"
        
        for cluster_id, cluster_data in clusters.items():
            text += f"Кластер {cluster_id}: {cluster_data['size']} траекторий"
            if 'quality' in cluster_data:
                text += f", качество={cluster_data['quality']:.1%}"
            text += "\n"
        
        # Добавляем информацию об опорных траекториях
        text += "\nОпорные траектории (направления):\n"
        for cluster_id, cluster_data in clusters.items():
            ref = cluster_data.get('reference')
            if ref is not None and len(ref) > 0:
                if isinstance(ref, np.ndarray):
                    text += f"Кластер {cluster_id}: [{ref[0]:.3f}, {ref[1]:.3f}, {ref[2]:.3f}]\n"
                else:
                    text += f"Кластер {cluster_id}: {ref}\n"
        
        self.details_text.setText(text)
    
    def set_details_text(self, text: str):
        """Установка текста в деталях"""
        self.details_text.setText(text)
    
    def on_selection_changed(self):
        """Обработчик выбора строки в таблице"""
        selected = self.results_table.selectedItems()
        if selected:
            row = selected[0].row()
            algorithm = self.results_table.item(row, 0).text()
            self.algorithm_selected.emit(algorithm)

    def display_full_results(self, clusters: dict, anomalies: dict, risks: dict, runway_optimization: dict = None):
        """Отображение полных результатов анализа"""
        self.clear_results()
        self._set_full_analysis_headers()
        sector_lines = []
    
        for cluster_id, cluster_data in clusters.items():
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
        
            # Название кластера
            self.results_table.setItem(row, 0, QTableWidgetItem(f"Пучок {cluster_id}"))
        
            # Размер кластера
            size = cluster_data.get('size', 0)
            self.results_table.setItem(row, 1, QTableWidgetItem(str(size)))
        
            # Количество аномалий
            anomaly_count = anomalies.get(cluster_id, {}).get('statistics', {}).get('anomalies_count', 0)
            self.results_table.setItem(row, 2, QTableWidgetItem(str(anomaly_count)))
        
            # Уровень риска
            risk_level = risks.get(cluster_id, {}).get('risk_level', 'неизвестен')
            risk_key = str(risk_level).lower()
            risk_item = QTableWidgetItem(self.RISK_LABELS.get(risk_key, str(risk_level)))
            risk_color = self.RISK_COLORS.get(risk_key)
            if risk_color is not None:
                risk_item.setForeground(risk_color)
            rule_stats = anomalies.get(cluster_id, {}).get('rule_statistics', {})
            rule_risks_count = rule_stats.get('total_rule_risks', 0)
            risk_item.setText(f"{risk_item.text()} / {rule_risks_count}")
            if rule_stats.get('high_severity_tracks', 0) > 0:
                risk_item.setForeground(QColor(255, 0, 0))
            elif rule_risks_count > 0:
                risk_item.setForeground(QColor(255, 165, 0))
            self.results_table.setItem(row, 3, risk_item)
        
            # Рекомендация
            sector = cluster_data.get('sector', {})
            workload = cluster_data.get('workload', {})
            pbn = cluster_data.get('pbn', {})
            quality = cluster_data.get('procedure_quality', {})
            entry_count = len(sector.get('entry_fixes', []))
            exit_count = len(sector.get('exit_fixes', []))
            workload_index = workload.get('workload_index', 0.0)
            peak_load = workload.get('peak_load', 0)
            workload_item = QTableWidgetItem(f"{workload_index:.1f}")
            if workload_index >= 80:
                workload_item.setForeground(QColor(255, 0, 0))
            elif workload_index >= 40:
                workload_item.setForeground(QColor(255, 165, 0))
            else:
                workload_item.setForeground(QColor(0, 128, 0))
            self.results_table.setItem(row, 4, workload_item)
            self.results_table.setItem(row, 5, QTableWidgetItem(str(peak_load)))

            floor = sector.get('floor_ft')
            ceiling = sector.get('ceiling_ft')

            if sector:
                floor_text = f"{floor:.0f}" if floor is not None else "нет"
                ceiling_text = f"{ceiling:.0f}" if ceiling is not None else "нет"
                risk_type_counts = rule_stats.get('risk_type_counts', {})
                risk_types_text = ", ".join(
                    f"{risk_type}: {count}" for risk_type, count in risk_type_counts.items()
                ) or "нет"
                sector_lines.append(
                    f"Пучок {cluster_id}: сектор={sector.get('method', 'неизвестно')}, "
                    f"floor={floor_text} ft, ceiling={ceiling_text} ft, "
                    f"входов={entry_count}, выходов={exit_count}, "
                    f"буфер={sector.get('safety_buffer_ft', 0):.0f} ft, "
                    f"W={workload_index:.1f}, пик={peak_load} ВС, "
                    f"конфликтов={workload.get('conflict_events_count', 0)}, "
                    f"жестких рисков={rule_risks_count} ({risk_types_text}), "
                    f"PBN замечаний={pbn.get('issues_count', 0)}, RF={pbn.get('rf_segments_count', 0)}, "
                    f"качество={quality.get('quality_score', 0):.1f}, "
                    f"стабильность={quality.get('stable_approach_percent', 0):.1f}%, "
                    f"средняя сложность={workload.get('avg_maneuver_complexity', 0):.2f}, "
                    f"среднее сопровождение={workload.get('avg_dwell_time_seconds', 0):.0f} сек, "
                    f"CSG={sector.get('csg_status', 'нет данных')}"
                )

        if sector_lines:
            details = "Управляемые 3D-секторы и нагрузка диспетчера:\n" + "\n".join(sector_lines)
            runway_text = self._runway_optimization_text(runway_optimization or {})
            if runway_text:
                details += "\n\nОптимизация ВПП:\n" + runway_text
            self.details_text.setText(details)

    def _runway_optimization_text(self, runway_optimization: dict) -> str:
        if not runway_optimization:
            return ""

        lines = []
        summary = runway_optimization.get('summary', {})
        lines.append(
            f"Прибытий={summary.get('arrivals_count', 0)}, "
            f"нарушений интервала={summary.get('spacing_violations', 0)}, "
            f"суммарная задержка={summary.get('total_delay_seconds', 0):.0f} сек"
        )

        for runway, data in runway_optimization.get('runways', {}).items():
            lines.append(
                f"ВПП {runway}: прибытий={data.get('arrivals_count', 0)}, "
                f"пик/час={data.get('landings_per_hour', 0)}, "
                f"использование={data.get('utilization', 0):.0%}, "
                f"нарушений={data.get('violations_count', 0)}, "
                f"ср. задержка={data.get('avg_delay_seconds', 0):.0f} сек"
            )

        reassignment = runway_optimization.get('reassignment', {})
        for plan in reassignment.get('plans', []):
            lines.append(
                f"Перенос {plan.get('source_runway')} -> {plan.get('target_runway')}: "
                f"{plan.get('moved_count', 0)} ВС"
            )

        tbo_count = len(runway_optimization.get('tbo_routes', []))
        if tbo_count:
            lines.append(f"TBO-маршрутов для восстановления интервала: {tbo_count}")
        return "\n".join(lines)
