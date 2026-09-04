"""Диалог выбора траекторий по параметрам ВС и маршрута."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class MultiSelectCombo(QComboBox):
    """Выпадающий список с множественным выбором."""

    def __init__(self, placeholder="Все", parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText(placeholder)
        self.view().pressed.connect(self._toggle_item)

    def add_check_item(self, text, value):
        item = QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(value, Qt.UserRole)
        item.setCheckState(Qt.Unchecked)
        self.model().appendRow(item)

    def selected_values(self):
        values = set()
        for row in range(self.model().rowCount()):
            item = self.model().item(row)
            if item and item.checkState() == Qt.Checked:
                values.add(item.data(Qt.UserRole))
        return values

    def clear_selection(self):
        self.model().blockSignals(True)
        for row in range(self.model().rowCount()):
            item = self.model().item(row)
            if item:
                item.setCheckState(Qt.Unchecked)
        self.model().blockSignals(False)
        self._update_text()

    def _toggle_item(self, index):
        item = self.model().itemFromIndex(index)
        if not item:
            return
        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
        self._update_text()

    def _update_text(self):
        selected = [
            self.model().item(row).text()
            for row in range(self.model().rowCount())
            if self.model().item(row).checkState() == Qt.Checked
        ]
        if not selected:
            self.lineEdit().setText("")
        elif len(selected) <= 2:
            self.lineEdit().setText(", ".join(selected))
        else:
            self.lineEdit().setText(f"Выбрано: {len(selected)}")


class TrackFilterDialog(QDialog):
    """Фильтрует треки по доступным полям TRACON."""

    def __init__(self, tracks_dict: dict, parent=None):
        super().__init__(parent)
        self.tracks_dict = tracks_dict or {}
        self.selected_track_ids = []
        self.setWindowTitle("Выбор траекторий ВС")
        self.resize(760, 560)
        self._init_ui()
        self._refresh_results()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Позывной, номер трека, владелец, тип ВС")
        form.addRow("Поиск:", self.search_edit)

        self.aircraft_combo = self._multi_combo("acid")
        self.type_combo = self._multi_combo("aircraft_category", fallback="aircrafttype")
        self.owner_combo = self._multi_combo("owner_name")
        self.airport_combo = self._multi_combo("airportid")
        self.other_port_combo = self._multi_combo("other_port")
        self.runway_combo = self._multi_combo("runwayname")

        form.addRow("ВС / позывной:", self.aircraft_combo)
        form.addRow("Тип / категория:", self.type_combo)
        form.addRow("Авиакомпания / владелец:", self.owner_combo)
        form.addRow("Аэропорт:", self.airport_combo)
        form.addRow("Город / порт прибытия или удаления:", self.other_port_combo)
        form.addRow("ВПП:", self.runway_combo)
        layout.addLayout(form)

        actions = QHBoxLayout()
        apply_btn = QPushButton("Применить фильтр")
        reset_btn = QPushButton("Сбросить")
        select_all_btn = QPushButton("Выбрать все найденные")
        actions.addWidget(apply_btn)
        actions.addWidget(reset_btn)
        actions.addStretch()
        actions.addWidget(select_all_btn)
        layout.addLayout(actions)

        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.results_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        apply_btn.clicked.connect(self._refresh_results)
        reset_btn.clicked.connect(self._reset_filters)
        select_all_btn.clicked.connect(self.results_list.selectAll)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

    def _multi_combo(self, field, fallback=None):
        combo = MultiSelectCombo()
        values = set()
        for track in self.tracks_dict.values():
            for name in [field, fallback]:
                if not name:
                    continue
                value = getattr(track, name, None)
                value = self._clean(value)
                if value:
                    values.add(value)
        for value in sorted(values):
            combo.add_check_item(value, value)
        return combo

    def _reset_filters(self):
        self.search_edit.clear()
        for combo in [
            self.aircraft_combo,
            self.type_combo,
            self.owner_combo,
            self.airport_combo,
            self.other_port_combo,
            self.runway_combo,
        ]:
            combo.clear_selection()
        self._refresh_results()

    def _refresh_results(self):
        self.results_list.clear()
        matches = self._matching_track_ids()
        for track_id in matches[:1000]:
            track = self.tracks_dict[track_id]
            item = QListWidgetItem(self._track_label(track_id, track))
            item.setData(Qt.UserRole, track_id)
            self.results_list.addItem(item)
        suffix = " Показаны первые 1000." if len(matches) > 1000 else ""
        self.summary_label.setText(f"Найдено траекторий: {len(matches)}.{suffix}")

    def _matching_track_ids(self):
        filters = {
            "acid": self.aircraft_combo.selected_values(),
            "type": self.type_combo.selected_values(),
            "owner_name": self.owner_combo.selected_values(),
            "airportid": self.airport_combo.selected_values(),
            "other_port": self.other_port_combo.selected_values(),
            "runwayname": self.runway_combo.selected_values(),
        }
        search = self.search_edit.text().strip().lower()
        matches = []
        for track_id, track in self.tracks_dict.items():
            if not self._matches_any(filters["acid"], self._clean(getattr(track, "acid", None))):
                continue
            if filters["type"] and not filters["type"].intersection({
                self._clean(getattr(track, "aircraft_category", None)),
                self._clean(getattr(track, "aircrafttype", None)),
            }):
                continue
            for field in ["owner_name", "airportid", "other_port", "runwayname"]:
                if not self._matches_any(filters[field], self._clean(getattr(track, field, None))):
                    break
            else:
                if search and search not in self._search_blob(track_id, track):
                    continue
                matches.append(track_id)
        return matches

    def _accept(self):
        selected = [item.data(Qt.UserRole) for item in self.results_list.selectedItems()]
        self.selected_track_ids = selected or self._matching_track_ids()
        self.accept()

    def _track_label(self, track_id, track):
        return (
            f"{track_id} | ВС: {self._clean(getattr(track, 'acid', None)) or 'н/д'} | "
            f"тип: {self._clean(getattr(track, 'aircraft_category', None)) or self._clean(getattr(track, 'aircrafttype', None)) or 'н/д'} | "
            f"владелец: {self._clean(getattr(track, 'owner_name', None)) or 'н/д'} | "
            f"порт: {self._clean(getattr(track, 'airportid', None)) or 'н/д'} / {self._clean(getattr(track, 'other_port', None)) or 'н/д'} | "
            f"ВПП: {self._clean(getattr(track, 'runwayname', None)) or 'н/д'}"
        )

    def _search_blob(self, track_id, track):
        values = [
            track_id,
            getattr(track, "acid", None),
            getattr(track, "aircrafttype", None),
            getattr(track, "aircraft_category", None),
            getattr(track, "owner_name", None),
            getattr(track, "airportid", None),
            getattr(track, "other_port", None),
            getattr(track, "runwayname", None),
        ]
        return " ".join(self._clean(value).lower() for value in values)

    def _matches_any(self, selected_values, value):
        return not selected_values or value in selected_values

    def _clean(self, value):
        text = str(value or "").strip()
        return "" if text.lower() in {"nan", "none", "null"} else text
