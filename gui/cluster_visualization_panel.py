"""Панель визуализации кластеров."""

from PyQt5.QtCore import QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QCheckBox, QDialog, QPushButton, QToolTip, QVBoxLayout, QWidget
import math
import numpy as np


class ClusterVisualizationPanel(QWidget):
    """Рисует траектории кластеров в 3D-виде и основных проекциях."""

    COLORS = [
        QColor("#1f77b4"),
        QColor("#ff7f0e"),
        QColor("#2ca02c"),
        QColor("#d62728"),
        QColor("#9467bd"),
        QColor("#8c564b"),
        QColor("#17becf"),
        QColor("#7f7f7f"),
    ]

    def __init__(self, show_fullscreen_button: bool = True):
        super().__init__()
        self.tracks = {}
        self.clusters = {}
        self.raw_clusters = {}
        self.title = "Визуализация кластеров"
        self.selected_track_ids = set()
        self.max_draw_tracks_per_cluster = 80
        self.max_scatter_points_per_cluster = 450
        self.max_display_clusters = 10
        self.show_routes = True
        self.show_clusters = True
        self.show_responsibility_zones = True
        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)
        self.setObjectName("ClusterVisualizationPanel")
        self.fullscreen_button = None
        if show_fullscreen_button:
            self.fullscreen_button = QPushButton("⛶", self)
            self.fullscreen_button.setToolTip("Показать визуализацию на весь экран")
            self.fullscreen_button.setFixedSize(34, 30)
            self.fullscreen_button.setStyleSheet(
                "QPushButton { background: rgba(255,255,255,220); border: 1px solid #94a3b8; "
                "border-radius: 4px; font-weight: bold; color: #0f172a; }"
                "QPushButton:hover { background: #e0f2fe; border-color: #0284c7; }"
            )
            self.fullscreen_button.clicked.connect(self.open_fullscreen_view)
        self.legend_checkboxes = []
        self._create_legend_checkboxes()

    def sizeHint(self):
        return QSize(800, 600)

    def clear(self):
        self.tracks = {}
        self.clusters = {}
        self.raw_clusters = {}
        self.title = "Визуализация кластеров"
        self.selected_track_ids = set()
        self.update()

    def set_clusters(self, tracks: dict, clusters: dict, title: str = "Визуализация кластеров"):
        self.tracks = tracks or {}
        self.raw_clusters = self._filter_display_clusters(clusters or {})
        self.clusters = self._merge_small_clusters_for_display(self.raw_clusters)
        self.title = title
        self.update()

    def set_max_display_clusters(self, value: int):
        self.max_display_clusters = max(1, int(value or 10))
        self.clusters = self._merge_small_clusters_for_display(self.raw_clusters)
        self.update()

    def set_selected_tracks(self, track_ids):
        self.selected_track_ids = set(track_ids or [])
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fullscreen_button is not None:
            self.fullscreen_button.move(self.width() - self.fullscreen_button.width() - 12, 10)
        self._position_legend_checkboxes()

    def _create_legend_checkboxes(self):
        options = [
            ("Маршруты", "show_routes"),
            ("Кластеры", "show_clusters"),
            ("Зоны ответственности", "show_responsibility_zones"),
        ]
        style = (
            "QCheckBox { background: rgba(248,250,252,235); color: #0f172a; "
            "font-size: 10pt; font-weight: 600; padding: 2px 6px; }"
        )
        for text, attr_name in options:
            checkbox = QCheckBox(text, self)
            checkbox.setChecked(getattr(self, attr_name))
            checkbox.setStyleSheet(style)
            checkbox.stateChanged.connect(lambda _state, name=attr_name: self._toggle_legend_item(name))
            self.legend_checkboxes.append(checkbox)
        self._position_legend_checkboxes()

    def _position_legend_checkboxes(self):
        if not self.legend_checkboxes:
            return
        x = 32
        y = max(0, self.height() - 126)
        for checkbox in self.legend_checkboxes:
            checkbox.adjustSize()
            checkbox.move(x, y)
            x += checkbox.width() + 14

    def _toggle_legend_item(self, attr_name):
        sender = self.sender()
        if sender is not None:
            setattr(self, attr_name, bool(sender.isChecked()))
            self.update()

    def _sync_checkbox_states(self):
        attrs = ("show_routes", "show_clusters", "show_responsibility_zones")
        for checkbox, attr_name in zip(self.legend_checkboxes, attrs):
            checkbox.setChecked(getattr(self, attr_name))

    def open_fullscreen_view(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.title)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Window)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        fullscreen_panel = ClusterVisualizationPanel(show_fullscreen_button=False)
        fullscreen_panel.set_max_display_clusters(self.max_display_clusters)
        fullscreen_panel.show_routes = self.show_routes
        fullscreen_panel.show_clusters = self.show_clusters
        fullscreen_panel.show_responsibility_zones = self.show_responsibility_zones
        fullscreen_panel._sync_checkbox_states()
        fullscreen_panel.set_clusters(self.tracks, self.raw_clusters, self.title)
        fullscreen_panel.set_selected_tracks(self.selected_track_ids)
        layout.addWidget(fullscreen_panel)

        dialog.showFullScreen()
        dialog.exec_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        self._draw_title(painter)

        bounds = self._bounds()
        if not self.clusters or bounds is None:
            self._draw_empty_state(painter, self.rect().adjusted(48, 52, -48, -48))
            return

        plot_rect = self._main_plot_rect()
        self._draw_panel_frame(painter, plot_rect, "3D маршруты ВПП и зоны ответственности диспетчеров", "iso")
        if self.show_responsibility_zones:
            self._draw_responsibility_zones(painter, plot_rect, bounds)
        self._draw_projection(painter, plot_rect, "iso", bounds)
        self._draw_legend(painter)
        self._draw_context_box(painter)

    def mouseMoveEvent(self, event):
        cluster_id = self._cluster_at_position(event.pos())
        if cluster_id is None:
            QToolTip.hideText()
            return
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            QToolTip.hideText()
            return
        QToolTip.showText(event.globalPos(), self._cluster_tooltip(cluster_id, cluster), self)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)

    def _plot_rects(self):
        area = self.rect().adjusted(18, 48, -18, -136)
        gap = 14
        cell_w = (area.width() - gap) / 2
        cell_h = (area.height() - gap) / 2
        return [
            QRectF(area.left(), area.top(), cell_w, cell_h),
            QRectF(area.left() + cell_w + gap, area.top(), cell_w, cell_h),
            QRectF(area.left(), area.top() + cell_h + gap, cell_w, cell_h),
            QRectF(area.left() + cell_w + gap, area.top() + cell_h + gap, cell_w, cell_h),
        ]

    def _main_plot_rect(self):
        return self.rect().adjusted(18, 50, -18, -148)

    def _draw_projection(self, painter, rect, projection, bounds):
        min_x, max_x, min_y, max_y, min_z, max_z = bounds

        for cluster_index, (cluster_id, cluster) in enumerate(self.clusters.items()):
            color = self.COLORS[cluster_index % len(self.COLORS)]
            if self.show_clusters:
                self._draw_cluster_area(painter, cluster, rect, projection, bounds, color)
                self._draw_sector_footprint(painter, cluster, rect, projection, bounds, color)
                self._draw_density_points(painter, cluster, rect, projection, bounds, color)
                self._draw_oas_surface(painter, cluster, rect, projection, bounds)
            if self.show_routes:
                self._draw_cluster_centerline(painter, cluster_id, cluster, rect, projection, bounds, color)

    def _draw_responsibility_zones(self, painter, rect, bounds):
        centers = []
        plot_rect = rect.adjusted(32, 24, -18, -28)
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        for index, (cluster_id, cluster) in enumerate(self.clusters.items()):
            center = self._cluster_center(cluster)
            if center is None:
                continue
            mapped = self._map_point(center[0], center[1], center[2], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, "iso")
            centers.append((cluster_id, mapped, self.COLORS[index % len(self.COLORS)]))

        if not centers:
            return

        cell = 28
        painter.setPen(Qt.NoPen)
        for y in range(int(plot_rect.top()), int(plot_rect.bottom()), cell):
            for x in range(int(plot_rect.left()), int(plot_rect.right()), cell):
                nearest_id, _, color = min(
                    centers,
                    key=lambda item: (item[1][0] - x) ** 2 + (item[1][1] - y) ** 2
                )
                fill = QColor(color)
                fill.setAlpha(30)
                painter.fillRect(QRectF(x, y, cell + 1, cell + 1), fill)

        for cluster_id, mapped, color in centers:
            marker = QColor(color)
            marker.setAlpha(150)
            painter.setBrush(marker)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(mapped[0] - 5, mapped[1] - 5, 10, 10)
        painter.setBrush(Qt.NoBrush)

    def _visible_track_ids(self, track_ids, top_anomaly):
        visible = []
        if top_anomaly:
            visible.append(top_anomaly)
        visible.extend([track_id for track_id in track_ids if track_id in self.selected_track_ids])
        return list(dict.fromkeys(visible))[:self.max_draw_tracks_per_cluster]

    def _limited_track_ids(self, track_ids, limit):
        if len(track_ids) <= limit:
            return list(track_ids)
        step = max(1, len(track_ids) // limit)
        return list(track_ids[::step])[:limit]

    def _draw_density_points(self, painter, cluster, rect, projection, bounds, color):
        sample_points = []
        track_ids = cluster.get("tracks", [])
        if not track_ids:
            return
        per_track_step = max(1, len(track_ids) // 40)
        for track_id in track_ids[::per_track_step]:
            track = self.tracks.get(track_id)
            if track is None or not getattr(track, "points", None):
                continue
            point_step = max(1, len(track.points) // 8)
            for point in track.points[::point_step]:
                sample_points.append(point)
                if len(sample_points) >= self.max_scatter_points_per_cluster:
                    break
            if len(sample_points) >= self.max_scatter_points_per_cluster:
                break

        scatter_color = QColor(color)
        scatter_color.setAlpha(45)
        painter.setPen(Qt.NoPen)
        painter.setBrush(scatter_color)
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        plot_rect = rect.adjusted(32, 20, -18, -26)
        for point in sample_points:
            x, y = self._map_point(point.x, point.y, point.z, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            painter.drawEllipse(x - 1, y - 1, 2, 2)
        painter.setBrush(Qt.NoBrush)

    def _draw_cluster_area(self, painter, cluster, rect, projection, bounds, color):
        points = self._sample_mapped_cluster_points(cluster, rect, projection, bounds)
        if len(points) < 3:
            return
        hull = self._convex_hull_2d(points)
        if len(hull) < 3:
            return

        fill_color = QColor(color)
        fill_color.setAlpha(34)
        border_color = QColor(color)
        border_color.setAlpha(115)
        path = QPainterPath()
        path.moveTo(hull[0][0], hull[0][1])
        for x, y in hull[1:]:
            path.lineTo(x, y)
        path.closeSubpath()
        painter.fillPath(path, fill_color)
        painter.setPen(QPen(border_color, 2))
        painter.drawPath(path)

    def _sample_mapped_cluster_points(self, cluster, rect, projection, bounds):
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        plot_rect = rect.adjusted(32, 20, -18, -26)
        mapped = []
        track_ids = cluster.get("tracks", [])
        if not track_ids:
            return mapped
        track_step = max(1, len(track_ids) // 60)
        for track_id in track_ids[::track_step]:
            track = self.tracks.get(track_id)
            if track is None or not getattr(track, "points", None):
                continue
            point_step = max(1, len(track.points) // 10)
            for point in track.points[::point_step]:
                mapped.append(self._map_point(point.x, point.y, point.z, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection))
        return mapped

    def _convex_hull_2d(self, points):
        unique = sorted(set(points))
        if len(unique) <= 1:
            return unique

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def _draw_track_markers(self, painter, track, rect, projection, bounds, color):
        if not getattr(track, "points", None):
            return
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        plot_rect = rect.adjusted(32, 20, -18, -26)
        marker_points = [track.points[0], track.points[-1]]
        if self._has_radar_jump(track):
            jump_index = self._radar_jump_index(track)
            if jump_index is not None:
                marker_points.append(track.points[jump_index])
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor(color))
        for point in marker_points:
            x, y = self._map_point(point.x, point.y, point.z, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            painter.drawRect(x - 3, y - 3, 6, 6)
        painter.setBrush(Qt.NoBrush)

    def _draw_centroid_label(self, painter, points, cluster_id):
        if not points:
            return
        x, y = points[len(points) // 2]
        painter.setPen(QColor("#111827"))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(x + 5, y - 5, f"Ц{cluster_id}")

    def _draw_cluster_centerline(self, painter, cluster_id, cluster, rect, projection, bounds, color):
        centerline = self._representative_points_for_cluster(cluster)
        if len(centerline) < 2:
            return

        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        plot_rect = rect.adjusted(32, 20, -18, -26)
        mapped = [
            self._map_point(point[0], point[1], point[2], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            for point in centerline
        ]
        line_color = QColor(color)
        line_color.setAlpha(235)
        pen = QPen(line_color, 6)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for start, end in zip(mapped, mapped[1:]):
            painter.drawLine(start[0], start[1], end[0], end[1])
        runway = self._cluster_runway(cluster)
        if runway != "н/д":
            painter.setPen(QColor("#0f172a"))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            label_x, label_y = mapped[-1]
            painter.drawText(label_x + 8, label_y - 8, f"ВПП {runway}")
        self._draw_centroid_label(painter, mapped, cluster_id)

    def _representative_points_for_cluster(self, cluster):
        centroid = cluster.get("centroid")
        if centroid is not None and len(centroid) > 1:
            return np.array(centroid, dtype=float)

        tracks = []
        track_ids = cluster.get("tracks", [])
        if not track_ids:
            return np.empty((0, 3))

        sample_ids = self._limited_track_ids(track_ids, 30)
        target_len = 60
        for track_id in sample_ids:
            track = self.tracks.get(track_id)
            if track is None or not getattr(track, "points", None):
                continue
            points = track.get_points_matrix()
            if len(points) < 2:
                continue
            old_t = np.linspace(0, 1, len(points))
            new_t = np.linspace(0, 1, target_len)
            resampled = np.column_stack([
                np.interp(new_t, old_t, points[:, axis])
                for axis in range(3)
            ])
            tracks.append(resampled)

        if not tracks:
            return np.empty((0, 3))
        return np.mean(np.array(tracks), axis=0)

    def _draw_oas_surface(self, painter, cluster, rect, projection, bounds):
        if projection not in ("xz", "yz"):
            return
        centroid = cluster.get("centroid")
        if centroid is None or len(centroid) < 2:
            return

        points = np.array(centroid)
        end = points[-1]
        start = points[0]
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        plot_rect = rect.adjusted(32, 20, -18, -26)
        gradient = 0.052

        if projection == "xz":
            axis_start = start[0]
            axis_end = end[0]
            z_start = end[2] + abs(axis_start - axis_end) * gradient
            mapped_start = self._map_point(axis_start, min_y, z_start, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            mapped_end = self._map_point(axis_end, min_y, end[2], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
        else:
            axis_start = start[1]
            axis_end = end[1]
            z_start = end[2] + abs(axis_start - axis_end) * gradient
            mapped_start = self._map_point(min_x, axis_start, z_start, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            mapped_end = self._map_point(min_x, axis_end, end[2], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)

        painter.setPen(QPen(QColor("#9ca3af"), 1, Qt.DotLine))
        painter.drawLine(mapped_start[0], mapped_start[1], mapped_end[0], mapped_end[1])
        painter.setFont(QFont("Arial", 7))
        painter.setPen(QColor("#6b7280"))
        painter.drawText(mapped_start[0] + 4, mapped_start[1] - 4, "OAS 3°")

    def _draw_sector_footprint(self, painter, cluster, rect, projection, bounds, color):
        sector = cluster.get("sector", {})
        footprints = sector.get("corrected_footprints") or sector.get("footprints", {})
        if projection == "iso":
            footprint = footprints.get("xy", [])
        else:
            footprint = footprints.get(projection, [])
        if len(footprint) < 2:
            return

        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        plot_rect = rect.adjusted(32, 20, -18, -26)
        points = []
        for point in footprint:
            if projection == "xz":
                mapped = self._map_point(point[0], min_y, point[1], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            elif projection == "yz":
                mapped = self._map_point(min_x, point[0], point[1], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection)
            else:
                mapped = self._map_point(point[0], point[1], min_z, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, "xy")
            points.append(mapped)

        sector_color = QColor(color)
        sector_color.setAlpha(150)
        pen = QPen(sector_color, 2, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for start, end in zip(points, points[1:] + points[:1]):
            painter.drawLine(start[0], start[1], end[0], end[1])

    def _draw_title(self, painter):
        painter.setPen(QColor("#1f2937"))
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        painter.drawText(QRectF(12, 10, self.width() - 24, 24), Qt.AlignCenter, self.title)

    def _draw_empty_state(self, painter, plot_rect):
        painter.setPen(QColor("#6b7280"))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(plot_rect, Qt.AlignCenter, "Нет данных кластеризации для отображения")

    def _draw_panel_frame(self, painter, rect, title, projection):
        painter.setPen(QPen(QColor("#d0d7de"), 1))
        painter.drawRect(rect)
        painter.setPen(QColor("#1f2937"))
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        painter.drawText(rect.adjusted(8, 4, -8, -4), Qt.AlignTop | Qt.AlignLeft, title)
        self._draw_grid(painter, rect.adjusted(32, 20, -18, -26))
        self._draw_axis_labels(painter, rect, projection)

    def _draw_grid(self, painter, plot_rect):
        painter.setPen(QPen(QColor("#eef2f7"), 1))
        for index in range(1, 5):
            x = plot_rect.left() + plot_rect.width() * index / 5
            y = plot_rect.top() + plot_rect.height() * index / 5
            painter.drawLine(int(x), int(plot_rect.top()), int(x), int(plot_rect.bottom()))
            painter.drawLine(int(plot_rect.left()), int(y), int(plot_rect.right()), int(y))

    def _draw_axis_labels(self, painter, rect, projection):
        labels = {
            "iso": ("X/Y", "Высота Z"),
            "xy": ("X", "Y"),
            "xz": ("X", "Z"),
            "yz": ("Y", "Z"),
        }[projection]
        painter.setPen(QColor("#6b7280"))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(rect.adjusted(32, 0, -18, -6), Qt.AlignBottom | Qt.AlignCenter, labels[0])
        painter.save()
        painter.translate(rect.left() + 12, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-rect.height() / 2, -8, rect.height(), 16), Qt.AlignCenter, labels[1])
        painter.restore()

    def _draw_legend(self, painter):
        legend_rect = QRectF(18, self.height() - 132, self.width() - 36, 114)
        painter.fillRect(legend_rect, QColor(248, 250, 252, 235))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(legend_rect)

        y = int(legend_rect.top()) + 66
        x = int(legend_rect.left()) + 14
        painter.setPen(QColor("#0f172a"))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(x, int(legend_rect.top()) + 44, "Легенда: цветная область - зона ответственности, толстая линия - основной маршрут ВПП")

        painter.setFont(QFont("Arial", 10))
        column_width = max(220, int((legend_rect.width() - 28) / 3))
        for index, (cluster_id, cluster) in enumerate(self.clusters.items()):
            if index >= 9:
                break
            col = index % 3
            row = index // 3
            item_x = x + col * column_width
            item_y = y + row * 24
            color = self.COLORS[index % len(self.COLORS)]

            painter.setPen(QPen(color, 6))
            painter.drawLine(item_x, item_y, item_x + 26, item_y)
            painter.setPen(QColor("#374151"))
            runway = self._cluster_runway(cluster)
            merged_count = len(cluster.get("merged_cluster_ids", []))
            merged_text = f", {merged_count} пучк." if merged_count > 1 else ""
            painter.drawText(item_x + 34, item_y + 5, f"К{cluster_id}: {cluster.get('size', 0)} ВС{merged_text}, ВПП {runway}")

    def _draw_context_box(self, painter):
        if not self.clusters:
            return
        rect = QRectF(self.width() - 286, 46, 274, 42)
        painter.fillRect(rect, QColor(248, 250, 252, 230))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(rect)
        total_anomalies = sum(cluster.get("anomalies_count", 0) for cluster in self.clusters.values())
        total_rule = sum(cluster.get("rule_risks_count", 0) for cluster in self.clusters.values())
        max_workload = max((cluster.get("workload", {}).get("workload_index", 0) for cluster in self.clusters.values()), default=0)
        painter.setPen(QColor("#111827"))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(rect.adjusted(8, 3, -8, -16), Qt.AlignLeft, "PBN / CRM")
        painter.setFont(QFont("Arial", 8))
        painter.drawText(
            rect.adjusted(8, 16, -8, -3),
            Qt.AlignLeft,
            f"аномалии: {total_anomalies}, правила: {total_rule}, Wmax: {max_workload:.1f}"
        )

    def _draw_temporal_charts(self, painter):
        track = self._representative_track()
        if track is None or len(getattr(track, "points", [])) < 3:
            return
        area = QRectF(18, self.height() - 66, self.width() - 36, 48)
        gap = 10
        chart_w = (area.width() - 2 * gap) / 3
        charts = [
            ("V(t)", self._speed_series(track), QColor("#2563eb")),
            ("θ(t)", self._gradient_series(track), QColor("#16a34a")),
            ("ψ(t)", self._heading_series(track), QColor("#dc2626")),
        ]
        for index, (title, values, color) in enumerate(charts):
            rect = QRectF(area.left() + index * (chart_w + gap), area.top(), chart_w, area.height())
            self._draw_small_chart(painter, rect, title, values, color)

    def _draw_small_chart(self, painter, rect, title, values, color):
        painter.setPen(QPen(QColor("#e5e7eb"), 1))
        painter.drawRect(rect)
        painter.setPen(QColor("#374151"))
        painter.setFont(QFont("Arial", 7, QFont.Bold))
        painter.drawText(rect.adjusted(4, 2, -4, -2), Qt.AlignTop | Qt.AlignLeft, title)
        if len(values) < 2:
            return
        values = np.array(values, dtype=float)
        if np.allclose(values.max(), values.min()):
            values = values + np.linspace(0, 1e-6, len(values))
        chart = rect.adjusted(6, 14, -6, -5)
        points = []
        for idx, value in enumerate(values):
            x = chart.left() + idx / (len(values) - 1) * chart.width()
            y = chart.bottom() - (value - values.min()) / (values.max() - values.min()) * chart.height()
            points.append((int(x), int(y)))
        painter.setPen(QPen(color, 1.5))
        for start, end in zip(points, points[1:]):
            painter.drawLine(start[0], start[1], end[0], end[1])

    def _bounds(self):
        xs = []
        ys = []
        zs = []
        for cluster in self.clusters.values():
            for track_id in cluster.get("tracks", []):
                track = self.tracks.get(track_id)
                if track is None:
                    continue
                for point in getattr(track, "points", []):
                    xs.append(point.x)
                    ys.append(point.y)
                    zs.append(point.z)

        if not xs or not ys or not zs:
            return None

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        if min_x == max_x:
            max_x += 1
        if min_y == max_y:
            max_y += 1
        if min_z == max_z:
            max_z += 1
        return min_x, max_x, min_y, max_y, min_z, max_z

    def _cluster_runway(self, cluster):
        counts = {}
        for track_id in cluster.get("tracks", []):
            track = self.tracks.get(track_id)
            if track is None:
                continue
            runway = str(getattr(track, "runwayname", "") or "").strip().upper()
            if not runway or runway in ("NAN", "NONE"):
                continue
            counts[runway] = counts.get(runway, 0) + 1
        return max(counts, key=counts.get) if counts else "н/д"

    def _filter_display_clusters(self, clusters):
        filtered = {}
        for cluster_id, cluster in clusters.items():
            if self._is_single_without_runway(cluster):
                continue
            filtered[cluster_id] = cluster
        return filtered

    def _is_single_without_runway(self, cluster):
        size = cluster.get("size", len(cluster.get("tracks", [])))
        return size <= 1 or self._cluster_runway(cluster) == "н/д"

    def _merge_small_clusters_for_display(self, clusters):
        if not clusters or len(clusters) <= self.max_display_clusters:
            return clusters or {}

        sorted_items = sorted(
            clusters.items(),
            key=lambda item: item[1].get("size", len(item[1].get("tracks", []))),
            reverse=True,
        )
        kept_count = max(1, self.max_display_clusters)
        kept = {
            cluster_id: self._copy_display_cluster(cluster_id, cluster)
            for cluster_id, cluster in sorted_items[:kept_count]
        }
        centers = {
            cluster_id: self._cluster_center(cluster)
            for cluster_id, cluster in kept.items()
        }

        for source_id, source_cluster in sorted_items[kept_count:]:
            source_center = self._cluster_center(source_cluster)
            target_id = self._nearest_cluster_id(source_center, centers)
            target = kept[target_id]
            target.setdefault("tracks", []).extend(source_cluster.get("tracks", []))
            target["size"] = target.get("size", 0) + source_cluster.get("size", len(source_cluster.get("tracks", [])))
            target["anomalies_count"] = target.get("anomalies_count", 0) + source_cluster.get("anomalies_count", 0)
            target["rule_risks_count"] = target.get("rule_risks_count", 0) + source_cluster.get("rule_risks_count", 0)
            target.setdefault("merged_cluster_ids", []).append(source_id)
            target.pop("centroid", None)
            centers[target_id] = self._cluster_center(target)

        return kept

    def _copy_display_cluster(self, cluster_id, cluster):
        copied = dict(cluster)
        copied["tracks"] = list(cluster.get("tracks", []))
        copied["merged_cluster_ids"] = [cluster_id]
        return copied

    def _nearest_cluster_id(self, source_center, centers):
        if source_center is None:
            return next(iter(centers))
        valid = [
            (cluster_id, center)
            for cluster_id, center in centers.items()
            if center is not None
        ]
        if not valid:
            return next(iter(centers))
        return min(valid, key=lambda item: float(np.linalg.norm(source_center - item[1])))[0]

    def _cluster_center(self, cluster):
        centroid = cluster.get("centroid")
        if centroid is not None and len(centroid) > 0:
            return np.mean(np.array(centroid, dtype=float), axis=0)

        points = []
        for track_id in self._limited_track_ids(cluster.get("tracks", []), 40):
            track = self.tracks.get(track_id)
            if track is None or not getattr(track, "points", None):
                continue
            matrix = track.get_points_matrix()
            if len(matrix):
                points.append(np.mean(matrix[:, :3], axis=0))
        if not points:
            return None
        return np.mean(np.array(points), axis=0)

    def _cluster_at_position(self, pos):
        bounds = self._bounds()
        if not self.clusters or bounds is None:
            return None
        rect = self._main_plot_rect()
        plot_rect = rect.adjusted(32, 24, -18, -28)
        if not plot_rect.contains(pos):
            return None

        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        nearest_line = None
        nearest_distance = 16.0
        centers = []
        for index, (cluster_id, cluster) in enumerate(self.clusters.items()):
            centerline = self._representative_points_for_cluster(cluster)
            if len(centerline) >= 2:
                mapped = [
                    self._map_point(point[0], point[1], point[2], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, "iso")
                    for point in centerline
                ]
                for start, end in zip(mapped, mapped[1:]):
                    distance = self._distance_to_segment(pos.x(), pos.y(), start, end)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_line = cluster_id
            center = self._cluster_center(cluster)
            if center is not None:
                mapped_center = self._map_point(center[0], center[1], center[2], min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, "iso")
                centers.append((cluster_id, mapped_center))

        if nearest_line is not None:
            return nearest_line
        if not centers:
            return None
        return min(centers, key=lambda item: (item[1][0] - pos.x()) ** 2 + (item[1][1] - pos.y()) ** 2)[0]

    def _distance_to_segment(self, px, py, start, end):
        ax, ay = start
        bx, by = end
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        nearest_x = ax + t * dx
        nearest_y = ay + t * dy
        return math.hypot(px - nearest_x, py - nearest_y)

    def _cluster_tooltip(self, cluster_id, cluster):
        runway = self._cluster_runway(cluster)
        aircraft_count = cluster.get("size", len(cluster.get("tracks", [])))
        conflict_count = self._merged_sum(cluster, "workload", "conflict_events_count")
        rule_risks = self._merged_sum(cluster, None, "rule_risks_count")
        avg_duration = self._average_route_duration(cluster)
        merged_count = len(cluster.get("merged_cluster_ids", []))
        title = f"Кластер {cluster_id}"
        if merged_count > 1:
            title += f" ({merged_count} объединенных пучков)"
        return (
            f"<b>{title}</b><br>"
            f"Основная ВПП/маршрут: {runway}<br>"
            f"ВС на маршруте: {aircraft_count}<br>"
            f"Риски столкновения/конфликты: {conflict_count}<br>"
            f"Жесткие процедурные риски: {rule_risks}<br>"
            f"Среднее время посадки-взлета: {self._format_duration(avg_duration)}"
        )

    def _merged_sum(self, cluster, nested_key, value_key):
        total = 0
        source_ids = cluster.get("merged_cluster_ids") or []
        source_clusters = [self.raw_clusters.get(source_id) for source_id in source_ids if source_id in self.raw_clusters]
        if not source_clusters:
            source_clusters = [cluster]
        for source in source_clusters:
            if not source:
                continue
            data = source.get(nested_key, {}) if nested_key else source
            total += int(data.get(value_key, 0) or 0)
        return total

    def _average_route_duration(self, cluster):
        durations = []
        for track_id in self._limited_track_ids(cluster.get("tracks", []), 300):
            track = self.tracks.get(track_id)
            if track is None:
                continue
            duration = self._track_duration_seconds(track)
            if duration is not None and duration > 0:
                durations.append(duration)
        return float(np.mean(durations)) if durations else 0.0

    def _track_duration_seconds(self, track):
        try:
            times = track.get_time_profile()
            if len(times) >= 2:
                return float(max(times) - min(times))
        except Exception:
            pass
        points = getattr(track, "points", [])
        if len(points) >= 2 and hasattr(points[0], "t") and hasattr(points[-1], "t"):
            return float(points[-1].t - points[0].t)
        return None

    def _format_duration(self, seconds):
        if seconds <= 0:
            return "н/д"
        minutes = int(seconds // 60)
        rest = int(seconds % 60)
        return f"{minutes} мин {rest:02d} сек"

    def _representative_track(self):
        for cluster in self.clusters.values():
            top_anomaly = cluster.get("top_anomaly")
            if top_anomaly in self.tracks:
                return self.tracks[top_anomaly]
        largest = max(self.clusters.values(), key=lambda item: item.get("size", 0), default=None)
        if not largest:
            return None
        for track_id in largest.get("tracks", []):
            if track_id in self.tracks:
                return self.tracks[track_id]
        return None

    def _speed_series(self, track):
        return list(track.get_velocity_profile())

    def _gradient_series(self, track):
        points = track.get_points_matrix()
        if len(points) < 2:
            return []
        deltas = np.diff(points, axis=0)
        horizontal = np.linalg.norm(deltas[:, :2], axis=1)
        return list(np.degrees(np.arctan2(deltas[:, 2], horizontal + 1e-8)))

    def _heading_series(self, track):
        points = track.get_points_matrix()
        if len(points) < 2:
            return []
        deltas = np.diff(points[:, :2], axis=0)
        headings = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
        return list(headings)

    def _has_radar_jump(self, track):
        return self._radar_jump_index(track) is not None

    def _radar_jump_index(self, track):
        points = track.get_points_matrix()
        times = track.get_time_profile()
        if len(points) < 3 or len(times) < 3:
            return None
        dt = np.diff(times)
        dt[dt <= 0] = 1.0
        velocities = np.diff(points, axis=0) / dt[:, None]
        accelerations = np.diff(velocities, axis=0)
        norms = np.linalg.norm(accelerations, axis=1)
        if len(norms) == 0:
            return None
        index = int(np.argmax(norms))
        return index + 1 if norms[index] > 96.5 else None

    def _map_point(self, x, y, z, min_x, max_x, min_y, max_y, min_z, max_z, plot_rect, projection):
        nx = (x - min_x) / (max_x - min_x)
        ny = (y - min_y) / (max_y - min_y)
        nz = (z - min_z) / (max_z - min_z)

        if projection == "xy":
            px_norm, py_norm = nx, ny
        elif projection == "xz":
            px_norm, py_norm = nx, nz
        elif projection == "yz":
            px_norm, py_norm = ny, nz
        else:
            px_norm = 0.5 + (nx - ny) * 0.42
            py_norm = 0.72 - nz * 0.52 + (nx + ny - 1) * 0.18

        px_norm = max(0.0, min(1.0, px_norm))
        py_norm = max(0.0, min(1.0, py_norm))
        px = plot_rect.left() + px_norm * plot_rect.width()
        py = plot_rect.bottom() - py_norm * plot_rect.height()
        return int(px), int(py)
