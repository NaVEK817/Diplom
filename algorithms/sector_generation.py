"""Построение управляемых секторов по пучкам траекторий."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN


@dataclass
class SectorGenerationConfig:
    safety_buffer_ft: float = 1215.2  # 2 x RNP 0.1 NM
    vertical_buffer_ft: float = 300.0
    max_points_per_sector: int = 5000
    fix_eps_ft: float = 2500.0
    fix_min_samples: int = 2


class SectorGenerator:
    """Создает 3D-секторы, точки входа/выхода и базовые ограничения ОрВД."""

    def __init__(self, config: Optional[SectorGenerationConfig] = None):
        self.config = config or SectorGenerationConfig()

    def generate(self, tracks_dict: dict, clusters: dict, restricted_zones: Optional[List[dict]] = None) -> Dict:
        sectors = {}
        for cluster_id, cluster in clusters.items():
            track_ids = cluster.get("tracks", [])
            tracks = [tracks_dict[track_id] for track_id in track_ids if track_id in tracks_dict]
            points = self._collect_points(tracks)
            if len(points) < 4:
                sectors[cluster_id] = self._empty_sector(cluster_id, "Недостаточно точек для построения оболочки")
                continue

            sampled = self._sample_points(points)
            sector = self._build_sector(cluster_id, sampled, tracks)
            sector["conflicts"] = self._detect_restricted_zone_conflicts(sector, restricted_zones or [])
            self._apply_geometric_constraints(sector, restricted_zones or [])
            sectors[cluster_id] = sector

        return sectors

    def _collect_points(self, tracks: list) -> np.ndarray:
        matrices = [track.get_points_matrix() for track in tracks if getattr(track, "points", None)]
        if not matrices:
            return np.empty((0, 3))
        return np.vstack(matrices).astype(float)

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= self.config.max_points_per_sector:
            return points
        indices = np.linspace(0, len(points) - 1, self.config.max_points_per_sector, dtype=int)
        return points[indices]

    def _build_sector(self, cluster_id, points: np.ndarray, tracks: list) -> dict:
        bounds = self._buffered_bounds(points)
        floor = max(0.0, float(np.percentile(points[:, 2], 2) - self.config.vertical_buffer_ft))
        ceiling = float(np.percentile(points[:, 2], 98) + self.config.vertical_buffer_ft)

        hull_data = self._convex_hull(points)
        entry_points, exit_points = self._boundary_points(tracks, hull_data, bounds)

        return {
            "cluster_id": cluster_id,
            "method": hull_data["method"],
            "safety_buffer_ft": self.config.safety_buffer_ft,
            "floor_ft": floor,
            "ceiling_ft": ceiling,
            "bounds": bounds,
            "hull_vertices": hull_data["vertices"],
            "hull_equations": hull_data["equations"],
            "footprints": self._projection_footprints(hull_data["vertices"], bounds),
            "entry_fixes": self._cluster_fixes(entry_points, "Вход"),
            "exit_fixes": self._cluster_fixes(exit_points, "Выход"),
            "entry_points_count": len(entry_points),
            "exit_points_count": len(exit_points),
            "volume_estimate": self._volume_estimate(hull_data, bounds),
        }

    def _convex_hull(self, points: np.ndarray) -> dict:
        try:
            hull = ConvexHull(points)
            return {
                "method": "Convex Hull 3D",
                "vertices": points[hull.vertices].tolist(),
                "equations": hull.equations.tolist(),
                "volume": float(hull.volume),
            }
        except QhullError:
            bounds = self._buffered_bounds(points)
            vertices = self._box_vertices(bounds)
            return {
                "method": "Ограничивающий параллелепипед",
                "vertices": vertices,
                "equations": [],
                "volume": self._box_volume(bounds),
            }

    def _buffered_bounds(self, points: np.ndarray) -> dict:
        buffer_xy = self.config.safety_buffer_ft
        buffer_z = self.config.vertical_buffer_ft
        return {
            "min_x": float(np.min(points[:, 0]) - buffer_xy),
            "max_x": float(np.max(points[:, 0]) + buffer_xy),
            "min_y": float(np.min(points[:, 1]) - buffer_xy),
            "max_y": float(np.max(points[:, 1]) + buffer_xy),
            "min_z": float(np.min(points[:, 2]) - buffer_z),
            "max_z": float(np.max(points[:, 2]) + buffer_z),
        }

    def _box_vertices(self, bounds: dict) -> List[List[float]]:
        return [
            [x, y, z]
            for x in (bounds["min_x"], bounds["max_x"])
            for y in (bounds["min_y"], bounds["max_y"])
            for z in (bounds["min_z"], bounds["max_z"])
        ]

    def _box_volume(self, bounds: dict) -> float:
        return (
            (bounds["max_x"] - bounds["min_x"])
            * (bounds["max_y"] - bounds["min_y"])
            * (bounds["max_z"] - bounds["min_z"])
        )

    def _point_inside(self, point: np.ndarray, hull_data: dict, bounds: dict) -> bool:
        equations = hull_data.get("equations", [])
        if equations:
            eq = np.array(equations)
            distances = eq[:, :3] @ point + eq[:, 3]
            return bool(np.all(distances <= self.config.safety_buffer_ft))

        return (
            bounds["min_x"] <= point[0] <= bounds["max_x"]
            and bounds["min_y"] <= point[1] <= bounds["max_y"]
            and bounds["min_z"] <= point[2] <= bounds["max_z"]
        )

    def _boundary_points(self, tracks: list, hull_data: dict, bounds: dict) -> Tuple[List[List[float]], List[List[float]]]:
        entries = []
        exits = []
        for track in tracks:
            matrix = track.get_points_matrix()
            if len(matrix) == 0:
                continue

            inside_mask = [self._point_inside(point, hull_data, bounds) for point in matrix]
            inside_indices = [index for index, inside in enumerate(inside_mask) if inside]
            if not inside_indices:
                continue

            entries.append(matrix[inside_indices[0]].tolist())
            exits.append(matrix[inside_indices[-1]].tolist())

        return entries, exits

    def _cluster_fixes(self, points: List[List[float]], prefix: str) -> List[dict]:
        if not points:
            return []

        data = np.array(points)
        if len(data) < self.config.fix_min_samples:
            return [self._fix(prefix, 1, data)]

        labels = DBSCAN(eps=self.config.fix_eps_ft, min_samples=self.config.fix_min_samples).fit_predict(data)
        fixes = []
        fix_index = 1
        for label in sorted(set(labels)):
            if label == -1:
                continue
            cluster_points = data[labels == label]
            fixes.append(self._fix(prefix, fix_index, cluster_points))
            fix_index += 1

        if not fixes:
            fixes.append(self._fix(prefix, 1, data))
        return fixes

    def _fix(self, prefix: str, index: int, points: np.ndarray) -> dict:
        center = np.mean(points, axis=0)
        spread = np.max(np.linalg.norm(points - center, axis=1)) if len(points) else 0.0
        return {
            "name": f"{prefix}-{index}",
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
            "track_count": int(len(points)),
            "radius_ft": float(spread),
        }

    def _projection_footprints(self, hull_vertices: List[List[float]], bounds: dict) -> dict:
        points = np.array(hull_vertices, dtype=float) if hull_vertices else np.array(self._box_vertices(bounds))
        return {
            "xy": self._projection_hull(points[:, [0, 1]]),
            "xz": self._projection_hull(points[:, [0, 2]]),
            "yz": self._projection_hull(points[:, [1, 2]]),
        }

    def _projection_hull(self, points_2d: np.ndarray) -> List[List[float]]:
        if len(points_2d) < 3:
            return points_2d.tolist()
        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices].tolist()
        except QhullError:
            min_xy = np.min(points_2d, axis=0)
            max_xy = np.max(points_2d, axis=0)
            return [
                [float(min_xy[0]), float(min_xy[1])],
                [float(max_xy[0]), float(min_xy[1])],
                [float(max_xy[0]), float(max_xy[1])],
                [float(min_xy[0]), float(max_xy[1])],
            ]

    def _volume_estimate(self, hull_data: dict, bounds: dict) -> float:
        return float(hull_data.get("volume") or self._box_volume(bounds))

    def _detect_restricted_zone_conflicts(self, sector: dict, restricted_zones: Iterable[dict]) -> List[dict]:
        conflicts = []
        sector_bounds = sector["bounds"]
        for zone in restricted_zones:
            zone_bounds = zone.get("bounds", {})
            if self._bounds_intersect(sector_bounds, zone_bounds):
                conflicts.append({
                    "zone_id": zone.get("id", "unknown"),
                    "zone_name": zone.get("name", "Ограниченная зона"),
                    "status": "Обнаружено пересечение",
                })
        return conflicts

    def _apply_geometric_constraints(self, sector: dict, restricted_zones: List[dict]) -> None:
        if not restricted_zones:
            sector["csg_status"] = "Ограничения не заданы"
            sector["csg_operations"] = []
            return

        corrected_bounds = dict(sector["bounds"])
        operations = []
        for zone in restricted_zones:
            zone_bounds = zone.get("bounds", {})
            if not self._bounds_intersect(corrected_bounds, zone_bounds):
                continue

            operation = self._trim_bounds_away_from_zone(corrected_bounds, zone_bounds)
            if operation:
                operation["zone_id"] = zone.get("id", "unknown")
                operation["zone_name"] = zone.get("name", "Ограниченная зона")
                operations.append(operation)

        if operations:
            sector["corrected_bounds"] = corrected_bounds
            sector["corrected_footprints"] = self._projection_footprints(
                self._box_vertices(corrected_bounds),
                corrected_bounds,
            )
            sector["csg_status"] = "Выполнена автоматическая геометрическая коррекция по AABB-ограничениям"
        else:
            sector["csg_status"] = "Пересечения с заданными ограничениями не обнаружены"
        sector["csg_operations"] = operations

    def _trim_bounds_away_from_zone(self, sector_bounds: dict, zone_bounds: dict) -> Optional[dict]:
        axes = [
            ("x", "min_x", "max_x"),
            ("y", "min_y", "max_y"),
            ("z", "min_z", "max_z"),
        ]
        overlaps = []
        for axis, min_key, max_key in axes:
            overlap = min(sector_bounds[max_key], zone_bounds[max_key]) - max(sector_bounds[min_key], zone_bounds[min_key])
            if overlap > 0:
                overlaps.append((overlap, axis, min_key, max_key))

        if not overlaps:
            return None

        _, axis, min_key, max_key = min(overlaps, key=lambda item: item[0])
        sector_center = (sector_bounds[min_key] + sector_bounds[max_key]) / 2
        zone_center = (zone_bounds[min_key] + zone_bounds[max_key]) / 2
        old_min = sector_bounds[min_key]
        old_max = sector_bounds[max_key]

        if sector_center <= zone_center:
            sector_bounds[max_key] = min(sector_bounds[max_key], zone_bounds[min_key])
            trimmed_side = max_key
        else:
            sector_bounds[min_key] = max(sector_bounds[min_key], zone_bounds[max_key])
            trimmed_side = min_key

        if sector_bounds[min_key] >= sector_bounds[max_key]:
            sector_bounds[min_key] = old_min
            sector_bounds[max_key] = old_max
            return None

        return {
            "type": "AABB trim",
            "axis": axis,
            "trimmed_side": trimmed_side,
            "old_min": float(old_min),
            "old_max": float(old_max),
            "new_min": float(sector_bounds[min_key]),
            "new_max": float(sector_bounds[max_key]),
        }

    def _bounds_intersect(self, a: dict, b: dict) -> bool:
        required = {"min_x", "max_x", "min_y", "max_y", "min_z", "max_z"}
        if not required.issubset(b):
            return False
        return not (
            a["max_x"] < b["min_x"]
            or a["min_x"] > b["max_x"]
            or a["max_y"] < b["min_y"]
            or a["min_y"] > b["max_y"]
            or a["max_z"] < b["min_z"]
            or a["min_z"] > b["max_z"]
        )

    def _empty_sector(self, cluster_id, reason: str) -> dict:
        return {
            "cluster_id": cluster_id,
            "error": reason,
            "entry_fixes": [],
            "exit_fixes": [],
            "conflicts": [],
            "footprints": {},
        }
