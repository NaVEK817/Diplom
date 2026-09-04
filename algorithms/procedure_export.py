"""Проектные экспорты центроидов и процедурных данных."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ProcedureExportConfig:
    mrp_lat: float = 37.6188056
    mrp_lon: float = -122.3754167
    ft_per_deg_lat: float = 364000.0


class ProcedureExporter:
    """Формирует GeoJSON и ARINC-подобную таблицу по центроидам и секторам."""

    def __init__(self, config: ProcedureExportConfig = None):
        self.config = config or ProcedureExportConfig()

    def export(self, clusters: dict, sectors: dict, tracks_dict: dict) -> Dict:
        procedures = {}
        features = []
        arinc_rows = []
        for cluster_id, cluster in clusters.items():
            centroid = np.array(cluster.get("centroid", []), dtype=float)
            if len(centroid) == 0:
                continue
            sector = sectors.get(cluster_id, {})
            track_ids = cluster.get("tracks", [])
            tracks = [tracks_dict[track_id] for track_id in track_ids if track_id in tracks_dict]
            waypoints = self._waypoints(cluster_id, centroid, sector, tracks)
            procedure_id = f"PROC-{cluster_id}"
            procedures[cluster_id] = {
                "procedure_id": procedure_id,
                "waypoints": waypoints,
                "geojson": self._geojson_feature(procedure_id, waypoints),
                "arinc_rows": self._arinc_rows(procedure_id, waypoints),
            }
            features.append(procedures[cluster_id]["geojson"])
            arinc_rows.extend(procedures[cluster_id]["arinc_rows"])

        return {
            "procedures": procedures,
            "geojson": {"type": "FeatureCollection", "features": features},
            "arinc_table": arinc_rows,
        }

    def _waypoints(self, cluster_id, centroid: np.ndarray, sector: dict, tracks: List) -> List[dict]:
        candidates = []
        for fix in sector.get("entry_fixes", []):
            candidates.append(("ENTRY", fix["name"], np.array([fix["x"], fix["y"], fix["z"]], dtype=float)))
        if len(centroid):
            for idx in np.linspace(0, len(centroid) - 1, min(5, len(centroid)), dtype=int):
                candidates.append(("TF", f"C{cluster_id}-{idx:02d}", centroid[idx]))
        for fix in sector.get("exit_fixes", []):
            candidates.append(("EXIT", fix["name"], np.array([fix["x"], fix["y"], fix["z"]], dtype=float)))

        waypoints = []
        for index, (kind, name, point) in enumerate(candidates, start=1):
            lat, lon = self._local_to_wgs84(point[0], point[1])
            alt_low, alt_high = self._altitude_constraint(point, tracks)
            speed = self._speed_constraint(point, tracks)
            waypoints.append({
                "id": name,
                "sequence": index,
                "type": "RF" if self._is_rf_point(index, candidates) else kind,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "lat": lat,
                "lon": lon,
                "altitude_constraint_ft": [alt_low, alt_high],
                "speed_constraint_kt": speed,
                "turn_radius_ft": self._local_radius(index, candidates),
            })
        return waypoints

    def _local_to_wgs84(self, x_ft: float, y_ft: float):
        lat = self.config.mrp_lat + y_ft / self.config.ft_per_deg_lat
        lon_scale = self.config.ft_per_deg_lat * np.cos(np.radians(self.config.mrp_lat))
        lon = self.config.mrp_lon + x_ft / (lon_scale + 1e-8)
        return float(lat), float(lon)

    def _altitude_constraint(self, point: np.ndarray, tracks: List):
        altitudes = []
        for track in tracks:
            matrix = track.get_points_matrix()
            if len(matrix) == 0:
                continue
            idx = int(np.argmin(np.linalg.norm(matrix[:, :2] - point[:2], axis=1)))
            altitudes.append(matrix[idx, 2])
        if not altitudes:
            return float(point[2]), float(point[2])
        mean = float(np.mean(altitudes))
        sigma = float(np.std(altitudes))
        return max(0.0, mean - 3 * sigma), mean + 3 * sigma

    def _speed_constraint(self, point: np.ndarray, tracks: List):
        speeds = []
        for track in tracks:
            matrix = track.get_points_matrix()
            profile = track.get_velocity_profile()
            if len(matrix) == 0 or len(profile) == 0:
                continue
            idx = int(np.argmin(np.linalg.norm(matrix[:, :2] - point[:2], axis=1)))
            speeds.append(profile[min(idx, len(profile) - 1)])
        return float(np.percentile(speeds, 85)) if speeds else 0.0

    def _geojson_feature(self, procedure_id: str, waypoints: List[dict]):
        coordinates = [[wp["lon"], wp["lat"], wp["z"]] for wp in waypoints]
        return {
            "type": "Feature",
            "properties": {"procedure_id": procedure_id},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        }

    def _arinc_rows(self, procedure_id: str, waypoints: List[dict]):
        rows = []
        for wp in waypoints:
            rows.append({
                "procedure_id": procedure_id,
                "seq": wp["sequence"],
                "waypoint_id": wp["id"],
                "path_type": wp["type"],
                "lat": wp["lat"],
                "lon": wp["lon"],
                "alt_low_ft": wp["altitude_constraint_ft"][0],
                "alt_high_ft": wp["altitude_constraint_ft"][1],
                "speed_kt": wp["speed_constraint_kt"],
                "radius_ft": wp["turn_radius_ft"],
            })
        return rows

    def _is_rf_point(self, index: int, candidates: List) -> bool:
        if index <= 1 or index >= len(candidates):
            return False
        p1 = candidates[index - 2][2][:2]
        p2 = candidates[index - 1][2][:2]
        p3 = candidates[index][2][:2]
        denom = np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p2) + 1e-8
        angle = np.degrees(np.arccos(np.clip(np.dot(p2 - p1, p3 - p2) / denom, -1, 1)))
        return angle > 3.0

    def _local_radius(self, index: int, candidates: List) -> float:
        if index <= 1 or index >= len(candidates):
            return 0.0
        p1 = candidates[index - 2][2]
        p2 = candidates[index - 1][2]
        p3 = candidates[index][2]
        a = np.linalg.norm(p2[:2] - p1[:2])
        b = np.linalg.norm(p3[:2] - p2[:2])
        c = np.linalg.norm(p3[:2] - p1[:2])
        area = abs(np.cross(p2[:2] - p1[:2], p3[:2] - p1[:2])) / 2
        return float((a * b * c) / (4 * area)) if area > 1e-8 else 0.0
