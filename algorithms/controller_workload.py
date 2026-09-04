"""Оценка загруженности диспетчера по 4D-траекториям внутри секторов."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class WorkloadConfig:
    time_bin_seconds: int = 300
    conflict_time_window_seconds: int = 60
    horizontal_separation_ft: float = 18228.0  # 3 NM
    vertical_separation_ft: float = 1000.0
    conflict_weight: float = 5.0
    peak_load_weight: float = 1.5


class ControllerWorkloadModel:
    """Считает occupancy, конфликтность, маневренность и интегральный индекс W."""

    def __init__(self, config: Optional[WorkloadConfig] = None):
        self.config = config or WorkloadConfig()

    def calculate(self, tracks_dict: dict, clusters: dict, sectors: dict) -> Dict:
        workload = {}
        for cluster_id, sector in sectors.items():
            track_ids = clusters.get(cluster_id, {}).get("tracks", [])
            tracks = [tracks_dict[track_id] for track_id in track_ids if track_id in tracks_dict]
            samples = self._sector_samples(tracks, sector)
            track_metrics = self._track_metrics(tracks, sector)
            occupancy = self._occupancy(samples)
            conflicts = self._conflicts(samples)

            peak_load = max((item["aircraft_count"] for item in occupancy), default=0)
            weighted_occupancy = self._weighted_occupancy(occupancy, track_metrics)
            conflicts_score = len(conflicts) * self.config.conflict_weight
            workload_index = weighted_occupancy + conflicts_score + peak_load * self.config.peak_load_weight

            workload[cluster_id] = {
                "workload_index": float(workload_index),
                "weighted_occupancy": float(weighted_occupancy),
                "peak_load": int(peak_load),
                "conflict_events_count": len(conflicts),
                "conflict_score": float(conflicts_score),
                "occupancy_timeline": occupancy,
                "conflicts": conflicts,
                "track_metrics": track_metrics,
                "avg_maneuver_complexity": float(np.mean([
                    metric["maneuver_complexity"] for metric in track_metrics.values()
                ])) if track_metrics else 0.0,
                "avg_dwell_time_seconds": float(np.mean([
                    metric["dwell_time_seconds"] for metric in track_metrics.values()
                ])) if track_metrics else 0.0,
            }
        return workload

    def _sector_samples(self, tracks: list, sector: dict) -> List[dict]:
        samples = []
        for track in tracks:
            points = track.get_points_matrix()
            times = track.get_time_profile()
            if len(points) == 0:
                continue

            absolute_start = self._parse_time(track.trackstart_time)
            velocities = self._velocity_vectors(points, times)
            for index, point in enumerate(points):
                if not self._inside_sector(point, sector):
                    continue
                samples.append({
                    "track_id": self._track_id(track),
                    "time": float(absolute_start + times[index]),
                    "point": point.astype(float),
                    "velocity": velocities[index],
                })
        return samples

    def _occupancy(self, samples: List[dict]) -> List[dict]:
        bins: Dict[int, set] = {}
        for sample in samples:
            bin_id = int(sample["time"] // self.config.time_bin_seconds)
            bins.setdefault(bin_id, set()).add(sample["track_id"])

        return [
            {
                "time_start": bin_id * self.config.time_bin_seconds,
                "time_end": (bin_id + 1) * self.config.time_bin_seconds,
                "aircraft_count": len(track_ids),
                "track_ids": sorted(track_ids),
            }
            for bin_id, track_ids in sorted(bins.items())
        ]

    def _weighted_occupancy(self, occupancy: List[dict], track_metrics: dict) -> float:
        total = 0.0
        for item in occupancy:
            if not item["track_ids"]:
                continue
            avg_complexity = np.mean([
                track_metrics.get(track_id, {}).get("maneuver_complexity", 1.0)
                for track_id in item["track_ids"]
            ])
            total += item["aircraft_count"] * avg_complexity
        return float(total)

    def _conflicts(self, samples: List[dict]) -> List[dict]:
        conflicts = []
        seen = set()
        samples_by_bin: Dict[int, List[dict]] = {}
        bin_size = self.config.conflict_time_window_seconds
        for sample in samples:
            samples_by_bin.setdefault(int(sample["time"] // bin_size), []).append(sample)

        for bin_id, bin_samples in samples_by_bin.items():
            if len(bin_samples) < 2:
                continue
            positions_xy = np.array([sample["point"][:2] for sample in bin_samples])
            tree = cKDTree(positions_xy)
            pairs = tree.query_pairs(self.config.horizontal_separation_ft)
            for left, right in pairs:
                a = bin_samples[left]
                b = bin_samples[right]
                if a["track_id"] == b["track_id"]:
                    continue
                if abs(a["time"] - b["time"]) > self.config.conflict_time_window_seconds:
                    continue
                vertical_distance = abs(a["point"][2] - b["point"][2])
                if vertical_distance > self.config.vertical_separation_ft:
                    continue
                key = tuple(sorted((a["track_id"], b["track_id"])) + [bin_id])
                if key in seen:
                    continue
                seen.add(key)

                convergence = self._convergence_score(a, b)
                if convergence <= 0:
                    continue

                horizontal_distance = float(np.linalg.norm(a["point"][:2] - b["point"][:2]))
                conflicts.append({
                    "track_a": a["track_id"],
                    "track_b": b["track_id"],
                    "time": float((a["time"] + b["time"]) / 2),
                    "horizontal_distance_ft": horizontal_distance,
                    "vertical_distance_ft": float(vertical_distance),
                    "convergence_score": float(convergence),
                    "severity": self._conflict_severity(horizontal_distance, vertical_distance, convergence),
                })
        return conflicts

    def _track_metrics(self, tracks: list, sector: dict) -> Dict[str, dict]:
        metrics = {}
        for track in tracks:
            points = track.get_points_matrix()
            times = track.get_time_profile()
            if len(points) < 2:
                continue
            inside_indices = [idx for idx, point in enumerate(points) if self._inside_sector(point, sector)]
            if not inside_indices:
                continue

            dwell_time = float(times[inside_indices[-1]] - times[inside_indices[0]])
            curvature = self._mean_curvature(points[inside_indices])
            descent_gradient = self._descent_gradient(points[inside_indices])
            maneuver_complexity = self._maneuver_complexity(curvature, descent_gradient, track)
            metrics[self._track_id(track)] = {
                "dwell_time_seconds": max(0.0, dwell_time),
                "curvature_score": float(curvature),
                "descent_gradient": float(descent_gradient),
                "maneuver_complexity": float(maneuver_complexity),
                "entry_time": float(self._parse_time(track.trackstart_time) + times[inside_indices[0]]),
                "exit_time": float(self._parse_time(track.trackstart_time) + times[inside_indices[-1]]),
            }
        return metrics

    def _mean_curvature(self, points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        vectors_a = points[1:-1] - points[:-2]
        vectors_b = points[2:] - points[1:-1]
        norms = np.linalg.norm(vectors_a, axis=1) * np.linalg.norm(vectors_b, axis=1) + 1e-8
        cosines = np.sum(vectors_a * vectors_b, axis=1) / norms
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))
        return float(np.mean(angles))

    def _descent_gradient(self, points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        horizontal = np.linalg.norm(points[-1, :2] - points[0, :2])
        vertical = points[0, 2] - points[-1, 2]
        return float(vertical / (horizontal + 1e-8))

    def _maneuver_complexity(self, curvature: float, descent_gradient: float, track) -> float:
        curvature_component = min(1.0, curvature / 0.25)
        gradient_component = min(1.0, abs(descent_gradient) / 0.12)
        aircraft_component = 0.2 if self._is_high_attention_aircraft(track) else 0.0
        return 1.0 + 0.7 * curvature_component + 0.5 * gradient_component + aircraft_component

    def _is_high_attention_aircraft(self, track) -> bool:
        aircraft_text = f"{track.aircrafttype or ''} {track.aircraft_category or ''}".lower()
        markers = ("heavy", "b747", "b777", "a340", "a380", "mil", "military", "ввс")
        return any(marker in aircraft_text for marker in markers)

    def _velocity_vectors(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.zeros_like(points)
        dt = np.diff(times)
        dt[dt == 0] = 1e-8
        segment_velocities = np.diff(points, axis=0) / dt[:, None]
        velocities = np.vstack([segment_velocities[0], segment_velocities])
        return velocities

    def _convergence_score(self, a: dict, b: dict) -> float:
        relative_position = b["point"] - a["point"]
        relative_velocity = b["velocity"] - a["velocity"]
        closing_rate = -float(np.dot(relative_position, relative_velocity))
        if closing_rate <= 0:
            return 0.0

        va = a["velocity"]
        vb = b["velocity"]
        cosine = float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8))
        opposing_motion = max(0.0, -cosine)
        return min(1.0, closing_rate / 10000.0) * (0.5 + 0.5 * opposing_motion)

    def _conflict_severity(self, horizontal_distance: float, vertical_distance: float, convergence: float) -> str:
        horizontal_ratio = horizontal_distance / self.config.horizontal_separation_ft
        vertical_ratio = vertical_distance / self.config.vertical_separation_ft
        if horizontal_ratio < 0.5 and vertical_ratio < 0.5 and convergence > 0.6:
            return "Высокая"
        if horizontal_ratio < 0.8 and vertical_ratio < 0.8:
            return "Средняя"
        return "Низкая"

    def _inside_sector(self, point: np.ndarray, sector: dict) -> bool:
        bounds = sector.get("corrected_bounds") or sector.get("bounds") or {}
        required = {"min_x", "max_x", "min_y", "max_y", "min_z", "max_z"}
        if not required.issubset(bounds):
            return False
        if not (
            bounds["min_x"] <= point[0] <= bounds["max_x"]
            and bounds["min_y"] <= point[1] <= bounds["max_y"]
            and bounds["min_z"] <= point[2] <= bounds["max_z"]
        ):
            return False

        equations = sector.get("hull_equations", [])
        if equations:
            eq = np.array(equations, dtype=float)
            distances = eq[:, :3] @ point + eq[:, 3]
            return bool(np.all(distances <= sector.get("safety_buffer_ft", 0.0)))
        return True

    def _parse_time(self, value: str) -> float:
        text = str(value or "").strip()
        try:
            if ":" in text:
                parts = [float(part) for part in text.split(":")]
                if len(parts) == 3:
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(text) == 6 and text.isdigit():
                return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])
        except ValueError:
            pass
        return 0.0

    def _track_id(self, track) -> str:
        return f"{track.track_num}_{track.eventid}"
