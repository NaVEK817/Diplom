"""Метрики качества, повторяемости и стабильности процедуры."""

from typing import Dict, List

import numpy as np


class ProcedureQualityAnalyzer:
    """Оценивает отклонение от центроида, коридор 2xRNP, RF и вертикальный профиль."""

    def __init__(self, rnp_nm: float = 0.1):
        self.rnp_ft = rnp_nm * 6076.12

    def analyze(self, tracks_dict: dict, clusters: dict, sectors: dict) -> Dict:
        results = {}
        for cluster_id, cluster in clusters.items():
            centroid = np.array(cluster.get("centroid", []), dtype=float)
            track_ids = cluster.get("tracks", [])
            tracks = [tracks_dict[track_id] for track_id in track_ids if track_id in tracks_dict]
            results[cluster_id] = self._cluster_quality(centroid, tracks, sectors.get(cluster_id, {}))
        return results

    def _cluster_quality(self, centroid: np.ndarray, tracks: List, sector: dict) -> dict:
        if len(centroid) < 2 or not tracks:
            return {"quality_score": 0.0, "repeatability": 0.0, "stable_approach_percent": 0.0}

        deviations = []
        corridor_ok = 0
        veb_ok = 0
        gradients = []
        for track in tracks:
            points = track.get_points_matrix()
            if len(points) < 2:
                continue
            aligned_track, aligned_centroid = self._align(points, centroid)
            deviations.append(self._mean_cosine_distance(aligned_track, aligned_centroid))
            lateral = np.linalg.norm(aligned_track[:, :2] - aligned_centroid[:, :2], axis=1)
            vertical = np.abs(aligned_track[:, 2] - aligned_centroid[:, 2])
            if np.percentile(lateral, 95) <= 2 * self.rnp_ft:
                corridor_ok += 1
            if np.percentile(vertical, 95) <= 300.0:
                veb_ok += 1
            gradients.append(self._descent_gradient(points))

        total = max(1, len(tracks))
        repeatability = 1.0 - float(np.mean(deviations)) if deviations else 0.0
        stable_percent = min(corridor_ok, veb_ok) / total * 100
        rf_consistency = self._rf_radius_consistency(centroid)
        gradient_mean = float(np.mean(gradients)) if gradients else 0.0
        gradient_deviation = abs(gradient_mean - 0.052)
        gradient_penalty = 1.0 if gradient_mean > 0.037 else 0.0
        quality_score = max(0.0, min(100.0, repeatability * 45 + stable_percent * 0.35 + rf_consistency * 20 - gradient_penalty * 20))

        return {
            "quality_score": float(quality_score),
            "repeatability": float(repeatability),
            "mean_cosine_deviation": float(np.mean(deviations)) if deviations else 0.0,
            "stable_approach_percent": float(stable_percent),
            "lateral_corridor_2rnp_ft": float(2 * self.rnp_ft),
            "vertical_budget_ft": 300.0,
            "rf_radius_consistency": float(rf_consistency),
            "mean_descent_gradient": float(gradient_mean),
            "gradient_deviation_from_3deg": float(gradient_deviation),
            "gradient_quality_warning": bool(gradient_mean > 0.035),
            "sector_floor_ft": sector.get("floor_ft"),
            "sector_ceiling_ft": sector.get("ceiling_ft"),
        }

    def _align(self, points: np.ndarray, centroid: np.ndarray):
        target_len = min(len(points), len(centroid))
        return self._resample(points, target_len), self._resample(centroid, target_len)

    def _resample(self, points: np.ndarray, target_len: int) -> np.ndarray:
        old_t = np.linspace(0, 1, len(points))
        new_t = np.linspace(0, 1, target_len)
        output = np.zeros((target_len, points.shape[1]))
        for dim in range(points.shape[1]):
            output[:, dim] = np.interp(new_t, old_t, points[:, dim])
        return output

    def _mean_cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        va = np.diff(a, axis=0)
        vb = np.diff(b, axis=0)
        denom = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-8
        cosines = np.sum(va * vb, axis=1) / denom
        return float(np.mean(1 - np.clip(cosines, -1, 1)))

    def _descent_gradient(self, points: np.ndarray) -> float:
        horizontal = np.linalg.norm(points[-1, :2] - points[0, :2])
        vertical_drop = points[0, 2] - points[-1, 2]
        return float(vertical_drop / (horizontal + 1e-8))

    def _rf_radius_consistency(self, points: np.ndarray) -> float:
        radii = []
        for i in range(1, len(points) - 1):
            radius = self._turn_radius(points[i - 1], points[i], points[i + 1])
            if radius > 0:
                radii.append(radius)
        if len(radii) < 2:
            return 1.0
        tolerance_ft = 0.1 * 6076.12
        return float(np.mean(np.abs(np.array(radii) - np.median(radii)) <= tolerance_ft))

    def _turn_radius(self, p1, p2, p3) -> float:
        a = np.linalg.norm(p2[:2] - p1[:2])
        b = np.linalg.norm(p3[:2] - p2[:2])
        c = np.linalg.norm(p3[:2] - p1[:2])
        area = abs(np.cross(p2[:2] - p1[:2], p3[:2] - p1[:2])) / 2
        return float((a * b * c) / (4 * area)) if area > 1e-8 else 0.0
