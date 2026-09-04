"""Валидация PBN/RNP и физической реализуемости процедур."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PbnValidationConfig:
    rnp_nm: float = 0.1
    bank_split_altitude_ft: float = 400.0
    max_bank_above_deg: float = 25.0
    max_bank_below_deg: float = 8.0
    coordinate_resolution_ft: float = 60.0
    vertical_angle_resolution_deg: float = 0.01
    min_rf_radius_nm: float = 0.3

    @property
    def rnp_ft(self) -> float:
        return self.rnp_nm * 6076.12


class PbnValidator:
    """Проверяет RF-сегменты, DTA, длины сегментов и точность базы данных."""

    def __init__(self, config: PbnValidationConfig = None):
        self.config = config or PbnValidationConfig()

    def validate(self, clusters: dict, sectors: dict) -> Dict:
        results = {}
        for cluster_id, cluster in clusters.items():
            centroid = np.array(cluster.get("centroid", []), dtype=float)
            sector = sectors.get(cluster_id, {})
            results[cluster_id] = self.validate_procedure(centroid, sector)
        return results

    def validate_procedure(self, centroid: np.ndarray, sector: dict) -> dict:
        issues = []
        rf_segments = self._rf_segments(centroid)
        bank_results = self._validate_bank(centroid, rf_segments)
        flyby_results = self._validate_flyby_lengths(centroid, rf_segments)
        precision_results = self._validate_database_precision(centroid)

        issues.extend(bank_results["issues"])
        issues.extend(flyby_results["issues"])
        issues.extend(precision_results["issues"])

        return {
            "is_valid": len([issue for issue in issues if issue["severity"] == "Высокая"]) == 0,
            "issues": issues,
            "rf_segments_count": len(rf_segments),
            "max_bank_deg": bank_results["max_bank_deg"],
            "min_segment_margin_ft": flyby_results["min_segment_margin_ft"],
            "database_precision_ok": precision_results["ok"],
            "rnp_ft": self.config.rnp_ft,
            "sector_floor_ft": sector.get("floor_ft"),
            "sector_ceiling_ft": sector.get("ceiling_ft"),
        }

    def _rf_segments(self, points: np.ndarray) -> List[dict]:
        segments = []
        if len(points) < 3:
            return segments
        for index in range(1, len(points) - 1):
            radius = self._turn_radius(points[index - 1], points[index], points[index + 1])
            if radius <= 0:
                continue
            angle = self._course_change(points[index - 1], points[index], points[index + 1])
            if radius <= self.config.min_rf_radius_nm * 6076.12 or abs(angle) >= 3.0:
                segments.append({
                    "index": index,
                    "radius_ft": float(radius),
                    "course_change_deg": float(angle),
                    "altitude_ft": float(points[index, 2]),
                })
        return segments

    def _validate_bank(self, points: np.ndarray, rf_segments: List[dict]) -> dict:
        issues = []
        bank_angles = []
        for segment in rf_segments:
            speed_ft_s = self._estimated_speed_ft_s(points, segment["index"])
            bank = float(np.degrees(np.arctan((speed_ft_s ** 2) / (32.174 * segment["radius_ft"]))))
            bank_angles.append(bank)
            limit = self.config.max_bank_above_deg if segment["altitude_ft"] > self.config.bank_split_altitude_ft else self.config.max_bank_below_deg
            if bank > limit:
                issues.append({
                    "type": "PBN/RNP: превышение крена RF",
                    "severity": "Высокая",
                    "detail": f"Точка {segment['index']}: крен {bank:.1f}°, допустимо {limit:.1f}°",
                })
        return {"issues": issues, "max_bank_deg": float(max(bank_angles)) if bank_angles else 0.0}

    def _validate_flyby_lengths(self, points: np.ndarray, rf_segments: List[dict]) -> dict:
        issues = []
        margins = []
        by_index = {segment["index"]: segment for segment in rf_segments}
        for index in range(1, len(points) - 2):
            start = by_index.get(index)
            end = by_index.get(index + 1)
            if start is None and end is None:
                continue
            segment_length = float(np.linalg.norm(points[index + 1, :2] - points[index, :2]))
            start_dta = self._dta(start) if start else 0.0
            end_dta = self._dta(end) if end else 0.0
            margin = segment_length - (start_dta + end_dta)
            margins.append(margin)
            if margin < 0:
                issues.append({
                    "type": "PBN/RNP: недостаточная длина прямого участка",
                    "severity": "Средняя",
                    "detail": f"Сегмент {index}-{index+1}: не хватает {abs(margin):.0f} ft для DTA",
                })
        return {"issues": issues, "min_segment_margin_ft": float(min(margins)) if margins else 0.0}

    def _validate_database_precision(self, points: np.ndarray) -> dict:
        issues = []
        if len(points) == 0:
            return {"ok": True, "issues": issues}
        rounded = np.round(points / self.config.coordinate_resolution_ft) * self.config.coordinate_resolution_ft
        residual = np.max(np.linalg.norm(points - rounded, axis=1))
        if residual > self.config.coordinate_resolution_ft:
            issues.append({
                "type": "PBN/RNP: точность координат",
                "severity": "Средняя",
                "detail": f"Остаточная погрешность координат {residual:.0f} ft",
            })
        gradients = self._vertical_angles(points)
        if len(gradients):
            angle_residual = np.max(np.abs(gradients - np.round(gradients / self.config.vertical_angle_resolution_deg) * self.config.vertical_angle_resolution_deg))
            if angle_residual > self.config.vertical_angle_resolution_deg:
                issues.append({
                    "type": "PBN/RNP: точность вертикальных углов",
                    "severity": "Низкая",
                    "detail": f"Погрешность угла {angle_residual:.3f}°",
                })
        return {"ok": not issues, "issues": issues}

    def _dta(self, segment: dict) -> float:
        return float(segment["radius_ft"] * np.tan(np.radians(abs(segment["course_change_deg"])) / 2))

    def _turn_radius(self, p1, p2, p3) -> float:
        a = np.linalg.norm(p2[:2] - p1[:2])
        b = np.linalg.norm(p3[:2] - p2[:2])
        c = np.linalg.norm(p3[:2] - p1[:2])
        area = abs(np.cross(p2[:2] - p1[:2], p3[:2] - p1[:2])) / 2
        return float((a * b * c) / (4 * area)) if area > 1e-8 else 0.0

    def _course_change(self, p1, p2, p3) -> float:
        v1 = p2[:2] - p1[:2]
        v2 = p3[:2] - p2[:2]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))

    def _estimated_speed_ft_s(self, points: np.ndarray, index: int) -> float:
        if len(points) < 2:
            return 220.0
        start = max(0, index - 1)
        end = min(len(points) - 1, index + 1)
        return max(80.0, float(np.linalg.norm(points[end, :2] - points[start, :2]) / max(1, end - start)))

    def _vertical_angles(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.array([])
        deltas = np.diff(points, axis=0)
        horizontal = np.linalg.norm(deltas[:, :2], axis=1)
        return np.degrees(np.arctan2(deltas[:, 2], horizontal + 1e-8))
