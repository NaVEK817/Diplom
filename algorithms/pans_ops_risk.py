"""Правила детального распознавания рисков по мотивам PANS-OPS."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PansOpsRiskConfig:
    optimal_glide_gradient: float = 0.052
    max_gradient_ab: float = 0.037
    max_gradient_cd: float = 0.035
    gradient_tolerance: float = 0.05
    final_speed_margin_kt: float = 20.0
    if_speed_margin_kt: float = 30.0
    bank_altitude_split_ft: float = 397.0  # 121 м
    max_bank_above_deg: float = 25.0
    max_bank_below_deg: float = 8.0
    impossible_acceleration_ft_s2: float = 96.5  # около 3g
    svd_residual_threshold_ft: float = 1500.0
    min_segment_seconds: float = 1.0


class PansOpsRiskDetector:
    """Проверяет траекторию по жестким эксплуатационным критериям."""

    VAT_LIMITS = {
        "A": (0, 90),
        "B": (91, 120),
        "C": (121, 140),
        "D": (141, 165),
        "E": (166, 210),
    }

    def __init__(self, config: PansOpsRiskConfig = None):
        self.config = config or PansOpsRiskConfig()

    def evaluate_tracks(self, tracks: List, track_ids: List[str]) -> Dict[str, dict]:
        return {
            track_id: self.evaluate_track(track)
            for track_id, track in zip(track_ids, tracks)
        }

    def evaluate_track(self, track) -> dict:
        points = track.get_points_matrix()
        times = track.get_time_profile()
        speeds = track.get_velocity_profile()
        category = self._aircraft_category(track, speeds)

        checks = [
            self._check_descent_profile(points, category),
            self._check_speed_stabilization(speeds, category),
            self._check_bank_angles(points, times, speeds),
            self._check_radar_failures(points, times),
        ]
        triggered = [risk for check in checks for risk in check["risks"]]
        max_severity = self._max_severity(triggered)

        return {
            "category": category,
            "risk_count": len(triggered),
            "max_severity": max_severity,
            "risks": triggered,
            "checks": {check["name"]: check for check in checks},
        }

    def _check_descent_profile(self, points: np.ndarray, category: str) -> dict:
        risks = []
        gradients = []
        if len(points) >= 2:
            deltas = np.diff(points, axis=0)
            horizontal = np.linalg.norm(deltas[:, :2], axis=1)
            vertical_drop = -deltas[:, 2]
            valid = horizontal > 1e-8
            gradients = (vertical_drop[valid] / horizontal[valid]).tolist()

        max_gradient = self.config.max_gradient_ab if category in ("A", "B") else self.config.max_gradient_cd
        threshold = max_gradient * (1 + self.config.gradient_tolerance)
        excessive = [gradient for gradient in gradients if gradient > threshold]
        if excessive:
            risks.append({
                "type": "Невыдерживание профиля",
                "severity": "Высокая",
                "detail": f"Градиент снижения до {max(excessive):.1%}, допустимо {threshold:.1%}",
            })

        if len(points) >= 2:
            oas_risk = self._below_protection_surface(points)
            if oas_risk is not None:
                risks.append(oas_risk)

        return {
            "name": "profile",
            "risks": risks,
            "max_gradient": float(max(gradients)) if gradients else 0.0,
            "allowed_gradient": float(threshold),
        }

    def _below_protection_surface(self, points: np.ndarray):
        runway_point = points[-1]
        horizontal_to_threshold = np.linalg.norm(points[:, :2] - runway_point[:2], axis=1)
        protected_altitude = runway_point[2] + horizontal_to_threshold * self.config.optimal_glide_gradient
        below = protected_altitude - points[:, 2]
        max_below = float(np.max(below)) if len(below) else 0.0
        if max_below > 150.0:
            return {
                "type": "Невыдерживание профиля",
                "severity": "Высокая",
                "detail": f"Траектория ниже расчетной защитной поверхности на {max_below:.0f} ft",
            }
        return None

    def _check_speed_stabilization(self, speeds: np.ndarray, category: str) -> dict:
        risks = []
        lower, upper = self.VAT_LIMITS.get(category, self.VAT_LIMITS["C"])
        if len(speeds) == 0:
            return {"name": "speed", "risks": risks, "vat_range": [lower, upper], "max_final_speed": 0.0}

        final_segment = speeds[max(0, int(len(speeds) * 0.8)):]
        intermediate_segment = speeds[int(len(speeds) * 0.3): max(int(len(speeds) * 0.7), 1)]
        max_final = float(np.max(final_segment)) if len(final_segment) else float(np.max(speeds))
        max_intermediate = float(np.max(intermediate_segment)) if len(intermediate_segment) else max_final

        if max_final > upper + self.config.final_speed_margin_kt:
            risks.append({
                "type": "Нестабилизированный заход",
                "severity": "Средняя",
                "detail": f"Скорость на конечном участке {max_final:.0f} kt, Vat max {upper} kt + 20 kt",
            })
        if max_intermediate > upper + self.config.if_speed_margin_kt:
            risks.append({
                "type": "Превышение скорости на промежуточном участке",
                "severity": "Средняя",
                "detail": f"Скорость IF {max_intermediate:.0f} kt, расчетный предел {upper + self.config.if_speed_margin_kt:.0f} kt",
            })

        return {
            "name": "speed",
            "risks": risks,
            "vat_range": [lower, upper],
            "max_final_speed": max_final,
            "max_intermediate_speed": max_intermediate,
        }

    def _check_bank_angles(self, points: np.ndarray, times: np.ndarray, speeds: np.ndarray) -> dict:
        risks = []
        bank_angles = []
        if len(points) < 3:
            return {"name": "bank", "risks": risks, "max_bank_deg": 0.0}

        for index in range(1, len(points) - 1):
            radius = self._turn_radius(points[index - 1], points[index], points[index + 1])
            if radius <= 0:
                continue
            speed_kt = speeds[index] if index < len(speeds) else np.mean(speeds) if len(speeds) else 0
            speed_ft_s = speed_kt * 1.68781
            bank_deg = float(np.degrees(np.arctan((speed_ft_s ** 2) / (32.174 * radius))))
            bank_angles.append(bank_deg)
            limit = self.config.max_bank_above_deg if points[index, 2] > self.config.bank_altitude_split_ft else self.config.max_bank_below_deg
            if bank_deg > limit:
                risks.append({
                    "type": "Опасное маневрирование",
                    "severity": "Высокая" if bank_deg > limit * 1.25 else "Средняя",
                    "detail": f"Расчетный крен {bank_deg:.1f}°, допустимо {limit:.1f}°",
                })
                break

        return {
            "name": "bank",
            "risks": risks,
            "max_bank_deg": float(max(bank_angles)) if bank_angles else 0.0,
        }

    def _turn_radius(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        a = np.linalg.norm(p2[:2] - p1[:2])
        b = np.linalg.norm(p3[:2] - p2[:2])
        c = np.linalg.norm(p3[:2] - p1[:2])
        area = abs(np.cross(p2[:2] - p1[:2], p3[:2] - p1[:2])) / 2
        if area < 1e-8:
            return 0.0
        return float((a * b * c) / (4 * area))

    def _check_radar_failures(self, points: np.ndarray, times: np.ndarray) -> dict:
        risks = []
        max_acceleration = 0.0
        if len(points) >= 3 and len(times) >= 3:
            dt = np.diff(times)
            dt[dt < self.config.min_segment_seconds] = self.config.min_segment_seconds
            velocities = np.diff(points, axis=0) / dt[:, None]
            acc_dt = dt[1:]
            accelerations = np.diff(velocities, axis=0) / acc_dt[:, None]
            accel_norms = np.linalg.norm(accelerations, axis=1)
            max_acceleration = float(np.max(accel_norms)) if len(accel_norms) else 0.0
            if max_acceleration > self.config.impossible_acceleration_ft_s2:
                risks.append({
                    "type": "Подозрение на технический сбой",
                    "severity": "Высокая",
                    "detail": f"Физически невозможное ускорение {max_acceleration:.1f} ft/s²",
                })

        svd_residual = self._svd_residual(points)
        if svd_residual > self.config.svd_residual_threshold_ft:
            risks.append({
                "type": "Подозрение на технический сбой",
                "severity": "Средняя",
                "detail": f"SVD-остаток восстановления {svd_residual:.0f} ft",
            })

        return {
            "name": "radar",
            "risks": risks,
            "max_acceleration_ft_s2": max_acceleration,
            "svd_residual_ft": float(svd_residual),
        }

    def _svd_residual(self, points: np.ndarray) -> float:
        if len(points) < 4:
            return 0.0
        centered = points - np.mean(points, axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        rank = min(2, len(singular_values))
        reconstructed = (centered @ vh[:rank].T) @ vh[:rank]
        residuals = np.linalg.norm(centered - reconstructed, axis=1)
        return float(np.percentile(residuals, 95))

    def _aircraft_category(self, track, speeds: np.ndarray) -> str:
        category = str(track.aircraft_category or "").strip().upper()
        for item in ("A", "B", "C", "D", "E"):
            if category == item or category.endswith(item):
                return item

        vat = float(np.percentile(speeds, 20)) if len(speeds) else 130.0
        for item, (lower, upper) in self.VAT_LIMITS.items():
            if lower <= vat <= upper:
                return item
        return "C"

    def _max_severity(self, risks: List[dict]) -> str:
        order = {"Нет": 0, "Низкая": 1, "Средняя": 2, "Высокая": 3}
        if not risks:
            return "Нет"
        return max((risk["severity"] for risk in risks), key=lambda value: order.get(value, 0))
