"""Sequencing & Merging для оценки эффективности использования ВПП."""

from dataclasses import dataclass
import random
import re
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RunwayOptimizationConfig:
    theoretical_capacity_per_hour: int = 36
    overload_threshold: float = 0.85
    default_required_spacing_sec: int = 120
    heavy_required_spacing_sec: int = 180
    tbo_poly_degree: int = 5
    genetic_population: int = 24
    genetic_generations: int = 30


class RunwayOptimizer:
    """Строит очередность прилета, оценивает интервалы, задержки и переносы потоков."""

    RUNWAY_PATTERN = re.compile(r"^\d{1,2}[LRC]?$", re.IGNORECASE)

    def __init__(self, config: Optional[RunwayOptimizationConfig] = None):
        self.config = config or RunwayOptimizationConfig()

    def optimize(self, tracks_dict: dict, clusters: dict, sectors: dict = None) -> Dict:
        arrivals = self._collect_arrivals(tracks_dict, clusters)
        by_runway = self._group_by_runway(arrivals)
        runway_results = {}
        violations = []

        for runway, runway_arrivals in by_runway.items():
            runway_arrivals = sorted(runway_arrivals, key=lambda item: item["eta_seconds"])
            sequence = [] if runway == "Не указана" else self._sequence(runway_arrivals)
            runway_violations = [item for item in sequence if item["spacing_violation"]]
            violations.extend(runway_violations)
            runway_results[runway] = {
                "arrivals_count": len(runway_arrivals),
                "landings_per_hour": 0 if runway == "Не указана" else self._landings_per_hour(runway_arrivals),
                "utilization": 0.0 if runway == "Не указана" else self._utilization(runway_arrivals),
                "sequence": sequence,
                "violations_count": len(runway_violations),
                "total_delay_seconds": float(sum(item["delay_seconds"] for item in sequence)),
                "avg_delay_seconds": float(np.mean([item["delay_seconds"] for item in sequence])) if sequence else 0.0,
            }

        reassignment = self._dynamic_reassignment(by_runway)
        tbo_routes = self._synthesize_tbo_routes(arrivals, violations)

        return {
            "runways": runway_results,
            "arrivals": arrivals,
            "violations": violations,
            "reassignment": reassignment,
            "tbo_routes": tbo_routes,
            "summary": {
                "runways_count": len(runway_results),
                "arrivals_count": len(arrivals),
                "spacing_violations": len(violations),
                "overloaded_runways": [
                    runway for runway, data in runway_results.items()
                    if runway != "Не указана" and data["utilization"] >= self.config.overload_threshold
                ],
                "total_delay_seconds": float(sum(data["total_delay_seconds"] for data in runway_results.values())),
            },
        }

    def _collect_arrivals(self, tracks_dict: dict, clusters: dict) -> List[dict]:
        arrivals = []
        for cluster_id, cluster in clusters.items():
            for track_id in cluster.get("tracks", []):
                track = tracks_dict.get(track_id)
                if track is None or not getattr(track, "points", None):
                    continue
                runway = self._normalized_runway(track.runwayname)
                eta = self._parse_time(track.trackstart_time) + float(track.points[-1].t)
                optimal_eta = self._optimal_eta(track)
                arrivals.append({
                    "track_id": track_id,
                    "cluster_id": cluster_id,
                    "runway": runway,
                    "eta_seconds": eta,
                    "optimal_eta_seconds": optimal_eta,
                    "delay_seconds": max(0.0, eta - optimal_eta),
                    "wake_category": self._wake_category(track),
                    "aircraft": str(track.aircraft_category or track.aircrafttype or ""),
                    "threshold_point": track.points[-1].to_dict(),
                    "start_point": track.points[0].to_dict(),
                })
        return arrivals

    def _group_by_runway(self, arrivals: List[dict]) -> Dict[str, List[dict]]:
        grouped = {}
        for arrival in arrivals:
            grouped.setdefault(arrival["runway"], []).append(arrival)
        return grouped

    def _sequence(self, arrivals: List[dict]) -> List[dict]:
        sequence = []
        previous = None
        scheduled_eta = None
        for index, arrival in enumerate(arrivals):
            required_spacing = self.config.default_required_spacing_sec
            actual_spacing = None
            spacing_violation = False
            overtaking_risk = False

            if previous is not None:
                required_spacing = self._required_spacing(previous, arrival)
                actual_spacing = arrival["eta_seconds"] - previous["eta_seconds"]
                spacing_violation = actual_spacing < required_spacing
                overtaking_risk = (
                    arrival["optimal_eta_seconds"] < previous["optimal_eta_seconds"]
                    and arrival["eta_seconds"] > previous["eta_seconds"]
                )

            if scheduled_eta is None:
                scheduled_eta = arrival["eta_seconds"]
            else:
                scheduled_eta = max(arrival["eta_seconds"], scheduled_eta + required_spacing)

            sequence.append({
                **arrival,
                "sequence_number": index + 1,
                "required_spacing_seconds": required_spacing,
                "actual_spacing_seconds": actual_spacing,
                "spacing_violation": spacing_violation,
                "overtaking_risk": overtaking_risk,
                "scheduled_eta_seconds": float(scheduled_eta),
                "sequence_delay_seconds": float(max(0.0, scheduled_eta - arrival["eta_seconds"])),
            })
            previous = arrival
        return sequence

    def _landings_per_hour(self, arrivals: List[dict]) -> int:
        if not arrivals:
            return 0
        bins = {}
        for arrival in arrivals:
            hour = int(arrival["eta_seconds"] // 3600)
            bins[hour] = bins.get(hour, 0) + 1
        return max(bins.values()) if bins else 0

    def _utilization(self, arrivals: List[dict]) -> float:
        return min(1.5, self._landings_per_hour(arrivals) / max(1, self.config.theoretical_capacity_per_hour))

    def _required_spacing(self, lead: dict, follower: dict) -> int:
        if lead["wake_category"] == "heavy" and follower["wake_category"] != "heavy":
            return self.config.heavy_required_spacing_sec
        return self.config.default_required_spacing_sec

    def _dynamic_reassignment(self, by_runway: Dict[str, List[dict]]) -> dict:
        candidates = []
        for runway, arrivals in by_runway.items():
            if runway == "Не указана" or self._utilization(arrivals) < self.config.overload_threshold:
                continue
            parallel = self._parallel_runway(runway, by_runway)
            if parallel is None:
                continue
            plan = self._genetic_reassignment(runway, parallel, by_runway[runway], by_runway.get(parallel, []))
            if plan["moved_tracks"]:
                candidates.append(plan)

        return {
            "plans": candidates,
            "status": "Предложены переносы на параллельные ВПП" if candidates else "Перегруженных ВПП с доступной параллельной парой не найдено",
        }

    def _genetic_reassignment(self, source: str, target: str, source_arrivals: List[dict], target_arrivals: List[dict]) -> dict:
        movable = source_arrivals[-min(10, len(source_arrivals)):]
        if not movable:
            return {"source_runway": source, "target_runway": target, "moved_tracks": [], "score": 0.0}

        population = [
            [random.random() < 0.25 for _ in movable]
            for _ in range(self.config.genetic_population)
        ]
        best = min(population, key=lambda genes: self._assignment_cost(genes, movable, source_arrivals, target_arrivals))
        best_cost = self._assignment_cost(best, movable, source_arrivals, target_arrivals)

        for _ in range(self.config.genetic_generations):
            next_population = [best]
            while len(next_population) < self.config.genetic_population:
                parent = random.choice(population)
                child = parent[:]
                mutation_index = random.randrange(len(child))
                child[mutation_index] = not child[mutation_index]
                next_population.append(child)
            population = next_population
            candidate = min(population, key=lambda genes: self._assignment_cost(genes, movable, source_arrivals, target_arrivals))
            candidate_cost = self._assignment_cost(candidate, movable, source_arrivals, target_arrivals)
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost

        moved = [arrival["track_id"] for gene, arrival in zip(best, movable) if gene]
        return {
            "source_runway": source,
            "target_runway": target,
            "moved_tracks": moved,
            "moved_count": len(moved),
            "score": float(best_cost),
            "criterion": "минимизация задержки и прокси-расхода топлива",
        }

    def _assignment_cost(self, genes: List[bool], movable: List[dict], source_arrivals: List[dict], target_arrivals: List[dict]) -> float:
        moved_ids = {arrival["track_id"] for gene, arrival in zip(genes, movable) if gene}
        new_source = [arrival for arrival in source_arrivals if arrival["track_id"] not in moved_ids]
        new_target = target_arrivals + [{**arrival, "runway": "target"} for arrival in movable if arrival["track_id"] in moved_ids]
        delay_cost = self._sequence_delay_cost(new_source) + self._sequence_delay_cost(new_target)
        fuel_proxy = len(moved_ids) * 45.0
        overload_penalty = max(0.0, self._utilization(new_target) - self.config.overload_threshold) * 1000.0
        return delay_cost + fuel_proxy + overload_penalty

    def _sequence_delay_cost(self, arrivals: List[dict]) -> float:
        return float(sum(item["sequence_delay_seconds"] for item in self._sequence(sorted(arrivals, key=lambda a: a["eta_seconds"]))))

    def _synthesize_tbo_routes(self, arrivals: List[dict], violations: List[dict]) -> List[dict]:
        delayed_ids = {item["track_id"] for item in violations}
        routes = []
        for arrival in arrivals:
            if arrival["track_id"] not in delayed_ids:
                continue
            start = np.array([arrival["start_point"]["x"], arrival["start_point"]["y"], arrival["start_point"]["z"]], dtype=float)
            end = np.array([arrival["threshold_point"]["x"], arrival["threshold_point"]["y"], arrival["threshold_point"]["z"]], dtype=float)
            duration = max(60.0, arrival["eta_seconds"] - self._parse_time_from_arrival(arrival))
            coefficients = self._minimum_jerk_coefficients(start, end, duration)
            routes.append({
                "track_id": arrival["track_id"],
                "runway": arrival["runway"],
                "duration_seconds": float(duration),
                "polynomial_degree": self.config.tbo_poly_degree,
                "coefficients": coefficients,
                "purpose": "TBO-маршрут для восстановления интервала прибытия",
            })
        return routes[:20]

    def _minimum_jerk_coefficients(self, start: np.ndarray, end: np.ndarray, duration: float) -> List[List[float]]:
        coeffs = []
        delta = end - start
        for dim in range(3):
            # p(t)=a0+a3*t^3+a4*t^4+a5*t^5, нулевые скорость/ускорение на концах
            a0 = start[dim]
            a1 = 0.0
            a2 = 0.0
            a3 = 10 * delta[dim] / (duration ** 3)
            a4 = -15 * delta[dim] / (duration ** 4)
            a5 = 6 * delta[dim] / (duration ** 5)
            coeffs.append([float(a0), a1, a2, float(a3), float(a4), float(a5)])
        return coeffs

    def _optimal_eta(self, track) -> float:
        points = track.get_points_matrix()
        times = track.get_time_profile()
        if len(points) < 2 or len(times) < 2:
            return self._parse_time(track.trackstart_time) + (times[-1] if len(times) else 0)
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        speed = max(120.0, float(np.percentile(track.get_velocity_profile(), 50)) if len(track.get_velocity_profile()) else 160.0)
        optimal_duration = length / (speed * 1.68781)
        return self._parse_time(track.trackstart_time) + optimal_duration

    def _normalized_runway(self, value) -> str:
        text = str(value or "").strip().upper()
        if text in ("", "NAN", "NONE", "NULL"):
            return "Не указана"
        return text if self.RUNWAY_PATTERN.match(text) else "Не указана"

    def _parallel_runway(self, runway: str, by_runway: Dict[str, List[dict]]) -> Optional[str]:
        match = re.match(r"^(\d{1,2})([LRC])$", runway, re.IGNORECASE)
        if not match:
            return None
        number, side = match.groups()
        candidates = {"L": ["R", "C"], "R": ["L", "C"], "C": ["L", "R"]}.get(side.upper(), [])
        for candidate_side in candidates:
            candidate = f"{number}{candidate_side}"
            if candidate in by_runway:
                return candidate
        return None

    def _wake_category(self, track) -> str:
        text = f"{track.aircraft_category or ''} {track.aircrafttype or ''}".upper()
        heavy_markers = ("B744", "B747", "B748", "B763", "B764", "B777", "B787", "A330", "A340", "A350", "A380", "HEAVY")
        return "heavy" if any(marker in text for marker in heavy_markers) else "medium"

    def _parse_time(self, value: str) -> float:
        text = str(value or "").strip()
        try:
            if ":" in text:
                parts = [float(part) for part in text.split(":")]
                if len(parts) == 3:
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
        except ValueError:
            pass
        return 0.0

    def _parse_time_from_arrival(self, arrival: dict) -> float:
        threshold = arrival["threshold_point"]
        return float(arrival["eta_seconds"] - threshold.get("t", 0.0))
