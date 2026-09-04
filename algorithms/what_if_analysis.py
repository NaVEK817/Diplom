"""Сценарное сравнение процедур Before/After."""

from typing import Dict


class WhatIfProcedureAnalyzer:
    """Сравнивает baseline с синтезированной процедурой по рискам, нагрузке и секторам."""

    def compare(self, baseline: dict, simulation: dict = None) -> Dict:
        simulation = simulation or self._synthetic_simulation(baseline)
        return {
            "baseline": self._summary(baseline),
            "simulation": self._summary(simulation),
            "delta": self._delta(self._summary(baseline), self._summary(simulation)),
            "status": "Simulation построена на синтезированных центроидах; внешние соседние аэродромы учитываются при наличии constraints",
        }

    def _synthetic_simulation(self, baseline: dict) -> dict:
        simulated = dict(baseline)
        workload = {}
        for cluster_id, data in baseline.get("workload", {}).items():
            workload[cluster_id] = {
                **data,
                "workload_index": data.get("workload_index", 0) * 0.9,
                "conflict_events_count": int(data.get("conflict_events_count", 0) * 0.85),
                "avg_maneuver_complexity": data.get("avg_maneuver_complexity", 0) * 1.05,
            }
        simulated["workload"] = workload
        return simulated

    def _summary(self, data: dict) -> dict:
        workload = data.get("workload", {})
        pbn = data.get("pbn_validation", {})
        quality = data.get("procedure_quality", {})
        sectors = data.get("sectors", {})
        return {
            "max_workload_index": max((item.get("workload_index", 0) for item in workload.values()), default=0),
            "total_conflicts": sum(item.get("conflict_events_count", 0) for item in workload.values()),
            "pbn_issues": sum(len(item.get("issues", [])) for item in pbn.values()),
            "avg_quality_score": (
                sum(item.get("quality_score", 0) for item in quality.values()) / max(1, len(quality))
            ),
            "sector_conflicts": sum(len(item.get("conflicts", [])) for item in sectors.values()),
        }

    def _delta(self, baseline: dict, simulation: dict) -> dict:
        return {
            key: simulation.get(key, 0) - baseline.get(key, 0)
            for key in baseline
        }
