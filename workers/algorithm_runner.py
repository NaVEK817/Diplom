"""Background worker for running clustering algorithms."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    factory: Callable[[], object]


def default_algorithm_specs(settings: Optional[dict] = None) -> list[AlgorithmSpec]:
    settings = settings or {}
    from algorithms.lscc import LaplacianSCC
    from algorithms.mean_shift import MeanShiftClustering, MeanShiftConfig
    from algorithms.ransac_mlesac import MLESACClustering, MLESACConfig, RANSACClustering, RANSACConfig
    from algorithms.scc import SpectralCurveClustering

    ransac_config = RANSACConfig(
        distance_threshold=settings.get("ransac_distance_threshold", settings.get("ransac_threshold", 0.05)),
        max_iterations=settings.get("ransac_max_iterations", 1000),
        line_distance_threshold=settings.get("ransac_line_distance_threshold", 0.5),
        max_line_iterations=settings.get("ransac_max_line_iterations", 300),
        min_cluster_tracks=settings.get("ransac_min_cluster_tracks", 2),
    )
    mlesac_config = MLESACConfig(
        distance_threshold=settings.get("ransac_distance_threshold", settings.get("ransac_threshold", 0.05)),
        max_iterations=settings.get("ransac_max_iterations", 1000),
        line_distance_threshold=settings.get("ransac_line_distance_threshold", 0.5),
        max_line_iterations=settings.get("ransac_max_line_iterations", 300),
        min_cluster_tracks=settings.get("ransac_min_cluster_tracks", 2),
        outlier_ratio=settings.get("mlesac_outlier_ratio", 0.3),
        outlier_prob=settings.get("mlesac_outlier_prob", 0.1),
    )
    bandwidth = settings.get("mean_shift_bandwidth", 0.0)
    mean_shift_config = MeanShiftConfig(
        bandwidth=bandwidth if bandwidth and bandwidth > 0 else None,
        max_iterations=settings.get("mean_shift_max_iterations", 300),
        min_cluster_size=settings.get("mean_shift_min_cluster_size", 3),
        bin_seeding=settings.get("mean_shift_bin_seeding", True),
    )

    return [
        AlgorithmSpec("RANSAC", lambda: RANSACClustering(ransac_config)),
        AlgorithmSpec("MLESAC", lambda: MLESACClustering(mlesac_config)),
        AlgorithmSpec("SCC", lambda: SpectralCurveClustering(
            n_clusters=settings.get("scc_n_clusters") or None,
            gamma=settings.get("scc_gamma", 1.0),
            affinity_threshold=settings.get("scc_affinity_threshold", 0.5),
            max_exact_tracks=settings.get("scc_max_exact_tracks", 700),
        )),
        AlgorithmSpec("LSCC", lambda: LaplacianSCC(
            n_clusters=settings.get("lscc_n_clusters") or None,
            n_scales=settings.get("lscc_n_scales", 3),
            affinity_metric=settings.get("lscc_affinity_metric", "hybrid"),
            use_normalized_laplacian=settings.get("lscc_use_normalized_laplacian", True),
        )),
        AlgorithmSpec("Сдвиг среднего", lambda: MeanShiftClustering(mean_shift_config)),
    ]


class AlgorithmRunnerWorker(QThread):
    """Runs configured clustering algorithms without knowing concrete classes."""

    finished = pyqtSignal(dict)
    progress = pyqtSignal(str, int)
    debug = pyqtSignal(str)
    error = pyqtSignal(str)
    algorithm_done = pyqtSignal(str, dict)

    def __init__(
        self,
        tracks_dict: dict,
        algorithm_specs: Optional[Iterable[AlgorithmSpec]] = None,
        debug_enabled: bool = False,
        max_tracks: Optional[int] = None,
        settings: Optional[dict] = None
    ):
        super().__init__()
        self.original_tracks_count = len(tracks_dict)
        self.tracks_dict = self._limited_tracks(tracks_dict, max_tracks)
        self.algorithm_specs = list(algorithm_specs) if algorithm_specs else default_algorithm_specs(settings)
        self.debug_enabled = debug_enabled
        self.max_tracks = max_tracks
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        results = {}

        for spec in self.algorithm_specs:
            if self._is_cancelled:
                break

            self.progress.emit(spec.name, 0)
            self._debug(f"{spec.name}: старт, траекторий={len(self.tracks_dict)}")
            start_time = time.time()

            try:
                result = self._run_algorithm(spec)
                if self._is_cancelled:
                    break
                result["time"] = time.time() - start_time
                results[spec.name] = result
                self.algorithm_done.emit(spec.name, result)
                self.progress.emit(spec.name, 100)
                self._debug(f"{spec.name}: завершен за {result['time']:.1f} сек, кластеров={result.get('num_clusters', 0)}")
            except Exception as error:
                self.error.emit(f"{spec.name}: {error}")
                results[spec.name] = {"error": str(error), "clusters": {}}
                self._debug(f"{spec.name}: ошибка {error}")

        self.finished.emit(results)

    def _limited_tracks(self, tracks_dict: dict, max_tracks: Optional[int]) -> dict:
        if max_tracks is None or max_tracks <= 0 or len(tracks_dict) <= max_tracks:
            return tracks_dict

        items = list(tracks_dict.items())
        step_indices = np.linspace(0, len(items) - 1, max_tracks, dtype=int)
        return {items[index][0]: items[index][1] for index in step_indices}

    def _run_algorithm(self, spec: AlgorithmSpec) -> dict:
        clusterer = spec.factory()
        if self.original_tracks_count != len(self.tracks_dict):
            self._debug(
                f"{spec.name}: используется выборка {len(self.tracks_dict)} "
                f"из {self.original_tracks_count} траекторий"
            )
        self._debug(f"{spec.name}: построение кластеров")
        clusters = clusterer.cluster(self.tracks_dict, verbose=False)
        self._debug(f"{spec.name}: подготовка результата")
        return self._prepare_result(clusters)

    def _debug(self, message: str):
        if self.debug_enabled:
            self.debug.emit(message)

    def _prepare_result(self, clusters: Dict) -> dict:
        total_tracks = sum(cluster["size"] for cluster in clusters.values())
        num_clusters = len(clusters)
        qualities = [cluster.get("quality", 0) for cluster in clusters.values()]
        avg_quality = np.mean(qualities) if qualities else 0
        avg_size = total_tracks / num_clusters if num_clusters > 0 else 0

        return {
            "clusters": clusters,
            "num_clusters": num_clusters,
            "total_tracks": total_tracks,
            "avg_quality": float(avg_quality),
            "avg_size": float(avg_size),
        }
