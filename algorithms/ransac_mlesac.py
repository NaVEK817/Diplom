import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize


@dataclass
class RANSACConfig:
    distance_threshold: float = 0.05
    max_iterations: int = 1000
    min_inlier_ratio: float = 0.3
    min_sample_size: int = 2
    line_distance_threshold: float = 0.5
    max_line_iterations: int = 300
    min_cluster_tracks: int = 2


@dataclass
class MLESACConfig(RANSACConfig):
    outlier_ratio: float = 0.3
    outlier_prob: float = 0.1


class RANSACClustering:
    """Robust extraction of trajectory bundles by 2D asymptote consensus."""

    algorithm_name = "RANSAC"

    def __init__(self, config: Optional[RANSACConfig] = None):
        self.config = config or RANSACConfig()

    def cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        return float(1 - np.dot(v1_norm, v2_norm))

    def fit_reference_direction(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 0:
            return np.zeros(3)

        normalized = normalize(vectors, norm="l2")
        mean_dir = np.mean(normalized, axis=0)
        return mean_dir / (np.linalg.norm(mean_dir) + 1e-8)

    def _collect_projected_points(
        self, trajectories: Dict, track_ids: List[str]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        points = []
        projected_by_track = {}

        for track_id in track_ids:
            matrix = trajectories[track_id].get_points_matrix()
            if len(matrix) < 2:
                continue

            projected = matrix[:, :2].astype(float)
            projected_by_track[track_id] = projected
            points.extend(projected)

        if not points:
            return np.empty((0, 2)), projected_by_track

        points = np.array(points)
        center = np.mean(points, axis=0)
        scale = np.std(points, axis=0) + 1e-8
        normalized_points = (points - center) / scale
        normalized_by_track = {
            track_id: (projected - center) / scale
            for track_id, projected in projected_by_track.items()
        }
        return normalized_points, normalized_by_track

    def _track_line_distance(
        self, points: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray
    ) -> float:
        offsets = points - line_point
        distances = np.abs(offsets[:, 0] * line_direction[1] - offsets[:, 1] * line_direction[0])
        return float(np.median(distances))

    def _line_direction_3d(
        self, line_direction: np.ndarray, trajectories: Dict, track_ids: List[str]
    ) -> np.ndarray:
        z_directions = []
        for track_id in track_ids:
            vector = trajectories[track_id].get_trajectory_vector()
            if np.linalg.norm(vector) > 0:
                z_directions.append(vector[2])

        z_component = float(np.median(z_directions)) if z_directions else 0.0
        direction = np.array([line_direction[0], line_direction[1], z_component])
        norm = np.linalg.norm(direction)
        return direction / (norm + 1e-8) if norm > 0 else np.zeros(3)

    def _line_distances(self, points: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray) -> np.ndarray:
        offsets = points - line_point
        return np.abs(offsets[:, 0] * line_direction[1] - offsets[:, 1] * line_direction[0])

    def _score_line(self, distances: np.ndarray) -> float:
        return float(np.sum(distances <= self.config.line_distance_threshold))

    def _fit_asymptote(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        if len(points) < 2:
            return None, None, -np.inf

        best_point = None
        best_direction = None
        best_score = -np.inf
        iterations = min(self.config.max_line_iterations, self.config.max_iterations)

        for _ in range(iterations):
            i, j = random.sample(range(len(points)), 2)
            p1 = points[i]
            p2 = points[j]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                continue

            direction = direction / norm
            score = self._score_line(self._line_distances(points, p1, direction))
            if score > best_score:
                best_score = score
                best_point = p1
                best_direction = direction

        return best_point, best_direction, best_score

    def _select_inliers(
        self,
        available: List[str],
        projected_by_track: Dict[str, np.ndarray],
        vectors: Dict[str, np.ndarray],
        reference_3d: np.ndarray,
        line_point: np.ndarray,
        line_direction: np.ndarray,
    ) -> Tuple[List[str], Dict[str, float]]:
        inliers = []
        line_distances = {}

        for track_id in available:
            points = projected_by_track.get(track_id)
            if points is None or len(points) < 2:
                continue

            line_distance = self._track_line_distance(points, line_point, line_direction)
            cosine_dist = self.cosine_distance(vectors[track_id], reference_3d)
            line_distances[track_id] = line_distance

            if (
                line_distance <= self.config.line_distance_threshold
                and cosine_dist <= self.config.distance_threshold * 6
            ):
                inliers.append(track_id)

        if len(inliers) < self.config.min_cluster_tracks and available:
            inliers = [min(available, key=lambda tid: line_distances.get(tid, float("inf")))]

        return inliers, line_distances

    def cluster(self, trajectories: Dict, verbose: bool = True) -> Dict:
        vectors = {
            track_id: track.get_trajectory_vector()
            for track_id, track in trajectories.items()
            if np.linalg.norm(track.get_trajectory_vector()) > 0
        }

        if verbose:
            print(f"{self.algorithm_name}: processing {len(vectors)} trajectories")

        clusters = {}
        used_tracks = set()
        cluster_id = 0

        while len(used_tracks) < len(vectors):
            available = [tid for tid in vectors if tid not in used_tracks]
            if not available:
                break

            projected_points, projected_by_track = self._collect_projected_points(trajectories, available)
            line_point, line_direction, line_score = self._fit_asymptote(projected_points)
            if line_point is None or line_direction is None:
                break

            reference_3d = self._line_direction_3d(line_direction, trajectories, available)
            inliers, distances = self._select_inliers(
                available, projected_by_track, vectors, reference_3d, line_point, line_direction
            )
            if not inliers:
                break

            refined_reference = self.fit_reference_direction(np.array([vectors[tid] for tid in inliers]))
            clusters[cluster_id] = {
                "reference": refined_reference,
                "tracks": inliers,
                "size": len(inliers),
                "algorithm": self.algorithm_name,
                "asymptote_point": line_point,
                "asymptote_direction": line_direction,
                "line_score": float(line_score),
                "mean_line_distance": float(np.mean([distances.get(tid, 0.0) for tid in inliers])),
            }

            used_tracks.update(inliers)
            if verbose:
                print(f"  cluster {cluster_id}: {len(inliers)} trajectories")
            cluster_id += 1

        return clusters


class MLESACClustering(RANSACClustering):
    """Likelihood-based RANSAC variant for the same asymptote extraction task."""

    algorithm_name = "MLESAC"

    def __init__(self, config: Optional[MLESACConfig] = None):
        super().__init__(config or MLESACConfig())

    @property
    def mlesac_config(self) -> MLESACConfig:
        return self.config

    def _score_line(self, distances: np.ndarray) -> float:
        variance = self.config.line_distance_threshold ** 2
        inlier_likelihood = np.exp(-(distances ** 2) / (2 * variance + 1e-8))
        mixed = (
            (1 - self.mlesac_config.outlier_ratio) * inlier_likelihood
            + self.mlesac_config.outlier_ratio * self.mlesac_config.outlier_prob
        )
        return float(np.sum(np.log(mixed + 1e-8)))
