import numpy as np
from sklearn.cluster import MeanShift as SklearnMeanShift
from sklearn.cluster import estimate_bandwidth
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class MeanShiftConfig:
    bandwidth: float = None  # если None - оценивается автоматически
    max_iterations: int = 300
    min_cluster_size: int = 3
    bin_seeding: bool = True
    use_density_weights: bool = True

class MeanShiftClustering:
    """
    Mean Shift алгоритм для кластеризации траекторий
    Использует плотность направлений для выделения пучков
    """
    
    def __init__(self, config: Optional[MeanShiftConfig] = None):
        self.config = config or MeanShiftConfig()
        
    def compute_feature_vectors(self, trajectories: Dict) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Build an abstract feature space where Mean Shift searches density modes."""
        features = []
        directions = []
        track_ids = []
        
        for track_id, track in trajectories.items():
            direction = track.get_trajectory_vector()
            points = track.get_points_matrix()
            if np.linalg.norm(direction) == 0 or len(points) < 2:
                continue

            centroid = np.mean(points, axis=0)
            start = points[0]
            end = points[-1]
            curvature = track.get_curvature_signature()
            curvature_mean = float(np.mean(curvature)) if len(curvature) else 0.0
            curvature_std = float(np.std(curvature)) if len(curvature) else 0.0

            features.append([
                direction[0], direction[1], direction[2],
                start[0], start[1], start[2],
                end[0], end[1], end[2],
                centroid[0], centroid[1], centroid[2],
                curvature_mean, curvature_std,
                len(points),
            ])
            directions.append(direction)
            track_ids.append(track_id)
        
        if not features:
            return np.array([]), [], np.array([])
        
        features = StandardScaler().fit_transform(np.array(features))
        return features, track_ids, np.array(directions)

    def compute_density_weighted_directions(self, trajectories: Dict) -> Tuple[np.ndarray, np.ndarray]:
        features, track_ids, _ = self.compute_feature_vectors(trajectories)
        return features, track_ids
    
    def estimate_bandwidth(self, vectors: np.ndarray) -> float:
        """Оценка полосы пропускания для Mean Shift"""
        if self.config.bandwidth is not None:
            return self.config.bandwidth
        
        if len(vectors) > 1:
            bandwidth = estimate_bandwidth(vectors, quantile=0.2, n_samples=min(len(vectors), 500))
            if bandwidth and bandwidth > 0:
                return float(max(bandwidth, 0.05))
        
        return 0.1  # Значение по умолчанию
    
    def cluster(self, trajectories: Dict, verbose: bool = True) -> Dict:
        """
        Кластеризация траекторий методом Mean Shift
        
        Args:
            trajectories: словарь {id: Track}
            verbose: вывод информации
            
        Returns:
            словарь кластеров
        """
        if verbose:
            print(f"Mean Shift: Кластеризация траекторий...")
        
        # Mean Shift работает в абстрактном пространстве признаков траектории.
        feature_vectors, track_ids, direction_vectors = self.compute_feature_vectors(trajectories)
        
        if len(feature_vectors) == 0:
            return {}
        
        # Оцениваем bandwidth
        bandwidth = self.estimate_bandwidth(feature_vectors)
        
        if verbose:
            print(f"  Оцененная ширина полосы: {bandwidth:.3f}")
        
        # Применяем Mean Shift
        mean_shift = SklearnMeanShift(
            bandwidth=bandwidth,
            max_iter=self.config.max_iterations,
            bin_seeding=self.config.bin_seeding,
            cluster_all=False
        )
        
        cluster_labels = mean_shift.fit_predict(feature_vectors)
        
        # Формируем результаты
        clusters = {}
        cluster_centers = mean_shift.cluster_centers_
        
        for cluster_id in range(len(cluster_centers)):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_track_ids = [track_ids[i] for i in cluster_indices]
            
            # Фильтруем маленькие кластеры
            if len(cluster_track_ids) >= self.config.min_cluster_size:
                reference = np.mean(direction_vectors[cluster_indices], axis=0)
                reference = reference / (np.linalg.norm(reference) + 1e-8)
                
                clusters[cluster_id] = {
                    'reference': reference,
                    'tracks': cluster_track_ids,
                    'size': len(cluster_track_ids),
                    'algorithm': 'MeanShift',
                    'bandwidth': bandwidth,
                    'density': len(cluster_indices) / len(feature_vectors)
                }
        
        # Объединяем некластеризованные траектории в отдельные кластеры?
        unclustered = np.where(cluster_labels == -1)[0]
        if len(unclustered) > 0:
            cluster_id = len(clusters)
            cluster_track_ids = [track_ids[i] for i in unclustered]
            
            if len(cluster_track_ids) >= self.config.min_cluster_size:
                # Для некластеризованных вычисляем среднее направление
                unclustered_vectors = direction_vectors[unclustered]
                reference = np.mean(unclustered_vectors, axis=0)
                reference = reference / (np.linalg.norm(reference) + 1e-8)
                
                clusters[cluster_id] = {
                    'reference': reference,
                    'tracks': cluster_track_ids,
                    'size': len(cluster_track_ids),
                    'algorithm': 'MeanShift',
                    'bandwidth': bandwidth,
                    'is_noise': True
                }
        
        if verbose:
            print(f"  Сформировано {len(clusters)} кластеров")
            for cid, cdata in clusters.items():
                noise_tag = " (шум)" if cdata.get('is_noise', False) else ""
                print(f"    Кластер {cid}: {cdata['size']} траекторий{noise_tag}")
        
        return clusters
    
    def adaptive_mean_shift(self, trajectories: Dict, max_clusters: int = 10) -> Dict:
        """
        Адаптивный Mean Shift с динамическим изменением bandwidth
        для поиска оптимального количества кластеров
        """
        if not trajectories:
            return {}
        
        best_clusters = {}
        best_quality = -1
        best_bandwidth = None
        
        # Пробуем разные значения bandwidth
        bandwidth_range = np.linspace(0.05, 0.5, 10)
        
        for bandwidth in bandwidth_range:
            self.config.bandwidth = bandwidth
            clusters = self.cluster(trajectories, verbose=False)
            
            if len(clusters) == 0 or len(clusters) > max_clusters:
                continue
            
            # Оцениваем качество кластеризации
            quality = self._evaluate_clustering_quality(clusters)
            
            if quality > best_quality:
                best_quality = quality
                best_clusters = clusters
                best_bandwidth = bandwidth
        
        print(f"  Лучшая ширина полосы: {best_bandwidth:.3f} (качество: {best_quality:.3f})")
        
        return best_clusters
    
    def _evaluate_clustering_quality(self, clusters: Dict) -> float:
        """Оценка качества кластеризации"""
        if not clusters:
            return 0.0
        
        # Критерии качества:
        # 1. Компактность кластеров
        # 2. Разделимость кластеров
        # 3. Сбалансированность размеров
        
        cluster_sizes = [c['size'] for c in clusters.values()]
        
        # Компактность (средняя плотность)
        compactness = np.mean([c.get('density', 0) for c in clusters.values()])
        
        # Сбалансированность (обратная дисперсия)
        if len(cluster_sizes) > 1:
            size_variance = np.var(cluster_sizes)
            balance = 1 / (1 + size_variance / len(cluster_sizes))
        else:
            balance = 1.0
        
        # Комбинированная оценка
        quality = 0.6 * compactness + 0.4 * balance
        
        return quality
