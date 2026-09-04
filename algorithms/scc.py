import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SpectralCurveClustering:
    """
    Spectral Clustering for Curves (SCC)
    Алгоритм для кластеризации кривых на основе спектральных методов
    """
    
    def __init__(self, n_clusters: int = None, similarity_metric: str = 'polar_curvature',
                 gamma: float = 1.0, affinity_threshold: float = 0.5,
                 max_exact_tracks: int = 700):
        """
        Args:
            n_clusters: количество кластеров (если None - определяется автоматически)
            similarity_metric: метрика сходства ('polar_curvature', 'curvature', 'distance', 'shape')
            gamma: параметр для RBF ядра
            affinity_threshold: порог для пороговой матрицы сходства
        """
        self.n_clusters = n_clusters
        self.similarity_metric = similarity_metric
        self.gamma = gamma
        self.affinity_threshold = affinity_threshold
        self.max_exact_tracks = max_exact_tracks
        self.cluster_labels = None
        
    def compute_curvature_similarity(self, track1, track2) -> float:
        """Вычисление сходства на основе кривизны траекторий"""
        curv1 = track1.get_curvature_signature()
        curv2 = track2.get_curvature_signature()
        
        if len(curv1) == 0 or len(curv2) == 0:
            return 0.0
        
        # Выравниваем длины массивов
        min_len = min(len(curv1), len(curv2))
        if min_len == 0:
            return 0.0
            
        curv1_aligned = self._resample_1d(curv1, min_len)
        curv2_aligned = self._resample_1d(curv2, min_len)
        
        # Корреляция Пирсона как мера сходства
        corr = np.corrcoef(curv1_aligned, curv2_aligned)[0, 1]
        if np.isnan(corr):
            return 0.0
        
        # Преобразуем в [0, 1]
        return max(0, (corr + 1) / 2)

    def _resample_1d(self, values: np.ndarray, target_len: int) -> np.ndarray:
        if len(values) == target_len:
            return values
        old_t = np.linspace(0, 1, len(values))
        new_t = np.linspace(0, 1, target_len)
        return np.interp(new_t, old_t, values)

    def compute_polar_curvature_similarity(self, track1, track2) -> float:
        """Affinity tensor surrogate: direction, spatial shape and polar curvature."""
        curvature_sim = self.compute_curvature_similarity(track1, track2)
        shape_sim = self.compute_shape_similarity(track1, track2)

        direction1 = track1.get_trajectory_vector()
        direction2 = track2.get_trajectory_vector()
        direction_sim = 0.0
        if np.linalg.norm(direction1) > 0 and np.linalg.norm(direction2) > 0:
            direction_sim = (np.dot(direction1, direction2) + 1) / 2

        return float(0.5 * curvature_sim + 0.3 * shape_sim + 0.2 * max(0.0, direction_sim))
    
    def compute_spatial_similarity(self, track1, track2) -> float:
        """Вычисление пространственного сходства траекторий"""
        points1 = track1.get_points_matrix()
        points2 = track2.get_points_matrix()
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # Вычисляем среднее расстояние между точками с выравниванием
        min_len = min(len(points1), len(points2))
        if min_len == 0:
            return 0.0
            
        # Интерполируем траектории для одинаковой длины
        indices1 = np.linspace(0, len(points1)-1, min_len).astype(int)
        indices2 = np.linspace(0, len(points2)-1, min_len).astype(int)
        
        distances = np.linalg.norm(points1[indices1] - points2[indices2], axis=1)
        mean_distance = np.mean(distances)
        
        # Преобразуем расстояние в сходство
        similarity = np.exp(-mean_distance / 100)  # 100 - масштабный коэффициент
        
        return similarity
    
    def compute_shape_similarity(self, track1, track2) -> float:
        """Вычисление сходства формы с использованием моментов"""
        points1 = track1.get_points_matrix()
        points2 = track2.get_points_matrix()
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # Нормализуем траектории
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)
        
        points1_norm = points1 - centroid1
        points2_norm = points2 - centroid2
        
        # Масштабируем
        scale1 = np.max(np.linalg.norm(points1_norm, axis=1))
        scale2 = np.max(np.linalg.norm(points2_norm, axis=1))
        
        points1_norm = points1_norm / (scale1 + 1e-8)
        points2_norm = points2_norm / (scale2 + 1e-8)
        
        # Сравниваем распределения направлений
        dirs1 = np.diff(points1_norm, axis=0)
        dirs2 = np.diff(points2_norm, axis=0)
        
        dirs1_norm = dirs1 / (np.linalg.norm(dirs1, axis=1, keepdims=True) + 1e-8)
        dirs2_norm = dirs2 / (np.linalg.norm(dirs2, axis=1, keepdims=True) + 1e-8)
        
        # Среднее косинусное сходство между направлениями
        similarities = []
        min_len = min(len(dirs1_norm), len(dirs2_norm))
        
        for i in range(min_len):
            sim = np.dot(dirs1_norm[i], dirs2_norm[i])
            similarities.append(max(0, (sim + 1) / 2))
        
        return np.mean(similarities) if similarities else 0.0
    
    def compute_similarity_matrix(self, tracks: Dict) -> np.ndarray:
        """Построение матрицы сходства для всех траекторий"""
        track_list = list(tracks.values())
        n_tracks = len(track_list)
        similarity_matrix = np.zeros((n_tracks, n_tracks))
        
        for i in range(n_tracks):
            for j in range(i, n_tracks):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    if self.similarity_metric == 'polar_curvature':
                        sim = self.compute_polar_curvature_similarity(track_list[i], track_list[j])
                    elif self.similarity_metric == 'curvature':
                        sim = self.compute_curvature_similarity(track_list[i], track_list[j])
                    elif self.similarity_metric == 'distance':
                        sim = self.compute_spatial_similarity(track_list[i], track_list[j])
                    elif self.similarity_metric == 'shape':
                        sim = self.compute_shape_similarity(track_list[i], track_list[j])
                    else:
                        raise ValueError(f"Unknown metric: {self.similarity_metric}")
                    
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Применяем порог
        similarity_matrix[similarity_matrix < self.affinity_threshold] = 0
        
        return similarity_matrix
    
    def determine_optimal_clusters(self, similarity_matrix: np.ndarray, 
                                   max_clusters: int = 10) -> int:
        """Определение оптимального количества кластеров по собственным значениям"""
        # Вычисляем лапласиан
        degree = np.diag(np.sum(similarity_matrix, axis=1))
        laplacian = degree - similarity_matrix
        
        # Вычисляем собственные значения
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)
        
        # Ищем разрыв в собственных значениях
        gaps = np.diff(eigenvalues[:max_clusters+1])
        if len(gaps) > 0:
            optimal_k = np.argmax(gaps) + 1
            return min(optimal_k, max_clusters)
        
        return min(max_clusters, similarity_matrix.shape[0])
    
    def cluster(self, tracks: Dict, verbose: bool = True) -> Dict:
        """
        Кластеризация траекторий методом SCC
        
        Args:
            tracks: словарь {id: Track}
            verbose: вывод информации
            
        Returns:
            словарь кластеров
        """
        if verbose:
            print(f"SCC: Построение матрицы сходства ({self.similarity_metric})...")
        if len(tracks) < 2:
            return {
                0: {
                    'reference': self._compute_cluster_reference(tracks, list(tracks.keys())),
                    'tracks': list(tracks.keys()),
                    'size': len(tracks),
                    'algorithm': f'SCC_{self.similarity_metric}'
                }
            } if tracks else {}

        if len(tracks) > self.max_exact_tracks:
            if verbose:
                print(
                    f"SCC: {len(tracks)} траекторий, включен быстрый режим без полной "
                    f"матрицы сходства {len(tracks)}x{len(tracks)}"
                )
            return self._cluster_large_dataset(tracks)
        
        # Строим матрицу сходства
        similarity_matrix = self.compute_similarity_matrix(tracks)
        
        # Определяем количество кластеров
        if self.n_clusters is None:
            self.n_clusters = self.determine_optimal_clusters(similarity_matrix)
            if verbose:
                print(f"  Оптимальное количество кластеров: {self.n_clusters}")
        
        # Применяем спектральную кластеризацию
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans'
        )
        
        self.cluster_labels = spectral.fit_predict(similarity_matrix)
        
        # Формируем результаты
        track_ids = list(tracks.keys())
        clusters = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_tracks = [track_ids[i] for i, label in enumerate(self.cluster_labels) if label == cluster_id]
            
            if cluster_tracks:
                # Вычисляем опорную траекторию для кластера
                reference = self._compute_cluster_reference(tracks, cluster_tracks)
                
                clusters[cluster_id] = {
                    'reference': reference,
                    'tracks': cluster_tracks,
                    'size': len(cluster_tracks),
                    'algorithm': f'SCC_{self.similarity_metric}'
                }
        
        if verbose:
            print(f"  Сформировано {len(clusters)} кластеров")
            for cid, cdata in clusters.items():
                print(f"    Кластер {cid}: {cdata['size']} траекторий")
        
        return clusters

    def _cluster_large_dataset(self, tracks: Dict) -> Dict:
        track_ids = list(tracks.keys())
        features = np.array([self._track_feature_vector(tracks[track_id]) for track_id in track_ids])
        n_tracks = len(track_ids)
        n_clusters = self.n_clusters or min(12, max(2, int(np.sqrt(n_tracks / 2))))

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(2048, max(256, n_tracks // 4)),
            n_init=5,
            max_iter=100,
        )
        labels = kmeans.fit_predict(features)
        self.cluster_labels = labels

        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_tracks = [track_ids[i] for i, label in enumerate(labels) if label == cluster_id]
            if not cluster_tracks:
                continue
            clusters[cluster_id] = {
                'reference': self._compute_cluster_reference(tracks, cluster_tracks),
                'tracks': cluster_tracks,
                'size': len(cluster_tracks),
                'algorithm': f'SCC_{self.similarity_metric}_fast',
                'quality': 0.0,
                'approximation': 'MiniBatchKMeans по признакам траекторий без O(n²) матрицы SCC'
            }
        return clusters

    def _track_feature_vector(self, track) -> np.ndarray:
        points = track.get_points_matrix()
        if len(points) == 0:
            return np.zeros(17)

        target_len = 5
        old_t = np.linspace(0, 1, len(points))
        new_t = np.linspace(0, 1, target_len)
        sampled = np.zeros((target_len, 3))
        for dim in range(3):
            sampled[:, dim] = np.interp(new_t, old_t, points[:, dim])

        start = sampled[0]
        end = sampled[-1]
        direction = track.get_trajectory_vector()
        centroid = np.mean(points, axis=0)
        spread = np.std(points, axis=0)
        curvature = track.get_curvature_signature()
        curvature_stats = np.array([
            float(np.mean(curvature)) if len(curvature) else 0.0,
            float(np.std(curvature)) if len(curvature) else 0.0,
        ])
        feature = np.concatenate([start, end, direction, centroid, spread, curvature_stats])
        scale = np.linalg.norm(feature) + 1e-8
        return feature / scale
    
    def _compute_cluster_reference(self, tracks: Dict, cluster_tracks: List) -> np.ndarray:
        """Вычисление опорной траектории для кластера"""
        if not cluster_tracks:
            return np.zeros(3)
        
        # Собираем все векторы направлений
        vectors = []
        for track_id in cluster_tracks:
            vec = tracks[track_id].get_trajectory_vector()
            if np.linalg.norm(vec) > 0:
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(3)
        
        vectors = np.array(vectors)
        # Нормализуем и усредняем
        normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        mean_dir = np.mean(normalized, axis=0)
        
        return mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
