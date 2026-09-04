import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from algorithms.scc import SpectralCurveClustering


class LaplacianSCC(SpectralCurveClustering):
    """
    Laplacian Spectral Clustering for Curves (LSCC)
    Улучшенная версия SCC с использованием нормализованного лапласиана
    и многомасштабного анализа
    """
    
    def __init__(self, n_clusters: int = None, n_scales: int = 3,
                 affinity_metric: str = 'hybrid', 
                 use_normalized_laplacian: bool = True):
        """
        Args:
            n_clusters: количество кластеров
            n_scales: количество масштабов для многомасштабного анализа
            affinity_metric: метрика сходства ('hybrid', 'curvature', 'spatial')
            use_normalized_laplacian: использовать нормализованный лапласиан
        """
        super().__init__(
            n_clusters=n_clusters,
            similarity_metric='polar_curvature' if affinity_metric == 'hybrid' else affinity_metric,
            affinity_threshold=0.0
        )
        self.n_scales = n_scales
        self.affinity_metric = affinity_metric
        self.use_normalized_laplacian = use_normalized_laplacian
        
    def compute_multiscale_affinity(self, track1, track2) -> float:
        """Вычисление многомасштабной меры сходства"""
        if self.affinity_metric == 'curvature':
            return self._compute_curvature_affinity(track1, track2)
        elif self.affinity_metric == 'spatial':
            return self._compute_spatial_affinity(track1, track2)
        else:  # hybrid
            curv_aff = self._compute_curvature_affinity(track1, track2)
            spat_aff = self._compute_spatial_affinity(track1, track2)
            return 0.6 * curv_aff + 0.4 * spat_aff  # взвешенная сумма
    
    def _compute_curvature_affinity(self, track1, track2, n_scales: int = None) -> float:
        """Вычисление сходства кривизны на нескольких масштабах"""
        if n_scales is None:
            n_scales = self.n_scales
            
        curv1 = track1.get_curvature_signature()
        curv2 = track2.get_curvature_signature()
        
        if len(curv1) == 0 or len(curv2) == 0:
            return 0.0
        
        affinities = []
        
        for scale in range(1, n_scales + 1):
            # Масштабируем сигнатуры
            window_size = max(1, scale)
            
            def smooth_curve(curve, window):
                if len(curve) < window:
                    return curve
                return np.convolve(curve, np.ones(window)/window, mode='valid')
            
            curv1_scaled = smooth_curve(curv1, window_size)
            curv2_scaled = smooth_curve(curv2, window_size)
            
            # Выравниваем длины
            min_len = min(len(curv1_scaled), len(curv2_scaled))
            if min_len > 0:
                curv1_scaled = curv1_scaled[:min_len]
                curv2_scaled = curv2_scaled[:min_len]
                
                # Корреляция на данном масштабе
                corr = np.corrcoef(curv1_scaled, curv2_scaled)[0, 1]
                if not np.isnan(corr):
                    affinities.append(max(0, (corr + 1) / 2))
        
        return np.mean(affinities) if affinities else 0.0
    
    def _compute_spatial_affinity(self, track1, track2) -> float:
        """Вычисление пространственного сходства с выравниванием по длине."""
        points1 = track1.get_points_matrix()
        points2 = track2.get_points_matrix()
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.0

        target_len = min(60, max(2, min(len(points1), len(points2))))
        points1 = self._resample_points(points1, target_len)
        points2 = self._resample_points(points2, target_len)
        distance = np.mean(np.linalg.norm(points1 - points2, axis=1))
        similarity = np.exp(-distance / 100)
        
        return similarity

    def _resample_points(self, points: np.ndarray, target_len: int) -> np.ndarray:
        if len(points) == target_len:
            return points

        source_t = np.linspace(0, 1, len(points))
        target_t = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, points.shape[1]))
        for dim in range(points.shape[1]):
            resampled[:, dim] = np.interp(target_t, source_t, points[:, dim])
        return resampled
    
    def build_affinity_matrix(self, tracks: Dict, use_knn: bool = True, 
                             n_neighbors: int = 10) -> np.ndarray:
        """Построение матрицы сходства с KNN усечением"""
        track_list = list(tracks.values())
        n_tracks = len(track_list)
        
        if n_tracks == 0:
            return np.zeros((0, 0))
        
        # Вычисляем полную матрицу сходства
        affinity_matrix = np.zeros((n_tracks, n_tracks))
        
        for i in range(n_tracks):
            for j in range(i, n_tracks):
                if i == j:
                    affinity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_multiscale_affinity(track_list[i], track_list[j])
                    affinity_matrix[i, j] = sim
                    affinity_matrix[j, i] = sim
        
        # Применяем KNN усечение для разреживания матрицы
        if use_knn:
            for i in range(n_tracks):
                # Находим K ближайших соседей
                neighbors = np.argsort(affinity_matrix[i])[-min(n_neighbors, n_tracks - 1)-1:-1]
                mask = np.ones(n_tracks, dtype=bool)
                mask[neighbors] = False
                mask[i] = False
                affinity_matrix[i, mask] = 0
                affinity_matrix[mask, i] = 0
        
        return affinity_matrix
    
    def compute_laplacian_embedding(self, affinity_matrix: np.ndarray, 
                                   n_components: int = 10) -> np.ndarray:
        """Вычисление лапласиановского вложения"""
        # Строим лапласиан
        if self.use_normalized_laplacian:
            # Нормализованный лапласиан L = I - D^{-1/2} A D^{-1/2}
            degree = np.sum(affinity_matrix, axis=1)
            degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
            laplacian = np.eye(len(affinity_matrix)) - degree_inv_sqrt @ affinity_matrix @ degree_inv_sqrt
        else:
            # Некормализованный лапласиан L = D - A
            degree = np.diag(np.sum(affinity_matrix, axis=1))
            laplacian = degree - affinity_matrix
        
        # Вычисляем собственные векторы
        n_components = min(n_components, len(affinity_matrix) - 1)
        
        try:
            if len(affinity_matrix) > 500:
                # Для больших матриц используем разреженные методы
                eigenvalues, eigenvectors = eigsh(laplacian, k=n_components, which='SM')
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                eigenvectors = eigenvectors[:, :n_components]
        except Exception as e:
            print(f"  Ошибка при вычислении собственных значений: {e}")
            # Fallback: используем все собственные векторы
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            eigenvectors = eigenvectors[:, :n_components]
        
        return eigenvectors
    
    def determine_optimal_clusters(self, eigenvectors: np.ndarray, 
                                   max_clusters: int = 15) -> int:
        """Определение оптимального количества кластеров методом силуэта"""
        from sklearn.metrics import silhouette_score
        
        best_k = 2
        best_score = -1
        
        for k in range(2, min(max_clusters, len(eigenvectors))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(eigenvectors)
            
            if len(np.unique(labels)) > 1:
                try:
                    score = silhouette_score(eigenvectors, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
        
        return best_k
    
    def cluster(self, tracks: Dict, verbose: bool = True) -> Dict:
        """
        Кластеризация траекторий методом LSCC
        """
        if verbose:
            print(f"LSCC: Начало кластеризации (метрика: {self.affinity_metric})...")

        if len(tracks) > self.max_exact_tracks:
            if verbose:
                print(
                    f"LSCC: {len(tracks)} траекторий, включен быстрый режим без полной "
                    f"матрицы сходства {len(tracks)}x{len(tracks)}"
                )
            clusters = self._cluster_large_dataset(tracks)
            for cluster in clusters.values():
                cluster['algorithm'] = 'LSCC_fast'
                cluster['approximation'] = 'MiniBatchKMeans по признакам траекторий без O(n²) матрицы LSCC'
            return clusters
        
        # Строим матрицу сходства
        affinity_matrix = self.build_affinity_matrix(tracks)
        
        if verbose:
            print(f"  Матрица сходства построена. Плотность: {(affinity_matrix > 0).mean():.2%}")
        
        # Вычисляем лапласиановское вложение
        embedding = self.compute_laplacian_embedding(affinity_matrix)
        
        # Определяем количество кластеров
        if self.n_clusters is None:
            self.n_clusters = self.determine_optimal_clusters(embedding)
            if verbose:
                print(f"  Оптимальное количество кластеров: {self.n_clusters}")
        
        # Кластеризуем вложенные точки
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)
        
        # Формируем результаты
        track_ids = list(tracks.keys())
        clusters = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_tracks = [track_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if cluster_tracks:
                reference = self._compute_cluster_reference(tracks, cluster_tracks)
                
                # Вычисляем качество кластера
                quality = self._compute_cluster_quality(affinity_matrix, cluster_labels, cluster_id)
                
                clusters[cluster_id] = {
                    'reference': reference,
                    'tracks': cluster_tracks,
                    'size': len(cluster_tracks),
                    'algorithm': 'LSCC',
                    'quality': quality
                }
        
        if verbose:
            print(f"  Сформировано {len(clusters)} кластеров")
            for cid, cdata in clusters.items():
                print(f"    Кластер {cid}: {cdata['size']} траекторий (качество: {cdata['quality']:.3f})")
        
        return clusters
    
    def _compute_cluster_reference(self, tracks: Dict, cluster_tracks: List) -> np.ndarray:
        """Вычисление опорной траектории для кластера"""
        if not cluster_tracks:
            return np.zeros(3)
        
        # Собираем векторы направлений
        vectors = []
        for track_id in cluster_tracks:
            vec = tracks[track_id].get_trajectory_vector()
            if np.linalg.norm(vec) > 0:
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(3)
        
        vectors = np.array(vectors)
        # Используем медиану для робастности
        median_dir = np.median(vectors, axis=0)
        median_dir = median_dir / (np.linalg.norm(median_dir) + 1e-8)
        
        # Взвешенное усреднение с близкими направлениями
        similarities = np.abs(np.dot(vectors, median_dir))
        weights = similarities / (np.sum(similarities) + 1e-8)
        
        weighted_dir = np.sum(vectors * weights[:, np.newaxis], axis=0)
        weighted_dir = weighted_dir / (np.linalg.norm(weighted_dir) + 1e-8)
        
        return weighted_dir
    
    def _compute_cluster_quality(self, affinity_matrix: np.ndarray, 
                                 labels: np.ndarray, cluster_id: int) -> float:
        """Вычисление качества кластера"""
        cluster_mask = labels == cluster_id
        if not np.any(cluster_mask):
            return 0.0
        
        # Внутрикластерная связность
        intra_cluster = affinity_matrix[cluster_mask][:, cluster_mask]
        intra_score = np.mean(intra_cluster) if intra_cluster.size > 0 else 0
        
        # Межкластерная связность
        inter_cluster = affinity_matrix[cluster_mask][:, ~cluster_mask]
        inter_score = np.mean(inter_cluster) if inter_cluster.size > 0 else 0
        
        # Качество = внутри / (внутри + меж)
        quality = intra_score / (intra_score + inter_score + 1e-8)
        
        return quality
