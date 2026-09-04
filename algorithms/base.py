import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AlgorithmType(Enum):
    """Типы алгоритмов кластеризации"""
    RANSAC = "ransac"
    MLESAC = "mlesac"
    SCC = "scc"
    LSCC = "lscc"
    MEAN_SHIFT = "mean_shift"


@dataclass
class ClusterResult:
    """Результат кластеризации"""
    cluster_id: int
    tracks: List[str]  # Список ID треков в кластере
    reference_trajectory: np.ndarray  # Опорная траектория (вектор направления)
    size: int = 0
    quality: float = 0.0  # Качество кластера (0-1)
    algorithm: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.size = len(self.tracks)


class BaseClusteringAlgorithm(ABC):
    """
    Абстрактный базовый класс для всех алгоритмов кластеризации траекторий
    """
    
    def __init__(self, name: str, verbose: bool = True):
        """
        Args:
            name: название алгоритма
            verbose: выводить ли информацию о процессе
        """
        self.name = name
        self.verbose = verbose
        self.clusters: Dict[int, ClusterResult] = {}
        
    @abstractmethod
    def cluster(self, trajectories: Dict) -> Dict[int, ClusterResult]:
        """
        Выполняет кластеризацию траекторий
        
        Args:
            trajectories: словарь {track_id: Track}
            
        Returns:
            словарь {cluster_id: ClusterResult}
        """
        pass
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Вычисление косинусной меры между двумя векторами
        
        Args:
            v1: первый вектор
            v2: второй вектор
            
        Returns:
            косинусное сходство (0-1)
        """
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        return float(np.dot(v1_norm, v2_norm))
    
    def cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Вычисление косинусного расстояния между двумя векторами
        
        Args:
            v1: первый вектор
            v2: второй вектор
            
        Returns:
            косинусное расстояние (0-2)
        """
        return 1.0 - self.cosine_similarity(v1, v2)
    
    def compute_reference_trajectory(self, vectors: np.ndarray, 
                                     weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Вычисление опорной траектории (усредненного направления)
        
        Args:
            vectors: массив векторов (N x 3)
            weights: веса для каждого вектора (опционально)
            
        Returns:
            опорный вектор направления
        """
        if len(vectors) == 0:
            return np.zeros(3)
        
        # Нормализуем все векторы
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / (norms + 1e-8)
        
        # Вычисляем взвешенное среднее
        if weights is not None:
            weights = np.array(weights) / (np.sum(weights) + 1e-8)
            mean_dir = np.sum(normalized * weights[:, np.newaxis], axis=0)
        else:
            mean_dir = np.mean(normalized, axis=0)
        
        # Нормализуем результат
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
        
        return mean_dir
    
    def compute_cluster_quality(self, cluster_vectors: np.ndarray, 
                                reference: np.ndarray) -> float:
        """
        Вычисление качества кластера
        
        Args:
            cluster_vectors: векторы траекторий в кластере
            reference: опорный вектор кластера
            
        Returns:
            качество кластера (0-1)
        """
        if len(cluster_vectors) == 0:
            return 0.0
        
        # Вычисляем среднее косинусное сходство с опорной траекторией
        similarities = [self.cosine_similarity(v, reference) for v in cluster_vectors]
        
        return float(np.mean(similarities))
    
    def filter_trajectories(self, trajectories: Dict, 
                           min_points: int = 2) -> Dict:
        """
        Фильтрует траектории с недостаточным количеством точек
        
        Args:
            trajectories: словарь траекторий
            min_points: минимальное количество точек
            
        Returns:
            отфильтрованный словарь
        """
        filtered = {}
        for track_id, track in trajectories.items():
            points = track.get_points_matrix() if hasattr(track, 'get_points_matrix') else track.get('points', [])
            if len(points) >= min_points:
                filtered[track_id] = track
        
        if self.verbose and len(filtered) < len(trajectories):
            print(f"  {self.name}: отфильтровано {len(trajectories) - len(filtered)} траекторий "
                  f"(меньше {min_points} точек)")
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики кластеризации
        
        Returns:
            словарь со статистикой
        """
        if not self.clusters:
            return {
                'algorithm': self.name,
                'num_clusters': 0,
                'total_tracks_clustered': 0,
                'avg_cluster_size': 0.0,
                'avg_quality': 0.0
            }
        
        sizes = [c.size for c in self.clusters.values()]
        qualities = [c.quality for c in self.clusters.values()]
        
        return {
            'algorithm': self.name,
            'num_clusters': len(self.clusters),
            'total_tracks_clustered': sum(sizes),
            'avg_cluster_size': np.mean(sizes),
            'std_cluster_size': np.std(sizes),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes),
            'avg_quality': np.mean(qualities),
            'min_quality': min(qualities),
            'max_quality': max(qualities)
        }
    
    def print_statistics(self):
        """Выводит статистику кластеризации"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print(f"Статистика кластеризации ({self.name})")
        print("="*50)
        print(f"Количество кластеров: {stats['num_clusters']}")
        print(f"Всего траекторий в кластерах: {stats['total_tracks_clustered']}")
        print(f"Средний размер кластера: {stats['avg_cluster_size']:.1f}")
        print(f"Стандартное отклонение: {stats['std_cluster_size']:.1f}")
        print(f"Размеры кластеров: min={stats['min_cluster_size']}, max={stats['max_cluster_size']}")
        print(f"Среднее качество кластеров: {stats['avg_quality']:.3f}")
        print(f"Качество кластеров: min={stats['min_quality']:.3f}, max={stats['max_quality']:.3f}")
        print("="*50)


class RANSACMixin:
    """Mixin для RANSAC-подобных алгоритмов"""
    
    def ransac_hypothesis(self, sample_vectors: np.ndarray) -> np.ndarray:
        """
        Генерация гипотезы на основе случайной выборки
        
        Args:
            sample_vectors: векторы из случайной выборки
            
        Returns:
            гипотетическая опорная траектория
        """
        return self.compute_reference_trajectory(sample_vectors)
    
    def evaluate_hypothesis(self, hypothesis: np.ndarray, 
                           all_vectors: np.ndarray,
                           threshold: float) -> np.ndarray:
        """
        Оценка гипотезы: возвращает маску инлайнеров
        
        Args:
            hypothesis: гипотетическая опорная траектория
            all_vectors: все векторы для оценки
            threshold: порог косинусного расстояния
            
        Returns:
            булева маска инлайнеров
        """
        distances = np.array([self.cosine_distance(v, hypothesis) for v in all_vectors])
        return distances < threshold


class SpectralMixin:
    """Mixin для спектральных алгоритмов"""
    
    def build_similarity_matrix(self, vectors: np.ndarray, 
                                sigma: float = 1.0) -> np.ndarray:
        """
        Построение матрицы сходства на основе RBF ядра
        
        Args:
            vectors: массив векторов (N x 3)
            sigma: параметр масштаба
            
        Returns:
            матрица сходства (N x N)
        """
        n = len(vectors)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                sim = self.cosine_similarity(vectors[i], vectors[j])
                # Применяем RBF ядро
                sim = np.exp(-((1 - sim) ** 2) / (2 * sigma ** 2))
                similarity[i, j] = sim
                similarity[j, i] = sim
        
        return similarity
    
    def build_laplacian(self, similarity: np.ndarray, 
                       normalized: bool = True) -> np.ndarray:
        """
        Построение лапласиана графа
        
        Args:
            similarity: матрица сходства
            normalized: использовать нормализованный лапласиан
            
        Returns:
            матрица лапласиана
        """
        # Степени вершин
        degrees = np.sum(similarity, axis=1)
        D = np.diag(degrees)
        
        if normalized:
            # Нормализованный лапласиан: L = I - D^{-1/2} A D^{-1/2}
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
            laplacian = np.eye(len(similarity)) - D_inv_sqrt @ similarity @ D_inv_sqrt
        else:
            # Ненормализованный лапласиан: L = D - A
            laplacian = D - similarity
        
        return laplacian