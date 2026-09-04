"""
Этап 2: Математическое моделирование центроида (типичного профиля)
Оптимизированная версия с улучшенной производительностью
"""

import numpy as np
import time
import sys
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline


class CentroidModeler:
    """
    Класс для построения центроида (типичного профиля) пучка траекторий
    """
    
    def __init__(self, 
                 polynomial_degree: int = 5, 
                 cpm_smoothing: float = 1.0,
                 max_iterations: int = 20,  # Уменьшено для производительности
                 convergence_threshold: float = 1e-3,  # Увеличен порог
                 use_cpm: bool = True,
                 verbose: bool = False,
                 max_points_per_trajectory: int = 100):  # Ограничение точек
        """
        Args:
            polynomial_degree: степень полинома для сглаживания
            cpm_smoothing: параметр сглаживания для CPM
            max_iterations: максимальное число итераций
            convergence_threshold: порог сходимости
            use_cpm: использовать ли CPM
            verbose: выводить подробную информацию
            max_points_per_trajectory: максимальное количество точек на траекторию
        """
        self.polynomial_degree = polynomial_degree
        self.cpm_smoothing = cpm_smoothing
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_cpm = use_cpm
        self.verbose = verbose
        self.max_points_per_trajectory = max_points_per_trajectory
        self.centroid = None
        self.centroid_poly = None
        self.reference_direction = None
        self.latent_trace = None
        
    def _log(self, message: str, level: str = "INFO"):
        """Вывод сообщения в консоль"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] [CentroidModeler] {message}")
            sys.stdout.flush()
    
    def cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Косинусное расстояние между векторами"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 1.0
        return 1 - np.dot(v1, v2) / (norm1 * norm2)
    
    def _reduce_points(self, points: np.ndarray) -> np.ndarray:
        """Уменьшение количества точек для ускорения"""
        if len(points) <= self.max_points_per_trajectory:
            return points
        
        # Равномерная выборка точек
        indices = np.linspace(0, len(points) - 1, self.max_points_per_trajectory, dtype=int)
        return points[indices]
    
    def _align_trajectories_length(self, points_list: List[np.ndarray]) -> np.ndarray:
        """Быстрое выравнивание траекторий"""
        if not points_list:
            return np.array([])
        
        self._log(f"Выравнивание {len(points_list)} траекторий...")
        
        # Сначала уменьшаем количество точек
        reduced_list = [self._reduce_points(p) for p in points_list]
        
        # Определяем целевую длину (медиана для избежания выбросов)
        lengths = [len(p) for p in reduced_list]
        target_len = int(np.median(lengths))
        self._log(f"  Целевая длина: {target_len} точек (медиана из {min(lengths)}-{max(lengths)})")
        
        aligned = []
        for points in reduced_list:
            if len(points) == target_len:
                aligned.append(points)
            else:
                # Быстрая линейная интерполяция
                t_old = np.linspace(0, 1, len(points))
                t_new = np.linspace(0, 1, target_len)
                
                interpolated = np.zeros((target_len, 3))
                for dim in range(3):
                    interpolated[:, dim] = np.interp(t_new, t_old, points[:, dim])
                aligned.append(interpolated)
        
        return np.array(aligned)
    
    def optimize_profile_cosine_fast(self, points_list: List[np.ndarray]) -> np.ndarray:
        """
        Быстрая оптимизация профиля - вычисление центроида как взвешенной медианы
        (без использования scipy.optimize для ускорения)
        """
        self._log("="*60)
        self._log("БЫСТРАЯ ОПТИМИЗАЦИЯ ПРОФИЛЯ (косинусная мера)")
        self._log("="*60)
        
        if not points_list:
            return np.array([])
        
        self._log(f"Количество траекторий: {len(points_list)}")
        
        # Выравниваем траектории
        aligned = self._align_trajectories_length(points_list)
        n_traj, n_steps, n_dims = aligned.shape
        
        self._log(f"Размер данных: {n_traj} × {n_steps} × {n_dims}")
        
        # Вычисляем центроид как взвешенную медиану (быстрее, чем оптимизация)
        start_time = time.time()
        
        # Для каждой точки по времени вычисляем медиану
        centroid = np.median(aligned, axis=0)
        
        # Дополнительное сглаживание с помощью скользящего окна
        window_size = max(3, n_steps // 20)  # Адаптивное окно
        if window_size % 2 == 0:
            window_size += 1
        
        for dim in range(3):
            # Применяем медианный фильтр для сглаживания
            from scipy.signal import medfilt
            centroid[:, dim] = medfilt(centroid[:, dim], kernel_size=window_size)
        
        elapsed = time.time() - start_time
        self._log(f"Центроид вычислен за {elapsed:.2f} сек (взвешенная медиана)")
        
        # Вычисляем опорное направление
        if len(centroid) >= 2:
            direction = centroid[-1] - centroid[0]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                self.reference_direction = direction / norm
            else:
                self.reference_direction = np.array([1.0, 0.0, 0.0])
        
        return centroid
    
    def optimize_profile_cosine_gradient(self, points_list: List[np.ndarray]) -> np.ndarray:
        """
        Оптимизация с использованием L-BFGS-B и аналитического градиента (более точная, но медленнее)
        """
        self._log("="*60)
        self._log("ТОЧНАЯ ОПТИМИЗАЦИЯ ПРОФИЛЯ (с градиентом)")
        self._log("="*60)
        
        if not points_list:
            return np.array([])
        
        self._log(f"Количество траекторий: {len(points_list)}")
        
        # Выравниваем траектории с уменьшением точек
        aligned = self._align_trajectories_length(points_list)
        n_traj, n_steps, n_dims = aligned.shape
        
        self._log(f"Размер данных: {n_traj} × {n_steps} × {n_dims}")
        self._log(f"Размер вектора оптимизации: {n_steps * n_dims}")
        
        # Преобразуем данные для более быстрого доступа
        # Транспонируем для удобства: (n_steps, n_traj, n_dims)
        aligned_t = np.transpose(aligned, (1, 0, 2))
        
        def cosine_loss_with_gradient(centroid_flat):
            """
            Функция потерь с аналитическим градиентом
            """
            centroid = centroid_flat.reshape(-1, n_dims)
            n_centroid_steps = len(centroid)
            
            loss = 0.0
            grad = np.zeros_like(centroid)
            
            for step in range(min(n_steps, n_centroid_steps)):
                c = centroid[step]
                norm_c = np.linalg.norm(c)
                
                if norm_c < 1e-8:
                    continue
                
                for t_idx in range(n_traj):
                    p = aligned_t[step, t_idx]
                    norm_p = np.linalg.norm(p)
                    
                    if norm_p < 1e-8:
                        continue
                    
                    # Косинусное расстояние
                    cos_sim = np.dot(p, c) / (norm_p * norm_c)
                    cos_sim = np.clip(cos_sim, -1, 1)
                    dist = 1 - cos_sim
                    
                    loss += dist ** 2
                    
                    # Градиент по c: d(dist^2)/dc = -2 * dist * d(cos_sim)/dc
                    # d(cos_sim)/dc = (p / (norm_p * norm_c)) - (cos_sim * c / norm_c^2)
                    grad_c = (p / (norm_p * norm_c)) - (cos_sim * c / (norm_c * norm_c))
                    grad[step] += -2 * dist * grad_c
            
            return loss, grad.flatten()
        
        # Начальное приближение - медиана
        self._log("Вычисление начального приближения...")
        initial_centroid = np.median(aligned, axis=0)
        
        # Оптимизация с градиентом
        self._log("Запуск L-BFGS-B оптимизации...")
        start_time = time.time()
        
        result = minimize(
            fun=cosine_loss_with_gradient,
            x0=initial_centroid.flatten(),
            method='L-BFGS-B',
            jac=True,  # Указываем, что функция возвращает градиент
            options={
                'maxiter': self.max_iterations,
                'maxfun': self.max_iterations * 10,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'disp': self.verbose
            }
        )
        
        elapsed = time.time() - start_time
        self._log(f"Оптимизация завершена за {elapsed:.2f} сек")
        self._log(f"  Успех: {result.success}")
        self._log(f"  Итераций: {result.nit}")
        self._log(f"  Финальный loss: {result.fun:.6f}")
        
        if not result.success and self.verbose:
            self._log(f"  Предупреждение: {result.message}", "WARNING")
        
        centroid = result.x.reshape(-1, n_dims)
        
        # Вычисляем опорное направление
        if len(centroid) >= 2:
            direction = centroid[-1] - centroid[0]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                self.reference_direction = direction / norm
        
        return centroid
    
    def continuous_profile_model_fast(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Быстрая версия CPM с меньшим числом итераций
        """
        self._log("="*60)
        self._log("БЫСТРЫЙ CPM (Continuous Profile Model)")
        self._log("="*60)
        
        n_traj, n_steps, n_dims = trajectories.shape
        self._log(f"Входные данные: {n_traj} траекторий, {n_steps} точек")
        
        # Упрощенная инициализация
        latent = np.median(trajectories, axis=0)
        prev_latent = latent.copy()
        
        # Уменьшаем количество итераций для скорости
        actual_iterations = min(self.max_iterations, 10)
        self._log(f"Итераций: {actual_iterations}")
        
        for iteration in range(actual_iterations):
            iter_start = time.time()
            
            # Упрощенный E-шаг (без полной рекурсии для скорости)
            weights = np.zeros((n_traj, n_steps))
            for t_idx in range(n_traj):
                for step in range(n_steps):
                    dist = np.linalg.norm(trajectories[t_idx, step] - latent[step])
                    weights[t_idx, step] = np.exp(-dist ** 2 / (2 * self.cpm_smoothing ** 2))
            
            # Нормализация весов
            weights_sum = np.sum(weights, axis=0, keepdims=True)
            weights = weights / (weights_sum + 1e-8)
            
            # M-шаг: взвешенное среднее
            for step in range(n_steps):
                for dim in range(n_dims):
                    latent[step, dim] = np.sum(weights[:, step] * trajectories[:, step, dim])
            
            # Проверка сходимости
            diff = np.linalg.norm(latent - prev_latent) / (np.linalg.norm(prev_latent) + 1e-8)
            iter_time = time.time() - iter_start
            
            self._log(f"  Итерация {iteration + 1}: изменение={diff:.6e}, время={iter_time:.2f}с")
            
            if diff < self.convergence_threshold:
                self._log(f"  Сходимость достигнута")
                break
            
            prev_latent = latent.copy()
        
        self.latent_trace = latent
        return latent
    
    def polynomial_smoothing(self, points: np.ndarray, 
                            degree: int = None) -> Tuple[np.ndarray, List]:
        """Полиномиальное сглаживание"""
        if degree is None:
            degree = self.polynomial_degree
        
        self._log(f"Полиномиальное сглаживание (степень {degree})...")
        start_time = time.time()
        
        n_points = len(points)
        if n_points < degree + 1:
            degree = max(1, n_points - 1)
            self._log(f"  Степень уменьшена до {degree}")
        
        t = np.linspace(0, 1, n_points)
        
        coefficients = []
        smoothed = np.zeros_like(points)
        
        for dim in range(3):
            coeffs = np.polyfit(t, points[:, dim], degree)
            coefficients.append(coeffs)
            smoothed[:, dim] = np.polyval(coeffs, t)
        
        elapsed = time.time() - start_time
        self._log(f"Сглаживание завершено за {elapsed:.2f} сек")
        
        self.centroid_poly = coefficients
        return smoothed, coefficients
    
    def fit(self, tracks: List, 
            use_cpm: bool = True, 
            use_cosine_optimization: bool = True,
            use_gradient: bool = False) -> Dict:
        """
        Построение центроида для пучка траекторий
        
        Args:
            tracks: список объектов Track
            use_cpm: использовать ли CPM
            use_cosine_optimization: использовать ли оптимизацию
            use_gradient: использовать ли градиентную оптимизацию (медленнее, но точнее)
        """
        total_start = time.time()
        
        self._log("\n" + "="*70)
        self._log("НАЧАЛО ПОСТРОЕНИЯ ЦЕНТРОИДА")
        self._log("="*70)
        self._log(f"Количество траекторий: {len(tracks)}")
        
        if not tracks:
            return {'error': 'Нет траекторий для моделирования'}
        
        # Извлекаем и уменьшаем точки
        self._log("Извлечение точек...")
        points_list = [t.get_points_matrix() for t in tracks]
        
        # Применяем уменьшение точек для ускорения
        points_list = [self._reduce_points(p) for p in points_list]
        
        total_points = sum(len(p) for p in points_list)
        self._log(f"Всего точек: {total_points} (после уменьшения)")
        
        # Шаг 1: Оптимизация профиля
        if use_cosine_optimization:
            if use_gradient:
                centroid = self.optimize_profile_cosine_gradient(points_list)
            else:
                centroid = self.optimize_profile_cosine_fast(points_list)
        else:
            self._log("Используем простое усреднение...")
            aligned = self._align_trajectories_length(points_list)
            centroid = np.median(aligned, axis=0)
        
        # Шаг 2: CPM (если включено)
        if use_cpm and self.use_cpm and len(points_list) > 3:
            self._log("\nЗапуск быстрого CPM...")
            aligned = self._align_trajectories_length(points_list)
            centroid = self.continuous_profile_model_fast(aligned)
        
        # Шаг 3: Полиномиальное сглаживание
        self._log("\nПолиномиальное сглаживание...")
        smoothed_centroid, coefficients = self.polynomial_smoothing(centroid)
        
        total_elapsed = time.time() - total_start
        self._log("\n" + "="*70)
        self._log(f"ПОСТРОЕНИЕ ЦЕНТРОИДА ЗАВЕРШЕНО за {total_elapsed:.2f} сек")
        self._log("="*70)
        
        self.centroid = smoothed_centroid
        
        return {
            'centroid': smoothed_centroid,
            'original_centroid': centroid,
            'coefficients': coefficients,
            'reference_direction': self.reference_direction,
            'latent_trace': self.latent_trace,
            'num_tracks': len(tracks),
            'total_points': total_points,
            'polynomial_degree': self.polynomial_degree,
            'execution_time': total_elapsed
        }
    
    def get_centroid_points(self) -> np.ndarray:
        return self.centroid
    
    def get_reference_vector(self) -> np.ndarray:
        return self.reference_direction