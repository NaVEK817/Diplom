"""
Этап 3: Автоматическое детектирование аномалий (выбросов)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Результат детектирования аномалии для одной траектории"""
    track_id: str
    rank: int
    cosine_distance: float
    squared_distance_sum: float
    is_anomaly: bool
    anomaly_score: float
    risk_level: str
    risk_description: str
    recommendation: str


class AnomalyDetector:
    """
    Класс для детектирования аномалий в траекториях
    """
    
    def __init__(self, 
                 use_cosine_metric: bool = True,
                 anomaly_threshold: float = 95.0):  # <-- ДОБАВЛЕН ПАРАМЕТР anomaly_threshold
        """
        Args:
            use_cosine_metric: использовать ли меру косинуса для сравнения
            anomaly_threshold: порог для определения аномалий (процентиль, 0-100)
        """
        self.use_cosine_metric = use_cosine_metric
        self.anomaly_threshold = anomaly_threshold  # <-- СОХРАНЯЕМ ПАРАМЕТР
        self.centroid = None
        self.results = []
        
    def cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Косинусное расстояние между векторами"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        return 1 - np.dot(v1_norm, v2_norm)
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Косинусное сходство"""
        return 1 - self.cosine_distance(v1, v2)
    
    def compute_trajectory_deviation(self, track_points: np.ndarray, 
                                      centroid: np.ndarray) -> Tuple[float, float]:
        """
        Расчет отклонения траектории от центроида
        
        Returns:
            (среднее_косинусное_расстояние, сумма_квадратов_расстояний)
        """
        if len(track_points) == 0 or len(centroid) == 0:
            return 1.0, float('inf')
        
        min_len = min(len(track_points), len(centroid))
        if min_len < 2:
            return 1.0, float('inf')

        track_aligned = self._resample_points(track_points, min_len)
        centroid_aligned = self._resample_points(centroid, min_len)
        track_aligned = np.diff(track_aligned, axis=0)
        centroid_aligned = np.diff(centroid_aligned, axis=0)
        
        distances = []
        squared_distances = []
        
        for i in range(len(track_aligned)):
            if self.use_cosine_metric:
                dist = self.cosine_distance(track_aligned[i], centroid_aligned[i])
            else:
                dist = np.linalg.norm(track_aligned[i] - centroid_aligned[i])
            
            distances.append(dist)
            squared_distances.append(dist ** 2)
        
        mean_cosine_distance = np.mean(distances)
        total_squared_distance = np.sum(squared_distances)
        
        return mean_cosine_distance, total_squared_distance

    def _resample_points(self, points: np.ndarray, target_len: int) -> np.ndarray:
        if len(points) == target_len:
            return points

        source_t = np.linspace(0, 1, len(points))
        target_t = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, points.shape[1]))
        for dim in range(points.shape[1]):
            resampled[:, dim] = np.interp(target_t, source_t, points[:, dim])
        return resampled
    
    def detect_anomalies(self, tracks: List, track_ids: List[str], 
                         centroid: np.ndarray) -> List[AnomalyResult]:
        """
        Детектирование аномалий с ранжированием
        
        Args:
            tracks: список траекторий
            track_ids: список ID траекторий
            centroid: точки центроида
        """
        self.centroid = centroid
        
        # Расчет отклонений для каждой траектории
        deviations = []
        
        for i, track in enumerate(tracks):
            points = track.get_points_matrix()
            cosine_dist, squared_sum = self.compute_trajectory_deviation(points, centroid)
            
            deviations.append({
                'track_id': track_ids[i],
                'track': track,
                'cosine_distance': cosine_dist,
                'squared_distance_sum': squared_sum,
                'index': i
            })
        
        # Ранжирование по величине отклонения
        deviations.sort(key=lambda x: x['cosine_distance'], reverse=True)
        
        for rank, dev in enumerate(deviations, 1):
            dev['rank'] = rank
        
        # Вычисление порога для аномалий
        if deviations:
            distances = [d['cosine_distance'] for d in deviations]
            # Используем переданный процентиль
            threshold = np.percentile(distances, self.anomaly_threshold)
        else:
            threshold = 0.5
        
        # Формирование результатов
        results = []
        
        for dev in deviations:
            # Нормированная оценка аномальности
            if deviations:
                max_dev = deviations[0]['squared_distance_sum']
                anomaly_score = dev['squared_distance_sum'] / (max_dev + 1e-8)
            else:
                anomaly_score = 0
            
            # Является ли траектория аномалией
            is_anomaly = dev['cosine_distance'] > threshold
            
            # Оценка уровня риска
            if dev['rank'] == 1:
                risk_level = "high"
                risk_description = "КРИТИЧЕСКОЕ ОТКЛОНЕНИЕ - потенциально опасная траектория"
                recommendation = "НЕМЕДЛЕННЫЙ АНАЛИЗ! Возможны ошибка пилотирования или сбой радара"
            elif anomaly_score > 0.7:
                risk_level = "high"
                risk_description = "Значительное отклонение от эталонной траектории"
                recommendation = "Требуется детальный анализ причин отклонения"
            elif anomaly_score > 0.4:
                risk_level = "medium"
                risk_description = "Умеренное отклонение от глиссады"
                recommendation = "Рекомендуется усиленный контроль"
            else:
                risk_level = "low"
                risk_description = "Отклонение в пределах нормы"
                recommendation = "Траектория соответствует эталону"
            
            results.append(AnomalyResult(
                track_id=dev['track_id'],
                rank=dev['rank'],
                cosine_distance=float(dev['cosine_distance']),
                squared_distance_sum=float(dev['squared_distance_sum']),
                is_anomaly=is_anomaly,
                anomaly_score=float(anomaly_score),
                risk_level=risk_level,
                risk_description=risk_description,
                recommendation=recommendation
            ))
        
        results.sort(key=lambda x: x.rank)
        self.results = results
        
        return results
    
    def get_anomaly_statistics(self, results: List[AnomalyResult] = None) -> Dict:
        """Получение статистики по аномалиям"""
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        anomalies = [r for r in results if r.is_anomaly]
        high_risk = [r for r in results if r.risk_level == 'high']
        
        return {
            'total_tracks': len(results),
            'anomalies_count': len(anomalies),
            'anomalies_percent': len(anomalies) / len(results) * 100,
            'high_risk_count': len(high_risk),
            'high_risk_percent': len(high_risk) / len(results) * 100,
            'max_cosine_distance': max(r.cosine_distance for r in results),
            'mean_cosine_distance': np.mean([r.cosine_distance for r in results]),
            'std_cosine_distance': np.std([r.cosine_distance for r in results]),
            'top_anomaly': results[0].track_id if results else None,
            'top_anomaly_score': results[0].anomaly_score if results else 0
        }
    # Добавьте этот метод в класс AnomalyDetector:

    def fit(self, centroid: np.ndarray, tracks: List = None):
        """
        Обучение детектора (сохранение центроида)

        Args:
            centroid: точки центроида
            tracks: список траекторий (опционально, для вычисления статистики)
        """
        self.centroid = centroid
        if tracks:
            # Опционально: предварительный расчет статистики
            pass

class RiskAssessor:
    """
    Класс для оценки рисков на основе детектированных аномалий
    """
    
    def __init__(self):
        self.risk_levels = {
            'low': {'color': 'green', 'threshold': 0.3, 'description': 'Нормальное состояние'},
            'medium': {'color': 'orange', 'threshold': 0.6, 'description': 'Требуется внимание'},
            'high': {'color': 'red', 'threshold': 0.8, 'description': 'КРИТИЧЕСКИЙ РИСК'}
        }
    
    def assess_anomaly_risk(self, anomaly_result: AnomalyResult) -> Dict:
        """
        Оценка риска для отдельной аномалии
        """
        if anomaly_result.risk_level == 'high':
            severity = "Критическое отклонение"
            action = "Немедленный анализ, возможен уход на второй круг"
        elif anomaly_result.risk_level == 'medium':
            severity = "Значительное отклонение"
            action = "Усиленный контроль, проверка параметров"
        else:
            severity = "Незначительное отклонение"
            action = "Мониторинг в штатном режиме"
        
        return {
            'track_id': anomaly_result.track_id,
            'risk_level': anomaly_result.risk_level,
            'severity': severity,
            'deviation': f"{anomaly_result.cosine_distance:.2%}",
            'rank': anomaly_result.rank,
            'recommended_action': action,
            'recommendation': anomaly_result.recommendation
        }
    
    def assess_overall_risk(self, anomaly_results: List[AnomalyResult]) -> Dict:
        """
        Оценка общего риска для всего пучка
        """
        if not anomaly_results:
            return {'overall_risk': 'unknown', 'risk_score': 0}
        
        # Средний аномальный скор
        avg_score = np.mean([r.anomaly_score for r in anomaly_results])
        
        # Процент аномалий
        anomaly_percent = len([r for r in anomaly_results if r.is_anomaly]) / len(anomaly_results) * 100
        
        # Общий риск-скор
        risk_score = avg_score * 0.6 + (anomaly_percent / 100) * 0.4
        risk_score = min(1.0, risk_score)
        
        if risk_score < 0.3:
            overall_risk = "Низкий"
            color = "green"
            icon = ""
            recommendation = "Система стабильна. Значимых отклонений не обнаружено."
        elif risk_score < 0.6:
            overall_risk = "Средний"
            color = "orange"
            icon = ""
            recommendation = "Обнаружены отклонения. Рекомендуется мониторинг."
        else:
            overall_risk = "Высокий"
            color = "red"
            icon = ""
            recommendation = "КРИТИЧЕСКИЙ РИСК! Обнаружены опасные отклонения."
        
        return {
            'overall_risk': overall_risk,
            'risk_score': float(risk_score),
            'risk_color': color,
            'risk_icon': icon,
            'recommendation': recommendation,
            'anomaly_percent': float(anomaly_percent),
            'avg_anomaly_score': float(avg_score)
        }
