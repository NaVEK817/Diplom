import numpy as np
from typing import List, Dict, Optional
from data.track_parser import Track

class TrajectoryProcessor:
    """Обработчик траекторий для вычисления метрик"""
    
    @staticmethod
    def calculate_trajectory_length(track: Track) -> float:
        """Вычисление длины траектории"""
        points = track.get_points_matrix()
        if len(points) < 2:
            return 0.0
        
        segment_lengths = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        return float(np.sum(segment_lengths))
    
    @staticmethod
    def calculate_glide_angle(track: Track) -> float:
        """Вычисление среднего угла глиссады"""
        points = track.get_points_matrix()
        if len(points) < 10:
            return 0.0
        
        # Берем начало и конец траектории
        start_z = np.mean(points[:5, 2])
        end_z = np.mean(points[-5:, 2])
        
        total_length = TrajectoryProcessor.calculate_trajectory_length(track)
        vertical_diff = start_z - end_z
        
        if total_length > 0:
            angle_rad = np.arctan2(vertical_diff, total_length)
            return float(np.degrees(angle_rad))
        return 0.0
    
    @staticmethod
    def calculate_curvature_radius(track: Track) -> float:
        """Вычисление среднего радиуса кривизны"""
        points = track.get_points_matrix()
        if len(points) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            angle = np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), 
                -1, 1
            ))
            
            if angle > 0.01:
                radius = np.linalg.norm(v1) / (2 * np.sin(angle/2))
                curvatures.append(radius)
        
        if curvatures:
            return float(np.mean(curvatures))
        return 0.0
    
    @staticmethod
    def calculate_kinematics(track: Track) -> Dict:
        """Вычисление кинематических параметров"""
        velocities = track.get_velocity_profile()
        times = track.get_time_profile()
        points = track.get_points_matrix()
        
        result = {
            'mean_speed': 0.0,
            'max_speed': 0.0,
            'mean_vertical_speed': 0.0,
            'max_acceleration': 0.0,
            'std_deviation': 0.0
        }
        
        if len(velocities) > 0:
            result['mean_speed'] = float(np.mean(velocities))
            result['max_speed'] = float(np.max(velocities))
        
        if len(points) > 1 and len(times) > 1:
            vertical_velocities = np.diff(points[:, 2]) / (np.diff(times) + 1e-8)
            result['mean_vertical_speed'] = float(np.mean(np.abs(vertical_velocities)))
        
        if len(velocities) > 1:
            accelerations = np.diff(velocities) / (np.diff(times) + 1e-8)
            result['max_acceleration'] = float(np.max(np.abs(accelerations)))
        
        # Стандартное отклонение от идеальной прямой
        if len(points) > 10:
            # Простая линейная регрессия
            t = np.arange(len(points))
            x = points[:, 0]
            coeffs = np.polyfit(t, x, 1)
            predicted = np.polyval(coeffs, t)
            result['std_deviation'] = float(np.std(x - predicted))
        
        return result
    
    @staticmethod
    def get_trajectory_bounds(tracks: List[Track]) -> Dict:
        """Получение границ всех траекторий"""
        all_points = []
        for track in tracks:
            points = track.get_points_matrix()
            all_points.extend(points)
        
        if not all_points:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0, 'min_z': 0, 'max_z': 0}
        
        all_points = np.array(all_points)
        return {
            'min_x': float(np.min(all_points[:, 0])),
            'max_x': float(np.max(all_points[:, 0])),
            'min_y': float(np.min(all_points[:, 1])),
            'max_y': float(np.max(all_points[:, 1])),
            'min_z': float(np.min(all_points[:, 2])),
            'max_z': float(np.max(all_points[:, 2]))
        }