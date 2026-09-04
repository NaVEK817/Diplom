import math

class MetricsCalculator:
    """Калькулятор метрик траектории (заготовка) - все значения int"""
    
    def __init__(self):
        pass
    
    def calculate_trajectory_length(self, points):
        """Расчёт длины траектории (возвращает int)"""
        length = 0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            length += int(math.sqrt(dx*dx + dy*dy))
        return length
    
    def calculate_glide_angle(self, altitude_points, distance_points):
        """Расчёт угла глиссады (возвращает int)"""
        if len(altitude_points) < 2:
            return 0
        delta_h = altitude_points[0] - altitude_points[-1]
        delta_d = distance_points[-1] - distance_points[0]
        if delta_d == 0:
            return 0
        return int(math.degrees(math.atan2(delta_h, delta_d)))
    
    def calculate_vertical_speed(self, altitude, time_delta):
        """Расчёт вертикальной скорости (возвращает int)"""
        if time_delta == 0:
            return 0
        return int(altitude / time_delta)
    
    def calculate_energy_parameter(self, speed, altitude, reference_speed=70):
        """Расчёт энергетического параметра E' (возвращает int)"""
        if reference_speed == 0:
            return 0
        result = (speed * speed) // (reference_speed * reference_speed) + altitude // 1000
        return int(result)