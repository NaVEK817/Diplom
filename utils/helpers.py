# utils/helpers.py
"""Вспомогательные функции"""

import numpy as np
from typing import List, Dict, Any


def format_duration(seconds: float) -> str:
    """Форматирует duration в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} мин"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} ч"


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Вычисляет статистику для списка значений"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }