# test_parser.py
import tempfile
import numpy as np
from pathlib import Path
from data.archive_extractor import ArchiveExtractor
from data.track_parser import TrackParser

# Создаем тестовый файл с вашим примером
test_content = """TRACK 12607553
0
01/01/2006
00:01:04
00:02:04
SJC
CALSTR5
HELO
H
0365
A
30L
715
925
166925
173507
12
31613,-38144,251,45,0
31336,-38150,261,45,4
31032,-38169,269,47,9
30714,-38202,273,49,18
30405,-38251,272,50,23
30099,-38312,265,50,27
29785,-38383,257,49,32
29483,-38457,251,48,41
29217,-38528,246,47,46
28986,-38594,242,46,50
28771,-38659,231,45,55
28562,-38727,218,45,60"""

# Создаем временный файл
with tempfile.NamedTemporaryFile(mode='w', suffix='.lt6', delete=False) as f:
    f.write(test_content)
    test_file = Path(f.name)

print(f"Создан тестовый файл: {test_file}")
print("="*60)

# Создаем экстрактор для тестов (без реального архива)
from archive_extractor import ArchiveExtractor
import zipfile
from pathlib import Path

# Создаем заглушку для ArchiveExtractor
class MockExtractor:
    def __init__(self, test_file_path):
        self.test_file_path = test_file_path
        self.extract_dir = test_file_path.parent
        
    def read_file_content(self, file_path):
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            return [line.rstrip('\n\r') for line in f.readlines()]
    
    def extract_dir_path(self):
        return self.extract_dir

# Используем MockExtractor для теста
mock_extractor = MockExtractor(test_file)
parser = TrackParser(mock_extractor, verbose=True, fill_missing=True)

# Парсим тестовый файл
tracks = parser.parse_track_file(test_file)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ТЕСТА")
print("="*60)

if tracks:
    track = tracks[0]
    print("Трек успешно распарсен!")
    print(f"  OPNUM: {track.opnum}")
    print(f"  EventID: {track.eventid}")
    print(f"  Дата: {track.trackstart_date}")
    print(f"  Время начала: {track.trackstart_time}")
    print(f"  Время конца: {track.trackend_time}")
    print(f"  Аэропорт: {track.airportid}")
    print(f"  ACID: {track.acid}")
    print(f"  Тип ВС: {track.aircrafttype}")
    print(f"  Категория: {track.aircraft_category}")
    print(f"  ВПП: {track.runwayname}")
    print(f"  Min/Max Alt: {track.min_alt}/{track.max_alt}")
    print(f"  Min/Max Range: {track.min_range}/{track.max_range}")
    print(f"  Количество точек: {len(track.points)}")
    
    print(f"\n  Первые 3 точки:")
    for i, point in enumerate(track.points[:3]):
        print(f"    {i+1}: x={point.x}, y={point.y}, z={point.z}, v={point.v}, t={point.t}")
    
    print(f"\n  Последние 3 точки:")
    for i, point in enumerate(track.points[-3:]):
        print(f"    {len(track.points)-3+i+1}: x={point.x}, y={point.y}, z={point.z}, v={point.v}, t={point.t}")
    
    # Проверяем вычисление вектора траектории
    vector = track.get_trajectory_vector()
    print(f"\n  Вектор траектории: {vector}")
    print(f"  Норма вектора: {np.linalg.norm(vector):.6f}")
    
    # Проверяем матрицу точек
    points_matrix = track.get_points_matrix()
    print(f"  Матрица точек shape: {points_matrix.shape}")
    
    # Проверяем кривизну
    curvature = track.get_curvature_signature()
    print(f"  Сигнатура кривизны: {len(curvature)} значений")
    if len(curvature) > 0:
        print(f"    Средняя кривизна: {np.mean(curvature):.4f}")
    
    # Проверяем DataFrame
    df = parser.get_dataframe()
    print(f"\n  DataFrame shape: {df.shape}")
    print(f"  Колонки: {list(df.columns)}")
    
else:
    print("Не удалось распарсить трек!")

# Очистка
test_file.unlink()
print(f"\nТестовый файл удален: {test_file}")
