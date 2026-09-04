import zipfile
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

class TrackParserFromArchive:
    """
    Парсер для чтения файлов из ZIP-архива
    Пустые строки сохраняются как NaN
    """
    
    def __init__(self, archive_path: str, extract_dir: str):
        """
        Инициализация парсера
        
        Args:
            archive_path: путь к ZIP-архиву
            extract_dir: директория для распаковки файлов
        """
        self.archive_path = archive_path
        self.extract_dir = Path(extract_dir)
        self.tracks_data = []
        
        # Создаем директорию для распаковки
        self.extract_dir.mkdir(parents=True, exist_ok=True)
    
    def check_archive_exists(self) -> bool:
        """Проверяет существование архива"""
        return os.path.exists(self.archive_path)
    
    def get_files_in_archive(self) -> List[str]:
        """Возвращает список файлов .lt6 в архиве"""
        if not self.check_archive_exists():
            raise FileNotFoundError(f"Архив не найден: {self.archive_path}")
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            return [f for f in zip_ref.namelist() if f.endswith('.lt6')]
    
    def extract_file(self, filename: str) -> str:
        """Извлекает файл из архива"""
        if not self.check_archive_exists():
            raise FileNotFoundError(f"Архив не найден: {self.archive_path}")
        
        target_path = self.extract_dir / filename
        
        # Если файл уже существует, не извлекаем заново
        if target_path.exists() and target_path.stat().st_size > 0:
            return str(target_path)
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            zip_ref.extract(filename, self.extract_dir)
            print(f"Извлечен: {filename}")
            return str(target_path)
    
    def get_value_or_nan(self, line: str):
        """
        Преобразует строку: если строка пустая -> NaN, иначе -> stripped значение
        """
        if line is None or line.strip() == '' or line.strip() == '\n':
            return np.nan
        return line.strip()
    
    def parse_track_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Парсит один файл с треками
        Пустые строки сохраняются как NaN
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        
        tracks = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i]
            
            # Ищем начало трека
            if current_line.startswith('TRACK'):
                track_data = {}
                track_data['source_file'] = os.path.basename(file_path)
                
                # 1: TRACK OPNUM
                parts = current_line.split()
                track_data['TRACK'] = parts[0]
                track_data['OPNUM'] = int(parts[1]) if len(parts) > 1 else np.nan
                i += 1
                
                # Поля 2-19 (18 полей)
                field_names = [
                    'eventid', 
                    'trackstart_date', 
                    'trackstart_time', 
                    'trackend_time',
                    'airportid', 
                    'ACID', 
                    'owner_name', 
                    'aircrafttype', 
                    'aircraft_category',
                    'beacon', 
                    'adflag', 
                    'waypoint', 
                    'other_port', 
                    'runwayname',
                    'min_alt', 
                    'max_alt', 
                    'min_range', 
                    'max_range'
                ]
                
                for field in field_names:
                    if i < len(lines):
                        value = self.get_value_or_nan(lines[i])
                        
                        # Преобразуем числовые поля
                        if field in ['min_alt', 'max_alt', 'min_range', 'max_range']:
                            if pd.notna(value) and str(value).isdigit():
                                track_data[field] = int(value)
                            else:
                                track_data[field] = np.nan
                        elif field == 'eventid':
                            if pd.notna(value) and str(value).isdigit():
                                track_data[field] = int(value)
                            else:
                                track_data[field] = value if pd.notna(value) else np.nan
                        else:
                            track_data[field] = value if pd.notna(value) else np.nan
                    else:
                        track_data[field] = np.nan
                    i += 1
                
                # 20: Count of trackpoints (может быть пустым)
                if i < len(lines):
                    count_value = self.get_value_or_nan(lines[i])
                    
                    # Проверяем, является ли строка числом или это первая точка
                    if pd.notna(count_value) and ',' in str(count_value):
                        # Нет поля Count - сразу точки
                        track_data['num_trackpoints'] = 0  # Временно, будет определено позже
                        # Не увеличиваем i, так как это первая точка
                    else:
                        # Есть поле Count
                        if pd.notna(count_value) and str(count_value).isdigit():
                            track_data['num_trackpoints'] = int(count_value)
                        else:
                            track_data['num_trackpoints'] = np.nan
                        i += 1
                else:
                    track_data['num_trackpoints'] = np.nan
                
                # 21: Чтение точек
                points = []
                
                # Определяем, сколько точек нужно прочитать
                expected_points = track_data.get('num_trackpoints')
                
                if pd.notna(expected_points) and expected_points > 0:
                    # Читаем ровно expected_points точек
                    for _ in range(int(expected_points)):
                        if i < len(lines):
                            point_line = lines[i].strip()
                            if point_line and ',' in point_line:
                                try:
                                    coords = list(map(float, point_line.split(',')))
                                    if len(coords) == 5:
                                        points.append({
                                            'x': coords[0],
                                            'y': coords[1],
                                            'z': coords[2],
                                            'v': coords[3],
                                            't': coords[4]
                                        })
                                except ValueError:
                                    pass
                        i += 1
                else:
                    # Читаем точки до следующего TRACK или конца файла
                    while i < len(lines):
                        next_line = lines[i].strip()
                        
                        # Если встретили новый трек - останавливаемся
                        if next_line.startswith('TRACK'):
                            break
                        
                        # Если пустая строка - пропускаем
                        if not next_line:
                            i += 1
                            continue
                        
                        # Если это точка - парсим
                        if ',' in next_line:
                            try:
                                coords = list(map(float, next_line.split(',')))
                                if len(coords) == 5:
                                    points.append({
                                        'x': coords[0],
                                        'y': coords[1],
                                        'z': coords[2],
                                        'v': coords[3],
                                        't': coords[4]
                                    })
                            except ValueError:
                                pass
                        i += 1
                    
                    # Обновляем количество точек
                    track_data['num_trackpoints'] = len(points)
                
                track_data['trackpoints'] = points
                tracks.append(track_data)
                
            else:
                i += 1
        
        print(f"  Файл {os.path.basename(file_path)}: {len(tracks)} треков")
        return tracks
    
    def parse_range(self, start_number: int, end_number: int) -> List[Dict[str, Any]]:
        """
        Парсит диапазон файлов вида 2006010i.lt6
        
        Args:
            start_number: начальное i (включительно)
            end_number: конечное i (включительно)
        """
        all_tracks = []
        
        for i in range(start_number, end_number + 1):
            filename = f"2006010{i}.lt6"
            print(f"\nОбработка: {filename}")
            
            try:
                file_path = self.extract_file(filename)
                tracks = self.parse_track_file(file_path)
                all_tracks.extend(tracks)
                print(f"  Всего треков: {len(all_tracks)}")
            except FileNotFoundError:
                print(f"  Файл не найден в архиве")
            except Exception as e:
                print(f"  Ошибка: {e}")
        
        self.tracks_data = all_tracks
        return all_tracks
    
    def parse_all_files(self) -> List[Dict[str, Any]]:
        """Парсит все файлы в архиве"""
        files = self.get_files_in_archive()
        all_tracks = []
        
        for filename in sorted(files):
            print(f"\nОбработка: {filename}")
            
            try:
                file_path = self.extract_file(filename)
                tracks = self.parse_track_file(file_path)
                all_tracks.extend(tracks)
                print(f"  Всего треков: {len(all_tracks)}")
            except Exception as e:
                print(f"  Ошибка: {e}")
        
        self.tracks_data = all_tracks
        return all_tracks
    
    def get_dataframe(self) -> pd.DataFrame:
        """Возвращает DataFrame с информацией о треках (без точек)"""
        if not self.tracks_data:
            return pd.DataFrame()
        
        df_data = []
        for track in self.tracks_data:
            track_copy = {}
            for k, v in track.items():
                if k != 'trackpoints':
                    track_copy[k] = v
            df_data.append(track_copy)
        
        df = pd.DataFrame(df_data)
        
        # Заменяем None на NaN (на всякий случай)
        df = df.replace([None], np.nan)
        
        return df
    
    def get_track_points_dataframe(self, track_index: int = 0) -> pd.DataFrame:
        """Возвращает точки конкретного трека в виде DataFrame"""
        if not self.tracks_data or track_index >= len(self.tracks_data):
            return pd.DataFrame()
        
        return pd.DataFrame(self.tracks_data[track_index]['trackpoints'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по данным"""
        if not self.tracks_data:
            return {}
        
        total_tracks = len(self.tracks_data)
        total_points = sum(len(t['trackpoints']) for t in self.tracks_data)
        
        # Анализ пропущенных значений (NaN)
        df = self.get_dataframe()
        
        field_stats = {}
        for field in ['airportid', 'ACID', 'aircrafttype', 'waypoint', 'runwayname', 
                      'owner_name', 'other_port', 'beacon']:
            if field in df.columns:
                null_count = df[field].isna().sum()
                field_stats[field] = {
                    'nan_count': int(null_count),
                    'nan_percent': float((null_count / total_tracks) * 100)
                }
        
        return {
            'total_tracks': total_tracks,
            'total_points': total_points,
            'avg_points_per_track': total_points / total_tracks if total_tracks > 0 else 0,
            'tracks_with_points': sum(1 for t in self.tracks_data if t['trackpoints']),
            'tracks_without_points': sum(1 for t in self.tracks_data if not t['trackpoints']),
            'field_statistics': field_stats
        }
    
    def save_to_csv(self, output_dir: str = None):
        """Сохраняет данные в CSV файлы"""
        if output_dir is None:
            output_dir = self.extract_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем информацию о треках
        df_tracks = self.get_dataframe()
        if not df_tracks.empty:
            tracks_csv = output_path / "all_tracks_info.csv"
            df_tracks.to_csv(tracks_csv, index=False)
            print(f"Информация о треках сохранена: {tracks_csv}")
        
        # Сохраняем точки (отдельно для каждого трека или один файл)
        all_points = []
        for idx, track in enumerate(self.tracks_data):
            if track['trackpoints']:
                points_df = pd.DataFrame(track['trackpoints'])
                points_df['track_opnum'] = track.get('OPNUM', np.nan)
                points_df['track_idx'] = idx
                all_points.append(points_df)
        
        if all_points:
            combined_points = pd.concat(all_points, ignore_index=True)
            points_csv = output_path / "all_track_points.csv"
            combined_points.to_csv(points_csv, index=False)
            print(f"Точки всех треков сохранены: {points_csv}")


# Пример использования
if __name__ == "__main__":
    # Настройки
    ARCHIVE_PATH = r"C:\Users\Mach\OneDrive\Desktop\Диплом\Project\resourses\data\SFO2006Q1-1.zip"
    EXTRACT_DIR = r"C:\Users\Mach\OneDrive\Desktop\Диплом\Project\resourses\data"
    
    parser = TrackParserFromArchive(ARCHIVE_PATH, EXTRACT_DIR)
    
    if parser.check_archive_exists():
        # Парсим диапазон файлов
        tracks = parser.parse_range(1, 10)
        
        # Статистика
        print("\n" + "="*60)
        print("СТАТИСТИКА")
        print("="*60)
        stats = parser.get_statistics()
        for key, value in stats.items():
            if key == 'field_statistics':
                print(f"\nПоля с пропусками (NaN):")
                for field, stat in value.items():
                    print(f"  {field}: {stat['nan_count']} пропусков ({stat['nan_percent']:.1f}%)")
            else:
                print(f"{key}: {value}")
        
        # DataFrame
        df = parser.get_dataframe()
        print(f"\nDataFrame shape: {df.shape}")
        print("\nПервые 10 строк:")
        print(df.head(10))
        
        # Проверка NaN
        print(f"\nNaN в колонке airportid: {df['airportid'].isna().sum()}")
        print(f"NaN в колонке ACID: {df['ACID'].isna().sum()}")
        
        # Сохраняем в CSV
        parser.save_to_csv()
        
    else:
        print(f"Архив не найден: {ARCHIVE_PATH}")