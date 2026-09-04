"""
Полный анализ траекторий: секторизация → центроиды → аномалии
"""

import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from typing import List, Dict, Any

from algorithms.ransac_mlesac import RANSACClustering, MLESACClustering
from algorithms.centroid_modeling import CentroidModeler
from algorithms.anomaly_detection import AnomalyDetector, RiskAssessor
from algorithms.sector_generation import SectorGenerator
from algorithms.controller_workload import ControllerWorkloadModel
from algorithms.pans_ops_risk import PansOpsRiskDetector
from algorithms.runway_optimization import RunwayOptimizer
from algorithms.pbn_validation import PbnValidator
from algorithms.procedure_export import ProcedureExporter
from algorithms.procedure_quality import ProcedureQualityAnalyzer
from algorithms.what_if_analysis import WhatIfProcedureAnalyzer


class FullAnalysisWorker(QThread):
    """
    Поток для выполнения полного анализа:
    Этап 1: Секторизация (выделение пучков) - RANSAC/MLESAC
    Этап 2: Моделирование центроидов - CPM с мерой косинуса
    Этап 3: Детектирование аномалий - ранжирование и выделение выбросов
    """
    
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    stage_completed = pyqtSignal(str, dict)
    
    def __init__(self, tracks_dict: dict, algorithm: str = 'mlesac',
                 use_cpm: bool = True, anomaly_threshold: float = 95.0,
                 restricted_zones: list = None, settings: dict = None):
        """
        Args:
            tracks_dict: словарь траекторий
            algorithm: алгоритм секторизации ('ransac', 'mlesac')
            use_cpm: использовать ли CPM для центроидов
            anomaly_threshold: порог аномалий (процентиль) - для совместимости
        """
        super().__init__()
        self.tracks_dict = tracks_dict
        self.algorithm = algorithm
        self.use_cpm = use_cpm
        self.anomaly_threshold = anomaly_threshold
        self.restricted_zones = restricted_zones or []
        self.settings = settings or {}
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        """Запуск полного анализа"""
        results = {
            'clusters': {},
            'sectors': {},
            'workload': {},
            'runway_optimization': {},
            'pbn_validation': {},
            'procedure_exports': {},
            'procedure_quality': {},
            'what_if': {},
            'centroids': {},
            'anomalies': {},
            'risks': {},
            'statistics': {}
        }
        
        start_time = time.time()
        
        # ========== ЭТАП 1: Секторизация (выделение пучков) ==========
        self.progress.emit(5, "Этап 1/3: Секторизация траекторий...")
        
        if self._is_cancelled:
            return
        
        try:
            if self.algorithm.lower() == 'ransac':
                from algorithms.ransac_mlesac import RANSACClustering, RANSACConfig
                clusterer = RANSACClustering(RANSACConfig(
                    distance_threshold=self.settings.get("ransac_distance_threshold", self.settings.get("ransac_threshold", 0.05)),
                    max_iterations=self.settings.get("ransac_max_iterations", 1000),
                    line_distance_threshold=self.settings.get("ransac_line_distance_threshold", 0.5),
                    max_line_iterations=self.settings.get("ransac_max_line_iterations", 300),
                    min_cluster_tracks=self.settings.get("ransac_min_cluster_tracks", 2),
                ))
            else:
                from algorithms.ransac_mlesac import MLESACClustering, MLESACConfig
                clusterer = MLESACClustering(MLESACConfig(
                    distance_threshold=self.settings.get("ransac_distance_threshold", self.settings.get("ransac_threshold", 0.05)),
                    max_iterations=self.settings.get("ransac_max_iterations", 1000),
                    line_distance_threshold=self.settings.get("ransac_line_distance_threshold", 0.5),
                    max_line_iterations=self.settings.get("ransac_max_line_iterations", 300),
                    min_cluster_tracks=self.settings.get("ransac_min_cluster_tracks", 2),
                    outlier_ratio=self.settings.get("mlesac_outlier_ratio", 0.3),
                    outlier_prob=self.settings.get("mlesac_outlier_prob", 0.1),
                ))
            
            clusters = clusterer.cluster(self.tracks_dict, verbose=False)
            if self._is_cancelled:
                return
            results['clusters'] = clusters
            
            self.stage_completed.emit('clustering', {
                'num_clusters': len(clusters),
                'total_tracks': sum(c['size'] for c in clusters.values())
            })
            
            self.progress.emit(30, f"Этап 1 завершен: найдено {len(clusters)} пучков")
            
        except Exception as e:
            self.error.emit(f"Ошибка на этапе секторизации: {e}")
            return

        self.progress.emit(34, "Этап 2/4: Построение 3D-оболочек секторов...")
        try:
            sector_generator = SectorGenerator()
            sectors = sector_generator.generate(self.tracks_dict, clusters, self.restricted_zones)
            if self._is_cancelled:
                return
            results['sectors'] = sectors
            for cluster_id, sector in sectors.items():
                clusters[cluster_id]['sector'] = sector
            self.stage_completed.emit('sectors', {
                'num_sectors': len(sectors),
                'total_entry_fixes': sum(len(s.get('entry_fixes', [])) for s in sectors.values()),
                'total_exit_fixes': sum(len(s.get('exit_fixes', [])) for s in sectors.values()),
            })
            self.progress.emit(38, f"Этап 2 завершен: построено {len(sectors)} 3D-секторов")
        except Exception as e:
            self.error.emit(f"Ошибка на этапе построения секторов: {e}")
            return

        self.progress.emit(39, "Расчет загруженности диспетчера по секторам...")
        try:
            workload_model = ControllerWorkloadModel()
            workload = workload_model.calculate(self.tracks_dict, clusters, sectors)
            if self._is_cancelled:
                return
            results['workload'] = workload
            for cluster_id, workload_data in workload.items():
                clusters[cluster_id]['workload'] = {
                    'workload_index': workload_data.get('workload_index', 0.0),
                    'peak_load': workload_data.get('peak_load', 0),
                    'conflict_events_count': workload_data.get('conflict_events_count', 0),
                    'avg_maneuver_complexity': workload_data.get('avg_maneuver_complexity', 0.0),
                    'avg_dwell_time_seconds': workload_data.get('avg_dwell_time_seconds', 0.0),
                }
            self.stage_completed.emit('workload', {
                'max_workload_index': max((w.get('workload_index', 0.0) for w in workload.values()), default=0.0),
                'total_conflicts': sum(w.get('conflict_events_count', 0) for w in workload.values()),
                'max_peak_load': max((w.get('peak_load', 0) for w in workload.values()), default=0),
            })
        except Exception as e:
            self.error.emit(f"Ошибка на этапе расчета загруженности диспетчера: {e}")
            return

        self.progress.emit(40, "Расчет очередности и оптимизации ВПП...")
        try:
            runway_optimizer = RunwayOptimizer()
            runway_optimization = runway_optimizer.optimize(self.tracks_dict, clusters, sectors)
            if self._is_cancelled:
                return
            results['runway_optimization'] = runway_optimization
            self.stage_completed.emit('runways', {
                'runways_count': runway_optimization.get('summary', {}).get('runways_count', 0),
                'spacing_violations': runway_optimization.get('summary', {}).get('spacing_violations', 0),
                'overloaded_count': len(runway_optimization.get('summary', {}).get('overloaded_runways', [])),
                'reassignment_plans': len(runway_optimization.get('reassignment', {}).get('plans', [])),
            })
        except Exception as e:
            self.error.emit(f"Ошибка на этапе оптимизации ВПП: {e}")
            return
        
        # ========== ЭТАП 2: Моделирование центроидов ==========
        self.progress.emit(42, "Этап 3/4: Моделирование центроидов...")
        
        if self._is_cancelled:
            return
        
        centroid_modeler = CentroidModeler(
            polynomial_degree=5,
            cpm_smoothing=2.0,
            max_iterations=self.settings.get("cpm_iterations", 20),
            convergence_threshold=1e-3,
            use_cpm=self.use_cpm,
            verbose=False
        )
        
        for cluster_id, cluster_data in clusters.items():
            if self._is_cancelled:
                return
            
            # Получаем треки в кластере
            track_ids = cluster_data['tracks']
            tracks_in_cluster = []
            for tid in track_ids:
                if tid in self.tracks_dict:
                    tracks_in_cluster.append(self.tracks_dict[tid])
            
            if len(tracks_in_cluster) < 2:
                results['centroids'][cluster_id] = {'error': 'Недостаточно треков'}
                continue
            
            # Строим центроид с использованием CPM и меры косинуса
            centroid_result = centroid_modeler.fit(
                tracks_in_cluster, 
                use_cpm=self.use_cpm,
                use_cosine_optimization=True
            )
            results['centroids'][cluster_id] = centroid_result
            
            # Добавляем центроид в информацию о кластере
            clusters[cluster_id]['centroid'] = centroid_result['centroid'].tolist()
            clusters[cluster_id]['reference_direction'] = centroid_result.get('reference_direction', [0, 0, 0]).tolist()
            clusters[cluster_id]['latent_trace'] = centroid_result.get('latent_trace', []).tolist() if centroid_result.get('latent_trace') is not None else []
            
            if clusters:
                self.progress.emit(42 + int((cluster_id + 1) / len(clusters) * 23),
                                 f"Этап 3: центроид для пучка {cluster_id + 1}/{len(clusters)}")
        
        self.stage_completed.emit('centroids', {
            'num_centroids': len(results['centroids']),
            'modeling_method': 'CPM' if self.use_cpm else 'косинусная оптимизация'
        })

        self.progress.emit(66, "Валидация PBN/RNP и подготовка проектных данных...")
        try:
            pbn_validation = PbnValidator().validate(clusters, results['sectors'])
            procedure_exports = ProcedureExporter().export(clusters, results['sectors'], self.tracks_dict)
            procedure_quality = ProcedureQualityAnalyzer().analyze(self.tracks_dict, clusters, results['sectors'])
            if self._is_cancelled:
                return
            results['pbn_validation'] = pbn_validation
            results['procedure_exports'] = procedure_exports
            results['procedure_quality'] = procedure_quality
            for cluster_id, cluster in clusters.items():
                cluster['pbn'] = {
                    'is_valid': pbn_validation.get(cluster_id, {}).get('is_valid', False),
                    'issues_count': len(pbn_validation.get(cluster_id, {}).get('issues', [])),
                    'rf_segments_count': pbn_validation.get(cluster_id, {}).get('rf_segments_count', 0),
                }
                cluster['procedure_quality'] = {
                    'quality_score': procedure_quality.get(cluster_id, {}).get('quality_score', 0.0),
                    'stable_approach_percent': procedure_quality.get(cluster_id, {}).get('stable_approach_percent', 0.0),
                    'repeatability': procedure_quality.get(cluster_id, {}).get('repeatability', 0.0),
                }
            self.stage_completed.emit('procedure_design', {
                'pbn_issues': sum(len(item.get('issues', [])) for item in pbn_validation.values()),
                'geojson_features': len(procedure_exports.get('geojson', {}).get('features', [])),
                'arinc_rows': len(procedure_exports.get('arinc_table', [])),
                'avg_quality': sum(item.get('quality_score', 0) for item in procedure_quality.values()) / max(1, len(procedure_quality)),
            })
        except Exception as e:
            self.error.emit(f"Ошибка на этапе PBN/RNP и проектных данных: {e}")
            return
        
        # ========== ЭТАП 3: Детектирование аномалий ==========
        self.progress.emit(70, "Этап 4/4: Детектирование аномалий...")
        
        if self._is_cancelled:
            return
        
        detector = AnomalyDetector(
            use_cosine_metric=True,
            anomaly_threshold=self.anomaly_threshold
        )
        rule_detector = PansOpsRiskDetector()
        risk_assessor = RiskAssessor()
        
        for cluster_id, cluster_data in clusters.items():
            if self._is_cancelled:
                return
            
            centroid = np.array(cluster_data.get('centroid', []))
            if len(centroid) == 0:
                results['anomalies'][cluster_id] = {'error': 'Нет центроида'}
                continue
            
            # Получаем треки в кластере
            track_ids = cluster_data['tracks']
            tracks_in_cluster = []
            track_ids_list = []
            
            for tid in track_ids:
                if tid in self.tracks_dict:
                    tracks_in_cluster.append(self.tracks_dict[tid])
                    track_ids_list.append(tid)
            
            # Детектируем аномалии с ранжированием
            anomaly_results = detector.detect_anomalies(tracks_in_cluster, track_ids_list, centroid)
            rule_risks = rule_detector.evaluate_tracks(tracks_in_cluster, track_ids_list)
            
            # Сохраняем результаты
            results['anomalies'][cluster_id] = {
                'results': [
                    {
                        'track_id': r.track_id,
                        'rank': r.rank,
                        'cosine_distance': r.cosine_distance,
                        'squared_distance_sum': r.squared_distance_sum,
                        'is_anomaly': r.is_anomaly,
                        'anomaly_score': r.anomaly_score,
                        'risk_level': r.risk_level,
                        'rule_based_risks': rule_risks.get(r.track_id, {}).get('risks', []),
                        'rule_risk_count': rule_risks.get(r.track_id, {}).get('risk_count', 0),
                        'rule_max_severity': rule_risks.get(r.track_id, {}).get('max_severity', 'Нет'),
                        'aircraft_category': rule_risks.get(r.track_id, {}).get('category', 'C'),
                        'risk_description': r.risk_description,
                        'recommendation': r.recommendation
                    } for r in anomaly_results
                ],
                'statistics': detector.get_anomaly_statistics(anomaly_results),
                'rule_statistics': self._aggregate_rule_risk_stats(rule_risks),
                'top_anomaly': anomaly_results[0].track_id if anomaly_results else None
            }
            
            # Оценка рисков
            risk_result = risk_assessor.assess_anomaly_risk(anomaly_results[0]) if anomaly_results else {}
            results['risks'][cluster_id] = risk_result
            
            # Обновляем информацию о кластере
            clusters[cluster_id]['anomalies_count'] = results['anomalies'][cluster_id]['statistics']['anomalies_count']
            clusters[cluster_id]['rule_risks_count'] = results['anomalies'][cluster_id]['rule_statistics']['total_rule_risks']
            clusters[cluster_id]['tracks_with_rule_risks'] = results['anomalies'][cluster_id]['rule_statistics']['tracks_with_rule_risks']
            clusters[cluster_id]['top_anomaly'] = results['anomalies'][cluster_id]['top_anomaly']
            clusters[cluster_id]['risk_level'] = risk_result.get('risk_level', 'неизвестен')
            
            if clusters:
                self.progress.emit(70 + int((cluster_id + 1) / len(clusters) * 30),
                                 f"Этап 4: аномалии для пучка {cluster_id + 1}/{len(clusters)}")
        
        # ========== Финальная статистика ==========
        anomaly_stats = self._aggregate_anomaly_stats(results['anomalies'])
        rule_risk_stats = self._aggregate_all_rule_risk_stats(results['anomalies'])
        results['what_if'] = WhatIfProcedureAnalyzer().compare(results)
        
        results['statistics'] = {
            'total_tracks': len(self.tracks_dict),
            'total_clusters': len(clusters),
            'total_sectors': len(results['sectors']),
            'max_workload_index': max((w.get('workload_index', 0.0) for w in results['workload'].values()), default=0.0),
            'total_conflict_events': sum(w.get('conflict_events_count', 0) for w in results['workload'].values()),
            'runway_optimization_stats': results['runway_optimization'].get('summary', {}),
            'pbn_issues_count': sum(len(item.get('issues', [])) for item in results['pbn_validation'].values()),
            'procedure_export_rows': len(results['procedure_exports'].get('arinc_table', [])),
            'avg_procedure_quality': sum(item.get('quality_score', 0) for item in results['procedure_quality'].values()) / max(1, len(results['procedure_quality'])),
            'what_if_delta': results['what_if'].get('delta', {}),
            'total_centroids': len(results['centroids']),
            'anomaly_stats': anomaly_stats,
            'rule_risk_stats': rule_risk_stats,
            'execution_time': time.time() - start_time,
            'algorithm': self.algorithm,
            'use_cpm': self.use_cpm,
            'modeling_method': 'CPM с косинусной метрикой'
        }
        
        self.progress.emit(100, "Анализ завершен!")
        self.finished.emit(results)
    
    def _aggregate_anomaly_stats(self, anomalies: dict) -> dict:
        """Агрегация статистики по аномалиям"""
        total_tracks = 0
        total_anomalies = 0
        high_risk_count = 0
        
        for cluster_id, cluster_data in anomalies.items():
            stats = cluster_data.get('statistics', {})
            total_tracks += stats.get('total_tracks', 0)
            total_anomalies += stats.get('anomalies_count', 0)
            high_risk_count += stats.get('high_risk_count', 0)
        
        return {
            'total_tracks_analyzed': total_tracks,
            'total_anomalies_detected': total_anomalies,
            'anomaly_percentage': (total_anomalies / total_tracks * 100) if total_tracks > 0 else 0,
            'high_risk_count': high_risk_count,
            'high_risk_percentage': (high_risk_count / total_tracks * 100) if total_tracks > 0 else 0
        }

    def _aggregate_rule_risk_stats(self, rule_risks: dict) -> dict:
        type_counts = {}
        total = 0
        tracks_with_risks = 0
        high_severity = 0

        for risk_data in rule_risks.values():
            risks = risk_data.get('risks', [])
            if risks:
                tracks_with_risks += 1
            if risk_data.get('max_severity') == 'Высокая':
                high_severity += 1
            for risk in risks:
                risk_type = risk.get('type', 'Неизвестный риск')
                type_counts[risk_type] = type_counts.get(risk_type, 0) + 1
                total += 1

        return {
            'total_rule_risks': total,
            'tracks_with_rule_risks': tracks_with_risks,
            'high_severity_tracks': high_severity,
            'risk_type_counts': type_counts,
        }

    def _aggregate_all_rule_risk_stats(self, anomalies: dict) -> dict:
        total = 0
        tracks = 0
        high = 0
        type_counts = {}
        for cluster_data in anomalies.values():
            stats = cluster_data.get('rule_statistics', {})
            total += stats.get('total_rule_risks', 0)
            tracks += stats.get('tracks_with_rule_risks', 0)
            high += stats.get('high_severity_tracks', 0)
            for risk_type, count in stats.get('risk_type_counts', {}).items():
                type_counts[risk_type] = type_counts.get(risk_type, 0) + count
        return {
            'total_rule_risks': total,
            'tracks_with_rule_risks': tracks,
            'high_severity_tracks': high,
            'risk_type_counts': type_counts,
        }
