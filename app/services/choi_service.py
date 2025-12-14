"""
Choi 알고리즘 통합 서비스

이 모듈은 Choi 알고리즘의 전체 파이프라인(필터링 -> 이상 탐지 -> KPI 분석)을
순차적으로 실행하고 결과를 종합하는 오케스트레이터 역할을 수행합니다.

Pipeline Steps:
1. 데이터 유효성 검증
2. 필터링 (6장)
3. 이상 탐지 (4장)
4. KPI 통계 분석 (5장)
5. 최종 결과 종합
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.models.judgement import PegSampleSeries, FilteringResult, MainKPIJudgement
from app.services.choi_strategy_factory import get_choi_strategy_factory
from app.exceptions import ChoiAlgorithmError

logger = logging.getLogger(__name__)

class ChoiService:
    """
    Choi 알고리즘 실행 서비스
    """
    
    def __init__(self):
        self.logger = logger
        self.factory = get_choi_strategy_factory()
    
    def analyze(self, 
                input_data: Dict[str, Dict[str, List[PegSampleSeries]]], 
                config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Choi 알고리즘 전체 분석 실행
        
        Args:
            input_data: KPI 토픽별/셀별/PEG별 데이터 
                        Format: {topic: {cell_id: [PegSampleSeries, ...]}}
            config_overrides: 설정 오버라이드 (Optional)
            
        Returns:
            Dict[str, Any]: 분석 결과 종합
        """
        start_time = datetime.now()
        
        try:
            # 1. Strategy 생성
            filtering_strategy, judgement_strategy = self.factory.create_strategy_pair()
            base_config = self.factory.get_configuration()
            
            # TODO: config_overrides 적용 로직 (필요 시 구현)
            
            results = {
                "metadata": {
                    "timestamp": start_time.isoformat(),
                    "algorithm_version": "1.0.0"
                },
                "cells": {}
            }
            
            # 셀 단위로 반복하지 않고, 전체 데이터를 넘겨서 처리하는 구조인지 확인 필요.
            # ChoiFiltering.apply()는 Dict[cell_id, List[PegSampleSeries]]를 받음 (단일 토픽 기준인듯?)
            # 하지만 input_data는 Topic 단위로 들어올 수 있음.
            # 구조: Topic -> Cell -> PEGs
            
            # 여기서는 편의상 "모든 셀의 모든 PEG 데이터"를 하나의 Dict로 Flatten해서 필터링/이상탐지에 넣거나,
            # 토픽별로 Loop를 돌려야 함.
            # Choi doc: 필터링과 이상탐지는 "셀 단위"로 수행됨.
            
            # 데이터 재구조화: Cell -> List[PegSampleSeries] (모든 토픽의 PEG 포함)
            per_cell_data = self._flatten_data_by_cell(input_data)
            
            # 2. 필터링 (6장)
            self.logger.info("Starting Filtering Step (Chapter 6)")
            # config.filtering을 dict로 변환하거나 객체 속성 접근
            filtering_config_dict = {
                "min_threshold": base_config.filtering.min_threshold,
                "max_threshold": base_config.filtering.max_threshold,
                "filter_ratio": base_config.filtering.filter_ratio
            }
            filtering_result = filtering_strategy.apply(per_cell_data, filtering_config_dict)
            
            # 3. 이상 탐지 (4장)
            self.logger.info("Starting Anomaly Detection Step (Chapter 4)")
            detection_config_dict = {
                "alpha_0": base_config.abnormal_detection.alpha_0,
                "beta_3": base_config.abnormal_detection.beta_3,
                "enable_range_check": True, # 기본값
                "enable_new_check": True,
                "enable_nd_check": True,
                "enable_zero_check": True,
                "enable_high_delta_check": True
            }
            anomaly_result = judgement_strategy.detect_abnormal_stats(per_cell_data, detection_config_dict)
            
            # 4. KPI 통계 분석 (5장)
            self.logger.info("Starting KPI Analysis Step (Chapter 5)")
            analysis_config_dict = {
                "beta_0": base_config.stats_analyzing.beta_0,
                "beta_1": base_config.stats_analyzing.beta_1,
                "beta_2": base_config.stats_analyzing.beta_2,
                "beta_3": base_config.abnormal_detection.beta_3, # High Delta uses beta_3
                "beta_4": base_config.stats_analyzing.beta_4,
                "beta_5": base_config.stats_analyzing.beta_5
            }
            
            # KPI 분석은 '토픽' 단위로 수행 (Main/Sub 관계)
            # ChoiJudgement.analyze_kpi_stats는 kpi_data(Topic구조), filtering_result, config를 받음
            kpi_judgements = judgement_strategy.analyze_kpi_stats(
                input_data, 
                filtering_result, 
                analysis_config_dict
            )
            
            # 5. 결과 종합
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_ms": processing_time
                },
                "filtering": {
                    "valid_time_slots": filtering_result.valid_time_slots,
                    "filter_efficiency": filtering_result.filter_ratio
                },
                "abnormal_detection": {
                    "anomalies": anomaly_result.display_results
                },
                "kpi_judgement": {
                    topic: judgement.model_dump() for topic, judgement in kpi_judgements.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Choi Algorithm analysis failed: {e}", exc_info=True)
            raise ChoiAlgorithmError(f"Analysis failed: {str(e)}")

    def _flatten_data_by_cell(self, kpi_data: Dict[str, Dict[str, List[PegSampleSeries]]]) -> Dict[str, List[PegSampleSeries]]:
        """
        토픽별 데이터를 셀별 데이터로 평탄화
        In: {topic: {cell_id: [pegs]}}
        Out: {cell_id: [all_pegs]}
        """
        flattened = {}
        for topic_data in kpi_data.values():
            for cell_id, pegs in topic_data.items():
                if cell_id not in flattened:
                    flattened[cell_id] = []
                flattened[cell_id].extend(pegs)
        return flattened
