"""
Choi 판정 서비스 구현

이 모듈은 TES.web_Choi.md 문서의 4장(이상 탐지)과 5장(통계 분석) 
판정 알고리즘을 Strategy 패턴으로 구현합니다.

주요 기능:
- 4장: Abnormal Stats Detecting Algorithm
  - Range, New, ND, Zero, High Delta 탐지
  - α0 규칙에 따른 결과 표시 로직
- 5장: Stats Analyzing Algorithm  
  - Can't Judge, High Variation, Improve/Degrade 판정
  - Similar/Delta 계층 판정 (β0-β5 임계값 적용)
  - Main/Sub KPI 결과 종합

PRD 참조: 섹션 2.2 (이상 탐지), 2.3 (통계 분석)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from ..models.judgement import (
    PegSampleSeries,
    FilteringResult,
    AbnormalDetectionResult,
    MainKPIJudgement,
    PegPeriodStats,
    PegCompareMetrics,
    PegCompareDecision,
    JudgementType,
    CompareDetail,
    KPIPositivity,
    SimpleKPIJudgement
)
from ..services.strategies import BaseJudgementStrategy
from ..services.anomaly_detectors import (
    AnomalyDetectorFactory,
    AnomalyDetectionResult as DetectorResult,
    DimsDataProvider,
    MockDimsDataProvider
)
from ..services.kpi_analyzers import (
    KPIAnalyzerFactory,
    KPIAnalysisResult,
    BaseKPIAnalyzer
)

logger = logging.getLogger(__name__)


class ChoiJudgement(BaseJudgementStrategy):
    """
    Choi 판정 알고리즘 구현 (4장, 5장)
    
    TES.web_Choi.md 문서의 4장, 5장 판정 알고리즘을 정확히 구현합니다.
    """
    
    def __init__(self, 
                 detector_factory: Optional[AnomalyDetectorFactory] = None,
                 analyzer_factory: Optional[KPIAnalyzerFactory] = None,
                 dims_provider: Optional[DimsDataProvider] = None):
        """
        Choi 판정 전략 초기화
        
        Args:
            detector_factory: 이상 탐지기 팩토리 (의존성 주입)
            analyzer_factory: KPI 분석기 팩토리 (의존성 주입)
            dims_provider: DIMS 데이터 제공자 (의존성 주입)
        """
        super().__init__("ChoiJudgement", "1.0.0")
        
        # 의존성 주입 (Dependency Injection)
        self.dims_provider = dims_provider or MockDimsDataProvider()
        self.detector_factory = detector_factory or AnomalyDetectorFactory(self.dims_provider)
        self.analyzer_factory = analyzer_factory or KPIAnalyzerFactory()
        
        # 이상 탐지기들 초기화 (Lazy Loading)
        self._detectors = None
        
        # KPI 분석기들 초기화 (Lazy Loading)
        self._analyzers = None
        
        self.logger.info(f"Choi Judgement 알고리즘 초기화 완료 "
                        f"(DIMS provider: {type(self.dims_provider).__name__}, "
                        f"Factories: detector, analyzer)")
    
    def apply(self,
              filtered_data: Dict[str, List[PegSampleSeries]],
              filtering_result: FilteringResult,
              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Algorithm Entry] 판정 알고리즘 전체 실행
        
        4장(이상 탐지)과 5장(통계 분석) 알고리즘을 순차적으로 실행하고 결과를 종합합니다.
        
        Args:
            filtered_data: 필터링된 PEG 데이터 (검증 대상)
            filtering_result: 필터링 결과 (유효 시간대 정보 등)
            config: 판정 설정을 담은 딕셔너리
            
        Returns:
            Dict[str, Any]: 최종 판정 결과 (abnormal_detection, kpi_judgement)
        """
        try:
            self.logger.info(f"🚀 Choi 판정 알고리즘 시작: {len(filtered_data)}개 Cell 처리")
            
            # 입력 검증
            if not self.validate_input(filtered_data, filtering_result, config):
                self.logger.error("❌ 입력 데이터 검증 실패")
                raise ValueError("Invalid input data for judgement")
            
            # 1. 4장: 이상 탐지 실행 (Range, New, ND, Zero, High Delta)
            self.logger.debug("👉 [Step 1] 4장: 이상 통계(Abnormal Stats) 탐지 실행")
            abnormal_detection_config = config.get("abnormal_detection", {})
            abnormal_result = self.detect_abnormal_stats(filtered_data, abnormal_detection_config)
            
            # 2. 5장: KPI 통계 분석 실행 (L2/L3 분석)
            self.logger.debug("👉 [Step 2] 5장: KPI 통계 분석(Stats Analysis) 실행")
            kpi_data = self._organize_data_by_kpi_topics(filtered_data, config.get("kpi_definitions", {}))
            stats_config = config.get("stats_analyzing", {})
            
            # KPI 분석기들 초기화 (Lazy Loading)
            if self._analyzers is None:
                self._analyzers = self.analyzer_factory.create_priority_ordered_analyzers()
                self.logger.debug(f"🛠️ KPI 분석기 초기화 완료: {len(self._analyzers)}개")
            
            kpi_judgement_result = self.analyze_kpi_stats(kpi_data, filtering_result, stats_config)
            
            # 3. 결과 종합
            result = {
                "abnormal_detection": abnormal_result,
                "kpi_judgement": kpi_judgement_result,
                "processing_metadata": {
                    "algorithm_version": self.version,
                    "processed_cells": len(filtered_data),
                    "processed_pegs": sum(len(series_list) for series_list in filtered_data.values())
                }
            }
            
            self.logger.info(f"✅ Choi 판정 알고리즘 완료: "
                           f"이상유형={len(abnormal_result.model_dump())}, "
                           f"KPI토픽={len(kpi_judgement_result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 판정 알고리즘 실행 중 치명적 오류: {e}", exc_info=True)
            raise RuntimeError(f"Judgement failed: {e}")
    
    def detect_abnormal_stats(self,
                             peg_data: Dict[str, List[PegSampleSeries]],
                             config: Dict[str, Any]) -> AbnormalDetectionResult:
        """
        [4장] 이상 통계 탐지 (Abnormal Stats Detection)
        
        Range, New, ND, Zero, High Delta 등 통계적 이상치를 탐지합니다.
        탐지 후 α0 규칙(최소 셀 수 조건)을 적용하여 최종 표시 여부를 결정합니다.
        
        Args:
            peg_data: 분석 대상 PEG 데이터
            config: 이상 탐지 설정 (임계값 등)
            
        Returns:
            AbnormalDetectionResult: 탐지된 이상 결과 및 표시 여부
        """
        try:
            self.logger.debug("🔎 [4장] 이상 통계 탐지 시작")
            
            # 설정값 추출
            alpha_0 = config.get("alpha_0", 2)
            beta_3 = config.get("beta_3", 500.0)
            detection_types = config.get("detection_types", {})
            enable_range_check = config.get("enable_range_check", True)
            
            # 이상 탐지기들 초기화 (Lazy Loading)
            if self._detectors is None:
                self._detectors = self.detector_factory.create_all_detectors()
                self.logger.debug(f"🛠️ 탐지기 초기화 완료: {len(self._detectors)}개")
            
            # 각 이상 탐지 규칙 실행 (SOLID 원칙 준수)
            detection_results = {}
            
            for detector_type, detector in self._detectors.items():
                if detection_types.get(detector_type, True):
                    try:
                        # 각 탐지기는 독립적으로 실행 (Single Responsibility)
                        result = detector.detect(peg_data, config)
                        detection_results[result.anomaly_type] = result
                        
                        if result.affected_cells:
                             self.logger.debug(f"⚠️ {detector_type} 탐지됨: {len(result.affected_cells)}개 Cell")
                             
                    except Exception as e:
                        self.logger.error(f"❌ {detector_type} 탐지 중 오류: {e}")
                        # 하나의 탐지기 실패가 전체를 중단시키지 않음 (견고한 오류 처리)
                        continue
            
            # 탐지 결과를 기존 형태로 변환
            converted_results = self._convert_detection_results(detection_results)
            
            # α0 규칙 적용하여 표시 여부 결정 (최소 셀 수 미만 시 숨김)
            display_results = self._apply_alpha_zero_rule(converted_results, alpha_0)
            
            # 결과 객체 생성
            result = AbnormalDetectionResult(
                range_violations=converted_results.get("Range", {}),
                new_statistics=converted_results.get("New", {}),
                nd_anomalies=converted_results.get("ND", {}),
                zero_anomalies=converted_results.get("Zero", {}),
                high_delta_anomalies=converted_results.get("High Delta", {}),
                display_results=display_results
            )
            
            displayed_count = sum(1 for display in display_results.values() if display)
            self.logger.info(f"✅ 이상 탐지 완료: {displayed_count}개 유형 표시 (α0 규칙 적용됨)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 이상 탐지 프로세스 중 오류: {e}", exc_info=True)
            raise
    
    def analyze_kpi_stats(self,
                          kpi_data: Dict[str, Dict[str, List[PegSampleSeries]]],
                          filtering_result: FilteringResult,
                          config: Dict[str, Any]) -> Dict[str, MainKPIJudgement]:
        """
        [5장] KPI 통계 분석 (KPI Stats Analysis)
        
        각 KPI 토픽(Main/Sub 그룹)에 대해 통계적 분석을 수행합니다.
        1. Main/Sub 각각에 대해 β 임계값 기반 규칙(High Variation, Improve, Degrade 등)을 적용합니다.
        2. Main KPI와 Sub KPI의 분석 결과를 종합하여 최종 판정(OK/NOK/POK)을 내립니다.
        
        Args:
            kpi_data: KPI 토픽별 데이터 ({ 'topic': {'main': [], 'subs': []} })
            filtering_result: 필터링 결과
            config: KPI 분석 설정 (β 값, 우선순위 등)
            
        Returns:
            Dict[str, MainKPIJudgement]: 토픽별 종합 판정 결과
        """
        try:
            self.logger.debug("🔎 [5장] KPI 통계 분석 시작")
            
            # β 설정값 추출 (기본값 설정)
            beta_values = {
                "beta_0": config.get("beta_0", 1000.0),
                "beta_1": config.get("beta_1", 5.0),
                "beta_2": config.get("beta_2", 10.0),
                "beta_3": config.get("beta_3", 500.0),
                "beta_4": config.get("beta_4", 10.0),
                "beta_5": config.get("beta_5", 3.0)
            }
            
            rule_priorities = config.get("rule_priorities", {})
            
            kpi_judgement_results = {}
            
            for topic_name, topic_data in kpi_data.items():
                self.logger.debug(f"👉 토픽 분석 시작: {topic_name}")
                
                try:
                    # 1. Main KPI 분석
                    main_judgement = self._analyze_main_kpi(
                        topic_data.get("main", []), 
                        beta_values, 
                        rule_priorities
                    )
                    
                    # 2. Sub KPI 데이터 준비 및 분석
                    sub_data_list = topic_data.get("subs", [])
                    sub_names = topic_data.get("sub_kpi_names", [])
                    
                    sub_map = {}
                    if len(sub_data_list) == len(sub_names):
                        for name, data in zip(sub_names, sub_data_list):
                            if data:
                                sub_map[name] = data
                    
                    sub_results = self._analyze_sub_kpis(
                        sub_map,
                        beta_values,
                        rule_priorities
                    )
                    
                    # 3. 최종 결과 종합 (5.4 규칙 적용)
                    final_judgement = self._combine_main_sub_results(
                        main_judgement, 
                        sub_results, 
                        topic_name,
                        topic_data.get("main_kpi_name", topic_name)
                    )
                    
                    if final_judgement:
                        kpi_judgement_results[topic_name] = final_judgement
                        
                except Exception as e:
                    self.logger.error(f"❌ 토픽 '{topic_name}' 분석 중 오류 스킵: {e}", exc_info=True)
                    continue
            
            self.logger.info(f"✅ KPI 분석 완료: {len(kpi_judgement_results)}개 토픽 판정됨")
            return kpi_judgement_results
            
        except Exception as e:
            self.logger.error(f"❌ KPI 분석 프로세스 중 치명적 오류: {e}", exc_info=True)
            raise
    
    # =============================================================================
    # KPI 분석 구현 메서드들 (5장)
    # =============================================================================
    
    def _organize_data_by_kpi_topics(self, 
                                   filtered_data: Dict[str, List[PegSampleSeries]], 
                                   kpi_definitions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        KPI 토픽별 데이터 재구성 (Data Organization)
        
        설정된 KPI 정의에 따라 Main KPI와 Sub KPI 데이터를 토픽 단위로 묶습니다.
        
        Args:
            filtered_data: 필터링된 PEG 데이터
            kpi_definitions: KPI 정의 설정 ({topic: {main: "...", subs: [...]}})
            
        Returns:
             Dict[str, Dict[str, Any]]: {'topic': {'main': [...], 'subs': [[...], ...]}} 구조
        """
        organized_data = {}
        
        try:
            for topic_name, definition in kpi_definitions.items():
                # Main KPI Data 추출
                main_kpi_name = definition.get("main")
                if not main_kpi_name:
                    self.logger.warning(f"⚠️ 토픽 '{topic_name}'에 Main KPI 정의가 없습니다.")
                    continue
                    
                main_data = filtered_data.get(main_kpi_name, [])
                if not main_data:
                    self.logger.debug(f"ℹ️ 토픽 '{topic_name}'의 Main KPI 데이터가 없습니다 ({main_kpi_name})")
                
                # Sub KPI Data 추출
                sub_data_list = []
                for sub_kpi_name in definition.get("subs", []):
                    # 각 Sub KPI 데이터는 List[PegSampleSeries] 형태
                    sub_data = filtered_data.get(sub_kpi_name, [])
                    sub_data_list.append(sub_data) 
                
                # Main 데이터가 존재하는 경우에만 토픽 구성 (Main 기반 분석 전제)
                if main_data:
                    organized_data[topic_name] = {
                        "main": main_data,
                        "subs": sub_data_list,
                        "main_kpi_name": main_kpi_name,
                        "sub_kpi_names": definition.get("subs", [])
                    }
                    self.logger.debug(f"📦 토픽 구성 완료: {topic_name} (Sub: {len(sub_data_list)}개)")
                    
            return organized_data
            
        except Exception as e:
            self.logger.error(f"❌ KPI 데이터 재구성 중 오류: {e}")
            return {}
    
    def _analyze_main_kpi(self, 
                         main_kpi_data: List[PegSampleSeries], 
                         beta_values: Dict[str, float], 
                         rule_priorities: Dict[str, int]) -> Optional[KPIAnalysisResult]:
        """Main KPI 단일 분석 수행"""
        if not main_kpi_data:
            return None
            
        return self._analyze_single_kpi(
            main_kpi_data, 
            main_kpi_data[0].peg_name, 
            beta_values, 
            rule_priorities
        )
    
    def _analyze_sub_kpis(self, 
                         sub_kpi_data_map: Dict[str, List[PegSampleSeries]], 
                         beta_values: Dict[str, float], 
                         rule_priorities: Dict[str, int]) -> Dict[str, KPIAnalysisResult]:
        """Sub KPI 목록에 대한 일괄 분석 수행"""
        sub_results = {}
        
        for kpi_name, kpi_data in sub_kpi_data_map.items():
            result = self._analyze_single_kpi(
                kpi_data, 
                kpi_name, 
                beta_values, 
                rule_priorities
            )
            
            if result:
                sub_results[kpi_name] = result
                
        return sub_results

    def _analyze_single_kpi(self,
                           kpi_series_list: List[PegSampleSeries],
                           kpi_name: str,
                           beta_values: Dict[str, float],
                           rule_priorities: Dict[str, int]) -> Optional[KPIAnalysisResult]:
        """
        단일 KPI 분석 (Single KPI Analysis)
        
        Chain of Responsibility 패턴을 사용하여 우선순위에 따라 분석기(Analyzer)를 순차적으로 적용합니다.
        가장 먼저 매칭되는 분석기의 결과를 반환합니다.
        
        Args:
            kpi_series_list: KPI 시계열 데이터
            kpi_name: KPI 이름
            beta_values: β 임계값들
            rule_priorities: 규칙 우선순위
            
        Returns:
            Optional[KPIAnalysisResult]: 분석 결과 (매칭된 규칙이 없으면 None)
        """
        try:
            if not kpi_series_list:
                self.logger.warning(f"⚠️ 데이터 없음: {kpi_name}")
                return None
            
            # 첫 번째 시리즈 사용 (Main KPI 또는 대표 Sub KPI)
            series = kpi_series_list[0]
            
            # 기본 통계 계산
            pre_stats = self._calculate_period_stats(series.pre_samples)
            post_stats = self._calculate_period_stats(series.post_samples)
            compare_metrics = self._calculate_compare_metrics(pre_stats, post_stats)
            
            # 우선순위 순서로 분석기 적용
            analysis_config = {**beta_values, **rule_priorities}
            
            for analyzer in self._analyzers:
                try:
                    result = analyzer.analyze(pre_stats, post_stats, compare_metrics, analysis_config)
                    if result:
                        self.logger.debug(f"📋 {kpi_name}: '{analyzer.analyzer_name}' 규칙 적용됨")
                        return result
                    else:
                        pass
                except Exception as e:
                    self.logger.error(f"❌ 분석기 '{analyzer.analyzer_name}' 실행 오류 ({kpi_name}): {e}")
                    continue
            
            # 모든 분석기가 적용되지 않은 경우
            self.logger.warning(f"⚠️ 매칭되는 분석 규칙 없음: {kpi_name} (기본값 사용)")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 단일 KPI 분석 오류 ({kpi_name}): {e}", exc_info=True)
            return None
    
    def _combine_main_sub_results(self, 
                                 main_result: Optional[KPIAnalysisResult], 
                                 sub_results: Dict[str, KPIAnalysisResult],
                                 topic_name: str,
                                 main_kpi_name_arg: str) -> Optional[MainKPIJudgement]:
        """
        [토픽 종합] Main/Sub KPI 결과 종합 (Topic Combination)
        
        5.4장 규칙을 적용하여 Main KPI와 Sub KPI의 분석 결과를 하나로 종합하고,
        최종 판정(Final Decision)을 도출합니다.
        
        Args:
            main_result: Main KPI 분석 결과
            sub_results: Sub KPI 분석 결과 딕셔너리 (kpi_name -> result)
            topic_name: 토픽 이름
            main_kpi_name_arg: Main KPI 명
            
        Returns:
            Optional[MainKPIJudgement]: 최종 판정 결과
        """
        try:
            if not main_result:
                self.logger.warning(f"⚠️ Main KPI 결과 없음 (Topic: {topic_name}) - 종합 불가")
                return None
            
            # 1. Main Result를 SimpleKPIJudgement로 변환 (처리 편의성)
            main_simple = self._convert_to_simple_judgement(main_result)
            
            # 2. Sub Results 변환 및 상세 정보 추출
            sub_simple_dict = {}
            sub_result_details = []
            
            for sub_name, sub_res in sub_results.items():
                sub_simple = self._convert_to_simple_judgement(sub_res)
                sub_simple_dict[sub_name] = sub_simple
                
                # 상세 결과 저장
                sub_result_details.append({
                    "kpi_name": sub_name,
                    "judgement": sub_simple.judgement_type,
                    "detail": sub_simple.compare_detail,
                    "reason": sub_simple.reasoning,
                    "metrics": sub_simple.metrics
                })
            
            # 3. 최종 판정 로직 적용 (_apply_final_summary_rules 활용)
            final_simple = self._apply_final_summary_rules(
                topic_name,
                main_simple,
                sub_simple_dict
            )
            
            self.logger.debug(f"🗳️ 토픽 '{topic_name}' 최종 판정: {final_simple.judgement_type} "
                            f"(Main: {main_simple.judgement_type}, Sub: {len(sub_results)}개)")

            # 4. MainKPIJudgement 객체 생성 및 반환
            return MainKPIJudgement(
                main_kpi_name=main_kpi_name_arg,
                main_result=main_result.judgement_type,
                main_decision=PegCompareDecision(
                    detail=main_result.compare_detail,
                    reason=main_result.reasoning,
                    thresholds_used=main_result.thresholds_used or {},
                    confidence=main_result.confidence
                ),
                sub_results=sub_result_details,
                final_result=final_simple.judgement_type,
                summary_text=final_simple.reasoning,
                # FIXME: 통계 객체 직접 접근 불가로 인한 빈 객체 사용 (추후 개선 필요)
                pre_stats=PegPeriodStats(),
                post_stats=PegPeriodStats(), 
                compare_metrics=PegCompareMetrics()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Main/Sub 결과 종합 중 오류 ({topic_name}): {e}", exc_info=True)
            return None

    def _convert_to_simple_judgement(self, result: KPIAnalysisResult) -> SimpleKPIJudgement:
        """
        [Helper] KPIAnalysisResult -> SimpleKPIJudgement 변환
        
        복잡한 분석 결과를 요약 로직 처리에 적합한 단순 형태로 변환합니다.
        """
        from app.models.judgement import SimpleKPIJudgement
        return SimpleKPIJudgement(
            judgement_type=result.judgement_type,
            compare_detail=result.compare_detail,
            reasoning=result.reasoning,
            confidence=result.confidence,
            metrics=result.metrics or {},
            thresholds_used=result.thresholds_used or {}
        )
    
    # =============================================================================
    # 통계 계산 유틸리티 메서드들
    # =============================================================================
    
    # =============================================================================
    # 통계 계산 유틸리티 메서드들
    # =============================================================================
    
    def _calculate_period_stats(self, samples: List[Optional[float]]) -> PegPeriodStats:
        """
        [Helper] 기간별 통계 계산 (Period Statistics)
        
        주어진 샘플 리스트에 대한 기본 통계(평균, 최소, 최대, 표준편차, ND비율, Zero비율 등)를 계산합니다.
        
        Args:
            samples: 샘플 데이터 리스트 (None 포함 가능)
            
        Returns:
            PegPeriodStats: 계산된 통계 객체
        """
        try:
            # None 값 제거 (유효 샘플만 추출)
            valid_samples = [s for s in samples if s is not None]
            
            # ND 및 Zero 비율 계산
            total_count = len(samples)
            nd_count = sum(1 for s in samples if s is None)
            nd_ratio = nd_count / total_count if total_count > 0 else 0
            
            if not valid_samples:
                # 유효 데이터가 없는 경우 (100% ND)
                return PegPeriodStats(
                    sample_count=0,
                    nd_ratio=nd_ratio,
                    zero_ratio=0.0,
                    mean=None, min=None, max=None, std=None, cv=None
                )
            
            np_samples = np.array(valid_samples)
            mean_val = float(np.mean(np_samples))
            std_val = float(np.std(np_samples))
            
            zero_count = sum(1 for s in valid_samples if s == 0.0)
            
            stats = PegPeriodStats(
                mean=mean_val,
                min=float(np.min(np_samples)),
                max=float(np.max(np_samples)),
                std=std_val,
                cv=std_val / mean_val if mean_val != 0 else None,
                nd_ratio=nd_ratio,
                zero_ratio=zero_count / len(valid_samples) if valid_samples else 0,
                sample_count=len(valid_samples)
            )
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ 통계 계산 중 오류: {e}")
            return PegPeriodStats(sample_count=0)
    
    def _calculate_compare_metrics(self, 
                                  pre_stats: PegPeriodStats, 
                                  post_stats: PegPeriodStats) -> PegCompareMetrics:
        """
        [Helper] 비교 지표 계산 (Compare Metrics)
        
        Pre 기간과 Post 기간의 통계를 비교하여 변화율(Delta), ND/Zero 존재 여부,
        트래픽 볼륨 등급 등을 계산합니다.
        
        Args:
            pre_stats: Pre 기간 통계
            post_stats: Post 기간 통계
            
        Returns:
            PegCompareMetrics: 비교 분석용 지표
        """
        try:
            # 변화율 계산 ((Post - Pre) / Pre * 100)
            delta_pct = None
            if pre_stats.mean is not None and pre_stats.mean != 0:
                delta_pct = ((post_stats.mean - pre_stats.mean) / pre_stats.mean) * 100
            
            # 플래그 설정
            has_nd = pre_stats.nd_ratio > 0 or post_stats.nd_ratio > 0
            has_zero = pre_stats.zero_ratio > 0 or post_stats.zero_ratio > 0
            
            # 트래픽 볼륨 분류 (High/Low) - β0 기준
            # FIXME: beta_0 값을 인자로 받거나 설정에서 가져와야 함 (현재 하드코딩 1000.0)
            beta_0 = 1000.0 
            traffic_class = "low"
            if (pre_stats.mean and pre_stats.mean >= beta_0) and (post_stats.mean and post_stats.mean >= beta_0):
                traffic_class = "high"
            
            return PegCompareMetrics(
                delta_pct=delta_pct,
                has_nd=has_nd,
                has_zero=has_zero,
                has_new=False,  # TODO: 'New' 상태 판별 로직 추가 필요 여부 확인
                out_of_range=False,
                traffic_volume_class=traffic_class
            )
            
        except Exception as e:
            self.logger.error(f"❌ 비교 지표 계산 중 오류: {e}")
            return PegCompareMetrics()
    
    # =============================================================================
    # 최종 KPI 결과 요약 로직 (PRD 2.3.5)
    # =============================================================================
    
    def summarize_final_kpi_results(self, 
                                  main_kpi_judgements: Dict[str, 'SimpleKPIJudgement'],
                                  sub_kpi_judgements: Dict[str, 'SimpleKPIJudgement']) -> Dict[str, 'SimpleKPIJudgement']:
        """
        [최종 요약] 전체 KPI 결과 요약 (Result Summarization)
        
        PRD 2.3.5 섹션의 규칙에 따라 각 Main KPI별로 최종 판정(OK/NOK/POK)을 요약합니다.
        
        요약 규칙:
        1. Main NOK -> NOK
        2. Main OK + any Sub NOK -> POK (Partially OK)
        3. Main OK + all Sub OK -> OK
        4. Main Can't judge -> Can't judge
        
        Args:
            main_kpi_judgements: Main KPI 판정 결과 딕셔너리
            sub_kpi_judgements: Sub KPI 판정 결과 딕셔너리
            
        Returns:
            Dict[str, SimpleKPIJudgement]: 최종 요약된 KPI 판정 결과
        """
        try:
            self.logger.info("📑 최종 KPI 결과 요약 시작")
            
            final_results = {}
            
            for main_kpi_name, main_judgement in main_kpi_judgements.items():
                
                # 해당 Main KPI의 Sub KPI들 찾기
                related_sub_kpis = {
                    sub_name: sub_judgement 
                    for sub_name, sub_judgement in sub_kpi_judgements.items()
                    if self._is_related_sub_kpi(main_kpi_name, sub_name)
                }
                
                # 최종 판정 적용
                final_judgement = self._apply_final_summary_rules(
                    main_kpi_name, main_judgement, related_sub_kpis
                )
                
                final_results[main_kpi_name] = final_judgement
                
                self.logger.debug(f"🗳️ KPI '{main_kpi_name}' 최종 판정: {final_judgement.judgement_type}")
            
            self.logger.info(f"✅ 최종 KPI 결과 요약 완료: {len(final_results)}개 KPI")
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 최종 KPI 결과 요약 중 오류: {e}")
            # 오류 시 원본 Main KPI 결과 반환 (Fail-safe)
            return main_kpi_judgements
    
    def _is_related_sub_kpi(self, main_kpi_name: str, sub_kpi_name: str) -> bool:
        """
        [Helper] Sub KPI와 Main KPI의 관련성 판단
        
        단순 이름 매칭 규칙을 사용하여 판단합니다.
        예: 'Avg' 등의 접미사를 제거한 기본 이름이 같으면 관련 KPI로 간주.
        """
        try:
            # 예: AirMacDLThruAvg와 AirMacDLThruMax가 관련
            main_base = main_kpi_name.replace("Avg", "").replace("Max", "").replace("Min", "")
            sub_base = sub_kpi_name.replace("Avg", "").replace("Max", "").replace("Min", "")
            
            # 같은 기본 이름을 가지면 관련 KPI로 판단
            is_related = main_base == sub_base and main_kpi_name != sub_kpi_name
            
            if is_related:
                # self.logger.debug(f"🔗 Sub KPI '{sub_kpi_name}'는 Main KPI '{main_kpi_name}'와 관련됨")
                pass
            
            return is_related
            
        except Exception as e:
            self.logger.error(f"❌ Sub KPI 관련성 판단 오류: {e}")
            return False
    
    def _apply_final_summary_rules(self, 
                                 main_kpi_name: str,
                                 main_judgement: 'SimpleKPIJudgement',
                                 related_sub_kpis: Dict[str, 'SimpleKPIJudgement']) -> 'SimpleKPIJudgement':
        """
        [Helper] 최종 요약 규칙 적용 로직
        
        Args:
            main_kpi_name: Main KPI 이름
            main_judgement: Main KPI 판정 결과
            related_sub_kpis: 관련 Sub KPI 판정 결과들
            
        Returns:
            SimpleKPIJudgement: 최종 요약 판정 결과
        """
        try:
            # 규칙 1: Main Can't judge -> Can't judge
            if main_judgement.judgement_type == JudgementType.CANT_JUDGE:
                return self._create_summary_judgement(
                    main_judgement,
                    "Main KPI 판정 불가로 인한 전체 판정 불가",
                    related_sub_kpis,
                    "rule_1_main_cant_judge"
                )
            
            # 규칙 2: Main NOK -> NOK
            if main_judgement.judgement_type == JudgementType.NOK:
                return self._create_summary_judgement(
                    main_judgement,
                    f"Main KPI NOK ({main_judgement.compare_detail}) → 전체 NOK",
                    related_sub_kpis,
                    "rule_2_main_nok"
                )
            
            # 규칙 3 & 4: Main OK인 경우 Sub KPI 검토
            if main_judgement.judgement_type == JudgementType.OK:
                return self._evaluate_main_ok_with_subs(
                    main_kpi_name, main_judgement, related_sub_kpis
                )
            
            # 예상하지 못한 경우 (방어적 프로그래밍)
            self.logger.warning(f"⚠️ 예상하지 못한 Main KPI 판정 타입: {main_judgement.judgement_type}")
            return main_judgement
            
        except Exception as e:
            self.logger.error(f"❌ 최종 요약 규칙 적용 오류: {e}")
            return main_judgement
    
    def _evaluate_main_ok_with_subs(self, 
                                  main_kpi_name: str,
                                  main_judgement: 'SimpleKPIJudgement',
                                  related_sub_kpis: Dict[str, 'SimpleKPIJudgement']) -> 'SimpleKPIJudgement':
        """
        [Helper] Main OK일 때의 Sub KPI 평가 로직
        
        Main KPI가 정상(OK)이라도 Sub KPI에 이상이 있으면 POK로 격하될 수 있습니다.
        """
        try:
            if not related_sub_kpis:
                # Sub KPI가 없으면 Main OK 그대로 유지
                return self._create_summary_judgement(
                    main_judgement,
                    "Main KPI OK, Sub KPI 없음 → OK",
                    related_sub_kpis,
                    "rule_3_main_ok_no_subs"
                )
            
            # Sub KPI들의 판정 상태 분석
            sub_analysis = self._analyze_sub_kpi_results(related_sub_kpis)
            
            # 규칙 4: Main OK + any Sub NOK -> POK (Partially OK)
            if sub_analysis["has_nok"]:
                from app.models.judgement import SimpleKPIJudgement
                pok_judgement = SimpleKPIJudgement(
                    judgement_type=JudgementType.POK,  # Partially OK
                    compare_detail=CompareDetail.PARTIALLY_OK,
                    reasoning=f"Main KPI OK이나 Sub KPI 중 NOK 존재 → POK",
                    confidence=min(main_judgement.confidence, sub_analysis["min_confidence"]),
                    metrics=main_judgement.metrics,
                    thresholds_used=main_judgement.thresholds_used
                )
                
                return self._create_summary_judgement(
                    pok_judgement,
                    f"Main OK + Sub NOK({sub_analysis['nok_count']}개) → POK",
                    related_sub_kpis,
                    "rule_4_main_ok_sub_nok"
                )
            
            # 규칙 3: Main OK + all Sub OK -> OK
            return self._create_summary_judgement(
                main_judgement,
                f"Main KPI OK + 모든 Sub KPI OK({sub_analysis['ok_count']}개) → OK",
                related_sub_kpis,
                "rule_3_main_ok_all_sub_ok"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Main OK Sub 평가 오류: {e}")
            return main_judgement
    
    def _analyze_sub_kpi_results(self, related_sub_kpis: Dict[str, 'SimpleKPIJudgement']) -> Dict[str, Any]:
        """
        [Helper] Sub KPI 결과 집합 분석
        
        OK/NOK/POK/Can't Judge 개수 및 최소 신뢰도 등을 계산합니다.
        """
        try:
            analysis = {
                "total_count": len(related_sub_kpis),
                "ok_count": 0,
                "nok_count": 0,
                "pok_count": 0,
                "cant_judge_count": 0,
                "has_nok": False,
                "has_cant_judge": False,
                "min_confidence": 1.0,
                "nok_details": []
            }
            
            for sub_name, sub_judgement in related_sub_kpis.items():
                analysis["min_confidence"] = min(analysis["min_confidence"], sub_judgement.confidence)
                
                if sub_judgement.judgement_type == JudgementType.OK:
                    analysis["ok_count"] += 1
                elif sub_judgement.judgement_type == JudgementType.NOK:
                    analysis["nok_count"] += 1
                    analysis["has_nok"] = True
                    analysis["nok_details"].append(f"{sub_name}({sub_judgement.compare_detail})")
                elif sub_judgement.judgement_type == JudgementType.POK:
                    analysis["pok_count"] += 1
                    analysis["has_nok"] = True  # POK도 NOK로 취급하여 상위 POK 유발
                    analysis["nok_details"].append(f"{sub_name}(POK)")
                elif sub_judgement.judgement_type == JudgementType.CANT_JUDGE:
                    analysis["cant_judge_count"] += 1
                    analysis["has_cant_judge"] = True
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Sub KPI 결과 분석 오류: {e}")
            return {"total_count": 0, "has_nok": False, "has_cant_judge": False, "min_confidence": 0.5}
    
    def _create_summary_judgement(self, 
                                base_judgement: 'SimpleKPIJudgement',
                                summary_reasoning: str,
                                related_sub_kpis: Dict[str, 'SimpleKPIJudgement'],
                                rule_applied: str) -> 'SimpleKPIJudgement':
        """[Helper] 요약된 판정 결과 객체 생성"""
        try:
            # 기존 메트릭스에 요약 정보 추가
            enhanced_metrics = {
                **base_judgement.metrics,
                "summary_rule_applied": rule_applied,
                "sub_kpi_count": len(related_sub_kpis),
                "sub_kpi_names": list(related_sub_kpis.keys()),
                "original_reasoning": base_judgement.reasoning
            }
            
            from app.models.judgement import SimpleKPIJudgement
            return SimpleKPIJudgement(
                judgement_type=base_judgement.judgement_type,
                compare_detail=base_judgement.compare_detail,
                reasoning=summary_reasoning,
                confidence=base_judgement.confidence,
                metrics=enhanced_metrics,
                thresholds_used=base_judgement.thresholds_used
            )
            
        except Exception as e:
            self.logger.error(f"요약 판정 생성 오류: {e}")
            return base_judgement

# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("✅ Choi Judgement Service 로드 완료")
