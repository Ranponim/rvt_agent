"""
L2 분석 서비스 구현

이 모듈은 L2 알고리즘(통계적 분석)의 전체 파이프라인을 실행합니다.
Choi 알고리즘이 1차 필터링을 한 후, 심층 분석이 필요한 경우(또는 전체 검사용) 이 서비스를 호출합니다.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

from app.models.judgement import PegSampleSeries
from app.models.l2_judgement import (
    AnalysisResultL2,
    L2AnalysisResponse,
    L2Severity,
    StatisticalTestResult,
    StatisticalTestType
)
from app.services.statistical_analyzers import (
    MannWhitneyAnalyzer,
    KSTestAnalyzer,
    ChangeRateAnalyzer
)

logger = logging.getLogger(__name__)

class L2Service:
    """L2 (Statistical) 분석 서비스"""
    
    def __init__(self):
        self.logger = logger
        # 분석기 인스턴스 초기화
        self.mw_analyzer = MannWhitneyAnalyzer()
        self.ks_analyzer = KSTestAnalyzer()
        self.rate_analyzer = ChangeRateAnalyzer()
        
    def analyze(self, input_data: Dict[str, List[PegSampleSeries]]) -> L2AnalysisResponse:
        """
        L2 전체 분석 실행 (Cell 단위)
        
        Args:
            input_data: {cell_id: [PegSampleSeries, ...]}
            
        Returns:
            L2AnalysisResponse: 분석 결과
        """
        start_time = datetime.now()
        results_by_cell = {}
        total_cells = 0
        
        try:
            for cell_id, peg_series_list in input_data.items():
                total_cells += 1
                cell_results = []
                
                for series in peg_series_list:
                    # 각 PEG에 대해 L2 분석 수행
                    result = self._analyze_single_peg(series, cell_id)
                    if result:
                        cell_results.append(result)
                
                results_by_cell[cell_id] = cell_results
            
            return L2AnalysisResponse(
                timestamp=datetime.now().isoformat(),
                analyzed_cells_count=total_cells,
                results=results_by_cell,
                metadata={
                    "algorithm": "L2_Statistical",
                    "analyzers": ["Mann-Whitney", "KS-Test", "ChangeRate"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"L2 Analysis Failed: {e}", exc_info=True)
            # 실패 시 부분 결과라도 반환
            return L2AnalysisResponse(
                timestamp=datetime.now().isoformat(),
                analyzed_cells_count=total_cells,
                results=results_by_cell,
                metadata={"error": str(e)}
            )

    def _analyze_single_peg(self, series: PegSampleSeries, cell_id: str) -> AnalysisResultL2:
        """단일 PEG 데이터에 대한 통계 분석 수행"""
        
        # None 제거 및 float 변환
        pre = [float(x) for x in series.pre_samples if x is not None]
        post = [float(x) for x in series.post_samples if x is not None]
        
        test_results: List[StatisticalTestResult] = []
        
        # 1. Mann-Whitney Test (중앙값 차이)
        mw_res = self.mw_analyzer.analyze(pre, post)
        if mw_res:
            test_results.append(mw_res)
            
        # 2. KS Test (분포 차이)
        ks_res = self.ks_analyzer.analyze(pre, post)
        if ks_res:
            test_results.append(ks_res)
            
        # 3. Change Rate (단순 변화율)
        rate_res = self.rate_analyzer.analyze(pre, post)
        if rate_res:
            test_results.append(rate_res)
            
        # 종합 판정 (심각도 결정)
        severity, score, summary = self._judge_severity(test_results)
        
        return AnalysisResultL2(
            kpi_name=series.peg_name,
            cell_id=cell_id,
            severity=severity,
            summary=summary,
            test_results=test_results,
            pre_samples_count=len(pre),
            post_samples_count=len(post),
            distribution_shift_score=score
        )

    def _judge_severity(self, test_results: List[StatisticalTestResult]):
        """검정 결과들을 종합하여 심각도 판정"""
        
        # 각 검정의 유의성 확인
        is_mw_sig = any(r.is_significant for r in test_results if r.test_type == StatisticalTestType.MANN_WHITNEY)
        is_ks_sig = any(r.is_significant for r in test_results if r.test_type == StatisticalTestType.KS_TEST)
        is_rate_sig = any(r.is_significant for r in test_results if r.test_type == StatisticalTestType.CHANGE_RATE)
        
        # 로직: 분포 변화(KS)가 있고, 중앙값 이동(MW)이나 변화율도 크면 Critical
        if is_ks_sig and (is_mw_sig or is_rate_sig):
            return L2Severity.CRITICAL, 0.9, "분포와 평균 모두 유의미한 변화 (Shifted & Deformed)"
            
        elif is_mw_sig or is_rate_sig:
             return L2Severity.WARNING, 0.6, "평균/중앙값 이동 감지됨 (Shifted)"
             
        elif is_ks_sig:
             return L2Severity.WARNING, 0.5, "분포 형태만 변화됨 (Deformed)"
             
        elif any(r.p_value is not None and r.p_value < 0.1 for r in test_results):
             return L2Severity.INFO, 0.2, "미세한 변화 징후 (Weak Signal)"
        
        return L2Severity.NORMAL, 0.0, "유의미한 변화 없음"
