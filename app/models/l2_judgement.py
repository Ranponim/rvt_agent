"""
L2 알고리즘 (Statistical Analysis) 데이터 모델

이 모듈은 L2 심층 분석결과를 담기 위한 Pydantic 모델들을 정의합니다.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.models.judgement import PegSampleSeries

class L2Severity(str, Enum):
    """L2 분석 심각도"""
    CRITICAL = "Critical"   # 통계적으로 매우 유의미한 악화 (P-value 매우 낮음 + 분포 큼)
    WARNING = "Warning"     # 유의미한 변화 감지됨
    INFO = "Info"           # 변화는 있으나 미미함
    NORMAL = "Normal"       # 변화 없음 (정상)

class StatisticalTestType(str, Enum):
    """통계 검정 종류"""
    MANN_WHITNEY = "Mann-Whitney U Test"
    KS_TEST = "Kolmogorov-Smirnov Test"
    MAHALANOBIS = "Mahalanobis Distance"
    CHANGE_RATE = "Change Rate"

class StatisticalTestResult(BaseModel):
    """개별 통계 검정 결과"""
    test_type: StatisticalTestType = Field(..., description="검정 종류")
    statistic: float = Field(..., description="검정 통계량 (U-stat, D-stat, Distance 등)")
    p_value: Optional[float] = Field(None, description="P-value (해당되는 경우)")
    is_significant: bool = Field(..., description="유의수준(0.05) 내 유의미성 여부")
    details: Dict[str, Any] = Field(default_factory=dict, description="추가 분석 정보")

class AnalysisResultL2(BaseModel):
    """
    단일 KPI에 대한 L2 심층 분석 결과
    """
    kpi_name: str = Field(..., description="KPI 이름")
    cell_id: str = Field(..., description="셀 ID")
    
    # 종합 판정
    severity: L2Severity = Field(..., description="L2 종합 심각도")
    summary: str = Field(..., description="분석 요약 텍스트")
    
    # 세부 검정 결과
    test_results: List[StatisticalTestResult] = Field(..., description="수행된 통계 검정 결과 리스트")
    
    # 분포 특성 (Metadata)
    pre_samples_count: int
    post_samples_count: int
    distribution_shift_score: float = Field(..., description="분포 변화 점수 (0.0 ~ 1.0)")

class L2AnalysisResponse(BaseModel):
    """
    L2 분석 전체 응답 (API 반환용)
    """
    timestamp: str
    analyzed_cells_count: int
    results: Dict[str, List[AnalysisResultL2]] = Field(..., description="Cell ID별 분석 결과 리스트")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
