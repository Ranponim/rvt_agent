"""
L2 통계 분석기 구현

이 모듈은 Scipy 및 Numpy를 사용하여 다음 통계 검정을 수행하는 클래스들을 제공합니다.
1. Mann-Whitney U Test: 중앙값(Median) 이동 검정
2. Kolmogorov-Smirnov (KS) Test: 분포(Distribution) 형태 변화 검정
3. Change Rate Check: 평균 변화율 단순 계산
"""

import numpy as np
from scipy import stats
from typing import List, Optional, Dict, Any
import logging

from app.models.l2_judgement import (
    StatisticalTestResult, 
    StatisticalTestType
)

logger = logging.getLogger(__name__)

class BaseStatisticalAnalyzer:
    """통계 분석기 기본 클래스"""
    def analyze(self, pre: List[float], post: List[float]) -> Optional[StatisticalTestResult]:
        raise NotImplementedError

class MannWhitneyAnalyzer(BaseStatisticalAnalyzer):
    """Mann-Whitney U Test: 두 집단의 중앙값 차이 검정"""
    
    def analyze(self, pre: List[float], post: List[float]) -> Optional[StatisticalTestResult]:
        try:
            if len(pre) < 3 or len(post) < 3: # 최소 샘플 수
                return None
                
            # Mann-Whitney U test (양측 검정)
            stat, p_value = stats.mannwhitneyu(pre, post, alternative='two-sided')
            
            # P-value < 0.05 이면 "유의미한 차이 있음"
            is_significant = p_value < 0.05
            
            return StatisticalTestResult(
                test_type=StatisticalTestType.MANN_WHITNEY,
                statistic=float(stat),
                p_value=float(p_value),
                is_significant=is_significant,
                details={
                    "pre_median": float(np.median(pre)),
                    "post_median": float(np.median(post))
                }
            )
        except Exception as e:
            logger.error(f"MW Test Failed: {e}")
            return None

class KSTestAnalyzer(BaseStatisticalAnalyzer):
    """Kolmogorov-Smirnov Test: 두 집단의 분포 형태 차이 검정"""
    
    def analyze(self, pre: List[float], post: List[float]) -> Optional[StatisticalTestResult]:
        try:
            if len(pre) < 3 or len(post) < 3:
                return None
                
            # KS test (두 표본 비교)
            stat, p_value = stats.ks_2samp(pre, post)
            
            is_significant = p_value < 0.05
            
            return StatisticalTestResult(
                test_type=StatisticalTestType.KS_TEST,
                statistic=float(stat),
                p_value=float(p_value),
                is_significant=is_significant
            )
        except Exception as e:
            logger.error(f"KS Test Failed: {e}")
            return None

class ChangeRateAnalyzer(BaseStatisticalAnalyzer):
    """단순 변화율 분석 (보조용)"""
    
    def analyze(self, pre: List[float], post: List[float]) -> Optional[StatisticalTestResult]:
        try:
            if not pre or not post:
                return None
                
            pre_mean = float(np.mean(pre))
            post_mean = float(np.mean(post))
            
            if pre_mean == 0:
                return None # 0으로 나눌 수 없음
                
            delta_pct = ((post_mean - pre_mean) / pre_mean) * 100.0
            
            # 변화율 자체는 '유의성' 기준이 모호하지만, 10% 이상이면 유의하다고 마킹 (예시)
            is_significant = abs(delta_pct) >= 10.0
            
            return StatisticalTestResult(
                test_type=StatisticalTestType.CHANGE_RATE,
                statistic=delta_pct,
                p_value=None, # P-value 없음
                is_significant=is_significant,
                details={
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "delta_pct": delta_pct
                }
            )
        except Exception as e:
            logger.error(f"Change Rate Analysis Failed: {e}")
            return None
