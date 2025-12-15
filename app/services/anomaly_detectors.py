"""
이상 탐지 규칙 구현 (4장)

이 모듈은 SOLID 원칙을 완벽히 준수하여 각 이상 탐지 규칙을
독립적인 클래스로 구현합니다.

SOLID 원칙 적용:
- Single Responsibility: 각 탐지기는 하나의 규칙만 담당
- Open/Closed: 새로운 탐지 규칙 추가 시 기존 코드 수정 없음
- Liskov Substitution: 모든 탐지기는 동일한 인터페이스 구현
- Interface Segregation: 각 탐지기는 필요한 메서드만 구현
- Dependency Inversion: 추상화에 의존, 구체 클래스에 의존하지 않음

PRD 참조: 섹션 2.2 (이상 탐지 알고리즘)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, Set
from dataclasses import dataclass, field
import logging
import numpy as np

from ..models.judgement import PegSampleSeries
from ..utils.logging_decorators import log_detector_execution
from ..exceptions import AbnormalDetectionError, DIMSDataError, ConfigurationError

logger = logging.getLogger(__name__)


# =============================================================================
# 이상 탐지 결과 데이터 클래스
# =============================================================================

@dataclass(frozen=True)
class AnomalyDetectionResult:
    """
    이상 탐지 결과 (불변 객체)
    
    각 탐지기의 결과를 표현하는 불변 데이터 클래스입니다.
    """
    anomaly_type: str
    affected_cells: Set[str]
    affected_pegs: Set[str]
    details: Dict[str, Any]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """후처리 검증: 신뢰도는 0.0 ~ 1.0 사이여야 함"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


# =============================================================================
# 이상 탐지기 인터페이스 (Protocol 사용 - 더 유연한 타입 힌팅)
# =============================================================================

class AnomalyDetector(Protocol):
    """
    이상 탐지기 프로토콜 (인터페이스)
    
    Protocol을 사용하여 덕 타이핑 지원 및 더 유연한 인터페이스를 제공합니다.
    모든 이상 탐지기는 이 인터페이스를 따라야 합니다.
    """
    
    def detect(self, 
               peg_data: Dict[str, List[PegSampleSeries]], 
               config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        이상 탐지 실행
        
        Args:
            peg_data: PEG 데이터 (Cell ID -> PEG 시리즈 리스트)
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: 탐지 결과
        """
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """설정 유효성 검증"""
        ...
    
    def get_detector_info(self) -> Dict[str, Any]:
        """탐지기 메타 정보 반환"""
        ...


# =============================================================================
# 추상 기본 탐지기 클래스
# =============================================================================

class BaseAnomalyDetector(ABC):
    """
    이상 탐지기 기본 추상 클래스
    
    공통 기능과 템플릿 메서드 패턴을 제공하여 코드 중복을 줄이고 일관성을 유지합니다.
    모든 구체적인 이상 탐지기는 이 클래스를 상속받아야 합니다.
    """
    
    def __init__(self, detector_name: str, version: str = "1.0.0"):
        """
        기본 탐지기 초기화
        
        Args:
            detector_name: 탐지기 이름 (Log 식별용)
            version: 탐지기 버전
        """
        self.detector_name = detector_name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"🛠️ 이상 탐지기 '{detector_name}' v{version} 초기화됨")
    
    @log_detector_execution()
    def detect(self, 
               peg_data: Dict[str, List[PegSampleSeries]], 
               config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Template Method] 이상 탐지 실행 흐름 제어
        
        1. 설정 검증
        2. 입력 데이터 검증
        3. 실제 탐지 로직 실행 (추상 메서드 호출)
        4. 결과 검증
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: 탐지 결과
        """
        try:
            # 1. 설정 검증
            if not self.validate_config(config):
                self.logger.error(f"❌ {self.detector_name}: 설정 검증 실패")
                raise ValueError(f"Invalid config for {self.detector_name}")
            
            # 2. 입력 데이터 검증
            if not self._validate_input_data(peg_data):
                self.logger.error(f"❌ {self.detector_name}: 입력 데이터 검증 실패")
                raise ValueError(f"Invalid input data for {self.detector_name}")
            
            # 3. 실제 탐지 로직 실행 (하위 클래스에서 구현)
            self.logger.debug(f"🚀 {self.detector_name} 탐지 시작...")
            detection_result = self._execute_detection(peg_data, config)
            
            # 4. 결과 검증
            if not self._validate_result(detection_result):
                self.logger.error(f"❌ {self.detector_name}: 잘못된 탐지 결과 반환됨")
                raise RuntimeError(f"Invalid detection result from {self.detector_name}")
            
            self.logger.info(f"✅ {self.detector_name} 탐지 완료: "
                           f"대상 Cell={len(peg_data)}, "
                           f"영향받은 Cell={len(detection_result.affected_cells)}, "
                           f"영향받은 PEG={len(detection_result.affected_pegs)}")
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"❌ {self.detector_name} 탐지 중 치명적 오류: {e}", exc_info=True)
            raise
    
    @abstractmethod
    def _execute_detection(self, 
                          peg_data: Dict[str, List[PegSampleSeries]], 
                          config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Abstract] 실제 탐지 로직 구현 (하위 클래스에서 반드시 구현)
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: 탐지 결과
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        기본 설정 검증
        
        Args:
            config: 검증할 설정
            
        Returns:
            bool: 유효성 여부
        """
        try:
            if not isinstance(config, dict):
                self.logger.error("설정(Config)은 딕셔너리 형태여야 합니다.")
                return False
            
            # 하위 클래스에서 추가 검증 수행
            return self._validate_specific_config(config)
            
        except Exception as e:
            self.logger.error(f"설정 검증 중 오류: {e}")
            return False
    
    @abstractmethod
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """[Abstract] 하위 클래스별 특화 설정 검증"""
        pass
    
    def _validate_input_data(self, peg_data: Dict[str, List[PegSampleSeries]]) -> bool:
        """
        입력 데이터 유효성 검증
        
        Args:
            peg_data: 검증할 PEG 데이터
            
        Returns:
            bool: 유효성 여부
        """
        try:
            if not peg_data:
                self.logger.warning("제공된 PEG 데이터가 비어 있습니다.")
                # 빈 데이터도 처리 가능한 경우가 있으므로 에러는 아님, 하지만 경고 로그
                return True
            
            for cell_id, peg_series_list in peg_data.items():
                if not cell_id:
                    self.logger.error("빈 Cell ID가 발견되었습니다.")
                    return False
                
                if peg_series_list is None:
                     self.logger.error(f"Cell {cell_id}의 PEG 시리즈 리스트가 None입니다.")
                     return False
                     
                if not peg_series_list:
                    self.logger.debug(f"ℹ️ Cell {cell_id}에 대한 PEG 시리즈가 없습니다.")
                    continue
                
                for series in peg_series_list:
                    if not isinstance(series, PegSampleSeries):
                        self.logger.error(f"Cell {cell_id}에 잘못된 타입의 PEG 시리즈가 포함됨: {type(series)}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"입력 데이터 검증 중 오류: {e}")
            return False
    
    def _validate_result(self, result: AnomalyDetectionResult) -> bool:
        """
        탐지 결과 유효성 검증
        
        Args:
            result: 검증할 결과
            
        Returns:
            bool: 유효성 여부
        """
        try:
            if not isinstance(result, AnomalyDetectionResult):
                self.logger.error(f"반환값이 AnomalyDetectionResult 인스턴스가 아닙니다. (실제 타입: {type(result)})")
                return False
            
            if not result.anomaly_type:
                self.logger.error("Anomaly Type이 비어있습니다.")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"결과 검증 중 오류: {e}")
            return False
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        탐지기 메타 정보 반환
        
        Returns:
            Dict[str, Any]: 탐지기 이름, 버전, 설명 등
        """
        return {
            "name": self.detector_name,
            "version": self.version,
            "type": "anomaly_detector",
            "description": f"Anomaly detector implementation: {self.detector_name}"
        }


# =============================================================================
# Range 이상 탐지기 (DIMS 의존성)
# =============================================================================

class RangeAnomalyDetector(BaseAnomalyDetector):
    """
    Range 이상 탐지기
    
    DIMS 문서의 [min, max] 범위를 벗어나는 통계를 탐지합니다.
    DIMS 제공자로부터 각 PEG에 대한 범위 정보를 받아와 실제 값과 비교합니다.
    
    Single Responsibility: Range 검사만 담당
    Dependency Inversion: DimsDataProvider 추상화에 의존
    """
    
    def __init__(self, dims_provider: Optional['DimsDataProvider'] = None):
        """
        Range 탐지기 초기화
        
        Args:
            dims_provider: DIMS 데이터 제공자 (의존성 주입)
        """
        super().__init__("RangeAnomalyDetector")
        self.dims_provider = dims_provider or MockDimsDataProvider()
        
        self.logger.info(f"📏 Range 탐지기 초기화됨 (Data Provider: {type(self.dims_provider).__name__})")
    
    def _execute_detection(self, 
                          peg_data: Dict[str, List[PegSampleSeries]], 
                          config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Analysis] Range 이상 탐지 실행
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: Range 탐지 결과
        """
        try:
            # Range 검사 활성화 여부 확인
            enable_range_check = config.get("enable_range_check", True)
            
            if not enable_range_check:
                self.logger.info("ℹ️ 설정에 의해 Range 이상 탐지가 비활성화되었습니다.")
                return AnomalyDetectionResult(
                    anomaly_type="Range",
                    affected_cells=set(),
                    affected_pegs=set(),
                    details={},
                    confidence=1.0,
                    metadata={
                        "detection_disabled": True,
                        "reason": "Range check disabled in configuration"
                    }
                )
            
            affected_cells = set()
            affected_pegs = set()
            details = {}
            dims_unavailable_count = 0
            
            for cell_id, peg_series_list in peg_data.items():
                for series in peg_series_list:
                    # DIMS에서 Range 정보 조회 (견고한 오류 처리)
                    try:
                        range_info = self.dims_provider.get_peg_range(series.peg_name)
                        
                        if not range_info:
                            dims_unavailable_count += 1
                            self.logger.debug(f"⚠️ {series.peg_name}의 Range 정보 없음, 건너뜀")
                            continue
                        
                        min_value, max_value = range_info["min"], range_info["max"]
                        
                        # Pre/Post 값들이 범위를 벗어나는지 검사
                        violations = self._check_range_violations(series, min_value, max_value)
                        
                    except Exception as e:
                        dims_unavailable_count += 1
                        self.logger.warning(f"⚠️ {series.peg_name}의 DIMS 데이터 접근 실패: {e}")
                        # 에러가 발생해도 계속 진행 (견고한 처리)
                        continue
                    
                    if violations:
                        affected_cells.add(cell_id)
                        affected_pegs.add(series.peg_name)
                        details[f"{cell_id}.{series.peg_name}"] = violations
                        
                        self.logger.debug(f"🚨 Range 위반 감지: {cell_id}.{series.peg_name} "
                                        f"(기준: [{min_value}, {max_value}])")
            
            # DIMS 데이터 가용성에 따른 신뢰도 조정
            total_pegs_checked = sum(len(peg_list) for peg_list in peg_data.values())
            dims_availability_ratio = 1.0 - (dims_unavailable_count / total_pegs_checked) if total_pegs_checked > 0 else 1.0
            
            # 메타데이터에 DIMS 가용성 정보 포함
            metadata = {
                "dims_unavailable_count": dims_unavailable_count,
                "total_pegs_checked": total_pegs_checked,
                "dims_availability_ratio": dims_availability_ratio,
                "detection_enabled": True
            }
            
            if dims_unavailable_count > 0:
                self.logger.warning(f"⚠️ Range 탐지 완료 (DIMS 데이터 부족: {dims_unavailable_count}/{total_pegs_checked}, "
                                  f"가용률: {dims_availability_ratio:.1%})")
            
            return AnomalyDetectionResult(
                anomaly_type="Range",
                affected_cells=affected_cells,
                affected_pegs=affected_pegs,
                details=details,
                confidence=max(0.5, 0.9 * dims_availability_ratio),  # DIMS 가용성에 따른 신뢰도
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Range 탐지 실행 중 오류: {e}")
            raise
    
    def _check_range_violations(self, 
                                series: PegSampleSeries, 
                                min_value: float, 
                                max_value: float) -> Dict[str, Any]:
        """
        [Helper] 개별 시리즈의 범위 위반 검사
        
        Pre와 Post 기간의 모든 샘플을 검사하여 범위를 벗어나는 샘플을 찾습니다.
        """
        try:
            violations = {}
            
            # Pre 기간 검사
            pre_violations = []
            for i, sample in enumerate(series.pre_samples):
                if sample is not None and not (min_value <= sample <= max_value):
                    pre_violations.append({"index": i, "value": sample, "period": "pre"})
            
            # Post 기간 검사
            post_violations = []
            for i, sample in enumerate(series.post_samples):
                if sample is not None and not (min_value <= sample <= max_value):
                    post_violations.append({"index": i, "value": sample, "period": "post"})
            
            if pre_violations or post_violations:
                violations = {
                    "range": {"min": min_value, "max": max_value},
                    "pre_violations": pre_violations,
                    "post_violations": post_violations,
                    "total_violations": len(pre_violations) + len(post_violations)
                }
            
            return violations
            
        except Exception as e:
            self.logger.error(f"❌ Range 위반 검사 중 오류: {e}")
            return {}
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """[Validation] Range 탐지기 특화 설정 검증"""
        # Range 탐지기는 필수로 요구되는 추가 설정이 없음
        return True


# =============================================================================
# ND (No Data) 이상 탐지기
# =============================================================================

class NDAnomalyDetector(BaseAnomalyDetector):
    """
    ND (No Data) 이상 탐지기
    
    pre/post 중 한쪽만 ND인 경우를 탐지합니다.
    - Pre만 ND: Post에 데이터가 생김 (이슈 아닐 수 있음, 상황에 따라 다름)
    - Post만 ND: Pre에는 있었는데 Post에 데이터가 없음 (심각한 이슈 가능성)
    
    Single Responsibility: ND 검사만 담당
    """
    
    def __init__(self):
        """ND 탐지기 초기화"""
        super().__init__("NDAnomalyDetector")
    
    def _execute_detection(self, 
                          peg_data: Dict[str, List[PegSampleSeries]], 
                          config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Analysis] ND 이상 탐지 실행
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: ND 탐지 결과
        """
        try:
            affected_cells = set()
            affected_pegs = set()
            details = {}
            
            for cell_id, peg_series_list in peg_data.items():
                for series in peg_series_list:
                    nd_info = self._analyze_nd_pattern(series)
                    
                    if nd_info["has_one_sided_nd"]:
                        affected_cells.add(cell_id)
                        affected_pegs.add(series.peg_name)
                        details[f"{cell_id}.{series.peg_name}"] = nd_info
                        
                        pattern_desc = "Pre Only" if nd_info["nd_pattern"] == "pre_only" else "Post Only"
                        
                        self.logger.debug(f"🚨 ND 이상 감지: {cell_id}.{series.peg_name} "
                                        f"({pattern_desc}, "
                                        f"Pre ND: {nd_info['pre_nd_ratio']:.1%}, "
                                        f"Post ND: {nd_info['post_nd_ratio']:.1%})")
            
            return AnomalyDetectionResult(
                anomaly_type="ND",
                affected_cells=affected_cells,
                affected_pegs=affected_pegs,
                details=details,
                confidence=1.0,  # ND 검사는 확실함
                metadata={"detection_enabled": True}
            )
            
        except Exception as e:
            self.logger.error(f"❌ ND 탐지 실행 중 오류: {e}")
            raise
    
    def _analyze_nd_pattern(self, series: PegSampleSeries) -> Dict[str, Any]:
        """
        [Helper] 시리즈의 ND 패턴 분석
        
        Args:
            series: PEG 시리즈
            
        Returns:
            Dict[str, Any]: ND 분석 결과
        """
        try:
            # Pre 기간 ND 분석
            pre_nd_count = sum(1 for sample in series.pre_samples if sample is None)
            pre_total = len(series.pre_samples)
            pre_nd_ratio = pre_nd_count / pre_total if pre_total > 0 else 0
            
            # Post 기간 ND 분석
            post_nd_count = sum(1 for sample in series.post_samples if sample is None)
            post_total = len(series.post_samples)
            post_nd_ratio = post_nd_count / post_total if post_total > 0 else 0
            
            # 한쪽만 ND인지 확인 (4장 규칙)
            pre_has_nd = pre_nd_ratio > 0
            post_has_nd = post_nd_ratio > 0
            has_one_sided_nd = (pre_has_nd and not post_has_nd) or (not pre_has_nd and post_has_nd)
            
            return {
                "pre_nd_count": pre_nd_count,
                "pre_nd_ratio": pre_nd_ratio,
                "post_nd_count": post_nd_count,
                "post_nd_ratio": post_nd_ratio,
                "has_one_sided_nd": has_one_sided_nd,
                "nd_pattern": "pre_only" if (pre_has_nd and not post_has_nd) else 
                            "post_only" if (not pre_has_nd and post_has_nd) else "none"
            }
            
        except Exception as e:
            self.logger.error(f"❌ ND 패턴 분석 중 오류: {e}")
            return {"has_one_sided_nd": False}
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """[Validation] ND 탐지기 특화 설정 검증"""
        # ND 탐지기는 추가 설정 불필요
        return True


# =============================================================================
# Zero 값 이상 탐지기
# =============================================================================

class ZeroAnomalyDetector(BaseAnomalyDetector):
    """
    Zero 값 이상 탐지기
    
    pre/post 중 한쪽만 0(Zero)인 경우를 탐지합니다.
    - Zero Tolerance를 사용하여 부동소수점 0 비교 문제를 해결합니다.
    - Pre가 0인데 Post가 0이 아님 -> 트래픽 발생 시작?
    - Pre가 0이 아닌데 Post가 0임 -> 트래픽 소멸 (장애 가능성)
    
    Single Responsibility: Zero 검사만 담당
    """
    
    def __init__(self, zero_tolerance: float = 1e-10):
        """
        Zero 탐지기 초기화
        
        Args:
            zero_tolerance: 0으로 간주할 허용 오차 (부동소수점 정밀도 대응)
        """
        super().__init__("ZeroAnomalyDetector")
        self.zero_tolerance = zero_tolerance
        
        self.logger.info(f"0️⃣ Zero 탐지기 초기화됨 (Tolerance: {zero_tolerance})")
    
    def _execute_detection(self, 
                          peg_data: Dict[str, List[PegSampleSeries]], 
                          config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Analysis] Zero 이상 탐지 실행
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: Zero 탐지 결과
        """
        try:
            affected_cells = set()
            affected_pegs = set()
            details = {}
            
            for cell_id, peg_series_list in peg_data.items():
                for series in peg_series_list:
                    zero_info = self._analyze_zero_pattern(series)
                    
                    if zero_info["has_one_sided_zero"]:
                        affected_cells.add(cell_id)
                        affected_pegs.add(series.peg_name)
                        details[f"{cell_id}.{series.peg_name}"] = zero_info
                        
                        pattern_desc = "Pre Only" if zero_info["zero_pattern"] == "pre_only" else "Post Only"
                        
                        self.logger.debug(f"🚨 Zero 이상 감지: {cell_id}.{series.peg_name} "
                                        f"({pattern_desc}, "
                                        f"Pre Zero: {zero_info['pre_zero_ratio']:.1%}, "
                                        f"Post Zero: {zero_info['post_zero_ratio']:.1%})")
            
            return AnomalyDetectionResult(
                anomaly_type="Zero",
                affected_cells=affected_cells,
                affected_pegs=affected_pegs,
                details=details,
                confidence=1.0,  # Zero 검사는 확실함
                metadata={"detection_enabled": True}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Zero 탐지 실행 중 오류: {e}")
            raise
    
    def _analyze_zero_pattern(self, series: PegSampleSeries) -> Dict[str, Any]:
        """
        [Helper] 시리즈의 Zero 패턴 분석
        
        Args:
            series: PEG 시리즈
            
        Returns:
            Dict[str, Any]: Zero 분석 결과
        """
        try:
            # Pre 기간 Zero 분석 (부동소수점 허용 오차 적용)
            pre_zero_count = sum(1 for sample in series.pre_samples 
                               if sample is not None and abs(sample) <= self.zero_tolerance)
            pre_valid_count = sum(1 for sample in series.pre_samples if sample is not None)
            pre_zero_ratio = pre_zero_count / pre_valid_count if pre_valid_count > 0 else 0
            
            # Post 기간 Zero 분석
            post_zero_count = sum(1 for sample in series.post_samples 
                                if sample is not None and abs(sample) <= self.zero_tolerance)
            post_valid_count = sum(1 for sample in series.post_samples if sample is not None)
            post_zero_ratio = post_zero_count / post_valid_count if post_valid_count > 0 else 0
            
            # 한쪽만 Zero인지 확인 (4장 규칙)
            pre_has_zero = pre_zero_ratio > 0
            post_has_zero = post_zero_ratio > 0
            has_one_sided_zero = (pre_has_zero and not post_has_zero) or (not pre_has_zero and post_has_zero)
            
            return {
                "pre_zero_count": pre_zero_count,
                "pre_zero_ratio": pre_zero_ratio,
                "post_zero_count": post_zero_count,
                "post_zero_ratio": post_zero_ratio,
                "has_one_sided_zero": has_one_sided_zero,
                "zero_pattern": "pre_only" if (pre_has_zero and not post_has_zero) else 
                              "post_only" if (not pre_has_zero and post_has_zero) else "none",
                "zero_tolerance": self.zero_tolerance
            }
            
        except Exception as e:
            self.logger.error(f"❌ Zero 패턴 분석 중 오류: {e}")
            return {"has_one_sided_zero": False}
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """[Validation] Zero 탐지기 특화 설정 검증"""
        # Zero 탐지기는 추가 설정 불필요
        return True


# =============================================================================
# New Statistics 이상 탐지기
# =============================================================================

class NewStatisticsDetector(BaseAnomalyDetector):
    """
    신규 통계 이상 탐지기
    
    이전 PKG 버전에는 없었지만 현재 버전에 새로 추가된 통계를 탐지합니다.
    DIMS 정보를 기반으로 신규 PEG 여부를 판단합니다.
    
    Single Responsibility: 신규 통계 검사만 담당
    Dependency Inversion: DimsDataProvider에 의존
    """
    
    def __init__(self, dims_provider: Optional['DimsDataProvider'] = None):
        """
        신규 통계 탐지기 초기화
        
        Args:
            dims_provider: DIMS 데이터 제공자 (의존성 주입)
        """
        super().__init__("NewStatisticsDetector")
        self.dims_provider = dims_provider or MockDimsDataProvider()
        
        self.logger.info(f"🆕 신규 통계 탐지기 초기화됨 (Data Provider: {type(self.dims_provider).__name__})")
    
    def _execute_detection(self, 
                          peg_data: Dict[str, List[PegSampleSeries]], 
                          config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Analysis] 신규 통계 탐지 실행
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: 신규 통계 탐지 결과
        """
        try:
            affected_cells = set()
            affected_pegs = set()
            details = {}
            
            for cell_id, peg_series_list in peg_data.items():
                for series in peg_series_list:
                    # DIMS 제공자를 통해 신규 PEG 여부 확인
                    if self.dims_provider.is_new_peg(series.peg_name):
                        affected_cells.add(cell_id)
                        affected_pegs.add(series.peg_name)
                        details[f"{cell_id}.{series.peg_name}"] = {
                            "peg_name": series.peg_name,
                            "detection_reason": "New PEG not present in previous PKG version",
                            "first_appearance": "current_version"
                        }
                        
                        self.logger.debug(f"✨ 신규 통계 감지: {cell_id}.{series.peg_name}")
            
            return AnomalyDetectionResult(
                anomaly_type="New",
                affected_cells=affected_cells,
                affected_pegs=affected_pegs,
                details=details,
                confidence=0.8,  # DIMS 데이터 정확도에 의존
                metadata={"detection_enabled": True}
            )
            
        except Exception as e:
            self.logger.error(f"❌ 신규 통계 탐지 중 오류: {e}")
            raise
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """[Validation] 신규 통계 탐지기 특화 설정 검증"""
        return True


# =============================================================================
# High Delta 이상 탐지기
# =============================================================================

class HighDeltaAnomalyDetector(BaseAnomalyDetector):
    """
    High Delta 이상 탐지기
    
    δ > β3 조건을 만족하는 높은 변화율을 탐지합니다.
    - 변화율(δ) = (Post - Pre) / Pre * 100
    - 예기치 않은 급격한 트래픽/성능 변화를 감지하는 것이 목적입니다.
    
    Single Responsibility: High Delta 검사만 담당
    """
    
    def __init__(self):
        """High Delta 탐지기 초기화"""
        super().__init__("HighDeltaAnomalyDetector")
    
    def _execute_detection(self, 
                          peg_data: Dict[str, List[PegSampleSeries]], 
                          config: Dict[str, Any]) -> AnomalyDetectionResult:
        """
        [Analysis] High Delta 이상 탐지 실행
        
        Args:
            peg_data: PEG 데이터
            config: 탐지 설정
            
        Returns:
            AnomalyDetectionResult: High Delta 탐지 결과
        """
        try:
            beta_3 = config.get("beta_3", 500.0)
            affected_cells = set()
            affected_pegs = set()
            details = {}
            
            for cell_id, peg_series_list in peg_data.items():
                for series in peg_series_list:
                    delta_info = self._calculate_delta(series, beta_3)
                    
                    if delta_info["is_high_delta"]:
                        affected_cells.add(cell_id)
                        affected_pegs.add(series.peg_name)
                        details[f"{cell_id}.{series.peg_name}"] = delta_info
                        
                        self.logger.debug(f"📈 High Delta 감지: {cell_id}.{series.peg_name} "
                                        f"(δ={delta_info['delta_percentage']:.1f}% > {beta_3}%)")
            
            return AnomalyDetectionResult(
                anomaly_type="High Delta",
                affected_cells=affected_cells,
                affected_pegs=affected_pegs,
                details=details,
                confidence=1.0,  # 수학적 계산이므로 확실함
                metadata={"detection_enabled": True}
            )
            
        except Exception as e:
            self.logger.error(f"❌ High Delta 탐지 실행 중 오류: {e}")
            raise
    
    def _calculate_delta(self, series: PegSampleSeries, beta_3: float) -> Dict[str, Any]:
        """
        [Helper] 변화율 계산 및 High Delta 판정
        
        Args:
            series: PEG 시리즈
            beta_3: High Delta 임계값
            
        Returns:
            Dict[str, Any]: Delta 계산 결과
        """
        try:
            # Pre/Post 평균 계산
            pre_valid = [s for s in series.pre_samples if s is not None]
            post_valid = [s for s in series.post_samples if s is not None]
            
            if not pre_valid or not post_valid:
                return {
                    "delta_percentage": None,
                    "is_high_delta": False,
                    "calculation_error": "Insufficient data for delta calculation"
                }
            
            pre_mean = np.mean(pre_valid)
            post_mean = np.mean(post_valid)
            
            # δ = (post-pre)/pre * 100 계산
            if pre_mean == 0:
                # 0으로 나누기 방지
                if post_mean == 0:
                    delta_pct = 0.0
                else:
                    # Pre가 0인데 Post가 있으면 무한대 증가로 간주 (방향에 따라 +/-)
                    delta_pct = 10000.0 if post_mean > 0 else -10000.0 # 임의의 큰 값
                calculation_note = "pre_mean_zero_special_case"
            else:
                delta_pct = ((post_mean - pre_mean) / pre_mean) * 100
                calculation_note = "normal_calculation"
            
            # High Delta 판정 (절댓값 기준)
            is_high_delta = abs(delta_pct) > beta_3
            
            return {
                "pre_mean": float(pre_mean),
                "post_mean": float(post_mean),
                "delta_percentage": float(delta_pct),
                "abs_delta": abs(delta_pct),
                "beta_3_threshold": beta_3,
                "is_high_delta": is_high_delta,
                "calculation_note": calculation_note,
                "pre_sample_count": len(pre_valid),
                "post_sample_count": len(post_valid)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Delta 계산 중 오류: {e}")
            return {"delta_percentage": None, "is_high_delta": False, "calculation_error": str(e)}
    
    def _validate_specific_config(self, config: Dict[str, Any]) -> bool:
        """[Validation] High Delta 탐지기 특화 설정 검증"""
        beta_3 = config.get("beta_3", 500.0)  # 기본값 제공
        
        if not isinstance(beta_3, (int, float)) or beta_3 <= 0:
            self.logger.error(f"beta_3 must be a positive number, got {beta_3}")
            return False
        
        # self.logger.debug(f"High Delta detector validated with beta_3={beta_3}")
        return True


# =============================================================================
# DIMS 데이터 제공자 인터페이스 (Dependency Inversion)
# =============================================================================

class DimsDataProvider(ABC):
    """
    DIMS 데이터 제공자 추상 인터페이스
    
    Dependency Inversion Principle 적용:
    - 고수준 모듈(탐지기)이 저수준 모듈(DIMS 접근)에 의존하지 않음
    - 추상화에 의존하여 테스트 및 확장성 확보
    """
    
    @abstractmethod
    def get_peg_range(self, peg_name: str) -> Optional[Dict[str, float]]:
        """
        [Abstract] PEG의 Range 정보 조회
        
        Args:
            peg_name: PEG 이름
            
        Returns:
            Optional[Dict[str, float]]: {"min": min_value, "max": max_value} 또는 None
        """
        pass
    
    @abstractmethod
    def is_new_peg(self, peg_name: str) -> bool:
        """
        [Abstract] 신규 PEG 여부 확인
        
        Args:
            peg_name: PEG 이름
            
        Returns:
            bool: 신규 PEG 여부
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """[Abstract] 제공자 메타 정보 반환"""
        pass


class MockDimsDataProvider(DimsDataProvider):
    """
    Mock DIMS 데이터 제공자
    
    테스트 및 개발 환경용 Mock 구현체입니다.
    실제 DIMS 연동 없이 테스트를 수행할 수 있게 해줍니다.
    """
    
    def __init__(self):
        """Mock 제공자 초기화"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Mock Range 데이터 (예시 데이터)
        self.mock_ranges = {
            "AirMacDLThruAvg": {"min": 500.0, "max": 5000.0},
            "AirMacULThruAvg": {"min": 100.0, "max": 2000.0},
            "ConnNoAvg": {"min": 0.0, "max": 100.0}
        }
        
        # Mock 신규 PEG 목록
        self.new_pegs = {"NewPEG2025", "TestPEG_v2"}
        
        self.logger.info(f"🧪 Mock DIMS 제공자 초기화됨 ({len(self.mock_ranges)} ranges)")
    
    def get_peg_range(self, peg_name: str) -> Optional[Dict[str, float]]:
        """Mock Range 정보 반환"""
        return self.mock_ranges.get(peg_name)
    
    def is_new_peg(self, peg_name: str) -> bool:
        """Mock 신규 PEG 여부 확인"""
        return peg_name in self.new_pegs
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Mock Provider 정보"""
        return {
            "provider_name": "MockDimsDataProvider",
            "description": "Mock DIMS data provider for testing",
            "provider_type": "mock",
            "version": "1.0.0",
            "available_ranges": len(self.mock_ranges),
            "available_new_pegs": len(self.new_pegs),
            "data_available": True
        }


# =============================================================================
# 탐지기 팩토리 (Factory Pattern + Dependency Injection)
# =============================================================================

class AnomalyDetectorFactory:
    """
    이상 탐지기 팩토리
    
    Factory Pattern과 Dependency Injection을 결합하여
    탐지기 인스턴스를 생성하고 관리합니다. 추후 DIMS 제공자 등 의존성을 관리하기 쉽습니다.
    """
    
    def __init__(self, dims_provider: Optional[DimsDataProvider] = None):
        """
        팩토리 초기화
        
        Args:
            dims_provider: DIMS 데이터 제공자 (의존성 주입)
        """
        self.dims_provider = dims_provider or MockDimsDataProvider()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"🛠️ 이상 탐지기 팩토리 초기화 완료")
    
    def create_range_detector(self) -> RangeAnomalyDetector:
        """Range 탐지기 생성"""
        return RangeAnomalyDetector(self.dims_provider)
    
    def create_nd_detector(self) -> NDAnomalyDetector:
        """ND 탐지기 생성"""
        return NDAnomalyDetector()
    
    def create_zero_detector(self, zero_tolerance: float = 1e-10) -> ZeroAnomalyDetector:
        """Zero 탐지기 생성"""
        return ZeroAnomalyDetector(zero_tolerance)
    
    def create_new_statistics_detector(self) -> NewStatisticsDetector:
        """신규 통계 탐지기 생성"""
        return NewStatisticsDetector(self.dims_provider)
    
    def create_high_delta_detector(self) -> HighDeltaAnomalyDetector:
        """High Delta 탐지기 생성"""
        return HighDeltaAnomalyDetector()
    
    def create_detector(self, detector_type: str) -> BaseAnomalyDetector:
        """이름으로 특정 탐지기 생성"""
        detector_map = {
            "Range": self.create_range_detector,
            "ND": self.create_nd_detector,
            "Zero": self.create_zero_detector,
            "New": self.create_new_statistics_detector,
            "High Delta": self.create_high_delta_detector,
            "range": self.create_range_detector,  # 소문자 별칭
            "nd": self.create_nd_detector,
            "zero": self.create_zero_detector,
            "new": self.create_new_statistics_detector,
            "high_delta": self.create_high_delta_detector
        }
        
        if detector_type not in detector_map:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        return detector_map[detector_type]()
    
    def create_all_detectors(self) -> Dict[str, BaseAnomalyDetector]:
        """모든 탐지기 생성"""
        return {
            "range": self.create_range_detector(),
            "nd": self.create_nd_detector(),
            "zero": self.create_zero_detector(),
            "new": self.create_new_statistics_detector(),
            "high_delta": self.create_high_delta_detector()
        }
    
    def get_available_detectors(self) -> List[str]:
        """사용 가능한 탐지기 목록"""
        return ["range", "nd", "zero", "new", "high_delta"]


# =============================================================================
# 초기화 및 로깅
# =============================================================================

logger.info("✅ Anomaly detectors module loaded successfully")
