"""
Choi 알고리즘 예외 처리 모듈
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import logging

class ChoiAlgorithmError(Exception):
    """
    Choi 알고리즘 관련 모든 예외의 기본 클래스
    """
    def __init__(
        self,
        message: str,
        error_code: str = "CHOI_ERROR",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
        
        self.context.update({
            "error_type": self.__class__.__name__,
            "timestamp": self.timestamp,
            "error_code": error_code
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }

class InsufficientDataError(ChoiAlgorithmError):
    def __init__(self, message: str = "데이터 부족", **kwargs):
        super().__init__(message, "INSUFFICIENT_DATA", context=kwargs)

class ConfigurationError(ChoiAlgorithmError):
    def __init__(self, message: str = "설정 오류", **kwargs):
        super().__init__(message, "CONFIGURATION_ERROR", context=kwargs)

class FilteringError(ChoiAlgorithmError):
    def __init__(self, message: str = "필터링 오류", **kwargs):
        super().__init__(message, "FILTERING_ERROR", context=kwargs)

class AbnormalDetectionError(ChoiAlgorithmError):
    def __init__(self, message: str = "이상 탐지 오류", **kwargs):
        super().__init__(message, "ABNORMAL_DETECTION_ERROR", context=kwargs)

class KPIAnalysisError(ChoiAlgorithmError):
    def __init__(self, message: str = "KPI 분석 오류", **kwargs):
        super().__init__(message, "KPI_ANALYSIS_ERROR", context=kwargs)

class DataValidationError(ChoiAlgorithmError):
    def __init__(self, message: str = "데이터 검증 오류", **kwargs):
        super().__init__(message, "DATA_VALIDATION_ERROR", context=kwargs)

class StrategyExecutionError(ChoiAlgorithmError):
    def __init__(self, message: str = "Strategy 실행 오류", **kwargs):
        super().__init__(message, "STRATEGY_EXECUTION_ERROR", context=kwargs)

class DIMSDataError(ChoiAlgorithmError):
    def __init__(self, message: str = "DIMS 데이터 오류", **kwargs):
        super().__init__(message, "DIMS_DATA_ERROR", context=kwargs)

class PerformanceError(ChoiAlgorithmError):
    def __init__(self, message: str = "성능 오류", **kwargs):
        super().__init__(message, "PERFORMANCE_ERROR", context=kwargs)

def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger_name: str = "app.exceptions"
) -> ChoiAlgorithmError:
    """
    일반 예외를 Choi 알고리즘 예외로 변환합니다.
    """
    logger = logging.getLogger(logger_name)
    
    if isinstance(exception, ChoiAlgorithmError):
        # logger.error(f"Choi 알고리즘 예외 발생: {exception.message}", extra=exception.context)
        return exception
    
    choi_exception = ChoiAlgorithmError(
        message=f"예상치 못한 오류가 발생했습니다: {str(exception)}",
        error_code="UNEXPECTED_ERROR",
        context=context or {},
        cause=exception
    )
    
    # logger.error(
    #     f"예상치 못한 예외를 Choi 예외로 변환: {str(exception)}",
    #     extra=choi_exception.context,
    #     exc_info=True
    # )
    
    return choi_exception

def create_exception(error_code: str, message: str, **kwargs) -> ChoiAlgorithmError:
    return ChoiAlgorithmError(message, error_code, context=kwargs)
