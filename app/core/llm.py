"""
LLM 인스턴스 생성 및 설정 모듈

OpenAI 호환 LLM (예: LM Studio, Ollama 등) 클라이언트를 초기화하고 반환합니다.
설정값은 app.core.config.settings에서 가져옵니다.
"""

import logging
from langchain_openai import ChatOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_llm():
    """
    LLM 인스턴스 생성 및 반환
    
    Returns:
        ChatOpenAI: 설정된 LLM 인스턴스
    """
    try:
        logger.info(f"🤖 LLM 초기화 시작 (Model: {settings.AGENT_MODEL_NAME}, URL: {settings.AGENT_API_URL})")
        
        llm_instance = ChatOpenAI(
            model=settings.AGENT_MODEL_NAME,
            base_url=settings.AGENT_API_URL,
            api_key="not-needed"
        )
        
        logger.debug("✅ LLM 인스턴스 생성 완료")
        return llm_instance
        
    except Exception as e:
        logger.error(f"❌ LLM 초기화 중 오류 발생: {e}", exc_info=True)
        raise

# 전역 LLM 인스턴스 (lazy initialization을 위해 필요시 get_llm() 호출 권장)
llm = get_llm()

