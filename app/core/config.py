import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "3GPP KPI Analysis Agent"
    VERSION: str = "0.1.0"
    
    # RAG Settings
    KNOWLEDGE_BASE_DIR: str = os.path.join(os.getcwd(), "app/knowledge_base")
    CHROMA_DB_DIR: str = os.path.join(os.getcwd(), "chroma_db")
    
    # Model Settings (Local LLM / OpenAI Compatible)
    # 로컬 LLM 구동을 위해 API Key 의존성 제거 (dummy 값 설정)
    AGENT_API_URL: str = os.getenv("AGENT_API_URL", "http://localhost:1234/v1") 
    AGENT_MODEL_NAME: str = os.getenv("AGENT_MODEL_NAME", "gpt-4o") 

    # PMDATA API Settings
    PMDATA_API_BASE_URL: str = os.getenv("PMDATA_API_BASE_URL", "http://165.213.69.30:8101")
    PMDATA_API_SUMMARIES_PATH: str = os.getenv("PMDATA_API_SUMMARIES_PATH", "/ems/summaries")
    PMDATA_API_TOTAL_PATH: str = os.getenv("PMDATA_API_TOTAL_PATH", "/ems/total")
    
    # Default Target info (Need to adjust based on real environment)
    PMDATA_EMS: str = os.getenv("PMDATA_EMS", "SVR25AUDUVZWA02") # Example default
    PMDATA_NE_ID: str = os.getenv("PMDATA_NE_ID", "ne_1234") # Example default 



settings = Settings()
