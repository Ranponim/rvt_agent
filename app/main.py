import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.agent_module.router import router as agent_router

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 수명 주기 관리
    
    서버 시작 시와 종료 시의 로직을 처리합니다.
    """
    logger.info("🚀 3GPP KPI Dashboard API 서버 시작 중...")
    yield
    logger.info("🛑 서버 종료 중...")

app = FastAPI(
    title="3GPP KPI Dashboard API (Dev)", 
    version="0.1.0",
    description="3GPP KPI 분석 에이전트 개발 서버",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드 접근 허용)
# 개발 환경: 모든 출처 허용 ("*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에이전트 모듈 라우터 포함
app.include_router(agent_router)

@app.get("/")
def read_root():
    """서버 상태 확인용 루트 엔드포인트"""
    logger.info("Health check endpoint called")
    return {"message": "Welcome to the 3GPP KPI Analysis Agent Dev Server (한글 지원)"}

if __name__ == "__main__":
    import uvicorn
    # 로컬 개발용 실행 설정
    # reload=True: 코드 변경 시 자동 재시작
    logger.info("Running uvicorn server locally...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

