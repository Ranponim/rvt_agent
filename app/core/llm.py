from langchain_openai import ChatOpenAI
from app.core.config import settings

def get_llm():
    return ChatOpenAI(
        model=settings.AGENT_MODEL_NAME,
        base_url=settings.AGENT_API_URL,
        api_key="not-needed"
    )

llm = get_llm()
