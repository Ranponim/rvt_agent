import logging
import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from app.agent_module.state import AgentState
from app.services.rag_service import rag_service
from app.core.llm import llm

# 로거 설정
logger = logging.getLogger(__name__)

# --- Helper Prompts ---

GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. 
Here is the retrieved document:
{context}

Here is the user question: 
{question}

If the document contains keyword(s) or semantic meaning useful to answer the question, assess it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

REWRITE_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the initial and formulate an improved question.
Input Question: {question}
Output with only the improved question and nothing else."""

DIAGNOSIS_PROMPT = """You are an expert telecom network engineer.
Based on the [Context Documents] below, analyze the [Anomaly] and provide a Root Cause and Action Plan.

[Context Documents]
{context}

[Anomaly]
- Title: {title}
- Description: {description}
- Related KPIs: {related_kpis}

[Instructions]
1. Summarize the 'Root Cause' in one sentence.
2. Provide a step-by-step 'Action Plan'.
3. If the context doesn't help, say "No specific manual found" and suggest general checks.

Format: JSON
{{
    "root_cause": "...",
    "action_plan": "..."
}}
"""

# --- Nodes ---

def start_rag_process(state: AgentState) -> Dict[str, Any]:
    """
    [Node] RAG 프로세스 시작 (Start RAG)
    분석 단계에서 감지된 이상 징후(Anomaly)를 RAG 큐에 적재합니다.
    
    Args:
        state (AgentState): 현재 상태
        
    Returns:
        Dict: rag_queue 초기화 및 로그
    """
    logger.info("🚀 [Node: Start RAG] 시작")
    anomalies = state.get("anomalies", [])
    
    # 실제 이상 징후만 필터링 (is_anomaly=True)
    real_anomalies = [a for a in anomalies if a.get("is_anomaly")]
    
    if not real_anomalies:
        logger.info("✅ 분석할 이상 징후가 없습니다. RAG를 건너뜁니다.")
        return {"next_step": "end", "logs": ["✅ No anomalies to analyze."]}
        
    logger.info(f"📋 RAG 분석 큐 생성: 총 {len(real_anomalies)} 건")
    return {
        "rag_queue": real_anomalies,
        "rag_completed_anomalies": [], # 완료된 항목 저장용
        "logs": [f"🚀 Starting RAG analysis for {len(real_anomalies)} items."]
    }

def pop_from_rag_queue(state: AgentState) -> Dict[str, Any]:
    """
    [Node] 큐에서 항목 추출 (Pop from Queue)
    RAG 큐에서 다음 분석 대상을 꺼내어 'current_rag_anomaly'로 설정합니다.
    분석을 위한 초기 검색 쿼리도 생성합니다.
    """
    queue = state.get("rag_queue", [])
    
    if not queue:
        logger.info("🏁 RAG 큐 소진. 처리를 종료합니다.")
        return {"current_rag_anomaly": None} # Queue finished
    
    current = queue[0]
    remaining = queue[1:]
    
    # 초기 검색 쿼리 생성 (제목 + 설명 조합)
    query = f"{current['title']} {current['description']}"
    
    logger.info(f"👉 [항목 처리 시작] {current['title']}")
    logger.debug(f"🔍 생성된 검색 쿼리: {query}")
    
    return {
        "rag_queue": remaining,
        "current_rag_anomaly": current,
        "search_query": query,
        "rag_retry_count": 0,
        "logs": [f"🔍 Analyzing: {current['title']}"]
    }

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node] 문서 검색 (Retrieve)
    Vector DB를 통해 관련 문서를 검색합니다.
    """
    query = state["search_query"]
    logger.info(f"📚 [Retrieve] 문서 검색 시작: '{query}'")
    
    try:
        docs = rag_service.search(query, k=3)
        logger.info(f"✅ 검색 완료: {len(docs)} 개의 문서 발견")
    except Exception as e:
        logger.error(f"❌ 문서 검색 중 오류 발생: {str(e)}", exc_info=True)
        docs = []

    return {
        "retrieved_docs": docs,
        "logs": [f"📚 Retrieved {len(docs)} documents."]
    }

def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node] 문서 평가 (Grade)
    검색된 문서가 질문과 관련이 있는지 LLM을 통해 평가합니다.
    """
    logger.info("🧐 [Grade] 문서 적합성 평가 시작")
    docs = state.get("retrieved_docs", [])
    query = state["search_query"]
    
    if not docs:
        logger.warning("⚠️ 평가할 문서가 없습니다.")
        return {"is_relevant": False, "logs": ["⚠️ No documents found."]}
        
    # 문서 내용 결합 (Simplified Bulk Grading)
    context = "\n\n".join(docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_PROMPT),
        ("human", "User question: {question}\n\nContext: {context}")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({"question": query, "context": context})
        score = response.content.lower().strip()
        is_relevant = "yes" in score
        
        logger.info(f"📝 평가 결과: {'적합(Relevant)' if is_relevant else '부적합(Not Relevant)'} (Score: {score})")
        
        return {
            "is_relevant": is_relevant,
            "logs": [f"🧐 Relevance Check: {is_relevant}"]
        }
    except Exception as e:
        logger.error(f"❌ 문서 평가 중 오류 발생: {str(e)}", exc_info=True)
        # 오류 발생 시 안전하게 부적합 처리 또는 재시도 로직 필요 (여기선 False)
        return {"is_relevant": False, "logs": [f"❌ Grading Error: {str(e)}"]}

def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node] 쿼리 재작성 (Rewrite)
    문서가 부적합할 경우, 검색 성능을 높이기 위해 쿼리를 재작성합니다.
    """
    current_query = state["search_query"]
    retry_count = state.get("rag_retry_count", 0) + 1
    
    logger.info(f"🔄 [Rewrite] 쿼리 재작성 시도 ({retry_count}회차)")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        ("human", "Input Question: {question}")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({"question": current_query})
        new_query = response.content.strip()
        
        logger.info(f"✨ 새로운 쿼리 생성: '{new_query}'")
        
        return {
            "search_query": new_query,
            "rag_retry_count": retry_count,
            "logs": [f"🔄 Rewriting Query ({retry_count}/3): {new_query}"]
        }
    except Exception as e:
        logger.error(f"❌ 쿼리 재작성 실패: {str(e)}", exc_info=True)
        return {
            "search_query": current_query, # 실패 시 기존 쿼리 유지
            "rag_retry_count": retry_count,
            "logs": [f"❌ Rewrite Failed: {str(e)}"]
        }

def generate_diagnosis_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node] 진단 생성 (Generate Diagnosis)
    검색된 문서를 바탕으로 원인(Root Cause)과 조치 방안(Action Plan)을 생성합니다.
    """
    logger.info("💡 [Generate] 최종 진단 생성 시작")
    anomaly = state["current_rag_anomaly"]
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(docs) if docs else "No specific documents found."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."), # Generic system msg prepended
        ("human", DIAGNOSIS_PROMPT)
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({
            "context": context,
            "title": anomaly["title"],
            "description": anomaly["description"],
            "related_kpis": str(anomaly["related_kpis"])
        })
        
        # Parse JSON
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        # Update Anomaly
        anomaly["root_cause"] = result.get("root_cause", "Analysis Failed")
        anomaly["action_plan"] = result.get("action_plan", "Please check manually.")
        
        logger.info(f"✅ 진단 생성 완료: {anomaly['root_cause']}")
        
    except Exception as e:
        logger.error(f"❌ 진단 생성 실패: {str(e)}", exc_info=True)
        anomaly["root_cause"] = "Analysis Error"
        anomaly["action_plan"] = f"Failed to generate analysis: {str(e)}"

    # We return the UPDATED anomaly in a list. 
    return {
        "current_rag_anomaly": None, # Finished processing
        "rag_completed_anomalies": [anomaly], 
        "logs": ["✅ Diagnosis Generated."]
    }

def finalize_rag_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node] RAG 종료 처리 (Finalize)
    RAG 과정을 통해 보강된 이상 징후 리스트를 최종 상태에 반영합니다.
    """
    logger.info("🏁 [Node: Finalize] RAG 분석 종료")
    enriched = state.get("rag_completed_anomalies", [])
    
    # (Optional) 기존 state['anomalies']와 병합 로직이 필요하다면 여기서 수행
    
    return {
        "anomalies": enriched,
        "logs": ["🏁 RAG Analysis Completed."]
    }

