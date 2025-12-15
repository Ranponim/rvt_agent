import logging
from langgraph.graph import StateGraph, END
from app.agent_module.state import AgentState
from app.agent_module.nodes import parse_data_node, analyze_kpi_node
from app.agent_module.rag_nodes import (
    start_rag_process,
    pop_from_rag_queue,
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_diagnosis_node,
    finalize_rag_node
)

# 로거 설정
logger = logging.getLogger(__name__)

# --- 조건부 엣지 (Conditional Edges) ---

def check_queue_condition(state: AgentState) -> str:
    """
    [Edge] 큐 상태 확인 (Check Queue)
    
    RAG 큐에 처리할 이상 항목(Anomaly)이 남아있는지 확인합니다.
    - 항목이 있으면 'retrieve' (검색) 단계로 이동
    - 없으면 'finalize' (종료) 단계로 이동
    """
    if state.get("current_rag_anomaly"):
        logger.debug("👉 [Edge] 큐에 항목 존재 -> 검색(Retrieve) 단계로 이동")
        return "retrieve"
    
    logger.debug("👉 [Edge] 큐 비어있음 -> 종료(Finalize) 단계로 이동")
    return "finalize"

def check_relevance_condition(state: AgentState) -> str:
    """
    [Edge] 관련성 확인 (Check Relevance)
    
    문서 등급(Grader) 결과를 바탕으로 다음 단계를 결정합니다.
    - 관련성 있음: 'generate' (진단 생성)
    - 관련성 없음: 'rewrite' (쿼리 재작성)
    """
    if state.get("is_relevant"):
        logger.debug("👉 [Edge] 문서 관련성 있음 -> 생성(Generate) 단계로 이동")
        return "generate"
    
    logger.debug("👉 [Edge] 문서 관련성 부족 -> 재작성(Rewrite) 단계로 이동")
    return "rewrite"

def check_retry_condition(state: AgentState) -> str:
    """
    [Edge] 재시도 횟수 확인 (Check Retry)
    
    쿼리 재작성 및 검색 재시도 횟수를 확인합니다.
    - 최대 재시도(3회) 미만: 'retrieve' (재검색)
    - 최대 재시도 도달: 'generate' (강제 생성 - Best Effort)
    """
    current_retry = state.get("rag_retry_count", 0)
    MAX_RETRIES = 3
    
    if current_retry < MAX_RETRIES:
        logger.warning(f"👉 [Edge] 재시도 조건 충족 ({current_retry}/{MAX_RETRIES}) -> 재검색(Retrieve) 단계로 이동")
        return "retrieve"
    
    logger.warning(f"👉 [Edge] 최대 재시도 초과 ({current_retry}) -> 강제 생성(Generate) 단계로 이동")
    return "generate" 

def create_agent_graph():
    """
    [Graph] Agentic RAG 워크플로우 생성 (Construct Graph)
    
    LangGraph를 사용하여 상태 기반의 에이전트 워크플로우를 구성합니다.
    - 노드: 데이터 파싱, KPI 분석, RAG 프로세스 (검색-평가-생성 루프)
    - 엣지: 실행 순서 및 분기 로직 정의
    """
    logger.info("🛠️ Agentic RAG 그래프 빌드 시작...")
    workflow = StateGraph(AgentState)

    # 1. 노드 추가 (Add Nodes)
    workflow.add_node("parse_data", parse_data_node)
    workflow.add_node("analyze_kpi", analyze_kpi_node)
    
    # RAG Nodes
    workflow.add_node("start_rag", start_rag_process)
    workflow.add_node("pop_queue", pop_from_rag_queue)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_diagnosis_node)
    workflow.add_node("finalize", finalize_rag_node)

    # 2. 엣지 연결 (Add Edges)
    workflow.set_entry_point("parse_data")
    
    workflow.add_edge("parse_data", "analyze_kpi")
    workflow.add_edge("analyze_kpi", "start_rag")
    workflow.add_edge("start_rag", "pop_queue")
    
    # Queue Loop (RAG Loop)
    workflow.add_conditional_edges(
        "pop_queue",
        check_queue_condition,
        {
            "retrieve": "retrieve",
            "finalize": "finalize"
        }
    )
    
    # RAG Retrieval & Grading Process
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        check_relevance_condition,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    # Retry Logic
    workflow.add_conditional_edges(
        "rewrite",
        check_retry_condition,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    
    # After generation, go back to queue for next anomaly
    workflow.add_edge("generate", "pop_queue")
    
    # Finalize
    workflow.add_edge("finalize", END)

    # 3. 컴파일 (Compile)
    app = workflow.compile()
    
    logger.info("✅ Agentic RAG 그래프 빌드 및 컴파일 완료")
    return app

# Singleton instance
agent_app = create_agent_graph()
