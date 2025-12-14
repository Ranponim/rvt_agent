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

# --- Conditional Edges ---

def check_queue_condition(state: AgentState) -> str:
    """Check if there are items in the queue to process."""
    if state.get("current_rag_anomaly"):
        return "retrieve"
    return "finalize"

def check_relevance_condition(state: AgentState) -> str:
    """Check grader result."""
    if state.get("is_relevant"):
        return "generate"
    return "rewrite"

def check_retry_condition(state: AgentState) -> str:
    """Check retry limit."""
    if state.get("rag_retry_count", 0) < 3:
        return "retrieve"
    return "generate" # Fallback to generation even if not relevant

def create_agent_graph():
    """
    Builds the Agentic RAG LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
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

    # 2. Add Edges
    workflow.set_entry_point("parse_data")
    
    workflow.add_edge("parse_data", "analyze_kpi")
    workflow.add_edge("analyze_kpi", "start_rag")
    workflow.add_edge("start_rag", "pop_queue")
    
    # Queue Loop
    workflow.add_conditional_edges(
        "pop_queue",
        check_queue_condition,
        {
            "retrieve": "retrieve",
            "finalize": "finalize"
        }
    )
    
    # RAG Cycle
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        check_relevance_condition,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    workflow.add_conditional_edges(
        "rewrite",
        check_retry_condition,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    
    # After generation, go back to queue
    workflow.add_edge("generate", "pop_queue")
    
    # Finalize
    workflow.add_edge("finalize", END)

    # 3. Compile
    app = workflow.compile()
    return app

# Singleton instance
agent_app = create_agent_graph()
