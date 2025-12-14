from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated
import operator
from app.models.kpi_data import PMData

class ValidationResult(TypedDict):
    is_anomaly: bool
    severity: str  # P1, P2, P3, NORMAL
    title: str
    description: str
    related_kpis: List[str]
    root_cause: Optional[str]
    action_plan: Optional[str]

class AgentState(TypedDict):
    """
    State for the LangGraph Agent.
    Maintains the context of the current analysis cycle.
    """
    # Input Data
    current_data_15min: Optional[Dict[str, Any]] # Raw JSON dict
    current_data_1hour: Optional[Dict[str, Any]] # Raw JSON dict
    
    # Processing Context
    parsed_data: Optional[PMData]
    
    # Analysis Results
    anomalies: List[ValidationResult] # Final results
    
    # Trace/Logs (for debugging & SSE)
    logs: Annotated[List[str], operator.add]
    
    # Next Step Control
    next_step: str

    # --- Agentic RAG State ---
    rag_queue: List[ValidationResult]  # Queue of anomalies to process
    current_rag_anomaly: Optional[ValidationResult] # Currently processing anomaly
    
    search_query: str                  # Current search query
    retrieved_docs: List[str]          # Retrieved document chunks
    rag_retry_count: int               # Rewrite attempt counter
    is_relevant: bool                  # Grader result
    
    rag_completed_anomalies: Annotated[List[ValidationResult], operator.add] # Accumulate enriched results

