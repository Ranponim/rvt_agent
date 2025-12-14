import json
import asyncio
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from typing import Dict, Any

from app.agent_module.graph import agent_app
from app.models.kpi_data import PMData

router = APIRouter(prefix="/agent", tags=["3GPP Agent"])

# --- 1. Real-time Streaming Endpoint (SSE) ---
@router.get("/stream")
async def stream_agent_events(request: Request):
    """
    Server-Sent Events (SSE) endpoint to visualize Agent thinking process.
    """
    async def event_generator():
        # In a real scenario, we would subscribe to a shared queue or Redis channel.
        # For this MVP, we simulate a "Keep Alive" loop or wait for trigger commands.
        # Since LangGraph execution is triggered by POST, we need a way to bridge them.
        
        # PROTOTYPE: Just emit a connection success message.
        # The actual 'graph events' would need to be broadcasted from the execution.
        yield {
            "event": "connect",
            "data": "Connected to Agent Stream. Waiting for analysis tasks..."
        }
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                # Heartbeat to keep connection alive
                await asyncio.sleep(5)
                # yield {"event": "heartbeat", "data": "ping"}
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())

# --- 2. Trigger Analysis Endpoint ---
@router.post("/analyze/15min")
async def trigger_analysis_15min(payload: Dict[str, Any]):
    """
    Directly input raw 15min.json data to trigger the agent.
    Returns the final state (Anomalies).
    """
    # 1. Run the Graph
    # We use 'invoke' for synchronous wait, or 'astream' if we want stream back.
    # To support SSE effectively, we would use a BackgroundTask to run 'astream' 
    # and push events to the SSE queue.
    
    # For MVP simplicity: We run it and return result.
    
    # [SIMULATION] Agent Memory: Load 1hour.json as History Context
    import os
    history_payload = None
    try:
        # Load 1hour.json from the adjacent api folder
        h_path = os.path.join("app", "aicrewpmdataapi", "1hour.json")
        with open(h_path, "r", encoding="utf-8") as f:
            history_payload = json.load(f)
    except Exception:
        pass # History load failed, proceed without it

    initial_state = {
        "current_data_15min": payload,
        "current_data_1hour": history_payload, # Inject History
        "logs": [],
        "anomalies": []
    }

    
    result = await agent_app.ainvoke(initial_state)
    
    return {
        "status": "completed",
        "anomalies": result.get("anomalies", []),
        "logs": result.get("logs", [])
    }

@router.get("/health")
def health_check():
    return {"status": "ok", "module": "3GPP Agent"}
