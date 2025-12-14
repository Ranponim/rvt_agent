from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.agent_module.router import router as agent_router

app = FastAPI(title="3GPP KPI Dashboard API (Dev)", version="0.1.0")

# CORS for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the Agent Module
app.include_router(agent_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the 3GPP KPI Analysis Agent Dev Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
