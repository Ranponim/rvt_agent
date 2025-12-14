# Project Tasks

## Choi Algorithm Implementation
- [x] Implement Abnormal Stats Detection (Chapter 4)
- [x] Implement KPI Statistical Analysis (Chapter 5)
    - [x] Basic Statistics (Mean, Std, CV, etc.)
    - [x] Comparison Logic (Pre vs Post)
    - [x] Rule-based Analyzers (CantJudge, Similar, Delta, etc.)
    - [x] Main/Sub KPI Combination Logic
- [x] Unit Tests for Choi Logic (`tests/test_choi_judgement_logic.py`)
- [x] Integration with Agent Node (`analyze_kpi_node`)
- [x] Integration Tests (`tests/test_integration_choi_rag.py`)
    - [x] Basic Integration (Low Delta Anomaly)
    - [x] Normal Case (No Anomaly)
    - [x] Cant Judge Case
    - [x] Multiple Anomalies
    - [x] RAG Triggering Verification

## Agentic RAG Implementation
- [x] Agent State Definition
- [x] RAG Graph Construction
- [x] Knowledge Base Setup (ChromaDB)
- [x] Document Ingestion Pipeline
- [x] RAG Node Implementation
    - [x] `retrieve_node` (Hybrid Search)
    - [x] `grade_documents_node` (Relevance Check)
    - [x] `generate_answer_node` (LLM Generation)
- [x] End-to-End RAG Testing (Mocked LLM)

## System Verification
- [x] Full Workflow Test (Input -> Choi -> Anomaly -> RAG -> Output) - *Verified via `tests/test_e2e_rag_mock.py`*
- [x] Documentation Update (Walkthrough / Design Doc)
