# Choi 알고리즘 구현 완성 계획

## 목표
`ChoiJudgement` 서비스 내의 미구현된 핵심 로직을 완성하고, 24/7 상시 모니터링을 위한 기능을 통합합니다.

## User Review Required
> [!IMPORTANT]
> **Choi vs L2 전략 결정**:
> - **Main (L1)**: Choi 알고리즘을 기본 판정 엔진으로 사용 (운영자에게 명확한 OK/NOK 제공)
> - **Sub (L2)**: 기존 Mahalanobis 서비스를 "고급 분석(Advanced Stats)" 용도로 유지 및 보조 지표로 활용
> - **구현 우선순위**: Choi 알고리즘의 미구현 로직 완성을 최우선으로 진행합니다.

> [!IMPORTANT]
> **24/7 모니터링 전략 (Operational Strategy)**:
> - **비교 주기**: 매 1시간 (Hourly Sliding Window)
> - **비교 대상 (N-1)**: **"24시간 전 동일 시간대"**를 Reference로 사용 (Daily Seasonality 등락에 따른 오탐 방지)
> - **유효 구간 추출**: Fixed Head/Tail Exclusion 대신 **Choi Filtering (Section 6, Median-based)** 사용
> - **통계 단위**: 5분 또는 15분 단위 입력을 받아 내부적으로 병렬 처리

## Proposed Changes

### 1. Choi Algorithm Completion (Core)
`app/services/choi_judgement_service.py`의 TODO 항목들을 구현합니다.

#### [MODIFY] [app/services/choi_judgement_service.py](file:///d:/Coding/n8n/app/services/choi_judgement_service.py)
- `_organize_data_by_kpi_topics`: `kpi_definitions` 설정을 기반으로 입력 데이터를 토픽별(Main/Sub)로 구조화.
- `_combine_main_sub_results`: 5.4장 규칙(Main OK + Sub NOK -> POK 등)을 적용하여 최종 판정 도출.
- `analyze_kpi_stats`: 전체 분석 파이프라인 연결.

### 2. Integration & Strategy Pattern
기존 코드와의 공존 및 통합을 위한 인터페이스 작업을 진행합니다.

#### [MODIFY] [app/services/choi_strategy_factory.py](file:///d:/Coding/n8n/app/services/choi_strategy_factory.py)
- 필요 시 L2(Mahalanobis) 서비스를 팩토리에서 함께 관리하거나 호출할 수 있는 구조 고려.

### 3. Agentic RAG Implementation (Self-Reflective)
검색 품질을 스스로 평가하고 개선하는 Agentic RAG를 구현합니다.

#### [NEW] [app/services/rag_service.py](file:///d:/Coding/n8n/app/services/rag_service.py)
- **RAGService Class**: 기초적인 Ingestion/Retrieval 기능 제공.

#### [NEW] [app/agent_module/rag_nodes.py](file:///d:/Coding/n8n/app/agent_module/rag_nodes.py)
기존 `nodes.py`에서 RAG 로직을 분리하여 확장합니다.
- `retrieve_node`: 문서 검색 수행.
- `grade_documents_node`: 문서 적합성 평가 (LLM Grader).
- `rewrite_query_node`: 쿼리 재작성 (LLM Rewriter).
- `generate_diagnosis_node`: 최종 답변 생성.

#### [MODIFY] [app/agent_module/graph.py](file:///d:/Coding/n8n/app/agent_module/graph.py)
- LangGraph 워크플로우에 순환(Cycle) 구조 추가: `Retrieve` -> `Grade` -> (`Rewrite` -> `Retrieve`) OR `Generate`.

### 4. Verification Plan

#### [NEW] [tests/test_rag_flow.py](file:///d:/Coding/n8n/tests/test_rag_flow.py)
- **Flow Test**: 부적절한 쿼리에 대해 Rewrite가 발생하는지 테스트(Node 단위).
- **Integration Test**: 전체 Agent Workflow가 정상적으로 Loop를 도는지 확인.

### 5. Running Tests
```bash
# 1. Create a dummy test file to verify environment
# 2. Run pytest
pytest tests/test_rag_flow.py -v
```
