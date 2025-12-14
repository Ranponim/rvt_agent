# Agentic RAG 상세 설계서 (Self-Reflective)

## 1. 개요
본 문서는 단순 검색-생성(Standard RAG)을 넘어, **에이전트가 검색 결과의 품질을 스스로 "검증(Grade)"하고 필요시 "재검색(Retry)"하는 Agentic RAG** 시스템의 설계서입니다.

### 핵심 차별점
- **Self-Correction**: 검색된 문서가 질문과 관련이 없으면, 쿼리를 수정하여 다시 검색합니다.
- **Answer Grading**: 생성된 답변이 문서에 근거하는지(Hallucination Check) 확인합니다.

---

## 2. Agentic 아키텍처 (LangGraph Flow)

에이전트는 다음과 같은 순환(Cycle) 구조를 가집니다.

```mermaid
graph TD
    Start([Start]) --> Retrieve[Retrieve Documents]
    Retrieve --> Grade{Grade Docs}
    
    Grade -->|Relevant| Generate[Generate Diagnosis]
    Grade -->|Not Relevant| Rewrite[Rewrite Query]
    
    Rewrite -->|Retry Loop (Max 3)| Retrieve
    
    Generate --> End([End])
    
    subgraph "Decision Process"
    Grade
    end
```

### 상태(State) 설계
`AgentState`에 RAG 전용 필드를 추가하여 추적합니다.

```python
class AgentState(TypedDict):
    # ... 기존 필드 ...
    anomalies: List[Dict]      # 감지된 이상 징후 목록
    
    # RAG 전용 State
    search_query: str          # 현재 검색 쿼리
    retrieved_docs: List[str]  # 검색된 문서 청크들
    retry_count: int           # 재시도 횟수 (Infinite Loop 방지)
    rag_status: str            # 'searching', 'grading', 'generating', 'done'
```

---

## 3. 상세 노드 설계

### 3.0. RAG Service (Core)
임베딩 및 Vector DB 로직은 `RAGService`에 캡슐화합니다 (Standard RAG와 동일).

### 3.1. `retrieve_node` (검색)
- **입력**: `search_query` (초기값은 Anomaly Description)
- **동작**: `RAGService.search()` 호출
- **출력**: `retrieved_docs` 업데이트

### 3.2. `grade_documents_node` (평가)
- **동작**: LLM에게 "이 문서가 이 문제(Query)를 해결하는 데 도움이 되는가?"를 Binary(Yes/No)로 질문.
- **판단 로직**:
  - **Yes**: `generate_node`로 이동.
  - **No**: `rewrite_query_node`로 이동. (단, `retry_count > 3`이면 강제 Generate 이동 - Fallback)

### 3.3. `rewrite_query_node` (재작성)
- **동작**: "현재 쿼리로는 관련 문서를 찾을 수 없습니다. 더 나은 검색어를 제안해 주세요."라고 LLM에 요청.
- **예시**:
  - 원본: *"Cell 123 congestion"* (문서 없음)
  - 재작성: *"LTE Downlink Throughput low with high Active Users troubleshooting"*

### 3.4. `generate_diagnosis_node` (생성)
- **동작**: 최종 선별된 문서들을 Context로 주입하여 원인 및 조치 가이드 생성.

---

## 4. 프롬프트 템플릿

### Grader Prompt
```text
당신은 검색된 문서의 적합성을 평가하는 감독관입니다.
사용자 질문: {query}
검색된 문서: {doc_content}

이 문서가 질문에 대한 답을 찾는데 도움이 되는 정보(키워드나 개념)를 포함하고 있습니까?
'예' 또는 '아니오'로만 답하세요.
```

### Rewriter Prompt
```text
초기 질문에 대한 검색 결과가 좋지 않습니다.
의도를 유지하면서 검색 엔진이 더 잘 이해할 수 있도록 질문을 영어 또는 전문 용어 위주로 재작성하세요.

초기 질문: {query}
재작성된 질문:
```

---

## 5. 구현 로드맵

1.  **`RAGService` 구현**: ChromaDB 연동 (공통)
2.  **노드 확장 (`node_rag.py`)**: `grade`, `rewrite` 함수 추가
3.  **Graph 수정**: LangGraph에 조건부 엣지(Conditional Edge) 추가

---

## 6. 구현 현황 및 사용 가이드 (Implementation & Usage)

### 6.1. 현재 구현 상태 (Current Implementation)
RAG 시스템은 아래 패키지와 서비스로 구현되어 검증되었습니다.

- **Core Service**: `app/services/rag_service.py`
  - **Vector DB**: ChromaDB (로컬 퍼시스턴트 모드)
  - **Embedding**: `jhgan/ko-sroberta-multitask` (HuggingFace)
  - **Status**: ✅ **Verified** (Indexing & Searching functional)
  
- **Graph Nodes**: `app/agent_module/rag_nodes.py`
  - `start_rag_process`: Anomaly를 RAG Queue로 변환
  - `pop_from_rag_queue`: Queue 관리 및 Query 생성
  - `retrieve_node`: ChromaDB Hybrid Search
  - `grade_documents_node`: LLM 기반 문서 적합성 평가
  - `generate_diagnosis_node`: LLM 기반 원인/조치 생성
  - **Status**: ✅ **Verified** (Unit Tests & Logic Inspection passed)

### 6.2. 사용 방법 (Usage Guide)

#### 1) 지식 베이스 구축 (Knowledge Base Setup)
RAG가 참조할 문서는 마크다운(.md) 형식으로 `app/knowledge_base/peg_docs/` 경로에 위치해야 합니다.
서비스 시작 시 자동으로 로드 및 인덱싱됩니다.

```bash
# 예시 문서 위치
app/knowledge_base/peg_docs/
├── packet_loss.md
├── rrc_setup_failure.md
└── template_peg.md
```

#### 2) 시스템 실행 (Running the System)
시스템은 `graph.py`를 통해 실행되며, `analyze_kpi_node`에서 이상 징후(Anomaly)가 발견되면 자동으로 RAG 프로세스로 전환됩니다.

**설정 파일 (`app/core/config.py`):**
```python
# LLM Endpoint 설정 (LM Studio 등 Local LLM 사용 시)
AGENT_API_URL="http://localhost:1234/v1"
AGENT_MODEL_NAME="gpt-4o" # 또는 로컬 모델명
```

#### 3) 테스트 및 검증 (Testing)
RAG 시스템의 각 컴포넌트는 아래 명령어로 독립 검증 가능합니다.

**Service 검증 (DB & Embedding):**
```bash
python -m pytest tests/test_rag_service_logic.py -v
```

**Flow Node 검증 (Queue Logic):**
```bash
python -m pytest tests/test_rag_flow_nodes.py -v
```

### 6.3. 트러블슈팅 (Troubleshooting)
- **문서 검색이 안될 때**: 
  - `d:/Coding/n8n/chroma_db` 폴더 삭제 후 재시작 (강제 재인덱싱)
  - `requirements.txt` 내 `sentence-transformers` 설치 확인
- **LLM 응답 오류**:
  - `AGENT_API_URL`이 실제 실행 중인 LLM 서버를 가리키는지 확인
  - 방화벽 및 네트워크 상태 확인
