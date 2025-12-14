# Choi 알고리즘 24/7 연속 모니터링 운영 전략 (SW Change Focus)

## 1. 개요 (Strategy Overview)
통신망 운용 환경에서는 **SW 변경(Upgrade/Patch)** 전후의 성능 변화를 검증하는 것이 가장 중요합니다. 
따라서 24/7 모니터링 Agent의 핵심 전략은 **"SW 버전 변경 이벤트(Version Change Event)"**를 실시간으로 감지하고, **변경 전(Pre)**과 **변경 후(Post)**의 성능을 즉시 비교 분석하는 것입니다.

## 2. 모니터링 시나리오 (Monitoring Scenarios)

사용자의 운용 환경에 맞춰 두 가지 시나리오를 정의합니다.

### **Scenario 1: 공식 SW 패키지 변경 (Primary)**
- **특징**: `version_desc` (Build ID)가 명시적으로 변경됨.
- **감지 방법**: API 응답의 `version_desc` 필드 모니터링.
- **대응**: **즉시 비교 검증 (Triggered Validation)** 수행.

### **Scenario 2: 단위 블록/디버깅 변경 (Secondary)**
- **특징**: 전체 SW 버전은 동일하지만 내부 로직(Block)만 변경됨. API상에서 버전 변화가 감지되지 않음.
- **감지 방법**: 데이터상으로 알 수 없음. (미래에 외부 Agent의 알림(Post Request)으로 처리 예정)
- **대응**: **일주기성 상시 감시 (Daily Seasonality Check)**로 이상 징후 포착.

---

## 3. Scenario 1: SW 버전 변경 감지 및 검증 프로세스

Agent는 24/7 루프를 돌며 **Data Polling**을 수행하다가, SW 버전이 바뀌는 순간을 포착하여 검증 프로세스를 트리거합니다.

### **Step 1: 버전 변경 감지 (Detection)**
- **Monitoring Target**: `app/aicrewpmdataapi/15min.json` (또는 실시간 API)의 `version_desc` 필드.
- **Logic**:
    - 매 주기(예: 15분)마다 데이터를 가져와 현재의 `Current Version String`을 확인.
    - `Last Known Version`과 다를 경우 **"Version Change Event"** 발생으로 간주.

### **Step 2: 데이터셋 확정 (Dataset Definition)**
버전 변경 시점을 기준($T_{change}$)으로 Pre와 Post 데이터셋을 동적으로 정의합니다.

- **Post Data (New SW)**:
    - **범위**: $T_{change}$ 이후 수집된 **N개** 샘플.
    - **수집 전략**: 변경 직후부터 데이터가 쌓일 때까지 대기(Wait)하거나, 실시간으로 들어오는 데이터를 하나씩 축적하여 N개가 되면 분석 시작.
    - **최소 샘플 수**: 신뢰도 확보를 위해 최소 4~12개 (1시간 분량) 권장.

- **Pre Data (Old SW)**:
    - **범위**: $T_{change}$ 직전 **N개** 샘플 (이전 버전에 속한 마지막 데이터).
    - **조건**: 반드시 **이전 버전의 데이터**여야 함. (버전 변경 중 발생한 과도기/점검 데이터는 Filtering으로 제외)

### **Step 3: Choi 알고리즘 실행 (Execution)**
확보된 Pre(Old) vs Post(New) 데이터셋을 `ChoiService`에 전달하여 비교 분석을 수행합니다.

1.  **Filtering**: Pre/Post 데이터 내 불안정 구간(Ramp-up/down) 자동 제거.
2.  **Validation**:
    - **OK**: 새 버전 성능이 기존 버전과 유사하거나 개선됨. -> **"Pass"** 기록.
    - **NOK/POK**: 새 버전에서 성능 저하 또는 특이 패턴 발생. -> **"Rollback Alert"** 또는 **"Warning"** 발송.

---

## 4. Scenario 2: 상시 감시 (Daily Seasonality)
*SW 버전이 바뀌지 않았더라도, 잠수함 패치(Block Change)나 외부 요인에 의한 성능 변화를 감지하기 위해 보조적으로 수행합니다.*

- **비교 대상**: 현재 시점(N) vs **어제 동일 시간대(N-1)**.
- **목적**: `Scenario 1`이 놓치는 변경사항이나 돌발적인 성능 저하 탐지.
- **동작**: 매 시간 수행 (기존 계획 유지).

---

## 5. Future Work: 외부 Trigger 연동
사용자 요청(Case 2 대응)에 따라, 외부 시스템(CI/CD 파이프라인 또는 타 Agent)에서 **"SW 변경 알림(WebHook/Post Request)"**을 수신하는 기능을 추가합니다.
- 알림 수신 시: "지금부터가 변경 후(Post) 시점이다"라고 인지하고 Step 2 프로세스 즉시 가동.
