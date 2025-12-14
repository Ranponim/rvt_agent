# 3GPP KPI 분석 에이전트 연동 가이드 (n8n)

이 문서는 n8n 워크플로우를 3GPP KPI 분석 에이전트에 연동하는 방법을 설명합니다.

## 1. 분석 요청 트리거 (HTTP Request Node)

이 에이전트는 REST API로 동작합니다. n8n에서 파일 소스로부터 읽은 JSON 데이터를 전송하여 분석을 요청할 수 있습니다.

- **URL**: `http://<AGENT_IP>:8000/agent/analyze/15min`
- **Method**: `POST`
- **인증 (Authentication)**: 없음 (또는 설정 시 API Key Header 추가)
- **헤더 (Headers)**:
  - `Content-Type`: `application/json`
- **본문 (Body Parameter)**:
  - "Raw (JSON)" 선택
  - 값: `{{ $json }}` (이전 노드에서 `15min.json` 내용을 출력한다고 가정)

### 응답 형식 (Response)
에이전트는 분석 결과가 담긴 JSON 객체를 반환합니다:
```json
{
  "status": "completed",
  "anomalies": [
    {
      "is_anomaly": true,
      "severity": "P2",
      "title": "Congestion Detected (폭주 감지됨)",
      "related_kpis": ["AirMacDLThruAvg"],
      "action_plan": "'pmPrbUsage' 파라미터를 확인하십시오..."
    }
  ]
}
```

## 2. 실시간 상태 스트리밍 (SSE)

n8n 대시보드나 별도 프론트엔드에서 에이전트의 "생각하는 과정"을 실시간으로 시각화할 때 사용합니다.

- **Endpoint**: `http://<AGENT_IP>:8000/agent/stream`
- **프로토콜**: Server-Sent Events (SSE)
- **참고**: n8n의 기본 `HTTP Request` 노드는 응답이 완료될 때까지 기다리므로 SSE를 직접 소비하기 어렵습니다. SSE는 주로 **React/Vue 프론트엔드**나 **Custom Code Node**에서 활용합니다.

## 3. 예시 워크플로우 (Concept)

1.  **Schedule Trigger**: 15분마다 실행.
2.  **Read Binary File**: 디스크나 FTP에서 `15min.json` 로드.
3.  **Parse JSON**: 바이너리를 JSON으로 변환.
4.  **HTTP Request**: `/agent/analyze/15min` 주소로 POST 요청.
5.  **IF Node**: `anomalies` 배열이 비어있는지 확인.
6.  **True (이상 발생 시)**: Slack/Email로 `root_cause` 및 `action_plan` 내용을 포함하여 알림 발송.
