# Choi vs L2 (Mahalanobis) 알고리즘 비교 분석

## 1. 개요
본 문서는 기존 시스템의 **L2 (Mahalanobis Analysis)** 알고리즘과 도입 예정인 **Choi 알고리즘**의 특징을 비교 분석하고, 두 알고리즘의 통합 및 공존 가능성을 검토합니다.

- **Choi 알고리즘**: 3GPP 도메인 지식 기반의 **규칙(Rule-based)** 중심 판정 시스템 (Deterministic)
- **L2 알고리즘**: 통계적 분포 차이 검정 기반의 **확률(Probabilistic)** 중심 분석 시스템 (Stochastic)

---

## 2. 상세 비교

### 2.1 차이점 (Differences)

| 특징 | Choi 알고리즘 (Rule-based) | L2 알고리즘 (Statistical - Mahalanobis/MW/KS) |
| :--- | :--- | :--- |
| **판정 기준** | **절대적 임계값** (Beta Values) 및 비즈니스 규칙 | **통계적 유의성** (P-value < 0.05, 분포 차이) |
| **주요 로직** | 6단계 필터링 → 5가지 이상 탐지 → 8단계 KPI 분석 | 변화율 스크리닝 → Mann-Whitney U & KS Test |
| **결과 형태** | **OK / POK / NOK / Can't Judge** (명확한 등급) | Critical / Warning / Caution (이상 점수 기반) |
| **데이터 요구** | Pre/Post 비교, 특정 샘플 수 이상, ND/Zero 처리 중요 | 분포를 형성할 수 있는 충분한 샘플 데이터 |
| **장점** | **설명 가능성(Explainability)** 높음, 운영자 직관과 일치 | **미세한 분포 변화** 감지 가능, 임계값 설정 불필요 |
| **단점** | 임계값(Beta)이 상황에 안 맞으면 오탐/미탐 가능 | 샘플 수가 적거나 노이즈가 많으면 신뢰도 하락 |

### 2.2 공통점 (Commonalities)

1.  **목적**: 네트워크 변경(파라미터 변경, 패치 등) 전후의 성능 변화를 감지하여 문제를 조기에 발견.
2.  **입력 데이터**: 시계열 PEG 데이터 (Time Series Data)를 입력으로 받아 Pre/Post 기간을 비교.
3.  **1차 필터링**:
    - Choi: 6-step Filtering (유효성 검사, 50% Rule)
    - L2: Change Rate Screening (단순 변화율로 1차 걸러냄)
4.  **이상 탐지**: Range 벗어남, 급격한 변화(High Delta) 등을 감지하는 기능 존재.

---

## 3. 우세한 알고리즘 (Superiority Analysis)

"어느 것이 더 좋은가?"는 사용 목적에 따라 다릅니다.

### **Choi 알고리즘이 우세한 경우 (운영/관제 관점)**
> **"명확한 조치가 필요할 때"**
- **SLA/KPI 관리가 엄격할 때**: "변화율 10% 이내" 같은 명확한 기준이 있을 때 적합합니다.
- **오탐(False Alarm) 피로도 감소**: 미세한 통계적 차이는 무시하고, 실제 서비스에 영향이 있는 '비즈니스 임계값' 이상의 변화만 잡습니다.
- **설명력 필요 시**: "왜 NOK인가?"에 대해 "Beta_3(500%)를 초과해서"라고 명확히 답할 수 있습니다. (L2는 "P-value가 0.03이라서"라고 답해야 함)

### **L2 알고리즘이 우세한 경우 (심층 분석/엔지니어링 관점)**
> **"알 수 없는 이상 징후를 찾을 때"**
- **데이터 분포 분석**: 평균은 같아도 분산이 커지거나 분포가 찌그러지는(Skewed) 변화를 감지할 때 탁월합니다.
- **임계값 설정이 어려울 때**: 새로운 KPI나 잘 모르는 지표에 대해 자동으로 이상을 탐지하고 싶을 때 유리합니다.

---

## 4. 복합 가능성 (Combination Strategy)

두 알고리즘은 상호 배타적이지 않으며, **상호 보완적(Complementary)** 으로 구성할 때 가장 강력한 시너지를 냅니다.

### **[제안] Hybrid Decision Model (복합 모델)**

**계층적 분석 구조 (Hierarchical Approach)**

1.  **Tier 1: Choi 알고리즘 (Primary Gatekeeper)**
    - 모든 셀/KPI에 대해 Choi 알고리즘을 우선 적용합니다.
    - **OK / NOK** 판정을 내립니다. 이는 즉각적인 운영 판단(Rollback 여부 등)의 기준이 됩니다.
    - 대부분의 일상적인 판정은 여기서 종결됩니다.

2.  **Tier 2: L2 알고리즘 (Secondary Insight - Optional)**
    - **Choi가 "OK" 또는 "Can't Judge"를 줬으나 위험해 보이는 경우**:
        - Choi 기준(평균 변화 등)으로는 OK지만, L2(분포 분석)에서 "심각한 분포 변화(High Distribution Change)"가 감지되면 **"Warning (Check Distribution)"** 태그를 붙입니다.
    - **심층 원인 분석 (RCA)**:
        - Choi가 NOK를 줬을 때, 그 원인이 "단순 평균 이동"인지 "분포의 붕괴"인지 L2 결과를 통해 엔지니어에게 추가 정보를 제공합니다.

### **구현 시나리오**

```python
def combined_analysis(data):
    # 1. Choi Analysis 실행 (Fast, Deterministic)
    choi_result = choi_service.analyze(data)
    
    # 2. 결과가 애매하거나(Can't Judge), OK지만 통계적 확인이 필요한 경우 L2 실행
    if choi_result.status == "OK" or choi_result.status == "Can't Judge":
        l2_result = l2_service.analyze(data)
        
        # Choi는 OK지만 통계적으로 유의미한 악화가 있는 경우
        if l2_result.is_significant_degrade():
             choi_result.add_warning("Statistical Degrade Detected")
             
    return choi_result
```

## 5. 결론 및 제언

1.  **Choi 알고리즘을 Main("L1")으로 채택**: 운영자에게 명확한 Action Item(OK/NOK)을 주기 위해 Choi 알고리즘을 기본 판정 로직으로 사용합니다.
2.  **L2를 "Advanced Option"으로 유지**: L2 코드를 삭제하지 않고, 상세 분석 탭이나 "고급 분석" 기능으로 남겨두어 엔지니어가 필요 시 분포 변화를 확인할 수 있게 합니다.
3.  **단계적 통합**: 우선 Choi 알고리즘을 완벽하게 구현하여 배포하고, 이후 안정화 단계에서 L2의 통계적 경고를 Choi 결과의 보조 지표(Tooltip 등)로 통합하는 2단계 접근을 추천합니다.
