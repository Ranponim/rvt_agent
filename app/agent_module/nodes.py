import json
from datetime import datetime
from typing import Dict, Any, List
import logging
import traceback

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.agent_module.state import AgentState, ValidationResult
from app.models.kpi_data import PMData
from app.core.config import settings
from app.core.llm import llm

# 로거 설정
logger = logging.getLogger(__name__)

def parse_data_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node 1] 데이터 파싱 (Parse Data)
    입력된 JSON 데이터를 Pydantic 모델(PMData)로 변환하여 유효성을 검증합니다.
    
    Args:
        state (AgentState): 현재 에이전트 상태
        
    Returns:
        Dict: 업데이트된 상태 (parsed_data, logs)
    """
    logger.info("📡 [Node: Parse Data] 시작")
    parsed = None
    
    raw_data = state.get("current_data_15min")
    if raw_data:
        try:
            # 딕셔너리에서 PMData 객체 생성 (Validation 수행)
            parsed = PMData(**raw_data)
            logger.info(f"✅ 데이터 파싱 성공. KPI 수: {len(parsed.kpi)}")
            return {"parsed_data": parsed, "logs": ["✅ 데이터 파싱 성공."]}
        except Exception as e:
            error_msg = f"❌ 데이터 파싱 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"logs": [error_msg]}
    
    logger.warning("⚠️ 파싱할 데이터가 없습니다 (Missing current_data_15min)")
    return {"logs": ["⚠️ 파싱할 데이터가 없습니다."]}


def analyze_kpi_node(state: AgentState) -> Dict[str, Any]:
    """
    [Node 2] KPI 분석 (Analyze KPI)
    Choi 알고리즘을 활용하여 KPI 데이터의 이상 징후를 감지합니다.
    L2/L3 분석 로직을 통합하여 수행합니다.
    
    Args:
        state (AgentState): 파싱된 데이터가 포함된 상태
        
    Returns:
        Dict: 감지된 이상 징후 리스트 (anomalies)
    """
    logger.info("🔍 [Node: Analyze KPI] 시작")
    
    # 상태 키 디버깅 (필요 시 주석 해제)
    # logger.debug(f"DEBUG_NODE: keys in state: {list(state.keys())}")
    
    data: PMData = state.get("parsed_data")
    if not data:
        logger.error("🛑 [Analyze KPI] parsed_data가 누락되었습니다. 분석을 중단합니다.")
        return {"next_step": "end", "logs": ["🛑 데이터 누락 분석 중단"]}

    anomalies: List[ValidationResult] = []
    
    # --- Choi Algorithm Integration ---
    from app.services.choi_strategy_factory import get_choi_strategy_factory
    from app.models.judgement import PegSampleSeries, FilteringResult, JudgementType
    
    try:
        logger.info("🛠️ Choi 알고리즘 전략 팩토리 초기화 중...")
        factory = get_choi_strategy_factory()
        judgement_strategy = factory.create_judgement_strategy()
        choi_config = factory._get_judgement_config_dict()
        
        # 1. 데이터 준비 (Convert Agent State to Choi Input)
        filtered_data = {}
        
        # Pre Data (Baseline - 1hour avg)
        pre_map = {}
        if state.get("current_data_1hour"):
            pre_pm = PMData(**state["current_data_1hour"])
            for item in pre_pm.kpi:
                pre_map[item.peg_name.split("(")[0]] = item.avg
            logger.debug(f"📊 Pre-Data 로드 완료: {len(pre_map)} 항목")
                
        # Post Data (Current)
        post_map = {}
        for item in data.kpi:
             post_map[item.peg_name.split("(")[0]] = item.avg
        logger.debug(f"📊 Post-Data 로드 완료: {len(post_map)} 항목")
             
        # Config에 정의된 KPI만 추출하여 Series 생성
        all_topics = choi_config.get("kpi_definitions", {})
        processed_kpis = set()
        
        for topic, definition in all_topics.items():
            # Main KPI 추출
            main_kpi = definition.get("main")
            if main_kpi and main_kpi not in processed_kpis:
                filtered_data[main_kpi] = [PegSampleSeries(
                    peg_name=main_kpi,
                    cell_id=data.cell_id if hasattr(data, 'cell_id') else "unknown",
                    pre_samples=[pre_map.get(main_kpi, 0.0)] if main_kpi in pre_map else [],
                    post_samples=[post_map.get(main_kpi, 0.0)] if main_kpi in post_map else [],
                    unit="unit"
                )]
                processed_kpis.add(main_kpi)
                
            # Sub KPIs 추출
            for sub in definition.get("subs", []):
                if sub not in processed_kpis:
                    filtered_data[sub] = [PegSampleSeries(
                        peg_name=sub,
                        cell_id=data.cell_id if hasattr(data, 'cell_id') else "unknown",
                        pre_samples=[pre_map.get(sub, 0.0)] if sub in pre_map else [],
                        post_samples=[post_map.get(sub, 0.0)] if sub in post_map else [],
                        unit="unit"
                    )]
                    processed_kpis.add(sub)
        
        logger.info(f"📦 분석 대상 데이터 구성 완료: {len(filtered_data)} KPIs")
        
        # 2. Choi Judging (알고리즘 수행)
        dummy_filter = FilteringResult(valid_time_slots={}, filter_ratio=0.0) # 필터링은 이미 수행되었다고 가정
        result = judgement_strategy.apply(filtered_data, dummy_filter, choi_config)
        
        # 3. Result Processing (결과 처리)
        kpi_judgements = result.get("kpi_judgement", {})
        
        logger.info(f"🧠 알고리즘 판정 완료. 판정 항목 수: {len(kpi_judgements)}")
        
        for topic, res in kpi_judgements.items():
            if res.final_result != JudgementType.OK:
                # 이상 징후 발견 (Anomaly Found)
                logger.warning(f"🚨 이상 징후 감지: {topic} - {res.final_result.value}")
                anomalies.append({
                    "is_anomaly": True,
                    "severity": "P1" if res.final_result == JudgementType.NOK else "P2",
                    "title": f"KPI Anomaly: {topic}",
                    "description": f"Judgement: {res.final_result.value}. {res.summary_text}",
                    "related_kpis": [res.main_kpi_name] + [s['kpi_name'] for s in res.sub_results],
                    "root_cause": "Choi Algorithm Analysis",
                    "action_plan": "Check related KPIs"
                })

    except Exception as e:
        logger.error(f"❌ Choi 알고리즘 수행 중 치명적 오류 발생: {str(e)}", exc_info=True)
        anomalies.append({
             "is_anomaly": True,
             "severity": "P3",
             "title": "Algorithm Error",
             "description": f"Choi Algorithm Failed: {str(e)}",
             "related_kpis": [],
             "root_cause": "System Error",
             "action_plan": "Debug Logic"
        })
    
    log_msg = f"🔍 Choi Algorithm Analyzed. Found {len(anomalies)} anomalies."
    logger.info(log_msg)
    return {"anomalies": anomalies, "logs": [log_msg]}





