import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.services.choi_strategy_factory import get_choi_strategy_factory
from app.models.judgement import PegSampleSeries, FilteringResult, JudgementType
# from app.services.choi_judgement_service import ChoiJudgementService

def create_sample_series(peg_name, pre_val, post_val, count=10):
    return [PegSampleSeries(
        peg_name=peg_name,
        cell_id="cell_01",
        pre_samples=[float(pre_val)] * count,
        post_samples=[float(post_val)] * count,
        unit="Mbps"
    )]

def run_debug():
    print("Starting Debug Run...")
    os.environ["CHOI_CONFIG_FILE"] = "config/choi_algorithm.yml"
    
    factory = get_choi_strategy_factory()
    factory.reload_configuration()
    
    judgement_strategy = factory.create_judgement_strategy()
    config = factory._get_judgement_config_dict()
    
    print("Config KPI Definitions Keys:", config.get("kpi_definitions").keys())
    
    # Pre=1000, Post=1000 => Delta 0% => Similar
    filtered_data = {
        "AirMacDLThruAvg": create_sample_series("AirMacDLThruAvg", 1000, 1000),
        "ConnNoAvg": create_sample_series("ConnNoAvg", 100, 101) 
    }
    
    print("Filtered Data Keys:", filtered_data.keys())
    
    dummy_filtering_result = FilteringResult(valid_time_slots={}, filter_ratio=0.0)
    
    try:
        result = judgement_strategy.apply(filtered_data, dummy_filtering_result, config)
        print("Result Keys:", result.keys())
        if "kpi_judgement" in result:
             print("KPI Judgement Keys:", result["kpi_judgement"].keys())
             if "Air_MAC_DL_Thru" in result["kpi_judgement"]:
                 res = result["kpi_judgement"]["Air_MAC_DL_Thru"]
                 print(f"Result: {res.final_result}, Detail: {res.main_decision.detail}")
                 for sub in res.sub_results:
                     print(f"Sub: {sub.kpi_name}, Judge: {sub.judgement}, Detail: {sub.detail}")
             else:
                 print("Air_MAC_DL_Thru NOT found in judgement.")
        else:
            print("No kpi_judgement in result.")
            
    except Exception as e:
        print(f"Exception during apply: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
