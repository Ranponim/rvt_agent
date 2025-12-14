# [KPI Name] Analysis Guide

## 1. Overview
- **Definition**: [Simple explanation of what this KPI means]
- **Target Value**: > 98.5%

## 2. Common Failure Causes
### Cause A: [Name, e.g., Congestion]
- **Symptom**: High PRB usage, Low Thpt
- **Related Counters**: `pmPrbUsageUl`, `pmPrbUsageDl`

### Cause B: [Name, e.g., Interference]
- **Symptom**: High RSSI, Low CQI
- **Related Counters**: `pmRadioRecInterferencePwr`

## 3. Troubleshooting Action
1. Check Alarm: [Specific Alarm IDs]
2. Parameter Check:
   - `sMeasure` (Default: 30) -> If too high, handover might fail.
