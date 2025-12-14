from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class VersionDesc(BaseModel):
    """Holds version descriptions. 
    Can be single key (15min) or multiple keys (1hour)."""
    # Using Dict because keys are dynamic timestamps (e.g., "20251212_0900+0000")
    # and values are version strings.
    versions: Dict[str, str] 

    class Config:
        extra = "allow"

class KPIItem(BaseModel):
    cell: str
    family_name: str
    units: str  # JSON string like "{'mcID': 0}"
    peg_name: str
    avg: float

class PMData(BaseModel):
    """Represents the root structure of 15min.json or 1hour.json"""
    version_desc: Dict[str, str]
    kpi: List[KPIItem]
