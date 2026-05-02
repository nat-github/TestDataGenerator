from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import math

from pydantic import BaseModel, Field, field_validator, model_validator


class DataType(str, Enum):
    N38 = "N38"
    VA1 = "VA1"
    VA3 = "VA3"
    VA256 = "VA256"
    DC = "DC"
    D = "D"
    DT = "DT"
    TS = "TS"
    A1 = "A1"
    A5 = "A5"
    NS = "NS"   # Numeric String
    AN = "AN"   # Alphanumeric


def _is_nan(v: Any) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


class ColumnConfig(BaseModel):
    table_name: str
    column_name: str
    data_type: str
    is_pk: bool = False
    is_fk: bool = False
    ref_table: Optional[str] = None
    ref_column: Optional[str] = None
    business_values: Optional[str] = None
    special_rules: Optional[str] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    is_business_key_component: bool = False
    event_time: bool = False
    partition_role: Optional[str] = None
    scd2_tracked: bool = False

    @field_validator("precision", "scale", "length", mode="before")
    @classmethod
    def coerce_optional_int(cls, v: Any) -> Optional[int]:
        if v is None or _is_nan(v):
            return None
        return v

    @field_validator("ref_table", "ref_column", "business_values", "special_rules", mode="before")
    @classmethod
    def coerce_optional_str(cls, v: Any) -> Optional[str]:
        if v is None or _is_nan(v):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("min_value", "max_value", mode="before")
    @classmethod
    def coerce_min_max(cls, v: Any) -> Optional[Any]:
        if v is None or _is_nan(v):
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return s  # date/datetime string — pass through for _coerce_safe_timestamp
        return v  # datetime objects from Excel pass through unchanged


class TableConfig(BaseModel):
    name: str
    columns: List[ColumnConfig]
    num_rows: int = 1000
    table_kind: str = "transactional"
    description: Optional[str] = None
    generation_mode: str = "snapshot"
    business_key_columns: List[str] = Field(default_factory=list)
    primary_key_columns: List[str] = Field(default_factory=list)
    partition_enabled: bool = False
    partition_columns: List[str] = Field(default_factory=list)
    event_time_column: Optional[str] = None
    scd2_enabled: bool = False
    scd2_tracked_columns: List[str] = Field(default_factory=list)
    delta_eligible: bool = False
    active: bool = True
    notes: Optional[str] = None
    seed: Optional[int] = None  # per-table generation seed

    @field_validator(
        "business_key_columns",
        "primary_key_columns",
        "partition_columns",
        "scd2_tracked_columns",
        mode="before",
    )
    @classmethod
    def normalize_list_fields(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [part.strip() for part in v.replace(",", ";").split(";") if part.strip()]
        return [str(v).strip()]


class RelationshipConfig(BaseModel):
    name: Optional[str] = None
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    # "one_to_many" means one parent row → many child rows (FK on child side)
    relationship_type: str = "one_to_many"
    preserve_on_delta: bool = False
    active: bool = True
    notes: Optional[str] = None
    inferred_by_llm: bool = False      # True when Claude inferred this relationship
    llm_confidence: Optional[float] = None  # 0-1 confidence from LLM inference


class GenerationConfig(BaseModel):
    output_dir: str = "output"
    default_records: int = 5000
    table_records: Dict[str, int] = Field(default_factory=dict)
    validate_relationships: bool = True
    seed: Optional[int] = None   # global generation seed
