from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import math

from pydantic import BaseModel, Field, field_validator


# Operators allowed inside a rule's `when:` condition.
# `and`/`or` are composite operators that nest further conditions.
RULE_OPERATORS = {
    "eq", "ne", "in", "not_in",
    "gt", "gte", "lt", "lte", "between",
    "is_null", "not_null", "matches",
    "and", "or",
}


class RuleAction(BaseModel):
    """What happens when a rule's `when:` condition matches.

    Any subset of these fields may be set; they override the column's
    base configuration for the matching row.
    """
    value: Optional[Any] = None              # constant value (use sentinel handling for null)
    min: Optional[Any] = None                # numeric / date min override
    max: Optional[Any] = None                # numeric / date max override
    distribution: Optional[Dict[str, Any]] = None
    null_rate: Optional[float] = None
    business_values: Optional[Any] = None    # list or "A;B;C"
    special_rules: Optional[str] = None
    set_null: bool = False                   # explicit null (preferred over value: null)


class RuleConfig(BaseModel):
    """A when/then rule applied to a column at row evaluation time.

    `when` is a nested dict where keys are either column names (mapped to
    operator dicts) or composite operators (`and`/`or` -> list of sub-conditions).
    Examples:
        when: { status: { eq: CLOSED } }
        when: { and: [ { status: { eq: ACTIVE } }, { tier: { in: [GOLD, PLATINUM] } } ] }
    """
    when: Dict[str, Any] = Field(default_factory=dict)
    then: RuleAction = Field(default_factory=RuleAction)
    note: Optional[str] = None


class WorkflowTransition(BaseModel):
    """One edge of a lifecycle state machine.

    ``from`` is a Python keyword, so the field is ``from_state`` with an
    alias — configs keep the natural ``{from: PLACED, to: SHIPPED}`` shape.
    """
    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")

    #: Chance of taking this edge, given the row is in ``from_state``.
    probability: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class WorkflowConfig(BaseModel):
    """Layer C — a state machine describing a row's lifecycle.

    Random generation draws each column independently, which happily
    produces a CANCELLED order carrying a delivery timestamp. A workflow
    instead walks the row through legal transitions and back-fills only the
    timestamps for states actually visited, in increasing order.

    ``timestamps`` maps *column name → state*: the column receives a value
    when that state is visited, and NULL when it is not.
    """
    name: str
    table: str
    state_column: str

    #: column → state that produces it
    timestamps: Dict[str, str] = Field(default_factory=dict)

    transitions: List[WorkflowTransition] = Field(default_factory=list)

    #: Where every row starts. Inferred when exactly one state is never a
    #: transition target; required otherwise.
    start_state: Optional[str] = None

    #: Window for the first timestamp, as ``[min, max]`` dates. Defaults to
    #: the last year when omitted.
    start_between: Optional[List[str]] = None

    #: Gap between consecutive state timestamps, ``[min_hours, max_hours]``.
    step_hours: List[float] = Field(default_factory=lambda: [1.0, 72.0])

    note: Optional[str] = None

    @property
    def states(self) -> List[str]:
        """Every state named by a transition, in first-seen order."""
        seen: List[str] = []
        for transition in self.transitions:
            for state in (transition.from_state, transition.to_state):
                if state not in seen:
                    seen.append(state)
        return seen

    def resolve_start_state(self) -> Optional[str]:
        """The declared start state, or the only state nothing transitions to."""
        if self.start_state:
            return self.start_state
        targets = {t.to_state for t in self.transitions}
        sources = [t.from_state for t in self.transitions]
        roots = [s for s in dict.fromkeys(sources) if s not in targets]
        return roots[0] if len(roots) == 1 else None


class CDCConfig(BaseModel):
    """Unified change-data-capture config, replacing the spread of
    delta_eligible / scd2_enabled / scd2_tracked_columns / partition_*
    fields on TableConfig.

    The legacy fields remain on TableConfig for backward compatibility;
    the parser fills both so downstream code keeps working unchanged.
    """
    mode: Literal["snapshot", "delta", "scd2"] = "snapshot"
    track: List[str] = Field(default_factory=list)         # tracked columns when mode=scd2
    event_time: Optional[str] = None                       # column used for monotonic ordering
    partition_by: List[str] = Field(default_factory=list)  # partition key columns

    @field_validator("track", "partition_by", mode="before")
    @classmethod
    def _normalize_lists(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [part.strip() for part in v.replace(",", ";").split(";") if part.strip()]
        return [str(v).strip()]


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
    # Statistical distribution descriptor from AutoConfigInferrer / manual config.
    # Format: {name, params, data_min, data_max} — used by generate_column_batch().
    distribution: Optional[Dict[str, Any]] = None
    # Canonical example value — used by wire-mock stub serialisers and API doc generators.
    example_value: Optional[Any] = None
    # Layer A: same-row when/then rules. Each rule rewrites this column's value
    # for rows whose `when:` condition matches. Evaluated post-generation.
    rules: Optional[List[RuleConfig]] = None
    # Layer B: same-row derived expression. When set, this column is computed
    # from other columns instead of randomly generated. Supports template
    # strings ("{first} {last}") and =-prefixed expressions ("={qty} * {price}").
    derived: Optional[str] = None

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
    # Output format for this table. "parquet" (default) | "json" | "wiremock"
    # Wire-mock serialiser is a future extension; field is present for forward compat.
    output_format: Optional[str] = None
    # Anchored generation: path to an existing dataset (.parquet / .csv) to load
    # verbatim instead of generating. The real rows are used to train SDV and to
    # anchor foreign keys — other tables generate around this fixed table.
    source: Optional[str] = None
    # Unified CDC config. When provided, the parser also back-fills the legacy
    # flat fields (delta_eligible, scd2_enabled, scd2_tracked_columns, etc.)
    # so downstream code keeps working unchanged.
    cdc: Optional[CDCConfig] = None

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
    inferred_by_ml: bool = False       # True when the heuristic ML inferrer produced this
    ml_confidence: Optional[float] = None   # 0-1 confidence from the ML inferrer
    inference_signals: Optional[Dict[str, float]] = None  # raw signal scores for audit / training


class GenerationConfig(BaseModel):
    output_dir: str = "output"
    default_records: int = 5000
    table_records: Dict[str, int] = Field(default_factory=dict)
    validate_relationships: bool = True
    seed: Optional[int] = None   # global generation seed
