"""
llm.schema_enricher — Use Claude to enrich bare column configs with realistic
generation rules, business values, data type suggestions, and null rates.

Usage
-----
    from llm.schema_enricher import SchemaEnricher
    from utils.config_parser import ConfigParser

    parser = ConfigParser("config/my_config.xlsx")
    parser.load_config(); parser.parse_tables()

    enricher = SchemaEnricher()
    enriched_yaml = enricher.enrich(parser.tables)

    with open("config/my_config_enriched.yaml", "w") as f:
        f.write(enriched_yaml)

CLI usage (via main.py):
    python main.py enrich --config config/bare.xlsx --output config/enriched.yaml
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from models.config_models import ColumnConfig, TableConfig
from llm.client import DEFAULT_MODEL, system_prompt
from llm.multi_provider import chat as llm_chat

logger = logging.getLogger(__name__)

# Columns to skip enrichment for (PKs and FKs are structurally determined)
_SKIP_ENRICHMENT_FLAGS = {"is_pk", "is_fk"}

# Maximum number of columns to send per LLM call (keep prompt bounded)
_MAX_COLS_PER_CALL = 80


@dataclass
class EnrichmentSuggestion:
    """Claude's suggestion for a single column."""
    table_name: str
    column_name: str
    suggested_business_values: Optional[str] = None   # semicolon-separated
    suggested_special_rules: Optional[str] = None      # e.g. REGEX:\d{4}
    suggested_data_type: Optional[str] = None          # e.g. VA256
    suggested_null_rate: Optional[float] = None        # 0.0 - 1.0
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class EnrichmentResult:
    """Results of a schema enrichment run."""
    suggestions: List[EnrichmentSuggestion] = field(default_factory=list)
    enriched_tables: Dict[str, TableConfig] = field(default_factory=dict)
    raw_response: Optional[str] = None
    model: Optional[str] = None
    columns_enriched: int = 0


class SchemaEnricher:
    """
    Ask Claude to suggest realistic generation rules for under-specified columns.

    Claude is given table + column context (name, data type, PK/FK flags, existing
    rules if any) and returns JSON suggestions per column that are then merged back
    into the TableConfig objects and optionally exported as YAML.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        confidence_threshold: float = 0.6,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._api_key = api_key
        self._provider = provider
        self._base_url = base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enrich(
        self,
        tables: Dict[str, TableConfig],
        output_yaml_path: Optional[str] = None,
    ) -> str:
        """
        Enrich table configs and return merged YAML string.

        Suggestions with confidence >= threshold are applied to a copy of the
        tables.  Columns that already have business_values or special_rules are
        still sent so Claude can improve them, but existing values are kept if
        Claude returns nothing.

        Parameters
        ----------
        tables:
            Dict of table_name -> TableConfig.
        output_yaml_path:
            If provided, write the enriched YAML to this file path.

        Returns
        -------
        YAML string of the enriched configuration.
        """
        active_tables = {name: cfg for name, cfg in tables.items() if cfg.active}
        if not active_tables:
            logger.warning("No active tables to enrich")
            return yaml.dump({"tables": []}, sort_keys=False)

        result = self._run_enrichment(active_tables)
        enriched_yaml = self._to_yaml(result.enriched_tables)

        if output_yaml_path:
            with open(output_yaml_path, "w", encoding="utf-8") as fh:
                fh.write(enriched_yaml)
            logger.info("Enriched config written to %s", output_yaml_path)

        logger.info(
            "Schema enrichment complete: %d column(s) updated (model: %s)",
            result.columns_enriched,
            self.model,
        )
        return enriched_yaml

    def get_suggestions(self, tables: Dict[str, TableConfig]) -> List[EnrichmentSuggestion]:
        """Return raw suggestions without applying them to the table configs."""
        active = {n: c for n, c in tables.items() if c.active}
        result = self._run_enrichment(active)
        return result.suggestions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_enrichment(self, tables: Dict[str, TableConfig]) -> EnrichmentResult:
        result = EnrichmentResult(model=self.model)

        import copy
        result.enriched_tables = {name: copy.deepcopy(cfg) for name, cfg in tables.items()}

        # Build column batch for LLM
        columns_for_enrichment = self._collect_enrichable_columns(tables)
        if not columns_for_enrichment:
            logger.info("No columns require enrichment")
            return result

        logger.info(
            "Sending %d column(s) across %d table(s) to Claude for schema enrichment...",
            len(columns_for_enrichment),
            len(tables),
        )

        schema_json = self._build_schema_json(columns_for_enrichment)
        raw = self._call_llm(schema_json, list(tables.keys()))
        result.raw_response = raw

        suggestions = self._parse_suggestions(raw, tables)
        result.suggestions = suggestions

        # Apply suggestions to enriched_tables
        for sug in suggestions:
            if sug.confidence < self.confidence_threshold:
                continue
            tbl = result.enriched_tables.get(sug.table_name)
            if tbl is None:
                continue
            for col in tbl.columns:
                if col.column_name != sug.column_name:
                    continue
                if sug.suggested_business_values and not col.business_values:
                    col.business_values = sug.suggested_business_values
                    result.columns_enriched += 1
                elif sug.suggested_special_rules and not col.special_rules:
                    col.special_rules = sug.suggested_special_rules
                    result.columns_enriched += 1
                break

        return result

    def _collect_enrichable_columns(
        self, tables: Dict[str, TableConfig]
    ) -> List[Dict[str, Any]]:
        """Collect columns that could benefit from enrichment."""
        cols: List[Dict[str, Any]] = []
        for table_name, cfg in tables.items():
            for col in cfg.columns:
                # Skip PKs and FKs — their values are structurally determined
                if col.is_pk or col.is_fk:
                    continue
                cols.append({
                    "table": table_name,
                    "column": col.column_name,
                    "data_type": col.data_type,
                    "nullable": col.nullable,
                    "current_business_values": col.business_values or "",
                    "current_special_rules": col.special_rules or "",
                    "min_value": col.min_value,
                    "max_value": col.max_value,
                })
                if len(cols) >= _MAX_COLS_PER_CALL:
                    break
            if len(cols) >= _MAX_COLS_PER_CALL:
                break
        return cols

    def _build_schema_json(self, columns: List[Dict[str, Any]]) -> str:
        return json.dumps(columns, indent=2, default=str)

    def _call_llm(self, schema_json: str, table_names: List[str]) -> str:
        tables_list = ", ".join(table_names)
        user_prompt = f"""\
You are enriching column generation rules for a synthetic test-data platform.

The tables in this schema are: {tables_list}

Below is a JSON array of columns that need generation rules.  Each column has:
- table, column: identifiers
- data_type: the declared type (N=numeric, VA=varchar, DC=decimal, D=date, DT=datetime, TS=timestamp, A=alpha, NS=numeric-string, AN=alphanumeric)
- current_business_values / current_special_rules: existing rules (may be empty)
- nullable, min_value, max_value: constraints

{schema_json}

## Your Task

For each column, suggest ONE of the following (choose the most appropriate):
1. **business_values**: a semicolon-separated list of realistic sample values
   (best for status codes, type enumerations, flags, country codes, currencies)
2. **special_rules**: a generation rule in one of these formats:
   - REGEX:<pattern>  e.g. REGEX:[A-Z]{{2}}[0-9]{{6}}
   - EMAIL, NAME, PHONE, ADDRESS, CITY, POSTCODE, IBAN, BIC, URL, UUID
3. **data_type_correction**: a corrected data type string (only if the declared type looks wrong)
4. **null_rate**: a float 0.0-1.0 (only if the column seems naturally sparse)

Return a JSON array where each element has:
{{
  "table": "<table name>",
  "column": "<column name>",
  "suggestion_type": "business_values" | "special_rules" | "data_type_correction" | "null_rate" | "none",
  "value": "<the suggested value as a string, or null if suggestion_type is 'none'>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Rules:
- Do NOT suggest business_values for free-text columns (names, addresses, descriptions).
- Do NOT suggest values you cannot confidently back with domain knowledge.
- Set confidence >= 0.9 only for well-known enumerations (ISO country codes, currency codes, boolean flags).
- Return suggestion_type = "none" with confidence = 0 if you have no useful suggestion.
- Do NOT include any text outside the JSON array.
"""
        content = llm_chat(
            messages=[{"role": "user", "content": user_prompt}],
            provider=self._provider,
            model=self.model,
            base_url=self._base_url,
            api_key=self._api_key,
            system=system_prompt(),
            max_tokens=8192,
        )
        return (content or "[]").strip()

    def _parse_suggestions(
        self, raw: str, tables: Dict[str, TableConfig]
    ) -> List[EnrichmentSuggestion]:
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            items: List[Dict[str, Any]] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON for schema enrichment: %s", exc)
            return []

        table_col_index: Dict[str, set] = {
            name: {col.column_name for col in cfg.columns}
            for name, cfg in tables.items()
        }

        suggestions: List[EnrichmentSuggestion] = []
        for item in items:
            table = str(item.get("table", "")).strip()
            column = str(item.get("column", "")).strip()
            stype = str(item.get("suggestion_type", "none")).strip()
            value = item.get("value")
            confidence = float(item.get("confidence", 0.0))
            reasoning = str(item.get("reasoning", ""))

            if stype == "none" or not value:
                continue
            if table not in table_col_index:
                continue
            if column not in table_col_index[table]:
                continue

            sug = EnrichmentSuggestion(
                table_name=table,
                column_name=column,
                confidence=confidence,
                reasoning=reasoning,
            )
            if stype == "business_values":
                sug.suggested_business_values = str(value)
            elif stype == "special_rules":
                sug.suggested_special_rules = str(value)
            elif stype == "data_type_correction":
                sug.suggested_data_type = str(value)
            elif stype == "null_rate":
                try:
                    sug.suggested_null_rate = float(value)
                except (TypeError, ValueError):
                    pass

            suggestions.append(sug)

        return suggestions

    # ------------------------------------------------------------------
    # YAML export
    # ------------------------------------------------------------------
    def _to_yaml(self, tables: Dict[str, TableConfig]) -> str:
        tables_list: List[Dict[str, Any]] = []
        for table_name, cfg in tables.items():
            cols: List[Dict[str, Any]] = []
            for col in cfg.columns:
                col_dict: Dict[str, Any] = {
                    "name": col.column_name,
                    "data_type": col.data_type,
                    "is_pk": col.is_pk,
                    "is_fk": col.is_fk,
                    "nullable": col.nullable,
                }
                if col.ref_table:
                    col_dict["ref_table"] = col.ref_table
                if col.ref_column:
                    col_dict["ref_column"] = col.ref_column
                if col.business_values:
                    col_dict["business_values"] = col.business_values
                if col.special_rules:
                    col_dict["special_rules"] = col.special_rules
                if col.min_value is not None:
                    col_dict["min_value"] = col.min_value
                if col.max_value is not None:
                    col_dict["max_value"] = col.max_value
                cols.append(col_dict)

            tbl_dict: Dict[str, Any] = {
                "name": table_name,
                "row_count": cfg.num_rows,
                "table_kind": cfg.table_kind,
                "generation_mode": cfg.generation_mode,
                "active": cfg.active,
                "columns": cols,
            }
            if cfg.business_key_columns:
                tbl_dict["business_key_columns"] = cfg.business_key_columns
            if cfg.primary_key_columns:
                tbl_dict["primary_key_columns"] = cfg.primary_key_columns
            if cfg.scd2_enabled:
                tbl_dict["scd2_enabled"] = cfg.scd2_enabled
                tbl_dict["scd2_tracked_columns"] = cfg.scd2_tracked_columns
            if cfg.delta_eligible:
                tbl_dict["delta_eligible"] = cfg.delta_eligible
            tables_list.append(tbl_dict)

        config: Dict[str, Any] = {
            "config_format": "sdp-yaml-v1",
            "tables": tables_list,
        }
        return yaml.dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False)

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    @staticmethod
    def format_suggestions(suggestions: List[EnrichmentSuggestion]) -> str:
        if not suggestions:
            return "[LLM Enrichment] No suggestions above confidence threshold."
        lines = [f"[LLM Enrichment] {len(suggestions)} suggestion(s):", "-" * 60]
        for sug in sorted(suggestions, key=lambda s: -s.confidence):
            conf = f"{sug.confidence * 100:.0f}%"
            val = (
                sug.suggested_business_values
                or sug.suggested_special_rules
                or sug.suggested_data_type
                or str(sug.suggested_null_rate)
            )
            lines.append(f"  {conf}  {sug.table_name}.{sug.column_name}: {val}")
            if sug.reasoning:
                lines.append(f"         {sug.reasoning}")
        return "\n".join(lines)
