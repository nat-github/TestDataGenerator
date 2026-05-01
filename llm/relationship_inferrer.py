"""
llm.relationship_inferrer — Infer FK/PK relationships between tables using Claude.

Usage
-----
    from llm.relationship_inferrer import RelationshipInferrer
    from utils.config_parser import ConfigParser

    parser = ConfigParser("config/my_config.xlsx")
    parser.load_config(); parser.parse_tables()

    inferrer = RelationshipInferrer()
    results = inferrer.infer(parser.tables)

    for rel in results.relationships:
        print(rel.source_table, rel.source_column, "->", rel.target_table, rel.target_column)
        print("  confidence:", rel.llm_confidence, "reason:", results.reasons[rel.name])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.config_models import RelationshipConfig, TableConfig
from llm.client import get_client, DEFAULT_MODEL, system_prompt

logger = logging.getLogger(__name__)

# Minimum confidence to include an inferred relationship in results
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class InferenceResult:
    """Results of a single LLM relationship-inference run."""
    relationships: List[RelationshipConfig] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)   # rel.name -> explanation
    raw_response: Optional[str] = None
    model: str = DEFAULT_MODEL
    input_tables: int = 0
    skipped_tables: List[str] = field(default_factory=list)


class RelationshipInferrer:
    """
    Use Claude to infer FK/PK relationships from table schemas.

    Claude is given the table names + column names + data types for all tables
    and asked to return a JSON list of candidate relationships with confidence
    scores and reasoning.  Only relationships with confidence >= threshold are
    returned.

    The call uses a stable system prompt eligible for Anthropic prompt caching
    (>1024 tokens) to reduce cost on repeated runs.
    """

    # Maximum columns to include per table in the prompt — keeps token count bounded.
    MAX_COLUMNS_PER_TABLE = 60

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._client = get_client(api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(
        self,
        tables: Dict[str, TableConfig],
        existing_relationships: Optional[List[RelationshipConfig]] = None,
    ) -> InferenceResult:
        """
        Infer relationships for the given tables.

        Parameters
        ----------
        tables:
            Dict of table_name -> TableConfig (from ConfigParser.parse_tables())
        existing_relationships:
            Known relationships to exclude from suggestions (avoid duplicates).

        Returns
        -------
        InferenceResult with inferred RelationshipConfig objects marked
        inferred_by_llm=True and llm_confidence set.
        """
        active_tables = {name: cfg for name, cfg in tables.items() if cfg.active}
        if not active_tables:
            logger.warning("No active tables found — nothing to infer relationships for")
            return InferenceResult(input_tables=0)

        schema_text = self._build_schema_text(active_tables)
        exclusions = self._build_exclusions(existing_relationships or [])

        logger.info(
            "Sending %d table schema(s) to Claude for relationship inference...",
            len(active_tables),
        )

        raw = self._call_llm(schema_text, exclusions)
        result = self._parse_response(raw, active_tables)
        result.input_tables = len(active_tables)
        result.model = self.model

        logger.info(
            "LLM inferred %d relationship(s) above confidence threshold %.2f",
            len(result.relationships),
            self.confidence_threshold,
        )
        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_schema_text(self, tables: Dict[str, TableConfig]) -> str:
        lines: List[str] = ["## Table Schemas\n"]
        for table_name, cfg in tables.items():
            lines.append(f"### Table: {table_name}")
            cols = cfg.columns[: self.MAX_COLUMNS_PER_TABLE]
            for col in cols:
                flags: List[str] = []
                if col.is_pk:
                    flags.append("PK")
                if col.is_fk:
                    flags.append("FK")
                if col.nullable is False:
                    flags.append("NOT NULL")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                lines.append(f"  - {col.column_name}: {col.data_type}{flag_str}")
            if len(cfg.columns) > self.MAX_COLUMNS_PER_TABLE:
                lines.append(f"  ... ({len(cfg.columns) - self.MAX_COLUMNS_PER_TABLE} more columns omitted)")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_exclusions(existing: List[RelationshipConfig]) -> str:
        if not existing:
            return "None — no relationships are currently configured."
        lines = ["Already-configured relationships (do NOT re-suggest these):"]
        for rel in existing:
            lines.append(
                f"  - {rel.source_table}.{rel.source_column} -> {rel.target_table}.{rel.target_column}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    def _call_llm(self, schema_text: str, exclusions: str) -> str:
        user_prompt = f"""\
You are analysing a database schema to discover foreign-key relationships.

{schema_text}

## Already-configured relationships
{exclusions}

## Your Task

Examine the table schemas above and identify columns that are likely foreign keys
pointing to primary keys in other tables.  Look for:
- Matching or near-matching column names across tables (e.g. account_id in a child
  table pointing to id in the accounts table).
- Columns whose names end in _id, _code, _key, _num, _nr, or _no and match a PK
  column in another table.
- Naming conventions such as <table>_id suggesting a reference to <table>.id.
- Shared business identifiers (e.g. customer_number, product_code).

Respond ONLY with a JSON array.  Each element must have these exact keys:
{{
  "source_table": "<child table name>",
  "source_column": "<FK column name in child table>",
  "target_table": "<parent table name>",
  "target_column": "<PK/unique column name in parent table>",
  "relationship_type": "many_to_one",
  "confidence": <float 0.0-1.0>,
  "reason": "<one concise sentence explaining why>"
}}

Rules:
- Only suggest relationships where both source_table AND target_table are in the schema above.
- Do NOT suggest a relationship if it is already in the "already-configured" list.
- Set confidence >= 0.9 only when column names are an exact or near-exact match to a PK.
- Set confidence 0.7-0.89 when the naming pattern strongly suggests a foreign key.
- Set confidence 0.5-0.69 when it is plausible but uncertain.
- Do NOT suggest relationships with confidence < 0.5.
- Return an empty array [] if you find no relationships.
- Do NOT include any text outside the JSON array.
"""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt(),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text if response.content else "[]"
        return content.strip()

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(
        self, raw: str, tables: Dict[str, TableConfig]
    ) -> InferenceResult:
        result = InferenceResult(raw_response=raw)

        # Strip markdown code fences if present
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )

        try:
            candidates: List[Dict[str, Any]] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON for relationship inference: %s", exc)
            logger.debug("Raw response: %s", raw)
            return result

        table_column_index: Dict[str, set] = {
            name: {col.column_name for col in cfg.columns}
            for name, cfg in tables.items()
        }

        for idx, item in enumerate(candidates):
            try:
                src_table = str(item.get("source_table", "")).strip().lower()
                src_col = str(item.get("source_column", "")).strip()
                tgt_table = str(item.get("target_table", "")).strip().lower()
                tgt_col = str(item.get("target_column", "")).strip()
                confidence = float(item.get("confidence", 0.0))
                reason = str(item.get("reason", ""))
                rel_type = str(item.get("relationship_type", "many_to_one"))

                # Skip low-confidence
                if confidence < self.confidence_threshold:
                    continue

                # Validate tables and columns exist in schema
                if src_table not in table_column_index:
                    logger.warning("LLM suggested unknown source table '%s' — skipping", src_table)
                    result.skipped_tables.append(src_table)
                    continue
                if tgt_table not in table_column_index:
                    logger.warning("LLM suggested unknown target table '%s' — skipping", tgt_table)
                    result.skipped_tables.append(tgt_table)
                    continue
                if src_col not in table_column_index[src_table]:
                    logger.warning(
                        "LLM suggested unknown column '%s.%s' — skipping", src_table, src_col
                    )
                    continue
                if tgt_col not in table_column_index[tgt_table]:
                    logger.warning(
                        "LLM suggested unknown column '%s.%s' — skipping", tgt_table, tgt_col
                    )
                    continue

                rel_name = f"llm_{src_table}_{src_col}_to_{tgt_table}_{tgt_col}"
                rel = RelationshipConfig(
                    name=rel_name,
                    source_table=src_table,
                    source_column=src_col,
                    target_table=tgt_table,
                    target_column=tgt_col,
                    relationship_type=rel_type,
                    inferred_by_llm=True,
                    llm_confidence=round(confidence, 3),
                )
                result.relationships.append(rel)
                result.reasons[rel_name] = reason

            except Exception as exc:
                logger.warning("Could not parse LLM relationship candidate %d: %s", idx, exc)

        # Sort by confidence descending
        result.relationships.sort(key=lambda r: r.llm_confidence or 0, reverse=True)
        return result

    # ------------------------------------------------------------------
    # Convenience: pretty print results
    # ------------------------------------------------------------------
    @staticmethod
    def format_results(result: InferenceResult) -> str:
        if not result.relationships:
            return "[LLM Inference] No relationships found above confidence threshold."

        lines = [
            f"[LLM Inference] Found {len(result.relationships)} relationship(s) "
            f"(model: {result.model}, tables analysed: {result.input_tables})",
            "-" * 72,
        ]
        for rel in result.relationships:
            conf_pct = f"{(rel.llm_confidence or 0) * 100:.0f}%"
            lines.append(
                f"  {conf_pct}  {rel.source_table}.{rel.source_column}"
                f"  ->  {rel.target_table}.{rel.target_column}"
            )
            reason = result.reasons.get(rel.name, "")
            if reason:
                lines.append(f"         Reason: {reason}")
        return "\n".join(lines)
