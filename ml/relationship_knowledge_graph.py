"""Opt-in knowledge-graph-enhanced relationship inferrer.

This module is intentionally additive: it does not change the default
`MLRelationshipInferrer` behaviour. Callers must explicitly opt into the
knowledge-graph mode.

Why this exists
---------------
The baseline heuristic inferrer is strong for simple `customer_id -> customer_id`
patterns, but it can over-link generic parent keys such as `CODE` when multiple
lookup tables share similarly-shaped primary keys. A schema-level knowledge graph
helps disambiguate those cases by reasoning over:

* table semantics (`acct_type`, `merchant_category`, `terminal_type`, ...)
* child-column semantics (`CARD_AC_TP_CODE`, `CARD_TXN_MRCH_CAT_CODE`, ...)
* FK hints already present in the config (`is_fk=True`)
* lookup-table structure (small reference tables vs large fact tables)
* key ambiguity (many tables exposing a generic PK called `CODE`)
* length compatibility (`A4 -> A4`, `N3 -> N3`, ...)

The implementation reuses the existing ML inferrer for the heuristic baseline,
pattern memory, and classifier activation, then applies graph-aware candidate
filters and score adjustments on top.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

from ml.relationship_inferrer import MLInferenceResult, MLRelationshipInferrer
from ml.relationship_signals import name_similarity
from models.config_models import RelationshipConfig, TableConfig

_TOKEN_SPLIT = re.compile(r"[_\W]+|(?<=[a-z])(?=[A-Z])")

# These words are common schema noise and should not dominate semantic matching.
_ENTITY_STOPWORDS = {
    "id", "key", "code", "codes", "number", "num", "no", "ref",
    "fk", "pk", "tbl", "table", "row", "data", "record",
    "ebx", "dc", "dl", "hdr", "header", "clrg", "clearing",
    "card", "cards", "trx", "txnseq", "seq", "sqn",
}

# Normalisation map for common business / banking abbreviations.
_TOKEN_ALIASES = {
    "acct": "account",
    "ac": "account",
    "pd": "product",
    "tp": "type",
    "trns": "transaction",
    "txn": "transaction",
    "trmnl": "terminal",
    "mrch": "merchant",
    "cat": "category",
    "ccy": "currency",
    "curr": "currency",
    "issur": "issuer",
    "issuer": "issuer",
    "bnk": "bank",
    "identn": "identification",
    "identif": "identification",
    "ident": "identification",
    "inpt": "input",
    "orig": "original",
    "src": "source",
    "ntw": "network",
    "atm": "atm",
    "pos": "pos",
    "svc": "service",
    "iso": "iso",
    "isocurrencies": "currency",
    "currencies": "currency",
    "n3": "numericcode",
    "numericcode": "numericcode",
    "isocurrencycode": "isocurrencycode",
    "bindescription": "bankidentificationdescription",
    "bin": "bankidentificationnumber",
    "merchantcategory": "merchantcategory",
    "merchantgroup": "merchantgroup",
}

_GENERIC_KEY_NAMES = {
    "id", "key", "code", "number", "num", "no", "identifier", "identification",
}


@dataclass
class KGContext:
    child_tokens: Set[str] = field(default_factory=set)
    parent_table_tokens: Set[str] = field(default_factory=set)
    parent_descriptor_tokens: Set[str] = field(default_factory=set)
    table_semantic_similarity: float = 0.0
    descriptor_similarity: float = 0.0
    target_name_similarity: float = 0.0
    child_fk_prior: float = 0.0
    length_compatibility: float = 0.0
    parent_lookup_bonus: float = 0.0
    generic_parent_penalty: float = 0.0
    ambiguity_penalty: float = 0.0
    child_lookup_pk_penalty: float = 0.0


@dataclass
class KGScoredRelationship:
    relationship: RelationshipConfig
    reason: str


class SchemaKnowledgeGraph:
    """Very lightweight schema graph used for relationship disambiguation."""

    def __init__(self, tables: Dict[str, TableConfig]):
        self.tables = tables
        self._table_tokens: Dict[str, Set[str]] = {
            name: self._build_table_tokens(cfg) for name, cfg in tables.items()
        }
        self._descriptor_tokens: Dict[str, Set[str]] = {
            name: self._build_descriptor_tokens(cfg) for name, cfg in tables.items()
        }
        self._lookup_tables: Set[str] = {
            name for name, cfg in tables.items() if self._is_lookup_table(cfg)
        }
        self._tables_with_fk_hints: Set[str] = {
            name for name, cfg in tables.items() if any(getattr(col, "is_fk", False) for col in cfg.columns)
        }
        self._pk_name_to_tables: Dict[str, Set[str]] = {}
        for table_name, cfg in tables.items():
            for col in cfg.columns:
                if col.is_pk or col.column_name in (cfg.primary_key_columns or []):
                    self._pk_name_to_tables.setdefault(col.column_name.lower(), set()).add(table_name)

    def build_context(self, child_cfg, child_col, parent_cfg, parent_col) -> KGContext:
        child_tokens = self._column_tokens(child_col.column_name)
        parent_table_tokens = set(self._table_tokens.get(parent_cfg.name, set()))
        parent_descriptor_tokens = set(self._descriptor_tokens.get(parent_cfg.name, set()))

        table_similarity = self._semantic_similarity(child_tokens, parent_table_tokens)
        descriptor_similarity = self._semantic_similarity(child_tokens, parent_descriptor_tokens)
        target_name_similarity = name_similarity(
            child_col.column_name,
            f"{parent_cfg.name}_{parent_col.column_name}",
        )

        child_fk_prior = 1.0 if getattr(child_col, "is_fk", False) else self._weak_fk_prior(child_col)
        length_compatibility = self._length_compatibility(child_col, parent_col)
        parent_lookup_bonus = 1.0 if parent_cfg.name in self._lookup_tables else 0.35
        ambiguity_penalty = self._ambiguity_penalty(parent_col, table_similarity)
        generic_parent_penalty = self._generic_parent_penalty(parent_col, table_similarity)
        child_lookup_pk_penalty = 1.0 if (
            child_cfg.name in self._lookup_tables and child_col.is_pk and not getattr(child_col, "is_fk", False)
        ) else 0.0

        return KGContext(
            child_tokens=child_tokens,
            parent_table_tokens=parent_table_tokens,
            parent_descriptor_tokens=parent_descriptor_tokens,
            table_semantic_similarity=table_similarity,
            descriptor_similarity=descriptor_similarity,
            target_name_similarity=target_name_similarity,
            child_fk_prior=child_fk_prior,
            length_compatibility=length_compatibility,
            parent_lookup_bonus=parent_lookup_bonus,
            generic_parent_penalty=generic_parent_penalty,
            ambiguity_penalty=ambiguity_penalty,
            child_lookup_pk_penalty=child_lookup_pk_penalty,
        )

    def is_plausible_child_column(self, table_cfg: TableConfig, column) -> bool:
        if getattr(column, "is_fk", False):
            return True
        if table_cfg.name in self._tables_with_fk_hints:
            return False
        if column.is_pk and table_cfg.name in self._lookup_tables:
            return False
        lname = column.column_name.lower()
        return (
            lname.endswith("_id")
            or lname.endswith("_key")
            or lname.endswith("_code")
            or lname.endswith("_no")
            or lname in {"id", "key", "code"}
        ) and not column.is_pk

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        raw = _TOKEN_SPLIT.split(str(text))
        out: List[str] = []
        for token in raw:
            token = token.strip().lower()
            if not token:
                continue
            out.append(_TOKEN_ALIASES.get(token, token))
        return out

    def _column_tokens(self, name: str) -> Set[str]:
        return {
            token for token in self._tokenize(name)
            if token and token not in _ENTITY_STOPWORDS and token not in _GENERIC_KEY_NAMES
        }

    def _build_table_tokens(self, cfg: TableConfig) -> Set[str]:
        tokens = set(self._column_tokens(cfg.name))
        for col in cfg.columns:
            if col.is_pk and self._is_generic_key_name(col.column_name):
                continue
            tokens.update(self._column_tokens(col.column_name))
        return tokens

    def _build_descriptor_tokens(self, cfg: TableConfig) -> Set[str]:
        tokens: Set[str] = set()
        for col in cfg.columns:
            if col.is_pk:
                continue
            tokens.update(self._column_tokens(col.column_name))
        return tokens

    @staticmethod
    def _semantic_similarity(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right) / max(1, len(left))
        joined_left = " ".join(sorted(left))
        joined_right = " ".join(sorted(right))
        fuzzy = name_similarity(joined_left, joined_right)
        return round(max(overlap, fuzzy), 4)

    @staticmethod
    def _dtype_length(column) -> Optional[int]:
        if getattr(column, "length", None):
            try:
                return int(column.length)
            except Exception:
                return None
        dtype = str(getattr(column, "data_type", "") or "").upper()
        match = re.fullmatch(r"[A-Z]+(\d+)", dtype)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _length_compatibility(self, child_col, parent_col) -> float:
        child_len = self._dtype_length(child_col)
        parent_len = self._dtype_length(parent_col)
        if child_len is None or parent_len is None:
            return 0.5
        if child_len == parent_len:
            return 1.0
        # Precision-first bias: exact length matters for code/key domains.
        if abs(child_len - parent_len) == 1:
            return 0.35
        return 0.0

    def _weak_fk_prior(self, child_col) -> float:
        if getattr(child_col, "is_pk", False):
            return 0.0
        lname = child_col.column_name.lower()
        if any(lname.endswith(suffix) for suffix in ("_id", "_key", "_code", "_no")):
            return 0.65
        return 0.15

    @staticmethod
    def _is_generic_key_name(column_name: str) -> bool:
        tokens = SchemaKnowledgeGraph._tokenize(column_name)
        if not tokens:
            return False
        return len(tokens) == 1 and tokens[0] in _GENERIC_KEY_NAMES

    def _ambiguity_penalty(self, parent_col, semantic_match: float) -> float:
        siblings = self._pk_name_to_tables.get(parent_col.column_name.lower(), set())
        if len(siblings) <= 1:
            return 0.0
        if semantic_match >= 0.75:
            return 0.0
        base = min((len(siblings) - 1) / 5.0, 1.0)
        if semantic_match >= 0.5:
            return round(base * 0.35, 4)
        return round(base, 4)

    def _generic_parent_penalty(self, parent_col, semantic_match: float) -> float:
        if not self._is_generic_key_name(parent_col.column_name):
            return 0.0
        if semantic_match >= 0.7:
            return 0.0
        if semantic_match >= 0.45:
            return 0.25
        return 0.75

    @staticmethod
    def _is_lookup_table(cfg: TableConfig) -> bool:
        pk_count = sum(1 for col in cfg.columns if col.is_pk)
        if pk_count != 1:
            return False
        if len(cfg.columns) > 4:
            return False
        non_pk = [col for col in cfg.columns if not col.is_pk]
        if not non_pk:
            return False
        textish = sum(1 for col in non_pk if str(col.data_type).upper().startswith(("A", "V", "S", "C")))
        return textish >= max(1, len(non_pk) - 1)


class KnowledgeGraphRelationshipInferrer(MLRelationshipInferrer):
    """Schema-graph-enhanced inferrer.

    This class deliberately subclasses the existing ML inferrer so it keeps the
    same output type, feedback-store behaviour, and classifier activation.
    """

    def infer(
        self,
        tables: Dict[str, TableConfig],
        existing_relationships: Optional[List[RelationshipConfig]] = None,
        sample_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> MLInferenceResult:
        active = {n: cfg for n, cfg in tables.items() if cfg.active}
        result = MLInferenceResult(input_tables=len(active))
        if not active:
            return result

        self._fit_classifier(result)
        excluded = self._build_exclusions(existing_relationships or [])
        graph = SchemaKnowledgeGraph(active)
        result.relationships = self._collect_relationships(active, excluded, graph, sample_data, result)
        result.relationships = self._dedupe_top_per_child_column(result.relationships)
        return result

    def _fit_classifier(self, result: MLInferenceResult) -> None:
        self._classifier_report = self.classifier.fit(*self.feedback_store.training_data())
        result.classifier_fitted = self._classifier_report.fitted
        result.classifier_examples = self._classifier_report.n_examples

    def _collect_relationships(
        self,
        active: Dict[str, TableConfig],
        excluded: set,
        graph: SchemaKnowledgeGraph,
        sample_data: Optional[Dict[str, pd.DataFrame]],
        result: MLInferenceResult,
    ) -> List[RelationshipConfig]:
        relationships: List[RelationshipConfig] = []
        for child_name, child_cfg, parent_name, parent_cfg, child_col, parent_col in self._iter_candidates(
            active,
            excluded,
            graph,
            sample_data,
        ):
            scored = self._score_candidate(
                child_name,
                child_cfg,
                parent_name,
                parent_cfg,
                child_col,
                parent_col,
                graph,
                sample_data,
                result,
            )
            if scored is None:
                continue
            relationships.append(scored.relationship)
            if scored.relationship.name:
                result.reasons[scored.relationship.name] = scored.reason
        return relationships

    def _iter_candidates(
        self,
        active: Dict[str, TableConfig],
        excluded: set,
        graph: SchemaKnowledgeGraph,
        sample_data: Optional[Dict[str, pd.DataFrame]],
    ):
        for child_name, child_cfg in active.items():
            for parent_name, parent_cfg in self._iter_parent_tables(active, child_name):
                yield from self._iter_candidate_pairs(
                    child_name,
                    child_cfg,
                    parent_name,
                    parent_cfg,
                    excluded,
                    graph,
                    sample_data,
                )

    @staticmethod
    def _iter_parent_tables(active: Dict[str, TableConfig], child_name: str):
        for parent_name, parent_cfg in active.items():
            if child_name != parent_name:
                yield parent_name, parent_cfg

    def _iter_candidate_pairs(
        self,
        child_name: str,
        child_cfg: TableConfig,
        parent_name: str,
        parent_cfg: TableConfig,
        excluded: set,
        graph: SchemaKnowledgeGraph,
        sample_data: Optional[Dict[str, pd.DataFrame]],
    ):
        for child_col in child_cfg.columns:
            if not graph.is_plausible_child_column(child_cfg, child_col):
                continue
            for parent_col in parent_cfg.columns:
                if not parent_col.is_pk and not self._looks_like_pk(parent_col, parent_cfg, sample_data):
                    continue
                candidate = (child_name, child_col.column_name, parent_name, parent_col.column_name)
                if candidate in excluded:
                    continue
                yield child_name, child_cfg, parent_name, parent_cfg, child_col, parent_col

    def _score_candidate(
        self,
        child_name: str,
        child_cfg: TableConfig,
        parent_name: str,
        parent_cfg: TableConfig,
        child_col,
        parent_col,
        graph: SchemaKnowledgeGraph,
        sample_data: Optional[Dict[str, pd.DataFrame]],
        result: MLInferenceResult,
    ) -> Optional[KGScoredRelationship]:
        candidate = (child_name, child_col.column_name, parent_name, parent_col.column_name)
        signals = self._compute_signals(child_cfg, child_col, parent_cfg, parent_col, sample_data)
        if signals["type_compatibility"] <= 0.0:
            return None

        base_confidence, source = self._final_confidence(signals, candidate)
        kg = graph.build_context(child_cfg, child_col, parent_cfg, parent_col)
        if kg.ambiguity_penalty >= 0.5 and kg.table_semantic_similarity < 0.45:
            return None

        graph_score = self._graph_score(kg)
        penalty = self._graph_penalty(kg)
        confidence = min(max(0.35 * base_confidence + 0.65 * graph_score - penalty, 0.0), 1.0)
        rel_name = f"{child_name}_{child_col.column_name}_to_{parent_name}_{parent_col.column_name}"
        if confidence < self.confidence_threshold:
            result.skipped_reasons[rel_name] = (
                f"below threshold ({confidence:.2f} < {self.confidence_threshold:.2f}); "
                f"base={base_confidence:.2f}; graph={graph_score:.2f}; penalty={penalty:.2f}"
            )
            return None

        enriched_signals = self._with_kg_signals(signals, kg, base_confidence, graph_score, penalty)
        rel = RelationshipConfig(
            name=rel_name,
            source_table=child_name,
            source_column=child_col.column_name,
            target_table=parent_name,
            target_column=parent_col.column_name,
            relationship_type="many_to_one",
            inferred_by_ml=True,
            ml_confidence=round(confidence, 4),
            inference_signals={k: round(v, 4) for k, v in enriched_signals.items()},
        )
        return KGScoredRelationship(
            relationship=rel,
            reason=self._explain_kg(enriched_signals, source, candidate),
        )

    @staticmethod
    def _graph_score(kg: KGContext) -> float:
        return (
            0.34 * kg.table_semantic_similarity
            + 0.18 * kg.descriptor_similarity
            + 0.14 * kg.target_name_similarity
            + 0.14 * kg.child_fk_prior
            + 0.12 * kg.length_compatibility
            + 0.08 * kg.parent_lookup_bonus
        )

    @staticmethod
    def _graph_penalty(kg: KGContext) -> float:
        return (
            0.42 * kg.generic_parent_penalty
            + 0.33 * kg.ambiguity_penalty
            + 0.25 * kg.child_lookup_pk_penalty
        )

    @staticmethod
    def _with_kg_signals(
        signals: Dict[str, float],
        kg: KGContext,
        base_confidence: float,
        graph_score: float,
        penalty: float,
    ) -> Dict[str, float]:
        enriched = dict(signals)
        enriched.update({
            "kg_table_semantic_similarity": round(kg.table_semantic_similarity, 4),
            "kg_descriptor_similarity": round(kg.descriptor_similarity, 4),
            "kg_target_name_similarity": round(kg.target_name_similarity, 4),
            "kg_child_fk_prior": round(kg.child_fk_prior, 4),
            "kg_length_compatibility": round(kg.length_compatibility, 4),
            "kg_parent_lookup_bonus": round(kg.parent_lookup_bonus, 4),
            "kg_generic_parent_penalty": round(kg.generic_parent_penalty, 4),
            "kg_ambiguity_penalty": round(kg.ambiguity_penalty, 4),
            "kg_child_lookup_pk_penalty": round(kg.child_lookup_pk_penalty, 4),
            "kg_base_confidence": round(base_confidence, 4),
            "kg_graph_score": round(graph_score, 4),
            "kg_penalty": round(penalty, 4),
        })
        return enriched

    @staticmethod
    def _explain_kg(signals: Dict[str, float], source: str, candidate) -> str:
        child_table, child_col, parent_table, parent_col = candidate
        return " | ".join([
            f"{child_table}.{child_col} -> {parent_table}.{parent_col}",
            f"source={source}+kg",
            f"base={signals.get('kg_base_confidence', 0):.2f}",
            f"table_sem={signals.get('kg_table_semantic_similarity', 0):.2f}",
            f"descriptor_sem={signals.get('kg_descriptor_similarity', 0):.2f}",
            f"fk_prior={signals.get('kg_child_fk_prior', 0):.2f}",
            f"len_match={signals.get('kg_length_compatibility', 0):.2f}",
            f"penalty={signals.get('kg_penalty', 0):.2f}",
        ])

