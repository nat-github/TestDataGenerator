"""Opt-in knowledge-graph-enhanced relationship inferrer.

This module is intentionally additive: it does not change the default
`MLRelationshipInferrer` behaviour. Callers must explicitly opt into the
knowledge-graph mode (`--ml-mode knowledge-graph`).

What the knowledge graph is
---------------------------
A real `networkx.DiGraph` is built over the schema with four node kinds:

* **table** nodes        — one per table, flagged as lookup vs fact
* **column** nodes       — one per column, flagged is_pk / is_fk
* **entity** nodes       — a lightweight ontology layer; tables that describe
                           the same business concept point at a shared entity
* **candidate_fk** edges — child column -> parent column edges added while
                           scoring, so degree/hub reasoning becomes possible

Reliability mechanisms layered on top of the heuristic baseline
---------------------------------------------------------------
1. **Inclusion dependency (primary, fully domain-agnostic)** — when sample data
   is supplied, a real FK requires the child's value set to sit (almost)
   inside the parent's. A name match with no value containment is dropped.
2. **Hub / degree prior** — a parent key referenced by many child columns is a
   genuine dimension and gets a small boost.
3. **Global one-parent assignment** — each child FK column commits to exactly
   one parent (the highest-scoring), resolving generic `CODE`/`ID` ambiguity.
4. **FK-cycle resolution** — cycles among distinct tables are almost always
   wrong; the weakest edge in each cycle is dropped (self-references kept).
5. **Semantic disambiguation** — table/descriptor token similarity, backed by a
   *pluggable* `SemanticProfile` (see `ml/semantic_profile.py`) so the domain
   vocabulary is swappable and the graph machinery stays generic.

The implementation reuses the existing ML inferrer for the heuristic baseline,
pattern memory and classifier activation, then applies the graph-aware filters
and score adjustments on top. With no sample data it falls back to the
name/structure path, behaving exactly as before.
"""
from __future__ import annotations

import logging

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from sdp.ml.relationship_inferrer import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MLInferenceResult,
    MLRelationshipInferrer,
)
from sdp.ml.relationship_signals import name_similarity
from sdp.ml.semantic_profile import SemanticProfile, default_profile
from sdp.models.config_models import RelationshipConfig, TableConfig

logger = logging.getLogger(__name__)

_TOKEN_SPLIT = re.compile(r"[_\W]+|(?<=[a-z])(?=[A-Z])")

# Below this child-into-parent value containment, a candidate is rejected
# outright *when sample data is available* — the inclusion-dependency gate.
_INCLUSION_GATE = 0.30


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
    hub_bonus: float = 0.0


@dataclass
class KGScoredRelationship:
    relationship: RelationshipConfig
    reason: str


class SchemaKnowledgeGraph:
    """A lightweight schema knowledge graph used for relationship disambiguation.

    Backed by a real `networkx.DiGraph` so degree/hub reasoning, entity grouping
    and cycle detection use graph primitives rather than ad-hoc bookkeeping.
    """

    def __init__(self, tables: Dict[str, TableConfig], profile: Optional[SemanticProfile] = None):
        self.tables = tables
        self.profile = profile or default_profile()

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
            name for name, cfg in tables.items()
            if any(getattr(col, "is_fk", False) for col in cfg.columns)
        }
        self._pk_name_to_tables: Dict[str, Set[str]] = {}
        for table_name, cfg in tables.items():
            for col in cfg.columns:
                if col.is_pk or col.column_name in (cfg.primary_key_columns or []):
                    self._pk_name_to_tables.setdefault(
                        col.column_name.lower(), set()
                    ).add(table_name)

        # Distinct child columns that have a candidate edge into a parent key.
        self._parent_candidate_children: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}

        self.graph = nx.DiGraph()
        self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self) -> None:
        """Populate the networkx graph with table, column and entity nodes."""
        for tname, cfg in self.tables.items():
            tnode = ("table", tname)
            self.graph.add_node(tnode, kind="table", lookup=(tname in self._lookup_tables))
            for col in cfg.columns:
                cnode = ("column", tname, col.column_name)
                self.graph.add_node(
                    cnode,
                    kind="column",
                    is_pk=bool(col.is_pk),
                    is_fk=bool(getattr(col, "is_fk", False)),
                )
                self.graph.add_edge(tnode, cnode, kind="has_column")

            # Ontology layer: tables sharing a semantic signature describe the
            # same business entity. This is what makes it a *knowledge* graph
            # rather than a plain schema graph.
            signature = self._entity_signature(cfg)
            if signature:
                enode = ("entity", signature)
                self.graph.add_node(enode, kind="entity")
                self.graph.add_edge(tnode, enode, kind="represents")

    def _entity_signature(self, cfg: TableConfig) -> Optional[Tuple[str, ...]]:
        tokens = sorted(self._table_tokens.get(cfg.name, set()))
        return tuple(tokens) if tokens else None

    def register_candidate(
        self, child_table: str, child_col: str, parent_table: str, parent_col: str
    ) -> None:
        """Record a candidate FK edge so hub/degree reasoning can use it."""
        key = (parent_table, parent_col)
        self._parent_candidate_children.setdefault(key, set()).add((child_table, child_col))
        self.graph.add_edge(
            ("column", child_table, child_col),
            ("column", parent_table, parent_col),
            kind="candidate_fk",
        )

    def hub_bonus(self, parent_table: str, parent_col: str) -> float:
        """Small boost for a parent key referenced by multiple child columns.

        A key that many children point at is a real dimension/lookup. The boost
        is deliberately tiny (<= 0.03) so it only ever breaks near-ties — it can
        never override semantic evidence.
        """
        children = self._parent_candidate_children.get((parent_table, parent_col), set())
        if len(children) <= 1:
            return 0.0
        return round(min(0.03, 0.01 * (len(children) - 1)), 4)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
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
            child_cfg.name in self._lookup_tables
            and child_col.is_pk
            and not getattr(child_col, "is_fk", False)
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
            hub_bonus=self.hub_bonus(parent_cfg.name, parent_col.column_name),
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

    # ------------------------------------------------------------------
    # Tokenisation — driven by the pluggable SemanticProfile
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        raw = _TOKEN_SPLIT.split(str(text))
        out: List[str] = []
        for token in raw:
            token = token.strip().lower()
            if not token:
                continue
            out.append(self.profile.aliases.get(token, token))
        return out

    def _column_tokens(self, name: str) -> Set[str]:
        return {
            token for token in self._tokenize(name)
            if token
            and token not in self.profile.stopwords
            and token not in self.profile.generic_key_names
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

    def _is_generic_key_name(self, column_name: str) -> bool:
        tokens = self._tokenize(column_name)
        if not tokens:
            return False
        return len(tokens) == 1 and tokens[0] in self.profile.generic_key_names

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

    def _is_lookup_table(self, cfg: TableConfig) -> bool:
        pk_count = sum(1 for col in cfg.columns if col.is_pk)
        if pk_count != 1:
            return False
        if len(cfg.columns) > 4:
            return False
        non_pk = [col for col in cfg.columns if not col.is_pk]
        if not non_pk:
            return False
        textish = sum(
            1 for col in non_pk
            if str(col.data_type).upper().startswith(("A", "V", "S", "C"))
        )
        return textish >= max(1, len(non_pk) - 1)


class KnowledgeGraphRelationshipInferrer(MLRelationshipInferrer):
    """Schema-knowledge-graph-enhanced inferrer.

    Subclasses the existing ML inferrer so it keeps the same output type,
    feedback-store behaviour and classifier activation. The graph reliability
    layers are applied additively on top of the heuristic baseline.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        feedback_store=None,
        weights=None,
        profile: Optional[SemanticProfile] = None,
    ) -> None:
        super().__init__(
            confidence_threshold=confidence_threshold,
            feedback_store=feedback_store,
            weights=weights,
        )
        self.profile = profile or default_profile()

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
        graph = SchemaKnowledgeGraph(active, profile=self.profile)

        scored = self._collect_relationships(active, excluded, graph, sample_data, result)
        reason_by_name = {s.relationship.name: s.reason for s in scored}

        relationships = self._dedupe_top_per_child_column([s.relationship for s in scored])
        relationships = self._resolve_cycles(relationships, result)

        result.relationships = relationships
        result.reasons = {
            r.name: reason_by_name[r.name]
            for r in relationships
            if r.name in reason_by_name
        }
        return result

    def _fit_classifier(self, result: MLInferenceResult) -> None:
        self._classifier_report = self.classifier.fit(*self.feedback_store.training_data())
        result.classifier_fitted = self._classifier_report.fitted
        result.classifier_examples = self._classifier_report.n_examples

    # ------------------------------------------------------------------
    # Candidate collection — two passes so hub/degree can see every edge
    # ------------------------------------------------------------------
    def _collect_relationships(
        self,
        active: Dict[str, TableConfig],
        excluded: set,
        graph: SchemaKnowledgeGraph,
        sample_data: Optional[Dict[str, pd.DataFrame]],
        result: MLInferenceResult,
    ) -> List[KGScoredRelationship]:
        # Pass 1 — enumerate type-compatible candidates and register their edges
        # so the parent hub/degree counts are complete before scoring begins.
        pending: List[tuple] = []
        for child_name, child_cfg, parent_name, parent_cfg, child_col, parent_col in (
            self._iter_candidates(active, excluded, graph, sample_data)
        ):
            signals = self._compute_signals(child_cfg, child_col, parent_cfg, parent_col, sample_data)
            if signals["type_compatibility"] <= 0.0:
                continue
            graph.register_candidate(
                child_name, child_col.column_name, parent_name, parent_col.column_name
            )
            pending.append(
                (child_name, child_cfg, parent_name, parent_cfg, child_col, parent_col, signals)
            )

        # Pass 2 — score every candidate now that hub knowledge is available.
        scored: List[KGScoredRelationship] = []
        for (child_name, child_cfg, parent_name, parent_cfg, child_col, parent_col, signals) in pending:
            kg_rel = self._score_candidate(
                child_name, child_cfg, parent_name, parent_cfg,
                child_col, parent_col, signals, graph, sample_data, result,
            )
            if kg_rel is not None:
                scored.append(kg_rel)
        return scored

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
                    child_name, child_cfg, parent_name, parent_cfg,
                    excluded, graph, sample_data,
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

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_candidate(
        self,
        child_name: str,
        child_cfg: TableConfig,
        parent_name: str,
        parent_cfg: TableConfig,
        child_col,
        parent_col,
        signals: Dict[str, float],
        graph: SchemaKnowledgeGraph,
        sample_data: Optional[Dict[str, pd.DataFrame]],
        result: MLInferenceResult,
    ) -> Optional[KGScoredRelationship]:
        rel_name = (
            f"{child_name}_{child_col.column_name}_to_"
            f"{parent_name}_{parent_col.column_name}"
        )
        candidate = (child_name, child_col.column_name, parent_name, parent_col.column_name)

        base_confidence, source = self._final_confidence(signals, candidate)
        kg = graph.build_context(child_cfg, child_col, parent_cfg, parent_col)

        if kg.ambiguity_penalty >= 0.5 and kg.table_semantic_similarity < 0.45:
            result.skipped_reasons[rel_name] = (
                f"ambiguous parent key with weak semantics "
                f"(ambiguity={kg.ambiguity_penalty:.2f}, table_sem={kg.table_semantic_similarity:.2f})"
            )
            return None

        has_data = self._has_value_data(
            child_name, child_col.column_name, parent_name, parent_col.column_name, sample_data,
        )
        value_subset = float(signals.get("value_subset", 0.0))

        # Inclusion-dependency gate — the strongest, fully domain-agnostic FK
        # signal. When real data is present, a candidate whose child values are
        # not contained in the parent is almost certainly not a foreign key.
        if has_data and value_subset < _INCLUSION_GATE:
            result.skipped_reasons[rel_name] = (
                f"value inclusion too low ({value_subset:.2f} < {_INCLUSION_GATE:.2f}) "
                f"despite sample data — not a foreign key"
            )
            return None

        graph_score = self._graph_score(kg)
        penalty = self._graph_penalty(kg)

        if has_data:
            # Data-led fusion: inclusion dependency dominates, structure/name
            # support it. This is the high-reliability path.
            confidence = (
                0.55 * value_subset
                + 0.30 * graph_score
                + 0.15 * base_confidence
                - 0.50 * penalty
                + kg.hub_bonus
            )
        else:
            # Name/structure-only path — unchanged from the pre-graph behaviour
            # apart from the (tiny, non-negative) hub bonus.
            confidence = (
                0.35 * base_confidence
                + 0.65 * graph_score
                - penalty
                + kg.hub_bonus
            )
        confidence = min(max(confidence, 0.0), 1.0)

        if confidence < self.confidence_threshold:
            result.skipped_reasons[rel_name] = (
                f"below threshold ({confidence:.2f} < {self.confidence_threshold:.2f}); "
                f"base={base_confidence:.2f}; graph={graph_score:.2f}; "
                f"value_subset={value_subset:.2f}; penalty={penalty:.2f}"
            )
            return None

        enriched_signals = self._with_kg_signals(
            signals, kg, base_confidence, graph_score, penalty, value_subset, has_data,
        )
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

    # ------------------------------------------------------------------
    # FK-cycle resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_cycles(
        relationships: List[RelationshipConfig],
        result: MLInferenceResult,
    ) -> List[RelationshipConfig]:
        """Drop the weakest edge of every multi-table FK cycle.

        Self-references (a table pointing at its own PK) are legitimate
        hierarchies and are left untouched.
        """
        survivors = list(relationships)
        # Bounded loop — each iteration removes one edge.
        for _ in range(len(survivors) + 1):
            digraph = nx.DiGraph()
            for rel in survivors:
                if rel.source_table != rel.target_table:
                    digraph.add_edge(rel.source_table, rel.target_table)
            try:
                cycles = [c for c in nx.simple_cycles(digraph) if len(c) >= 2]
            except (nx.NetworkXError, RecursionError, MemoryError) as exc:
                # Cycle enumeration is exponential in the worst case. Bailing
                # out leaves the surviving relationships as-is, which is safe
                # — but silently returning a graph that may still contain
                # cycles is worth recording.
                logger.warning(
                    "Cycle detection aborted (%s: %s) — remaining relationships "
                    "were not checked for cycles", type(exc).__name__, exc,
                )
                break
            if not cycles:
                break
            cycle_tables = set(cycles[0])
            on_cycle = [
                rel for rel in survivors
                if rel.source_table != rel.target_table
                and rel.source_table in cycle_tables
                and rel.target_table in cycle_tables
            ]
            if not on_cycle:
                break
            weakest = min(on_cycle, key=lambda r: r.ml_confidence or 0.0)
            survivors.remove(weakest)
            result.skipped_reasons[weakest.name] = (
                f"dropped to break FK cycle among tables {sorted(cycle_tables)} "
                f"(weakest edge, confidence={weakest.ml_confidence:.2f})"
            )
        return survivors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_value_data(
        child_table: str,
        child_col: str,
        parent_table: str,
        parent_col: str,
        sample_data: Optional[Dict[str, pd.DataFrame]],
    ) -> bool:
        if not sample_data:
            return False
        child_df = sample_data.get(child_table)
        parent_df = sample_data.get(parent_table)
        return (
            child_df is not None
            and parent_df is not None
            and child_col in getattr(child_df, "columns", [])
            and parent_col in getattr(parent_df, "columns", [])
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
        value_subset: float,
        has_data: bool,
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
            "kg_hub_bonus": round(kg.hub_bonus, 4),
            "kg_value_inclusion": round(value_subset, 4),
            "kg_inclusion_evidence": 1.0 if has_data else 0.0,
            "kg_base_confidence": round(base_confidence, 4),
            "kg_graph_score": round(graph_score, 4),
            "kg_penalty": round(penalty, 4),
        })
        return enriched

    @staticmethod
    def _explain_kg(signals: Dict[str, float], source: str, candidate) -> str:
        child_table, child_col, parent_table, parent_col = candidate
        parts = [
            f"{child_table}.{child_col} -> {parent_table}.{parent_col}",
            f"source={source}+kg",
            f"base={signals.get('kg_base_confidence', 0):.2f}",
            f"table_sem={signals.get('kg_table_semantic_similarity', 0):.2f}",
            f"descriptor_sem={signals.get('kg_descriptor_similarity', 0):.2f}",
            f"fk_prior={signals.get('kg_child_fk_prior', 0):.2f}",
            f"len_match={signals.get('kg_length_compatibility', 0):.2f}",
        ]
        if signals.get("kg_inclusion_evidence"):
            parts.append(f"value_inclusion={signals.get('kg_value_inclusion', 0):.2f}")
        if signals.get("kg_hub_bonus"):
            parts.append(f"hub=+{signals.get('kg_hub_bonus', 0):.2f}")
        parts.append(f"penalty={signals.get('kg_penalty', 0):.2f}")
        return " | ".join(parts)
