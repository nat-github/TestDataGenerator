"""ML-based (heuristic + adaptive) relationship inferrer.

Mirrors the interface of `llm.relationship_inferrer.RelationshipInferrer` so
the two engines are interchangeable. No LLM call, no API cost.

Pipeline per candidate (source.col -> target.col):

  1. Type compatibility — gate; incompatible pairs are dropped immediately.
  2. Compute four signals → raw heuristic confidence (weighted sum).
  3. Pattern memory boost / penalty from the feedback store.
  4. Optional learned-classifier override once enough feedback exists.
  5. Threshold filter; emit a RelationshipConfig per surviving candidate.

The inferrer is "data-aware" when callers pass `sample_data` — a dict of
table_name -> pandas DataFrame. Without sample data the value-subset signal
is skipped and confidence is computed from name/type/PK signals only; this
still works but with reduced precision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from sdp.models.config_models import RelationshipConfig, TableConfig
from sdp.ml.relationship_signals import (
    DEFAULT_WEIGHTS,
    combine,
    name_similarity,
    pk_likeness,
    type_compatibility,
    value_subset,
)
from sdp.ml.relationship_feedback_store import FeedbackStore
from sdp.ml.relationship_classifier import RelationshipClassifier, MIN_TRAINING_EXAMPLES

logger = logging.getLogger(__name__)


# Default minimum confidence for an inferred relationship to be reported.
DEFAULT_CONFIDENCE_THRESHOLD = 0.55

# Pattern-memory adjustments. A history of accept means "I've seen this work
# before" — bump the score. History of reject means "I've seen this be wrong"
# — push it below threshold.
PATTERN_ACCEPT_BOOST = 0.15
PATTERN_REJECT_PENALTY = 0.40
COLUMN_PATTERN_ACCEPT_BOOST = 0.07   # cross-project, weaker signal


@dataclass
class MLInferenceResult:
    """Mirror of llm.InferenceResult so callers can treat both uniformly."""
    relationships: List[RelationshipConfig] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    skipped_reasons: Dict[str, str] = field(default_factory=dict)  # debug for filtered-out candidates
    input_tables: int = 0
    classifier_fitted: bool = False
    classifier_examples: int = 0


class MLRelationshipInferrer:
    """Heuristic + adaptive relationship inferrer."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        feedback_store: Optional[FeedbackStore] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.feedback_store = feedback_store or FeedbackStore()
        self.weights = weights or DEFAULT_WEIGHTS
        self.classifier = RelationshipClassifier()
        self._classifier_report = None

    # ------------------------------------------------------------------
    # Public API — matches llm.RelationshipInferrer.infer signature
    # ------------------------------------------------------------------
    def infer(
        self,
        tables: Dict[str, TableConfig],
        existing_relationships: Optional[List[RelationshipConfig]] = None,
        sample_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> MLInferenceResult:
        """Infer relationships from the given table schemas.

        Parameters
        ----------
        tables:
            Dict of table_name -> TableConfig.
        existing_relationships:
            Already-known relationships to exclude (avoids re-suggestion).
        sample_data:
            Optional dict of table_name -> DataFrame providing real values for
            the value-subset signal. Without it, the signal is skipped.
        """
        active = {n: cfg for n, cfg in tables.items() if cfg.active}
        result = MLInferenceResult(input_tables=len(active))

        if not active:
            logger.warning("ml_inferrer: no active tables; nothing to infer")
            return result

        # Train classifier from accumulated feedback (no-op below threshold).
        self._classifier_report = self.classifier.fit(*self.feedback_store.training_data())
        result.classifier_fitted = self._classifier_report.fitted
        result.classifier_examples = self._classifier_report.n_examples
        if self._classifier_report.fitted:
            logger.info(
                "ml_inferrer: classifier active (n=%d, coefs=%s)",
                self._classifier_report.n_examples,
                self._classifier_report.coefs,
            )

        excluded = self._build_exclusions(existing_relationships or [])

        # Iterate every (child_table, child_col, parent_table, parent_col) candidate.
        for child_name, child_cfg in active.items():
            for parent_name, parent_cfg in active.items():
                if child_name == parent_name:
                    continue

                for child_col in child_cfg.columns:
                    for parent_col in parent_cfg.columns:
                        # Parent must be (or look like) a unique identifier.
                        if not parent_col.is_pk and not self._looks_like_pk(parent_col, parent_cfg, sample_data):
                            continue

                        candidate = (child_name, child_col.column_name, parent_name, parent_col.column_name)
                        if candidate in excluded:
                            continue

                        signals = self._compute_signals(
                            child_cfg, child_col, parent_cfg, parent_col, sample_data,
                        )

                        if signals["type_compatibility"] == 0.0:
                            continue  # gate

                        confidence, source = self._final_confidence(signals, candidate)
                        rel_name = f"{child_name}_{child_col.column_name}_to_{parent_name}_{parent_col.column_name}"

                        if confidence < self.confidence_threshold:
                            result.skipped_reasons[rel_name] = (
                                f"below threshold ({confidence:.2f} < {self.confidence_threshold:.2f}); "
                                f"signals={ {k: round(v, 2) for k, v in signals.items()} }"
                            )
                            continue

                        rel = RelationshipConfig(
                            name=rel_name,
                            source_table=child_name,
                            source_column=child_col.column_name,
                            target_table=parent_name,
                            target_column=parent_col.column_name,
                            relationship_type="many_to_one",
                            inferred_by_ml=True,
                            ml_confidence=round(confidence, 4),
                            inference_signals={k: round(v, 4) for k, v in signals.items()},
                        )
                        result.relationships.append(rel)
                        result.reasons[rel_name] = self._explain(signals, source, candidate)

        # Resolve duplicates: when the same child column has multiple high-confidence
        # parent candidates, keep only the top-scoring one (the strongest match).
        result.relationships = self._dedupe_top_per_child_column(result.relationships)

        logger.info(
            "ml_inferrer: produced %d relationship(s) above threshold %.2f",
            len(result.relationships),
            self.confidence_threshold,
        )
        return result

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------
    def _compute_signals(
        self,
        child_cfg: TableConfig,
        child_col,
        parent_cfg: TableConfig,
        parent_col,
        sample_data: Optional[Dict[str, pd.DataFrame]],
    ) -> Dict[str, float]:
        sig: Dict[str, float] = {
            "type_compatibility": type_compatibility(child_col.data_type, parent_col.data_type),
            "name_similarity": name_similarity(child_col.column_name, parent_col.column_name),
            "pk_likeness": pk_likeness(
                values=[],
                total_rows=parent_cfg.num_rows or 1,
                is_pk_declared=parent_col.is_pk,
            ),
        }

        # value_subset requires sample data; skip when absent.
        if sample_data and child_cfg.name in sample_data and parent_cfg.name in sample_data:
            child_df = sample_data[child_cfg.name]
            parent_df = sample_data[parent_cfg.name]
            if (child_col.column_name in child_df.columns
                    and parent_col.column_name in parent_df.columns):
                sig["value_subset"] = value_subset(
                    child_df[child_col.column_name].dropna().tolist(),
                    parent_df[parent_col.column_name].dropna().tolist(),
                )
                # Re-score pk_likeness with the actual sample to catch declared
                # PKs that aren't actually unique in the data.
                sig["pk_likeness"] = pk_likeness(
                    parent_df[parent_col.column_name].dropna().tolist(),
                    total_rows=len(parent_df),
                    is_pk_declared=parent_col.is_pk,
                )
            else:
                sig["value_subset"] = 0.0
        else:
            # Without sample data, bias slightly upwards on PK-like targets so
            # name + type + declared-PK alone can still cross threshold.
            sig["value_subset"] = 0.0

        return sig

    # ------------------------------------------------------------------
    # Final confidence — combines heuristic + memory + classifier
    # ------------------------------------------------------------------
    def _final_confidence(
        self,
        signals: Dict[str, float],
        candidate,
    ) -> tuple[float, str]:
        """Returns (confidence, source) where source ∈ {heuristic, memory, classifier}."""
        child_table, child_col, parent_table, parent_col = candidate

        # Heuristic baseline
        baseline = combine(signals, self.weights)

        # Pattern memory
        accept, reject = self.feedback_store.lookup_pattern(
            child_table, child_col, parent_table, parent_col,
        )
        col_accept, col_reject = self.feedback_store.lookup_column_pattern(child_col, parent_col)

        memory_adjust = 0.0
        if accept and accept > reject:
            memory_adjust += PATTERN_ACCEPT_BOOST
        if reject and reject > accept:
            memory_adjust -= PATTERN_REJECT_PENALTY
        if col_accept and col_accept > col_reject:
            memory_adjust += COLUMN_PATTERN_ACCEPT_BOOST

        signals["pattern_memory_score"] = round(memory_adjust, 4)

        # Classifier override (only when fitted)
        if self.classifier.is_fitted:
            classifier_p = self.classifier.predict(signals)
            if classifier_p is not None:
                # Blend: 70% classifier, 30% heuristic — classifier still benefits
                # from the heuristic anchor early on after activation.
                confidence = 0.7 * classifier_p + 0.3 * (baseline + memory_adjust)
                return min(max(confidence, 0.0), 1.0), "classifier"

        confidence = baseline + memory_adjust
        return min(max(confidence, 0.0), 1.0), "heuristic"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_pk(column, table_cfg: TableConfig, sample_data) -> bool:
        """Cheap pre-filter — only consider parent-side candidates that look
        like primary keys. Saves O(n²) work on tables with 100+ columns."""
        if column.is_pk:
            return True
        if column.column_name in (table_cfg.primary_key_columns or []):
            return True
        # If the column name ends in _id or _key, it's plausibly a PK.
        lname = column.column_name.lower()
        if lname.endswith("_id") or lname.endswith("_key") or lname == "id" or lname == "key":
            return True
        # Otherwise: with sample data, accept if uniqueness > 95%.
        if sample_data and table_cfg.name in sample_data:
            df = sample_data[table_cfg.name]
            if column.column_name in df.columns:
                vals = df[column.column_name].dropna()
                if len(vals) and (len(set(vals)) / len(vals)) > 0.95:
                    return True
        return False

    @staticmethod
    def _build_exclusions(existing: List[RelationshipConfig]) -> set:
        return {
            (r.source_table.lower(), r.source_column, r.target_table.lower(), r.target_column)
            for r in existing
        }

    @staticmethod
    def _dedupe_top_per_child_column(rels: List[RelationshipConfig]) -> List[RelationshipConfig]:
        """For each (child_table, child_column), keep only the highest-confidence rel."""
        best: Dict[tuple, RelationshipConfig] = {}
        for r in rels:
            key = (r.source_table, r.source_column)
            existing = best.get(key)
            if existing is None or (r.ml_confidence or 0) > (existing.ml_confidence or 0):
                best[key] = r
        return list(best.values())

    @staticmethod
    def _explain(signals: Dict[str, float], source: str, candidate) -> str:
        child_table, child_col, parent_table, parent_col = candidate
        parts = [
            f"{child_table}.{child_col} -> {parent_table}.{parent_col}",
            f"source={source}",
            f"name_sim={signals.get('name_similarity', 0):.2f}",
            f"value_subset={signals.get('value_subset', 0):.2f}",
            f"pk_likeness={signals.get('pk_likeness', 0):.2f}",
        ]
        if signals.get("pattern_memory_score"):
            parts.append(f"memory_adj={signals['pattern_memory_score']:+.2f}")
        return " | ".join(parts)
