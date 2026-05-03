"""End-to-end tests for the ML relationship inferrer.

Covers:
  - simple two-table FK detection (with and without sample data)
  - three-table chain (orders -> customers, items -> orders)
  - ambiguous candidates (two PKs with similar names)
  - already-known relationships are excluded from suggestions
  - threshold filter respected
  - feedback memory affects subsequent runs (the "learning" loop)
  - SME-rejected pairs stop appearing
  - confidence scores are reasonable
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.config_models import ColumnConfig, RelationshipConfig, TableConfig
from ml.relationship_feedback_store import FeedbackEntry, FeedbackStore
from ml.relationship_inferrer import MLRelationshipInferrer


def _col(name: str, dtype: str = "N10", *, is_pk=False, table="t") -> ColumnConfig:
    return ColumnConfig(
        table_name=table, column_name=name, data_type=dtype,
        is_pk=is_pk, nullable=not is_pk,
    )


def _table(name: str, columns: list[ColumnConfig], num_rows: int = 100) -> TableConfig:
    return TableConfig(name=name, columns=columns, num_rows=num_rows)


def _make_inferrer(tmp_path: Path, threshold: float = 0.55) -> MLRelationshipInferrer:
    """Each test gets an isolated feedback store so tests don't bleed state."""
    return MLRelationshipInferrer(
        confidence_threshold=threshold,
        feedback_store=FeedbackStore(tmp_path / "fb.jsonl"),
    )


# ---------------------------------------------------------------------------
# Two-table happy path
# ---------------------------------------------------------------------------


def test_two_table_fk_detected_without_sample_data(tmp_path: Path):
    customers = _table("customers", [
        _col("customer_id", "N10", is_pk=True, table="customers"),
        _col("name", "VA64", table="customers"),
    ])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
        _col("amount", "DC", table="orders"),
    ])
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "orders": orders})

    rels = [r for r in result.relationships
            if r.source_table == "orders" and r.source_column == "customer_id"]
    assert rels, f"expected orders.customer_id -> customers.customer_id; got {result.relationships}"
    rel = rels[0]
    assert rel.target_table == "customers"
    assert rel.target_column == "customer_id"
    assert rel.inferred_by_ml is True
    assert rel.ml_confidence >= 0.55


def test_two_table_fk_detected_with_sample_data_higher_confidence(tmp_path: Path):
    customers = _table("customers", [
        _col("customer_id", "N10", is_pk=True, table="customers"),
        _col("name", "VA64", table="customers"),
    ])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    sample = {
        "customers": pd.DataFrame({"customer_id": list(range(50)),
                                   "name": [f"u{i}" for i in range(50)]}),
        "orders":    pd.DataFrame({"order_id": list(range(100)),
                                   "customer_id": [i % 50 for i in range(100)]}),
    }
    inferrer = _make_inferrer(tmp_path)
    res_no_data = inferrer.infer({"customers": customers, "orders": orders})
    res_with_data = inferrer.infer({"customers": customers, "orders": orders}, sample_data=sample)

    rel_no = next(r for r in res_no_data.relationships
                  if r.source_table == "orders" and r.source_column == "customer_id")
    rel_yes = next(r for r in res_with_data.relationships
                   if r.source_table == "orders" and r.source_column == "customer_id")
    # Sample data confirms full subset → confidence should be at least as high
    assert rel_yes.ml_confidence >= rel_no.ml_confidence


def test_no_fk_when_types_incompatible(tmp_path: Path):
    """An int FK candidate matched against a string PK should be rejected by the type gate."""
    customers = _table("customers", [
        _col("customer_id", "VA32", is_pk=True, table="customers"),
    ])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),  # int, not string!
    ])
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "orders": orders})

    # The mismatched-type candidate should not appear
    bad = [r for r in result.relationships
           if r.source_table == "orders" and r.target_table == "customers"]
    assert not bad


def test_known_relationships_are_excluded_from_suggestions(tmp_path: Path):
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    existing = [RelationshipConfig(
        name="orders_customer_id_to_customers_customer_id",
        source_table="orders", source_column="customer_id",
        target_table="customers", target_column="customer_id",
    )]
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "orders": orders}, existing_relationships=existing)

    # The already-known relationship should not be re-inferred
    dupes = [r for r in result.relationships
             if r.source_column == "customer_id" and r.target_table == "customers"]
    assert not dupes


# ---------------------------------------------------------------------------
# Three-table chain
# ---------------------------------------------------------------------------


def test_three_table_chain_detects_both_fks(tmp_path: Path):
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    items = _table("order_items", [
        _col("item_id", "N10", is_pk=True, table="order_items"),
        _col("order_id", "N10", table="order_items"),
    ])
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "orders": orders, "order_items": items})

    pairs = {(r.source_table, r.source_column, r.target_table, r.target_column)
             for r in result.relationships}
    assert ("orders", "customer_id", "customers", "customer_id") in pairs
    assert ("order_items", "order_id", "orders", "order_id") in pairs


# ---------------------------------------------------------------------------
# Ambiguity / dedup
# ---------------------------------------------------------------------------


def test_ambiguous_candidate_keeps_only_top_match(tmp_path: Path):
    """A child column with two plausible PK targets should yield exactly one
    relationship (the highest-confidence one) thanks to dedup."""
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    clients = _table("clients", [_col("customer_id", "N10", is_pk=True, table="clients")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "clients": clients, "orders": orders})

    matches = [r for r in result.relationships
               if r.source_table == "orders" and r.source_column == "customer_id"]
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def test_threshold_filters_weak_candidates(tmp_path: Path):
    """A high threshold should prune weak matches."""
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("zzz_unrelated", "N10", table="orders"),  # weak name match
    ])
    strict = _make_inferrer(tmp_path, threshold=0.95)
    result = strict.infer({"customers": customers, "orders": orders})
    # Either no matches at all, or only matches that satisfy the strict threshold
    for r in result.relationships:
        assert r.ml_confidence >= 0.95


# ---------------------------------------------------------------------------
# Adaptive learning — pattern memory
# ---------------------------------------------------------------------------


def test_pattern_memory_boosts_previously_accepted_pairs(tmp_path: Path):
    """Acceptance feedback should raise confidence on subsequent runs."""
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    store = FeedbackStore(tmp_path / "fb.jsonl")

    # Run 1 — no feedback yet
    inferrer1 = MLRelationshipInferrer(feedback_store=store)
    res1 = inferrer1.infer({"customers": customers, "orders": orders})
    rel1 = next(r for r in res1.relationships if r.source_column == "customer_id")
    baseline = rel1.ml_confidence

    # SME accepts it — record feedback
    store.append(FeedbackEntry(
        source_table="orders", source_column="customer_id",
        target_table="customers", target_column="customer_id",
        accepted=True,
        signals={"name_similarity": 1.0, "type_compatibility": 1.0,
                 "value_subset": 0.0, "pk_likeness": 1.0},
    ))

    # Run 2 — same data, now with feedback
    inferrer2 = MLRelationshipInferrer(feedback_store=store)
    res2 = inferrer2.infer({"customers": customers, "orders": orders})
    rel2 = next(r for r in res2.relationships if r.source_column == "customer_id")

    assert rel2.ml_confidence > baseline


def test_pattern_memory_rejection_drops_below_threshold(tmp_path: Path):
    """A previously-rejected pair should be filtered out next time."""
    # Build a pair that scores moderately (not overwhelming) so the rejection
    # penalty can drop it below threshold.
    customers = _table("customers", [_col("ref", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("ref_id", "N10", table="orders"),
    ])
    store = FeedbackStore(tmp_path / "fb.jsonl")

    # Run 1 — pair is suggested
    inferrer1 = MLRelationshipInferrer(feedback_store=store, confidence_threshold=0.5)
    res1 = inferrer1.infer({"customers": customers, "orders": orders})
    bad_pair = next((r for r in res1.relationships
                    if r.source_column == "ref_id" and r.target_column == "ref"), None)
    if bad_pair is None:
        pytest.skip("baseline did not produce the candidate, can't test rejection")

    # SME rejects it
    store.append(FeedbackEntry(
        source_table="orders", source_column="ref_id",
        target_table="customers", target_column="ref",
        accepted=False,
        signals=dict(bad_pair.inference_signals or {}),
    ))

    # Run 2 — the rejected pair should not survive
    inferrer2 = MLRelationshipInferrer(feedback_store=store, confidence_threshold=0.5)
    res2 = inferrer2.infer({"customers": customers, "orders": orders})
    rejected = [r for r in res2.relationships
                if r.source_column == "ref_id" and r.target_column == "ref"]
    assert not rejected


def test_classifier_activates_after_enough_feedback(tmp_path: Path):
    """Once the feedback store has enough labelled examples, the classifier
    should mark itself as fitted in the result metadata."""
    from ml.relationship_classifier import MIN_PER_CLASS, MIN_TRAINING_EXAMPLES

    store = FeedbackStore(tmp_path / "fb.jsonl")
    # Stuff the store with enough balanced labelled examples
    for i in range(MIN_TRAINING_EXAMPLES):
        store.append(FeedbackEntry(
            source_table="t1", source_column=f"col{i}",
            target_table="t2", target_column=f"col{i}",
            accepted=True,
            signals={"name_similarity": 0.9, "type_compatibility": 1.0,
                     "value_subset": 1.0, "pk_likeness": 0.9, "pattern_memory_score": 0.0},
        ))
    for i in range(MIN_PER_CLASS + 5):
        store.append(FeedbackEntry(
            source_table="t1", source_column=f"bad{i}",
            target_table="t2", target_column=f"other{i}",
            accepted=False,
            signals={"name_similarity": 0.1, "type_compatibility": 1.0,
                     "value_subset": 0.0, "pk_likeness": 0.1, "pattern_memory_score": 0.0},
        ))

    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])

    inferrer = MLRelationshipInferrer(feedback_store=store)
    result = inferrer.infer({"customers": customers, "orders": orders})

    assert result.classifier_fitted is True
    assert result.classifier_examples >= MIN_TRAINING_EXAMPLES


# ---------------------------------------------------------------------------
# Inactive tables
# ---------------------------------------------------------------------------


def test_inactive_tables_excluded(tmp_path: Path):
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    customers.active = False
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "orders": orders})

    # No relationships — only one active table
    assert not result.relationships


def test_empty_input_returns_empty_result(tmp_path: Path):
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({})
    assert not result.relationships
    assert result.input_tables == 0


# ---------------------------------------------------------------------------
# Inference signals are persisted on output
# ---------------------------------------------------------------------------


def test_inference_signals_attached_to_result(tmp_path: Path):
    customers = _table("customers", [_col("customer_id", "N10", is_pk=True, table="customers")])
    orders = _table("orders", [
        _col("order_id", "N10", is_pk=True, table="orders"),
        _col("customer_id", "N10", table="orders"),
    ])
    inferrer = _make_inferrer(tmp_path)
    result = inferrer.infer({"customers": customers, "orders": orders})

    rel = next(r for r in result.relationships if r.source_column == "customer_id")
    assert rel.inference_signals is not None
    assert "name_similarity" in rel.inference_signals
    assert "type_compatibility" in rel.inference_signals
    assert "pk_likeness" in rel.inference_signals
