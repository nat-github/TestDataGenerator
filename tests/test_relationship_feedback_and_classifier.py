"""Tests for the feedback store and the adaptive classifier.

Covers:
  - JSONL persistence round-trip
  - pattern memory lookups (exact + column-only)
  - malformed-line tolerance
  - cold-start classifier refusal
  - classifier activation thresholds
  - retraining from new feedback
  - prediction probability range
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdp.ml.relationship_classifier import (
    FEATURE_ORDER,
    MIN_PER_CLASS,
    MIN_TRAINING_EXAMPLES,
    RelationshipClassifier,
)
from sdp.ml.relationship_feedback_store import FeedbackEntry, FeedbackStore


# ---------------------------------------------------------------------------
# FeedbackStore basic persistence
# ---------------------------------------------------------------------------


def _entry(src_t="orders", src_c="customer_id", tgt_t="customers", tgt_c="customer_id",
           accepted=True, signals=None) -> FeedbackEntry:
    return FeedbackEntry(
        source_table=src_t,
        source_column=src_c,
        target_table=tgt_t,
        target_column=tgt_c,
        accepted=accepted,
        signals=signals or {"name_similarity": 0.9, "value_subset": 1.0,
                            "type_compatibility": 1.0, "pk_likeness": 0.9},
    )


def test_feedback_store_round_trip(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    e = _entry()
    store.append(e)

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].source_table == "orders"
    assert loaded[0].accepted is True


def test_feedback_store_appends_multiple(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    written = store.append_many([
        _entry(accepted=True),
        _entry(src_c="cust_id", accepted=False),
        _entry(src_c="customer_ref", accepted=True),
    ])
    assert written == 3
    assert len(store.load()) == 3


def test_feedback_store_stats(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append_many([
        _entry(accepted=True),
        _entry(accepted=True),
        _entry(accepted=False),
    ])
    stats = store.stats()
    assert stats == {"total": 3, "accepted": 2, "rejected": 1}


def test_feedback_store_tolerates_malformed_lines(tmp_path: Path):
    path = tmp_path / "fb.jsonl"
    valid = json.dumps({
        "source_table": "a", "source_column": "x", "target_table": "b",
        "target_column": "y", "accepted": True,
    })
    path.write_text(f"{valid}\n\nnot-json-at-all\n{valid}\n", encoding="utf-8")

    store = FeedbackStore(path)
    entries = store.load()
    # 2 valid lines, 1 malformed silently skipped
    assert len(entries) == 2


def test_feedback_store_pattern_lookup_exact(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append_many([
        _entry(accepted=True),
        _entry(accepted=True),
        _entry(accepted=False),
        _entry(src_c="something_else", accepted=True),
    ])

    accept, reject = store.lookup_pattern(
        "orders", "customer_id", "customers", "customer_id",
    )
    assert (accept, reject) == (2, 1)


def test_feedback_store_pattern_lookup_case_insensitive_tables(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(FeedbackEntry(
        source_table="Orders", source_column="customer_id",
        target_table="Customers", target_column="customer_id",
        accepted=True,
    ))
    accept, _ = store.lookup_pattern("orders", "customer_id", "customers", "customer_id")
    assert accept == 1


def test_feedback_store_column_pattern_generalises_across_tables(tmp_path: Path):
    """The column-only key should match across different table names — that's
    how cross-project knowledge accumulates."""
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append_many([
        FeedbackEntry(source_table="orders", source_column="customer_id",
                      target_table="customers", target_column="customer_id", accepted=True),
        FeedbackEntry(source_table="invoices", source_column="customer_id",
                      target_table="clients", target_column="customer_id", accepted=True),
    ])
    accept, reject = store.lookup_column_pattern("customer_id", "customer_id")
    assert (accept, reject) == (2, 0)


def test_feedback_store_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """SDP_FEEDBACK_PATH env var should take precedence when no path is passed."""
    target = tmp_path / "env_fb.jsonl"
    monkeypatch.setenv("SDP_FEEDBACK_PATH", str(target))
    store = FeedbackStore()
    assert store.path == target


def test_feedback_entry_to_from_json_round_trip():
    e = _entry()
    e2 = FeedbackEntry.from_json(e.to_json())
    assert e2.source_table == e.source_table
    assert e2.signals == e.signals
    assert e2.accepted == e.accepted


# ---------------------------------------------------------------------------
# Training-data extraction
# ---------------------------------------------------------------------------


def test_training_data_skips_entries_without_signals(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append_many([
        _entry(accepted=True),                 # has signals
        FeedbackEntry(source_table="x", source_column="a",
                      target_table="y", target_column="b",
                      accepted=True, signals={}),  # no signals — skipped
    ])
    X, y = store.training_data()
    assert len(X) == 1
    assert len(y) == 1


# ---------------------------------------------------------------------------
# RelationshipClassifier — cold start and activation
# ---------------------------------------------------------------------------


def test_classifier_refuses_below_threshold():
    clf = RelationshipClassifier()
    X = [{"name_similarity": 0.9, "value_subset": 1.0}] * 5
    y = [1, 1, 0, 0, 1]
    report = clf.fit(X, y)
    assert report.fitted is False
    assert "at least" in report.reason.lower()
    assert clf.predict({"name_similarity": 1.0}) is None


def test_classifier_refuses_when_one_class_missing():
    clf = RelationshipClassifier()
    # Exactly threshold examples but all the same class
    X = [{"name_similarity": 0.5}] * MIN_TRAINING_EXAMPLES
    y = [1] * MIN_TRAINING_EXAMPLES
    report = clf.fit(X, y)
    assert report.fitted is False
    assert "per class" in report.reason.lower()


def test_classifier_fits_with_balanced_data():
    clf = RelationshipClassifier()
    X = []
    y = []
    # Generate strong-positive and strong-negative examples
    for _ in range(MIN_TRAINING_EXAMPLES):
        X.append({"name_similarity": 0.9, "value_subset": 1.0,
                  "type_compatibility": 1.0, "pk_likeness": 0.9, "pattern_memory_score": 0.0})
        y.append(1)
    for _ in range(MIN_PER_CLASS + 5):
        X.append({"name_similarity": 0.1, "value_subset": 0.0,
                  "type_compatibility": 1.0, "pk_likeness": 0.2, "pattern_memory_score": 0.0})
        y.append(0)

    report = clf.fit(X, y)
    assert report.fitted is True
    assert report.coefs is not None
    assert set(report.coefs.keys()) == set(FEATURE_ORDER)


def test_classifier_predictions_increase_with_signals():
    clf = RelationshipClassifier()
    X, y = [], []
    for _ in range(MIN_TRAINING_EXAMPLES):
        X.append({"name_similarity": 0.95, "value_subset": 1.0,
                  "type_compatibility": 1.0, "pk_likeness": 0.95, "pattern_memory_score": 0.1})
        y.append(1)
    for _ in range(MIN_PER_CLASS + 5):
        X.append({"name_similarity": 0.05, "value_subset": 0.0,
                  "type_compatibility": 1.0, "pk_likeness": 0.05, "pattern_memory_score": 0.0})
        y.append(0)
    clf.fit(X, y)

    strong = clf.predict({"name_similarity": 0.95, "value_subset": 1.0,
                          "type_compatibility": 1.0, "pk_likeness": 0.95, "pattern_memory_score": 0.0})
    weak = clf.predict({"name_similarity": 0.1, "value_subset": 0.0,
                        "type_compatibility": 1.0, "pk_likeness": 0.1, "pattern_memory_score": 0.0})
    assert strong is not None and weak is not None
    assert strong > weak
    assert 0 <= strong <= 1 and 0 <= weak <= 1


def test_classifier_retrains_after_new_feedback():
    """Re-fitting on a larger dataset should reset the model."""
    clf = RelationshipClassifier()
    # First training set
    X1, y1 = [], []
    for _ in range(MIN_TRAINING_EXAMPLES):
        X1.append({"name_similarity": 0.5, "value_subset": 0.5,
                   "type_compatibility": 1.0, "pk_likeness": 0.5, "pattern_memory_score": 0.0})
        y1.append(1)
    for _ in range(MIN_PER_CLASS + 1):
        X1.append({"name_similarity": 0.0, "value_subset": 0.0,
                   "type_compatibility": 0.0, "pk_likeness": 0.0, "pattern_memory_score": 0.0})
        y1.append(0)
    r1 = clf.fit(X1, y1)
    assert r1.fitted

    # Second training run with stronger positive signal
    X2, y2 = list(X1), list(y1)
    for _ in range(20):
        X2.append({"name_similarity": 1.0, "value_subset": 1.0,
                   "type_compatibility": 1.0, "pk_likeness": 1.0, "pattern_memory_score": 0.5})
        y2.append(1)
    r2 = clf.fit(X2, y2)
    assert r2.fitted
    assert r2.n_examples > r1.n_examples


def test_classifier_handles_missing_features_gracefully():
    """Inference with a partial signal dict should default missing features to 0."""
    clf = RelationshipClassifier()
    X, y = [], []
    for _ in range(MIN_TRAINING_EXAMPLES):
        X.append({"name_similarity": 0.9, "value_subset": 1.0,
                  "type_compatibility": 1.0, "pk_likeness": 0.9, "pattern_memory_score": 0.0})
        y.append(1)
    for _ in range(MIN_PER_CLASS):
        X.append({"name_similarity": 0.1, "value_subset": 0.0,
                  "type_compatibility": 1.0, "pk_likeness": 0.1, "pattern_memory_score": 0.0})
        y.append(0)
    clf.fit(X, y)

    # Only one signal present — should still produce a probability
    p = clf.predict({"name_similarity": 0.5})
    assert p is not None
    assert 0 <= p <= 1
