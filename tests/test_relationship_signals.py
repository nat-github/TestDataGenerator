"""Unit tests for the relationship-inference signal computers.

Each signal is exercised independently with edge cases — these are pure
functions, so the tests are tight, fast, and deterministic.
"""
from __future__ import annotations

import pytest

from sdp.ml.relationship_signals import (
    DEFAULT_WEIGHTS,
    combine,
    name_similarity,
    pk_likeness,
    type_compatibility,
    value_subset,
)


# ---------------------------------------------------------------------------
# type_compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("N10", "N10", 1.0),
        ("N10", "N19", 1.0),       # both numeric
        ("VA10", "VA64", 1.0),     # both string
        ("DC", "N10", 1.0),        # decimal is numeric
        ("D", "DT", 1.0),          # both datetime
        ("TS", "DT", 1.0),         # ts and dt
        ("N10", "VA10", 0.0),      # numeric vs string
        ("N10", "D", 0.0),         # numeric vs datetime
        ("VA10", "DT", 0.0),       # string vs datetime
        ("N10", None, 0.0),        # missing type
        ("", "N10", 0.0),          # empty type
        (None, None, 0.0),
    ],
)
def test_type_compatibility(left, right, expected):
    assert type_compatibility(left, right) == expected


def test_type_compatibility_handles_unknown_prefix():
    """Unknown data-type prefixes return 0 (conservative — incompatible)."""
    assert type_compatibility("X10", "N10") == 0.0
    assert type_compatibility("X10", "X10") == 0.0  # unknown family is None


# ---------------------------------------------------------------------------
# name_similarity
# ---------------------------------------------------------------------------


def test_name_similarity_identical_names():
    assert name_similarity("customer_id", "customer_id") == 1.0


def test_name_similarity_only_stopword_overlap_low():
    """customer_id <-> order_id should NOT score high just from sharing 'id'."""
    score = name_similarity("customer_id", "order_id")
    assert score < 0.6


def test_name_similarity_camel_case_and_snake_case_match():
    """customerID and customer_id should be considered very similar."""
    score = name_similarity("customerID", "customer_id")
    assert score > 0.7


def test_name_similarity_abbreviation_partial_match():
    """cust_id <-> customer_id should score reasonably high."""
    score = name_similarity("cust_id", "customer_id")
    assert score > 0.5


def test_name_similarity_typo_tolerance():
    """One-character typo should still score high via Levenshtein."""
    score = name_similarity("customer_id", "custmer_id")
    assert score > 0.85


def test_name_similarity_unrelated_columns_low():
    score = name_similarity("balance", "country_code")
    assert score < 0.4


def test_name_similarity_empty_inputs():
    assert name_similarity("", "anything") == 0.0
    assert name_similarity("anything", "") == 0.0
    assert name_similarity("", "") == 0.0


def test_name_similarity_pure_id_columns():
    """Two columns named exactly `id` should match — the stopword fallback uses
    raw string ratio so exact matches still score 1.0."""
    assert name_similarity("id", "id") == 1.0


# ---------------------------------------------------------------------------
# value_subset
# ---------------------------------------------------------------------------


def test_value_subset_full_containment():
    assert value_subset([1, 2, 3], [1, 2, 3, 4, 5]) == 1.0


def test_value_subset_partial_overlap():
    score = value_subset([1, 2, 3, 4], [1, 2, 5, 6])
    assert score == pytest.approx(0.5)


def test_value_subset_no_overlap():
    assert value_subset([1, 2, 3], [4, 5, 6]) == 0.0


def test_value_subset_empty_inputs():
    assert value_subset([], [1, 2, 3]) == 0.0
    assert value_subset([1, 2, 3], []) == 0.0
    assert value_subset([], []) == 0.0


def test_value_subset_int_float_string_normalisation():
    """Pandas often stores integer FK columns as floats. The signal should
    still match int parents to float children."""
    assert value_subset([1.0, 2.0, 3.0], [1, 2, 3, 4]) == 1.0
    assert value_subset(["123", "456"], [123, 456, 789]) == 1.0


def test_value_subset_handles_nan():
    """NaN values in the child set should be ignored (treated as missing)."""
    score = value_subset([1, 2, float("nan"), 3], [1, 2, 3])
    assert score == 1.0


def test_value_subset_distinct_only():
    """Duplicate child values shouldn't hurt the score — operates on distinct sets."""
    assert value_subset([1, 1, 1, 1], [1, 2]) == 1.0


# ---------------------------------------------------------------------------
# pk_likeness
# ---------------------------------------------------------------------------


def test_pk_likeness_declared_pk_returns_one():
    """A declared PK shortcut: trust the schema."""
    assert pk_likeness([], total_rows=100, is_pk_declared=True) == 1.0


def test_pk_likeness_unique_high_cardinality():
    """100 unique values out of 100 rows — strong PK."""
    score = pk_likeness(list(range(100)), total_rows=100)
    assert score >= 0.9


def test_pk_likeness_low_uniqueness():
    """50 distinct values out of 100 — moderate score."""
    vals = [i % 50 for i in range(100)]
    score = pk_likeness(vals, total_rows=100)
    # 0.6 * 0.5 + 0.4 * 0.5 = 0.5
    assert 0.4 < score < 0.6


def test_pk_likeness_low_cardinality_penalty():
    """Only 5 distinct values across 100 rows — bad PK candidate."""
    vals = [i % 5 for i in range(100)]
    score = pk_likeness(vals, total_rows=100)
    assert score < 0.1


def test_pk_likeness_empty():
    assert pk_likeness([], total_rows=0) == 0.0
    assert pk_likeness([None, None], total_rows=2) == 0.0


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------


def test_combine_uses_default_weights():
    score = combine({
        "type_compatibility": 1.0,
        "name_similarity": 1.0,
        "value_subset": 1.0,
        "pk_likeness": 1.0,
    })
    assert score == pytest.approx(1.0, rel=1e-3)


def test_combine_zero_type_gates_to_zero():
    """type_compatibility == 0 forces total to 0 regardless of other signals."""
    score = combine({
        "type_compatibility": 0.0,
        "name_similarity": 1.0,
        "value_subset": 1.0,
        "pk_likeness": 1.0,
    })
    assert score == 0.0


def test_combine_partial_signals_ok():
    """Missing signals are simply skipped, weights renormalised."""
    score = combine({"name_similarity": 0.5, "value_subset": 0.5})
    assert score == pytest.approx(0.5, rel=1e-3)


def test_combine_custom_weights():
    custom = {"name_similarity": 1.0}  # only weight name
    score = combine({"name_similarity": 0.8, "value_subset": 0.0}, custom)
    assert score == pytest.approx(0.8, rel=1e-3)


def test_combine_default_weights_keys_match_signals():
    """The default weight dict must reference real signal names — guards
    against typos that would silently produce zero scores."""
    expected = {"name_similarity", "value_subset", "pk_likeness"}
    assert set(DEFAULT_WEIGHTS.keys()) == expected
