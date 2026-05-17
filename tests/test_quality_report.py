"""Tests for `validators/quality_report.py`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdp.validators.quality_report import (
    ColumnQualityMetrics,
    QualityReport,
    TableQualityMetrics,
    quality_report,
    quality_report_from_paths,
)


def _np(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Univariate-only mode
# ---------------------------------------------------------------------------


def test_univariate_only_runs_without_source():
    syn = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    report = quality_report({"t": syn})
    assert report.has_source is False
    assert report.overall_fidelity is None
    assert "t" in report.tables
    assert report.tables["t"].fidelity_score is None
    assert report.tables["t"].row_count_synthetic == 5


def test_numeric_column_summary_stats_present():
    syn = pd.DataFrame({"amt": [10, 20, 30, 40, 50]})
    report = quality_report({"t": syn})
    col = report.tables["t"].columns[0]
    assert col.is_numeric is True
    assert col.mean == pytest.approx(30.0)
    assert col.median == pytest.approx(30.0)
    assert col.min == 10
    assert col.max == 50
    assert col.unique_count == 5


def test_categorical_column_yields_top_values():
    syn = pd.DataFrame({"status": ["A"] * 7 + ["B"] * 2 + ["C"]})
    report = quality_report({"t": syn})
    col = report.tables["t"].columns[0]
    assert col.is_numeric is False
    assert col.top_values is not None
    assert col.top_values[0] == ("A", 7)
    assert ("B", 2) in col.top_values
    assert col.unique_count == 3


def test_null_rate_calculated_correctly():
    syn = pd.DataFrame({"x": [1, 2, None, None, 5]})
    report = quality_report({"t": syn})
    col = report.tables["t"].columns[0]
    assert col.null_count == 2
    assert col.null_rate == pytest.approx(0.4)


def test_correlation_matrix_for_numeric_columns_only():
    rng = _np(1)
    syn = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
        "label": ["X"] * 100,                    # non-numeric — skipped
    })
    report = quality_report({"t": syn})
    t = report.tables["t"]
    assert t.correlation_columns == ["a", "b"]
    assert t.correlation_synthetic is not None
    assert len(t.correlation_synthetic) == 2


def test_correlation_skipped_when_fewer_than_two_numeric_columns():
    syn = pd.DataFrame({"a": [1, 2, 3], "label": ["X", "Y", "Z"]})
    report = quality_report({"t": syn})
    assert report.tables["t"].correlation_synthetic is None
    assert report.tables["t"].correlation_columns is None


# ---------------------------------------------------------------------------
# Fidelity vs source
# ---------------------------------------------------------------------------


def test_identical_source_yields_high_fidelity():
    df = pd.DataFrame({
        "amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "status": ["A", "B"] * 5,
    })
    report = quality_report({"t": df}, source={"t": df.copy()})
    assert report.has_source is True
    assert report.overall_fidelity is not None
    assert report.overall_fidelity > 0.95
    # Per-column scores high
    for col in report.tables["t"].columns:
        assert col.distribution_score is not None
        assert col.distribution_score > 0.8


def test_disjoint_categorical_distributions_yield_low_score():
    syn = pd.DataFrame({"status": ["A"] * 100})
    src = pd.DataFrame({"status": ["B"] * 100})
    report = quality_report({"t": syn}, source={"t": src})
    col = report.tables["t"].columns[0]
    assert col.tv_distance == pytest.approx(1.0)
    assert col.distribution_score == pytest.approx(0.0)


def test_shifted_numeric_distribution_yields_lower_score():
    rng = _np(2)
    syn = pd.Series(rng.normal(0, 1, 1000))
    src = pd.Series(rng.normal(10, 1, 1000))   # mean shifted by 10σ
    report = quality_report({"t": pd.DataFrame({"x": syn})},
                            source={"t": pd.DataFrame({"x": src})})
    col = report.tables["t"].columns[0]
    assert col.ks_statistic is not None
    assert col.ks_statistic > 0.9
    assert col.distribution_score is not None
    assert col.distribution_score < 0.1


def test_mostly_overlapping_categories_yield_intermediate_score():
    syn = pd.DataFrame({"k": ["A"] * 70 + ["B"] * 20 + ["C"] * 10})
    src = pd.DataFrame({"k": ["A"] * 65 + ["B"] * 25 + ["C"] * 10})
    report = quality_report({"t": syn}, source={"t": src})
    col = report.tables["t"].columns[0]
    assert col.tv_distance is not None
    assert 0.0 < col.tv_distance < 0.2
    assert col.distribution_score is not None
    assert col.distribution_score > 0.8


def test_correlation_distance_low_when_correlations_match():
    rng = _np(3)
    base = rng.normal(0, 1, 1000)
    syn = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.1, 1000)})
    src = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.1, 1000)})
    report = quality_report({"t": syn}, source={"t": src})
    assert report.tables["t"].correlation_distance is not None
    assert report.tables["t"].correlation_distance < 0.1


def test_correlation_distance_high_when_correlations_differ():
    rng = _np(4)
    n = 1000
    base = rng.normal(0, 1, n)
    syn = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.05, n)})    # strongly correlated
    src = pd.DataFrame({"a": base, "b": rng.normal(0, 1, n)})              # uncorrelated
    report = quality_report({"t": syn}, source={"t": src})
    assert report.tables["t"].correlation_distance is not None
    assert report.tables["t"].correlation_distance > 0.1


def test_privacy_nn_flags_exact_duplicates():
    """When the synthetic data contains exact source rows, the NN proxy fires."""
    src = pd.DataFrame({"x": list(range(100)), "y": list(range(100))})
    syn = src.iloc[:50].copy()  # exact duplicates of the first 50 source rows
    report = quality_report({"t": syn}, source={"t": src}, privacy_threshold=1e-9)
    rate = report.tables["t"].privacy_nn_too_close_rate
    assert rate is not None
    assert rate == pytest.approx(1.0)


def test_privacy_nn_returns_zero_for_distant_rows():
    rng = _np(5)
    src = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    syn = pd.DataFrame({"x": rng.normal(100, 1, 500), "y": rng.normal(100, 1, 500)})
    report = quality_report({"t": syn}, source={"t": src}, privacy_threshold=0.01)
    rate = report.tables["t"].privacy_nn_too_close_rate
    assert rate is not None
    assert rate < 0.05


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe_still_yields_report():
    syn = pd.DataFrame({"x": [], "y": []})
    report = quality_report({"t": syn})
    assert report.tables["t"].row_count_synthetic == 0


def test_all_null_column_does_not_crash():
    syn = pd.DataFrame({"x": [None, None, None]})
    report = quality_report({"t": syn})
    col = report.tables["t"].columns[0]
    assert col.null_rate == 1.0


def test_table_in_synthetic_but_not_in_source():
    """When a table has no source counterpart, fidelity stays None for that table."""
    report = quality_report(
        {"users": pd.DataFrame({"x": [1, 2]}), "extra": pd.DataFrame({"x": [3, 4]})},
        source={"users": pd.DataFrame({"x": [1, 2]})},
    )
    assert report.tables["users"].fidelity_score is not None
    assert report.tables["extra"].fidelity_score is None


def test_columns_in_synthetic_but_not_in_source():
    """Column-level fidelity is computed only for columns present in both."""
    syn = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    src = pd.DataFrame({"a": [1, 2, 3]})
    report = quality_report({"t": syn}, source={"t": src})
    a = next(c for c in report.tables["t"].columns if c.column == "a")
    b = next(c for c in report.tables["t"].columns if c.column == "b")
    assert a.distribution_score is not None
    assert b.distribution_score is None  # b absent from source


# ---------------------------------------------------------------------------
# Render outputs
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_through_json():
    syn = pd.DataFrame({"x": [1, 2, 3]})
    report = quality_report({"t": syn})
    payload = report.to_dict()
    text = json.dumps(payload)            # must serialise cleanly
    parsed = json.loads(text)
    assert parsed["tables"]["t"]["row_count_synthetic"] == 3


def test_to_markdown_includes_table_and_column():
    syn = pd.DataFrame({"x": [1, 2, 3]})
    report = quality_report({"users": syn})
    md = report.to_markdown()
    assert "Quality Report" in md
    assert "users" in md
    assert "x" in md


def test_to_html_self_contained():
    syn = pd.DataFrame({"x": [1, 2, 3]})
    report = quality_report({"t": syn})
    html = report.to_html()
    assert "<!doctype html>" in html
    assert "<style>" in html
    assert "Quality Report" in html


# ---------------------------------------------------------------------------
# Path-based loader
# ---------------------------------------------------------------------------


def test_quality_report_from_paths(tmp_path: Path):
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    out = tmp_path / "syn"
    out.mkdir()
    df.to_parquet(out / "users.parquet", index=False)

    report = quality_report_from_paths(out)
    assert "users" in report.tables
    assert report.tables["users"].row_count_synthetic == 4
