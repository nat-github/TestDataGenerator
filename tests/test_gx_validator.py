"""Tests for the Great Expectations adapter (`validators/gx_validator.py`).

The pure-function half (`derive_expectations_*`) runs without GX installed.
The integration half is auto-skipped when GX isn't on the path.
"""
from __future__ import annotations

import pandas as pd
import pytest

from sdp.models.config_models import ColumnConfig, TableConfig
from sdp.validators.gx_validator import (
    HAS_GX,
    derive_expectations_for_column,
    derive_expectations_for_table,
    format_report,
    validate_tables,
)


# ---------------------------------------------------------------------------
# Pure-function tests — derive_expectations_*
# ---------------------------------------------------------------------------


def _col(**kwargs) -> ColumnConfig:
    base = dict(table_name="t", column_name="x", data_type="VA32")
    base.update(kwargs)
    return ColumnConfig(**base)


def _kinds(expectations) -> list[str]:
    return [k for k, _, _ in expectations]


def test_pk_column_yields_unique_and_not_null():
    exps = derive_expectations_for_column(_col(column_name="id", is_pk=True))
    kinds = _kinds(exps)
    assert "values_to_be_unique" in kinds
    assert "values_to_not_be_null" in kinds


def test_non_nullable_column_yields_not_null():
    exps = derive_expectations_for_column(_col(column_name="name", nullable=False))
    assert "values_to_not_be_null" in _kinds(exps)


def test_business_values_yields_in_set():
    exps = derive_expectations_for_column(_col(
        column_name="status",
        business_values="ACTIVE;PENDING;CLOSED",
    ))
    kind = next(((k, c, kw) for k, c, kw in exps if k == "values_to_be_in_set"), None)
    assert kind is not None
    assert kind[2]["value_set"] == ["ACTIVE", "PENDING", "CLOSED"]


def test_business_values_handles_pipe_separator():
    exps = derive_expectations_for_column(_col(
        column_name="kind",
        business_values="A|B|C",
    ))
    in_set = next((kw for k, _, kw in exps if k == "values_to_be_in_set"), None)
    assert in_set is not None
    assert in_set["value_set"] == ["A", "B", "C"]


def test_min_max_yield_between():
    exps = derive_expectations_for_column(_col(
        column_name="age", data_type="N10", min_value=18, max_value=99,
    ))
    btw = next((kw for k, _, kw in exps if k == "values_to_be_between"), None)
    assert btw == {"min_value": 18.0, "max_value": 99.0}


def test_special_rule_email_yields_regex():
    exps = derive_expectations_for_column(_col(
        column_name="email",
        special_rules="EMAIL",
    ))
    regex = next((kw["regex"] for k, _, kw in exps if k == "values_to_match_regex"), None)
    assert regex is not None
    assert "@" in regex


def test_special_rule_with_locale_suffix_still_picks_format():
    exps = derive_expectations_for_column(_col(
        column_name="email", special_rules="EMAIL:de_DE",
    ))
    assert any(k == "values_to_match_regex" for k, _, _ in exps)


def test_special_rule_unknown_does_not_yield_regex():
    exps = derive_expectations_for_column(_col(
        column_name="name", special_rules="NAME",
    ))
    assert not any(k == "values_to_match_regex" for k, _, _ in exps)


def test_explicit_regex_rule_passes_through():
    exps = derive_expectations_for_column(_col(
        column_name="code", special_rules=r"REGEX:^X\d{4}$",
    ))
    regex = next((kw["regex"] for k, _, kw in exps if k == "values_to_match_regex"), None)
    assert regex == r"^X\d{4}$"


def test_length_yields_value_lengths_between():
    exps = derive_expectations_for_column(_col(
        column_name="iban", data_type="VA34", length=34,
    ))
    blen = next((kw for k, _, kw in exps if k == "value_lengths_to_be_between"), None)
    assert blen == {"min_value": 1, "max_value": 34}


def test_numeric_data_type_yields_type_list():
    exps = derive_expectations_for_column(_col(
        column_name="qty", data_type="N10",
    ))
    types = next((kw["type_list"] for k, _, kw in exps if k == "values_to_be_in_type_list"), None)
    assert types is not None
    assert "int64" in types or "Int64" in types


def test_string_data_type_yields_type_list():
    exps = derive_expectations_for_column(_col(
        column_name="name", data_type="VA64",
    ))
    types = next((kw["type_list"] for k, _, kw in exps if k == "values_to_be_in_type_list"), None)
    assert "object" in types or "string" in types


def test_table_level_row_count_between():
    tc = TableConfig(name="t", columns=[_col()], num_rows=100)
    exps = derive_expectations_for_table("t", tc, row_count_tolerance=0.5)
    rc = next((kw for k, _, kw in exps if k == "row_count_between"), None)
    assert rc == {"min_value": 50, "max_value": 150}


def test_table_level_row_count_exact_when_no_tolerance():
    tc = TableConfig(name="t", columns=[_col()], num_rows=100)
    exps = derive_expectations_for_table("t", tc, row_count_tolerance=0.0)
    rc = next((kw for k, _, kw in exps if k == "row_count_equal"), None)
    assert rc == {"value": 100}


def test_column_exists_expectation_emitted_per_column():
    tc = TableConfig(
        name="t",
        columns=[_col(column_name="a"), _col(column_name="b")],
        num_rows=10,
    )
    exps = derive_expectations_for_table("t", tc)
    cols = [c for k, c, _ in exps if k == "column_exists"]
    assert cols == ["a", "b"]


# ---------------------------------------------------------------------------
# Integration tests — auto-skipped without GX
# ---------------------------------------------------------------------------


pytestmark_int = pytest.mark.skipif(not HAS_GX, reason="great-expectations not installed")


@pytestmark_int
def test_validate_passes_for_compliant_dataframe():
    cols = [
        _col(column_name="id", data_type="N10", is_pk=True, nullable=False),
        _col(column_name="status", business_values="A;B;C"),
        _col(column_name="age", data_type="N10", min_value=18, max_value=99),
    ]
    tc = TableConfig(name="users", columns=cols, num_rows=5)
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "status": ["A", "B", "C", "A", "B"],
        "age": [25, 35, 45, 55, 65],
    })
    report = validate_tables({"users": tc}, dataframes={"users": df})
    assert report.success
    assert report.tables["users"].failed_expectations == 0


@pytestmark_int
def test_validate_catches_pk_duplicates():
    cols = [_col(column_name="id", data_type="N10", is_pk=True, nullable=False)]
    tc = TableConfig(name="t", columns=cols, num_rows=4)
    df = pd.DataFrame({"id": [1, 2, 2, 3]})
    report = validate_tables({"t": tc}, dataframes={"t": df})
    assert not report.success
    failures = [r for r in report.tables["t"].results if not r.success]
    assert any(r.expectation_type == "values_to_be_unique" for r in failures)


@pytestmark_int
def test_validate_catches_business_value_violation():
    cols = [_col(column_name="status", business_values="A;B;C")]
    tc = TableConfig(name="t", columns=cols, num_rows=3)
    df = pd.DataFrame({"status": ["A", "B", "Z"]})
    report = validate_tables({"t": tc}, dataframes={"t": df})
    assert not report.success
    failures = [r for r in report.tables["t"].results if not r.success]
    assert any(r.expectation_type == "values_to_be_in_set" for r in failures)


@pytestmark_int
def test_validate_catches_min_max_violation():
    cols = [_col(column_name="age", data_type="N10", min_value=18, max_value=99)]
    tc = TableConfig(name="t", columns=cols, num_rows=3)
    df = pd.DataFrame({"age": [17, 50, 100]})
    report = validate_tables({"t": tc}, dataframes={"t": df})
    assert not report.success
    failures = [r for r in report.tables["t"].results if not r.success]
    assert any(r.expectation_type == "values_to_be_between" for r in failures)


@pytestmark_int
def test_validate_catches_email_format_violation():
    cols = [_col(column_name="email", data_type="VA128", special_rules="EMAIL")]
    tc = TableConfig(name="t", columns=cols, num_rows=3)
    df = pd.DataFrame({"email": ["a@b.com", "not-an-email", "ok@x.io"]})
    report = validate_tables({"t": tc}, dataframes={"t": df})
    failures = [r for r in report.tables["t"].results if not r.success]
    assert any(r.expectation_type == "values_to_match_regex" for r in failures)


@pytestmark_int
def test_validate_skips_inactive_tables():
    cols = [_col(column_name="x")]
    tc_active = TableConfig(name="a", columns=cols, num_rows=2)
    tc_inactive = TableConfig(name="b", columns=cols, num_rows=2, active=False)
    df = pd.DataFrame({"x": ["a", "b"]})
    report = validate_tables(
        {"a": tc_active, "b": tc_inactive},
        dataframes={"a": df},
    )
    assert "a" in report.tables
    assert "b" not in report.tables


@pytestmark_int
def test_validate_marks_missing_table_as_failed():
    cols = [_col(column_name="x")]
    tc = TableConfig(name="a", columns=cols, num_rows=2)
    report = validate_tables({"a": tc}, dataframes={})  # no data
    assert not report.success
    assert "No data found" in (report.tables["a"].error or "")


@pytestmark_int
def test_format_report_renders_pass_summary():
    cols = [_col(column_name="x")]
    tc = TableConfig(name="t", columns=cols, num_rows=2)
    df = pd.DataFrame({"x": ["a", "b"]})
    report = validate_tables({"t": tc}, dataframes={"t": df})
    text = format_report(report, verbose=False)
    assert "PASS" in text
    assert "t" in text


@pytestmark_int
def test_format_report_lists_failed_expectations():
    cols = [_col(column_name="status", business_values="A;B")]
    tc = TableConfig(name="t", columns=cols, num_rows=3)
    df = pd.DataFrame({"status": ["A", "B", "Z"]})
    report = validate_tables({"t": tc}, dataframes={"t": df})
    text = format_report(report, verbose=False)
    assert "FAIL" in text
    assert "values_to_be_in_set" in text


@pytestmark_int
def test_row_count_within_tolerance_passes():
    cols = [_col(column_name="x")]
    tc = TableConfig(name="t", columns=cols, num_rows=10)
    df = pd.DataFrame({"x": ["a"] * 12})  # 20% over → within ±50%
    report = validate_tables({"t": tc}, dataframes={"t": df}, row_count_tolerance=0.5)
    rc = next(
        (r for r in report.tables["t"].results if r.expectation_type == "row_count_between"),
        None,
    )
    assert rc is not None and rc.success


@pytestmark_int
def test_row_count_outside_tolerance_fails():
    cols = [_col(column_name="x")]
    tc = TableConfig(name="t", columns=cols, num_rows=10)
    df = pd.DataFrame({"x": ["a"] * 100})  # 10× — way outside ±50%
    report = validate_tables({"t": tc}, dataframes={"t": df}, row_count_tolerance=0.5)
    rc = next(
        (r for r in report.tables["t"].results if r.expectation_type == "row_count_between"),
        None,
    )
    assert rc is not None and not rc.success
