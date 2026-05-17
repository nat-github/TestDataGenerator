"""Tests for data contract testing — sdp.contracts (checker + diff).

The diff tests are pure (no Great Expectations). The checker tests need the
`gx` extra and are skipped automatically when it is not installed.
"""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import pytest

from sdp.contracts import diff_contracts, run_contract_test
from sdp.contracts.checker import format_contract_report
from sdp.contracts.diff import format_contract_diff
from sdp.contracts.model import ChangeClass, Severity, Verdict
from sdp.models.config_models import ColumnConfig, TableConfig
from sdp.validators.gx_validator import HAS_GX

requires_gx = pytest.mark.skipif(not HAS_GX, reason="contract testing needs the 'gx' extra")


def _col(name: str, dtype: str = "N38", *, is_pk: bool = False, nullable: bool = True,
         business_values: str = None, min_value=None, max_value=None, length=None,
         table: str = "t") -> ColumnConfig:
    return ColumnConfig(
        table_name=table, column_name=name, data_type=dtype,
        is_pk=is_pk, nullable=(False if is_pk else nullable),
        business_values=business_values, min_value=min_value, max_value=max_value,
        length=length,
    )


def _table(name: str, columns: List[ColumnConfig], num_rows: int = 3) -> TableConfig:
    return TableConfig(name=name, columns=columns, num_rows=num_rows)


def _users_contract() -> Dict[str, TableConfig]:
    return {
        "users": _table("users", [
            _col("id", "N38", is_pk=True, table="users"),
            _col("status", "VA3", nullable=False, business_values="A;B;C", table="users"),
        ], num_rows=3),
    }


# ---------------------------------------------------------------------------
# contract test — verdict + severity
# ---------------------------------------------------------------------------

@requires_gx
def test_contract_test_passes_on_conforming_data():
    frames = {"users": pd.DataFrame({"id": [1, 2, 3], "status": ["A", "B", "C"]})}
    report = run_contract_test(_users_contract(), dataframes=frames, contract_name="users")

    assert report.verdict is Verdict.PASS
    assert report.passed is True
    assert report.error_failures == []
    assert report.warning_failures == []


@requires_gx
def test_contract_test_fails_on_error_severity_violation():
    # A null primary key violates NOT NULL — an error-severity check.
    frames = {"users": pd.DataFrame({"id": [1, None, 3], "status": ["A", "B", "C"]})}
    report = run_contract_test(_users_contract(), dataframes=frames)

    assert report.verdict is Verdict.FAIL
    assert report.passed is False
    assert report.error_failures, "expected an error-severity failure"
    assert all(c.severity is Severity.ERROR for c in report.error_failures)


@requires_gx
def test_contract_test_warns_on_quality_only_violation():
    # 'X' is outside the declared business_values — a warning-severity check.
    frames = {"users": pd.DataFrame({"id": [1, 2, 3], "status": ["A", "B", "X"]})}
    report = run_contract_test(_users_contract(), dataframes=frames)

    assert report.verdict is Verdict.WARN
    assert report.passed is True            # WARN still counts as honoured
    assert report.error_failures == []
    assert report.warning_failures, "expected a warning-severity failure"


@requires_gx
def test_contract_test_report_is_json_serialisable():
    frames = {"users": pd.DataFrame({"id": [1, 2, 3], "status": ["A", "B", "X"]})}
    report = run_contract_test(_users_contract(), dataframes=frames)
    # Strict JSON — must not raise on NaN / numpy scalars.
    encoded = json.dumps(report.to_dict())
    assert json.loads(encoded)["verdict"] == "warn"
    assert isinstance(format_contract_report(report, verbose=True), str)


# ---------------------------------------------------------------------------
# contract diff — breaking-change classification
# ---------------------------------------------------------------------------

def test_diff_identical_contracts_has_no_changes():
    diff = diff_contracts(_users_contract(), _users_contract())
    assert diff.changes == []
    assert diff.has_breaking is False
    assert isinstance(format_contract_diff(diff), str)


def test_diff_removed_column_is_breaking():
    new = {"users": _table("users", [_col("id", "N38", is_pk=True, table="users")])}
    diff = diff_contracts(_users_contract(), new)
    assert diff.has_breaking
    kinds = {c.kind for c in diff.breaking}
    assert "column_removed" in kinds


def test_diff_new_optional_column_is_additive_required_is_breaking():
    optional_new = {"users": _table("users", [
        _col("id", "N38", is_pk=True, table="users"),
        _col("status", "VA3", nullable=False, business_values="A;B;C", table="users"),
        _col("nickname", "VA256", nullable=True, table="users"),
    ])}
    diff = diff_contracts(_users_contract(), optional_new)
    assert not diff.has_breaking
    assert any(c.kind == "column_added" for c in diff.additive)

    required_new = {"users": _table("users", [
        _col("id", "N38", is_pk=True, table="users"),
        _col("status", "VA3", nullable=False, business_values="A;B;C", table="users"),
        _col("region", "VA3", nullable=False, table="users"),
    ])}
    diff2 = diff_contracts(_users_contract(), required_new)
    assert diff2.has_breaking
    assert any(c.kind == "required_column_added" for c in diff2.breaking)


def test_diff_enum_shrunk_is_breaking_expanded_is_additive():
    shrunk = {"users": _table("users", [
        _col("id", "N38", is_pk=True, table="users"),
        _col("status", "VA3", nullable=False, business_values="A;B", table="users"),
    ])}
    diff = diff_contracts(_users_contract(), shrunk)
    assert diff.has_breaking
    assert any(c.kind == "enum_shrunk" for c in diff.breaking)

    expanded = {"users": _table("users", [
        _col("id", "N38", is_pk=True, table="users"),
        _col("status", "VA3", nullable=False, business_values="A;B;C;D", table="users"),
    ])}
    diff2 = diff_contracts(_users_contract(), expanded)
    assert not diff2.has_breaking
    assert any(c.kind == "enum_expanded" for c in diff2.additive)


def test_diff_type_narrowed_is_breaking_widened_is_additive():
    old = {"t": _table("t", [_col("note", "VA256", table="t")])}
    narrowed = {"t": _table("t", [_col("note", "VA3", table="t")])}
    widened = {"t": _table("t", [_col("note", "VA256", table="t")])}

    d_narrow = diff_contracts(old, narrowed)
    assert any(c.kind == "type_narrowed" and c.classification is ChangeClass.BREAKING
               for c in d_narrow.changes)

    d_widen = diff_contracts(narrowed, widened)
    assert any(c.kind == "type_widened" and c.classification is ChangeClass.ADDITIVE
               for c in d_widen.changes)


def test_diff_removed_table_is_breaking():
    old = {
        "users": _table("users", [_col("id", "N38", is_pk=True, table="users")]),
        "orders": _table("orders", [_col("oid", "N38", is_pk=True, table="orders")]),
    }
    new = {"users": old["users"]}
    diff = diff_contracts(old, new)
    assert diff.has_breaking
    assert any(c.kind == "table_removed" for c in diff.breaking)


def test_diff_range_narrowed_is_breaking():
    old = {"t": _table("t", [_col("amount", "DC", min_value=0, max_value=100, table="t")])}
    new = {"t": _table("t", [_col("amount", "DC", min_value=10, max_value=100, table="t")])}
    diff = diff_contracts(old, new)
    assert diff.has_breaking
    assert any(c.kind == "range_narrowed" for c in diff.breaking)
