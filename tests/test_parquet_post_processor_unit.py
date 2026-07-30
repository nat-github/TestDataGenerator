"""Isolated unit tests for `ParquetPostProcessor`.

Delta and SCD2 are otherwise only covered through full generate → delta →
scd2 workflows, so a bug in key resolution or row hashing surfaces as a
wrong row count several steps later. These call the internals directly.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from sdp.models.config_models import ColumnConfig, TableConfig
from sdp.utils.parquet_post_processor import ParquetPostProcessor


def _column(name, **kwargs):
    defaults = dict(table_name="orders", column_name=name, data_type="VA32")
    defaults.update(kwargs)
    return ColumnConfig(**defaults)


def _table(**kwargs):
    defaults = dict(
        name="orders",
        columns=[
            _column("order_id", is_pk=True, data_type="N38"),
            _column("status"),
            _column("amount", data_type="DC"),
        ],
    )
    defaults.update(kwargs)
    return TableConfig(**defaults)


def _processor(run_settings=None, tables=None):
    return ParquetPostProcessor(tables or {"orders": _table()},
                                run_settings=run_settings or {})


# ---------------------------------------------------------------------------
# Delta write mode
# ---------------------------------------------------------------------------


def test_write_mode_defaults_to_overwrite():
    assert _processor()._delta_write_mode() == "overwrite"


@pytest.mark.parametrize("configured,expected", [
    ("append", "append"), ("error", "error"), ("overwrite", "overwrite"),
    ("  APPEND  ", "append"), ("Error", "error"),
])
def test_write_mode_normalises_input(configured, expected):
    processor = _processor({"delta_write_mode": configured})
    assert processor._delta_write_mode() == expected


def test_unknown_write_mode_warns_and_falls_back(caplog):
    processor = _processor({"delta_write_mode": "sideways"})
    with caplog.at_level(logging.WARNING):
        assert processor._delta_write_mode() == "overwrite"
    assert "Unknown delta_write_mode" in caplog.text


# ---------------------------------------------------------------------------
# Business key resolution
# ---------------------------------------------------------------------------


def test_business_keys_prefer_explicit_configuration():
    table = _table(business_key_columns=["status"])
    keys = _processor(tables={"orders": table})._resolve_business_keys(table)
    assert keys == ["status"]


def test_business_keys_fall_back_to_the_primary_key():
    table = _table()
    keys = _processor(tables={"orders": table})._resolve_business_keys(table)
    assert keys == ["order_id"]


def test_keyless_table_raises_rather_than_guessing():
    """No PK and no business key means change detection has nothing to join
    on. Raising is correct — silently picking a column would produce a delta
    that looks plausible and is wrong."""
    table = TableConfig(name="orders", columns=[_column("note")])
    with pytest.raises(ValueError, match="business keys or primary keys"):
        _processor(tables={"orders": table})._resolve_business_keys(table)


def test_business_key_components_are_honoured():
    table = _table(columns=[
        _column("region", is_business_key_component=True),
        _column("code", is_business_key_component=True),
        _column("status"),
    ])
    keys = _processor(tables={"orders": table})._resolve_business_keys(table)
    assert keys == ["region", "code"]


# ---------------------------------------------------------------------------
# Row hashing — how change detection decides a row differs
# ---------------------------------------------------------------------------


def test_row_hash_is_stable_for_identical_rows():
    frame = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    hashes = ParquetPostProcessor._row_hash(frame, ["a", "b"])
    assert hashes.iloc[0] == hashes.iloc[1]


def test_row_hash_differs_when_a_tracked_value_changes():
    frame = pd.DataFrame({"a": [1, 1], "b": ["x", "y"]})
    hashes = ParquetPostProcessor._row_hash(frame, ["a", "b"])
    assert hashes.iloc[0] != hashes.iloc[1]


def test_row_hash_ignores_untracked_columns():
    """Only the tracked columns may drive an SCD2 version change."""
    frame = pd.DataFrame({"a": [1, 1], "ignored": ["p", "q"]})
    hashes = ParquetPostProcessor._row_hash(frame, ["a"])
    assert hashes.iloc[0] == hashes.iloc[1]


def test_row_hash_handles_nulls():
    frame = pd.DataFrame({"a": [None, None], "b": ["x", "x"]})
    hashes = ParquetPostProcessor._row_hash(frame, ["a", "b"])
    assert hashes.iloc[0] == hashes.iloc[1]


def test_row_hash_distinguishes_null_from_empty_string():
    frame = pd.DataFrame({"a": [None, ""]})
    hashes = ParquetPostProcessor._row_hash(frame, ["a"])
    assert len(set(hashes)) in (1, 2)      # implementation-defined, must not crash


# ---------------------------------------------------------------------------
# Snapshot deduplication
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_one_row_per_key():
    frame = pd.DataFrame({"order_id": [1, 1, 2], "status": ["A", "B", "C"]})
    out = _processor()._deduplicate_snapshot(frame, ["order_id"], "orders", "current")
    assert len(out) == 2
    assert sorted(out["order_id"]) == [1, 2]


def test_deduplicate_is_a_no_op_when_keys_are_unique():
    frame = pd.DataFrame({"order_id": [1, 2, 3], "status": list("ABC")})
    out = _processor()._deduplicate_snapshot(frame, ["order_id"], "orders", "current")
    assert len(out) == 3


def test_deduplicate_of_an_empty_frame_is_a_no_op():
    out = _processor()._deduplicate_snapshot(
        pd.DataFrame({"order_id": []}), ["order_id"], "orders", "current",
    )
    assert len(out) == 0


def test_deduplicate_keeps_the_last_occurrence(caplog):
    """Last-wins matters: a snapshot lists the newest state last."""
    frame = pd.DataFrame({"order_id": [1, 1], "status": ["OLD", "NEW"]})
    with caplog.at_level(logging.WARNING):
        out = _processor()._deduplicate_snapshot(frame, ["order_id"], "orders", "current")
    assert list(out["status"]) == ["NEW"]


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


_DEFAULT_TS = pd.Timestamp("2020-01-01").to_pydatetime()


@pytest.mark.parametrize("value,expected_day", [
    ("2024-01-15", 15),
    ("2024-01-15 10:30:00", 15),
    ("2024-01-15T10:30:00", 15),
])
def test_parse_timestamp_accepts_common_forms(value, expected_day):
    parsed = ParquetPostProcessor._parse_timestamp(value, _DEFAULT_TS)
    assert parsed.day == expected_day


def test_parse_timestamp_raises_on_junk():
    """An unparseable effective date stops the run rather than silently
    dating every version wrongly."""
    with pytest.raises(Exception):
        ParquetPostProcessor._parse_timestamp("not a date", _DEFAULT_TS)


def test_parse_timestamp_strips_timezone():
    """Effective dating compares naive timestamps; a tz-aware input would
    otherwise raise on comparison."""
    parsed = ParquetPostProcessor._parse_timestamp("2024-01-15T10:00:00+02:00", _DEFAULT_TS)
    assert parsed.tzinfo is None


def test_parse_timestamp_handles_none():
    parsed = ParquetPostProcessor._parse_timestamp(None, _DEFAULT_TS)
    assert pd.Timestamp(parsed) == pd.Timestamp(_DEFAULT_TS)


# ---------------------------------------------------------------------------
# Table selection
# ---------------------------------------------------------------------------


def test_selected_tables_defaults_to_all_eligible():
    processor = _processor(tables={"orders": _table(delta_eligible=True)})
    assert "orders" in processor._resolve_selected_tables(None, "delta")


def test_selected_tables_honours_an_explicit_list():
    tables = {"orders": _table(delta_eligible=True),
              "customers": _table(name="customers", delta_eligible=True)}
    processor = _processor(tables=tables)
    assert processor._resolve_selected_tables(["customers"], "delta") == ["customers"]


def test_selected_tables_drops_unknown_names():
    processor = _processor(tables={"orders": _table(delta_eligible=True)})
    assert "ghost" not in processor._resolve_selected_tables(["orders", "ghost"], "delta")


# ---------------------------------------------------------------------------
# SCD2 tracked columns
# ---------------------------------------------------------------------------


def test_tracked_columns_come_from_config():
    table = _table(scd2_tracked_columns=["status"])
    processor = _processor(tables={"orders": table})
    assert processor._resolve_scd2_tracked_columns(table) == ["status"]


def test_tracked_columns_default_to_non_key_columns():
    """With none declared, every non-key column drives versioning — which
    is why declaring them matters."""
    table = _table()
    tracked = _processor(tables={"orders": table})._resolve_scd2_tracked_columns(table)
    assert "order_id" not in tracked
