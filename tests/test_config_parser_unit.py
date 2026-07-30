"""Isolated unit tests for `ConfigParser`.

Existing coverage exercises the parser through full generate/delta runs,
which means a parsing bug surfaces as a mysterious downstream failure. These
tests call the parsing helpers directly — most are static, so they need no
config file at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sdp.utils.config_parser import ConfigIssue, ConfigParser


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("true", True), ("TRUE", True), ("Yes", True), ("y", True), ("1", True), (1, True),
    ("false", False), ("no", False), ("n", False), ("0", False), (0, False),
])
def test_to_bool_recognised_forms(value, expected):
    assert ConfigParser._to_bool(value) is expected


def test_to_bool_uses_default_for_unrecognised():
    assert ConfigParser._to_bool("maybe", default=True) is True
    assert ConfigParser._to_bool(None, default=True) is True
    assert ConfigParser._to_bool(float("nan"), default=True) is True


@pytest.mark.parametrize("value,expected", [
    ("42", 42), ("-7", -7), ("3.5", 3.5), ("-0.25", -0.25),
    ("true", True), ("FALSE", False),
    ("plain", "plain"), ("  padded  ", "padded"),
    ("", None), ("   ", None), (None, None),
])
def test_coerce_setting_value(value, expected):
    assert ConfigParser._coerce_setting_value(value) == expected


def test_coerce_setting_value_passes_non_strings_through():
    assert ConfigParser._coerce_setting_value(7) == 7
    assert ConfigParser._coerce_setting_value(float("nan")) is None


@pytest.mark.parametrize("value,expected", [
    ("a;b;c", ["a", "b", "c"]),
    ("a, b , c", ["a", "b", "c"]),          # commas normalise to semicolons
    ("  spaced  ", ["spaced"]),
    ("a;;b", ["a", "b"]),                    # empties dropped
    (["x", " y "], ["x", "y"]),
    ("", []), (None, []),
])
def test_split_multi_value(value, expected):
    assert ConfigParser._split_multi_value(value) == expected


def test_split_multi_value_handles_nan():
    assert ConfigParser._split_multi_value(float("nan")) == []


def test_optional_string_trims_and_nulls():
    assert ConfigParser._optional_string("  hi  ") == "hi"
    assert ConfigParser._optional_string("") is None
    assert ConfigParser._optional_string(None) is None
    assert ConfigParser._optional_string(float("nan")) is None


def test_excel_row_is_one_based_with_header():
    """Row 0 of the frame is row 2 of the sheet — off-by-one here makes
    every lint message point at the wrong line."""
    assert ConfigParser._excel_row(0) == 2
    assert ConfigParser._excel_row(10) == 12


# ---------------------------------------------------------------------------
# Data-type grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared,expected", [
    ("DC(18,2)", ("DC", None, 18, 2)),
    ("NS(15)",   ("NS", 15, None, None)),
    ("A34",      ("A", 34, None, None)),
    ("N19",      ("N", 19, None, None)),
    ("VA256",    ("VA", 256, None, None)),
    ("D",        ("D", None, None, None)),
    ("TS",       ("TS", None, None, None)),
])
def test_parse_data_type_details(declared, expected):
    assert ConfigParser("x").parse_data_type_details(declared) == expected


def test_parse_data_type_details_tolerates_non_strings():
    base, *_ = ConfigParser("x").parse_data_type_details(None)
    assert base == "None"


def test_parse_data_type_details_is_case_insensitive():
    assert ConfigParser("x").parse_data_type_details("dc(9,3)")[0] == "DC"


# ---------------------------------------------------------------------------
# Rules field
# ---------------------------------------------------------------------------


def test_parse_rules_field_accepts_a_list_of_dicts():
    rules = ConfigParser._parse_rules_field(
        [{"when": {"status": {"eq": "A"}}, "then": {"set_value": "X"}}]
    )
    assert rules is not None and len(rules) == 1


def test_parse_rules_field_accepts_a_json_string():
    """Excel cannot hold a list, so a cell holds JSON."""
    payload = json.dumps([{"when": {"s": {"eq": "A"}}, "then": {"set_value": "X"}}])
    rules = ConfigParser._parse_rules_field(payload)
    assert rules is not None and len(rules) == 1


def test_parse_rules_field_rejects_malformed_json():
    assert ConfigParser._parse_rules_field("{not json") is None


def test_parse_rules_field_ignores_non_lists():
    assert ConfigParser._parse_rules_field({"when": {}}) is None
    assert ConfigParser._parse_rules_field(42) is None


def test_parse_rules_field_empty_inputs():
    assert ConfigParser._parse_rules_field(None) is None
    assert ConfigParser._parse_rules_field("") is None
    assert ConfigParser._parse_rules_field("   ") is None
    assert ConfigParser._parse_rules_field(float("nan")) is None
    assert ConfigParser._parse_rules_field([]) is None


def test_parse_rules_field_skips_non_dict_entries():
    rules = ConfigParser._parse_rules_field(
        ["not a dict", {"when": {"s": {"eq": "A"}}, "then": {"set_value": "X"}}]
    )
    assert rules is not None and len(rules) == 1


# ---------------------------------------------------------------------------
# CDC block
# ---------------------------------------------------------------------------


def test_expand_cdc_snapshot_mode():
    out = ConfigParser._expand_cdc_block({"mode": "snapshot"})
    assert out["generation_mode"] == "snapshot"
    assert out.get("scd2_enabled") in (False, None)


def test_expand_cdc_delta_mode():
    out = ConfigParser._expand_cdc_block({"mode": "delta"})
    assert out["delta_eligible"] is True


def test_expand_cdc_scd2_implies_delta():
    """SCD2 is delta plus history — a config saying scd2 must not have to
    say delta as well."""
    out = ConfigParser._expand_cdc_block({"mode": "scd2", "track": ["status"]})
    assert out["scd2_enabled"] is True
    assert out["delta_eligible"] is True
    assert out["scd2_tracked_columns"] == ["status"]


def test_expand_cdc_partition_and_event_time():
    out = ConfigParser._expand_cdc_block({
        "mode": "delta", "partition_by": ["region"], "event_time": "updated_at",
    })
    assert out["partition_columns"] == ["region"]
    assert out["event_time_column"] == "updated_at"


def test_expand_cdc_ignores_unknown_keys():
    out = ConfigParser._expand_cdc_block({"mode": "snapshot", "nonsense": 1})
    assert "nonsense" not in out


def test_expand_cdc_empty_inputs():
    assert ConfigParser._expand_cdc_block(None) == {}
    assert ConfigParser._expand_cdc_block({}) == {}
    assert ConfigParser._expand_cdc_block("not a mapping") == {}


# ---------------------------------------------------------------------------
# ConfigIssue rendering
# ---------------------------------------------------------------------------


def test_issue_renders_location_context():
    issue = ConfigIssue(level="error", message="boom", sheet="Columns",
                        row=7, column="data_type", table="orders")
    text = str(issue)
    assert "[ERROR]" in text and "sheet=Columns" in text
    assert "row=7" in text and "col=data_type" in text and "table=orders" in text
    assert text.endswith("boom")


def test_issue_without_location_is_still_readable():
    # `[WARN] ` is padded to align with `[ERROR]`, so collapse whitespace.
    rendered = " ".join(str(ConfigIssue(level="warning", message="hmm")).split())
    assert rendered == "[WARN] hmm"


# ---------------------------------------------------------------------------
# Settings lookup
# ---------------------------------------------------------------------------


def test_get_setting_returns_default_when_absent():
    parser = ConfigParser("x")
    assert parser.get_setting("nope", "fallback") == "fallback"
    assert parser.get_setting("nope") is None


def test_get_setting_reads_loaded_settings():
    parser = ConfigParser("x")
    parser.run_settings = {"seed": 7}
    assert parser.get_setting("seed") == 7


def test_workflows_default_to_empty():
    """Layer C is opt-in; a parser that has not loaded anything must not
    claim workflows exist."""
    assert ConfigParser("x").workflows == []
