"""Tests for the new CDC simplification, rules/derived columns, and JSON loader.

Covers:
  - new `cdc:` block in YAML and JSON authoring formats
  - legacy flat fields still working (delta / scd2 / scd2_tracked_columns)
  - Excel cdc_mode + cdc_track alias columns
  - column-level rules (when/then) and derived expressions
  - JSON loader round-trip
  - rule evaluator unit tests (operators, derived expressions, topo sort)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from sdp.models.config_models import CDCConfig, ColumnConfig, RuleAction, RuleConfig, TableConfig
from sdp.utils.config_parser import ConfigParser
from sdp.utils.rule_evaluator import (
    apply_to_dataframe,
    apply_when_then,
    evaluate_derived,
    evaluate_when,
    topo_sort_derived,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config_dict(use_cdc_block: bool = True) -> dict:
    """Two-table config with one rule and one derived column."""
    cfg: dict[str, Any] = {
        "config_format": "sdp-yaml-v1",
        "run_settings": {"default_records_per_table": 50},
        "tables": [
            {
                "name": "accounts",
                "rows": 50,
                "primary_key_columns": ["account_id"],
                "columns": [
                    {"name": "account_id", "data_type": "N10", "is_pk": True, "nullable": False},
                    {"name": "status", "data_type": "VA10", "values": ["ACTIVE", "CLOSED"], "nullable": False},
                    {
                        "name": "closure_date",
                        "data_type": "D",
                        "nullable": True,
                        "rules": [
                            {"when": {"status": {"eq": "ACTIVE"}}, "then": {"set_null": True}},
                            {"when": {"status": {"eq": "CLOSED"}}, "then": {"value": "2024-06-15"}},
                        ],
                    },
                    {"name": "first_name", "data_type": "VA32"},
                    {"name": "last_name", "data_type": "VA32"},
                    {"name": "full_name", "data_type": "VA64", "derived": "{first_name} {last_name}"},
                ],
            },
        ],
    }
    if use_cdc_block:
        cfg["tables"][0]["cdc"] = {
            "mode": "scd2",
            "track": ["status", "closure_date"],
            "event_time": "account_id",
            "partition_by": [],
        }
    else:
        cfg["tables"][0]["scd2"] = True
        cfg["tables"][0]["track_changes"] = ["status", "closure_date"]
    return cfg


# ---------------------------------------------------------------------------
# CDC block — YAML / JSON / Excel-aliases
# ---------------------------------------------------------------------------


def test_cdc_block_yaml_populates_cdc_object_and_legacy_fields(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(use_cdc_block=True)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))

    parser = ConfigParser(str(yaml_path))
    assert parser.load_config()
    tables = parser.parse_tables()

    accounts = tables["accounts"]
    # New unified field
    assert accounts.cdc is not None
    assert accounts.cdc.mode == "scd2"
    assert accounts.cdc.track == ["status", "closure_date"]
    assert accounts.cdc.event_time == "account_id"
    # Legacy fields back-filled for backward compat
    assert accounts.scd2_enabled is True
    assert accounts.delta_eligible is True   # scd2 implies delta
    assert accounts.scd2_tracked_columns == ["status", "closure_date"]


def test_legacy_flat_fields_still_work(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(use_cdc_block=False)
    yaml_path = tmp_path / "legacy.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))

    parser = ConfigParser(str(yaml_path))
    assert parser.load_config()
    tables = parser.parse_tables()
    accounts = tables["accounts"]

    assert accounts.scd2_enabled is True
    assert accounts.scd2_tracked_columns == ["status", "closure_date"]
    # Even without explicit cdc:, parser should synthesize a CDCConfig
    assert accounts.cdc is not None
    assert accounts.cdc.mode == "scd2"


def test_json_config_loads_with_cdc_and_rules(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(use_cdc_block=True)
    json_path = tmp_path / "cfg.json"
    json_path.write_text(json.dumps(cfg))

    parser = ConfigParser(str(json_path))
    assert parser.load_config()
    tables = parser.parse_tables()
    accounts = tables["accounts"]

    closure = next(c for c in accounts.columns if c.column_name == "closure_date")
    assert closure.rules is not None
    assert len(closure.rules) == 2
    assert closure.rules[0].when == {"status": {"eq": "ACTIVE"}}
    assert closure.rules[0].then.set_null is True


def test_excel_cdc_mode_and_cdc_track_aliases(tmp_path: Path) -> None:
    """The Excel Tables sheet supports two new optional columns: cdc_mode + cdc_track."""
    columns_df = pd.DataFrame(
        [
            {"table_name": "accounts", "column_name": "account_id", "data_type": "N10", "is_pk": True, "nullable": False},
            {"table_name": "accounts", "column_name": "status", "data_type": "VA10", "business_values": "A;B"},
            {"table_name": "accounts", "column_name": "balance", "data_type": "N10"},
        ]
    )
    tables_df = pd.DataFrame(
        [
            {
                "table_name": "accounts",
                "row_count": 10,
                "cdc_mode": "scd2",
                "cdc_track": "status;balance",
            }
        ]
    )
    xlsx = tmp_path / "cdc_excel.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)
        tables_df.to_excel(writer, sheet_name="Tables", index=False)

    parser = ConfigParser(str(xlsx))
    assert parser.load_config()
    tables = parser.parse_tables()
    accounts = tables["accounts"]

    assert accounts.scd2_enabled is True
    assert accounts.delta_eligible is True
    assert accounts.scd2_tracked_columns == ["status", "balance"]
    assert accounts.cdc is not None
    assert accounts.cdc.mode == "scd2"


def test_excel_rules_column_as_json_string(tmp_path: Path) -> None:
    rules_payload = json.dumps([
        {"when": {"status": {"eq": "CLOSED"}}, "then": {"set_null": True}},
    ])
    columns_df = pd.DataFrame(
        [
            {"table_name": "accounts", "column_name": "account_id", "data_type": "N10", "is_pk": True, "nullable": False},
            {"table_name": "accounts", "column_name": "status", "data_type": "VA10", "business_values": "ACTIVE;CLOSED"},
            {"table_name": "accounts", "column_name": "closure_date", "data_type": "D", "nullable": True, "rules": rules_payload},
        ]
    )
    xlsx = tmp_path / "rules_excel.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)

    parser = ConfigParser(str(xlsx))
    assert parser.load_config()
    tables = parser.parse_tables()
    closure = next(c for c in tables["accounts"].columns if c.column_name == "closure_date")
    assert closure.rules is not None and len(closure.rules) == 1
    assert closure.rules[0].then.set_null is True


# ---------------------------------------------------------------------------
# Rule evaluator unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row,when,expected",
    [
        ({"status": "ACTIVE"}, {"status": {"eq": "ACTIVE"}}, True),
        ({"status": "ACTIVE"}, {"status": {"eq": "CLOSED"}}, False),
        ({"status": "ACTIVE"}, {"status": "ACTIVE"}, True),  # shorthand eq
        ({"tier": "GOLD"}, {"tier": {"in": ["GOLD", "PLATINUM"]}}, True),
        ({"tier": "SILVER"}, {"tier": {"in": ["GOLD", "PLATINUM"]}}, False),
        ({"balance": 200}, {"balance": {"gt": 100}}, True),
        ({"balance": 50}, {"balance": {"gt": 100}}, False),
        ({"balance": "150"}, {"balance": {"gte": 100}}, True),  # string coercion
        ({"balance": 150}, {"balance": {"between": [100, 200]}}, True),
        ({"balance": 250}, {"balance": {"between": [100, 200]}}, False),
        ({"closed_at": None}, {"closed_at": {"is_null": True}}, True),
        ({"closed_at": "2024-01-01"}, {"closed_at": {"is_null": True}}, False),
        ({"phone": "+1-555-1234"}, {"phone": {"matches": r"^\+1-"}}, True),
    ],
)
def test_evaluate_when_operators(row: dict, when: dict, expected: bool) -> None:
    assert evaluate_when(when, row) is expected


def test_evaluate_when_and_or_composition() -> None:
    row = {"status": "ACTIVE", "tier": "GOLD"}
    assert evaluate_when(
        {"and": [{"status": {"eq": "ACTIVE"}}, {"tier": {"in": ["GOLD"]}}]},
        row,
    ) is True
    assert evaluate_when(
        {"or": [{"status": {"eq": "CLOSED"}}, {"tier": {"in": ["GOLD"]}}]},
        row,
    ) is True
    assert evaluate_when(
        {"and": [{"status": {"eq": "CLOSED"}}, {"tier": {"in": ["GOLD"]}}]},
        row,
    ) is False


def test_apply_when_then_sets_null_and_constant() -> None:
    row = {"status": "CLOSED", "closed_at": "2024-01-01"}
    rule = RuleConfig(when={"status": {"eq": "CLOSED"}}, then=RuleAction(set_null=True))
    fired = apply_when_then(rule, "closed_at", row)
    assert fired is True
    assert row["closed_at"] is None

    row2 = {"status": "PROCESSING"}
    rule2 = RuleConfig(when={"status": {"eq": "PROCESSING"}}, then=RuleAction(value="P"))
    apply_when_then(rule2, "stage", row2)
    assert row2["stage"] == "P"


def test_evaluate_derived_template_and_expression() -> None:
    row = {"first_name": "Ada", "last_name": "Lovelace", "qty": 3, "price": 12.5}
    assert evaluate_derived("{first_name} {last_name}", row) == "Ada Lovelace"
    assert evaluate_derived("={qty} * {price}", row) == 37.5
    assert evaluate_derived("=upper({first_name})", row) == "ADA"
    assert evaluate_derived("=concat({first_name}, ' ', {last_name})", row) == "Ada Lovelace"


def test_topo_sort_derived_handles_dependencies() -> None:
    cols = [
        ColumnConfig(table_name="t", column_name="full_name", data_type="VA64",
                     derived="{first_name} {last_name}"),
        ColumnConfig(table_name="t", column_name="display", data_type="VA64",
                     derived="=upper({full_name})"),
        ColumnConfig(table_name="t", column_name="first_name", data_type="VA32"),
        ColumnConfig(table_name="t", column_name="last_name", data_type="VA32"),
    ]
    ordered = topo_sort_derived(cols)
    names = [c.column_name for c in ordered]
    # full_name must appear before display
    assert names.index("full_name") < names.index("display")


def test_apply_to_dataframe_layer_a_and_b() -> None:
    cols = [
        ColumnConfig(table_name="t", column_name="status", data_type="VA10"),
        ColumnConfig(table_name="t", column_name="closed_at", data_type="D",
                     rules=[
                         RuleConfig(when={"status": {"eq": "ACTIVE"}}, then=RuleAction(set_null=True)),
                         RuleConfig(when={"status": {"eq": "CLOSED"}}, then=RuleAction(value="2024-06-15")),
                     ]),
        ColumnConfig(table_name="t", column_name="first", data_type="VA10"),
        ColumnConfig(table_name="t", column_name="last", data_type="VA10"),
        ColumnConfig(table_name="t", column_name="full", data_type="VA32",
                     derived="{first} {last}"),
    ]
    tc = TableConfig(name="t", columns=cols)
    df = pd.DataFrame([
        {"status": "ACTIVE", "closed_at": "1999-01-01", "first": "Ada", "last": "Lovelace", "full": ""},
        {"status": "CLOSED", "closed_at": "1999-01-01", "first": "Alan", "last": "Turing", "full": ""},
    ])
    out = apply_to_dataframe(df, tc)
    assert out.loc[0, "closed_at"] is None
    assert out.loc[1, "closed_at"] == "2024-06-15"
    assert out.loc[0, "full"] == "Ada Lovelace"
    assert out.loc[1, "full"] == "Alan Turing"


def test_cdc_config_implies_delta_when_scd2() -> None:
    cdc = CDCConfig(mode="scd2", track=["a", "b"])
    assert cdc.mode == "scd2"
    # The parser is what sets delta_eligible=True for scd2; here we test the model
    # accepts the values as authored.
    assert cdc.track == ["a", "b"]


def test_cdc_config_normalizes_string_lists() -> None:
    cdc = CDCConfig(mode="scd2", track="a;b;c", partition_by="x;y")
    assert cdc.track == ["a", "b", "c"]
    assert cdc.partition_by == ["x", "y"]


def test_derived_with_unknown_name_returns_none_or_empty() -> None:
    """Tolerant: typos shouldn't crash a 1M-row run."""
    row = {"first": "A", "last": "B"}
    # Unknown name in expression → None
    assert evaluate_derived("=missing_col + 1", row) is None
    # Unknown name in template → empty string substitution
    assert evaluate_derived("{first}-{missing}", row) == "A-"
