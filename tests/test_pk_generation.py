"""Regression tests for primary-key generation integrity.

Guards the bug where the SDV generation path emitted opaque string ids
("sdv-id-XXXX") for primary-key columns. Because the declared data type was
numeric, the cast on Parquet export wiped every value to NaN — primary keys
came out 100% null.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sdp.generators.data_generator import DataGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_USERS = REPO_ROOT / "examples" / "configs" / "yaml" / "01_simple_users.yaml"


def _generate(num_rows: int):
    """Generate the `users` table via the SDV path; return (generator, fitted, df)."""
    gen = DataGenerator(str(SIMPLE_USERS), seed=42)
    assert gen.load_configuration()
    gen.create_sdv_metadata()
    fitted = gen.train_synthesizer()
    data = gen.generate_data({"users": num_rows})
    return gen, fitted, data["users"]


def test_sdv_path_primary_key_is_not_null_and_unique():
    gen, fitted, df = _generate(30)
    # The bug was SDV-path specific — make sure we actually exercised it.
    assert fitted, "expected the SDV synthesizer to fit for this config"
    pk = df["user_id"]
    assert pk.isna().sum() == 0, "primary key column must have no nulls"
    assert pk.nunique() == len(pk), "primary key values must be unique"


def test_primary_key_survives_parquet_export(tmp_path: Path):
    gen, _fitted, _df = _generate(40)
    gen.export_to_parquet(str(tmp_path))

    pk = pd.read_parquet(tmp_path / "users.parquet")["user_id"]
    assert pk.isna().sum() == 0, "primary key must survive the Parquet export cast"
    assert pk.nunique() == len(pk), "primary key values must stay unique after export"
    # user_id is declared N10 (numeric) — values must be numeric, not 'sdv-id-...'.
    assert pd.to_numeric(pk, errors="coerce").notna().all(), \
        "numeric-typed primary key must hold numeric values"


# A one-to-one / shared-primary-key schema: person_detail.person_id is BOTH the
# child's primary key AND a foreign key to person.person_id.
_ONE_TO_ONE_YAML = """\
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 60
tables:
  - name: person
    rows: 60
    primary_key_columns: [person_id]
    columns:
      - name: person_id
        data_type: N10
        is_pk: true
        nullable: false
      - name: full_name
        data_type: VA64
        special_rules: NAME
  - name: person_detail
    rows: 60
    primary_key_columns: [person_id]
    columns:
      - name: person_id
        data_type: N10
        is_pk: true
        is_fk: true
        ref_table: person
        ref_column: person_id
        nullable: false
      - name: detail_note
        data_type: VA64
        special_rules: NAME
"""


def test_one_to_one_fk_keeps_child_primary_key_unique(tmp_path: Path):
    """Regression: when a child's FK column is also its sole primary key
    (one-to-one / shared-PK), FK resolution must assign *unique* parent keys.
    It previously sampled with replacement, duplicating the child primary key.
    """
    cfg = tmp_path / "one_to_one.yaml"
    cfg.write_text(_ONE_TO_ONE_YAML, encoding="utf-8")

    gen = DataGenerator(str(cfg), seed=42)
    assert gen.load_configuration()
    gen.create_sdv_metadata()
    gen.train_synthesizer()
    data = gen.generate_data({"person": 60, "person_detail": 60})

    parent_ids = set(data["person"]["person_id"])
    child_pk = data["person_detail"]["person_id"]

    assert child_pk.isna().sum() == 0, "child primary key must have no nulls"
    assert child_pk.nunique() == len(child_pk), \
        "child PK is also the FK — it must stay unique (one-to-one)"
    assert set(child_pk).issubset(parent_ids), \
        "every child FK value must reference an existing parent key"


# ---------------------------------------------------------------------------
# Composite primary keys — only the COMBINATION of members must be unique.
# The uniqueness fixer must never rewrite a foreign-key member (that would
# break referential integrity), and members may legitimately repeat.
# ---------------------------------------------------------------------------
_COMPOSITE_PK_YAML = """\
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 20
tables:
  - name: orders
    rows: 20
    primary_key_columns: [order_id]
    columns:
      - name: order_id
        data_type: N10
        is_pk: true
        nullable: false
  - name: order_lines
    rows: 20
    primary_key_columns: [order_id, line_no]
    columns:
      - name: order_id
        data_type: N10
        is_pk: true
        is_fk: true
        ref_table: orders
        ref_column: order_id
        nullable: false
      - name: line_no
        data_type: N10
        is_pk: true
        nullable: false
      - name: note
        data_type: VA64
        special_rules: NAME
"""


def _load_generator(tmp_path: Path, yaml_text: str) -> DataGenerator:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    gen = DataGenerator(str(cfg), seed=7)
    assert gen.load_configuration()
    return gen


def test_composite_pk_fixer_dedups_combination_not_members(tmp_path: Path):
    """A duplicate composite-PK *combination* is fixed; the FK member is kept."""
    gen = _load_generator(tmp_path, _COMPOSITE_PK_YAML)
    tc = gen.tables_config["order_lines"]

    # (10, 1) appears twice — the composite key is violated. order_id repeats
    # legitimately (one order, many lines) and must NOT be rewritten.
    df = pd.DataFrame({
        "order_id": [10, 10, 10, 20, 20],
        "line_no":  [1,  1,  2,  1,  2],
        "note": list("abcde"),
    })
    original_order_id = list(df["order_id"])

    fixed = gen._validate_and_fix_pk_uniqueness(df, tc)

    assert list(fixed["order_id"]) == original_order_id, \
        "composite-PK fix must not rewrite the foreign-key member"
    combos = fixed[["order_id", "line_no"]]
    assert len(combos) == len(combos.drop_duplicates()), \
        "composite PK combination must be unique after the fix"


def test_composite_pk_fixer_is_noop_when_already_unique(tmp_path: Path):
    """A valid composite PK (members repeat, combinations unique) is untouched."""
    gen = _load_generator(tmp_path, _COMPOSITE_PK_YAML)
    tc = gen.tables_config["order_lines"]

    df = pd.DataFrame({
        "order_id": [10, 10, 20, 20, 30],
        "line_no":  [1,  2,  1,  2,  1],
        "note": list("abcde"),
    })
    before = df.copy()
    fixed = gen._validate_and_fix_pk_uniqueness(df, tc)
    pd.testing.assert_frame_equal(fixed[["order_id", "line_no"]],
                                  before[["order_id", "line_no"]])


def test_composite_pk_generation_end_to_end_keeps_fk_integrity(tmp_path: Path):
    """Full generate: composite-PK combinations unique, FK member valid."""
    gen = _load_generator(tmp_path, _COMPOSITE_PK_YAML)
    gen.create_sdv_metadata()
    gen.train_synthesizer()
    data = gen.generate_data({"orders": 20, "order_lines": 40})

    lines = data["order_lines"]
    combos = lines[["order_id", "line_no"]]
    assert len(combos) == len(combos.drop_duplicates()), \
        "generated composite PK must be unique as a combination"
    parent_ids = set(data["orders"]["order_id"])
    assert set(lines["order_id"]).issubset(parent_ids), \
        "composite-PK FK member must reference an existing parent key"


# ---------------------------------------------------------------------------
# NOT NULL precedence — a `nullable: false` column never receives nulls, even
# when a contradicting NULL_PCT special rule is also declared on it.
# ---------------------------------------------------------------------------
_NULL_PRECEDENCE_YAML = """\
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 200
tables:
  - name: events
    rows: 200
    primary_key_columns: [event_id]
    columns:
      - name: event_id
        data_type: N10
        is_pk: true
        nullable: false
      - name: required_code
        data_type: VA8
        nullable: false
        special_rules: "NULL_PCT=50"
      - name: optional_code
        data_type: VA8
        nullable: true
        special_rules: "NULL_PCT=50"
"""


def test_not_null_column_ignores_null_pct_rule(tmp_path: Path):
    """`nullable: false` must override a contradicting NULL_PCT special rule."""
    gen = _load_generator(tmp_path, _NULL_PRECEDENCE_YAML)
    by_name = {c.column_name: c for c in gen.tables_config["events"].columns}

    # The hard NOT NULL constraint forces a 0 null probability...
    assert gen._get_null_probability(by_name["required_code"]) == 0.0
    # ...while a nullable column still honours the NULL_PCT rule.
    assert gen._get_null_probability(by_name["optional_code"]) == pytest.approx(0.5)


def test_not_null_column_has_no_nulls_despite_null_pct(tmp_path: Path):
    """End-to-end: a NOT NULL column with a NULL_PCT rule generates zero nulls."""
    gen = _load_generator(tmp_path, _NULL_PRECEDENCE_YAML)
    gen.create_sdv_metadata()
    gen.train_synthesizer()
    df = gen.generate_data({"events": 200})["events"]

    assert df["required_code"].isna().sum() == 0, \
        "NOT NULL column must have no nulls even with a NULL_PCT rule"
