"""Regression tests for primary-key generation integrity.

Guards the bug where the SDV generation path emitted opaque string ids
("sdv-id-XXXX") for primary-key columns. Because the declared data type was
numeric, the cast on Parquet export wiped every value to NaN — primary keys
came out 100% null.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

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
