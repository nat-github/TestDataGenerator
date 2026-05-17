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
