"""Tests for anchored generation.

Anchored generation lets a table declare ``source: <path>`` in the config. That
table is loaded verbatim from a real ``.parquet`` / ``.csv`` file instead of
being generated; the other tables generate around it and resolve their foreign
keys against the anchor's *real* key values. The real rows also feed SDV training.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sdp.generators.data_generator import DataGenerator


def _anchor_customers() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [101, 202, 303, 404, 505],
        "full_name": ["Ann", "Bob", "Cara", "Dan", "Eve"],
    })


def _config_yaml(source_path: str) -> str:
    return f"""\
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 40
tables:
  - name: customer
    source: {source_path}
    primary_key_columns: [customer_id]
    columns:
      - name: customer_id
        data_type: N10
        is_pk: true
        nullable: false
      - name: full_name
        data_type: VA64
  - name: account
    rows: 40
    primary_key_columns: [account_id]
    columns:
      - name: account_id
        data_type: N10
        is_pk: true
        nullable: false
      - name: customer_id
        data_type: N10
        is_fk: true
        ref_table: customer
        ref_column: customer_id
        nullable: false
      - name: balance
        data_type: DC
        min_value: 0
        max_value: 10000
"""


def _write_config(tmp_path: Path, anchor: pd.DataFrame, *, fmt: str = "parquet") -> Path:
    src = tmp_path / f"customers.{fmt}"
    if fmt == "parquet":
        anchor.to_parquet(src, index=False)
    else:
        anchor.to_csv(src, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_config_yaml(src.as_posix()), encoding="utf-8")
    return cfg


def _run(cfg_path: Path, counts):
    gen = DataGenerator(str(cfg_path), seed=42)
    assert gen.load_configuration()
    gen.create_sdv_metadata()
    gen.train_synthesizer()
    return gen, gen.generate_data(counts)


def test_anchor_table_is_used_verbatim(tmp_path: Path):
    anchor = _anchor_customers()
    cfg = _write_config(tmp_path, anchor)

    gen, data = _run(cfg, {"customer": 40, "account": 40})

    out_customer = data["customer"]
    # The anchor table comes out exactly as the source file — rows, keys, values.
    assert len(out_customer) == len(anchor)
    assert set(out_customer["customer_id"]) == set(anchor["customer_id"])
    assert list(out_customer["full_name"]) == list(anchor["full_name"])


def test_anchor_table_survives_parquet_export(tmp_path: Path):
    anchor = _anchor_customers()
    cfg = _write_config(tmp_path, anchor)

    gen, _data = _run(cfg, {"customer": 40, "account": 40})
    gen.export_to_parquet(str(tmp_path / "out"))

    exported = pd.read_parquet(tmp_path / "out" / "customer.parquet")
    assert set(exported["customer_id"]) == set(anchor["customer_id"])


def test_child_fk_references_only_real_anchor_keys(tmp_path: Path):
    anchor = _anchor_customers()
    cfg = _write_config(tmp_path, anchor)

    gen, data = _run(cfg, {"customer": 40, "account": 40})

    real_keys = set(anchor["customer_id"])
    child_fk = set(data["account"]["customer_id"])
    assert child_fk, "account table generated no FK values"
    assert child_fk.issubset(real_keys), \
        f"child FK values not in real anchor keys: {child_fk - real_keys}"


def test_csv_source_is_supported(tmp_path: Path):
    anchor = _anchor_customers()
    cfg = _write_config(tmp_path, anchor, fmt="csv")

    gen = DataGenerator(str(cfg), seed=1)
    assert gen.load_configuration()
    assert "customer" in gen.anchor_data
    assert set(gen.anchor_data["customer"]["customer_id"]) == set(anchor["customer_id"])


def test_missing_source_file_raises(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_config_yaml((tmp_path / "nope.parquet").as_posix()), encoding="utf-8")

    gen = DataGenerator(str(cfg), seed=1)
    # load_configuration swallows the error — call the loader directly to assert it.
    gen.config_parser.load_config()
    gen.tables_config = gen.config_parser.parse_tables()
    with pytest.raises(FileNotFoundError):
        gen._load_anchor_tables()


def test_source_missing_configured_column_raises(tmp_path: Path):
    bad = pd.DataFrame({"customer_id": [1, 2, 3]})  # full_name column is absent
    src = tmp_path / "customers.parquet"
    bad.to_parquet(src, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_config_yaml(src.as_posix()), encoding="utf-8")

    gen = DataGenerator(str(cfg), seed=1)
    gen.config_parser.load_config()
    gen.tables_config = gen.config_parser.parse_tables()
    with pytest.raises(ValueError, match="missing configured"):
        gen._load_anchor_tables()
