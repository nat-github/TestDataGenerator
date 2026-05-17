from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdp.generators.data_generator import DataGenerator
from sdp.utils.config_parser import ConfigParser
from sdp.utils.helpers import DataHelpers


def _write_columns_workbook(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Columns", index=False)


def test_helpers_generate_values_from_regex_rules() -> None:
    helpers = DataHelpers()

    samples = [
        helpers.generate_special_value("REGEX:AD\\d{3}", "A5", max_length=5)
        for _ in range(10)
    ]
    assert all(re.fullmatch(r"AD\d{3}", value) for value in samples)

    complex_samples = [
        helpers.generate_special_value(
            "REGEX:(INV|CRN)-[A-Z]{2}\\d{4}",
            "VA20",
            max_length=20,
        )
        for _ in range(10)
    ]
    assert all(re.fullmatch(r"(INV|CRN)-[A-Z]{2}\d{4}", value) for value in complex_samples)

    combined_samples = [
        helpers.generate_special_value(
            r"REGEX:ADR-\d{4}(-[A-Z]{2})?;;NULL_RATE=0.25",
            "VA20",
            max_length=20,
        )
        for _ in range(10)
    ]
    assert all(re.fullmatch(r"ADR-\d{4}(-[A-Z]{2})?", value) for value in combined_samples)
    assert helpers.get_special_rule_null_probability(r"REGEX:ADR-\d{4}(-[A-Z]{2})?;;NULL_RATE=0.25") == pytest.approx(0.25)


def test_config_parser_rejects_invalid_or_length_incompatible_regex(tmp_path: Path) -> None:
    invalid_length_workbook = tmp_path / "invalid_length.xlsx"
    _write_columns_workbook(
        invalid_length_workbook,
        [
            {
                "table_name": "demo",
                "column_name": "code",
                "data_type": "A5",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:[A-Z]{6}",
                "min_value": None,
                "max_value": None,
            }
        ],
    )

    parser = ConfigParser(str(invalid_length_workbook))
    assert parser.load_config() is True
    parser.parse_tables()
    parser.parse_relationships()
    assert parser.validate_config() is False

    unsupported_workbook = tmp_path / "unsupported_regex.xlsx"
    _write_columns_workbook(
        unsupported_workbook,
        [
            {
                "table_name": "demo",
                "column_name": "code",
                "data_type": "VA20",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:(?=AD)AD\\d{3}",
                "min_value": None,
                "max_value": None,
            }
        ],
    )

    parser = ConfigParser(str(unsupported_workbook))
    assert parser.load_config() is True
    parser.parse_tables()
    parser.parse_relationships()
    assert parser.validate_config() is False

    conflicting_workbook = tmp_path / "conflicting_special_rules.xlsx"
    _write_columns_workbook(
        conflicting_workbook,
        [
            {
                "table_name": "demo",
                "column_name": "code",
                "data_type": "VA20",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:AD\\d{3};;EMAIL",
                "min_value": None,
                "max_value": None,
            }
        ],
    )

    parser = ConfigParser(str(conflicting_workbook))
    assert parser.load_config() is True
    parser.parse_tables()
    parser.parse_relationships()
    assert parser.validate_config() is False

    pk_null_workbook = tmp_path / "pk_null_modifier.xlsx"
    _write_columns_workbook(
        pk_null_workbook,
        [
            {
                "table_name": "demo",
                "column_name": "demo_id",
                "data_type": "A5",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:AD\\d{3};;NULL_PCT=10",
                "min_value": None,
                "max_value": None,
            }
        ],
    )

    parser = ConfigParser(str(pk_null_workbook))
    assert parser.load_config() is True
    parser.parse_tables()
    parser.parse_relationships()
    assert parser.validate_config() is False


def test_data_generator_uses_regex_special_rules_for_pk_and_non_pk(tmp_path: Path) -> None:
    workbook = tmp_path / "regex_generation.xlsx"
    _write_columns_workbook(
        workbook,
        [
            {
                "table_name": "demo",
                "column_name": "demo_id",
                "data_type": "A5",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:AD\\d{3}",
                "min_value": None,
                "max_value": None,
            },
            {
                "table_name": "demo",
                "column_name": "status_code",
                "data_type": "A3",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:(ACT|NEW)",
                "min_value": None,
                "max_value": None,
            },
            {
                "table_name": "demo",
                "column_name": "reference_text",
                "data_type": "VA20",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:(INV|CRN)-\\d{6};;NULL_PCT=25",
                "min_value": None,
                "max_value": None,
            },
        ],
    )

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True

    data = generator.generate_data({"demo": 50})
    df = data["demo"]

    assert len(df) == 50
    assert df["demo_id"].nunique() == 50
    assert all(re.fullmatch(r"AD\d{3}", value) for value in df["demo_id"])
    assert all(re.fullmatch(r"(ACT|NEW)", value) for value in df["status_code"])
    non_null_reference_text = df["reference_text"].dropna()
    assert len(non_null_reference_text) < len(df)
    assert all(re.fullmatch(r"(INV|CRN)-\d{6}", value) for value in non_null_reference_text)


def test_sdv_generation_reconciles_regex_special_rules(tmp_path: Path) -> None:
    workbook = tmp_path / "regex_generation_sdv.xlsx"
    _write_columns_workbook(
        workbook,
        [
            {
                "table_name": "demo",
                "column_name": "demo_id",
                "data_type": "A5",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:AD\\d{3}",
                "min_value": None,
                "max_value": None,
            },
            {
                "table_name": "demo",
                "column_name": "status_code",
                "data_type": "A3",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:(ACT|NEW)",
                "min_value": None,
                "max_value": None,
            },
            {
                "table_name": "demo",
                "column_name": "reference_text",
                "data_type": "VA20",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:(INV|CRN)-\\d{6};;NULL_PCT=25",
                "min_value": None,
                "max_value": None,
            },
        ],
    )

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True

    class FakeSynthesizer:
        def sample(self, num_rows):
            count = num_rows["demo"]
            return {
                "demo": pd.DataFrame(
                    {
                        "demo_id": ["bad"] * count,
                        "status_code": ["ZZZ"] * count,
                        "reference_text": ["not-a-match"] * count,
                    }
                )
            }

    generator.synthesizer = cast(Any, FakeSynthesizer())
    generator.is_fitted = True

    data = generator.generate_data({"demo": 40})
    df = data["demo"]

    assert len(df) == 40
    assert df["demo_id"].nunique() == 40
    assert all(re.fullmatch(r"AD\d{3}", value) for value in df["demo_id"])
    assert all(re.fullmatch(r"(ACT|NEW)", value) for value in df["status_code"])
    non_null_reference_text = df["reference_text"].dropna()
    assert len(non_null_reference_text) < len(df)
    assert all(re.fullmatch(r"(INV|CRN)-\d{6}", value) for value in non_null_reference_text)

def test_fixed_length_text_columns_do_not_end_with_whitespace(tmp_path: Path) -> None:
    workbook = tmp_path / "fixed_length_text.xlsx"
    _write_columns_workbook(
        workbook,
        [
            {
                "table_name": "demo",
                "column_name": "demo_id",
                "data_type": "N10",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
            },
            {
                "table_name": "demo",
                "column_name": "status_code",
                "data_type": "A5",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "ACT;NEW",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
            },
            {
                "table_name": "demo",
                "column_name": "channel_code",
                "data_type": "AN8",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "AB12;ZX34",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
            },
        ],
    )

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True

    data = generator.generate_data({"demo": 20})
    df = data["demo"]

    assert not df["status_code"].str.endswith(" ").any()
    assert not df["channel_code"].str.endswith(" ").any()

    output_dir = tmp_path / "parquet_out"
    generator.export_to_parquet(str(output_dir))
    exported = pd.read_parquet(output_dir / "demo.parquet")

    assert not exported["status_code"].astype("string").str.endswith(" ").any()
    assert not exported["channel_code"].astype("string").str.endswith(" ").any()
