from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from deltalake import DeltaTable

from sdp.generators.data_generator import DataGenerator
from main import parse_arguments, validate_config_file
from sdp.utils.config_parser import ConfigParser
from sdp.utils.helpers import DataHelpers
from sdp.utils.parquet_post_processor import ParquetPostProcessor


def _write_workbook(path: Path, with_optional_sheets: bool = True) -> None:
    columns_df = pd.DataFrame(
        [
            {
                "table_name": "parent",
                "column_name": "id",
                "data_type": "N10",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "parent",
                "column_name": "status",
                "data_type": "VA10",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "A;B",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": True,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": True,
            },
            {
                "table_name": "child",
                "column_name": "id",
                "data_type": "N10",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "parent_id",
                "data_type": "N10",
                "is_pk": False,
                "is_fk": True,
                "ref_table": "parent",
                "ref_column": "id",
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": True,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)
        if with_optional_sheets:
            pd.DataFrame(
                [
                    {"setting_name": "default_records_per_table", "value": 77, "description": ""},
                    {"setting_name": "operation_column", "value": "operation_type", "description": ""},
                ]
            ).to_excel(writer, sheet_name="Run_Settings", index=False)
            pd.DataFrame(
                [
                    {
                        "table_name": "parent",
                        "table_kind": "dimension",
                        "row_count": 12,
                        "generation_mode": "scd2_ready",
                        "business_key_columns": "id",
                        "primary_key_columns": "id",
                        "partition_enabled": False,
                        "partition_columns": "",
                        "event_time_column": "",
                        "scd2_enabled": True,
                        "scd2_tracked_columns": "status",
                        "delta_eligible": True,
                        "active": True,
                    },
                    {
                        "table_name": "child",
                        "table_kind": "transactional",
                        "row_count": 33,
                        "generation_mode": "delta_ready",
                        "business_key_columns": "id",
                        "primary_key_columns": "id",
                        "partition_enabled": False,
                        "partition_columns": "",
                        "event_time_column": "",
                        "scd2_enabled": False,
                        "scd2_tracked_columns": "",
                        "delta_eligible": True,
                        "active": True,
                    },
                ]
            ).to_excel(writer, sheet_name="Tables", index=False)
            pd.DataFrame(
                [
                    {
                        "relationship_name": "child_parent",
                        "source_table": "child",
                        "source_columns": "parent_id",
                        "target_table": "parent",
                        "target_columns": "id",
                        "cardinality": "many_to_one",
                        "preserve_on_delta": True,
                        "active": True,
                    }
                ]
            ).to_excel(writer, sheet_name="Relationships", index=False)


def _write_yaml_config(path: Path, with_optional_sections: bool = True) -> None:
    config: dict[str, Any] = {
        "config_format": "sdp-yaml-v1",
        "tables": [
            {
                "name": "parent",
                "table_kind": "dimension",
                "row_count": 12,
                "generation_mode": "scd2_ready",
                "business_key_columns": ["id"],
                "primary_key_columns": ["id"],
                "partition_enabled": False,
                "partition_columns": [],
                "event_time_column": None,
                "scd2_enabled": True,
                "scd2_tracked_columns": ["status"],
                "delta_eligible": True,
                "active": True,
                "columns": [
                    {
                        "name": "id",
                        "data_type": "N10",
                        "is_pk": True,
                        "nullable": False,
                        "is_business_key_component": True,
                    },
                    {
                        "name": "status",
                        "data_type": "VA10",
                        "business_values": "A;B",
                        "nullable": True,
                        "scd2_tracked": True,
                    },
                ],
            },
            {
                "name": "child",
                "table_kind": "transactional",
                "row_count": 33,
                "generation_mode": "delta_ready",
                "business_key_columns": ["id"],
                "primary_key_columns": ["id"],
                "partition_enabled": False,
                "partition_columns": [],
                "event_time_column": None,
                "scd2_enabled": False,
                "scd2_tracked_columns": [],
                "delta_eligible": True,
                "active": True,
                "columns": [
                    {
                        "name": "id",
                        "data_type": "N10",
                        "is_pk": True,
                        "nullable": False,
                        "is_business_key_component": True,
                    },
                    {
                        "name": "parent_id",
                        "data_type": "N10",
                        "is_fk": True,
                        "ref_table": "parent",
                        "ref_column": "id",
                        "nullable": True,
                    },
                ],
            },
        ],
    }

    if with_optional_sections:
        config["run_settings"] = {
            "default_records_per_table": 77,
            "operation_column": "operation_type",
        }
        config["relationships"] = [
            {
                "name": "child_parent",
                "source_table": "child",
                "source_columns": ["parent_id"],
                "target_table": "parent",
                "target_columns": ["id"],
                "relationship_type": "many_to_one",
                "preserve_on_delta": True,
                "active": True,
            }
        ]

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_sdv_edge_case_workbook(path: Path) -> None:
    columns_df = pd.DataFrame(
        [
            {
                "table_name": "parent",
                "column_name": "id",
                "data_type": "N10",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "parent",
                "column_name": "status_code",
                "data_type": "VA10",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "NEW;ACTIVE;CLOSED",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "parent",
                "column_name": "event_ts",
                "data_type": "TS",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "9999-12-31 00:00:00;not-a-date;2024-01-01 10:30:00",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": True,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "id",
                "data_type": "N10",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "parent_id",
                "data_type": "N10",
                "is_pk": False,
                "is_fk": True,
                "ref_table": "parent",
                "ref_column": "id",
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "status_code",
                "data_type": "VA10",
                "is_pk": False,
                "is_fk": True,
                "ref_table": "parent",
                "ref_column": "status_code",
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
        ]
    )

    tables_df = pd.DataFrame(
        [
            {
                "table_name": "parent",
                "table_kind": "dimension",
                "row_count": 12,
                "generation_mode": "snapshot",
                "business_key_columns": "id;status_code",
                "primary_key_columns": "id;status_code",
                "partition_enabled": False,
                "partition_columns": "",
                "event_time_column": "event_ts",
                "scd2_enabled": False,
                "scd2_tracked_columns": "",
                "delta_eligible": False,
                "active": True,
            },
            {
                "table_name": "child",
                "table_kind": "transactional",
                "row_count": 16,
                "generation_mode": "snapshot",
                "business_key_columns": "id",
                "primary_key_columns": "id",
                "partition_enabled": False,
                "partition_columns": "",
                "event_time_column": "",
                "scd2_enabled": False,
                "scd2_tracked_columns": "",
                "delta_eligible": False,
                "active": True,
            },
        ]
    )

    relationships_df = pd.DataFrame(
        [
            {
                "relationship_name": "child_parent_composite",
                "source_table": "child",
                "source_columns": "parent_id;status_code",
                "target_table": "parent",
                "target_columns": "id;status_code",
                "cardinality": "many_to_one",
                "preserve_on_delta": True,
                "active": True,
            }
        ]
    )

    run_settings_df = pd.DataFrame(
        [
            {"setting_name": "default_records_per_table", "value": 12, "description": ""},
            {"setting_name": "synthesizer_sample_size", "value": 20, "description": ""},
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)
        tables_df.to_excel(writer, sheet_name="Tables", index=False)
        relationships_df.to_excel(writer, sheet_name="Relationships", index=False)
        run_settings_df.to_excel(writer, sheet_name="Run_Settings", index=False)


def _write_business_key_only_workbook(path: Path) -> None:
    columns_df = pd.DataFrame(
        [
            {
                "table_name": "codes",
                "column_name": "group_code",
                "data_type": "N4",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": 1000,
                "max_value": 1002,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "codes",
                "column_name": "description",
                "data_type": "VA20",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "A;B;C",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
        ]
    )

    tables_df = pd.DataFrame(
        [
            {
                "table_name": "codes",
                "table_kind": "dimension",
                "row_count": 12,
                "generation_mode": "snapshot",
                "business_key_columns": "group_code",
                "primary_key_columns": "",
                "partition_enabled": False,
                "partition_columns": "",
                "event_time_column": "",
                "scd2_enabled": False,
                "scd2_tracked_columns": "",
                "delta_eligible": False,
                "active": True,
            }
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)
        tables_df.to_excel(writer, sheet_name="Tables", index=False)


def _write_regex_relationship_workbook(path: Path) -> None:
    columns_df = pd.DataFrame(
        [
            {
                "table_name": "parent",
                "column_name": "parent_code",
                "data_type": "A7",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:PAR\\d{4}",
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "parent",
                "column_name": "status",
                "data_type": "A3",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "ACT;NEW",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "child_id",
                "data_type": "N10",
                "is_pk": True,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": True,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "parent_code",
                "data_type": "A7",
                "is_pk": False,
                "is_fk": True,
                "ref_table": "parent",
                "ref_column": "parent_code",
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "child",
                "column_name": "doc_code",
                "data_type": "VA10",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "REGEX:CHD-\\d{3}",
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
        ]
    )

    tables_df = pd.DataFrame(
        [
            {
                "table_name": "parent",
                "table_kind": "dimension",
                "row_count": 8,
                "generation_mode": "snapshot",
                "business_key_columns": "parent_code",
                "primary_key_columns": "parent_code",
                "partition_enabled": False,
                "partition_columns": "",
                "event_time_column": "",
                "scd2_enabled": False,
                "scd2_tracked_columns": "",
                "delta_eligible": False,
                "active": True,
            },
            {
                "table_name": "child",
                "table_kind": "transactional",
                "row_count": 12,
                "generation_mode": "snapshot",
                "business_key_columns": "child_id",
                "primary_key_columns": "child_id",
                "partition_enabled": False,
                "partition_columns": "",
                "event_time_column": "",
                "scd2_enabled": False,
                "scd2_tracked_columns": "",
                "delta_eligible": False,
                "active": True,
            },
        ]
    )

    relationships_df = pd.DataFrame(
        [
            {
                "relationship_name": "child_parent_code",
                "source_table": "child",
                "source_columns": "parent_code",
                "target_table": "parent",
                "target_columns": "parent_code",
                "cardinality": "many_to_one",
                "preserve_on_delta": True,
                "active": True,
            }
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)
        tables_df.to_excel(writer, sheet_name="Tables", index=False)
        relationships_df.to_excel(writer, sheet_name="Relationships", index=False)


def test_config_parser_supports_legacy_columns_only(tmp_path: Path):
    workbook = tmp_path / "legacy.xlsx"
    _write_workbook(workbook, with_optional_sheets=False)

    parser = ConfigParser(str(workbook))
    assert parser.load_config() is True
    tables = parser.parse_tables()
    relationships = parser.parse_relationships()

    assert set(tables.keys()) == {"parent", "child"}
    assert tables["parent"].num_rows == 1000
    assert len(relationships) == 1
    assert relationships[0].source_column == "parent_id"


def test_config_parser_reads_optional_sheets_and_legacy_cli(tmp_path: Path):
    workbook = tmp_path / "enhanced.xlsx"
    _write_workbook(workbook, with_optional_sheets=True)

    parser = ConfigParser(str(workbook))
    assert parser.load_config() is True
    tables = parser.parse_tables()
    relationships = parser.parse_relationships()
    assert parser.validate_config() is True

    assert parser.get_setting("default_records_per_table") == 77
    assert tables["parent"].num_rows == 12
    assert tables["parent"].scd2_enabled is True
    assert tables["child"].delta_eligible is True
    assert len(relationships) == 1
    assert relationships[0].preserve_on_delta is True

    args = parse_arguments(["--config", str(workbook), "--output", str(tmp_path / "out")])
    assert args.command == "generate"


def test_config_parser_reads_yaml_and_generation_uses_same_core_logic(tmp_path: Path):
    config_file = tmp_path / "enhanced.yaml"
    _write_yaml_config(config_file, with_optional_sections=True)

    parser = ConfigParser(str(config_file))
    assert parser.load_config() is True
    tables = parser.parse_tables()
    relationships = parser.parse_relationships()
    assert parser.validate_config() is True

    assert validate_config_file(str(config_file)) is True
    assert parser.get_setting("default_records_per_table") == 77
    assert tables["parent"].num_rows == 12
    assert tables["parent"].scd2_enabled is True
    assert tables["child"].delta_eligible is True
    assert len(relationships) == 1
    assert relationships[0].preserve_on_delta is True

    args = parse_arguments(["--config", str(config_file), "--output", str(tmp_path / "out")])
    assert args.command == "generate"

    generator = DataGenerator(str(config_file))
    assert generator.load_configuration() is True
    data = generator.generate_data({"parent": 5, "child": 7})

    assert set(data.keys()) == {"parent", "child"}
    assert len(data["parent"]) == 5
    assert len(data["child"]) == 7
    assert set(data["child"]["parent_id"]).issubset(set(data["parent"]["id"]))


def test_delta_and_scd2_post_processing(tmp_path: Path):
    previous_dir = tmp_path / "previous"
    current_dir = tmp_path / "current"
    delta_dir = tmp_path / "delta"
    scd2_dir = tmp_path / "scd2"
    previous_dir.mkdir()
    current_dir.mkdir()

    parent_prev = pd.DataFrame(
        [
            {"id": 1, "status": "A"},
            {"id": 2, "status": "A"},
        ]
    )
    parent_curr = pd.DataFrame(
        [
            {"id": 1, "status": "B"},
            {"id": 3, "status": "A"},
        ]
    )
    child_prev = pd.DataFrame(
        [
            {"id": 10, "parent_id": 1},
            {"id": 11, "parent_id": 2},
        ]
    )
    child_curr = pd.DataFrame(
        [
            {"id": 10, "parent_id": 1},
            {"id": 12, "parent_id": 3},
        ]
    )

    parent_prev.to_parquet(previous_dir / "parent.parquet", index=False)
    parent_curr.to_parquet(current_dir / "parent.parquet", index=False)
    child_prev.to_parquet(previous_dir / "child.parquet", index=False)
    child_curr.to_parquet(current_dir / "child.parquet", index=False)

    workbook = tmp_path / "enhanced.xlsx"
    _write_workbook(workbook, with_optional_sheets=True)
    parser = ConfigParser(str(workbook))
    assert parser.load_config() is True
    parser.parse_tables()
    parser.run_settings["delta_partition_column"] = "run_partition_date"
    parser.tables["parent"].partition_columns = ["as_of_date"]

    processor = ParquetPostProcessor(parser.tables, parser.run_settings)
    delta_summary = processor.generate_delta(str(previous_dir), str(current_dir), str(delta_dir))
    assert delta_summary["parent"] == {"rows": 3, "inserts": 1, "updates": 1, "deletes": 1}
    assert delta_summary["child"] == {"rows": 2, "inserts": 1, "updates": 0, "deletes": 1}

    parent_delta_root = delta_dir / "parent"
    parent_log_dir = parent_delta_root / "_delta_log"
    assert parent_log_dir.exists()
    assert sorted(path.name for path in parent_log_dir.glob("*.json")) == ["00000000000000000000.json"]

    parent_delta_table = DeltaTable(str(parent_delta_root))
    assert parent_delta_table.version() == 0
    assert parent_delta_table.metadata().partition_columns == ["as_of_date"]
    assert parent_delta_table.to_pyarrow_dataset().count_rows() == 3

    first_parent_log = [json.loads(line) for line in (parent_log_dir / "00000000000000000000.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("protocol" in entry for entry in first_parent_log)
    assert any("metaData" in entry for entry in first_parent_log)
    assert any("add" in entry for entry in first_parent_log)
    assert any("commitInfo" in entry for entry in first_parent_log)
    assert not any("remove" in entry for entry in first_parent_log)

    child_delta_root = delta_dir / "child"
    child_delta_table = DeltaTable(str(child_delta_root))
    assert child_delta_table.metadata().partition_columns == ["run_partition_date"]
    assert child_delta_table.to_pyarrow_dataset().count_rows() == 2

    second_delta_summary = processor.generate_delta(str(previous_dir), str(current_dir), str(delta_dir))
    assert second_delta_summary["parent"] == {"rows": 3, "inserts": 1, "updates": 1, "deletes": 1}
    assert sorted(path.name for path in parent_log_dir.glob("*.json")) == [
        "00000000000000000000.json",
        "00000000000000000001.json",
    ]

    parent_delta_table_after_rerun = DeltaTable(str(parent_delta_root))
    assert parent_delta_table_after_rerun.version() == 1
    assert parent_delta_table_after_rerun.metadata().partition_columns == ["as_of_date"]
    assert parent_delta_table_after_rerun.to_pyarrow_dataset().count_rows() == 3

    second_parent_log = [json.loads(line) for line in (parent_log_dir / "00000000000000000001.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("remove" in entry for entry in second_parent_log)
    assert any("add" in entry for entry in second_parent_log)
    assert any("commitInfo" in entry for entry in second_parent_log)

    scd2_summary = processor.generate_scd2(str(previous_dir), str(current_dir), str(scd2_dir), ["parent"])
    assert scd2_summary["parent"]["new_versions"] == 2
    parent_scd2 = pd.read_parquet(scd2_dir / "parent.parquet")
    assert {"effective_from_ts", "effective_to_ts", "is_current", "version_num"}.issubset(parent_scd2.columns)
    assert (parent_scd2["id"] == 1).sum() == 2


def test_sdv_metadata_skips_unsupported_composite_relationships(tmp_path: Path):
    workbook = tmp_path / "sdv_metadata.xlsx"
    _write_sdv_edge_case_workbook(workbook)

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True

    metadata = generator.create_sdv_metadata()
    metadata_dict = metadata.to_dict()

    parent_columns = metadata_dict["tables"]["parent"]["columns"]
    assert parent_columns["status_code"]["sdtype"] == "categorical"
    assert "order_by" not in parent_columns["status_code"]
    assert metadata_dict.get("relationships", []) == []


def test_synthesizer_fit_handles_invalid_datetime_business_values(tmp_path: Path):
    workbook = tmp_path / "sdv_fit.xlsx"
    _write_sdv_edge_case_workbook(workbook)

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True
    generator.create_sdv_metadata()

    assert generator.train_synthesizer(sample_size=20) is True
    assert generator.is_fitted is True


def test_synthesizer_fit_handles_business_key_primary_key_columns(tmp_path: Path):
    workbook = tmp_path / "business_key_only.xlsx"
    _write_business_key_only_workbook(workbook)

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True
    generator.create_sdv_metadata()

    assert generator.train_synthesizer(sample_size=20) is True
    assert generator.is_fitted is True


def test_helpers_generate_realistic_bounded_values():
    helpers = DataHelpers()

    name_value = helpers.generate_realistic_dutch_data("CTPTY_NM", "VA254", max_length=254)
    address_value = helpers.generate_realistic_dutch_data("INPTY_ADR_LINE1", "VA140", max_length=140)
    city_value = helpers.generate_realistic_dutch_data("CITY_NM", "VA50", max_length=50)
    currency_value = helpers.generate_realistic_dutch_data("ACCT_CCY", "A3", max_length=3)
    reason_value = helpers.generate_realistic_dutch_data("RSN_CD", "A4", max_length=4)

    assert isinstance(name_value, str)
    assert 3 <= len(name_value) <= 254
    assert any(ch.isalpha() for ch in name_value)
    assert len(address_value) <= 140
    assert any(ch.isalpha() for ch in address_value)
    assert city_value in {
        "Amsterdam", "Rotterdam", "Den Haag", "Utrecht", "Eindhoven",
        "Tilburg", "Groningen", "Almere", "Breda", "Nijmegen",
        "Enschede", "Haarlem", "Arnhem", "Zaanstad", "Zwolle",
        "Leeuwarden", "Leiden", "Maastricht", "Dordrecht", "Amersfoort",
    } or isinstance(city_value, str)
    assert currency_value in {"EUR", "GBP", "CHF", "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "HRK", "USD", "CAD", "MXN", "JPY", "CNY", "INR", "AUD", "NZD", "SGD"}
    assert reason_value in {"AC01", "AM04", "FF01", "FR01", "MS02", "RC01"}


def test_generator_preserves_rule_precedence_and_va_length(tmp_path: Path):
    workbook = tmp_path / "quality.xlsx"
    columns_df = pd.DataFrame(
        [
            {
                "table_name": "sample",
                "column_name": "acct_ccy",
                "data_type": "VA10",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": "EUR;USD",
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "sample",
                "column_name": "bank_bic",
                "data_type": "VA11",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": "BIC",
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
            {
                "table_name": "sample",
                "column_name": "customer_name",
                "data_type": "VA254",
                "is_pk": False,
                "is_fk": False,
                "ref_table": None,
                "ref_column": None,
                "business_values": None,
                "special_rules": None,
                "min_value": None,
                "max_value": None,
                "nullable": False,
                "is_business_key_component": False,
                "event_time": False,
                "partition_role": "",
                "scd2_tracked": False,
            },
        ]
    )

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        columns_df.to_excel(writer, sheet_name="Columns", index=False)

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True
    table = generator.tables_config["sample"]
    by_name = {column.column_name: column for column in table.columns}

    currency_values = {generator._generate_enhanced_value(by_name["acct_ccy"], "sample") for _ in range(20)}
    assert currency_values.issubset({"EUR", "USD"})

    bic_value = generator._generate_enhanced_value(by_name["bank_bic"], "sample")
    assert isinstance(bic_value, str)
    assert 8 <= len(bic_value) <= 11

    customer_samples = [generator._generate_enhanced_value(by_name["customer_name"], "sample") for _ in range(10)]
    assert all(isinstance(value, str) for value in customer_samples)
    assert all(1 <= len(value) <= 254 for value in customer_samples)
    assert any(len(value) < 80 for value in customer_samples)
    assert all(any(ch.isalpha() for ch in value) for value in customer_samples)


def test_sdv_generation_reconciles_constrained_columns_and_keeps_relationships(tmp_path: Path):
    workbook = tmp_path / "regex_relationships.xlsx"
    _write_regex_relationship_workbook(workbook)

    generator = DataGenerator(str(workbook))
    assert generator.load_configuration() is True

    class FakeSynthesizer:
        def sample(self, num_rows):
            return {
                "parent": pd.DataFrame(
                    {
                        "parent_code": ["broken"] * num_rows["parent"],
                        "status": ["BAD"] * num_rows["parent"],
                    }
                ),
                "child": pd.DataFrame(
                    {
                        "child_id": list(range(1, num_rows["child"] + 1)),
                        "parent_code": ["wrong"] * num_rows["child"],
                        "doc_code": ["oops"] * num_rows["child"],
                    }
                ),
            }

    generator.synthesizer = cast(Any, FakeSynthesizer())
    generator.is_fitted = True

    data = generator.generate_data({"parent": 8, "child": 12})
    parent_df = data["parent"]
    child_df = data["child"]

    assert parent_df["parent_code"].nunique() == 8
    assert parent_df["parent_code"].str.fullmatch(r"PAR\d{4}").all()
    assert set(parent_df["status"]).issubset({"ACT", "NEW"})
    assert child_df["doc_code"].str.fullmatch(r"CHD-\d{3}").all()
    assert set(child_df["parent_code"]).issubset(set(parent_df["parent_code"]))


