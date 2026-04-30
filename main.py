#!/usr/bin/env python3
"""Main SDV-based Data Generator with optional delta and SCD2 parquet flows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from generators.data_generator import DataGenerator
from utils.config_parser import ConfigParser
from utils.data_validator import DataValidator
from utils.parquet_post_processor import ParquetPostProcessor


logger = logging.getLogger(__name__)
KNOWN_COMMANDS = {"generate", "delta", "scd2"}


def _normalize_argv(argv: Sequence[str]) -> List[str]:
    if not argv:
        return ["generate"]
    if argv[0] in KNOWN_COMMANDS:
        return list(argv)
    return ["generate", *argv]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SDV-Based Test Data Generator with snapshot, delta, and SCD2 flows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate parquet snapshots from an Excel or YAML config")
    generate_parser.add_argument("--config", required=True, help="Path to Excel or YAML configuration file")
    generate_parser.add_argument("--output", default="output", help="Output directory for parquet files")
    generate_parser.add_argument("--default-records", type=int, default=None, help="Default records per table")
    generate_parser.add_argument("--records", nargs="+", help="Table-specific records: table_name:count")
    generate_parser.add_argument("--validate", action="store_true", help="Validate relationships after generation")
    generate_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    generate_parser.add_argument("--stream", action="store_true", help="Use streaming/chunked generation and direct export")
    generate_parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for streaming generation")

    delta_parser = subparsers.add_parser("delta", help="Generate parquet deltas from two snapshot folders")
    delta_parser.add_argument("--config", required=True, help="Path to Excel or YAML configuration file")
    delta_parser.add_argument("--previous", required=True, help="Previous snapshot parquet directory")
    delta_parser.add_argument("--current", required=True, help="Current snapshot parquet directory")
    delta_parser.add_argument("--output", required=True, help="Output directory for delta parquet files")
    delta_parser.add_argument("--tables", nargs="+", help="Optional list of table names to process")
    delta_parser.add_argument("--partition-column", help="Override the delta partition column for all processed tables")
    delta_parser.add_argument("--partition-columns", nargs="+", help="Override the delta partition columns for all processed tables")
    delta_parser.add_argument("--partition-start-date", help="Override the synthetic delta partition start date (YYYYMMDD or timestamp)")
    delta_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    scd2_parser = subparsers.add_parser("scd2", help="Build SCD2 parquet outputs from snapshot folders")
    scd2_parser.add_argument("--config", required=True, help="Path to Excel or YAML configuration file")
    scd2_parser.add_argument("--previous", required=True, help="Previous snapshot or SCD2 parquet directory")
    scd2_parser.add_argument("--current", required=True, help="Current snapshot parquet directory")
    scd2_parser.add_argument("--output", required=True, help="Output directory for SCD2 parquet files")
    scd2_parser.add_argument("--tables", nargs="+", help="Optional list of table names to process")
    scd2_parser.add_argument("--effective-ts", help="Effective timestamp for the current snapshot rows")
    scd2_parser.add_argument("--previous-effective-ts", help="Bootstrap effective timestamp for previous snapshot rows")
    scd2_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    return parser


def parse_arguments(argv: Sequence[str] | None = None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    return parser.parse_args(_normalize_argv(argv))


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")


def create_output_directory(output_dir: str) -> bool:
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path.absolute()}")
        return True
    except Exception as exc:
        logger.error(f"Error creating output directory: {exc}")
        return False


def validate_config_file(config_path: str) -> bool:
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return False
    if not config_file.is_file():
        logger.error(f"Configuration path is not a file: {config_file}")
        return False
    if config_file.suffix.lower() not in [".xlsx", ".xls", ".yaml", ".yml"]:
        logger.warning(f"Configuration file may not be Excel or YAML format: {config_file}")
    logger.info(f"Configuration file: {config_file.absolute()}")
    return True


def get_record_counts(generator: DataGenerator, args) -> Dict[str, int]:
    workbook_default = generator.config_parser.get_setting("default_records_per_table", 1000)
    default_records = args.default_records if args.default_records is not None else int(workbook_default)
    records_config = {
        table_name: max(
            1,
            int(default_records if args.default_records is not None else (table_config.num_rows or default_records)),
        )
        for table_name, table_config in generator.tables_config.items()
        if table_config.active
    }

    if args.records:
        for record_arg in args.records:
            if ":" not in record_arg:
                logger.warning(f"Invalid record format: {record_arg}")
                continue
            try:
                table_name, count = record_arg.split(":", 1)
                table_name = table_name.strip().lower()
                count = max(1, int(count))
                if table_name in records_config:
                    records_config[table_name] = count
                    logger.info(f"  {table_name}: {count} records (from command line)")
                else:
                    logger.warning(f"Table '{table_name}' not found in configuration")
            except ValueError:
                logger.warning(f"Invalid record format: {record_arg}")

    return records_config


def verify_export(output_dir: str) -> int:
    output_path = Path(output_dir)
    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files were found in {output_path}")

    total_file_records = 0
    logger.info(f"Export successful: {len(parquet_files)} parquet files created")
    for file in parquet_files:
        file_data = pd.read_parquet(file)
        file_records = len(file_data)
        total_file_records += file_records
        logger.info(f"  {file.name}: {file_records} records ({file.stat().st_size} bytes)")
    return total_file_records


def load_config_context(config_path: str) -> ConfigParser:
    parser = ConfigParser(config_path)
    if not parser.load_config():
        raise ValueError("Failed to load configuration")
    parser.parse_tables()
    parser.parse_relationships()
    if not parser.validate_config():
        raise ValueError("Configuration validation failed")
    return parser


def run_generate(args) -> int:
    logger.info("SDV Test Data Generator")
    logger.info("=" * 50)

    if not validate_config_file(args.config):
        return 1
    if not create_output_directory(args.output):
        return 1

    logger.info("Initializing SDV data generator...")
    generator = DataGenerator(args.config)
    if not generator.load_configuration():
        logger.error("Failed to load configuration")
        return 1

    logger.info("Creating SDV metadata...")
    generator.create_sdv_metadata()

    table_names = [table_name for table_name, cfg in generator.tables_config.items() if cfg.active]
    if not table_names:
        logger.error("No active tables found in configuration")
        return 1

    logger.info(f"Tables detected: {len(table_names)}")
    for table_name in table_names:
        logger.info(f"  {table_name}")

    records_config = get_record_counts(generator, args)
    logger.info("\nGeneration settings:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Total tables to generate: {len(records_config)}")

    logger.info("\nTraining SDV synthesizer...")
    if generator.train_synthesizer():
        logger.info("SDV synthesizer trained successfully")
    else:
        logger.warning("SDV synthesizer training failed - using fallback generation")

    logger.info("\nStarting data generation...")
    if args.stream:
        if hasattr(generator, "generate_and_export_stream"):
            logger.info("Using streaming generation + incremental export")
            generator.generate_and_export_stream(records_config, output_dir=args.output, chunk_size=args.chunk_size)
            total_file_records = verify_export(args.output)
            logger.info(f"Streaming export created parquet files with {total_file_records} total records")
            return 0
        logger.warning("Streaming mode requested, but this generator build does not expose a dedicated streaming export method; falling back to standard generation")

    data = generator.generate_data(records_config)
    if not data:
        logger.error("No data generated")
        return 1

    logger.info("\nValidating generated data...")
    empty_tables = 0
    for table_name, table_data in data.items():
        if table_data.empty:
            logger.warning(f"Table {table_name} is empty")
            empty_tables += 1
        else:
            logger.info(f"Table {table_name}: {len(table_data)} records")

    logger.info(f"\nExporting to {args.output}...")
    generator.export_to_parquet(args.output)
    generator.save_model_artifacts(generator.config_parser.get_setting("model_artifact_path", None))
    total_file_records = verify_export(args.output)

    report = generator.get_generation_report()
    logger.info("\nGeneration Report:")
    logger.info(f"  Total records: {report['total_records']:,}")
    logger.info(f"  File records verified: {total_file_records:,}")
    logger.info(f"  Relationships configured: {report['relationships_configured']}")
    logger.info(f"  Synthesizer fitted: {report['synthesizer_fitted']}")
    logger.info(f"  Empty tables: {empty_tables}")

    if args.validate:
        logger.info("\nRunning relationship validation...")
        validator = DataValidator()
        is_valid = validator.validate_relationships(data, generator.relationships)
        val_report = validator.get_validation_report()
        logger.info(f"  Valid relationships: {val_report.get('valid_count', 0)}")
        logger.info(f"  Invalid relationships: {val_report.get('invalid_count', 0)}")
        if not is_valid:
            logger.warning("Some relationship issues were found")

    logger.info(f"\nAll files saved to: {Path(args.output).absolute()}")
    return 0


def run_delta(args) -> int:
    if not validate_config_file(args.config):
        return 1
    if not create_output_directory(args.output):
        return 1

    parser = load_config_context(args.config)
    run_settings = dict(parser.run_settings)
    if args.partition_columns:
        run_settings["delta_partition_columns"] = ";".join(args.partition_columns)
    elif args.partition_column:
        run_settings["delta_partition_column"] = args.partition_column
    if args.partition_start_date:
        run_settings["delta_partition_start_date"] = args.partition_start_date

    processor = ParquetPostProcessor(parser.tables, run_settings)
    summary = processor.generate_delta(args.previous, args.current, args.output, args.tables)
    if not summary:
        logger.warning("No delta output was generated")
        return 0

    logger.info("\nDelta generation summary:")
    for table_name, metrics in summary.items():
        logger.info(f"  {table_name}: {metrics}")
    return 0


def run_scd2(args) -> int:
    if not validate_config_file(args.config):
        return 1
    if not create_output_directory(args.output):
        return 1

    parser = load_config_context(args.config)
    processor = ParquetPostProcessor(parser.tables, parser.run_settings)
    summary = processor.generate_scd2(
        previous_dir=args.previous,
        current_dir=args.current,
        output_dir=args.output,
        selected_tables=args.tables,
        effective_timestamp=args.effective_ts,
        previous_effective_timestamp=args.previous_effective_ts,
    )
    if not summary:
        logger.warning("No SCD2 output was generated")
        return 0

    logger.info("\nSCD2 generation summary:")
    for table_name, metrics in summary.items():
        logger.info(f"  {table_name}: {metrics}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    configure_logging(getattr(args, "verbose", False))

    try:
        if args.command == "generate":
            return run_generate(args)
        if args.command == "delta":
            return run_delta(args)
        if args.command == "scd2":
            return run_scd2(args)
        logger.error(f"Unknown command: {args.command}")
        return 1
    except FileNotFoundError as exc:
        logger.error(f"File error: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
