#!/usr/bin/env python3
"""Main SDV-based Data Generator with optional delta and SCD2 parquet flows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from generators.data_generator import DataGenerator
from utils.config_parser import ConfigParser
from utils.data_validator import DataValidator
from utils.parquet_post_processor import ParquetPostProcessor


logger = logging.getLogger(__name__)
KNOWN_COMMANDS = {"generate", "delta", "scd2", "lint", "enrich", "collibra-import",
                  "infer-config", "pii-scan"}


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
    generate_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    generate_parser.add_argument("--infer-relationships", action="store_true", help="Use LLM to infer missing relationships")
    generate_parser.add_argument("--llm-confidence", type=float, default=0.7, help="Minimum LLM confidence threshold (0-1)")
    generate_parser.add_argument("--er-diagram", action="store_true", help="Generate ER diagram after data generation")
    generate_parser.add_argument("--er-format", nargs="+", default=["mermaid"],
                                 choices=["mermaid", "dot", "png"],
                                 help="ER diagram output format(s): mermaid (default), dot, png")
    generate_parser.add_argument("--er-output", default=None,
                                 help="Output directory for ER diagram (default: same as --output)")
    generate_parser.add_argument("--upload-to", default=None,
                                 help="Upload generated files to cloud: azure://<container>[/prefix] or s3://<bucket>[/prefix]")

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

    lint_parser = subparsers.add_parser("lint", help="Validate config and report issues with exact sheet/row/column location")
    lint_parser.add_argument("--config", required=True, help="Path to Excel or YAML configuration file")
    lint_parser.add_argument("--verbose", action="store_true", help="Show all issues including warnings")

    enrich_parser = subparsers.add_parser("enrich", help="Use LLM to enrich schema with semantic suggestions")
    enrich_parser.add_argument("--config", required=True, help="Path to Excel or YAML configuration file")
    enrich_parser.add_argument("--output", required=True, help="Output path for enriched YAML file")
    enrich_parser.add_argument("--confidence", type=float, default=0.7, help="Minimum LLM confidence threshold (0-1)")
    enrich_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    collibra_parser = subparsers.add_parser(
        "collibra-import",
        help="Import dataset definition from Collibra and write a YAML config"
    )
    collibra_parser.add_argument("--dataset", required=True,
                                 help="Collibra dataset name or asset ID to import")
    collibra_parser.add_argument("--output", required=True,
                                 help="Output path for generated YAML config (e.g. config/from_collibra.yaml)")
    collibra_parser.add_argument("--asset-type", default="Data Set",
                                 help="Collibra asset type name (default: 'Data Set')")
    collibra_parser.add_argument("--domain", default=None,
                                 help="Collibra domain name to narrow the search")
    collibra_parser.add_argument("--base-url", default=None,
                                 help="Collibra base URL (overrides COLLIBRA_BASE_URL env var)")
    collibra_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── infer-config ──────────────────────────────────────────────────────────
    infer_parser = subparsers.add_parser(
        "infer-config",
        help="Infer a YAML config from a sample CSV, Parquet, or Excel data file"
    )
    infer_parser.add_argument("--input", required=True,
                              help="Path to sample data file (.csv, .parquet, .xlsx)")
    infer_parser.add_argument("--output", required=True,
                              help="Output path for the generated YAML config")
    infer_parser.add_argument("--table-name", default=None,
                              help="Override the inferred table name (default: filename stem)")
    infer_parser.add_argument("--no-distributions", action="store_true",
                              help="Skip scipy distribution fitting (faster)")
    infer_parser.add_argument("--no-pii-scan", action="store_true",
                              help="Skip PII column detection")
    infer_parser.add_argument("--pii-confidence", type=float, default=0.70,
                              help="Minimum PII detection confidence to emit a warning (default: 0.70)")
    infer_parser.add_argument("--sample-size", type=int, default=5000,
                              help="Max rows to read for inference (default: 5000)")
    infer_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── pii-scan ──────────────────────────────────────────────────────────────
    pii_parser = subparsers.add_parser(
        "pii-scan",
        help="Scan a data file or config for PII / sensitive columns"
    )
    pii_parser.add_argument("--input", required=True,
                            help="Data file (.csv, .parquet, .xlsx) or config (.yaml, .yml, .xlsx)")
    pii_parser.add_argument("--confidence", type=float, default=0.60,
                            help="Minimum confidence score to report (default: 0.60)")
    pii_parser.add_argument("--sample-size", type=int, default=5000,
                            help="Max rows to read when scanning a data file (default: 5000)")
    pii_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

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


def _run_relationship_inference(generator: DataGenerator, confidence: float) -> None:
    try:
        from llm.relationship_inferrer import RelationshipInferrer
        inferrer = RelationshipInferrer()
        new_rels = inferrer.infer(generator.tables_config, generator.relationships, min_confidence=confidence)
        if new_rels:
            generator.relationships.extend(new_rels)
            logger.info(f"LLM inferred {len(new_rels)} additional relationship(s)")
        else:
            logger.info("LLM found no additional relationships to add")
    except Exception as exc:
        logger.warning(f"LLM relationship inference skipped: {exc}")


def run_generate(args) -> int:
    logger.info("SDV Test Data Generator")
    logger.info("=" * 50)

    if not validate_config_file(args.config):
        return 1
    if not create_output_directory(args.output):
        return 1

    seed: Optional[int] = getattr(args, "seed", None)

    logger.info("Initializing SDV data generator...")
    generator = DataGenerator(args.config, seed=seed)
    if not generator.load_configuration():
        logger.error("Failed to load configuration")
        return 1

    if getattr(args, "infer_relationships", False):
        confidence = getattr(args, "llm_confidence", 0.7)
        _run_relationship_inference(generator, confidence)

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
    if seed is not None:
        logger.info(f"  Seed: {seed}")

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
    if report.get("seed") is not None:
        logger.info(f"  Seed: {report['seed']}")
    if report.get("generation_path_summary"):
        logger.info(f"  Generation paths: {report['generation_path_summary']}")

    if args.validate:
        logger.info("\nRunning relationship validation...")
        validator = DataValidator()
        is_valid = validator.validate_relationships(data, generator.relationships)
        val_report = validator.get_validation_report()
        logger.info(f"  Valid relationships: {val_report.get('valid_count', 0)}")
        logger.info(f"  Invalid relationships: {val_report.get('invalid_count', 0)}")
        if not is_valid:
            logger.warning("Some relationship issues were found")

    # --- ER diagram ---
    if getattr(args, "er_diagram", False):
        _run_er_diagram(generator, args)

    # --- Cloud upload ---
    if getattr(args, "upload_to", None):
        _run_upload(args.output, args.upload_to)

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


def run_lint(args) -> int:
    if not validate_config_file(args.config):
        return 1

    parser = ConfigParser(args.config)
    if not parser.load_config():
        logger.error("Failed to load configuration")
        return 1
    parser.parse_tables()
    parser.parse_relationships()

    issues = parser.lint_config()
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    print(parser.format_lint_report(issues))

    if errors:
        logger.error(f"Lint failed: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    if warnings:
        logger.warning(f"Lint passed with {len(warnings)} warning(s)")
    else:
        logger.info("Lint passed: no issues found")
    return 0


def run_enrich(args) -> int:
    if not validate_config_file(args.config):
        return 1

    parser = ConfigParser(args.config)
    if not parser.load_config():
        logger.error("Failed to load configuration")
        return 1
    parser.parse_tables()
    parser.parse_relationships()

    try:
        from llm.schema_enricher import SchemaEnricher
        enricher = SchemaEnricher(min_confidence=args.confidence)
        enricher.enrich(parser.tables, output_yaml_path=args.output)
        logger.info(f"Enriched schema written to {args.output}")
        suggestions = enricher.get_suggestions(parser.tables)
        logger.info(f"Applied {len(suggestions)} LLM suggestion(s)")
        return 0
    except Exception as exc:
        logger.error(f"Schema enrichment failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def _run_er_diagram(generator, args) -> None:
    try:
        from utils.er_diagram import ERDiagramGenerator
        er_output = getattr(args, "er_output", None) or args.output
        er_formats = getattr(args, "er_format", ["mermaid"])
        gen = ERDiagramGenerator(generator.tables_config, generator.relationships)
        written = gen.save(er_output, formats=er_formats)
        if written:
            logger.info(f"\nER diagram(s) saved:")
            for p in written:
                logger.info(f"  {p}")
            if any(str(p).endswith(".mmd") for p in written):
                logger.info("  Tip: open .mmd in VS Code (Mermaid extension) or paste into https://mermaid.live")
    except Exception as exc:
        logger.warning(f"ER diagram generation failed (non-fatal): {exc}")


def _run_upload(output_dir: str, uri: str) -> None:
    try:
        from utils.cloud_uploader import upload_output
        logger.info(f"\nUploading output to {uri} ...")
        paths = upload_output(output_dir, uri)
        logger.info(f"Upload complete: {len(paths)} file(s)")
    except ImportError as exc:
        logger.warning(f"Cloud upload skipped — missing dependency: {exc}")
    except Exception as exc:
        logger.error(f"Cloud upload failed: {exc}")


def run_collibra_import(args) -> int:
    configure_logging(getattr(args, "verbose", False))
    try:
        from utils.collibra_importer import CollibraImporter
        importer = CollibraImporter(base_url=getattr(args, "base_url", None))
        out = importer.import_dataset(
            dataset_name=args.dataset,
            output_path=args.output,
            asset_type=getattr(args, "asset_type", "Data Set"),
            domain=getattr(args, "domain", None),
        )
        logger.info(f"Collibra import complete. Config written to: {out}")
        logger.info(f"Next step: python main.py generate --config {out} --output output/snapshot_v1")
        return 0
    except Exception as exc:
        logger.error(f"Collibra import failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def run_infer_config(args) -> int:
    """Infer a YAML config from a sample data file."""
    configure_logging(getattr(args, "verbose", False))
    try:
        import yaml
        from ml.auto_config import AutoConfigInferrer

        inferrer = AutoConfigInferrer(
            fit_distributions=not getattr(args, "no_distributions", False),
            scan_pii=not getattr(args, "no_pii_scan", False),
            sample_size=getattr(args, "sample_size", 5000),
            pii_confidence=getattr(args, "pii_confidence", 0.70),
        )
        config = inferrer.infer_from_file(
            args.input,
            table_name=getattr(args, "table_name", None),
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

        table_cfg = config["tables"][0]
        n_cols = len(table_cfg.get("columns", []))
        pii_cols = [c for c in table_cfg.get("columns", []) if "_pii_note" in c]
        lines = [
            f"",
            f"Config written to: {out}",
            f"  Table: {table_cfg['name']} | {n_cols} column(s) inferred",
        ]
        if pii_cols:
            lines.append(f"  {len(pii_cols)} column(s) flagged as PII -- review _pii_note fields")
        lines += ["", "Next step:", f"  python main.py generate --config {out} --output output/snapshot_v1"]
        sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", errors="replace"))
        return 0
    except Exception as exc:
        logger.error(f"infer-config failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def run_pii_scan(args) -> int:
    """Scan a data file or config for PII / sensitive columns."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from ml.pii_detector import PIIDetector
        detector = PIIDetector(confidence_threshold=getattr(args, "confidence", 0.60))

        input_path = Path(args.input)
        suffix = input_path.suffix.lower()
        findings = []

        if suffix in (".csv", ".parquet", ".xlsx", ".xls"):
            # Scan actual data values
            sample_size = getattr(args, "sample_size", 5000)
            if suffix == ".csv":
                df = pd.read_csv(input_path, nrows=sample_size)
            elif suffix == ".parquet":
                df = pd.read_parquet(input_path)
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)
            else:
                df = pd.read_excel(input_path, nrows=sample_size)
            findings = detector.scan_dataframe(df, table_name=input_path.stem)

        elif suffix in (".yaml", ".yml"):
            # Scan config column names (no data values)
            cp = ConfigParser(str(input_path))
            tables_cfg, _ = cp.parse_config()
            findings = detector.scan_config(tables_cfg)

        else:
            logger.error(f"Unsupported file type: {suffix}. Use .csv, .parquet, .xlsx, .yaml")
            return 1

        report = detector.format_report(findings)
        sys.stdout.buffer.write((report + "\n").encode("utf-8", errors="replace"))
        return 0
    except Exception as exc:
        logger.error(f"pii-scan failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


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
        if args.command == "lint":
            return run_lint(args)
        if args.command == "enrich":
            return run_enrich(args)
        if args.command == "collibra-import":
            return run_collibra_import(args)
        if args.command == "infer-config":
            return run_infer_config(args)
        if args.command == "pii-scan":
            return run_pii_scan(args)
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
