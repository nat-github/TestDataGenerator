#!/usr/bin/env python3
"""Main SDV-based Data Generator with optional delta and SCD2 parquet flows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from sdp.generators.data_generator import DataGenerator
from sdp.utils.config_parser import ConfigParser
from sdp.utils.data_validator import DataValidator
from sdp.utils.parquet_post_processor import ParquetPostProcessor


logger = logging.getLogger(__name__)
KNOWN_COMMANDS = {"generate", "delta", "scd2", "lint", "enrich", "collibra-import",
                  "infer-config", "pii-scan", "infer-relationships", "record-feedback",
                  "mock-init", "mock-render", "mock-lint", "mock-enrich",
                  "validate-data", "quality-report", "contract-test", "contract-diff"}


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
    # Not `required=True`: --list-engines is a valid invocation on its own.
    # run_generate rejects a missing --config for every other path.
    generate_parser.add_argument("--config", default=None, help="Path to Excel or YAML configuration file")
    generate_parser.add_argument("--output", default="output", help="Output directory for parquet files")
    generate_parser.add_argument("--default-records", type=int, default=None, help="Default records per table")
    generate_parser.add_argument("--records", nargs="+", help="Table-specific records: table_name:count")
    generate_parser.add_argument("--validate", action="store_true", help="Validate relationships after generation")
    generate_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    generate_parser.add_argument("--stream", action="store_true", help="Use streaming/chunked generation and direct export")
    generate_parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for streaming generation")
    generate_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    generate_parser.add_argument("--engine", default=None,
                                 help="Generation engine (default: sdv). See --list-engines")
    generate_parser.add_argument("--engine-option", action="append", default=None, metavar="KEY=VALUE",
                                 help="Engine-specific option, repeatable (e.g. epochs=300)")
    generate_parser.add_argument("--list-engines", action="store_true",
                                 help="List available generation engines and exit")
    generate_parser.add_argument("--infer-relationships", action="store_true",
                                 help="Infer missing FK relationships before generation")
    generate_parser.add_argument("--method", choices=["ml", "llm", "both"], default="ml",
                                 help="Inference engine: ml (heuristic, free), llm (Claude, costs), or both (ML first, LLM only on low-confidence edges)")
    generate_parser.add_argument("--llm-confidence", type=float, default=0.7,
                                 help="Minimum LLM confidence threshold (0-1)")
    generate_parser.add_argument("--ml-confidence", type=float, default=0.55,
                                 help="Minimum ML confidence threshold (0-1)")
    generate_parser.add_argument("--feedback-store", default=None,
                                 help="Path to ML feedback JSONL (default: ml_feedback/relationship_feedback.jsonl)")
    generate_parser.add_argument("--er-diagram", action="store_true", help="Generate ER diagram after data generation")
    generate_parser.add_argument("--er-format", nargs="+", default=["mermaid"],
                                 choices=["mermaid", "dot", "png"],
                                 help="ER diagram output format(s): mermaid (default), dot, png")
    generate_parser.add_argument("--er-output", default=None,
                                 help="Output directory for ER diagram (default: same as --output)")
    generate_parser.add_argument("--upload-to", default=None,
                                 help="Upload generated files to cloud: azure://<container>[/prefix] or s3://<bucket>[/prefix]")
    generate_parser.add_argument("--validate-with-gx", action="store_true",
                                 help="Run Great Expectations validation after generation (requires --extras gx)")
    generate_parser.add_argument("--gx-tolerance", type=float, default=0.5,
                                 help="Row-count tolerance for GX validation (0.5 = ±50%%; default 0.5)")
    generate_parser.add_argument("--gx-fail-on-error", action="store_true",
                                 help="Exit non-zero when GX validation fails (default: report and continue)")
    # --- Delta Lake direct write (one command, no separate `delta` step) ---
    generate_parser.add_argument("--write-delta", action="store_true",
                                 help="Write each table as a Delta Lake table at <output>/<table>/ "
                                      "instead of (or alongside) a flat parquet snapshot. Each run "
                                      "appends a new partition under the same _delta_log.")
    generate_parser.add_argument("--delta-partition-col", default="BOOKING_TM",
                                 help="Default partition column for tables converted to Delta (default: BOOKING_TM). "
                                      "Per-table overrides via YAML `delta_partition_col: <COLNAME>` — useful when "
                                      "different sources partition by different columns in the same run.")
    generate_parser.add_argument("--delta-partition-value", default=None,
                                 help="Partition value for THIS run (e.g. 20260531). Default: today's date as YYYYMMDD.")
    generate_parser.add_argument("--delta-tables", nargs="+", default=None,
                                 help="Override which tables go to Delta. If omitted, tables with "
                                      "`write_delta: true` in the config are used. If neither is set, all tables.")

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

    scd2_parser = subparsers.add_parser("scd2", help="Build SCD2 parquet outputs from snapshot folders (or generate them with --simulate)")
    scd2_parser.add_argument("--config", required=True, help="Path to Excel or YAML configuration file")
    scd2_parser.add_argument("--previous", help="Previous snapshot or SCD2 parquet directory (omit when using --simulate)")
    scd2_parser.add_argument("--current", help="Current snapshot parquet directory (omit when using --simulate)")
    scd2_parser.add_argument("--output", required=True, help="Output directory for SCD2 parquet files")
    scd2_parser.add_argument("--tables", nargs="+", help="Optional list of table names to process")
    scd2_parser.add_argument("--effective-ts", help="Effective timestamp for the current snapshot rows")
    scd2_parser.add_argument("--previous-effective-ts", help="Bootstrap effective timestamp for previous snapshot rows")
    # --- self-contained mode: generate v1 + v2 internally, then diff ---
    scd2_parser.add_argument("--simulate", action="store_true",
                             help="Generate the previous+current snapshots from --config internally and diff them "
                                  "(no --previous/--current needed). Produces real version history in one command.")
    scd2_parser.add_argument("--change-fraction", type=float, default=0.3,
                             help="With --simulate: fraction of rows whose tracked column changes between v1 and v2 (default 0.3)")
    scd2_parser.add_argument("--change-columns", nargs="+", default=None,
                             help="With --simulate: specific tracked column(s) to change (default: per-table scd2_tracked_columns from config)")
    scd2_parser.add_argument("--default-records", type=int, default=None,
                             help="With --simulate: rows per table for the generated baseline snapshot")
    scd2_parser.add_argument("--seed", type=int, default=None,
                             help="With --simulate: random seed for the generated baseline + which rows change")
    scd2_parser.add_argument("--keep-snapshots", action="store_true",
                             help="With --simulate: keep the intermediate v1/v2 snapshot folders instead of deleting them")
    scd2_parser.add_argument("--no-effective-dates", action="store_true",
                             help="Drop the effective_from_ts/effective_to_ts columns from the output. The differing "
                                  "version dates stay in the data's own *_crt_dts column(s).")
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

    # ── infer-relationships ────────────────────────────────────────────────
    rel_parser = subparsers.add_parser(
        "infer-relationships",
        help="Deduce FK relationships from a config and emit ER diagram + YAML for SME review",
    )
    rel_parser.add_argument("--config", required=True,
                            help="Path to Excel / YAML / JSON configuration file")
    rel_parser.add_argument("--method", choices=["ml", "llm", "both"], default="ml",
                            help="Inference engine (default: ml)")
    rel_parser.add_argument("--ml-confidence", type=float, default=0.55,
                            help="Minimum ML confidence threshold (0-1)")
    rel_parser.add_argument("--ml-mode", choices=["standard", "knowledge-graph"], default="standard",
                            help="ML inference mode: standard (existing heuristic path) or knowledge-graph (opt-in semantic disambiguation)")
    rel_parser.add_argument("--llm-confidence", type=float, default=0.7,
                            help="Minimum LLM confidence threshold (0-1)")
    rel_parser.add_argument("--config-output", required=True,
                            help="Output YAML path with inferred relationships annotated for SME review")
    rel_parser.add_argument("--simple-yaml", dest="simple_yaml", action="store_true", default=None,
                            help="Write a minimal YAML with only tables and relationships; omit inference metadata, confidence, and recommendations")
    rel_parser.add_argument("--review-yaml", dest="simple_yaml", action="store_false",
                            help="Force the richer review YAML with inference metadata, confidence, and recommendations")
    rel_parser.add_argument("--er-output", default=None,
                            help="Output path for the ER diagram. Extension determines format (.mmd|.dot|.png)")
    rel_parser.add_argument("--feedback-store", default=None,
                            help="Path to ML feedback JSONL (default: ml_feedback/relationship_feedback.jsonl)")
    rel_parser.add_argument("--sample-data", default=None,
                            help="Optional directory of sample parquet/CSV files for value-subset signal")
    rel_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── record-feedback ────────────────────────────────────────────────────
    fb_parser = subparsers.add_parser(
        "record-feedback",
        help="Compare an inferred config against an SME-reviewed config and persist accept/reject deltas for adaptive learning",
    )
    fb_parser.add_argument("--inferred", required=True,
                           help="Path to the YAML emitted by infer-relationships")
    fb_parser.add_argument("--reviewed", required=True,
                           help="Path to the YAML after the SME has accepted / removed / added relationships")
    fb_parser.add_argument("--feedback-store", default=None,
                           help="Path to ML feedback JSONL (default: ml_feedback/relationship_feedback.jsonl)")
    fb_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── mock-init ──────────────────────────────────────────────────────────
    mi_parser = subparsers.add_parser(
        "mock-init",
        help="Convert an OpenAPI spec / Postman collection / HAR capture into a sdp-mock-v1 YAML",
    )
    mi_parser.add_argument("--from", dest="source", required=True,
                           help="Path to source artefact (OpenAPI YAML/JSON, Postman collection JSON, or HAR file)")
    mi_parser.add_argument("--output", required=True,
                           help="Destination path for the generated sdp-mock-v1 YAML")
    mi_parser.add_argument(
        "--source-type", choices=["auto", "openapi", "postman", "har"], default="auto",
        help="Override the auto-detection of the source format (default: auto)",
    )
    mi_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── mock-render ────────────────────────────────────────────────────────
    mr_parser = subparsers.add_parser(
        "mock-render",
        help="Render mocks (WireMock stubs, JSON fixtures, …) from a sdp-mock-v1 config",
    )
    mr_parser.add_argument("--config", required=True,
                           help="Path to sdp-mock-v1 YAML/JSON config")
    mr_parser.add_argument("--output", required=True,
                           help="Output directory for generated artefacts")
    mr_parser.add_argument("--format", default="wiremock",
                           help="Comma-separated list of formats: wiremock,json,pact,postman,openapi-examples (default: wiremock)")
    mr_parser.add_argument("--pact-consumer", default="consumer",
                           help="Consumer name for Pact contracts (default: consumer)")
    mr_parser.add_argument("--pact-provider", default="provider",
                           help="Provider name for Pact contracts (default: provider)")
    mr_parser.add_argument("--openapi-source",
                           help="Source OpenAPI spec to enrich with examples (required when --format includes openapi-examples)")
    mr_parser.add_argument("--openapi-overwrite", action="store_true",
                           help="When enriching OpenAPI examples, overwrite hand-authored ones (default: preserve)")
    mr_parser.add_argument("--examples", type=int, default=None,
                           help="How many concrete example stubs per endpoint (overrides MockConfig)")
    mr_parser.add_argument("--match-mode", choices=["concrete", "any"], default="concrete",
                           help="WireMock path matching: 'concrete' (literal per example) or 'any' (regex)")
    mr_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output")
    mr_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── mock-lint ──────────────────────────────────────────────────────────
    ml_parser = subparsers.add_parser(
        "mock-lint",
        help="Validate a sdp-mock-v1 config and surface issues without raising",
    )
    ml_parser.add_argument("--config", required=True, help="Path to sdp-mock-v1 YAML/JSON")
    ml_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── mock-enrich ────────────────────────────────────────────────────────
    me_parser = subparsers.add_parser(
        "mock-enrich",
        help="Use an LLM to fill missing schema examples and draft 4xx/5xx responses",
    )
    me_parser.add_argument("--config", required=True, help="Path to sdp-mock-v1 YAML/JSON")
    me_parser.add_argument("--output", required=True, help="Destination path for the enriched config")
    me_parser.add_argument("--no-fill-examples", action="store_true",
                           help="Skip the fill-missing-examples pass")
    me_parser.add_argument("--no-draft-errors", action="store_true",
                           help="Skip the draft-error-responses pass")
    me_parser.add_argument("--llm-provider", default=None,
                           help="LLM provider override (anthropic, openai, lm-studio, ollama, ...)")
    me_parser.add_argument("--llm-model", default=None,
                           help="LLM model override (provider-specific)")
    me_parser.add_argument("--llm-base-url", default=None,
                           help="LLM base URL override (for LM Studio / Ollama / custom)")
    me_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # ── validate-data (Great Expectations) ─────────────────────────────────
    vd_parser = subparsers.add_parser(
        "validate-data",
        help="Run Great Expectations validation on generated Parquet output (requires --extras gx)",
    )
    vd_parser.add_argument("--config", required=True,
                           help="Path to the config used to generate the data")
    vd_parser.add_argument("--input", required=True,
                           help="Directory containing <table>.parquet files to validate")
    vd_parser.add_argument("--tolerance", type=float, default=0.5,
                           help="Row-count tolerance (0.5 = ±50%%; default 0.5)")
    vd_parser.add_argument("--report-json", default=None,
                           help="Optional path to write the full validation report as JSON")
    vd_parser.add_argument("--verbose", action="store_true",
                           help="Print every expectation, including passed ones")
    vd_parser.add_argument("--fail-on-error", action="store_true",
                           help="Exit non-zero when any expectation fails")

    # ── quality-report ─────────────────────────────────────────────────────
    qr_parser = subparsers.add_parser(
        "quality-report",
        help="Statistical quality / fidelity / privacy report on generated Parquet output",
    )
    qr_parser.add_argument("--generated", required=True,
                           help="Directory containing the generated <table>.parquet files")
    qr_parser.add_argument("--source", default=None,
                           help="Optional directory of source Parquet files for fidelity comparison")
    qr_parser.add_argument("--output-html", default=None,
                           help="Optional path to write a self-contained HTML report")
    qr_parser.add_argument("--output-json", default=None,
                           help="Optional path to write the structured JSON report")
    qr_parser.add_argument("--privacy-threshold", type=float, default=0.0,
                           help="Distance below which a synthetic row is flagged as too close to source (0.0 = exact duplicates)")
    qr_parser.add_argument("--no-utility", action="store_true",
                           help="Skip the TSTR utility check (the slowest metric — it fits two models per table)")
    qr_parser.add_argument("--no-bias", action="store_true",
                           help="Skip representation / outcome-disparity checks")
    qr_parser.add_argument("--target", action="append", default=None, metavar="TABLE=COLUMN",
                           help="Column to predict and measure outcomes against, per table "
                                "(repeatable; auto-selected when omitted)")
    qr_parser.add_argument("--verbose", action="store_true", help="Print full markdown report")

    # --- contract-test ---
    ct_parser = subparsers.add_parser(
        "contract-test",
        help="Verify a dataset against a data contract (the config) using Great Expectations",
    )
    ct_parser.add_argument("--contract", required=True,
                           help="Path to the contract config (.xlsx | .yaml | .json)")
    ct_parser.add_argument("--data", required=True,
                           help="Directory of <table>.parquet files to verify")
    ct_parser.add_argument("--tolerance", type=float, default=0.5,
                           help="Permitted row-count deviation from the contract (default 0.5 = ±50%%)")
    ct_parser.add_argument("--report-json", default=None,
                           help="Optional path to write the contract report as JSON")
    ct_parser.add_argument("--fail-on", choices=["error", "warning", "none"], default="error",
                           help="Exit non-zero when failures reach this severity (default: error)")
    ct_parser.add_argument("--verbose", action="store_true", help="Show passing checks too")

    # --- contract-diff ---
    cd_parser = subparsers.add_parser(
        "contract-diff",
        help="Detect breaking changes between two contract versions",
    )
    cd_parser.add_argument("--old", required=True, help="Path to the previous contract config")
    cd_parser.add_argument("--new", required=True, help="Path to the new contract config")
    cd_parser.add_argument("--report-json", default=None,
                           help="Optional path to write the diff report as JSON")
    cd_parser.add_argument("--fail-on-breaking", action="store_true",
                           help="Exit non-zero when any breaking change is detected")
    cd_parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

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
        from sdp.llm.relationship_inferrer import RelationshipInferrer
        inferrer = RelationshipInferrer()
        new_rels = inferrer.infer(generator.tables_config, generator.relationships, min_confidence=confidence)
        if new_rels:
            generator.relationships.extend(new_rels)
            logger.info(f"LLM inferred {len(new_rels)} additional relationship(s)")
        else:
            logger.info("LLM found no additional relationships to add")
    except Exception as exc:
        logger.warning(f"LLM relationship inference skipped: {exc}")


# ---------------------------------------------------------------------------
# SCD2 / Delta / versions_per_key — ported from the patched old generator.
# All of these are opt-in and have no effect on existing configs/commands.
# ---------------------------------------------------------------------------
def _looks_like_date_column(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith("_dts") or "dts" in n or "date" in n


def _apply_change(df: pd.DataFrame, col: str, idx) -> None:
    """Write a clearly-different value into df.loc[idx, col], matching dtype."""
    n = len(idx)
    if pd.api.types.is_datetime64_any_dtype(df[col]) or _looks_like_date_column(col):
        s = pd.to_datetime(df[col], errors="coerce")
        tz = getattr(getattr(s, "dt", None), "tz", None)
        fill = pd.Timestamp("2026-01-01", tz=tz) if tz is not None else pd.Timestamp("2026-01-01")
        base = s.loc[idx].fillna(fill)
        s.loc[idx] = base + pd.to_timedelta(range(1, n + 1), unit="D")
        df[col] = s
    elif pd.api.types.is_numeric_dtype(df[col]):
        df.loc[idx, col] = list(range(1, n + 1))
    else:
        df.loc[idx, col] = [f"SCD2_CHANGED_{i}" for i in range(n)]


def _read_versions_per_key(config_path: str) -> Dict[str, int]:
    """Read per-table `versions_per_key` from YAML/JSON. config_parser ignores it."""
    p = Path(config_path)
    suffix = p.suffix.lower()
    specs: Dict[str, int] = {}
    try:
        if suffix in (".yaml", ".yml"):
            import yaml
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            return {}
        for t in raw.get("tables", []) or []:
            name = t.get("name") or t.get("table_name")
            vpk = t.get("versions_per_key")
            if name and vpk:
                try:
                    specs[name] = int(vpk)
                except (TypeError, ValueError):
                    pass
    except Exception as exc:
        logger.warning(f"Could not read versions_per_key from {config_path}: {exc}")
    return specs


def _expand_versions_from_config(config_path: str, output_dir: str, generator, seed) -> None:
    """Config-driven, schema-preserving versioning: repeat each business key
    1..N times (N = versions_per_key) and vary every column listed in
    scd2_tracked_columns. No columns are added or removed; FK integrity holds."""
    specs = _read_versions_per_key(config_path)
    if not specs:
        return
    import numpy as np
    rng = np.random.default_rng(seed if seed is not None else 7)
    out = Path(output_dir)
    logger.info("\nApplying config-driven versions_per_key (schema unchanged)...")
    for table, vmax in specs.items():
        if not vmax or vmax < 2:
            continue
        pq = out / f"{table}.parquet"
        if not pq.exists():
            continue
        tc = generator.tables_config.get(table)
        if tc is None:
            continue
        tracked = list(getattr(tc, "scd2_tracked_columns", []) or [])
        df = pd.read_parquet(pq)
        if df.empty or not tracked:
            logger.info(f"  versions: {table} skipped (no scd2_tracked_columns)")
            continue
        date_cols = [c for c in tracked
                     if c in df.columns and (pd.api.types.is_datetime64_any_dtype(df[c])
                                             or _looks_like_date_column(c))]
        attr_cols = [c for c in tracked if c in df.columns and c not in date_cols]
        if not date_cols and not attr_cols:
            logger.info(f"  versions: {table} skipped (tracked columns not in data)")
            continue
        counts = rng.integers(1, vmax + 1, size=len(df))
        rep_index = np.repeat(np.arange(len(df)), counts)
        expanded = df.iloc[rep_index].reset_index(drop=True)
        version_no = np.concatenate([np.arange(c) for c in counts])
        # Date columns: vectorised shift.
        if date_cols:
            jitter = rng.integers(0, 30, size=len(expanded))
            offset_days = np.where(version_no == 0, 0, version_no * 90 + jitter)
            offset = pd.to_timedelta(offset_days, unit="D")
            for dc in date_cols:
                base = pd.to_datetime(expanded[dc], errors="coerce")
                tz = getattr(getattr(base, "dt", None), "tz", None)
                fill = pd.Timestamp("2026-01-01", tz=tz) if tz is not None else pd.Timestamp("2026-01-01")
                expanded[dc] = base.fillna(fill) + offset
        # Attribute columns: per-key cycling.
        if attr_cols:
            col_cfg = {c.column_name: c for c in tc.columns}
            for ac in attr_cols:
                cc = col_cfg.get(ac)
                if cc is None:
                    continue
                try:
                    bv_list = generator.helpers.parse_business_values(getattr(cc, "business_values", None)) or []
                except Exception:
                    bv_list = []
                special = getattr(cc, "special_rules", None)
                data_type = getattr(cc, "data_type", None)
                col_pos = expanded.columns.get_loc(ac)
                seen_for_key: set = set()
                shortfalls = 0
                for i in range(len(expanded)):
                    if version_no[i] == 0:
                        seen_for_key = {expanded.iat[i, col_pos]}
                        continue
                    new_val = None
                    if bv_list:
                        new_val = next((v for v in bv_list if v not in seen_for_key), None)
                        if new_val is None:
                            new_val = bv_list[(int(version_no[i]) - 1) % len(bv_list)]
                            shortfalls += 1
                    elif special:
                        for _ in range(10):
                            try:
                                cand = generator.helpers.generate_special_value(special, data_type, column_name=ac)
                            except Exception:
                                cand = None
                                break
                            if cand is not None and cand not in seen_for_key:
                                new_val = cand
                                break
                        if new_val is None:
                            new_val = cand
                    if new_val is None:
                        new_val = int(version_no[i]) if pd.api.types.is_numeric_dtype(expanded[ac]) \
                                  else f"V{int(version_no[i])}"
                    seen_for_key.add(new_val)
                    expanded.iat[i, col_pos] = new_val
                if shortfalls:
                    logger.warning(f"  versions: {table}.{ac} has fewer business_values "
                                   f"({len(bv_list)}) than versions_per_key={vmax}; "
                                   f"{shortfalls} version(s) had to repeat a value")
        expanded.to_parquet(pq, index=False)
        repeats = int((counts > 1).sum())
        parts = []
        if date_cols:
            parts.append(f"dates vary in {date_cols}")
        if attr_cols:
            parts.append(f"attrs vary in {attr_cols}")
        logger.info(f"  versions: {table} {len(df)} -> {len(expanded)} rows "
                    f"({repeats} keys repeated, up to {vmax} each; " + "; ".join(parts) + ")")


def _read_delta_table_selection(config_path: str) -> Optional[List[str]]:
    """Read per-table `write_delta: true` flags from YAML/JSON. Returns list or
    None (None => caller falls back to converting all tables)."""
    p = Path(config_path)
    suffix = p.suffix.lower()
    selected: List[str] = []
    try:
        if suffix in (".yaml", ".yml"):
            import yaml
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            return None
        for t in raw.get("tables", []) or []:
            name = t.get("name") or t.get("table_name")
            if name and t.get("write_delta"):
                selected.append(name)
    except Exception as exc:
        logger.warning(f"Could not read write_delta flags from {config_path}: {exc}")
        return None
    return selected if selected else None


def _read_delta_partition_overrides(config_path: str) -> Dict[str, str]:
    """Read per-table `delta_partition_col` overrides from YAML/JSON.

    Bypasses config_parser / config_models (same pattern as `write_delta` /
    `versions_per_key`). Returns `{table_name: partition_col}` for tables that
    specify it; tables without it fall back to the CLI default
    `--delta-partition-col`. Used when different source tables need different
    partition columns in the same run (e.g. BOOKING_TM for one, LOAD_DT for
    another).
    """
    p = Path(config_path)
    suffix = p.suffix.lower()
    overrides: Dict[str, str] = {}
    try:
        if suffix in (".yaml", ".yml"):
            import yaml
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            return overrides
        for t in raw.get("tables", []) or []:
            name = t.get("name") or t.get("table_name")
            col = t.get("delta_partition_col")
            if name and col:
                overrides[name] = str(col)
    except Exception as exc:
        logger.warning(f"Could not read delta_partition_col overrides from {config_path}: {exc}")
        return {}
    return overrides


def _write_delta_outputs(output_dir: str, partition_col: str, partition_value,
                         selected_tables: Optional[List[str]] = None,
                         partition_overrides: Optional[Dict[str, str]] = None) -> None:
    """Convert selected <output>/<table>.parquet files into Delta tables at
    <output>/<table>/ partitioned by partition_col=partition_value (append mode).

    `partition_overrides` (optional): per-table `{name: col}` mapping for tables
    that need a different partition column than the global default. Tables
    absent from the mapping use `partition_col`.
    """
    try:
        from deltalake import write_deltalake
    except ImportError:
        raise ImportError("deltalake is required for --write-delta. Install: pip install deltalake")
    from datetime import date

    out = Path(output_dir)
    if not partition_value:
        partition_value = date.today().strftime("%Y%m%d")
    flat_parquets = sorted(p for p in out.glob("*.parquet") if p.is_file())
    if not flat_parquets:
        logger.warning("No parquet files to convert to Delta")
        return
    overrides = partition_overrides or {}
    selection_msg = f" for {len(selected_tables)} selected table(s)" if selected_tables else " (all tables)"
    logger.info(f"\nWriting Delta Lake tables (default partition {partition_col}={partition_value}"
                f"{', overrides for ' + str(len(overrides)) + ' table(s)' if overrides else ''})"
                f"{selection_msg} -> {out}/")
    converted = 0
    for pq in flat_parquets:
        table = pq.stem
        if selected_tables is not None and table not in selected_tables:
            logger.info(f"  Skip:  {table:30s} (no write_delta flag -> kept as flat parquet)")
            continue
        df = pd.read_parquet(pq)
        pq.unlink()
        col = overrides.get(table, partition_col)
        if col in df.columns:
            logger.warning(f"  {table}: existing column '{col}' will be overwritten with the run's partition value")
        df[col] = str(partition_value)
        delta_path = out / table
        write_deltalake(str(delta_path), df, mode="append", partition_by=[col])
        converted += 1
        logger.info(f"  Delta: {table:30s} {len(df):6d} rows -> "
                    f"{delta_path}/{col}={partition_value}/  (+commit in _delta_log/)")
    if selected_tables and converted == 0:
        logger.warning(f"  --write-delta requested but none of {selected_tables} matched any output parquet")


def _generate_snapshot(config_path: str, output_dir: str, default_records, seed) -> None:
    """Generate one full snapshot (all active tables) into output_dir. Used by scd2 --simulate."""
    generator = DataGenerator(config_path, seed=seed)
    if not generator.load_configuration():
        raise ValueError("Failed to load configuration for snapshot generation")
    generator.create_sdv_metadata()
    workbook_default = generator.config_parser.get_setting("default_records_per_table", 1000)
    base = default_records if default_records is not None else int(workbook_default)
    records_config = {
        name: max(1, int(base if default_records is not None else (cfg.num_rows or base)))
        for name, cfg in generator.tables_config.items() if cfg.active
    }
    if not generator.train_synthesizer():
        logger.warning("SDV training failed for snapshot - using fallback generation")
    data = generator.generate_data(records_config)
    if not data:
        raise ValueError("Snapshot generation produced no data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.export_to_parquet(output_dir)


def _derive_changed_snapshot(processor: ParquetPostProcessor, previous_dir: str, current_dir: str,
                             fraction: float, seed, selected_tables, change_columns) -> None:
    """Copy previous_dir -> current_dir, changing tracked columns on a fraction of rows."""
    import random as _random
    rng = _random.Random(seed if seed is not None else 7)
    prev = Path(previous_dir)
    cur = Path(current_dir)
    cur.mkdir(parents=True, exist_ok=True)
    for pq in sorted(prev.glob("*.parquet")):
        table = pq.stem
        df = pd.read_parquet(pq)
        tc = processor.tables_config.get(table)
        do_table = (not selected_tables) or (table in selected_tables)
        changed_cols = None
        if tc is not None and do_table and not df.empty:
            keys = set(processor._resolve_business_keys(tc))
            if change_columns:
                wanted = list(change_columns)
            elif getattr(tc, "scd2_tracked_columns", None):
                wanted = list(tc.scd2_tracked_columns)
            else:
                wanted = processor._resolve_scd2_tracked_columns(tc)[:1]
            targets = [c for c in wanted if c in df.columns and c not in keys]
            if targets:
                n = max(1, int(len(df) * fraction))
                idx = rng.sample(list(df.index), min(n, len(df)))
                for col in targets:
                    _apply_change(df, col, idx)
                changed_cols = ", ".join(targets)
                logger.info(f"  simulate: changed [{changed_cols}] on {len(idx)}/{len(df)} rows of {table}")
        if changed_cols is None:
            logger.info(f"  simulate: {table} copied unchanged (no change column resolved)")
        df.to_parquet(cur / pq.name, index=False)


def _strip_effective_date_columns(output_dir: str, selected_tables) -> None:
    """Remove effective_from_ts / effective_to_ts from the SCD2 output parquets."""
    drop = ["effective_from_ts", "effective_to_ts"]
    out = Path(output_dir)
    for pq in sorted(out.glob("*.parquet")):
        if selected_tables and pq.stem not in selected_tables:
            continue
        df = pd.read_parquet(pq)
        present = [c for c in drop if c in df.columns]
        if present:
            df.drop(columns=present).to_parquet(pq, index=False)
            logger.info(f"  dropped {present} from {pq.name}")


def run_generate(args) -> int:
    logger.info("SDV Test Data Generator")
    logger.info("=" * 50)

    if getattr(args, "list_engines", False):
        _print_engines()
        return 0

    if not args.config:
        logger.error("--config is required (omit it only with --list-engines)")
        return 1
    if not validate_config_file(args.config):
        return 1
    if not create_output_directory(args.output):
        return 1

    seed: Optional[int] = getattr(args, "seed", None)

    logger.info("Initializing SDV data generator...")
    generator = DataGenerator(
        args.config, seed=seed,
        engine=getattr(args, "engine", None),
        # CLI options win over the config's `engine_options` setting.
        engine_options=_parse_engine_options(getattr(args, "engine_option", None)),
    )
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

    anchored = sorted(getattr(generator, "anchor_data", {}).keys())
    if anchored:
        logger.info(f"Anchored tables (loaded from real source data): {', '.join(anchored)}")

    records_config = get_record_counts(generator, args)
    logger.info("\nGeneration settings:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Total tables to generate: {len(records_config)}")
    if seed is not None:
        logger.info(f"  Seed: {seed}")

    engine_name = generator.resolve_engine_name()
    logger.info(f"\nTraining synthesizer (engine: {engine_name})...")
    if generator.train_synthesizer():
        logger.info(f"Synthesizer trained successfully (engine: {engine_name})")
    elif engine_name == "rule-based":
        # Not a failure — this engine has no model by design.
        logger.info("Generating directly from config rules (no model fitted)")
    else:
        logger.warning(f"Synthesizer training failed (engine: {engine_name}) - using fallback generation")

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

    stats = generator.engine_stats
    if stats:
        logger.info(
            f"\nEngine cost: {stats['engine']} — fit {stats['fit_seconds']}s, "
            f"sample {stats['sample_seconds']}s "
            f"({stats['fit_rows']} training rows, {stats['sampled_rows']} sampled)"
        )

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
    _expand_versions_from_config(args.config, args.output, generator, seed)
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

    # --- Great Expectations validation ---
    if getattr(args, "validate_with_gx", False):
        gx_rc = _run_gx_validation(
            generator,
            output_dir=args.output,
            tolerance=getattr(args, "gx_tolerance", 0.5),
            verbose=getattr(args, "verbose", False),
        )
        if gx_rc != 0 and getattr(args, "gx_fail_on_error", False):
            return gx_rc

    # --- Delta Lake direct write (kept LAST so GX/upload see flat parquet) ---
    if getattr(args, "write_delta", False):
        selected = args.delta_tables or _read_delta_table_selection(args.config)
        partition_overrides = _read_delta_partition_overrides(args.config)
        _write_delta_outputs(
            args.output,
            partition_col=args.delta_partition_col,
            partition_value=args.delta_partition_value,
            selected_tables=selected,
            partition_overrides=partition_overrides,
        )

    logger.info(f"\nAll files saved to: {Path(args.output).absolute()}")
    return 0


def _run_gx_validation(generator, *, output_dir: str, tolerance: float, verbose: bool) -> int:
    """Run GX validation against the generated Parquet output and print a summary."""
    try:
        from sdp.validators.gx_validator import (
            HAS_GX, validate_tables, format_report,
        )
    except Exception as exc:
        logger.error(f"Could not import sdp.validators.gx_validator: {exc}")
        return 1
    if not HAS_GX:
        logger.error(
            "Great Expectations is not installed. Run: poetry install --extras gx"
        )
        return 1
    logger.info("\nRunning Great Expectations validation...")
    report = validate_tables(
        generator.tables_config,
        output_dir=output_dir,
        row_count_tolerance=tolerance,
    )
    print(format_report(report, verbose=verbose))
    return 0 if report.success else 2


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

    previous_dir = args.previous
    current_dir = args.current
    temp_dirs: List[Path] = []

    if getattr(args, "simulate", False):
        out = Path(args.output)
        previous_dir = str(out.parent / f"{out.name}_sim_v1")
        current_dir = str(out.parent / f"{out.name}_sim_v2")
        temp_dirs = [Path(previous_dir), Path(current_dir)]
        logger.info("Simulate mode: generating baseline snapshot (v1)...")
        _generate_snapshot(args.config, previous_dir, args.default_records, args.seed)
        logger.info("Simulate mode: deriving changed snapshot (v2)...")
        _derive_changed_snapshot(processor, previous_dir, current_dir,
                                 args.change_fraction, args.seed, args.tables, args.change_columns)
    else:
        if not previous_dir or not current_dir:
            logger.error("scd2 needs --previous and --current snapshot directories "
                         "(or use --simulate to generate them from --config).")
            return 1

    try:
        summary = processor.generate_scd2(
            previous_dir=previous_dir,
            current_dir=current_dir,
            output_dir=args.output,
            selected_tables=args.tables,
            effective_timestamp=args.effective_ts,
            previous_effective_timestamp=args.previous_effective_ts,
        )
    finally:
        if getattr(args, "simulate", False) and not getattr(args, "keep_snapshots", False):
            import shutil
            for d in temp_dirs:
                shutil.rmtree(d, ignore_errors=True)

    if not summary:
        logger.warning("No SCD2 output was generated")
        return 0

    logger.info("\nSCD2 generation summary:")
    for table_name, metrics in summary.items():
        logger.info(f"  {table_name}: {metrics}")
    if getattr(args, "simulate", False) and getattr(args, "keep_snapshots", False):
        logger.info(f"  (kept intermediate snapshots: {previous_dir}, {current_dir})")
    if getattr(args, "no_effective_dates", False):
        logger.info("\nRemoving effective_from_ts/effective_to_ts (version dates kept in *_crt_dts columns)...")
        _strip_effective_date_columns(args.output, args.tables)
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
        from sdp.llm.schema_enricher import SchemaEnricher
        enricher = SchemaEnricher(confidence_threshold=args.confidence)
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
        from sdp.utils.er_diagram import ERDiagramGenerator
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
        from sdp.utils.cloud_uploader import upload_output
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
        from sdp.utils.collibra_importer import CollibraImporter
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
        from sdp.ml.auto_config import AutoConfigInferrer

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
        from sdp.ml.pii_detector import PIIDetector
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


def run_infer_relationships(args) -> int:
    """Infer FK relationships from a config and emit ER diagram + reviewable YAML."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.utils.config_parser import ConfigParser
        from sdp.ml.relationship_inferrer import MLRelationshipInferrer
        from sdp.ml.relationship_knowledge_graph import KnowledgeGraphRelationshipInferrer
        from sdp.ml.relationship_feedback_store import FeedbackStore

        if not validate_config_file(args.config):
            return 1

        parser = ConfigParser(args.config)
        if not parser.load_config():
            logger.error("Failed to load config")
            return 1
        tables = parser.parse_tables()
        existing = parser.parse_relationships()

        inferred: list = []
        method = args.method
        sample_data = _load_relationship_sample_data(getattr(args, "sample_data", None), tables)

        if method in ("ml", "both"):
            store = FeedbackStore(args.feedback_store) if args.feedback_store else FeedbackStore()
            inferrer_cls = KnowledgeGraphRelationshipInferrer if getattr(args, "ml_mode", "standard") == "knowledge-graph" else MLRelationshipInferrer
            ml_inferrer = inferrer_cls(
                confidence_threshold=args.ml_confidence,
                feedback_store=store,
            )
            ml_result = ml_inferrer.infer(
                tables,
                existing_relationships=existing,
                sample_data=sample_data,
            )
            inferred.extend(ml_result.relationships)
            logger.info(
                f"ML inferred {len(ml_result.relationships)} relationship(s) "
                f"using mode={getattr(args, 'ml_mode', 'standard')}; "
                f"classifier_fitted={ml_result.classifier_fitted} "
                f"(n={ml_result.classifier_examples} feedback examples)"
            )

        if method in ("llm", "both"):
            try:
                from sdp.llm.relationship_inferrer import RelationshipInferrer as LLMInferrer
                llm = LLMInferrer(confidence_threshold=args.llm_confidence)
                # When method=both, only ask LLM about relationships ML missed
                seen_pairs = {(r.source_table, r.source_column, r.target_table, r.target_column)
                              for r in inferred} if method == "both" else set()
                llm_result = llm.infer(tables, existing_relationships=existing)
                for rel in llm_result.relationships:
                    pair = (rel.source_table, rel.source_column, rel.target_table, rel.target_column)
                    if pair not in seen_pairs:
                        inferred.append(rel)
                logger.info(f"LLM contributed {len(llm_result.relationships)} relationship(s)")
            except Exception as exc:
                logger.warning(f"LLM inference unavailable, continuing with ML only: {exc}")

        inference_summary = _build_inference_summary(
            inferred,
            ml_mode=getattr(args, "ml_mode", "standard") if method in ("ml", "both") else None,
        )

        # Write YAML output
        output_path = Path(args.config_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simple_yaml = _should_write_simple_yaml(args)
        if simple_yaml:
            _write_simple_relationship_yaml(output_path, tables, existing, inferred)
        else:
            _write_reviewable_yaml(output_path, existing, inferred, inference_summary)
        yaml_mode = "simple" if simple_yaml else "review"
        logger.info(f"✅ Wrote {yaml_mode} YAML: {output_path}")

        # Optional ER diagram
        if args.er_output:
            _write_er_diagram(Path(args.er_output), tables, existing + inferred)

        # Print summary
        print(f"\n=== Inference summary ===")
        print(f"Method:                 {method}")
        if method in ("ml", "both"):
            print(f"ML mode:                {getattr(args, 'ml_mode', 'standard')}")
        print(f"Existing relationships: {len(existing)}")
        print(f"Inferred relationships: {len(inferred)}")
        if inference_summary.get("average_confidence") is not None:
            print(f"Average confidence:     {inference_summary['average_confidence']:.2f}")
        print(f"Recommendation:         {inference_summary['recommendation']}")
        for rel in inferred:
            conf = rel.ml_confidence if rel.inferred_by_ml else rel.llm_confidence
            tag = "ML" if rel.inferred_by_ml else "LLM"
            print(f"  [{tag} {conf:.2f}] {rel.source_table}.{rel.source_column} -> {rel.target_table}.{rel.target_column}")
        return 0
    except Exception as exc:
        logger.error(f"infer-relationships failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def _relationship_confidence(rel) -> Optional[float]:
    return rel.ml_confidence if rel.inferred_by_ml else rel.llm_confidence


def _should_write_simple_yaml(args) -> bool:
    """Resolve the effective YAML mode for infer-relationships.

    Default behaviour remains unchanged for the standard ML path, but the
    opt-in knowledge-graph mode now defaults to a cleaner, simple YAML unless
    callers explicitly request the richer review document via --review-yaml.
    """
    explicit = getattr(args, "simple_yaml", None)
    if explicit is not None:
        return bool(explicit)
    return getattr(args, "ml_mode", "standard") == "knowledge-graph"


def _confidence_band(confidence: Optional[float]) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def _relationship_recommendation(confidence: Optional[float]) -> str:
    band = _confidence_band(confidence)
    if band == "high":
        return "Spot-check, then likely keep"
    if band == "medium":
        return "Review before accepting"
    if band == "low":
        return "Review carefully or reject"
    return "Manual review required"


def _build_inference_summary(inferred_rels, ml_mode: Optional[str] = None) -> Dict[str, object]:
    confidences = [c for c in (_relationship_confidence(r) for r in inferred_rels) if c is not None]
    average_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    bands = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for rel in inferred_rels:
        bands[_confidence_band(_relationship_confidence(rel))] += 1

    recommendation = "No inferred relationships. Review schema hints or provide sample data."
    if inferred_rels:
        if bands["low"] == 0 and (average_confidence or 0.0) >= 0.80:
            recommendation = "High-confidence set. Spot-check key relationships, then keep the rest if they look right."
        elif bands["low"] <= max(1, len(inferred_rels) // 4):
            recommendation = "Mostly medium/high confidence. Review the medium-confidence relationships before accepting."
        else:
            recommendation = "Several low-confidence relationships exist. Review every inferred relationship carefully."
        if ml_mode == "standard" and bands["low"] > 0:
            recommendation += " If ambiguity remains, try --ml-mode knowledge-graph."

    return {
        "average_confidence": average_confidence,
        "confidence_bands": bands,
        "recommendation": recommendation,
    }


def _relationship_sort_key(rel) -> tuple:
    source_cols = rel.get("source_columns") or [rel.get("source_column") or ""]
    target_cols = rel.get("target_columns") or [rel.get("target_column") or ""]
    return (
        str(rel.get("source_table", "")),
        str(source_cols[0]),
        str(rel.get("target_table", "")),
        str(target_cols[0]),
        str(rel.get("name", "")),
    )


def _build_review_relationship_entry(rel, *, inferred: bool) -> Dict[str, object]:
    entry: Dict[str, object] = {
        "name": rel.name,
        "source_table": rel.source_table,
        "source_columns": [rel.source_column],
        "target_table": rel.target_table,
        "target_columns": [rel.target_column],
        "relationship_type": rel.relationship_type,
        "active": rel.active,
    }
    if not inferred:
        entry["review_status"] = "existing"
        return entry

    confidence = _relationship_confidence(rel)
    entry["review_status"] = "pending_review"
    entry["confidence_band"] = _confidence_band(confidence)
    entry["review_recommendation"] = _relationship_recommendation(confidence)
    if rel.inferred_by_ml:
        entry["inferred_by_ml"] = True
        entry["ml_confidence"] = rel.ml_confidence
    if rel.inferred_by_llm:
        entry["inferred_by_llm"] = True
        entry["llm_confidence"] = rel.llm_confidence
    entry["notes"] = "REVIEW: keep to accept, delete to reject, or edit the columns/table names to correct it."
    return entry


def _build_simple_table_entry(table_cfg: object) -> Dict[str, object]:
    return {
        "name": table_cfg.name,
        "rows": table_cfg.num_rows,
        "primary_key_columns": list(table_cfg.primary_key_columns or []),
        "columns": [
            {
                "name": col.column_name,
                "data_type": col.data_type,
                "is_pk": bool(col.is_pk),
                "is_fk": bool(col.is_fk),
                "nullable": bool(col.nullable),
            }
            for col in table_cfg.columns
        ],
    }


def _build_simple_relationship_entry(rel) -> Dict[str, object]:
    return {
        "name": rel.name,
        "source_table": rel.source_table,
        "source_columns": [rel.source_column],
        "target_table": rel.target_table,
        "target_columns": [rel.target_column],
        "relationship_type": rel.relationship_type,
        "active": rel.active,
    }


def _write_simple_relationship_yaml(path: Path, tables, existing_rels, inferred_rels) -> None:
    """Write a minimal YAML containing only table schemas and relationships."""
    import yaml as _yaml

    payload = {
        "config_format": "sdp-yaml-v1",
        "tables": [
            _build_simple_table_entry(table_cfg)
            for _, table_cfg in sorted(tables.items(), key=lambda item: item[0])
        ],
        "relationships": sorted(
            [_build_simple_relationship_entry(r) for r in [*existing_rels, *inferred_rels]],
            key=_relationship_sort_key,
        ),
    }
    path.write_text(_yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_reviewable_yaml(path: Path, existing_rels, inferred_rels, inference_summary: Optional[Dict[str, object]] = None) -> None:
    """Emit a YAML containing both the original relationships (kept) and the
    inferred ones (annotated with confidence + signals) so the SME can edit
    in place — delete what's wrong, keep what's right.
    """
    import yaml as _yaml
    rel_dicts = []
    debug_signals = {}
    for r in existing_rels:
        rel_dicts.append(_build_review_relationship_entry(r, inferred=False))
    for r in inferred_rels:
        rel_dicts.append(_build_review_relationship_entry(r, inferred=True))
        if r.name and r.inference_signals:
            debug_signals[r.name] = dict(r.inference_signals)

    rel_dicts = sorted(rel_dicts, key=_relationship_sort_key)

    payload = {
        "config_format": "sdp-yaml-v1",
        "_review_metadata": {
            "instructions": "Review each entry under 'relationships'. Keep accurate ones, delete inaccurate ones, "
                            "fix any column-name mistakes, then run `python main.py record-feedback "
                            "--inferred <this-file> --reviewed <your-edited-file>` to teach the system.",
            "existing_count": len(existing_rels),
            "inferred_count": len(inferred_rels),
            "summary": inference_summary or _build_inference_summary(inferred_rels),
        },
        "relationships": rel_dicts,
    }
    if debug_signals:
        payload["_review_debug"] = {
            "inference_signals_by_relationship": debug_signals,
        }
    path.write_text(_yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_relationship_sample_data(sample_data_arg: Optional[str], tables: Dict[str, object]) -> Optional[Dict[str, pd.DataFrame]]:
    """Load optional sample data for the value-subset signal.

    The CLI advertises `--sample-data`; keep it opt-in so existing behaviour is
    unchanged when callers do not provide it.
    """
    if not sample_data_arg:
        return None

    root = Path(sample_data_arg)
    if not root.exists() or not root.is_dir():
        logger.warning(f"sample-data path is not a readable directory: {root}")
        return None

    loaded: Dict[str, pd.DataFrame] = {}
    wanted = {str(name).lower() for name in tables.keys()}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        if stem not in wanted:
            continue
        try:
            if path.suffix.lower() == ".csv":
                loaded[stem] = pd.read_csv(path, nrows=5000)
            elif path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
                loaded[stem] = frame.head(5000) if len(frame) > 5000 else frame
        except Exception as exc:
            logger.warning(f"sample-data: skipped {path.name}: {exc}")

    if loaded:
        logger.info(f"Loaded sample data for {len(loaded)} table(s) from {root}")
        return loaded

    logger.warning(f"No matching sample CSV/Parquet files found in {root}")
    return None


def _write_er_diagram(out_path: Path, tables, relationships) -> None:
    """Emit an ER diagram in the format implied by the file extension."""
    try:
        from sdp.utils.er_diagram import ERDiagramGenerator
    except Exception as exc:
        logger.warning(f"ER diagram generator unavailable: {exc}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower().lstrip(".")
    fmt = {"mmd": "mermaid", "dot": "dot", "png": "png"}.get(suffix, "mermaid")
    gen = ERDiagramGenerator(tables, relationships)
    if fmt == "png":
        if gen.generate_png(out_path):
            logger.info(f"✅ Wrote ER diagram: {out_path}")
        return
    text = gen.generate_dot() if fmt == "dot" else gen.generate_mermaid()
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"✅ Wrote ER diagram: {out_path}")


def run_record_feedback(args) -> int:
    """Diff inferred-vs-reviewed YAML and persist accept/reject deltas."""
    configure_logging(getattr(args, "verbose", False))
    try:
        import yaml as _yaml
        from sdp.ml.relationship_feedback_store import FeedbackStore, FeedbackEntry

        inferred_doc = _yaml.safe_load(Path(args.inferred).read_text(encoding="utf-8")) or {}
        reviewed_doc = _yaml.safe_load(Path(args.reviewed).read_text(encoding="utf-8")) or {}
        debug_signals = ((inferred_doc.get("_review_debug") or {}).get("inference_signals_by_relationship") or {})

        def _index(doc) -> dict:
            out = {}
            for r in (doc.get("relationships") or []):
                src_cols = r.get("source_columns") or [r.get("source_column")]
                tgt_cols = r.get("target_columns") or [r.get("target_column")]
                for s_col, t_col in zip(src_cols or [], tgt_cols or []):
                    if not s_col or not t_col:
                        continue
                    key = (str(r.get("source_table", "")).lower(), s_col,
                           str(r.get("target_table", "")).lower(), t_col)
                    out[key] = r
            return out

        inferred = _index(inferred_doc)
        reviewed = _index(reviewed_doc)

        # Only score entries the inferrer originally proposed (others are user-authored).
        inferred_only = {k: v for k, v in inferred.items()
                         if v.get("inferred_by_ml") or v.get("inferred_by_llm")}

        store = FeedbackStore(args.feedback_store) if args.feedback_store else FeedbackStore()
        accept_count = reject_count = 0
        for key, original in inferred_only.items():
            kept = key in reviewed
            original_name = original.get("name")
            entry = FeedbackEntry(
                source_table=key[0],
                source_column=key[1],
                target_table=key[2],
                target_column=key[3],
                accepted=kept,
                signals=dict(original.get("inference_signals") or debug_signals.get(original_name) or {}),
                predicted_confidence=original.get("ml_confidence") or original.get("llm_confidence"),
                note="recorded via record-feedback",
            )
            store.append(entry)
            if kept:
                accept_count += 1
            else:
                reject_count += 1

        # Detect SME-added relationships (in reviewed but not in inferred) — these
        # are positive examples the inferrer missed. Recorded with empty signals
        # so they only contribute to pattern-memory, not classifier training.
        added = 0
        for key, r in reviewed.items():
            if key in inferred:
                continue
            entry = FeedbackEntry(
                source_table=key[0], source_column=key[1],
                target_table=key[2], target_column=key[3],
                accepted=True, signals={},
                note="SME-added (inferrer missed)",
            )
            store.append(entry)
            added += 1

        print(f"=== Feedback recorded ===")
        print(f"  Accepted (kept):  {accept_count}")
        print(f"  Rejected (gone):  {reject_count}")
        print(f"  SME-added:        {added}")
        print(f"  Store path:       {store.path}")
        stats = store.stats()
        print(f"  Total in store:   {stats['total']} ({stats['accepted']} accepted, {stats['rejected']} rejected)")
        return 0
    except Exception as exc:
        logger.error(f"record-feedback failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


# ===========================================================================
# Stubs / Mocks track — `mock-init`, `mock-render`, `mock-lint`
# ===========================================================================


def run_mock_init(args) -> int:
    """Convert an OpenAPI / Postman / HAR artefact into a sdp-mock-v1 YAML config."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import dump_mock_config

        source_type = _detect_mock_source_type(args.source, getattr(args, "source_type", "auto"))

        if source_type == "openapi":
            from sdp.mocks.openapi_importer import import_openapi
            cfg = import_openapi(args.source)
        elif source_type == "postman":
            from sdp.mocks.postman_importer import import_postman
            cfg = import_postman(args.source)
        elif source_type == "har":
            from sdp.mocks.har_importer import import_har
            cfg = import_har(args.source)
        else:
            logger.error(f"Could not detect source format for {args.source} — pass --source-type explicitly")
            return 1

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_mock_config(cfg, out_path)

        print(f"=== mock-init ===")
        print(f"  Source:      {args.source}")
        print(f"  Source type: {source_type}")
        print(f"  Output:      {out_path}")
        print(f"  Endpoints:   {len(cfg.endpoints)}")
        print(f"  Schemas:     {len(cfg.schemas)}")
        for ep in cfg.endpoints[:10]:
            statuses = ",".join(str(r.status) for r in ep.responses)
            print(f"    {ep.method:6s} {ep.path}  -> [{statuses}]  ({ep.name})")
        if len(cfg.endpoints) > 10:
            print(f"    ... and {len(cfg.endpoints) - 10} more")
        return 0
    except Exception as exc:
        logger.error(f"mock-init failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def _detect_mock_source_type(source_path: str, override: str) -> str:
    """Heuristic detection between openapi / postman / har sources."""
    if override and override != "auto":
        return override
    p = Path(source_path)
    if not p.exists():
        return "auto"
    suffix = p.suffix.lower()
    raw = ""
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")[:4096]
    except Exception:
        pass
    # HAR files always contain a top-level "log": with "version" + "entries"
    if '"log"' in raw and '"entries"' in raw:
        return "har"
    # Postman collections start with an `info` block referencing the schema
    if "schema.getpostman.com" in raw or '"_postman_id"' in raw:
        return "postman"
    # OpenAPI specs declare `openapi:` (3.x) or `swagger:` (2.x)
    if "openapi:" in raw or "swagger:" in raw or '"openapi"' in raw or '"swagger"' in raw:
        return "openapi"
    # Fallback by extension
    if suffix in (".yaml", ".yml"):
        return "openapi"
    return "auto"


def run_mock_render(args) -> int:
    """Render mocks (WireMock stubs / JSON fixtures / …) from a sdp-mock-v1 config."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import load_mock_config

        cfg = load_mock_config(args.config)
        formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_files = 0
        for fmt in formats:
            target = out_dir / fmt
            if fmt == "wiremock":
                from sdp.mocks.renderers.wiremock import render_wiremock
                files = render_wiremock(
                    cfg, target,
                    seed=args.seed,
                    examples_per_endpoint=args.examples,
                    match_mode=args.match_mode,
                )
                print(f"  wiremock:        {len(files)} mapping(s) -> {target}")
                total_files += len(files)
            elif fmt in ("json", "json-fixture", "fixture"):
                from sdp.mocks.renderers.json_fixture import render_json_fixtures
                files = render_json_fixtures(cfg, target, seed=args.seed)
                print(f"  json:            {len(files)} fixture(s) -> {target}")
                total_files += len(files)
            elif fmt == "pact":
                from sdp.mocks.renderers.pact import render_pact
                files = render_pact(
                    cfg, target,
                    consumer=getattr(args, "pact_consumer", "consumer"),
                    provider=getattr(args, "pact_provider", "provider"),
                    seed=args.seed,
                    examples_per_endpoint=args.examples,
                )
                print(f"  pact:            {len(files)} contract(s) -> {target}")
                total_files += len(files)
            elif fmt == "postman":
                from sdp.mocks.renderers.postman import render_postman
                files = render_postman(
                    cfg, target,
                    seed=args.seed,
                    examples_per_endpoint=args.examples,
                )
                print(f"  postman:         {len(files)} collection(s) -> {target}")
                total_files += len(files)
            elif fmt in ("openapi-examples", "openapi"):
                source = getattr(args, "openapi_source", None)
                if not source:
                    logger.error("openapi-examples format requires --openapi-source <spec>")
                    continue
                from sdp.mocks.renderers.openapi_examples import render_openapi_examples
                files = render_openapi_examples(
                    cfg, target,
                    source_spec=source,
                    seed=args.seed,
                    overwrite_existing=getattr(args, "openapi_overwrite", False),
                )
                print(f"  openapi-examples: {len(files)} file(s) -> {target}")
                total_files += len(files)
            else:
                logger.warning(f"Unknown format: {fmt} — skipping")

        print(f"\n=== mock-render ===")
        print(f"  Config:    {args.config}")
        print(f"  Output:    {out_dir}")
        print(f"  Formats:   {formats}")
        print(f"  Examples:  {args.examples or 'config-default'}")
        print(f"  Seed:      {args.seed if args.seed is not None else 'random'}")
        print(f"  Total:     {total_files} file(s)")
        return 0 if total_files > 0 else 1
    except Exception as exc:
        logger.error(f"mock-render failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def run_mock_enrich(args) -> int:
    """Use an LLM to fill missing examples and draft missing 4xx/5xx responses."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import load_mock_config, dump_mock_config
        from sdp.mocks.llm_enricher import enrich

        cfg = load_mock_config(args.config)
        enriched, result = enrich(
            cfg,
            fill_missing_examples=not getattr(args, "no_fill_examples", False),
            draft_error_responses=not getattr(args, "no_draft_errors", False),
            provider=getattr(args, "llm_provider", None),
            model=getattr(args, "llm_model", None),
            base_url=getattr(args, "llm_base_url", None),
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_mock_config(enriched, out_path)

        print(f"=== mock-enrich ===")
        print(f"  Source:           {args.config}")
        print(f"  Output:           {out_path}")
        print(f"  Examples added:   {result.examples_added}")
        print(f"  Errors drafted:   {result.error_responses_added}")
        if result.schemas_touched:
            print(f"  Schemas touched:  {', '.join(sorted(set(result.schemas_touched)))}")
        if result.endpoints_touched:
            print(f"  Endpoints touched:{', '.join(sorted(set(result.endpoints_touched)))}")
        for warn in result.warnings:
            print(f"  WARN: {warn}")
        return 0
    except Exception as exc:
        logger.error(f"mock-enrich failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def run_validate_data(args) -> int:
    """Run Great Expectations validation on a directory of generated Parquet files."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.validators.gx_validator import (
            HAS_GX, validate_tables, format_report,
        )
    except Exception as exc:
        logger.error(f"Could not import sdp.validators.gx_validator: {exc}")
        return 1
    if not HAS_GX:
        logger.error(
            "Great Expectations is not installed. Run: poetry install --extras gx"
        )
        return 1

    if not validate_config_file(args.config):
        return 1

    parser = load_config_context(args.config)
    tables = parser.tables

    report = validate_tables(
        tables,
        output_dir=args.input,
        row_count_tolerance=args.tolerance,
    )

    print(format_report(report, verbose=getattr(args, "verbose", False)))

    if getattr(args, "report_json", None):
        import json
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_serialise_report(report), encoding="utf-8")
        logger.info(f"\nWrote JSON report to {out_path}")

    if not report.success and getattr(args, "fail_on_error", False):
        return 2
    return 0


def _serialise_report(report) -> str:
    """Convert a ValidationReport into a JSON string."""
    import json

    payload = {
        "success": report.success,
        "total_expectations": report.total_expectations,
        "total_passed": report.total_passed,
        "total_failed": report.total_failed,
        "tables": {
            name: {
                "row_count": t.row_count,
                "success": t.success,
                "error": t.error,
                "total_expectations": t.total_expectations,
                "passed": t.passed_expectations,
                "failed": t.failed_expectations,
                "pass_rate": round(t.pass_rate, 4),
                "results": [
                    {
                        "expectation_type": r.expectation_type,
                        "column": r.column,
                        "success": r.success,
                        "observed_value": r.observed_value,
                        "unexpected_count": r.unexpected_count,
                        "unexpected_percent": r.unexpected_percent,
                        "details": r.details,
                    }
                    for r in t.results
                ],
            }
            for name, t in report.tables.items()
        },
    }
    return json.dumps(payload, indent=2, default=str)


def _parse_engine_options(raw: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """``["epochs=300", "batch_size=500"]`` → ``{"epochs": 300, ...}``.

    Digit-only values become ints — engine options are overwhelmingly
    numeric (epochs, batch_size), and passing "300" where an int is
    expected fails deep inside the engine with a poor message.
    """
    if not raw:
        return None
    options: Dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--engine-option expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        value = value.strip()
        options[key.strip()] = int(value) if value.isdigit() else value
    return options


def _print_engines() -> None:
    """Print the engine registry — name, availability, description."""
    from sdp.synthesizers import DEFAULT_ENGINE, describe

    print("Available generation engines:\n")
    for name, description, available in describe():
        mark = "  " if available else "! "
        default = "  (default)" if name == DEFAULT_ENGINE else ""
        print(f"{mark}{name:<16}{description}{default}")
    print("\n  ! = registered but dependencies unavailable")
    print("  Select with --engine NAME, or `synthesizer_engine` in Run_Settings.")


def _parse_targets(raw: Optional[List[str]]) -> Optional[Dict[str, str]]:
    """``["orders=status", "customers=tier"]`` → ``{"orders": "status", ...}``."""
    if not raw:
        return None
    targets: Dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"--target expects TABLE=COLUMN, got {item!r}"
            )
        table, column = item.split("=", 1)
        targets[table.strip()] = column.strip()
    return targets


def run_quality_report(args) -> int:
    """Run a statistical quality / fidelity / privacy report on generated data."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.validators.quality_report import quality_report_from_paths

        report = quality_report_from_paths(
            synthetic_dir=args.generated,
            source_dir=args.source,
            privacy_threshold=getattr(args, "privacy_threshold", 0.0),
            with_utility=not getattr(args, "no_utility", False),
            with_bias=not getattr(args, "no_bias", False),
            targets=_parse_targets(getattr(args, "target", None)),
        )
    except Exception as exc:
        logger.error(f"quality-report failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    md = report.to_markdown(max_columns_shown=200)
    if getattr(args, "verbose", False):
        print(md)
    else:
        # Print the header lines + a one-line per-table summary
        print("=" * 60)
        print("Synthetic Data Quality Report")
        print("=" * 60)
        if report.overall_fidelity is not None:
            print(f"Overall fidelity score: {report.overall_fidelity:.3f} "
                  "(1.0 = identical to source, 0.0 = disjoint)")
        elif report.has_source:
            print("Overall fidelity score: not computable")
        else:
            print("(univariate-only — no source data provided)")
        if report.overall_utility is not None:
            print(f"Overall utility (TSTR): {report.overall_utility:.3f} "
                  "(1.0 = as useful as real data)")
        if report.bias_flagged_tables:
            print(f"Bias flags: {', '.join(report.bias_flagged_tables)}")
        print()
        for name, t in report.tables.items():
            line = f"  {name:30s}  rows={t.row_count_synthetic:>8}"
            if t.fidelity_score is not None:
                line += f"  fidelity={t.fidelity_score:.3f}"
            if t.utility_ratio is not None:
                line += f"  utility={t.utility_ratio:.3f}"
            if t.privacy_nn_too_close_rate is not None:
                line += f"  privacy_too_close={t.privacy_nn_too_close_rate:.1%}"
            if t.biased_columns:
                line += f"  bias={len(t.biased_columns)}col"
            print(line)

    if getattr(args, "output_html", None):
        out = Path(args.output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.to_html(), encoding="utf-8")
        logger.info(f"\nWrote HTML report to {out}")
    if getattr(args, "output_json", None):
        import json
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, default=str),
                       encoding="utf-8")
        logger.info(f"\nWrote JSON report to {out}")

    return 0


def run_mock_lint(args) -> int:
    """Validate a sdp-mock-v1 config and pretty-print the report."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import lint_mock_config

        report = lint_mock_config(args.config)
        print(f"=== mock-lint: {report['path']} ===")
        if report["ok"]:
            print(f"  OK  ({report.get('endpoints', 0)} endpoints, {report.get('schemas', 0)} schemas)")
        else:
            print("  FAIL")
        for err in report.get("errors", []):
            print(f"  ERROR:   {err}")
        for warn in report.get("warnings", []):
            print(f"  WARN:    {warn}")
        return 0 if report["ok"] else 1
    except Exception as exc:
        logger.error(f"mock-lint failed: {exc}")
        return 1


def run_contract_test_cmd(args) -> int:
    """Verify a directory of Parquet data against a data contract (the config)."""
    configure_logging(getattr(args, "verbose", False))
    if not validate_config_file(args.contract):
        return 1
    data_dir = Path(args.data)
    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    try:
        from sdp.contracts import ContractError, run_contract_test
        from sdp.contracts.checker import format_contract_report
    except Exception as exc:
        logger.error(f"Could not import the contract checker: {exc}")
        return 1
    try:
        parser = load_config_context(args.contract)
    except Exception as exc:
        logger.error(f"Failed to load contract: {exc}")
        return 1
    try:
        report = run_contract_test(
            parser.tables,
            data_dir=data_dir,
            contract_name=Path(args.contract).name,
            row_count_tolerance=args.tolerance,
        )
    except ContractError as exc:
        logger.error(str(exc))
        return 1

    print(format_contract_report(report, verbose=getattr(args, "verbose", False)))

    if getattr(args, "report_json", None):
        import json
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote contract report to {out_path}")

    fail_on = getattr(args, "fail_on", "error")
    verdict = report.verdict.value
    if fail_on == "error" and verdict == "fail":
        return 2
    if fail_on == "warning" and verdict in ("fail", "warn"):
        return 2
    return 0


def run_contract_diff_cmd(args) -> int:
    """Detect breaking changes between two contract versions."""
    configure_logging(getattr(args, "verbose", False))
    if not validate_config_file(args.old) or not validate_config_file(args.new):
        return 1
    try:
        from sdp.contracts import diff_contracts
        from sdp.contracts.diff import format_contract_diff
    except Exception as exc:
        logger.error(f"Could not import the contract differ: {exc}")
        return 1
    try:
        old_parser = load_config_context(args.old)
        new_parser = load_config_context(args.new)
    except Exception as exc:
        logger.error(f"Failed to load a contract: {exc}")
        return 1

    diff = diff_contracts(
        old_parser.tables, new_parser.tables,
        old_name=Path(args.old).name, new_name=Path(args.new).name,
    )
    print(format_contract_diff(diff))

    if getattr(args, "report_json", None):
        import json
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(diff.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote diff report to {out_path}")

    if getattr(args, "fail_on_breaking", False) and diff.has_breaking:
        return 2
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
        if args.command == "infer-relationships":
            return run_infer_relationships(args)
        if args.command == "record-feedback":
            return run_record_feedback(args)
        if args.command == "mock-init":
            return run_mock_init(args)
        if args.command == "mock-render":
            return run_mock_render(args)
        if args.command == "mock-lint":
            return run_mock_lint(args)
        if args.command == "mock-enrich":
            return run_mock_enrich(args)
        if args.command == "validate-data":
            return run_validate_data(args)
        if args.command == "quality-report":
            return run_quality_report(args)
        if args.command == "contract-test":
            return run_contract_test_cmd(args)
        if args.command == "contract-diff":
            return run_contract_diff_cmd(args)
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
