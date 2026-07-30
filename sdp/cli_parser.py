"""Argument parser for the ``sdp`` CLI.

Split out of ``cli.py``: 350 lines of argparse wiring is a self-contained
concern, and keeping it beside the command handlers made both harder to
read.
"""
from __future__ import annotations

import argparse
import logging



logger = logging.getLogger(__name__)


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
    generate_parser.add_argument("--epsilon", type=float, default=None,
                                 help="Privacy budget per table for --engine dp-marginal "
                                      "(lower = more private, less accurate; default 1.0)")
    generate_parser.add_argument("--privacy-report-json", default=None,
                                 help="Write the DP privacy accounting to this path")
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
    lint_parser.add_argument("--strict-schema", action="store_true",
                             help="Treat JSON Schema violations as errors (exit non-zero). "
                                  "Reported as warnings by default. YAML/JSON configs only.")

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
