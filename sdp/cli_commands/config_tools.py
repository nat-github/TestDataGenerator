"""Config-authoring commands: lint, enrich, collibra-import, infer-config, pii-scan.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from sdp.generators.data_generator import DataGenerator
from sdp.services.common import (
    configure_logging,
    create_output_directory,
    load_config_context,
    validate_config_file,
    verify_export,
)
from sdp.utils.config_parser import ConfigParser
from sdp.utils.data_validator import DataValidator
from sdp.utils.parquet_post_processor import ParquetPostProcessor

logger = logging.getLogger(__name__)


def _schema_issues(config_path: str, *, strict: bool) -> List["object"]:
    """Validate against `schemas/sdp_config.schema.json`, as ConfigIssues.

    Returns an empty list for Excel configs (the schema describes the
    YAML/JSON document shape) and for an unavailable schema, which is a
    packaging problem rather than a config problem.
    """
    from sdp.utils.config_parser import ConfigIssue
    from sdp.utils.schema_validator import (
        SchemaUnavailable,
        schema_supported,
        validate_config_file as validate_against_schema,
    )

    if not schema_supported(config_path):
        return []

    try:
        violations = validate_against_schema(config_path)
    except SchemaUnavailable as exc:
        logger.warning(f"Schema validation unavailable: {exc}")
        return []

    level = "error" if strict else "warning"
    return [
        ConfigIssue(level=level, message=f"schema: {violation.message}",
                    sheet="schema", field_name=violation.path or None)
        for violation in violations
    ]


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

    # JSON Schema check. Reported as warnings by default: the schema is
    # deliberately permissive and the parser is deliberately tolerant, so a
    # violation means "this is probably not what you meant", not "this
    # cannot run". --strict-schema promotes them for CI.
    strict = getattr(args, "strict_schema", False)
    issues.extend(_schema_issues(args.config, strict=strict))

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
        logger.debug('Full traceback:', exc_info=True)
        return 1

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
        logger.debug('Full traceback:', exc_info=True)
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
            "",
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
        logger.debug('Full traceback:', exc_info=True)
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
        logger.debug('Full traceback:', exc_info=True)
        return 1
