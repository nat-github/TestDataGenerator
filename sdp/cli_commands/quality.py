"""Data-quality commands: validate-data, quality-report, contract-test, contract-diff.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


from sdp.services.common import (
    configure_logging,
    load_config_context,
    validate_config_file,
)

logger = logging.getLogger(__name__)


def run_validate_data(args) -> int:
    """Run Great Expectations validation on a directory of generated Parquet files."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.validators.gx_validator import (
            HAS_GX, validate_tables, format_report,
        )
    except ImportError as exc:
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

    try:
        report = validate_tables(
            tables,
            output_dir=args.input,
            row_count_tolerance=args.tolerance,
        )
    except (FileNotFoundError, OSError) as exc:
        # A missing --input directory is a user error, not a crash. It used
        # to escape as an uncaught traceback out of the command handler.
        logger.error(f"validate-data failed: {exc}")
        return 1

    print(format_report(report, verbose=getattr(args, "verbose", False)))

    if getattr(args, "report_json", None):
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_serialise_report(report), encoding="utf-8")
        logger.info(f"\nWrote JSON report to {out_path}")

    if not report.success and getattr(args, "fail_on_error", False):
        return 2
    return 0

def _serialise_report(report) -> str:
    """Convert a ValidationReport into a JSON string."""

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
        logger.debug('Full traceback:', exc_info=True)
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
    except ImportError as exc:
        logger.error(f"Could not import the contract checker: {exc}")
        return 1
    try:
        parser = load_config_context(args.contract)
    except (ValueError, OSError) as exc:
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
    except ImportError as exc:
        logger.error(f"Could not import the contract differ: {exc}")
        return 1
    try:
        old_parser = load_config_context(args.old)
        new_parser = load_config_context(args.new)
    except (ValueError, OSError) as exc:
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
