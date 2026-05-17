"""Contract checker — verifies data against a contract using Great Expectations.

This module does **not** implement assertions. It calls
:func:`sdp.validators.gx_validator.validate_tables` (which derives a GX suite
from the config) and re-frames the raw GX results as a contract verdict:

* each GX expectation is tagged with a :class:`~sdp.contracts.model.Severity`
  (schema/integrity failures are ``ERROR``, data-quality failures are
  ``WARNING``);
* the table- and contract-level :class:`~sdp.contracts.model.Verdict` is
  derived from those severities.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sdp.contracts.model import (
    ContractCheck,
    ContractTableResult,
    ContractTestReport,
    Severity,
    Verdict,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ContractError(RuntimeError):
    """Raised when a contract test cannot run (e.g. GX missing, no data)."""


# GX expectation kinds whose failure breaks downstream consumers outright.
# Everything else is treated as a data-quality WARNING.
_ERROR_KINDS = frozenset({
    "column_exists",
    "expect_column_to_exist",
    "values_to_be_unique",
    "expect_column_values_to_be_unique",
    "values_to_not_be_null",
    "expect_column_values_to_not_be_null",
    "values_to_be_in_type_list",
    "expect_column_values_to_be_in_type_list",
    "compound_columns_to_be_unique",
    "expect_compound_columns_to_be_unique",
    "column_is_temporal",
})


def _severity_for(expectation_type: str) -> Severity:
    """Map a GX expectation kind onto a contract severity."""
    return Severity.ERROR if expectation_type in _ERROR_KINDS else Severity.WARNING


def _detail_for(result: Any) -> str:
    """Build a short human-readable detail string from a GX ExpectationResult."""
    if result.success:
        return "ok"
    bits = []
    if getattr(result, "unexpected_count", 0):
        pct = getattr(result, "unexpected_percent", 0.0) or 0.0
        bits.append(f"{result.unexpected_count} unexpected value(s) ({pct:.1f}%)")
    observed = getattr(result, "observed_value", None)
    if observed is not None and not bits:
        bits.append(f"observed={observed!r}")
    return "; ".join(bits) or "expectation failed"


def run_contract_test(
    tables_config: Dict[str, Any],
    *,
    data_dir: Optional[PathLike] = None,
    dataframes: Optional[Dict[str, Any]] = None,
    contract_name: str = "",
    row_count_tolerance: float = 0.5,
) -> ContractTestReport:
    """Verify a dataset against a data contract.

    Parameters
    ----------
    tables_config:
        ``Dict[str, TableConfig]`` — the contract (as returned by
        ``ConfigParser.parse_tables()``).
    data_dir:
        Directory of ``<table>.parquet`` files to validate (the *real* data).
    dataframes:
        Pre-loaded DataFrames keyed by table name — an alternative to
        ``data_dir`` (handy in tests). Wins over ``data_dir`` if both given.
    contract_name:
        Display name for the contract (e.g. the config file name).
    row_count_tolerance:
        Permitted deviation of row count from the contract's declared
        ``num_rows`` (``0.5`` → ±50%).

    Returns
    -------
    ContractTestReport
        A report carrying a per-table, per-check severity-tagged result and an
        overall :class:`~sdp.contracts.model.Verdict`.
    """
    try:
        from sdp.validators.gx_validator import HAS_GX, validate_tables
    except Exception as exc:  # pragma: no cover - defensive
        raise ContractError(f"Could not import the GX validator: {exc}") from exc

    if not HAS_GX:
        raise ContractError(
            "Great Expectations is not installed — contract testing needs it. "
            "Run: poetry install --extras gx"
        )
    if data_dir is None and dataframes is None:
        raise ContractError("Provide either data_dir or dataframes to test against.")

    gx_report = validate_tables(
        tables_config,
        output_dir=str(data_dir) if data_dir is not None else None,
        dataframes=dataframes,
        row_count_tolerance=row_count_tolerance,
    )

    report = ContractTestReport(contract_name=contract_name or "data-contract")
    for table_name, table_report in gx_report.tables.items():
        checks = []
        for r in table_report.results:
            checks.append(ContractCheck(
                table=table_name,
                column=r.column,
                check=r.expectation_type,
                severity=_severity_for(r.expectation_type),
                passed=r.success,
                observed=getattr(r, "observed_value", None),
                detail=_detail_for(r),
            ))
        report.tables[table_name] = ContractTableResult(
            table=table_name,
            row_count=table_report.row_count,
            checks=checks,
            error=table_report.error,
        )
    return report


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

_VERDICT_LABEL = {
    Verdict.PASS: "PASS  — contract honoured",
    Verdict.WARN: "WARN  — contract honoured, data-quality warnings",
    Verdict.FAIL: "FAIL  — contract violated",
}


def format_contract_report(report: ContractTestReport, *, verbose: bool = False) -> str:
    """Render a :class:`ContractTestReport` as a human-readable text block."""
    lines = []
    lines.append("=" * 64)
    lines.append(f"Data contract test — {report.contract_name}")
    lines.append("=" * 64)
    lines.append(f"Verdict: {_VERDICT_LABEL[report.verdict]}")
    lines.append(
        f"  tables={len(report.tables)}  checks={len(report.all_checks)}  "
        f"errors={len(report.error_failures)}  warnings={len(report.warning_failures)}"
    )
    lines.append("")

    for name, table in report.tables.items():
        if table.error:
            lines.append(f"  [FAIL] {name}: {table.error}")
            continue
        status = "PASS" if table.passed else (
            "FAIL" if table.error_failures else "WARN"
        )
        lines.append(
            f"  [{status}] {name}  ({table.row_count} rows, "
            f"{len(table.checks)} checks, {len(table.failures)} failed)"
        )
        shown = table.failures if not verbose else table.checks
        for c in shown:
            mark = "ok" if c.passed else c.severity.value.upper()
            col = f".{c.column}" if c.column else ""
            lines.append(f"      {mark:>7}  {c.check}{col} — {c.detail}")
    lines.append("")
    return "\n".join(lines)
