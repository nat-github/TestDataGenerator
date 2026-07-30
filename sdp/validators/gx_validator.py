"""Great Expectations adapter — auto-derive expectations from ColumnConfig.

What this gives you:
  - One call (`validate_tables(...)`) takes the platform's tables_config
    plus a directory of generated Parquet files (or in-memory DataFrames)
    and runs Great Expectations against every table.
  - Expectations are derived from `ColumnConfig` so authors don't have to
    write any GX code by hand:

      ColumnConfig field            →  Expectation
      ────────────────────────────  ─────────────────────────────────────
      is_pk = True                  →  expect_column_values_to_be_unique
                                       + expect_column_values_to_not_be_null
      nullable = False              →  expect_column_values_to_not_be_null
      business_values = "A;B;C"     →  expect_column_values_to_be_in_set
      special_rules = "REGEX:^.."   →  expect_column_values_to_match_regex
      special_rules = "EMAIL"       →  expect_column_values_to_match_regex(<email regex>)
      special_rules = "UUID"        →  expect_column_values_to_match_regex(<uuid regex>)
      special_rules = "IPV4"        →  expect_column_values_to_match_regex(<ipv4 regex>)
      special_rules = "IBAN"        →  expect_column_values_to_match_regex(<iban regex>)
      min_value / max_value         →  expect_column_values_to_be_between
      length                         →  expect_column_value_lengths_to_be_between
      data_type starting with "N"   →  expect_column_values_to_be_in_type_list(int/float)

  - Plus table-level: row count between (num_rows / 2) and (num_rows * 2)
    by default, configurable via `row_count_tolerance`.

Failures are reported per (table, column) so authors can pinpoint exactly
which expectation tripped.

The GX dependency is optional — install with `poetry install --extras gx`.
This module degrades gracefully when GX isn't present (`HAS_GX = False`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


try:
    import great_expectations as gx
    from great_expectations.core.expectation_suite import ExpectationSuite
    HAS_GX = True
except ImportError:  # pragma: no cover
    HAS_GX = False
    gx = None  # type: ignore[assignment]
    ExpectationSuite = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Format regexes — used when special_rules implies a known shape
# ---------------------------------------------------------------------------

_FORMAT_REGEX: Dict[str, str] = {
    "EMAIL": r"^[\w\.\+-]+@[\w\.-]+\.[A-Za-z]{2,}$",
    "UUID":  r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "IPV4":  r"^(\d{1,3}\.){3}\d{1,3}$",
    "IPV6":  r"^[0-9a-fA-F:]+$",
    "MAC":   r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$",
    "URL":   r"^https?://[^\s]+$",
    # IBAN: 2-letter country, 2-digit check, up to 30 alphanumerics
    "IBAN":  r"^[A-Z]{2}[0-9]{2}[A-Z0-9]+$",
    "SWIFT": r"^[A-Z0-9]{8,11}$",
    "BIC":   r"^[A-Z0-9]{8,11}$",
    "SSN":   r"^\d{3}-?\d{2}-?\d{4}$",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ExpectationResult:
    """One expectation's pass/fail outcome."""
    expectation_type: str
    column: Optional[str]
    success: bool
    observed_value: Any = None
    unexpected_count: int = 0
    unexpected_percent: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableValidationReport:
    """Aggregated outcome for one table."""
    table_name: str
    row_count: int
    success: bool
    total_expectations: int
    passed_expectations: int
    failed_expectations: int
    results: List[ExpectationResult] = field(default_factory=list)
    error: Optional[str] = None  # populated when the validator itself failed

    @property
    def pass_rate(self) -> float:
        if self.total_expectations == 0:
            return 1.0
        return self.passed_expectations / self.total_expectations


@dataclass
class ValidationReport:
    """Overall outcome across every table."""
    success: bool
    tables: Dict[str, TableValidationReport] = field(default_factory=dict)

    @property
    def total_passed(self) -> int:
        return sum(t.passed_expectations for t in self.tables.values())

    @property
    def total_failed(self) -> int:
        return sum(t.failed_expectations for t in self.tables.values())

    @property
    def total_expectations(self) -> int:
        return sum(t.total_expectations for t in self.tables.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_tables(
    tables_config: Dict[str, Any],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    dataframes: Optional[Dict[str, Any]] = None,
    row_count_tolerance: float = 0.5,
) -> ValidationReport:
    """Validate generated data against expectations derived from `tables_config`.

    Provide *either* ``output_dir`` (a directory of `<table>.parquet` files) or
    ``dataframes`` (a dict of pandas DataFrames keyed by table name). When both
    are passed, ``dataframes`` wins.

    Parameters
    ----------
    tables_config:
        ``Dict[str, TableConfig]`` — same shape `ConfigParser.parse_tables()`
        returns.
    output_dir:
        Directory holding `<table>.parquet` files written by the data
        generator.
    dataframes:
        Pre-loaded DataFrames. Useful in tests so you don't have to round-trip
        through Parquet.
    row_count_tolerance:
        How much the actual row count may deviate from `TableConfig.num_rows`.
        ``0.5`` means actual ∈ [num_rows * 0.5, num_rows * 1.5]. Pass ``0`` to
        require an exact match (rarely a good idea — SDV produces a few extra
        rows when fitting fails). Default 0.5.
    """
    if not HAS_GX:
        raise RuntimeError(
            "Great Expectations is not installed. "
            "Run: poetry install --extras gx"
        )


    if dataframes is None:
        if output_dir is None:
            raise ValueError("Either output_dir or dataframes must be provided")
        dataframes = _load_parquet_dir(Path(output_dir))

    tables_report: Dict[str, TableValidationReport] = {}
    for table_name, table_cfg in tables_config.items():
        if not getattr(table_cfg, "active", True):
            continue
        df = dataframes.get(table_name)
        if df is None:
            tables_report[table_name] = TableValidationReport(
                table_name=table_name,
                row_count=0,
                success=False,
                total_expectations=0,
                passed_expectations=0,
                failed_expectations=0,
                error=f"No data found for table {table_name!r}",
            )
            continue

        try:
            tables_report[table_name] = _validate_one_table(
                table_name, table_cfg, df, row_count_tolerance
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("validate_tables: %s raised %s", table_name, exc)
            tables_report[table_name] = TableValidationReport(
                table_name=table_name,
                row_count=len(df) if hasattr(df, "__len__") else 0,
                success=False,
                total_expectations=0,
                passed_expectations=0,
                failed_expectations=0,
                error=f"{type(exc).__name__}: {exc}",
            )

    overall_success = all(t.success and not t.error for t in tables_report.values())
    return ValidationReport(success=overall_success, tables=tables_report)


# ---------------------------------------------------------------------------
# Per-table validation
# ---------------------------------------------------------------------------


def _validate_one_table(
    table_name: str,
    table_cfg: Any,
    df: Any,  # pandas.DataFrame
    row_count_tolerance: float,
) -> TableValidationReport:
    expectations = derive_expectations_for_table(
        table_name, table_cfg, row_count_tolerance=row_count_tolerance,
    )
    results = _run_expectations(df, expectations)

    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    return TableValidationReport(
        table_name=table_name,
        row_count=len(df),
        success=(failed == 0),
        total_expectations=len(results),
        passed_expectations=passed,
        failed_expectations=failed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Expectation derivation — pure logic, no GX dependency
# ---------------------------------------------------------------------------


def derive_expectations_for_table(
    table_name: str,
    table_cfg: Any,
    *,
    row_count_tolerance: float = 0.5,
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Walk a TableConfig and produce a list of (kind, column, kwargs) tuples.

    Each tuple maps onto a single GX expectation. Pure-function, easy to test
    without GX installed.
    """
    out: List[Tuple[str, Optional[str], Dict[str, Any]]] = []

    # Table-level: row count
    expected_rows = getattr(table_cfg, "num_rows", None) or 0
    if expected_rows > 0 and row_count_tolerance > 0:
        lower = max(1, int(expected_rows * (1 - row_count_tolerance)))
        upper = int(expected_rows * (1 + row_count_tolerance))
        out.append(("row_count_between", None, {"min_value": lower, "max_value": upper}))
    elif expected_rows > 0:
        out.append(("row_count_equal", None, {"value": expected_rows}))

    # Per-column. A composite primary key (more than one is_pk column) must NOT
    # assert uniqueness on each member individually — only the *combination* is
    # unique. Members still get a not-null check; the compound key gets one
    # table-level uniqueness check.
    columns = list(getattr(table_cfg, "columns", []))
    pk_names = [c.column_name for c in columns if getattr(c, "is_pk", False)]
    composite_pk = len(pk_names) > 1

    for col in columns:
        out.extend(derive_expectations_for_column(
            col, emit_pk_uniqueness=not composite_pk,
        ))

    if composite_pk:
        out.append(("compound_columns_to_be_unique", None, {"column_list": pk_names}))

    return out


def derive_expectations_for_column(
    col: Any,
    *,
    emit_pk_uniqueness: bool = True,
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Produce expectations for a single ColumnConfig.

    ``emit_pk_uniqueness`` is set False for members of a *composite* primary
    key — those are unique only in combination, not individually.
    """
    out: List[Tuple[str, Optional[str], Dict[str, Any]]] = []
    name = col.column_name

    # Column existence — every table is expected to contain its declared columns
    out.append(("column_exists", name, {}))

    # PK ⇒ (individually unique, unless composite) + not null
    if getattr(col, "is_pk", False):
        if emit_pk_uniqueness:
            out.append(("values_to_be_unique", name, {}))
        out.append(("values_to_not_be_null", name, {}))
    elif getattr(col, "nullable", True) is False:
        out.append(("values_to_not_be_null", name, {}))

    # Business values → membership
    bv_raw = getattr(col, "business_values", None)
    if bv_raw:
        bv_list = _parse_business_values(bv_raw)
        if bv_list:
            if (getattr(col, "data_type", None) or "").strip().upper() in ("D", "DT", "TS"):
                # Datetime membership must compare *normalized datetimes* — a
                # pandas Timestamp never equals an ISO date string, so the plain
                # set check would false-fail every row.
                out.append(("temporal_values_in_set", name, {"value_set": bv_list}))
            else:
                out.append(("values_to_be_in_set", name, {"value_set": bv_list}))

    # Min/max
    min_v = getattr(col, "min_value", None)
    max_v = getattr(col, "max_value", None)
    if min_v is not None or max_v is not None:
        # Numeric only — skip if the bounds are non-numeric date strings; the
        # type checker handles those.
        try:
            if min_v is not None:
                min_v = float(min_v)
            if max_v is not None:
                max_v = float(max_v)
            out.append((
                "values_to_be_between", name,
                {"min_value": min_v, "max_value": max_v},
            ))
        except (TypeError, ValueError):
            pass

    # Length
    length = getattr(col, "length", None)
    if length:
        out.append((
            "value_lengths_to_be_between", name,
            {"min_value": 1, "max_value": int(length)},
        ))

    # Special rules — derive a regex when we recognise the rule
    rule = getattr(col, "special_rules", None)
    if rule:
        regex = _regex_from_rule(rule)
        if regex:
            out.append(("values_to_match_regex", name, {"regex": regex}))

    # Type expectation from data_type
    dtype = getattr(col, "data_type", None)
    if (dtype or "").strip().upper() in ("D", "DT", "TS"):
        # Datetime columns get a dedicated, reliable temporal check.
        out.append(("column_is_temporal", name, {}))
    else:
        type_list = _gx_type_list(dtype)
        if type_list:
            out.append(("values_to_be_in_type_list", name, {"type_list": type_list}))

    return out


def _parse_business_values(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw if v is not None and str(v).strip()]
    if isinstance(raw, str):
        # Support semicolon and pipe separators
        parts = re.split(r"[;|]", raw)
        return [p.strip() for p in parts if p.strip()]
    return [str(raw)]


def _regex_from_rule(rule: str) -> Optional[str]:
    """Turn a special_rules string into a regex when we can.

    Returns ``None`` for rules we don't have a deterministic shape for —
    ``NAME``, ``ADDRESS``, etc. are too varied to validate without LLM-ish
    matching.
    """
    if not isinstance(rule, str):
        return None
    rule_clean = rule.strip()
    if rule_clean.upper().startswith("REGEX:"):
        # The pattern runs until the next rule separator (';'); strip trailing
        # rule tokens such as ';;NULL_RATE=0.20' so they don't corrupt the regex.
        pattern = rule_clean[6:].split(";", 1)[0].strip()
        return pattern or None
    # Strip locale suffix (`EMAIL:de_DE` → `EMAIL`)
    primary = rule_clean.split(":", 1)[0].strip().upper()
    return _FORMAT_REGEX.get(primary)


def _gx_type_list(data_type: Optional[str]) -> List[str]:
    """Map platform data_type to a list of acceptable pandas/numpy types."""
    if not data_type:
        return []
    dt = data_type.strip().upper()
    if dt.startswith("N") and dt != "NS":
        # N10/N19/N38 — int family
        return ["int", "int64", "int32", "Int64", "Int32", "float64"]
    if dt == "DC":
        return ["float", "float64", "Decimal", "object"]
    if dt.startswith("VA") or dt == "AN" or dt.startswith("A"):
        return ["str", "object", "string"]
    if dt == "NS":
        return ["str", "object", "string"]
    if dt in ("D", "DT", "TS"):
        # Date/datetime columns are checked by the dedicated `column_is_temporal`
        # expectation — GX's values_to_be_in_type_list is unreliable for datetime
        # dtypes (e.g. it reports datetime64[us, UTC] as 'Timestamp' and fails).
        return []
    return []


# ---------------------------------------------------------------------------
# Run expectations against a DataFrame
# ---------------------------------------------------------------------------


def _run_expectations(
    df: Any,
    expectations: List[Tuple[str, Optional[str], Dict[str, Any]]],
) -> List[ExpectationResult]:
    """Apply derived expectations to a DataFrame using GX 1.x's pandas validator."""
    if not HAS_GX:
        raise RuntimeError("GX not installed")

    import pandas as pd  # noqa: F401

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="sdp_pandas")
    data_asset = data_source.add_dataframe_asset(name="sdp_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("sdp_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results: List[ExpectationResult] = []
    for kind, column, kwargs in expectations:
        try:
            res = _evaluate_expectation(batch, df, kind, column, kwargs)
        except Exception as exc:  # pragma: no cover
            logger.warning("expectation %s on %s failed to evaluate: %s", kind, column, exc)
            res = ExpectationResult(
                expectation_type=kind, column=column, success=False,
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        results.append(res)
    return results


def _evaluate_expectation(
    batch: Any,
    df: Any,
    kind: str,
    column: Optional[str],
    kwargs: Dict[str, Any],
) -> ExpectationResult:
    """Run one of our supported expectation kinds.

    We map our compact internal kind names onto GX's expectation classes.
    Where GX's API is awkward (e.g. column existence) we evaluate directly
    against the DataFrame for clarity.
    """

    if kind == "column_exists":
        success = column in df.columns
        return ExpectationResult(
            expectation_type=kind, column=column, success=success,
            details={"observed_columns": list(df.columns)} if not success else {},
        )

    if kind == "row_count_between":
        actual = len(df)
        success = kwargs["min_value"] <= actual <= kwargs["max_value"]
        return ExpectationResult(
            expectation_type=kind, column=None, success=success,
            observed_value=actual,
            details=dict(kwargs, observed=actual),
        )

    if kind == "row_count_equal":
        actual = len(df)
        success = actual == kwargs["value"]
        return ExpectationResult(
            expectation_type=kind, column=None, success=success,
            observed_value=actual,
            details=dict(kwargs, observed=actual),
        )

    if kind == "compound_columns_to_be_unique":
        # Table-level: the *combination* of the composite-PK columns is unique.
        col_list = kwargs.get("column_list", [])
        missing = [c for c in col_list if c not in df.columns]
        if missing:
            return ExpectationResult(
                expectation_type=kind, column=None, success=False,
                details={"error": f"columns missing: {missing}"},
            )
        expectation = _build_gx_expectation(kind, "", kwargs)
        raw = batch.validate(expectation)
        summary = _summarise_gx_result(raw)
        return ExpectationResult(
            expectation_type=kind, column=None, success=bool(raw.success),
            observed_value=summary.get("observed_value"),
            unexpected_count=summary.get("unexpected_count", 0),
            unexpected_percent=summary.get("unexpected_percent", 0.0),
            details=summary,
        )

    # Column-level: route through GX's expectation classes
    if column is None or column not in df.columns:
        return ExpectationResult(
            expectation_type=kind, column=column, success=False,
            details={"error": f"column {column!r} missing"},
        )

    if kind == "column_is_temporal":
        # Direct temporal check — robust where GX's type-list check is not.
        # A native datetime dtype passes; an object/string column passes only
        # when every non-null value parses as a date/datetime.
        import pandas as pd

        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return ExpectationResult(
                expectation_type=kind, column=column, success=True,
                observed_value=str(series.dtype),
            )
        non_null = series.dropna()
        if len(non_null) == 0:
            return ExpectationResult(
                expectation_type=kind, column=column, success=True,
                observed_value=str(series.dtype),
            )
        parsed = pd.to_datetime(non_null, errors="coerce")
        bad = int(parsed.isna().sum())
        return ExpectationResult(
            expectation_type=kind, column=column, success=(bad == 0),
            observed_value=str(series.dtype),
            unexpected_count=bad,
            unexpected_percent=round(100.0 * bad / len(non_null), 2),
            details={"unparseable_values": bad},
        )

    if kind == "temporal_values_in_set":
        # Datetime set-membership compared on normalized (tz-naive) datetimes —
        # a pandas Timestamp never equals the ISO string in the declared set.
        import pandas as pd

        allowed = pd.to_datetime(
            pd.Series(list(kwargs.get("value_set", [])), dtype="object"),
            errors="coerce",
        ).dropna()
        if getattr(allowed.dt, "tz", None) is not None:
            allowed = allowed.dt.tz_localize(None)
        allowed_set = set(allowed)

        series = pd.to_datetime(df[column], errors="coerce")
        non_null = series.dropna()
        if getattr(non_null.dt, "tz", None) is not None:
            non_null = non_null.dt.tz_localize(None)

        if not allowed_set or len(non_null) == 0:
            return ExpectationResult(
                expectation_type=kind, column=column, success=True,
            )
        bad = int((~non_null.isin(allowed_set)).sum())
        return ExpectationResult(
            expectation_type=kind, column=column, success=(bad == 0),
            unexpected_count=bad,
            unexpected_percent=round(100.0 * bad / len(non_null), 2),
            details={"out_of_set_values": bad},
        )

    expectation = _build_gx_expectation(kind, column, kwargs)
    if expectation is None:
        return ExpectationResult(
            expectation_type=kind, column=column, success=False,
            details={"error": f"unsupported expectation kind: {kind}"},
        )

    raw = batch.validate(expectation)
    summary = _summarise_gx_result(raw)
    return ExpectationResult(
        expectation_type=kind,
        column=column,
        success=bool(raw.success),
        observed_value=summary.get("observed_value"),
        unexpected_count=summary.get("unexpected_count", 0),
        unexpected_percent=summary.get("unexpected_percent", 0.0),
        details=summary,
    )


def _build_gx_expectation(kind: str, column: str, kwargs: Dict[str, Any]):
    """Map our `kind` strings onto GX 1.x expectation classes."""
    import great_expectations.expectations as gxe

    if kind == "values_to_be_unique":
        return gxe.ExpectColumnValuesToBeUnique(column=column)
    if kind == "values_to_not_be_null":
        return gxe.ExpectColumnValuesToNotBeNull(column=column)
    if kind == "values_to_be_in_set":
        return gxe.ExpectColumnValuesToBeInSet(column=column, value_set=kwargs["value_set"])
    if kind == "values_to_match_regex":
        return gxe.ExpectColumnValuesToMatchRegex(column=column, regex=kwargs["regex"])
    if kind == "values_to_be_between":
        return gxe.ExpectColumnValuesToBeBetween(
            column=column,
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
        )
    if kind == "value_lengths_to_be_between":
        return gxe.ExpectColumnValueLengthsToBeBetween(
            column=column,
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
        )
    if kind == "values_to_be_in_type_list":
        return gxe.ExpectColumnValuesToBeInTypeList(
            column=column, type_list=kwargs["type_list"],
        )
    if kind == "compound_columns_to_be_unique":
        return gxe.ExpectCompoundColumnsToBeUnique(column_list=kwargs["column_list"])
    return None


def _summarise_gx_result(raw: Any) -> Dict[str, Any]:
    """Pull the bits of a GX ExpectationValidationResult we want to surface.

    GX's result.result dict shape varies slightly per expectation kind, so
    we defensively pull out commonly-present keys.
    """
    if raw is None:
        return {}
    inner = getattr(raw, "result", None) or {}
    return {
        "observed_value": inner.get("observed_value"),
        "unexpected_count": inner.get("unexpected_count", 0),
        "unexpected_percent": inner.get("unexpected_percent", 0.0),
        "element_count": inner.get("element_count"),
        "missing_count": inner.get("missing_count"),
        "partial_unexpected_list": inner.get("partial_unexpected_list", [])[:5],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parquet_dir(path: Path) -> Dict[str, Any]:
    """Read every `<name>.parquet` in `path` into a dict[name → DataFrame]."""
    import pyarrow.parquet as pq

    out: Dict[str, Any] = {}
    if not path.exists():
        raise FileNotFoundError(f"Output dir not found: {path}")
    for parquet_file in sorted(path.glob("*.parquet")):
        out[parquet_file.stem] = pq.read_table(parquet_file).to_pandas()
    return out


def format_report(report: ValidationReport, *, verbose: bool = False) -> str:
    """Pretty-print a ValidationReport for CLI output."""
    lines: List[str] = []
    status = "PASS" if report.success else "FAIL"
    lines.append(f"=== Great Expectations validation: {status} ===")
    lines.append(
        f"Tables: {len(report.tables)}, "
        f"expectations: {report.total_expectations}, "
        f"passed: {report.total_passed}, "
        f"failed: {report.total_failed}"
    )
    lines.append("")
    for name, t in report.tables.items():
        flag = "PASS" if t.success and not t.error else "FAIL"
        lines.append(
            f"  {flag}  {name:30s}  rows={t.row_count:>8}  "
            f"expectations={t.passed_expectations}/{t.total_expectations}"
        )
        if t.error:
            lines.append(f"         ERROR: {t.error}")
        for r in t.results:
            if r.success and not verbose:
                continue
            ok = "ok" if r.success else "X "
            col_part = f"[{r.column}]" if r.column else ""
            lines.append(f"         {ok}  {r.expectation_type} {col_part}")
            if not r.success:
                if r.unexpected_count:
                    lines.append(
                        f"             unexpected={r.unexpected_count} "
                        f"({r.unexpected_percent:.1f}%)"
                    )
                err = (r.details or {}).get("error")
                if err:
                    lines.append(f"             {err}")
                samples = (r.details or {}).get("partial_unexpected_list") or []
                if samples:
                    lines.append(f"             samples: {samples[:5]}")
    return "\n".join(lines)
