"""Result objects for data contract testing — plain dataclasses, no GX import.

Two report families:

* :class:`ContractTestReport` — the outcome of validating data against a
  contract (produced by :mod:`sdp.contracts.checker`).
* :class:`ContractDiff` — the outcome of comparing two contract versions
  (produced by :mod:`sdp.contracts.diff`).
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """How serious a failed contract check is for downstream consumers."""

    ERROR = "error"       # schema / integrity — breaks consumers
    WARNING = "warning"   # data quality — degrades, but does not break


class Verdict(str, Enum):
    """Overall outcome of a contract test."""

    PASS = "pass"   # every check passed
    WARN = "warn"   # only warning-severity checks failed
    FAIL = "fail"   # at least one error-severity check failed


class ChangeClass(str, Enum):
    """Classification of a single change between two contract versions."""

    BREAKING = "breaking"   # consumers will break — e.g. column removed
    ADDITIVE = "additive"   # safe — e.g. new optional column
    REVIEW = "review"       # ambiguous — a human must judge (e.g. a rename)


def _json_scalar(v: Any) -> Any:
    """Coerce an observed value into something a strict JSON encoder accepts.

    GX observed values may be NaN, numpy scalars, etc. — strict JSON rejects
    NaN/Inf and cannot encode numpy types, so normalise here.
    """
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if (v == v and v not in (float("inf"), float("-inf"))) else None
    return str(v)


# ---------------------------------------------------------------------------
# Contract test report
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ContractCheck:
    """One assertion's pass/fail outcome within a contract test."""

    table: str
    column: Optional[str]
    check: str               # human-readable check name (e.g. "values_to_be_unique")
    severity: Severity
    passed: bool
    observed: Any = None     # observed value / unexpected count, when available
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "check": self.check,
            "severity": self.severity.value,
            "passed": self.passed,
            "observed": _json_scalar(self.observed),
            "detail": self.detail,
        }


@dataclasses.dataclass
class ContractTableResult:
    """Per-table contract outcome."""

    table: str
    row_count: int
    checks: List[ContractCheck] = dataclasses.field(default_factory=list)
    error: Optional[str] = None   # set when the table could not be checked at all

    @property
    def failures(self) -> List[ContractCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def error_failures(self) -> List[ContractCheck]:
        return [c for c in self.failures if c.severity is Severity.ERROR]

    @property
    def warning_failures(self) -> List[ContractCheck]:
        return [c for c in self.failures if c.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return self.error is None and not self.error_failures and not self.warning_failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "row_count": self.row_count,
            "error": self.error,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclasses.dataclass
class ContractTestReport:
    """Outcome of validating a dataset against a data contract."""

    contract_name: str
    tables: Dict[str, ContractTableResult] = dataclasses.field(default_factory=dict)

    @property
    def all_checks(self) -> List[ContractCheck]:
        return [c for t in self.tables.values() for c in t.checks]

    @property
    def error_failures(self) -> List[ContractCheck]:
        return [c for c in self.all_checks if not c.passed and c.severity is Severity.ERROR]

    @property
    def warning_failures(self) -> List[ContractCheck]:
        return [c for c in self.all_checks if not c.passed and c.severity is Severity.WARNING]

    @property
    def table_errors(self) -> List[str]:
        return [t.table for t in self.tables.values() if t.error]

    @property
    def verdict(self) -> Verdict:
        if self.error_failures or self.table_errors:
            return Verdict.FAIL
        if self.warning_failures:
            return Verdict.WARN
        return Verdict.PASS

    @property
    def passed(self) -> bool:
        """True when the contract is honoured (PASS or WARN — no hard errors)."""
        return self.verdict is not Verdict.FAIL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "verdict": self.verdict.value,
            "summary": {
                "tables": len(self.tables),
                "checks": len(self.all_checks),
                "errors": len(self.error_failures),
                "warnings": len(self.warning_failures),
                "table_errors": self.table_errors,
            },
            "tables": {name: t.to_dict() for name, t in self.tables.items()},
        }


# ---------------------------------------------------------------------------
# Contract diff report
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ContractChange:
    """One classified change between two contract versions."""

    kind: str               # e.g. "column_removed", "type_narrowed", "enum_shrunk"
    target: str             # "table" or "table.column"
    classification: ChangeClass
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "classification": self.classification.value,
            "detail": self.detail,
        }


@dataclasses.dataclass
class ContractDiff:
    """Outcome of comparing an old and a new contract version."""

    old_name: str
    new_name: str
    changes: List[ContractChange] = dataclasses.field(default_factory=list)

    @property
    def breaking(self) -> List[ContractChange]:
        return [c for c in self.changes if c.classification is ChangeClass.BREAKING]

    @property
    def additive(self) -> List[ContractChange]:
        return [c for c in self.changes if c.classification is ChangeClass.ADDITIVE]

    @property
    def review(self) -> List[ContractChange]:
        return [c for c in self.changes if c.classification is ChangeClass.REVIEW]

    @property
    def has_breaking(self) -> bool:
        return bool(self.breaking)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_name": self.old_name,
            "new_name": self.new_name,
            "has_breaking_changes": self.has_breaking,
            "summary": {
                "breaking": len(self.breaking),
                "additive": len(self.additive),
                "review": len(self.review),
            },
            "changes": [c.to_dict() for c in self.changes],
        }
