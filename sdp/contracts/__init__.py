"""Data contract testing for the Synthetic Data Platform.

A *data contract* is a versioned agreement about a dataset's schema, types,
nullability, allowed values, volume and referential integrity. This package
treats an existing platform config (`TableConfig` set) as the contract and:

* **verifies** real/incoming data against it (`checker.run_contract_test`), and
* **detects breaking changes** between two contract versions (`diff.diff_contracts`).

The assertion engine is Great Expectations — reused via
:mod:`sdp.validators.gx_validator`; nothing is reimplemented here.

See `Data_Contract_Testing.md` for the concepts and the why.
"""
from __future__ import annotations

from sdp.contracts.checker import ContractError, run_contract_test
from sdp.contracts.diff import diff_contracts
from sdp.contracts.model import (
    ChangeClass,
    ContractChange,
    ContractCheck,
    ContractDiff,
    ContractTableResult,
    ContractTestReport,
    Severity,
    Verdict,
)

__all__ = [
    "run_contract_test",
    "diff_contracts",
    "ContractError",
    "ContractTestReport",
    "ContractTableResult",
    "ContractCheck",
    "ContractDiff",
    "ContractChange",
    "Severity",
    "Verdict",
    "ChangeClass",
]
