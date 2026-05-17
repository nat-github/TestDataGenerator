"""Streamlit page — Data contract testing.

Two workflows:

* **Test data** against a contract — upload a contract config + Parquet data,
  get a severity-tagged PASS / WARN / FAIL verdict.
* **Compare versions** — upload two contract versions, see which changes are
  breaking / additive / review.

Same engine as `sdp contract-test` and `sdp contract-diff` — runs in-process.
See `Data_Contract_Testing.md` for the concepts.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Same path-shim as the entry script — makes the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sdp.contracts import diff_contracts  # noqa: E402
from sdp.contracts.checker import ContractError, run_contract_test  # noqa: E402
from sdp.contracts.model import ChangeClass, Verdict  # noqa: E402
from sdp.utils.config_parser import ConfigParser  # noqa: E402

_CONFIG_SUFFIXES = {".xlsx", ".xls", ".yaml", ".yml", ".json"}

st.set_page_config(page_title="Data Contracts", page_icon="📜", layout="wide")

st.title("Data Contracts")
st.caption(
    "Verify real data against a contract, and detect breaking changes between "
    "contract versions. Same engine as `sdp contract-test` / `sdp contract-diff`."
)


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------

with st.expander("📘  What is a data contract — and why test it?"):
    st.markdown(
        """
A **data contract** is a versioned agreement between a data *producer* and its
*consumers* about a dataset: its schema, types, nullability, allowed values,
volume and referential integrity.

**Why it matters.** Without a contract, a producer renames a column or narrows
an enum and consumers' pipelines, dashboards and ML models break *silently and
downstream* — found in production, days later. A contract makes the expectations
**explicit and testable**, so a breaking change fails at the producer's CI
stage ("shift-left"), not in a consumer's 2 a.m. page.

**How it works here.**
- The platform config (tables, columns, types, constraints) **is** the contract.
- **Test data** runs Great Expectations checks derived from the contract against
  real data. Schema/integrity failures are **errors**; data-quality failures are
  **warnings**. The verdict is `PASS` / `WARN` / `FAIL`.
- **Compare versions** diffs two contracts and classifies every change as
  **breaking** (consumers break), **additive** (safe), or **review** (ambiguous).
"""
    )

tab_test, tab_diff = st.tabs(["✅  Test data against a contract", "🔀  Compare contract versions"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tables(uploaded) -> Dict[str, Any]:
    """Parse an uploaded contract config into a {table: TableConfig} dict."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in _CONFIG_SUFFIXES:
        raise ValueError(f"Unsupported contract type {suffix!r}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    parser = ConfigParser(tmp_path)
    if not parser.load_config():
        raise ValueError("Failed to load the contract config")
    parser.parse_tables()
    parser.parse_relationships()
    return parser.tables


_VERDICT_RENDER = {
    Verdict.PASS: ("success", "PASS — contract honoured"),
    Verdict.WARN: ("warning", "WARN — honoured, with data-quality warnings"),
    Verdict.FAIL: ("error", "FAIL — contract violated"),
}


# ---------------------------------------------------------------------------
# Tab 1 — test data against a contract
# ---------------------------------------------------------------------------

with tab_test:
    st.subheader("Test a dataset against its contract")
    contract_file = st.file_uploader(
        "Contract config", type=["xlsx", "xls", "yaml", "yml", "json"], key="ct_contract",
    )
    data_files = st.file_uploader(
        "Data — one or more `<table>.parquet` files",
        type=["parquet"], accept_multiple_files=True, key="ct_data",
    )
    tolerance = st.slider(
        "Row-count tolerance (±)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="How far the row count may deviate from the contract's declared num_rows.",
    )

    if st.button("Run contract test", type="primary", disabled=not (contract_file and data_files)):
        try:
            tables = _load_tables(contract_file)
            frames = {Path(f.name).stem: pd.read_parquet(f) for f in data_files}
            report = run_contract_test(
                tables, dataframes=frames,
                contract_name=contract_file.name, row_count_tolerance=tolerance,
            )
        except ContractError as exc:
            st.error(f"Contract test could not run: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001 - surface any parse/read error
            st.error(f"Failed: {exc}")
            st.stop()

        kind, label = _VERDICT_RENDER[report.verdict]
        getattr(st, kind)(f"**{label}**  —  {report.contract_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tables", len(report.tables))
        c2.metric("Checks", len(report.all_checks))
        c3.metric("Errors", len(report.error_failures))
        c4.metric("Warnings", len(report.warning_failures))

        rows = [
            {
                "status": "pass" if c.passed else "FAIL",
                "severity": c.severity.value,
                "table": c.table,
                "column": c.column or "",
                "check": c.check,
                "detail": c.detail,
            }
            for c in report.all_checks
        ]
        # Failures first.
        rows.sort(key=lambda r: (r["status"] == "pass", r["table"]))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report.to_dict(), indent=2, default=str),
            file_name="contract_test_report.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Tab 2 — compare contract versions
# ---------------------------------------------------------------------------

with tab_diff:
    st.subheader("Compare two contract versions")
    col_old, col_new = st.columns(2)
    old_file = col_old.file_uploader(
        "Previous contract", type=["xlsx", "xls", "yaml", "yml", "json"], key="cd_old",
    )
    new_file = col_new.file_uploader(
        "New contract", type=["xlsx", "xls", "yaml", "yml", "json"], key="cd_new",
    )

    if st.button("Compare versions", type="primary", disabled=not (old_file and new_file)):
        try:
            old_tables = _load_tables(old_file)
            new_tables = _load_tables(new_file)
            diff = diff_contracts(
                old_tables, new_tables,
                old_name=old_file.name, new_name=new_file.name,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed: {exc}")
            st.stop()

        if diff.has_breaking:
            st.error(f"**{len(diff.breaking)} breaking change(s)** — consumers will break.")
        elif diff.changes:
            st.success("No breaking changes — safe to publish.")
        else:
            st.info("No changes — the contracts are identical.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Breaking", len(diff.breaking))
        c2.metric("Review", len(diff.review))
        c3.metric("Additive", len(diff.additive))

        if diff.changes:
            order = {ChangeClass.BREAKING: 0, ChangeClass.REVIEW: 1, ChangeClass.ADDITIVE: 2}
            ordered = sorted(diff.changes, key=lambda c: order[c.classification])
            st.dataframe(
                pd.DataFrame([
                    {
                        "classification": c.classification.value,
                        "target": c.target,
                        "kind": c.kind,
                        "detail": c.detail,
                    }
                    for c in ordered
                ]),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Download diff (JSON)",
                data=json.dumps(diff.to_dict(), indent=2, default=str),
                file_name="contract_diff.json",
                mime="application/json",
            )
