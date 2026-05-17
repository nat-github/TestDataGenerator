# Data Contract Testing

The platform can verify that data honours a **data contract**, and detect when
a contract changes in a way that breaks consumers. This document explains what
data contracts are, why they matter, and how the feature works.

---

## 1. What is a data contract?

A **data contract** is a versioned, machine-readable agreement between a data
*producer* (the team that writes a dataset) and its *consumers* (the pipelines,
dashboards, reports and ML models that read it). It pins down:

- **Schema** — which columns exist, and their types
- **Nullability** — which columns are guaranteed populated
- **Allowed values** — enums / business value sets
- **Ranges & formats** — numeric bounds, lengths, regex patterns
- **Keys & integrity** — primary keys, referential relationships
- **Volume** — expected row counts

It is the dataset's *API*: a stable promise the producer makes and consumers
depend on.

## 2. Why it matters

Without a contract, this happens:

> A producer renames `customer_id` to `cust_id`, or drops `BR` from the set of
> allowed country codes. Nothing fails *for them*. Three days later a consumer's
> nightly pipeline silently produces wrong numbers, or a dashboard breaks, or an
> ML model degrades — and someone gets paged at 2 a.m. to trace it back.

A data contract turns that silent, downstream, late failure into a **loud,
local, early** one:

- **Shift-left** — a breaking change fails in the *producer's* CI / pull request,
  before it ships, instead of in a consumer's production run.
- **Explicit expectations** — "what this dataset guarantees" stops being tribal
  knowledge and becomes a tested artefact.
- **Trust** — consumers can build on the dataset knowing it won't move under them
  without warning.

## 3. How it's implemented here

The platform already has a machine-readable schema — `TableConfig` / `ColumnConfig`
(from an Excel / YAML / JSON config) — and already compiles it into a Great
Expectations suite (`sdp/validators/gx_validator.py`). So:

> **The config _is_ the contract.** No new format to learn.

Two operations sit on top:

| Operation | Question it answers |
|---|---|
| `contract-test` | *Does this data honour the contract?* |
| `contract-diff` | *Did the contract itself change in a breaking way?* |

The assertion engine is **Great Expectations** — reused, not reimplemented.
`contract-test` re-frames raw GX results with a **severity** and an overall
**verdict**.

### Severity & verdict

Each derived check is tagged:

| Severity | Failing checks | Meaning |
|---|---|---|
| **error** | column missing, type mismatch, PK uniqueness / NOT NULL violations | schema/integrity — **breaks consumers** |
| **warning** | enum membership, range, length, regex, row-count | data quality — **degrades, doesn't break** |

The verdict rolls those up:

| Verdict | Condition |
|---|---|
| `PASS` | every check passed |
| `WARN` | only warning-severity checks failed |
| `FAIL` | at least one error-severity check failed |

### Breaking-change classification (`contract-diff`)

A pure structural comparison of two contract versions classifies each change:

| Classification | Examples |
|---|---|
| 🔴 **breaking** | column/table removed, PK changed, type narrowed, enum value removed, range tightened, length reduced, new NOT NULL column |
| 🟢 **additive** | new optional column/table, enum expanded, range widened, length increased, nullability relaxed |
| 🟡 **review** | type changed across families, column became NOT NULL — a human must judge |

---

## 4. Using it

### CLI

```bash
# Verify a directory of <table>.parquet files against a contract
sdp contract-test --contract config/Acct_bkng.xlsx --data data/incoming/ \
  --report-json reports/contract.json --fail-on error

# Detect breaking changes between two contract versions
sdp contract-diff --old config/acct_v1.yaml --new config/acct_v2.yaml \
  --fail-on-breaking
```

Exit codes make both CI-ready: `contract-test` exits `2` when failures reach the
`--fail-on` severity (`error` | `warning` | `none`); `contract-diff` exits `2`
with `--fail-on-breaking` when any breaking change is found.

### Python SDK

```python
from sdp import SyntheticDataPlatform

sdp = SyntheticDataPlatform()

report = sdp.contract_test(contract="config/Acct_bkng.xlsx", data="data/incoming/")
print(report.verdict.value)              # 'pass' | 'warn' | 'fail'
for check in report.error_failures:
    print(check.table, check.column, check.check, check.detail)

diff = sdp.contract_diff(old="config/acct_v1.yaml", new="config/acct_v2.yaml")
if diff.has_breaking:
    for change in diff.breaking:
        print(change.target, change.kind, change.detail)
```

### REST API (`api` extra)

```bash
# Verify data — upload the contract + a ZIP of <table>.parquet files
curl -X POST http://localhost:8000/contract-test \
  -F "contract=@config/Acct_bkng.xlsx" -F "data=@generated.zip" -F "tolerance=0.5"

# Compare two contract versions
curl -X POST http://localhost:8000/contract-diff \
  -F "old=@config/acct_v1.yaml" -F "new=@config/acct_v2.yaml"
```

### Streamlit UI

The **Data Contracts** page (`sdp/ui/pages/3_Data_Contracts.py`) offers both
workflows with file uploads, a colour-coded verdict, a per-check results table,
and JSON download — plus an inline "what is a data contract" primer.

```bash
poetry run streamlit run sdp/ui/streamlit_app.py
```

---

## 5. CI integration

`contract-test` is built to gate a pipeline. A producer's CI step:

```yaml
# .github/workflows/data-contract.yml (sketch)
- name: Verify the dataset honours its contract
  run: sdp contract-test --contract contracts/orders.yaml --data build/orders/ --fail-on error
```

And to block breaking contract edits in review:

```yaml
- name: Block breaking contract changes
  run: sdp contract-diff --old "$(git show origin/main:contracts/orders.yaml)" \
                         --new contracts/orders.yaml --fail-on-breaking
```

---

## 6. The closed loop

Because the *same* config drives both synthetic data generation **and** contract
verification, the platform closes a loop most tools leave open:

```
   config  ──generate──▶  conformant synthetic test data
     │
     └──────contract-test──▶  verify real/production data
```

One artefact: generate test data that *provably* matches the contract, and
verify that production data still does.

---

## 7. Roadmap (not yet implemented)

- **`contract:` block** — explicit contract metadata (name, version, owner) plus
  **freshness** (newest row within N hours) and explicit **volume** SLAs.
- **ODCS interop** — import/export the open Data Contract Standard so contracts
  interchange with other tooling.
- **HTML reports** for `contract-test`.

These are deliberately deferred; today's feature reuses the existing config and
GX engine with zero new dependencies.
