# SDK & REST API

The Synthetic Data Platform can be driven programmatically two ways that sit on
top of the same code paths as the `sdp` CLI:

- **Python SDK** (`sdp.sdk`) — an in-process facade. Core install, no extras.
- **REST API** (`sdp.api`) — an HTTP layer over the SDK. Needs the `api` extra.

### How the SDK relates to the CLI

`SyntheticDataPlatform.generate()` calls
`sdp.services.generation.generate_dataset()` — **the same function the CLI
calls** — rather than building an argument vector and invoking the CLI.
Behaviour is identical because the code is shared, not because the inputs are
translated into flags.

That matters for what you get back. Because the SDK holds the service's
`GenerationOutcome`, `GenerationResult` exposes row counts, the generation
report, engine cost and DP privacy accounting:

```python
result = sdp.generate(config="config/loans.yaml", engine="dp-marginal", epsilon=0.5)

result.success            # bool
result.row_counts         # {"loans": 2000}
result.report             # total_records, relationships_configured, seed, ...
result.engine_stats       # {"engine": "dp-marginal", "fit_seconds": 0.003, ...}
result.privacy_report     # epsilon accounting — None for non-DP engines
result.frames             # tables as DataFrames
```

The one exception is `extra_args`, an escape hatch for raw CLI flags that
have no typed parameter. Passing it routes that call through argparse, and
the richer fields will be empty.

Other SDK methods (`lint`, `run`, contract helpers) still dispatch through
the CLI where no service function exists yet.

---

## Python SDK

### Install

Nothing extra — the SDK ships with the core package:

```python
from sdp import SyntheticDataPlatform
```

### `SyntheticDataPlatform`

```python
sdp = SyntheticDataPlatform(log_level="WARNING")   # raise to "INFO" for progress logs
```

#### `generate(...) -> GenerationResult`

```python
result = sdp.generate(
    config="config/Acct_bkng.xlsx",
    output="output/run_01",          # omit → a temp dir is created and kept
    default_records=1000,
    records={"orders": 500, "customers": 50},   # per-table overrides
    seed=42,
    validate=True,
    infer_relationships=False,
    method="ml",                     # ml | llm | both
)

result.success          # bool — exit code 0
result.output_dir       # Path to the Parquet output
result.tables           # ['customers', 'orders', ...]
result.total_records    # int
result.frames           # dict[str, pandas.DataFrame] — lazily read from Parquet
```

Parquet is **always** written to `output_dir`; `frames` reads it back for
in-process use. When `output` is omitted a temp directory is created and *kept*
— the caller owns cleanup.

#### `lint(config) -> LintResult`

```python
report = sdp.lint("config/Acct_bkng.xlsx")
report.ok               # bool — no error-level issues
report.errors           # list of issue dicts (level, message, sheet, row, column, ...)
report.warnings
report.report           # formatted text report
```

#### SCD2 & Delta extras

The `generate`, `scd2`, and `delta` methods accept additional **opt-in keyword
arguments** that mirror the CLI flags of the same name. Default values
preserve the previous behaviour, so existing calls are unaffected.

```python
# 1) generate writes each table as a Delta Lake table (partitioned, per-run append)
sdp.generate(
    config="cfg.yaml", output="out/", seed=42,
    write_delta=True,
    delta_partition_col="BOOKING_TM",
    delta_partition_value="20260531",
    delta_tables=["customers", "orders"],     # optional; else `write_delta: true` flags in YAML
)

# 2) scd2 self-contained mode — generates baseline + changed snapshot internally
sdp.scd2(
    config="cfg.yaml", output="hist/",
    simulate=True, default_records=1000, seed=42,
    change_fraction=0.3,
    no_effective_dates=True,                  # drop effective_from/to; keep dates in *_crt_dts
    previous_effective_ts="2026-01-01 00:00:00",
    effective_ts="2026-05-28 00:00:00",
)
# Classic two-snapshot mode is unchanged:
sdp.scd2("cfg.yaml", "v1/", "v2/", "hist/")

# 3) delta with optional partition overrides
sdp.delta(
    config="cfg.yaml", previous="v1/", current="v2/", output="delta/",
    partition_column="LOAD_DATE",             # or partition_columns=["A","B"]
    partition_start_date="20260101",
)
```

Three YAML fields drive the new generate-side features (read directly by
`main.py`/`sdp.cli`; the schema parser ignores them, so adding them to a
config never breaks anything):

| Field | Purpose |
|---|---|
| `versions_per_key: N` | repeat each business key 1..N times, varying every column in `scd2_tracked_columns` (dates shifted ~90 d; `business_values` cycle uniquely per key; `special_rules` regenerate). Output schema is unchanged — no extra columns. |
| `write_delta: true` | per-table marker — when `generate` is called with `--write-delta`, only flagged tables become Delta, the rest stay as flat parquet. Override per call with `--delta-tables` / `delta_tables=`. |
| `delta_partition_col: <COL>` | per-table override of the Delta partition column. Default for tables without it is the CLI flag `--delta-partition-col` / SDK `delta_partition_col=`. Use when different sources in the same run need different partition columns (e.g. one `BOOKING_TM`, one `LOAD_DT`). |

Because `delta_partition_col` is read from the YAML, **SDK and API callers
don't need any new arguments** — pass the same config file and the per-table
overrides apply automatically. The SDK/API `delta_partition_col` argument
remains the run-wide default for tables that don't set it themselves.

#### Other operations

| Method | Wraps |
|---|---|
| `delta(config, previous, current, output, partition_column=, partition_columns=, partition_start_date=, ...)` | `sdp delta` |
| `scd2(config, previous=, current=, output=, simulate=, default_records=, seed=, change_fraction=, change_columns=, keep_snapshots=, no_effective_dates=, effective_ts=, previous_effective_ts=, ...)` | `sdp scd2` |
| `infer_relationships(config, config_output, method=, ml_mode=, ...)` | `sdp infer-relationships` |
| `validate_data(config, input_dir, ...)` | `sdp validate-data` |
| `quality_report(generated, source=, ...)` | `sdp quality-report` |
| `mock_init(source, output, ...)` | `sdp mock-init` |
| `mock_render(config, output, formats=, ...)` | `sdp mock-render` |
| `contract_test(contract, data, ...) -> ContractTestReport` | `sdp contract-test` |
| `contract_diff(old, new) -> ContractDiff` | `sdp contract-diff` |
| `run(*argv) -> CommandResult` | **any** CLI command (escape hatch) |

`contract_test` / `contract_diff` return rich report objects (verdict, severity-tagged
checks, classified changes) — see `Data_Contract_Testing.md`.

```python
# Escape hatch — run any CLI command verbatim, in-process:
sdp.run("enrich", "--config", "config/bare.xlsx", "--output", "config/enriched.yaml")
```

These return a `CommandResult` (`command`, `exit_code`, `success`, `argv`).

---

## REST API

### Install & run

```bash
pip install "synthetic-data-platform[api]"      # or: poetry install --extras api

sdp-api --host 0.0.0.0 --port 8000              # console script
# equivalently:
uvicorn sdp.api:app --host 0.0.0.0 --port 8000
python -m sdp.api --port 8000
```

Interactive OpenAPI docs: `http://localhost:8000/docs`.

The API is **stateless** — every request uploads its own config, is processed
in an isolated temp directory, and nothing is persisted between requests.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz` | Liveness + version probe |
| `POST` | `/generate` | Generate synthetic data from an uploaded config (optionally as Delta Lake) |
| `POST` | `/scd2` | Build SCD2 history — with `simulate=true`, prior snapshots aren't needed |
| `POST` | `/delta` | Compute a CDC delta between two uploaded snapshot ZIPs |
| `POST` | `/lint` | Validate a config, return structured issues |
| `POST` | `/infer-relationships` | Infer FK relationships, return reviewable YAML |
| `POST` | `/contract-test` | Verify uploaded data (ZIP of Parquet) against a contract |
| `POST` | `/contract-diff` | Detect breaking changes between two contract versions |

#### `POST /generate`

Multipart form upload.

| Field | Type | Default | Notes |
|---|---|---|---|
| `config` | file | — | `.xlsx` / `.yaml` / `.yml` / `.json` |
| `seed` | int | — | Reproducible output |
| `default_records` | int | — | Rows per table |
| `records` | str | — | Per-table overrides: `orders:500,customers:50` |
| `validate` | bool | `false` | Validate FK relationships |
| `infer_relationships` | bool | `false` | Infer missing FKs first |
| `method` | str | `ml` | `ml` / `llm` / `both` |
| `response_format` | str | `zip` | `zip` (Parquet download) or `json` (row preview) |
| `write_delta` | bool | `false` | Write each table as a Delta Lake table at `<output>/<table>/` (with per-run partition append) |
| `delta_partition_col` | str | `BOOKING_TM` | Partition column name when `write_delta=true` |
| `delta_partition_value` | str | today YYYYMMDD | Partition value for this run |
| `delta_tables` | str | — | Comma-separated table list to convert to Delta (overrides per-table `write_delta: true` flags) |

```bash
# Download generated Parquet as a ZIP
curl -X POST http://localhost:8000/generate \
  -F "config=@config/Acct_bkng.xlsx" \
  -F "default_records=1000" -F "seed=42" \
  -o generated.zip

# Get a JSON preview instead
curl -X POST http://localhost:8000/generate \
  -F "config=@examples/configs/yaml/01_simple_users.yaml" \
  -F "default_records=20" -F "response_format=json"

# Write Delta tables (returned in the ZIP, with _delta_log + BOOKING_TM=…/ partitions)
curl -X POST http://localhost:8000/generate \
  -F "config=@config/Natural_Person_template_versioned.yaml" \
  -F "default_records=1000" -F "seed=42" \
  -F "write_delta=true" \
  -F "delta_partition_col=BOOKING_TM" \
  -F "delta_partition_value=20260531" \
  -o generated.zip
```

#### `POST /scd2`

Build SCD2 history from a config. With `simulate=true` the endpoint generates
the baseline + changed snapshot internally so the caller only uploads a config.

| Field | Type | Default | Notes |
|---|---|---|---|
| `config` | file | — | `.xlsx` / `.yaml` / `.yml` / `.json` |
| `simulate` | bool | `false` | Generate baseline + changed snapshot internally |
| `default_records` | int | — | With simulate: rows per table for the baseline |
| `seed` | int | — | With simulate: random seed |
| `change_fraction` | float | `0.3` | With simulate: fraction of rows that change between v1 and v2 |
| `change_columns` | str | — | With simulate: comma-separated tracked columns to change (default: per-table `scd2_tracked_columns` in config) |
| `no_effective_dates` | bool | `false` | Drop `effective_from_ts` / `effective_to_ts` from output |
| `effective_ts` | str | — | Effective timestamp for the current snapshot rows |
| `previous_effective_ts` | str | — | Bootstrap effective timestamp for previous snapshot rows |
| `tables` | str | — | Comma-separated list of tables to process |

```bash
curl -X POST http://localhost:8000/scd2 \
  -F "config=@config/Natural_Person_template_versioned.yaml" \
  -F "simulate=true" -F "default_records=1000" -F "seed=42" \
  -F "previous_effective_ts=2026-01-01 00:00:00" \
  -F "effective_ts=2026-05-28 00:00:00" \
  -o scd2.zip
```

#### `POST /delta`

Compute a CDC delta between two uploaded snapshot ZIPs and return the Delta
Lake output as a ZIP. Each snapshot ZIP is the ZIP you get back from
`/generate` — drop-in input.

| Field | Type | Default | Notes |
|---|---|---|---|
| `config` | file | — | `.xlsx` / `.yaml` / `.yml` / `.json` |
| `previous` | file | — | ZIP of the previous snapshot's parquet files |
| `current` | file | — | ZIP of the current snapshot's parquet files |
| `tables` | str | — | Comma-separated list of tables to process |
| `partition_column` | str | — | Override delta partition column for all tables |
| `partition_columns` | str | — | Comma-separated override of delta partition columns |
| `partition_start_date` | str | — | Override synthetic delta partition start date (YYYYMMDD or timestamp) |

```bash
curl -X POST http://localhost:8000/delta \
  -F "config=@config/Acct_bkng.xlsx" \
  -F "previous=@v1.zip" \
  -F "current=@v2.zip" \
  -F "partition_column=LOAD_DATE" \
  -o delta.zip
```

#### `POST /lint`

```bash
curl -X POST http://localhost:8000/lint -F "config=@config/Acct_bkng.xlsx"
```

Returns `{ ok, error_count, warning_count, issues[], report }`.

#### `POST /infer-relationships`

```bash
curl -X POST http://localhost:8000/infer-relationships \
  -F "config=@config/bare.xlsx" -F "method=ml" -F "ml_mode=knowledge-graph"
```

Returns `{ config_yaml }` — the reviewable inferred config as YAML text.

### Error handling

| Status | Meaning |
|---|---|
| `415` | Unsupported config file type |
| `422` | Bad input (malformed `records`, config that fails to load) |
| `500` | Generation/inference failed — try `POST /lint` to diagnose |

---

## Notes

- The SDK and API run generation **synchronously**. For large jobs prefer the
  CLI or run the API behind a queue.
- API endpoints offload the blocking SDK call to a worker thread, so the event
  loop stays responsive; run multiple `uvicorn` workers for concurrency.
- LLM-backed operations (`method=llm`, `enrich`) still require the relevant
  provider credentials (e.g. `ANTHROPIC_API_KEY`) in the server environment.
