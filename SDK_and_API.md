# SDK & REST API

The Synthetic Data Platform can be driven programmatically two ways that sit on
top of the same code paths as the `sdp` CLI:

- **Python SDK** (`sdp.sdk`) — an in-process facade. Core install, no extras.
- **REST API** (`sdp.api`) — an HTTP layer over the SDK. Needs the `api` extra.

Both translate their inputs into the exact argument vector the CLI parser
expects, so behaviour is identical to running `sdp ...` on the command line.

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

#### Other operations

| Method | Wraps |
|---|---|
| `delta(config, previous, current, output, ...)` | `sdp delta` |
| `scd2(config, previous, current, output, ...)` | `sdp scd2` |
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
| `POST` | `/generate` | Generate synthetic data from an uploaded config |
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
