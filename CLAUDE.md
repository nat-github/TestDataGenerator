# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic Data Platform — generates realistic, relationship-aware Parquet data from Excel or YAML configs. Supports snapshot generation, delta (CDC) processing, and SCD2 history. Uses SDV (Synthetic Data Vault) with automatic fallback to rule-based generation.

## Commands

```bash
# Install dependencies
poetry install

# After `poetry install` (or installing the wheel) the CLI is also available as
# the `sdp` console script — `sdp generate ...` == `python main.py generate ...`.
# The examples below use `python main.py`; both forms are equivalent.

# Generate a snapshot
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --default-records 1000

# Generate from a JSON config
python main.py generate --config config/sample_rules_and_cdc.json --output output/sample_run --seed 42

# Generate using rules + derived columns + cdc:
python main.py generate --config config/sample_rules_and_cdc.yaml --output output/sample_run --seed 42

# Reproducible run (--seed makes output deterministic)
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --seed 42

# Infer missing FK relationships before generation (default --method ml — free, no API key needed)
python main.py generate --config config/bare.xlsx --output output/run_01 --infer-relationships
# Use Claude instead, or both engines (LLM picks up what ML missed). LLM path needs ANTHROPIC_API_KEY.
python main.py generate --config config/bare.xlsx --output output/run_01 --infer-relationships --method llm --llm-confidence 0.7

# Standalone: infer relationships and emit a reviewable YAML + ER diagram for SME review
python main.py infer-relationships --config config/bare.xlsx --config-output config/inferred.yaml --er-output diagrams/inferred.mmd

# Feed the SME's edits (kept / removed / added entries) back into the adaptive feedback store
python main.py record-feedback --inferred config/inferred.yaml --reviewed config/inferred.yaml.reviewed

# Stubs / Mocks track — convert an OpenAPI / Postman / HAR artefact into editable sdp-mock-v1 YAML
python main.py mock-init --from examples/openapi/medium_tasks.yaml --output mocks/tasks.yaml
python main.py mock-init --from session.har --output mocks/captured.yaml          # HAR auto-detected
python main.py mock-init --from collection.json --output mocks/postman.yaml       # Postman auto-detected

# Render any of: wiremock, json, pact, postman, openapi-examples
python main.py mock-render --config mocks/tasks.yaml --output stubs/tasks \
  --format wiremock,json,pact,postman --examples 5 --seed 42 --match-mode any \
  --pact-consumer client-app --pact-provider tasks-svc

# OpenAPI examples enrichment — round-trip a spec with example: blocks injected
python main.py mock-render --config mocks/tasks.yaml --output stubs/oas \
  --format openapi-examples --openapi-source examples/openapi/medium_tasks.yaml

# LLM-assisted authoring — fill missing examples + draft 4xx/5xx error envelopes
python main.py mock-enrich --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml

# Validate a mock config without raising
python main.py mock-lint --config mocks/tasks.yaml

# Validate config with full sheet/row/column error context (no data generated)
python main.py lint --config config/Acct_bkng.xlsx
# YAML/JSON configs are also checked against schemas/sdp_config.schema.json.
# Violations are warnings by default; --strict-schema makes them errors (for CI).
python main.py lint --config config/orders.yaml --strict-schema

# Enrich a bare config with Claude-suggested business_values and special_rules
python main.py enrich --config config/bare.xlsx --output config/enriched.yaml --confidence 0.7

# Data contract testing — verify real data against the config-as-contract
python main.py contract-test --contract config/Acct_bkng.xlsx --data data/incoming/ --fail-on error
# Detect breaking changes between two contract versions
python main.py contract-diff --old config/acct_v1.yaml --new config/acct_v2.yaml --fail-on-breaking

# Legacy mode (backward compatible, treated as generate)
python main.py --config config/Acct_bkng.xlsx --output output/run_01

# Delta between two snapshots
python main.py delta --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/delta_run

# SCD2 history
python main.py scd2 --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/scd2_run

# Generate with ER diagram (--er-format accepts: mermaid dot png)
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --er-diagram --er-format mermaid dot

# Upload output to cloud after generation
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --upload-to azure://my-container/prefix
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --upload-to s3://my-bucket/prefix

# Import dataset definition from Collibra data catalog
python main.py collibra-import --dataset "Account Booking" --output config/from_collibra.yaml
python main.py collibra-import --dataset "Customer" --domain "Finance" --output config/customer.yaml

# Inspect parquet output
poetry run python readParquet.py output/run_01

# Build the installable wheel (→ dist/synthetic_data_platform-<ver>-py3-none-any.whl)
poetry build

# Python SDK (in-process; ships with the core install)
python -c "from sdp import SyntheticDataPlatform; \
  SyntheticDataPlatform().generate(config='config/Acct_bkng.xlsx', output='output/run_01', seed=42)"

# REST API (needs the api extra)
poetry install --extras api
poetry run sdp-api --host 0.0.0.0 --port 8000      # docs at http://localhost:8000/docs

# Run all tests (pytest lives in the dev group — `poetry install` includes it,
# but the published wheel does not depend on it)
poetry run pytest tests/ -v

# Lint (ruff config in pyproject.toml — narrow rule set, kept at zero findings)
poetry run ruff check .

# Build a slim image without the ui/mcp/gx/mimesis extras
docker build -f Dockerfile.slim -t sdp:slim .

# Run a single test
poetry run pytest tests/test_config_and_parquet_flows.py::test_name -v
```

### Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required when the LLM provider is `anthropic` (the default) |
| `SDP_LLM_PROVIDER` | Pick LLM backend: `anthropic` (default), `openai`, `lm-studio`, `ollama`, `azure-openai`, `groq`, `together`, `openrouter` |
| `SDP_LLM_MODEL` | Model identifier — defaults to provider-specific (e.g. `claude-sonnet-4-6`, `gpt-4o-mini`, `llama3.2`) |
| `SDP_LLM_BASE_URL` | Override base URL for OpenAI-compatible providers (LM Studio, Ollama, vLLM, custom hosts) |
| `SDP_LLM_API_KEY` | Generic LLM API key fallback; or set provider-specific (`OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`) |
| `SDP_FEEDBACK_PATH` | Override default location of the ML relationship feedback store (default: `ml_feedback/relationship_feedback.jsonl`) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure upload (preferred) |
| `AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_KEY` | Azure upload (alternative) |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | AWS S3 upload |
| `COLLIBRA_BASE_URL` | Collibra import base URL |
| `COLLIBRA_USERNAME` + `COLLIBRA_PASSWORD` | Collibra import credentials |

### Optional dependencies (install as needed)

```bash
pip install azure-storage-blob       # --upload-to azure://...
pip install boto3                    # --upload-to s3://...
pip install matplotlib               # --er-format png
poetry install --extras mimesis      # MIMESIS_* special rules
```

## Architecture

### Package layout

All code lives under a single installable package, **`sdp/`**. In the module
tables below, a path like `utils/config_parser.py` means `sdp/utils/config_parser.py`.
The package builds to a wheel (`poetry build`) and installs two console scripts:
`sdp` (CLI) and `sdp-api` (REST API). See `Packaging.md`.

Four entry surfaces share the same code paths: `sdp/cli.py` (CLI),
`sdp/sdk.py` (Python SDK — `SyntheticDataPlatform`), `sdp/api/` (REST API,
FastAPI), `sdp/mcp_server/` (MCP). See `SDK_and_API.md`.

### Entry Point and the service layer

`sdp/cli.py` (~360 lines) is **argument dispatch only**. The work sits behind
it:

| Module | Owns |
|---|---|
| `sdp/cli_parser.py` | the argparse tree (`build_parser`) |
| `sdp/cli_commands/*.py` | one module per command group — `cdc`, `config_tools`, `mocks`, `quality`, `relationships` |
| `sdp/services/generation.py` | `generate_dataset()` — the generation run itself |
| `sdp/services/common.py` | config validation, output dirs, export verification |

**The service layer is the important part.** `generate_dataset(GenerationRequest)
-> GenerationOutcome` is called by both `cli.run_generate` and
`sdk.generate`. Nothing under `sdp/services/` imports `sdp.cli`, so the SDK
is no longer a CLI wrapper: it gets row counts, the generation report, engine
cost and DP privacy accounting instead of just an exit code.

`GenerationRequest`'s field names deliberately match the `generate`
subcommand's argparse destinations, so an argparse `Namespace` and a
`GenerationRequest` are interchangeable — which is what let the orchestration
move without rewriting the helpers it calls.

The CLI presentation layer (`cli.py`, `cli_parser.py`, `cli_commands/`) may
`print`; everything else must log, and `test_hardening_fixes.py` enforces it.
A root `main.py` shim re-exports `sdp.cli`, so `python main.py ...` and
`from main import ...` keep working.

### Core Data Flow

```
Excel / YAML Config
      │
 ConfigParser (utils/config_parser.py)
      │  Parses 4 sheets: Run_Settings, Tables, Columns (required), Relationships
      │  → TableConfig, ColumnConfig, RelationshipConfig (models/config_models.py)
      │
 DataGenerator (generators/data_generator.py)
      │  1. create_sdv_metadata()      — builds SDV Metadata object
      │  2. train_synthesizer()        — fits HMASynthesizer on sample data; marks is_fitted
      │  3. generate_data()            — SDV if fitted, else rule-based fallback
      │  4. export_to_parquet()        — Arrow type casting → .parquet files
      │
 ParquetPostProcessor (utils/parquet_post_processor.py)   [delta / scd2 only]
         Compares two snapshot dirs → writes Delta Lake tables or SCD2 history
      │
 ERDiagramGenerator (utils/er_diagram.py)               [--er-diagram only]
         Produces Mermaid (.mmd), Graphviz DOT, or PNG ER diagrams
      │
 upload_output (utils/cloud_uploader.py)                [--upload-to only]
         Uploads output directory to Azure Blob Storage or AWS S3
      │
 CollibraImporter (utils/collibra_importer.py)          [collibra-import only]
         Fetches dataset/column definitions from Collibra REST API v2 → YAML
```

### Key Modules

| Module | Responsibility |
|---|---|
| `generators/data_generator.py` | Main orchestrator (~1,160 lines). Composes six mixins below; what remains here is the run sequence, per-column value generation, rules/derived columns, and the fallback path. |
| `generators/_primary_keys.py` | `PrimaryKeyMixin` — PK generation, sequences, uniqueness repair (single + composite) |
| `generators/_arrow_export.py` | `ArrowExportMixin` — Arrow type casting (decimal/bigint) and Parquet export |
| `generators/_sdv_metadata.py` | `SDVMetadataMixin` — SDV `Metadata` construction, sdtype mapping, training-sample sanitisation |
| `generators/_relationships.py` | `RelationshipMixin` — FK resolution, relationship groups, referential integrity |
| `generators/_model_cache.py` | `ModelCacheMixin` — fingerprint-keyed synthesizer artifacts, SDV-version validation |
| `generators/_anchors.py` | `AnchorMixin` — `source:` tables loaded verbatim from real data |
| `utils/config_parser.py` | Excel & YAML parsing, validation with row/column error context, `lint_config()` |
| `utils/helpers.py` | Regex generation, Faker integration (18 locales), type coercion, NULL rate logic, 60+ special rules |
| `utils/parquet_post_processor.py` | Delta (I/U/D) and SCD2 effective-dating logic with delta-log rollback safety. Delta writes honour `delta_write_mode` (`overwrite` default / `append` / `error`); the overwrite path warns before discarding an existing table's change history. |
| `utils/rule_evaluator.py` | Layer A (when/then) and Layer B (derived expressions) post-generation pass — operates on plain dicts so it can be reused by stub/mock renderers later |
| `utils/schema_validator.py` | JSON Schema enforcement for YAML/JSON configs against `schemas/sdp_config.schema.json`. Reports every violation with a document path (`workflows[0].transitions[1].probability`). Wired into `lint` as warnings; `--strict-schema` promotes them to errors. Excel configs are skipped — the schema describes a document shape. |
| `utils/workflow_engine.py` | Layer C — lifecycle state machines. Walks each row through declared transitions, back-fills only the visited states' timestamps in increasing order, NULLs the rest. `validate_workflow()` is called by `lint`. |
| `utils/mimesis_provider.py` | Optional Mimesis adapter — handles `MIMESIS_*` special rules. Lazy import; the library is an optional poetry extra |
| `utils/data_validator.py` | Post-generation FK relationship validation |
| `utils/er_diagram.py` | ER diagram generation: Mermaid (zero-dep), Graphviz DOT, PNG (matplotlib optional) |
| `utils/cloud_uploader.py` | Azure Blob Storage and AWS S3 upload — credentials from env vars only |
| `utils/collibra_importer.py` | Collibra REST API v2 importer — dataset → YAML config |
| `models/config_models.py` | Pydantic v2 models: `TableConfig`, `ColumnConfig`, `RelationshipConfig`, `DataType` enum |
| `llm/client.py` | Shared Anthropic client factory + cache-friendly system prompt (used by the Anthropic-specific path) |
| `llm/multi_provider.py` | Unified `chat()` abstraction — routes to Anthropic, OpenAI, LM Studio, Ollama, Azure, Groq, Together, OpenRouter via `SDP_LLM_PROVIDER` |
| `llm/relationship_inferrer.py` | LLM relationship inference — calls `multi_provider.chat`; provider-agnostic |
| `llm/schema_enricher.py` | LLM schema enrichment — calls `multi_provider.chat`; suggests `business_values`, `special_rules`, `data_type` corrections |
| `ml/relationship_signals.py` | Pure-function signal computers (type, name, value-subset, pk-likeness) + weighted `combine` |
| `ml/relationship_feedback_store.py` | JSONL-backed pattern memory + classifier training corpus, with `SDP_FEEDBACK_PATH` env override |
| `ml/relationship_classifier.py` | Cold-start-safe logistic-regression classifier; refuses below `MIN_TRAINING_EXAMPLES`/`MIN_PER_CLASS` |
| `ml/relationship_inferrer.py` | ML/heuristic relationship inferrer — same `infer(...)` surface as the LLM path, no API key needed |
| `models/mock_models.py` | Pydantic models for the stubs/mocks track: `MockConfig`, `EndpointConfig`, `RequestMatcher`, `ResponseTemplate`, `SchemaConfig`, `FieldSpec`, `ScenarioConfig` |
| `mocks/config_parser.py` | `sdp-mock-v1` YAML/JSON loader, dumper, and linter |
| `mocks/openapi_importer.py` | OpenAPI 3.x spec → `MockConfig` (handles $refs, allOf, format hints, multi-status responses) |
| `mocks/template_engine.py` | Walks `MockConfig` schemas/fields, generates JSON values via `utils/helpers.py` |
| `mocks/renderers/wiremock.py` | `MockConfig` → WireMock-compatible mapping JSON files; honours scenarios |
| `mocks/renderers/json_fixture.py` | `MockConfig` → standalone JSON response/request bodies and per-schema canonical examples |
| `mocks/renderers/pact.py` | `MockConfig` → Pact v3 consumer-driven contract files |
| `mocks/renderers/postman.py` | `MockConfig` → Postman v2.1 importable collection with saved example responses |
| `mocks/renderers/openapi_examples.py` | Round-trip an OpenAPI spec, injecting `example:` blocks into schemas/responses/requests |
| `mocks/scenario_engine.py` | Compiles `ScenarioConfig.states` into renderer-agnostic `ScenarioStep` records (used by the WireMock renderer for stateful stubs) |
| `mocks/llm_enricher.py` | Two-pass LLM enrichment via `multi_provider`: fill missing schema examples + draft missing 4xx/5xx error envelopes |
| `mocks/postman_importer.py` | Postman v2.1 collection JSON → `MockConfig` (folders → tags, saved responses → templates, variable substitution) |
| `mocks/har_importer.py` | HAR (HTTP Archive) → `MockConfig`; clusters entries by templated path, filters volatile headers |
| `sdp/ui/streamlit_app.py` | Streamlit UI for the data-generation track (≤ 10k rows). Single page: pick config → settings → generate → preview → download. |
| `sdp/mcp_server/server.py` | MCP server exposing `generate_data`, `lint_config`, `validate_data`, `infer_relationships`, `mock_init`, `mock_render`, `mock_enrich`, `list_examples`, `llm_diagnose` as tools so Claude Desktop / Claude Code / **LM Studio** / Cursor can drive the platform. LLM-using tools accept `llm_provider`/`llm_model`/`llm_base_url` for full provider portability. |
| `validators/gx_validator.py` | Great Expectations 1.x adapter — auto-derives an expectation suite from each table's `ColumnConfig` (PK→unique+not-null, business_values→in_set, min/max→between, REGEX:→matches_regex, EMAIL/UUID/IBAN/IPV4 etc.→regex with canonical shape, length→value_lengths_to_be_between, num_rows→row_count_between with tolerance). |
| `validators/quality_report.py` | Statistical quality reports — univariate stats + correlation matrix (always); KS test (numeric), TV-distance + chi-square (categorical), correlation-matrix delta, NN-distance privacy proxy, plus composed utility and bias sections (when source data is provided). JSON / HTML / Markdown outputs. Used by `quality-report` CLI, Streamlit UI, and MCP tool. |
| `synthesizers/base.py` | `Synthesizer` ABC (`fit`/`sample`), `EngineStats` cost accounting, and the shared multi-table sampling helper used by both the HMA engine and directly-assigned synthesizers. |
| `synthesizers/registry.py` | Lazy name → engine-class registry (`sdv`, `gaussian-copula`, `ctgan`, `tvae`, `rule-based`). Registration is by import path so unused engines cost nothing at import time. |
| `synthesizers/sdv_hma.py` | The default engine — SDV `HMASynthesizer`, the only one that models cross-table structure natively. |
| `synthesizers/single_table.py` | Per-table engines (Gaussian copula, CTGAN, TVAE). No cross-table modelling; referential integrity comes from FK resolution after sampling. |
| `synthesizers/dp_marginal.py` | Differentially private marginals — the only engine with a formal guarantee. Laplace-noised histograms over **config-declared** domains (`business_values`, `min_value`/`max_value`), so the domain never comes from the data. Budget composes within a table; the guarantee is per-row-per-table and says so. Columns with no declared domain are generated from config rules and spend nothing. |
| `synthesizers/rule_based.py` | Explicit no-model path — makes "generate from config rules" a named choice rather than only a failure mode. |
| `validators/utility.py` | TSTR utility — trains a model on synthetic data and scores it on held-out *real* rows, against the same model trained on real data. Answers "can this data still do the job?", which fidelity cannot: a generator can match every marginal while destroying the relationships between columns. ROC AUC ratios are chance-adjusted. |
| `validators/bias.py` | Bias drift — per-group representation shift and outcome disparity amplification (does the gap in outcome rates *between* groups widen in synthetic data?). Descriptive, not prescriptive. pandas only. |
| `contracts/checker.py` | Data contract testing — verifies data against the config-as-contract by reusing `gx_validator`, re-framing each GX result with a severity (error/warning) and an overall PASS/WARN/FAIL verdict. |
| `contracts/diff.py` | Breaking-change detection between two contract versions — pure structural diff classifying each change breaking / additive / review. |
| `contracts/model.py` | Dataclasses for the contract reports (`ContractTestReport`, `ContractDiff`) — JSON-serialisable, no GX dependency. |

### Generation Strategy (Hybrid)

1. **SDV path**: Builds metadata → generates 100-row internal sample → fits `HMASynthesizer` → samples full data. Two-tier retry with increasingly aggressive sanitization.
2. **Fallback path** (always available when SDV fails): regex patterns (`REGEX:\d{4}`), business value enumerations (semicolon-separated), Faker generators (`EMAIL`, `NAME`, `PHONE`), numeric/date ranges, NULL rates.

FK resolution runs after generation on both paths to ensure referential integrity.

### Configuration Formats

**Excel workbook** (primary): four optional/required sheets — `Columns` is required; `Run_Settings`, `Tables`, `Relationships` are optional. New optional `Tables` columns: `cdc_mode` (snapshot|delta|scd2) and `cdc_track` (semicolon list). New optional `Columns` columns: `rules` (JSON-encoded list of when/then rules) and `derived` (template / =-expression).

**YAML** (`config_format: sdp-yaml-v1`): code-friendly alternative; same semantics as Excel. See `Yaml_Config_Schema.md` for canonical format.

**JSON** (`config_format: sdp-json-v1`): same field set as YAML in JSON syntax. See `Json_Config_Schema.md`. JSON Schema for IDE validation lives at `schemas/sdp_config.schema.json`.

### Unified CDC block (replaces six legacy fields)

```yaml
cdc:
  mode: scd2          # snapshot | delta | scd2 (scd2 implies delta)
  track: [status]     # SCD2 tracked columns
  event_time: ts      # monotonic ordering column
  partition_by: []    # partition keys
```

Legacy flat fields (`delta_eligible`, `scd2_enabled`, `scd2_tracked_columns`, `partition_columns`, `event_time_column`, `generation_mode`) still parse — the parser back-fills both legacy and `cdc` views so downstream code keeps working.

### Anchored generation (`source:`)

A table may declare `source: <path>` (per-table — Excel `Tables` sheet column or
YAML `source:` field). That table is loaded verbatim from a real `.parquet`/`.csv`
dataset instead of being generated; the loaded rows live in
`DataGenerator.anchor_data`. Other tables generate around it —
`_inject_anchor_tables()` substitutes the real data before FK resolution so child
FKs reference the anchor's real keys, and the real rows feed `HMASynthesizer`
training so generated tables mimic its distributions. `_apply_relationship_group`
never rewrites an anchor table's own FK columns. The source dataset must contain
every configured column. See `Yaml_Config_Schema.md` → *Anchored generation*.

### Rules, derived columns and workflows (Layers A, B, C)

Per-column `rules:` applies when/then logic at row evaluation time; per-column `derived:` computes a value from other columns; top-level `workflows:` walks each row through a lifecycle state machine.

All three run after FK resolution, in the order **C → A → B** — a workflow assigns the lifecycle, rules react to the state it assigned, derived columns compute from the result. A Layer A rule targeting the state column therefore **overrides** the workflow.

| Layer | Engine | Scope |
|---|---|---|
| A — when/then rules | `utils/rule_evaluator.py` | per column |
| B — derived columns | `utils/rule_evaluator.py` | per column |
| C — workflows | `utils/workflow_engine.py` | per table, YAML/JSON only |

Layer C guarantees, per row: only declared transitions are followed; each visited state's timestamp column is set, strictly increasing along the path; unvisited states' columns are NULL. That makes a CANCELLED order with a delivery timestamp structurally impossible. Outgoing probabilities that sum below 1 leave the remainder as the chance of stopping in that state. Cycles are allowed but require an explicit `start_state` and are capped at 50 steps.

Both engines operate on plain dicts/DataFrames so the stub/mock track can reuse them. See `Rules_and_Workflows.md`.

### Data Types

`N38` (int/bigint), `VA1/VA3/VA256` (varchar), `DC` (decimal128), `D` (date), `DT` (datetime), `TS` (timestamp with tz), `A1/A5` (alpha), `NS` (numeric string), `AN` (alphanumeric).

### Special Rules (60+ rules in `utils/helpers.py`)

The `special_rules` column/field controls value generation strategy. Categories:
- **Faker-backed**: `NAME`, `FIRST_NAME`, `LAST_NAME`, `EMAIL`, `PHONE`, `ADDRESS`, `COMPANY`
- **Locale suffix**: `NAME:de_DE`, `EMAIL:fr_FR` — any of 18 supported locales
- **Global random locale**: `GLOBAL_NAME`, `GLOBAL_ADDRESS`, `GLOBAL_COMPANY`, `GLOBAL_PHONE`
- **Banking**: `IBAN`, `IBAN:de_DE`, `SWIFT`/`BIC`, `US_ROUTING`, `US_ACCOUNT`, `UK_SORTCODE`, `UK_ACCOUNT`, `AU_BSB`, `IN_IFSC`, `CLABE`, `CA_TRANSIT`
- **National IDs**: `SSN`, `US_EIN`, `UK_NI`, `IN_PAN`, `IN_AADHAAR`, `AU_TFN`, `AU_ABN`, `BR_CPF`, `BR_CNPJ`, `FR_SIREN`, `FR_SIRET`, `DE_TAX`, `ZA_ID`, `SG_NRIC`
- **Government docs**: `PASSPORT`, `PASSPORT:US/DE/FR/...`, `DRIVERS_LICENCE`, `DRIVERS_LICENCE:US`
- **Tax/VAT**: `EU_VAT`, `EU_VAT:DE/FR/...`, `GSTIN`
- **Network/tech**: `IPV4`, `IPV6`, `MAC`, `UUID`, `URL`
- **Healthcare**: `NHS_NUMBER`, `AU_MEDICARE`, `DE_KRANKENVERSICHERUNG`, `US_NPI`
- **Crypto**: `BTC_ADDRESS`, `ETH_ADDRESS`, `SOL_ADDRESS`
- **Product codes**: `EAN13`, `EAN8`, `UPC_A`, `ISBN13`, `CN_USCC`

Full reference in `Usage.md` section 8 and `Regex_Rules.md`.

### Delta & SCD2

- **Delta**: compares previous/current snapshot parquets; writes Delta Lake format with I/U/D operation column. Partition strategies are configurable.
- **SCD2**: bootstraps previous snapshot into versioned history; merges current data with effective dating columns (`effective_from_ts`, `effective_to_ts`, `is_current`, `version_num`). Only columns listed in `scd2_tracked_columns` trigger version changes.

## Testing

Tests live in `tests/`:
- `test_config_and_parquet_flows.py` — integration tests for Excel/YAML parsing, generate → delta → scd2 workflows, relationship validation.
- `test_regex_rules.py` — unit tests for regex pattern generation and special rule parsing.
- `test_rules_and_cdc.py` — CDC block, when/then rules, derived columns, JSON loader, rule evaluator.
- `test_mimesis_provider.py` — Mimesis adapter dispatch + locale handling.
- `test_relationship_signals.py` — pure signal computers for ML inferrer.
- `test_relationship_feedback_and_classifier.py` — JSONL persistence, pattern memory, classifier cold-start/activation.
- `test_relationship_inferrer.py` — end-to-end ML inferrer (two/three-table chains, dedup, threshold, learning loop).
- `test_cli_relationship_inference.py` — `infer-relationships` + `record-feedback` CLI dispatch and round-trip.
- `test_multi_provider.py` — unified LLM provider abstraction (config resolution, OpenAI-compat HTTP shape, Anthropic SDK dispatch, auth headers, error handling).
- `test_mock_models.py` — `MockConfig` pydantic shape, validation, $ref resolution, lint reports.
- `test_openapi_importer.py` — OpenAPI 3.x → `MockConfig` (format hints, $refs, allOf, parameters, response headers) plus end-to-end import of all three example specs.
- `test_template_engine.py` — value generation precedence (example > enum > special_rule > format > pattern > type), determinism, $refs, end-to-end renders.
- `test_renderers.py` — WireMock + JSON fixture renderers (concrete vs any-mode paths, deterministic seeds, header injection, request matchers).
- `test_renderers_phase_e.py` — Pact / Postman / OpenAPI-examples renderers (contract shape, collection import, example preservation vs overwrite).
- `test_scenarios.py` — scenario engine compiler + WireMock scenario block emission (after-N chains, multi-transition, priority ordering).
- `test_llm_enricher.py` — `mock-enrich` two-pass behaviour with the LLM call mocked (fence handling, exception tolerance, status-code filtering).
- `test_reverse_importers.py` — Postman + HAR importers (path templating, variable substitution, header filtering, body type inference).
- `test_cli_mocks.py` + `test_cli_mocks_phase_e_h_i.py` — `mock-*` argparse plumbing and dispatch end-to-end.
- `test_ui_streamlit.py` — Streamlit AppTest smoke tests (no-exception load, widget presence, 10k cap).
- `test_mcp_server.py` — MCP tool registry + per-tool behaviour (no real MCP transport spun up).
- `test_sdk_and_api.py` — Python SDK facade (`generate`/`lint`/`run`, seed reproducibility) and REST API endpoints (`/healthz`, `/generate`, `/lint`); API tests skip without the `api` extra.
- `test_contracts.py` — data contract testing: `contract-test` verdict/severity and `contract-diff` breaking-change classification; checker tests skip without the `gx` extra.
- `test_pk_generation.py` — regression guard: primary-key columns must be non-null and unique through the SDV path and Parquet export.
- `test_dp_marginal.py` — the DP engine, targeting the properties that make the ε claim real rather than just the code path: domain comes from config not data, epsilons compose, noise is genuinely applied, undeclared columns never reflect the training data, reported accounting matches what was spent.
- `test_synthesizer_engines.py` — engine registry, shared sampling helper (including the `num_rows`→`scale` fallback), engine contract, `DataGenerator` wiring, and the back-compat guarantee that a directly-assigned `generator.synthesizer` still samples.
- `test_utility_and_bias.py` — TSTR utility and bias drift. Fixtures carry a *known* answer (a learnable signal that synthetic data either preserves or destroys; a group skew that is either faithful or amplified), so the tests check the metrics measure what they claim — including the case fidelity misses: identical marginals, zero utility.

- `test_cli_commands.py` — the CLI command modules, which were the lowest-coverage part of the package and where both critical review findings lived. Covers `pii-scan` (the `parse_config()` crash), `lint` including `--strict-schema`, `quality-report`, `validate-data`, `contract-diff`, `delta`/`scd2`, and a regression that a **failed** Delta write leaves the flat parquet in place.
- `test_schema_validation.py` — JSON Schema enforcement: the `workflows` block the schema previously did not cover, violation paths, warning-vs-strict levels, Excel skipping, and a parametrised check that **every shipped example config validates against the schema** so the examples cannot drift from it.
- `test_workflows.py` — Layer C: the model (`from`/`to` aliasing, start-state inference), validation, the walk (only legal transitions, probability distribution, residual-stop rule, cycle capping), and the invariants on generated data — unvisited states NULL, visited states populated, timestamps strictly increasing. Plus config parsing and an end-to-end run through `generate_dataset`.
- `test_service_layer.py` — the extracted units directly: `services/common.py`, `GenerationRequest`/`GenerationOutcome`, `generate_dataset` (end-to-end, seed reproducibility, failure paths, engine stats, DP report), the mixin composition (including a guard that no method is defined twice across mixins, which would make behaviour depend on MRO order), the `cli_commands` handler surface, and size ratchets on `cli.py` / `data_generator.py`.
- `test_error_handling.py` — degraded paths must still work *and* leave a trace: invalid rules, malformed `cdc:` blocks, unparseable `null_rate`, unknown Faker locales, bad distribution descriptors. Plus two codebase-wide guards — no bare `except:`, and a ratchet on broad handlers that swallow without any signal.
- `test_hardening_fixes.py` — regression guards for defects found in platform assessment: the missing `Path` import in `collibra_importer` (NameError on every write), the hardcoded destructive delta overwrite, engine flags unreachable from the SDK, deprecated packaging metadata, and an AST-based check that library modules never `print()`.

CI (`.github/workflows/ci.yml`) runs the full suite with all extras on every
push and pull request. Linting is configured via `[tool.ruff]` in
`pyproject.toml` — a deliberately narrow rule set (undefined names, unused
imports/variables, mutable defaults, bare excepts) kept at zero findings, so
it can be widened without a repo-wide restyle.

## Error-handling conventions

Two rules, enforced by `tests/test_error_handling.py`:

**1. Name the exceptions that can actually occur.** `except Exception` hides
genuine bugs — a typo raising `NameError` looked identical to a malformed
value. Narrowing is not free: when the distribution handler was first
narrowed it omitted `AttributeError`, which a non-dict descriptor raises, and
that would have turned a degraded path into a crash. The test that pins it
exists because of that near-miss.

**2. Every swallowed failure leaves a trace** — a log line, a note on the
report, or a returned error. The one exemption is per-cell code: the Arrow
coercion helpers in `generators/_arrow_export.py` run once per cell, so a
single bad column would emit a log line per row. Those handlers name their
exception types and stay silent, and the module docstring says why.

Severity follows who can act on it: config errors the user can fix
(`null_rate` that will not parse, an invalid `cdc:` block, an unknown Faker
locale) are `warning`; internal fallbacks that are normal operation (a
distribution candidate that does not fit) are `debug`.

A ratchet test caps the number of broad handlers that swallow without any
signal. Fixing one lowers the number; it must never rise.

### Where `except Exception` is correct — do not "fix" these

Reviews keep flagging the raw count of broad handlers. Three groups are
deliberate and should stay:

**`mcp_server/server.py` (13).** Every one returns `{"ok": False, "error":
...}`. That *is* the MCP tool contract — a tool that raises hands the agent
a transport error instead of a message it can act on. Narrowing them would
make the server worse.

**Third-party boundaries.** LLM inference, cloud upload and the Delta write
call SDKs that raise a wide and undocumented range of types. These log (and
the Delta one re-raises), so nothing is hidden; enumerating their exceptions
would be guesswork that silently stops catching things on the next SDK
upgrade.

**Per-cell coercion in `generators/_arrow_export.py` (5).** Documented in
that module: they run once per cell, so logging is not an option. Their
exception types *are* named — they are counted as broad only because a
`ValueError` tuple still reads as broad to a naive count.

What was narrowed instead: import guards to `ImportError`, config-file
reads to `(OSError, TypeError, AttributeError, ValueError, yaml.YAMLError)`,
and `load_config_context` calls to `(ValueError, OSError)`.

## Relationship Inference

Two interchangeable engines behind the same interface (`infer(tables, ...) -> InferenceResult`):

### ML / heuristic path (`ml/relationship_inferrer.py`) — default
- Free, deterministic, no network call. Computes four signals (type, name, value-subset, pk-likeness), gates on type compatibility, blends with pattern memory, and optionally overrides with a learned logistic-regression classifier once feedback accumulates.
- Triggered via `--method ml` on `generate --infer-relationships`, or via the standalone `python main.py infer-relationships` subcommand.
- Adaptive: SME edits to the reviewable YAML are folded back via `python main.py record-feedback`. After ~30 labelled examples (with both classes), the classifier activates.
- Full design notes in `ML_Relationship_Inference.md`.

### GenAI / LLM path (`llm/relationship_inferrer.py`)
- `RelationshipInferrer.infer(tables)` sends table schemas to an LLM and returns `RelationshipConfig` objects with `inferred_by_llm=True` and `llm_confidence` (0–1).
- Triggered via `--method llm` (or `both` to run ML first and only ask the LLM about candidates ML missed).
- Provider-agnostic via `llm/multi_provider.py`: works with hosted Anthropic / OpenAI / Groq / Together / OpenRouter / Azure OpenAI, **and** local LLMs through LM Studio or Ollama (no API key needed). Switch with `SDP_LLM_PROVIDER`.
- Results are merged into `generator.relationships` before SDV metadata is built, so inferred FKs affect both SDV and fallback generation.
- Failures are non-fatal — generation continues without inferred relationships.

See `LLM_Ecosystem.md` for the full picture of providers, local runtimes, model families, and orchestration frameworks. See `examples/llm_quickstart.py` for a runnable demo.

### Schema Enrichment (`llm/schema_enricher.py`)
- `SchemaEnricher.enrich(tables)` sends column contexts to Claude and gets back suggestions for `business_values`, `special_rules`, `data_type`, or `null_rate`.
- Only suggestions above `confidence_threshold` are applied.
- Outputs a merged YAML config file.
- Triggered via `python main.py enrich --config ... --output enriched.yaml`.

Both modules use a stable system prompt with `cache_control: ephemeral` for Anthropic prompt caching, minimising cost on repeated calls.

## Docs

- `Usage.md` — full user guide for all three CLI commands and config formats
- `SDK_and_API.md` — Python SDK facade (`SyntheticDataPlatform`) and REST API (FastAPI) reference
- `Packaging.md` — building the wheel, package layout, extras, console scripts
- `Yaml_Config_Schema.md` — canonical YAML schema reference
- `Json_Config_Schema.md` — JSON schema reference (sdp-json-v1)
- `Rules_and_Workflows.md` — Layer A (when/then rules), Layer B (derived columns), Layer C (planned workflows)
- `Stubs_Mocks_Plan.md` — planned WireMock / stubs / mocks track (uses same rule engine)
- `Regex_Rules.md` — regex pattern syntax supported in `special_rules`
- `ML_Relationship_Inference.md` — ML relationship inferrer: signals, feedback loop, classifier activation, CLI
- `LLM_Ecosystem.md` — reference map of providers, open-weight model families, local runtimes (Ollama, LM Studio, vLLM, llama.cpp), provider abstractions (LiteLLM, OpenRouter), and orchestration frameworks (LangChain, LlamaIndex, DSPy, Haystack, …)
- `Stubs_Mocks_Plan.md` — parallel-track plan for API stubs/mocks generation (`MockConfig` model, OpenAPI ingest, WireMock/Pact/Postman renderers)
- `Bruno_Workflow.md` — end-to-end recipe: OpenAPI spec → `mock-init` → `mock-render` → WireMock standalone → Bruno API client
- `Data_Validation.md` — Great Expectations integration: auto-derived suites from `ColumnConfig`, the `validate-data` subcommand, and how to extend
- `Quality_Reports.md` — Statistical fidelity / utility / bias / privacy reports: univariate, KS / TV-distance, correlation delta, TSTR utility, bias drift, NN privacy proxy, CLI + UI + MCP entry points
- `Differential_Privacy.md` — The `dp-marginal` engine: what ε means, why config-declared domains make honest DP possible here, composition, the per-table limit, where the noise bites, and the caveat that ε over config-generated input is vacuous
- `Synthesizer_Engines.md` — Pluggable generation engines: the `Synthesizer` contract, built-in engines and their relationship trade-offs, engine selection and options, cost reporting, and how to write your own
- `Data_Contract_Testing.md` — data contracts: what/why, `contract-test` (verdict + severity), `contract-diff` (breaking-change detection), CLI / SDK / API / Streamlit
- `Docker_Quickstart.md` — Run with or without Docker (additive); CLI / Streamlit / MCP run modes inside the image
- `MCP_Integration.md` — what MCP is, why it matters, how the server is implemented, and how to wire it into LM Studio / Claude Desktop / Claude Code / Cursor / Zed (with multi-provider LLM passthrough)
- `Examples_Walkthrough.md` — demo playbook: 11 example configs (YAML/JSON/XLSX) covering every feature, with copy-pasteable commands and a 10-minute stakeholder demo arc
- `UI_Quickstart.md` — install + run the Streamlit UI (`poetry install --extras ui`)
- `MCP_Integration.md` — what MCP is, how to wire the platform's MCP server into Claude Desktop / Claude Code / Cursor, and demo prompts
- `PRD_Roadmap.md` — product vision and future roadmap
- `Pending_Items.md` — active engineering backlog
