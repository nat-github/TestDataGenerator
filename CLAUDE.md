# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic Data Platform — generates realistic, relationship-aware Parquet data from Excel or YAML configs. Supports snapshot generation, delta (CDC) processing, and SCD2 history. Uses SDV (Synthetic Data Vault) with automatic fallback to rule-based generation.

## Commands

```bash
# Install dependencies
poetry install

# Generate a snapshot
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --default-records 1000

# Generate from a JSON config
python main.py generate --config config/sample_workflow.json --output output/sample_run --seed 42

# Generate using rules + derived columns + cdc:
python main.py generate --config config/sample_workflow.yaml --output output/sample_run --seed 42

# Reproducible run (--seed makes output deterministic)
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --seed 42

# LLM-assisted: infer missing FK relationships using Claude (requires ANTHROPIC_API_KEY)
python main.py generate --config config/bare.xlsx --output output/run_01 --infer-relationships --llm-confidence 0.7

# Validate config with full sheet/row/column error context (no data generated)
python main.py lint --config config/Acct_bkng.xlsx

# Enrich a bare config with Claude-suggested business_values and special_rules
python main.py enrich --config config/bare.xlsx --output config/enriched.yaml --confidence 0.7

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

# Run all tests
poetry run pytest tests/ -v

# Run a single test
poetry run pytest tests/test_config_and_parquet_flows.py::test_name -v
```

### Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required for `--infer-relationships` and `enrich` commands |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure upload (preferred) |
| `AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_KEY` | Azure upload (alternative) |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | AWS S3 upload |
| `COLLIBRA_BASE_URL` | Collibra import base URL |
| `COLLIBRA_USERNAME` + `COLLIBRA_PASSWORD` | Collibra import credentials |

### Optional dependencies (install as needed)

```bash
pip install azure-storage-blob   # --upload-to azure://...
pip install boto3                # --upload-to s3://...
pip install matplotlib           # --er-format png
```

## Architecture

### Entry Point

`main.py` — CLI with four subcommands (`generate`, `delta`, `scd2`, `collibra-import`). Dispatches to `DataGenerator`, `ParquetPostProcessor`, `ERDiagramGenerator`, cloud uploaders, and `CollibraImporter`.

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
| `generators/data_generator.py` | Main orchestrator — SDV training, data generation, FK resolution, Parquet export |
| `utils/config_parser.py` | Excel & YAML parsing, validation with row/column error context, `lint_config()` |
| `utils/helpers.py` | Regex generation, Faker integration (18 locales), type coercion, NULL rate logic, 60+ special rules |
| `utils/parquet_post_processor.py` | Delta (I/U/D) and SCD2 effective-dating logic with delta-log rollback safety |
| `utils/rule_evaluator.py` | Layer A (when/then) and Layer B (derived expressions) post-generation pass — operates on plain dicts so it can be reused by stub/mock renderers later |
| `utils/data_validator.py` | Post-generation FK relationship validation |
| `utils/er_diagram.py` | ER diagram generation: Mermaid (zero-dep), Graphviz DOT, PNG (matplotlib optional) |
| `utils/cloud_uploader.py` | Azure Blob Storage and AWS S3 upload — credentials from env vars only |
| `utils/collibra_importer.py` | Collibra REST API v2 importer — dataset → YAML config |
| `models/config_models.py` | Pydantic v2 models: `TableConfig`, `ColumnConfig`, `RelationshipConfig`, `DataType` enum |
| `llm/client.py` | Shared Anthropic client factory with prompt caching and `lru_cache` |
| `llm/relationship_inferrer.py` | LLM relationship inference — sends schemas to Claude, returns `RelationshipConfig` objects with confidence |
| `llm/schema_enricher.py` | LLM schema enrichment — suggests `business_values`, `special_rules`, `data_type` corrections |

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

### Rules and derived columns (Layer A + B)

Per-column `rules:` list applies when/then logic at row evaluation time; per-column `derived:` produces values computed from other columns. Both run after FK resolution. Engine lives at `utils/rule_evaluator.py` and operates on plain dicts so it can be reused by the future stub/mock track. See `Rules_and_Workflows.md`.

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

Tests live in `tests/`. Two test files:
- `test_config_and_parquet_flows.py` — integration tests for Excel/YAML parsing, generate → delta → scd2 workflows, relationship validation.
- `test_regex_rules.py` — unit tests for regex pattern generation and special rule parsing.

No linting is configured (pending item in `Pending_Items.md`).

## GenAI / LLM Features

The `llm/` package integrates Claude (Anthropic) for two capabilities:

### Relationship Inference (`llm/relationship_inferrer.py`)
- `RelationshipInferrer.infer(tables)` sends table schemas to Claude and returns `RelationshipConfig` objects with `inferred_by_llm=True` and `llm_confidence` (0–1).
- Triggered via `--infer-relationships` on `generate`, or directly from Python.
- Results are merged into `generator.relationships` before SDV metadata is built, so inferred FKs affect both SDV and fallback generation.
- Requires `ANTHROPIC_API_KEY`.  Failures are non-fatal — generation continues without inferred relationships.

### Schema Enrichment (`llm/schema_enricher.py`)
- `SchemaEnricher.enrich(tables)` sends column contexts to Claude and gets back suggestions for `business_values`, `special_rules`, `data_type`, or `null_rate`.
- Only suggestions above `confidence_threshold` are applied.
- Outputs a merged YAML config file.
- Triggered via `python main.py enrich --config ... --output enriched.yaml`.

Both modules use a stable system prompt with `cache_control: ephemeral` for Anthropic prompt caching, minimising cost on repeated calls.

## Docs

- `Usage.md` — full user guide for all three CLI commands and config formats
- `Yaml_Config_Schema.md` — canonical YAML schema reference
- `Json_Config_Schema.md` — JSON schema reference (sdp-json-v1)
- `Rules_and_Workflows.md` — Layer A (when/then rules), Layer B (derived columns), Layer C (planned workflows)
- `Stubs_Mocks_Plan.md` — planned WireMock / stubs / mocks track (uses same rule engine)
- `Regex_Rules.md` — regex pattern syntax supported in `special_rules`
- `PRD_Roadmap.md` — product vision and future roadmap
- `Pending_Items.md` — active engineering backlog
