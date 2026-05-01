# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FDL Synthetic Data Platform — generates realistic, relationship-aware Parquet data from Excel or YAML configs. Supports snapshot generation, delta (CDC) processing, and SCD2 history. Uses SDV (Synthetic Data Vault) with automatic fallback to rule-based generation.

## Commands

```bash
# Install dependencies
poetry install

# Generate a snapshot
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --default-records 1000

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

## Architecture

### Entry Point

`main.py` — CLI with three subcommands (`generate`, `delta`, `scd2`). Dispatches to `DataGenerator` and `ParquetPostProcessor`.

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
```

### Key Modules

| Module | Responsibility |
|---|---|
| `generators/data_generator.py` | Main orchestrator — SDV training, data generation, FK resolution, Parquet export |
| `utils/config_parser.py` | Excel & YAML parsing, validation with row/column error context, `lint_config()` |
| `utils/helpers.py` | Regex generation, Faker integration, type coercion, NULL rate logic, business value enumeration |
| `utils/parquet_post_processor.py` | Delta (I/U/D) and SCD2 effective-dating logic with delta-log rollback safety |
| `utils/data_validator.py` | Post-generation FK relationship validation |
| `models/config_models.py` | Pydantic v2 models: `TableConfig`, `ColumnConfig`, `RelationshipConfig`, `DataType` enum |
| `llm/client.py` | Shared Anthropic client factory with prompt caching and `lru_cache` |
| `llm/relationship_inferrer.py` | LLM relationship inference — sends schemas to Claude, returns `RelationshipConfig` objects with confidence |
| `llm/schema_enricher.py` | LLM schema enrichment — suggests `business_values`, `special_rules`, `data_type` corrections |

### Generation Strategy (Hybrid)

1. **SDV path**: Builds metadata → generates 100-row internal sample → fits `HMASynthesizer` → samples full data. Two-tier retry with increasingly aggressive sanitization.
2. **Fallback path** (always available when SDV fails): regex patterns (`REGEX:\d{4}`), business value enumerations (semicolon-separated), Faker generators (`EMAIL`, `NAME`, `PHONE`), numeric/date ranges, NULL rates.

FK resolution runs after generation on both paths to ensure referential integrity.

### Configuration Formats

**Excel workbook** (primary): four optional/required sheets — `Columns` is required; `Run_Settings`, `Tables`, `Relationships` are optional.

**YAML** (`config_format: fdl-yaml-v1`): code-friendly alternative; same semantics as Excel. See `Yaml_Config_Schema.md` for canonical format.

### Data Types

`N38` (int/bigint), `VA1/VA3/VA256` (varchar), `DC` (decimal128), `D` (date), `DT` (datetime), `TS` (timestamp with tz), `A1/A5` (alpha), `NS` (numeric string), `AN` (alphanumeric).

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
- `Regex_Rules.md` — regex pattern syntax supported in `special_rules`
- `PRD_Roadmap.md` — product vision and future roadmap
- `Pending_Items.md` — active engineering backlog (deterministic seeds, audit reports, Pydantic v2 migration, streaming)
