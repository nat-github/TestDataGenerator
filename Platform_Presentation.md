---
marp: true
theme: default
paginate: true
title: Synthetic Data Platform — Walkthrough
description: Detailed presentation for leadership + dev/test teams
---

# Synthetic Data Platform

A walkthrough for **leadership** and **engineering / QA** teams.

> One config in. Realistic test data, API mocks, and validation reports out.
> Same engine drives a CLI, a Streamlit UI, and an MCP server for AI assistants.

<!-- Speaker note: Open with one line — "We're going to look at what we built, what it's worth, and how to use it." Then dive in. -->

---

# How this deck is laid out

| Section | For whom | Slides |
|---|---|---|
| 1. Executive summary | **Leadership** | the pitch, the matrix, the numbers |
| 2. Feature tour | Everyone | what each capability is + a demo line |
| 3. Architecture | **Dev** | how the parts fit; engine / tracks / LLM / MCP |
| 4. How to use | **Dev / QA** | CLI, UI, MCP, examples |
| 5. Quality & reliability | **Leadership + QA** | tests, validation, determinism |
| 6. Roadmap | **Leadership** | what's stable, what's next, what's deferred |
| 7. Live demo arc | Demo-day | 10-minute stakeholder script |
| 8. References | **Dev** | doc index, file map |

> Trim by deleting whole sections — each is self-contained.

---

# Section 1 — Executive Summary

> *For leadership.*

---

## The problem

Modern dev teams need realistic, relationship-aware test data **fast**, in **many formats**, against **many schemas**, without the legal, security, or compliance risk of using production data.

Off-the-shelf tools usually give you one of these:

- A single-table data generator (no FK awareness)
- An API mocking tool (no realistic data)
- A schema-validation tool (no generation)
- A privacy-scrubbing tool (no synthesis)

**No single product covers data + APIs + AI + validation under one roof.**

---

## The solution — in one sentence

> A platform that turns one config into realistic Parquet **data**, **API mocks** (WireMock / Pact / Postman / OpenAPI), and **validation reports**, with optional AI assist for relationship inference and schema enrichment — driven by CLI, browser UI, or any AI assistant.

---

## Key numbers (current state)

| | |
|---|---:|
| Tests passing | **391** (+ 1 self-skip) |
| Output formats | **6** (Parquet, JSON, WireMock, Pact, Postman, OpenAPI examples) |
| Validation tracks | **3** (FK integrity, GX schema conformance, statistical quality reports) |
| Special-rule library | **60+** (banking, IDs, network, healthcare, crypto, locale-aware) |
| Supported locales | **18** Faker locales + Mimesis adapter |
| LLM providers supported | **8** (Anthropic, OpenAI, LM Studio, Ollama, Azure, Groq, Together, OpenRouter) |
| MCP tools exposed | **10** + 1 resource template |
| Example configs shipped | **11** (YAML / JSON / XLSX) + 3 OpenAPI specs + 3 mock fixtures |
| Docs | **17+** focused markdown files |
| Run modes | **CLI**, **Streamlit UI**, **MCP server**, **Docker / docker-compose** |

---

## Capability matrix — Domain × Status × Surfaces

| Domain | Status | Surfaces |
|---|---|---|
| **Configuration** (Excel / YAML / JSON, validation, linting) | Production-ready | CLI, UI, MCP |
| **Data generation core** (SDV + rule-based fallback) | Production-ready | CLI, UI, MCP |
| **Special rules** (60+ rules, 18 locales, Mimesis) | Production-ready | All |
| **Distributions + business values** (statistical, enums) | Production-ready | All |
| **Rules + derived columns** (when/then + computed) | Production-ready | CLI, MCP |
| **CDC: snapshot / delta / SCD2** | Production-ready | CLI, MCP |
| **Cloud upload** (Azure / S3) | Production-ready | CLI |
| **Collibra import** | Production-ready | CLI |
| **PII scanning** | Production-ready | CLI |
| **ER diagrams** (Mermaid / DOT / PNG) | Production-ready | CLI |
| **LLM relationship inference** (8 providers incl. local) | Production-ready | CLI, MCP |
| **ML relationship inference** (heuristic + adaptive classifier) | Production-ready | CLI, MCP |
| **LLM schema enrichment** | Production-ready | CLI, MCP |
| **Multi-provider LLM** (Anthropic / LM Studio / …) | Production-ready | All |
| **API mocks: OpenAPI ingest** | Production-ready | CLI, MCP |
| **API mocks: Postman + HAR reverse import** | Production-ready | CLI, MCP |
| **API mocks: 5 renderers** (WireMock / JSON / Pact / Postman / OpenAPI examples) | Production-ready | CLI, MCP |
| **API mocks: stateful scenarios** | Production-ready | CLI, MCP |
| **API mocks: LLM-assisted authoring** | Production-ready | CLI, MCP |
| **Great Expectations validation** | Production-ready | CLI, MCP |
| **Synthetic-data quality reports** (univariate + fidelity vs source + privacy proxy) | **Production-ready** | **CLI, UI, MCP** |
| **Streamlit UI** (≤ 10k rows) | Production-ready | UI |
| **MCP server** (10 tools, 1 resource template) | Production-ready | All MCP-aware clients |
| **Docker / containerisation** (additive — Poetry path still works) | **Production-ready** | — |
| **CI / CD** | Intentionally deferred | — |
| **Performance benchmarks** | Missing | — |
| **Type checking / linting** | Missing | — |

---

## Three audiences, one engine

```
                    ┌───────────────────────────┐
                    │   CLI                      │  hands-on developer
                    │   python main.py …         │
                    │   25+ subcommands          │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
            ┌────────────┐  ┌──────────┐  ┌────────────┐
            │ Streamlit  │  │  MCP     │  │ Direct     │
            │ UI         │  │  server  │  │ Python API │
            │ (SMEs)     │  │ (agents) │  │ (devs)     │
            └─────┬──────┘  └────┬─────┘  └─────┬──────┘
                  │              │              │
                  └──────────────┼──────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │  Engine (shared by all):   │
                    │  generators/, mocks/, ml/, │
                    │  llm/, utils/, models/,    │
                    │  validators/               │
                    └────────────────────────────┘
```

**Architectural payoff:** every feature reachable from every surface. Add once, available everywhere.

---

## Who benefits, how

| Audience | Pain we remove |
|---|---|
| Backend engineers | "I need 100K rows of plausible data with FK integrity for integration tests, ten minutes ago." |
| QA / SDETs | "I need a WireMock stub from this OpenAPI spec, plus a 429 scenario after the third call." |
| Data engineers | "I need delta + SCD2 outputs to test our pipeline's CDC path against a known oracle." |
| SMEs / non-technical reviewers | "I want to click *Generate* and see what comes out — not type CLI commands." |
| AI-augmented developers | "I want my IDE assistant to drive the platform conversationally." |
| Compliance / security | "Show me where the PII is, and prove the data we share is synthetic, not real." |

---

## ROI angles for leadership

| Cost we avoid | How |
|---|---|
| Production-data exposure incidents | Fully synthetic; PII detector flags risk; locale-aware fakers replace real values |
| "Borrowed" prod snapshots in dev / staging | Generate exactly what's needed, on demand, deterministic with `--seed` |
| Bespoke per-team stub authoring | OpenAPI spec → WireMock + Pact + Postman in one command |
| API-key spend on hosted LLMs | Local LM Studio / Ollama path needs zero API keys |
| Schema-onboarding tax | LLM enrichment + ML relationship inference cuts a multi-day SME meeting to a 5-minute review |
| Hand-written test fixtures rotting | Single source-of-truth config drives every output format |

---

# Section 2 — Feature tour

> *Mixed audience. Each card is one capability + one demo line.*

---

## Data generation — the core

**What:** Excel / YAML / JSON config in. Realistic Parquet out. SDV (Synthetic Data Vault) trains on a small sample, then samples N rows; rule-based fallback fires automatically when SDV can't fit.

**Demo line:**
```bash
python main.py generate --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/ecom --default-records 500 --seed 7 --validate
```

**Worth highlighting:**
- `--seed` makes it deterministic — same config + same seed = byte-identical output
- `--validate` runs the FK integrity check after generation
- Topological order is automatic — parents generated before children

---

## Special rules — 60+ realistic generators

| Category | Examples |
|---|---|
| Personal (locale-aware) | `NAME`, `ADDRESS`, `PHONE`, `EMAIL`, `COMPANY`, plus `:de_DE` / `:ja_JP` / `:fr_FR` etc. |
| Banking | `IBAN`, `SWIFT`, `US_ROUTING`, `UK_SORTCODE`, `IN_IFSC`, `CLABE`, `AU_BSB` |
| National IDs | `SSN`, `IN_PAN`, `IN_AADHAAR`, `BR_CPF`, `FR_SIREN`, `DE_TAX`, `ZA_ID`, `SG_NRIC` |
| Tax / VAT | `EU_VAT`, `EU_VAT:DE`, `GSTIN` |
| Network / tech | `IPV4`, `IPV6`, `MAC`, `UUID`, `URL` |
| Healthcare | `NHS_NUMBER`, `AU_MEDICARE`, `US_NPI` |
| Crypto | `BTC_ADDRESS`, `ETH_ADDRESS`, `SOL_ADDRESS` |
| Product codes | `EAN13`, `EAN8`, `UPC_A`, `ISBN13`, `CN_USCC` |
| Custom regex | `REGEX:[A-Z]{2}-[0-9]{6}` |
| Mimesis (optional extra) | `MIMESIS_*` rules via the optional `mimesis` package |

> Locale-aware = realistic to the right country, not just "Latin-script noise."

---

## Rules + derived columns

**What:** Two layers of business logic applied after FK resolution:
- **Layer A — when/then rules:** conditional value/null assignment per row
- **Layer B — derived columns:** values computed from other columns

```yaml
- name: closure_date
  data_type: D
  nullable: true
  rules:
    - when: { status: { eq: ACTIVE } }
      then: { set_null: true }
    - when: { status: { eq: CLOSED } }
      then: { value: 2024-06-15 }

- name: full_name
  data_type: VA64
  derived: "{first_name} {last_name}"
```

**Why it matters:** business invariants ("closed accounts have closure dates") are declarative, not buried in test fixtures.

---

## CDC: snapshot / delta / SCD2

| Mode | Output | Use case |
|---|---|---|
| **snapshot** | Plain Parquet at a point in time | Default. The test data you want today. |
| **delta** | Delta Lake table with `_operation: I/U/D` | Pipeline tests that consume change feeds |
| **scd2** | Versioned history with `effective_from_ts`, `effective_to_ts`, `is_current`, `version_num` | Dimension tables, audit history |

```bash
# Generate v1 → generate v2 → compute delta
python main.py delta --config X --previous output/v1 --current output/v2 --output output/delta
python main.py scd2  --config X --previous output/v1 --current output/v2 --output output/scd2
```

> Real Delta Lake output — Spark, Polars, deltalake-py, Trino all read it natively.

---

## Relationship inference — two engines, same surface

**Problem:** SME hands you a schema with no `relationships:` block. You don't want to ask them to write one.

**Two interchangeable engines** behind the same `infer(...)` call:

| Engine | Cost | Pattern |
|---|---|---|
| **ML / heuristic** (default) | Free, deterministic | 4 signals (type, name, value-subset, pk-likeness) + pattern memory + adaptive classifier |
| **LLM** | API call (or local LM Studio) | Schemas → Claude/GPT/local model → annotated FK suggestions |
| **Both** | Hybrid | ML first, LLM only on candidates ML missed |

```bash
python main.py infer-relationships --config bare.yaml \
  --config-output suggested.yaml --er-output diagram.mmd \
  --method both
```

**Adaptive:** SME edits the suggested YAML, runs `record-feedback`, the ML classifier learns from each accept/reject. After ~30 examples it activates.

---

## Multi-provider LLM — 8 backends, one abstraction

```python
# llm/multi_provider.py — chat() routes to whichever backend is configured
from llm.multi_provider import chat

text = chat(
    messages=[...],
    provider="lm-studio",           # or "anthropic" / "openai" / "ollama" / ...
    model="meta-llama-3.1-8b-instruct",
    base_url="http://localhost:1234/v1",
)
```

| Provider | Where it runs |
|---|---|
| `anthropic` (default) | Hosted Claude |
| `openai` | Hosted GPT |
| **`lm-studio`** | **Local — no API key** |
| **`ollama`** | **Local — no API key** |
| `azure-openai` | Enterprise |
| `groq` / `together` / `openrouter` | Cost-optimised hosted alternatives |

Switch by setting `SDP_LLM_PROVIDER=lm-studio`. Every LLM-using feature works against every backend.

---

## API mocks track — parallel sibling to data generation

The mocks track has its own config model (`MockConfig` / `sdp-mock-v1`) rooted in HTTP semantics, not relational. Independent of the data track but shares the engine layer (helpers, rule evaluator, LLM provider).

```
Inputs:                          Outputs:
  OpenAPI 3.x spec      →        WireMock mappings
  Postman collection    →        Pact v3 contracts
  HAR capture           →        Postman v2.1 collection
  Hand-authored YAML    →        OpenAPI spec with example: blocks
                                 JSON / XML fixtures
```

Stateful **scenarios**: `after 3 calls to list_accounts, switch to 429`. Pure config.

LLM-assisted authoring: `mock-enrich` fills missing examples and drafts realistic 4xx/5xx error envelopes — using whatever LLM you configured.

---

## API mocks — pipeline in 4 commands

```bash
# 1. OpenAPI / Postman / HAR → editable mock config
python main.py mock-init --from spec.yaml --output mocks/x.yaml

# 2. Validate
python main.py mock-lint --config mocks/x.yaml

# 3. Render every format (or pick a subset)
python main.py mock-render --config mocks/x.yaml --output stubs/x \
  --format wiremock,json,pact,postman --examples 5 --seed 42 --match-mode any

# 4. Optional: LLM-fill missing examples + draft errors
python main.py mock-enrich --config mocks/x.yaml --output mocks/x_enriched.yaml
```

Then run WireMock standalone against `stubs/x/wiremock/` and point Bruno / Postman / your code at it.

---

## Validation track — three complementary tools

| Validator | Validates | Run with |
|---|---|---|
| **FK integrity** (`utils/data_validator.py`) | Every child FK references a real parent | `generate --validate` |
| **Great Expectations** (`validators/gx_validator.py`) | **Schema conformance**: PK uniqueness, ranges, regex, enums, type, row counts | `generate --validate-with-gx` or `validate-data` subcommand |
| **Quality reports** (`validators/quality_report.py`) | **Statistical fidelity**: univariate stats, KS test, TV-distance, correlation delta, NN-distance privacy proxy | `quality-report` subcommand or Streamlit UI button |

> Schema conformance ≠ statistical fidelity. GX answers *"does it satisfy
> the constraints?"*; quality reports answer *"is it shaped like real
> data?"*. Both worth running.

GX expectations are **auto-derived from `ColumnConfig`** — no hand-written suite required. Mappings:

| `ColumnConfig` field | Auto-expectation |
|---|---|
| `is_pk: true` | `unique` + `not_null` |
| `business_values: A;B;C` | `to_be_in_set` |
| `min_value` / `max_value` | `to_be_between` |
| `special_rules: EMAIL/IBAN/UUID/IPV4/...` | `match_regex` with canonical shape |
| `length: N` | `value_lengths_to_be_between(1, N)` |
| `TableConfig.num_rows` | `row_count_between` with tolerance |

---

## Validation reports look like this

```
=== Great Expectations validation: FAIL ===
Tables: 1, expectations: 14, passed: 10, failed: 4

  FAIL  users                           rows=      10  expectations=10/14
         X   values_to_be_unique [user_id]
             unexpected=2 (20.0%)
             samples: [2, 2]
         X   values_to_match_regex [email]
             unexpected=5 (50.0%)
             samples: ['bad', 'bad', ...]
         X   values_to_be_in_set [status]
             unexpected=1 (10.0%)
             samples: ['UNKNOWN']
         X   values_to_be_between [age]
             unexpected=1 (10.0%)
             samples: [17]
```

Failures show up to 5 offending samples per expectation — debug without going back to the generator.

`--report-json <path>` dumps the full report for downstream dashboards.

---

## Quality reports — fidelity, distribution, privacy

**What:** statistical comparison of synthetic data to the source distribution.

**Two modes:**

| Mode | What it gives | When to use |
|---|---|---|
| **Univariate-only** (no source) | Per-column stats, top-N values, correlation matrix | "Does this synthetic data look plausible on its own?" |
| **Fidelity vs source** | Adds KS test (numeric), TV-distance (categorical), correlation delta, NN privacy proxy | "Is this synthetic data faithful to the source distribution while not leaking individual rows?" |

**Outputs:** structured JSON, self-contained HTML, inline markdown.

```bash
# CLI
python main.py quality-report \
  --generated output/run_01 --source data/real_sample \
  --output-html quality.html --verbose
```

**Streamlit UI**: one click in the *Quality report* section. Optional source upload for fidelity comparison.

**MCP**: `quality_report(generated_dir, source_dir=None)` — agents can summarise fidelity in natural language.

---

## Quality report — sample output

```
============================================================
Synthetic Data Quality Report
============================================================
Overall fidelity score: 0.957 (1.0 = identical to source, 0.0 = disjoint)

  customers                       rows=     200  fidelity=0.962  privacy_too_close=0.0%
  orders                          rows=     500  fidelity=0.944  privacy_too_close=1.2%
  order_items                     rows=    1500  fidelity=0.965  privacy_too_close=0.0%
```

Reading guide:

| `fidelity_score` | What it means |
|---|---|
| **≥ 0.85** | Marginal distributions match closely. Safe drop-in replacement for source on most analyses. |
| **0.6 – 0.85** | Acceptable for functional / integration tests; not for downstream stats. |
| **< 0.6** | Significant divergence. Investigate. |

`privacy_too_close` flags synthetic rows uncomfortably near a source row — a simple membership-inference proxy.

---

## PII / privacy scanning

```bash
# Scan a directory of generated parquet for PII categories
python main.py pii-scan --input output/run_01 --verbose
```

The scanner uses **column-name heuristics + value-pattern matching**, classifying columns by category (PERSON, EMAIL, PHONE, ADDRESS, GOVERNMENT_ID, FINANCIAL, HEALTHCARE, NETWORK, etc.).

**Compliance angle:** for any dataset, you can answer *"where's the PII?"* in one command — useful for both proving the synthetic data has the same PII surface as production *and* proving downstream consumers know what they're handling.

---

## Cloud upload + Collibra import

| Capability | Command |
|---|---|
| Azure Blob Storage upload | `--upload-to azure://<container>/<prefix>` (uses `AZURE_STORAGE_CONNECTION_STRING` or account+key) |
| AWS S3 upload | `--upload-to s3://<bucket>/<prefix>` (uses `AWS_*` env vars) |
| Collibra dataset → YAML | `python main.py collibra-import --dataset "Account Booking" --output config/from_collibra.yaml` |

Lightweight, optional, environment-variable-driven (no secrets in configs).

---

## Streamlit UI — for non-CLI users

Single page, ≤ 10,000 rows per table cap (the right tool for that volume; CLI for larger):

1. Pick a config (bundled example dropdown OR upload XLSX/YAML/JSON)
2. Set records + seed
3. Click **Generate** or **Lint**
4. Tabbed table preview (first 100 rows each)
5. Download all output as a single ZIP

```bash
poetry install --extras ui
poetry run streamlit run ui/streamlit_app.py
```

Same generator the CLI uses — runs in-process, no subprocess. Tested with Streamlit's AppTest harness.

---

## MCP server — agents drive the platform

The platform exposes **10 tools + 1 resource template** over Model Context Protocol. Any MCP-aware client (LM Studio, Claude Desktop, Claude Code, Cursor, Zed, OpenAI Agents SDK) can call them.

| Tool | Purpose |
|---|---|
| `generate_data` | Run the synthesizer (10k row cap as safety) |
| `lint_config` | Validate without generating |
| `validate_data` | Great Expectations validation |
| `quality_report` | Statistical fidelity + privacy report |
| `infer_relationships` | ML or LLM relationship inference |
| `mock_init` | OpenAPI / Postman / HAR → sdp-mock-v1 |
| `mock_render` | Render WireMock / JSON / Pact / Postman / OpenAPI examples |
| `mock_enrich` | LLM enrichment of mock configs |
| `list_examples` | Discover bundled configs |
| `llm_diagnose` | Sanity-check the LLM connection |

LLM-using tools accept `llm_provider` / `llm_model` / `llm_base_url` — point at LM Studio / Ollama / hosted Anthropic per-call.

---

## Example: a full conversational session in LM Studio

User in LM Studio:

> *"List the synthetic-data examples this platform ships with."*

Model calls `list_examples` → 11 YAML / 11 JSON / 7 XLSX configs + 3 OpenAPI specs.

> *"Generate 500 rows from the e-commerce one into `output/from_lm_studio`."*

Model calls `generate_data(config_path, output_dir, default_records=500)` → done in 2 seconds.

> *"Now validate it with Great Expectations."*

Model calls `validate_data(config_path, output_dir)` → returns the structured pass/fail summary.

> *"Convert `examples/openapi/medium_tasks.yaml` into a sdp-mock-v1 config and render WireMock + Postman stubs."*

Model chains `mock_init` → `mock_render`. Tells you where artefacts landed.

**Zero CLI typed. No API key. The local model in LM Studio orchestrated the whole flow via MCP.**

---

# Section 3 — Architecture

> *For dev. Skim if you're leadership.*

---

## Engine + tracks — the parallel-track design

```
                     ┌───────────────────────────────┐
                     │   Engine layer (shared)       │
                     │   • utils/helpers.py — 60+    │
                     │     special rules + Faker     │
                     │   • utils/mimesis_provider.py │
                     │   • utils/rule_evaluator.py   │
                     │   • llm/multi_provider.py     │
                     │   • llm/client.py             │
                     │   • validators/gx_validator.py│
                     └────┬─────────────────┬────────┘
                          │                 │
              ┌───────────┘                 └────────────┐
              ▼                                          ▼
   ┌────────────────────┐                     ┌────────────────────┐
   │  Data track         │                     │  Mocks track        │
   │  TableConfig +      │                     │  MockConfig +       │
   │  RelationshipConfig │                     │  EndpointConfig +   │
   │  (sdp-yaml-v1)      │                     │  Req/Resp templates │
   ├────────────────────┤                     │  (sdp-mock-v1)      │
   │ Inputs:             │                     ├────────────────────┤
   │  Excel / YAML /     │                     │ Inputs:             │
   │  JSON / Collibra /  │                     │  OpenAPI / Postman /│
   │  sample data        │                     │  HAR / hand-YAML    │
   ├────────────────────┤                     ├────────────────────┤
   │ Outputs:            │                     │ Outputs:            │
   │  Parquet, Delta,    │                     │  WireMock, Pact,    │
   │  SCD2               │                     │  Postman, OpenAPI   │
   │                     │                     │  examples, JSON/XML │
   └────────────────────┘                     └────────────────────┘
```

The two tracks share the **engine** and have **independent config models, ingest paths, and renderers**. Adding a renderer to one track doesn't touch the other.

---

## Data generation pipeline (data track)

```
   sdp-yaml-v1 / sdp-json-v1 / Excel / Collibra
                        │
                        ▼
          utils/config_parser.ConfigParser
                        │
                        ▼
   TableConfig + ColumnConfig + RelationshipConfig (Pydantic v2)
                        │
                        ▼
          generators/data_generator.DataGenerator
                        ├── create_sdv_metadata()
                        ├── train_synthesizer()  (HMA on a 100-row sample)
                        ├── generate_data()      (SDV path or rule fallback)
                        ├── apply rules + derived columns
                        ├── apply FK resolution
                        └── export_to_parquet()
                        │
       ┌────────────────┼────────────────┬────────────────┐
       ▼                ▼                ▼                ▼
   FK validator   GX validator   ER diagram     Cloud upload
   (built-in)     (optional)     (optional)     (optional)
```

---

## Mocks generation pipeline (mocks track)

```
  OpenAPI 3.x / Postman v2.1 / HAR / hand-authored YAML
                        │
                        ▼
                mock-init detects format
                        │
   ┌────────────────────┼────────────────────┐
   ▼                    ▼                    ▼
openapi_importer  postman_importer    har_importer
   │                    │                    │
   └────────────────────┼────────────────────┘
                        ▼
          MockConfig + EndpointConfig + ResponseTemplate
          + SchemaConfig + FieldSpec + ScenarioConfig
                        │
                        ▼
              mocks/template_engine.py
              (resolves field values via utils/helpers.py
               + utils/mimesis_provider.py)
                        │
       ┌────────────────┼────────────────┬────────────────┬────────────────┐
       ▼                ▼                ▼                ▼                ▼
   wiremock         json_fixture       pact            postman      openapi_examples
   renderer         renderer           renderer        renderer     enricher
                        │
                  scenario_engine.py overlays stateful scenarios
```

---

## LLM provider abstraction

```
  any caller (relationship inferrer / schema enricher / mock enricher)
                          │
                          ▼
              llm/multi_provider.chat()
                          │
              resolve_config() → (provider, model, base_url, api_key)
                          │
      ┌───────────────────┼───────────────────────────────┐
      ▼                   ▼                               ▼
  anthropic-sdk      requests POST /v1/chat/completions   …
  (Anthropic)        (OpenAI / LM Studio / Ollama / Azure /
                      Groq / Together / OpenRouter)
```

**Single function**, dispatches to the right backend. Configuration order: explicit args → `SDP_LLM_*` env vars → built-in defaults.

---

## MCP architecture

```
  AI client (LM Studio / Claude Desktop / Cursor / Zed / etc.)
                          │
                          │ JSON-RPC (stdio or HTTP)
                          ▼
              mcp_server/server.py (FastMCP)
              9 tools + 1 resource template
                          │
        ┌─────────────────┼──────────────────┬──────────────────┐
        ▼                 ▼                  ▼                  ▼
    generators/        mocks/             ml/                llm/multi_provider
    utils/             validators/        relationship_      (8 backends, no
                       gx_validator       inferrer           direct LLM calls
                                                              from the server)
```

**The MCP server itself never makes LLM calls directly.** Tools that need an LLM (`mock_enrich`, `infer_relationships --method=llm`, `llm_diagnose`) route through `multi_provider.chat()` — so the same MCP server works against any provider per call.

---

## Test discipline

| Test file | Coverage |
|---|---|
| `test_config_and_parquet_flows.py` | Excel/YAML parsing, generate → delta → scd2 |
| `test_regex_rules.py` | Regex generation + special rules |
| `test_rules_and_cdc.py` | CDC block, when/then, derived, JSON loader |
| `test_mimesis_provider.py` | Mimesis dispatch + locales |
| `test_relationship_signals.py` | Pure signal computers |
| `test_relationship_feedback_and_classifier.py` | JSONL persistence, pattern memory, classifier activation |
| `test_relationship_inferrer.py` | End-to-end ML inferrer |
| `test_cli_relationship_inference.py` | infer-relationships + record-feedback |
| `test_multi_provider.py` | LLM provider abstraction |
| `test_mock_models.py` | MockConfig validation |
| `test_openapi_importer.py` | OpenAPI 3.x → MockConfig |
| `test_template_engine.py` | Value generation precedence |
| `test_renderers.py` + `test_renderers_phase_e.py` | All 5 mock renderers |
| `test_scenarios.py` | Scenario compiler + WireMock state blocks |
| `test_llm_enricher.py` | mock-enrich (LLM mocked) |
| `test_reverse_importers.py` | Postman + HAR ingest |
| `test_cli_mocks*.py` | mock-* CLI plumbing |
| `test_ui_streamlit.py` | Streamlit AppTest smoke tests |
| `test_mcp_server.py` | MCP tool registry + per-tool behaviour |
| `test_gx_validator.py` | Great Expectations adapter |

**Total: 369 passing tests, deterministic with `--seed`.**

---

# Section 4 — How to use

> *For dev / QA. Demo-able commands.*

---

## CLI surface (the full menu)

| Subcommand | What it does |
|---|---|
| `generate` | Generate Parquet from a config |
| `delta` | Compute delta between two snapshots |
| `scd2` | Build SCD2 history |
| `lint` | Validate config (sheet/row/column error context) |
| `enrich` | LLM-enrich a bare config |
| `infer-config` | Infer config from sample data files |
| `infer-relationships` | ML / LLM relationship inference |
| `record-feedback` | Feed SME edits into the adaptive classifier |
| `pii-scan` | Scan generated parquet for PII columns |
| `collibra-import` | Import dataset definition from Collibra |
| `mock-init` | OpenAPI / Postman / HAR → sdp-mock-v1 |
| `mock-render` | Render mocks (WireMock / JSON / Pact / Postman / OpenAPI examples) |
| `mock-lint` | Validate a sdp-mock-v1 config |
| `mock-enrich` | LLM enrichment of a mock config |
| `validate-data` | Great Expectations validation |

---

## The 11 example configs (all formats)

Located under `examples/configs/`. Each has YAML + JSON, plus XLSX where the format supports the feature.

| # | Example | Demonstrates |
|---|---|---|
| 01 | Simple users | Faker, regex, business_values |
| 02 | E-commerce | 3 tables + FK + composite keys |
| 03 | Special-rules showcase | banking / national-ID / network / healthcare / crypto |
| 04 | Distributions + business values | normal + uniform numeric distributions |
| 05 | Multi-locale | five locales side by side |
| 06 | Rules + derived | when/then + computed columns |
| 07 | CDC delta workflow | snapshot → snapshot → delta |
| 08 | SCD2 history | versioned dimensions |
| 09 | PII columns | full PII surface for the scanner |
| 10 | Bare for LLM enrichment | sparse config for `enrich` |
| 11 | Bare for relationship inference | no relationships block — ML/LLM infer |

Plus 3 OpenAPI specs (simple_books / medium_tasks / complex_payments) and 3 mocks fixtures (scenarios / Postman / HAR).

`Examples_Walkthrough.md` covers each end-to-end with copy-paste commands.

---

## Streamlit UI walkthrough

```bash
poetry install --extras ui
poetry run streamlit run ui/streamlit_app.py
# Opens http://localhost:8501
```

Five sections, top to bottom:

1. **Pick config** — dropdown of bundled examples OR upload XLSX/YAML/JSON
2. **Settings** — records (≤ 10k cap), seed, output format
3. **Actions** — Generate / Lint config
4. **Preview** — tabbed DataFrames, first 100 rows each
5. **Download** — ZIP of every Parquet output

For larger runs / CDC / mocks / LLM features → drop back to the CLI.

---

## Run with Docker (or without)

Pure Poetry path stays untouched. Docker is **additive** — choose per box.

```bash
# 1. Build (one-time, ~5–10 min)
docker build -t sdp:latest .

# 2. CLI as one-shot
docker run --rm -v "$PWD:/work" sdp:latest \
  generate --config /work/examples/configs/yaml/01_simple_users.yaml \
           --output /work/output --default-records 200 --seed 42

# 3. Streamlit UI on :8501
docker run --rm -p 8501:8501 -v "$PWD:/work" sdp:latest streamlit

# 4. docker-compose alternative
docker compose up streamlit
```

The image bundles every optional extra (gx, ui, mcp, mimesis) so a single
`docker run` covers every feature. Non-root user, multi-stage build, ~1GB.

| Pick Docker when | Pick local Poetry when |
|---|---|
| You don't want Python on the host | You want fastest iteration |
| You're on a shared / CI box | You need IDE integration |
| You want a reproducible runtime | You're extending the platform itself |

Full guide: `Docker_Quickstart.md`.

---

## MCP setup — LM Studio in 4 steps

```json
// in LM Studio's mcp.json (Settings → Developer → MCP → "Reveal mcp.json"):
{
  "mcpServers": {
    "synthetic-data-platform": {
      "command": "C:/path/to/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/path/to/TestDataGeneration",
      "env": {
        "SDP_LLM_PROVIDER": "lm-studio",
        "SDP_LLM_MODEL": "meta-llama-3.1-8b-instruct",
        "SDP_LLM_BASE_URL": "http://localhost:1234/v1"
      }
    }
  }
}
```

1. Save `mcp.json`. 2. Restart LM Studio. 3. Verify in MCP panel — should see "synthetic-data-platform" with 9 tools. 4. Try `"Diagnose the LLM connection"` to confirm.

Same shape for Claude Desktop (`claude_desktop_config.json`), Claude Code (`~/.claude.json`), Cursor, Zed.

**The killer detail:** the `env` block points the platform's own LLM calls *back at LM Studio*. Closed loop, fully offline, no API keys anywhere.

---

## Documentation map (where each thing lives)

| Doc | Audience | What it covers |
|---|---|---|
| `CLAUDE.md` | Engineers | Engineering reference (modules, config, generation strategy, testing) |
| `Usage.md` | Engineers | Full user guide |
| `Yaml_Config_Schema.md` | Engineers | Canonical YAML schema |
| `Json_Config_Schema.md` | Engineers | JSON schema reference |
| `Examples_Walkthrough.md` | Everyone | Demo playbook for every feature |
| `Bruno_Workflow.md` | QA / dev | OpenAPI → mocks → WireMock → Bruno |
| `Stubs_Mocks_Plan.md` | Architects | Parallel-track design rationale |
| `Rules_and_Workflows.md` | Engineers | Layer A / B (when-then + derived) |
| `Regex_Rules.md` | Engineers | Regex syntax in `special_rules` |
| `ML_Relationship_Inference.md` | Engineers | ML inferrer design + adaptive feedback loop |
| `LLM_Ecosystem.md` | Architects | Provider/runtime/framework reference |
| `Data_Validation.md` | Engineers | Great Expectations integration |
| `Quality_Reports.md` | Engineers + analysts | Statistical fidelity / privacy reports |
| `Docker_Quickstart.md` | Engineers / Ops | Run with or without Docker |
| `MCP_Integration.md` | Engineers + leadership | What MCP is + multi-client setup |
| `UI_Quickstart.md` | SMEs | Run + use the Streamlit UI |
| `Claude_Architect_Certification.md` | Engineers prepping for CCA-F | Self-contained study guide |

---

# Section 5 — Quality & reliability

> *For QA + leadership.*

---

## Reliability assurances

| Property | How it's enforced |
|---|---|
| **Determinism** | `--seed N` makes any run byte-identical. Tests assert this. |
| **Referential integrity** | `--validate` confirms every child FK references a real parent. Topological generation prevents the issue at source. |
| **Schema conformance** | `--validate-with-gx` runs Great Expectations against every column constraint declared in the config. |
| **Type safety** | Pydantic v2 models reject invalid configs at parse time, with row/column error context. |
| **No silent skips** | Empty tables, missing schemas, parse errors all log loudly and fail the run unless explicitly tolerated. |
| **No PII leak** | All values are synthesized — Faker / Mimesis / regex / business_values. PII scanner double-checks the output surface. |
| **Idempotent CDC** | Delta + SCD2 outputs are reproducible from the same input snapshots. Delta-log rollback safety. |

---

## Test pyramid

```
                    ┌─────────────────────────────┐
                    │   End-to-end (smoke)        │  ~30
                    │   Streamlit AppTest         │
                    │   MCP tool registry         │
                    │   CLI dispatch round-trips  │
                    └─────────────────────────────┘
                ┌───────────────────────────────────┐
                │   Integration                     │  ~150
                │   generate → delta → scd2 flows   │
                │   OpenAPI → MockConfig → renderer │
                │   GX validator vs real DataFrames │
                └───────────────────────────────────┘
        ┌─────────────────────────────────────────────┐
        │   Unit                                       │  ~190
        │   Pure functions: signals, regex, rule      │
        │   evaluator, schema validation, importers,  │
        │   provider config resolution                 │
        └─────────────────────────────────────────────┘
```

**Total: 369 passing tests in ~120 seconds.**

---

## Honest weaknesses

| Gap | Why it matters | Mitigation today |
|---|---|---|
| **No CI/CD pipeline** | Regressions can land without being caught pre-merge | Tests are fast — culture is "always run them locally" |
| **No type-checking** (mypy / ruff) | Drift in type annotations not caught | Pydantic catches the worst at runtime |
| **No Docker image** | Onboarding requires Poetry + Python 3.13 | Documented; could change with one container |
| **No performance benchmarks** | Don't know p99 latency at scale | Most users in safe range; not a real-world bottleneck yet |
| **No synthetic-data quality reports** | Can't prove distribution fidelity vs source | GX catches conformance failures, but not fidelity |

> Ranked here in roughly the order they should be addressed.

---

# Section 6 — Roadmap

> *For leadership.*

---

## What's stable, ship-ready

Everything in the capability matrix marked **Production-ready**. That's:

- Data generation core, all 6 output formats
- Special rules + locales + Mimesis
- Rules engine + derived columns
- CDC: snapshot, delta, SCD2
- ML + LLM relationship inference (with adaptive feedback)
- LLM schema enrichment
- Multi-provider LLM (8 backends)
- Mocks track: ingest from 3 formats × render to 5 formats × stateful scenarios
- Great Expectations validation
- PII scanning, cloud upload, Collibra import
- All three surfaces: CLI, Streamlit UI, MCP server

> **Recommendation:** ship internally now; collect real-user friction reports; *then* prioritise.

---

## Tier 1 — quick wins (1–3 days each)

| Item | Status |
|---|---|
| **Docker image + `docker-compose.yml`** | **Done** — `Dockerfile` + `docker-compose.yml` + `Docker_Quickstart.md`. Both Poetry and Docker paths supported. |
| **`ruff` + `mypy --strict`** | Pending. Catches a class of bugs before tests run. Few hours each. |
| **CI/CD via GitHub Actions** | Intentionally deferred. |

---

## Tier 2 — medium investments (1–2 weeks each)

| Item | Why |
|---|---|
| ~~**Synthetic-data quality reports**~~ | **Shipped.** Univariate + correlation + KS / TV / NN privacy proxy. CLI + UI + MCP. |
| **GraphQL support** | OpenAPI track only covers REST. GraphQL is huge in modern stacks. |
| **AsyncAPI 2.x support** | Event-driven systems blind spot today. |
| **Performance benchmark suite** | Establish baselines; CI catches regressions. |
| **Pact `matchingRules`** | Real contract testing wants fuzzy matching. |
| **OpenAPI `oneOf` / `anyOf` with discriminator** | Real specs use polymorphism heavily. |

---

## Tier 3 — strategic directions (multi-week)

| Item | Why |
|---|---|
| **Test-data subsetting** | Given a real DB, extract a referentially-consistent subset. Different problem, adjacent value. |
| **Privacy / GDPR features** | Differential-privacy noise, k-anonymity, named-entity scrubbing. |
| **Web UI for SME review loops** | The ML inferrer's review flow is YAML-editing today; a small accept/reject UI widens the audience. |
| **Time-series / temporal patterns** | Generators are row-independent today; real data has trend/seasonality. |
| **RAG layer for docs Q&A** | Grounded "ask the platform anything" using the 15+ markdown docs as corpus. |

---

## Things to NOT do (yet)

These are common asks that are *not* the right next move:

- **Plugin system** — premature; current 60+ rules cover most needs
- **Helm chart** — only matters if a Docker image exists first
- **gRPC support** — niche; wait for a concrete user
- **OAuth flow simulation in mocks** — easier in WireMock proxies than in our config model
- **Real-time / streaming output** — adds complexity for a small slice of cases

> The pattern: don't add breadth until current depth is validated by real users.

---

# Section 7 — Live demo arc (10 minutes)

> *Scripted commands for a stakeholder demo.*

---

## Demo 1 — basic generation (45 s)

> *"One config, real data, one second."*

```bash
python main.py generate \
  --config examples/configs/yaml/01_simple_users.yaml \
  --output output/demo01 --default-records 200 --seed 42
poetry run python readParquet.py output/demo01
```

> Show: 200 rows, realistic names/emails/phones, deterministic.

---

## Demo 2 — relational integrity (1 min)

> *"With relationships, the data stays referentially honest."*

```bash
python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/demo02 --default-records 500 --seed 7 --validate
```

> Show: 3 tables, 2,200 rows, FK validator confirms every child references a real parent.

---

## Demo 3 — special rules + locales (1 min)

> *"60+ rules covering banking, IDs, network, healthcare."*

```bash
python main.py generate \
  --config examples/configs/yaml/03_special_rules_showcase.yaml \
  --output output/demo03 --default-records 50 --seed 1
```

> Open a row in pandas, scan the 25 columns: IBAN, SWIFT, SSN, IPv6, EAN13, BTC address — all format-valid.

---

## Demo 4 — relationship inference (2 min)

> *"The platform infers FKs for you, free or with an LLM."*

```bash
# Free, ML-only
python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/demo04_inferred.yaml \
  --er-output output/demo04.mmd \
  --method ml
```

> Show: ER diagram in `output/demo04.mmd`. SME edits the YAML, runs `record-feedback`, classifier learns.

---

## Demo 5 — rules + derived (1 min)

> *"Business invariants declarative, not buried in fixtures."*

```bash
python main.py generate \
  --config examples/configs/yaml/06_rules_and_derived.yaml \
  --output output/demo05 --default-records 200 --seed 21
```

> Show: ACTIVE accounts have null `closure_date`; CLOSED have a fixed value; `full_name` is `{first} {last}` everywhere.

---

## Demo 6 — CDC delta (1.5 min)

> *"Two snapshots, one delta, real Delta Lake."*

```bash
python main.py generate --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
    --output output/demo06/snap_v1 --default-records 1000 --seed 1
python main.py generate --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
    --output output/demo06/snap_v2 --default-records 1000 --seed 2
python main.py delta --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
    --previous output/demo06/snap_v1 --current output/demo06/snap_v2 \
    --output output/demo06/delta
```

> Show: `_operation` column with I/U/D rows; readable by Spark, Polars, deltalake-py, Trino.

---

## Demo 7 — Mocks: OpenAPI → Bruno (2 min)

> *"Same platform produces API mocks. Bruno hits a real WireMock pointed at our generated stubs."*

```bash
python main.py mock-init \
  --from examples/openapi/medium_tasks.yaml --output mocks/tasks.yaml

python main.py mock-render \
  --config mocks/tasks.yaml --output stubs/tasks \
  --format wiremock,postman --examples 5 --seed 42 --match-mode any

java -jar ~/tools/wiremock.jar --root-dir stubs/tasks/wiremock --port 8081
# Open Bruno → Import collection → mocks/tasks.yaml. Hit endpoints.
```

---

## Demo 8 — Stateful scenarios (1 min)

> *"Stateful: rate limit kicks in on the fourth call. Pure config."*

```bash
python main.py mock-render \
  --config examples/mocks/accounts_with_scenarios.yaml \
  --output stubs/accounts --format wiremock --match-mode any --seed 42
java -jar ~/tools/wiremock.jar --root-dir stubs/accounts/wiremock --port 8080

curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 429!
```

---

## Demo 9 — Validation (1 min, optional)

> *"Schema conformance + FK integrity, one flag."*

```bash
python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/demo09 --default-records 500 --seed 7 \
  --validate --validate-with-gx
```

> Show: Great Expectations report — passed/failed per (table, column).

---

## Demo 10 — Quality reports (1 min, optional headline)

> *"Schema conformance ≠ statistical fidelity. Both worth measuring."*

```bash
python main.py quality-report \
  --generated output/demo09 \
  --output-html output/demo09_quality.html
open output/demo09_quality.html
```

> Show: per-table fidelity score + correlation distance + privacy-too-close rate.
> If you have source data, pass `--source <dir>` for the full fidelity comparison.

> **Or in the Streamlit UI:** click "Generate quality report" in section 5.
> Optionally upload a Parquet/CSV source file for fidelity comparison.

---

## Demo 11 — MCP via LM Studio (2 min, optional headline)

> *"And — an AI assistant can drive the whole platform conversationally, using whatever LLM you have configured."*

In LM Studio chat (with the MCP server wired up):

> *"List the synthetic-data examples this platform ships with."*

> *"Generate 500 rows from the e-commerce one into `output/from_lm_studio`, then validate it with Great Expectations and tell me the result."*

> *"Now convert `examples/openapi/medium_tasks.yaml` into a sdp-mock-v1 config and render WireMock + Postman stubs from it."*

---

# Section 8 — References

> *For dev. The doc + code map.*

---

## File / module map

```
TestDataGeneration/
├── main.py                              # CLI entry (15+ subcommands)
├── ui/streamlit_app.py                  # Streamlit UI
├── mcp_server/server.py                 # MCP server (9 tools)
│
├── models/
│   ├── config_models.py                 # TableConfig / ColumnConfig / RelationshipConfig
│   └── mock_models.py                   # MockConfig / EndpointConfig / FieldSpec
│
├── generators/data_generator.py         # SDV + rule-based + FK resolution
│
├── utils/
│   ├── config_parser.py                 # Excel + YAML + JSON loader
│   ├── helpers.py                       # 60+ special rules + Faker
│   ├── mimesis_provider.py              # MIMESIS_* dispatch
│   ├── rule_evaluator.py                # Layer A/B engine
│   ├── parquet_post_processor.py        # delta + SCD2
│   ├── data_validator.py                # FK validator
│   ├── er_diagram.py                    # Mermaid / DOT / PNG
│   ├── cloud_uploader.py                # Azure / S3
│   └── collibra_importer.py             # Collibra → YAML
│
├── ml/
│   ├── relationship_signals.py          # 4 signal computers
│   ├── relationship_feedback_store.py   # JSONL pattern memory
│   ├── relationship_classifier.py       # Adaptive logistic regression
│   └── relationship_inferrer.py         # Orchestrator (free, deterministic)
│
├── llm/
│   ├── multi_provider.py                # 8-backend abstraction
│   ├── client.py                        # System prompt + cache_control
│   ├── relationship_inferrer.py         # LLM-based inferrer
│   └── schema_enricher.py               # LLM-based enricher
│
├── mocks/
│   ├── config_parser.py                 # sdp-mock-v1 loader
│   ├── openapi_importer.py              # OpenAPI 3.x → MockConfig
│   ├── postman_importer.py              # Postman v2.1 → MockConfig
│   ├── har_importer.py                  # HAR → MockConfig
│   ├── template_engine.py               # Field → JSON value
│   ├── scenario_engine.py               # Stateful state machine compiler
│   ├── llm_enricher.py                  # mock-enrich
│   └── renderers/
│       ├── wiremock.py                  # WireMock mappings
│       ├── json_fixture.py              # Standalone JSON
│       ├── pact.py                      # Pact v3
│       ├── postman.py                   # Postman v2.1 collection
│       └── openapi_examples.py          # OpenAPI spec round-trip
│
├── validators/
│   └── gx_validator.py                  # Great Expectations adapter
│
├── examples/
│   ├── configs/{yaml,json,xlsx}/        # 11 reference configs
│   ├── openapi/                         # 3 OpenAPI specs
│   ├── mocks/                           # 3 mocks fixtures
│   └── llm_quickstart.py                # multi-provider demo
│
└── tests/                               # 369 passing tests
```

---

## Closing

Three audiences. One engine. Six output formats. Eight LLM backends.
**Production-ready in 95% of the matrix; the gaps are productisation, not capability.**

> Built incrementally over 30+ commits, with tests on every change.
> Designed so adding a feature once exposes it on every surface.

**Next decision points** (for leadership):

1. **Ship internally** — collect real-user friction reports; prioritise based on what hurts.
2. **Tier 1 productisation** (CI/CD, Docker, type-checking) — 1 week, big credibility lift.
3. **Tier 2 capability** (quality reports, GraphQL, perf benchmarks) — pick the *one* that closes a real loop.

Questions?

---

*Generated as a unified deck for leadership + dev/test teams.
Trim by deleting whole sections — each is self-contained.
Convert to PPTX via Pandoc, or render with Marp / Slidev / reveal-md as-is.*
