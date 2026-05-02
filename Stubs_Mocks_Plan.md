# Stubs & Mocks Extension — Plan

*One-stop-shop for synthetic data across every layer of the test stack.*

---

## The Big Idea

Today the platform generates **Parquet files** — data at rest, used by data pipelines and analytics tests.

The extension makes the same config, the same ML intelligence, and the same generation engine also produce:

- **WireMock stubs** — HTTP API responses your services call
- **JSON/XML fixtures** — payload files for unit and integration tests
- **SQL INSERT scripts** — seed data for relational databases
- **Pact/contract files** — consumer-driven contract test payloads
- **Postman collections** — ready-to-run API test suites

**One YAML config → every format your test environment needs.**

---

## Why This Makes Sense

The platform already knows:
- What data types each column holds
- What realistic values look like (business values, distributions, special rules)
- Which columns are PII (needs masking/replacement)
- How tables relate to each other (FK relationships)

All of that knowledge is equally useful whether you're writing it to a Parquet file or an HTTP JSON response. The **data generation core does not change** — only the **output serialiser** changes.

We already planted the seeds for this in the previous session:

```python
# models/config_models.py — already in the codebase
class ColumnConfig(BaseModel):
    distribution:  Optional[Dict[str, Any]] = None  # drives realistic values
    example_value: Optional[Any] = None             # canonical value for stub headers/examples

class TableConfig(BaseModel):
    output_format: Optional[str] = None             # "parquet" | "json" | "wiremock" | ...
```

These three fields were added specifically so stubs and mocks can be bolted on without touching the existing config format.

---

## Target State — What "One Stop Shop" Looks Like

```
YAML Config (same config you write today)
              │
              ▼
    ┌─────────────────────┐
    │   Data Generator     │   ← unchanged core, same ML features
    │   (generates rows)   │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────────────────────────────┐
    │                  Output Serialiser (new)                     │
    ├──────────────────────────────────────────────────────────────┤
    │  parquet  │  json  │  xml  │  sql  │  wiremock  │  pact     │
    └──────────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │            Output folder                         │
    │  output/run_01/                                  │
    │  ├── parquet/  df_cac_acg_entr.parquet           │  ← today
    │  ├── json/     df_cac_acg_entr.json              │  ← new
    │  ├── sql/      df_cac_acg_entr_seed.sql          │  ← new
    │  ├── wiremock/ GET_accounts_200.json             │  ← new
    │  └── pact/     consumer-provider.json            │  ← new
    └─────────────────────────────────────────────────┘
```

---

## Feature Mapping — What Carries Over

| Existing Feature | How It Extends to Stubs/Mocks |
|---|---|
| Business values (`values: EUR;USD;INR`) | Enum constraints in JSON Schema / OpenAPI examples |
| Special rules (`IBAN`, `EMAIL`, `NAME`) | Generates realistic values in API response bodies |
| PII Detector | Flags which response fields must never contain real data |
| Distribution Fitter | Realistic numeric values in response payloads (amounts, counts, scores) |
| Auto-Config (`infer-config`) | Can now also read an OpenAPI spec and infer stub config |
| FK Relationships | Parent-child IDs stay consistent across JSON responses (same as Parquet) |
| Null rates | `nullable: true` in JSON Schema; random null injection in responses |
| Delta / SCD2 | Webhook payloads — "what changed" events (INSERT/UPDATE/DELETE) |
| Cloud upload | Upload WireMock stubs to Azure/S3 alongside Parquet output |
| LLM enrichment | Claude can suggest API response structure from endpoint descriptions |

---

## The Plan — Six Phases

---

### Phase 1 — Output Abstraction Layer
*Foundation: decouple "generate data" from "write data"*

**What:** Introduce a clean `OutputSerializer` interface so the generator does not need to know what format it's writing. Today it hardcodes Parquet. After this phase, it hands rows to a serialiser and doesn't care what happens next.

**Steps:**
1. Create `serialisers/` package
2. Define `BaseSerializer` abstract class with one method: `write(table_name, dataframe, config, output_dir)`
3. Move existing Parquet logic into `serialisers/parquet_serializer.py`
4. Wire `DataGenerator.export_to_parquet()` to call the serialiser instead
5. Make serialiser selection driven by `output_format` on `TableConfig` or a new CLI flag `--output-format`

**Config change (backward compatible):**
```yaml
# Per-table (existing field, now active)
tables:
  - name: accounts
    output_format: json   # parquet | json | xml | sql | wiremock | pact

# Or global (new run_settings field)
run_settings:
  output_format: wiremock
```

**CLI change:**
```bash
python main.py generate --config config/accounts.yaml --output output/v1 --output-format json
```

**Effort:** Medium. Core refactor but no new features — existing tests must still pass.

---

### Phase 2 — JSON & XML Fixture Serialisers
*Most common need: flat test fixture files*

**What:** Take the generated DataFrame and write it as JSON (array of objects) or XML.

**Steps:**
1. `serialisers/json_serializer.py` — writes `[{col: val, ...}, ...]` per table
2. `serialisers/xml_serializer.py` — wraps each row in a configurable root/record element
3. Add config options for JSON structure: flat array vs. nested (for API-shaped payloads)
4. Handle PII-flagged columns: replace with `special_rules`-generated synthetic values (already done by the generator — no extra work needed)
5. Add `--output-format json` and `--output-format xml` to CLI

**Output examples:**

```json
// output/json/accounts.json
[
  { "ACCT_ID": "DE89370400440532013000", "ACCT_CCY": "EUR", "BOOKG_AMT_NMRC": 1420 },
  { "ACCT_ID": "GB29NWBK60161331926819", "ACCT_CCY": "USD", "BOOKG_AMT_NMRC": 890 }
]
```

```xml
<!-- output/xml/accounts.xml -->
<accounts>
  <account>
    <ACCT_ID>DE89370400440532013000</ACCT_ID>
    <ACCT_CCY>EUR</ACCT_CCY>
  </account>
</accounts>
```

**Effort:** Low. pandas has `df.to_json()` and `df.to_xml()` — thin wrappers only.

---

### Phase 3 — SQL Seed Script Serialiser
*For teams whose tests load data into a real database*

**What:** Generate `INSERT INTO` statements that can be run against any relational database to seed test data.

**Steps:**
1. `serialisers/sql_serializer.py`
2. Config: specify target dialect — `sql_dialect: postgres | mysql | mssql | oracle | sqlite`
3. Generate `CREATE TABLE` (from platform types → SQL types mapping table)
4. Generate `INSERT INTO` statements (batched, e.g. 500 rows per statement)
5. Honour FK relationships: parent tables first, child tables second (same topological sort as Parquet path)
6. Add `TRUNCATE` / `DELETE` preamble option for idempotent test runs

**Platform type → SQL type mapping:**
| Platform type | PostgreSQL | MySQL | SQL Server |
|---|---|---|---|
| `N` | `INTEGER` | `INT` | `INT` |
| `N19` | `BIGINT` | `BIGINT` | `BIGINT` |
| `DC` | `DECIMAL(18,2)` | `DECIMAL(18,2)` | `DECIMAL(18,2)` |
| `VA50` | `VARCHAR(50)` | `VARCHAR(50)` | `NVARCHAR(50)` |
| `DT` | `TIMESTAMP` | `DATETIME` | `DATETIME2` |
| `D` | `DATE` | `DATE` | `DATE` |

**Output example:**
```sql
-- output/sql/accounts_seed.sql
TRUNCATE TABLE accounts;
INSERT INTO accounts (ACCT_ID, ACCT_CCY, BOOKG_AMT_NMRC) VALUES
  ('DE89370400440532013000', 'EUR', 1420),
  ('GB29NWBK60161331926819', 'USD', 890);
```

**Effort:** Medium. SQL escaping and type mapping need care; dialect differences need a thin adapter per database.

---

### Phase 4 — WireMock Stub Serialiser
*The main stub format: HTTP API responses your services call during tests*

**What:** Generate WireMock-compatible JSON stub files. Each table in the config becomes a set of stub mappings — one per row (detail endpoint) and one collection response (list endpoint).

**Steps:**
1. `serialisers/wiremock_serializer.py`
2. Add new config section to YAML to describe the API shape:
   ```yaml
   tables:
     - name: accounts
       output_format: wiremock
       api:
         base_path: /api/v1/accounts
         id_column: ACCT_ID          # used to build /accounts/{id} URL
         method: GET
         status_code: 200
         response_wrapper: null      # or "data" to wrap in { "data": [...] }
   ```
3. Generate:
   - `GET /accounts` → array response (all rows)
   - `GET /accounts/{id}` → single-row response per unique `id_column` value
   - `POST /accounts` → request-body schema + 201 response
4. Use `example_value` field (already in `ColumnConfig`) to populate WireMock body patterns
5. Add request matching: path template, query params, headers
6. Output: standard WireMock `__files/` and `mappings/` directory structure

**Output example:**
```json
// output/wiremock/mappings/GET_accounts_DE89370400440532013000.json
{
  "request": {
    "method": "GET",
    "urlPathPattern": "/api/v1/accounts/DE89370400440532013000"
  },
  "response": {
    "status": 200,
    "headers": { "Content-Type": "application/json" },
    "jsonBody": {
      "ACCT_ID": "DE89370400440532013000",
      "ACCT_CCY": "EUR",
      "BOOKG_AMT_NMRC": 1420
    }
  }
}
```

**Effort:** Medium-high. The data generation is trivial (already done); the WireMock mapping structure needs careful templating. Also need to decide how many individual stubs to generate (10 rows? 100? configurable).

---

### Phase 5 — Contract & Postman Serialisers
*For teams doing consumer-driven contract testing or API-level test suites*

**What:**
- **Pact files** — JSON files describing the expected interactions between a consumer and provider. Used with Pact framework in Java, JS, .NET, Python.
- **Postman collections** — importable into Postman for manual or Newman-automated API testing.
- **OpenAPI examples** — inject `examples:` blocks into an existing OpenAPI spec file.

**Steps:**
1. `serialisers/pact_serializer.py` — generates Pact v2/v3 JSON interaction files
2. `serialisers/postman_serializer.py` — generates Postman Collection v2.1 JSON
3. `serialisers/openapi_enricher.py` — reads an existing `openapi.yaml`, injects `example:` values from generated rows
4. Add config: `consumer_name`, `provider_name` (for Pact), `collection_name` (for Postman)

**Output example (Pact):**
```json
{
  "consumer": { "name": "transaction-service" },
  "provider": { "name": "account-service" },
  "interactions": [{
    "description": "get account by ID",
    "request": { "method": "GET", "path": "/api/v1/accounts/DE89370400440532013000" },
    "response": {
      "status": 200,
      "body": { "ACCT_ID": "DE89370400440532013000", "ACCT_CCY": "EUR" }
    }
  }]
}
```

**Effort:** Medium. JSON structure is well-defined by Pact/Postman specs — mainly template work.

---

### Phase 6 — ML Extension for Stubs/Mocks
*Bring the intelligence layer into the stub generation world*

**What:** Extend the three ML features to work with API/stub inputs and outputs.

#### 6a — Auto-Config from OpenAPI Spec
**Existing:** `infer-config` reads CSV/Parquet/Excel → writes YAML
**Extension:** `infer-config --input openapi.yaml` reads an OpenAPI spec → writes YAML with API block pre-filled

Steps:
1. Detect `.yaml` / `.json` input that contains `openapi:` key
2. Parse `paths`, `components/schemas`
3. Map OpenAPI types to platform types
4. Detect `enum` values → `business_values`
5. Detect `format: iban`, `format: email` → `special_rules`
6. Pre-fill `api:` block with the path, method, status codes from the spec

#### 6b — PII Detection in API Responses
**Existing:** PII detector scans column names and values in data files
**Extension:** Also scan OpenAPI schema property names and example values for PII

Steps:
1. Extend `PIIDetector.scan_dataframe()` to accept a dict (API schema) as well as a DataFrame
2. Add `scan_openapi_schema(schema_dict)` method
3. Wire into the `infer-config` OpenAPI path

#### 6c — Delta as Webhook Events
**Existing:** Delta computes I/U/D rows between two Parquet snapshots
**Extension:** Emit those changes as webhook event payloads (JSON POST bodies)

Steps:
1. Add `--output-format webhook` to `delta` command
2. `serialisers/webhook_serializer.py` wraps each I/U/D row in an event envelope:
   ```json
   { "event_type": "ACCOUNT_UPDATED", "timestamp": "...", "payload": { ... } }
   ```
3. Configurable event type naming (from YAML) and envelope schema

#### 6d — LLM: Suggest API Shape from Endpoint Description
**Existing:** LLM enrichment (`enrich` command) suggests `business_values` and `special_rules`
**Extension:** New LLM prompt asks Claude to suggest the `api:` block given a plain-English endpoint description

```bash
python main.py enrich --config config/accounts.yaml \
  --api-description "Returns account transaction history for a given IBAN" \
  --output config/accounts_enriched.yaml
```

Claude suggests: `base_path`, `id_column`, `response_wrapper`, query params.

**Effort:** Medium. New prompt, new output fields, same LLM infrastructure.

---

## What Changes vs. What Stays the Same

| Component | Status | Notes |
|---|---|---|
| YAML config format | **Unchanged** | New optional `api:` block added; existing configs still work |
| `DataGenerator` core | **Unchanged** | Generates rows exactly as today |
| ML features (PII, dist, auto-config) | **Extended** | New input types (OpenAPI spec) added |
| `ColumnConfig` / `TableConfig` models | **Minor additions** | `api:` block on `TableConfig`; `output_format` already present |
| Parquet output | **Unchanged** | Still the default, still works as before |
| Delta / SCD2 | **Extended** | Optional webhook serialiser added |
| Cloud upload | **Extended** | Uploads all formats, not just Parquet |
| CLI | **Extended** | New `--output-format` flag; `generate` dispatches to serialiser |
| Tests | **Extended** | New serialiser tests; existing tests unchanged |

---

## Implementation Order & Priorities

| Priority | Phase | Effort | Value |
|---|---|---|---|
| 1 | Phase 1 — Output Abstraction | Medium | Enables everything else |
| 2 | Phase 2 — JSON / XML | Low | Immediate value for most teams |
| 3 | Phase 4 — WireMock | Medium-high | Highest demand for API testing |
| 4 | Phase 3 — SQL Seeds | Medium | Common need for DB-backed tests |
| 5 | Phase 6a — OpenAPI infer | Medium | Closes the auto-config loop |
| 6 | Phase 6c — Webhook events | Medium | Natural extension of delta |
| 7 | Phase 5 — Pact / Postman | Medium | Niche but high-value for contract testing |
| 8 | Phase 6b,d — LLM extensions | Medium | Polish on top of a working platform |

Do Phase 1 first — it is the prerequisite for everything. Once the abstraction layer exists, each serialiser is an independent parallel workstream.

---

## New Config Block (Draft)

This is what the YAML will look like once the `api:` block is added — backwards compatible, all new fields optional:

```yaml
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 100
  output_format: wiremock          # global default (optional)

tables:
  - name: accounts
    rows: 50
    output_format: wiremock        # per-table override (optional)
    api:
      base_path: /api/v1/accounts
      id_column: ACCT_ID
      method: GET
      status_code: 200
      response_wrapper: null       # or "data", "result", etc.
      consumer_name: payment-service
      provider_name: account-service
    columns:
      - name: ACCT_ID
        type: VA18
        pk: true
        special_rules: IBAN
        example_value: DE89370400440532013000   # already in ColumnConfig
      - name: ACCT_CCY
        type: VA3
        values: EUR;USD;INR
      - name: BOOKG_AMT_NMRC
        type: N19
        distribution:
          name: norm
          params: [1200.0, 450.0]
          data_min: 1.0
          data_max: 50000.0
```

---

## New CLI Commands (Draft)

```bash
# Generate WireMock stubs (same config, different output format)
python main.py generate --config config/accounts.yaml --output output/stubs --output-format wiremock

# Generate JSON fixtures
python main.py generate --config config/accounts.yaml --output output/fixtures --output-format json

# Generate SQL seed scripts (postgres dialect)
python main.py generate --config config/accounts.yaml --output output/sql --output-format sql --sql-dialect postgres

# Generate all formats at once
python main.py generate --config config/accounts.yaml --output output/all --output-format all

# Delta as webhook events
python main.py delta --config config/accounts.yaml \
  --previous output/snap_v1 --current output/snap_v2 \
  --output output/events --output-format webhook

# Infer config from OpenAPI spec (Phase 6a)
python main.py infer-config --input api/openapi.yaml --output config/accounts.yaml

# Enrich config with LLM-suggested API shape (Phase 6d)
python main.py enrich --config config/accounts.yaml \
  --api-description "Returns paginated transaction history for an IBAN" \
  --output config/accounts_enriched.yaml
```

---

## Summary

```
TODAY                              AFTER EXTENSION
─────────────────────────────      ─────────────────────────────────────────
One config → Parquet files         One config → Parquet + JSON + XML +
                                                SQL + WireMock + Pact +
                                                Postman + Webhook events

Test data layer only               Every layer of the test stack:
                                   • Data layer     (Parquet, SQL)
                                   • API layer      (WireMock, Pact)
                                   • Unit test      (JSON/XML fixtures)
                                   • Contract test  (Pact, Postman)
                                   • Event layer    (Webhook payloads)

Infer config from data files       Infer config from data files OR
                                   OpenAPI specs

PII detection on data              PII detection on data AND API schemas
```

The core platform stays exactly as it is. Every feature you built for Parquet generation comes along for free — the serialiser layer is purely additive.

---

*Plan written: 2026-05-03 | Next step: start Phase 1 (Output Abstraction Layer)*
