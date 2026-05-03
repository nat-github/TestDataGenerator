# Stubs & Mocks — Plan

*A parallel track for generating API stubs, mocks, contracts, and fixtures —
sharing the value-generation engine but rooted in HTTP/OpenAPI semantics, not
the relational table model.*

---

## 1. The Big Idea

Today the platform generates **Parquet files** from `TableConfig` +
`RelationshipConfig` (data at rest, used by data pipelines and analytics tests).

The stubs/mocks track is a **sibling capability** that generates:

- **WireMock stub mappings** — HTTP request/response pairs your services hit during tests
- **Pact contract files** — consumer-driven contract test interactions
- **Postman collections** — runnable API test suites
- **OpenAPI examples** — `examples:` blocks injected into existing specs
- **JSON / XML response fixtures** — payload bodies for unit and integration tests
- **HAR / cURL recipes** — capture-and-replay artefacts

It does **not** extend `TableConfig`. It has its own config model (`MockConfig`)
rooted in HTTP semantics: paths, operations, request matchers, response
templates, status codes, scenarios.

The two tracks share the **engine** (value generation, rule evaluation, LLM
provider) and nothing else.

---

## 2. Why Parallel, Not Extension

An earlier draft of this plan extended `TableConfig` with an `api:` block.
That works only when an API resource maps 1:1 to a database table — which is
rarely true for real APIs. Things that don't fit the table model:

- **Stateful endpoints** — idempotency keys, OAuth dances, paginated cursors
- **Multiple operations per path** — `GET / POST / PUT / DELETE /accounts/{id}`
- **Request shapes with no FK** — `POST /payments` body validation
- **Cross-cutting responses** — 4xx error envelopes, rate-limit headers, CORS preflight
- **Hypermedia / nested resources** — HAL/HATEOAS links, embedded sub-resources
- **Streaming, SSE, WebSocket, gRPC, GraphQL** — request/response duality breaks

Forcing all of this through `TableConfig` would be a leaky abstraction. A
separate model rooted in OpenAPI's vocabulary (paths × operations × schemas ×
responses) is the right shape.

---

## 3. Architecture — Shared Engine, Separate Models

```
                        ┌─────────────────────────────────┐
                        │       Shared engine layer        │
                        │  • utils/helpers.py (60+ rules) │
                        │  • utils/mimesis_provider.py    │
                        │  • utils/rule_evaluator.py      │
                        │  • llm/multi_provider.py        │
                        │  • llm/client.py (system prompt)│
                        └──────┬──────────────────┬───────┘
                               │                  │
            ┌──────────────────┘                  └──────────────────┐
            ▼                                                         ▼
  ┌─────────────────────┐                           ┌─────────────────────────┐
  │   Data track         │                           │   Stubs/Mocks track     │
  │   (today)            │                           │   (this plan)           │
  ├─────────────────────┤                           ├─────────────────────────┤
  │  TableConfig +       │                           │  MockConfig +           │
  │  RelationshipConfig  │                           │  EndpointConfig +       │
  │                      │                           │  Req/Resp templates     │
  ├─────────────────────┤                           ├─────────────────────────┤
  │  Inputs:             │                           │  Inputs:                │
  │  • Excel             │                           │  • OpenAPI / Swagger    │
  │  • YAML / JSON       │                           │  • Postman collection   │
  │  • Collibra          │                           │  • HAR capture          │
  │  • Sample files      │                           │  • Hand-authored YAML   │
  ├─────────────────────┤                           ├─────────────────────────┤
  │  Outputs:            │                           │  Outputs:               │
  │  • Parquet (default) │                           │  • WireMock mappings    │
  │  • Delta Lake        │                           │  • Pact contracts       │
  │  • SCD2              │                           │  • Postman collections  │
  │  • SQL seeds (1)     │                           │  • OpenAPI examples     │
  │  • JSON Lines (1)    │                           │  • JSON/XML fixtures    │
  └─────────────────────┘                           └─────────────────────────┘
        (1) JSON Lines / SQL serialisers for tabular data live on the
            data track; they reshape rows, they don't model HTTP.
```

The shared engine is provider-agnostic and format-blind. Both tracks call it
when they need a realistic IBAN, a regex-conformant invoice number, or a
locale-aware name. Neither track knows about the other's config model.

---

## 4. Primary Inputs

The stubs/mocks track accepts API-shaped inputs, not table configs:

| Input | Source | Use |
|---|---|---|
| **OpenAPI 3.0 / 3.1 spec** | `openapi.yaml`, `swagger.json` | Discover paths, operations, schemas, examples — convert to internal `MockConfig` |
| **Hand-authored mock YAML** (`sdp-mock-v1`) | YAML written by the user | Direct `MockConfig` definition for endpoints with no spec yet |
| **Postman collection** | Postman v2.1 export | Reverse-engineer endpoints + example bodies |
| **HAR file** | Browser/proxy capture | Replay-style stubs from real traffic |
| **OpenAPI 2.0 / Swagger** | Older specs | Same as 3.x via converter |

There is **no** ingestion path from `TableConfig` → `MockConfig`. The two
config models are independent. (A future helper could *suggest* a `MockConfig`
from a `TableConfig` for the narrow case where an API mirrors a table, but
that's a convenience tool, not the primary path.)

---

## 5. Internal Model — `MockConfig`

A first sketch of the pydantic model, parallel to `models/config_models.py`:

```python
# models/mock_models.py — net new

class MockConfig(BaseModel):
    """Top-level container for an API mock specification."""
    config_format: Literal["sdp-mock-v1"]
    info: MockInfo                          # title, version, description
    servers: List[ServerConfig]             # base URLs
    endpoints: List[EndpointConfig]
    schemas: Dict[str, SchemaConfig]        # reusable response/request shapes
    scenarios: List[ScenarioConfig] = []    # optional stateful sequences
    settings: MockSettings = ...            # latency, error rates, dialects


class EndpointConfig(BaseModel):
    """One HTTP operation on one path."""
    name: str                               # human-readable id
    path: str                               # e.g. /accounts/{id}
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    request: RequestMatcher
    responses: List[ResponseTemplate]       # 200 + 4xx variants
    examples: int = 5                       # how many concrete stubs to render
    priority: int = 5                       # WireMock priority for overlap resolution


class RequestMatcher(BaseModel):
    """How an incoming request is matched."""
    path_params: Dict[str, ValueSpec] = {}  # /{id} → spec
    query_params: Dict[str, ValueSpec] = {}
    headers: Dict[str, ValueSpec] = {}
    body_schema: Optional[SchemaConfig] = None
    body_match_mode: Literal["exact", "json-equal", "json-path", "regex"] = "json-equal"


class ResponseTemplate(BaseModel):
    """How a response is generated."""
    status: int                             # 200, 404, 500, …
    headers: Dict[str, ValueSpec] = {}
    body_schema: Optional[SchemaConfig] = None
    body_template: Optional[str] = None     # Jinja-style template for raw bodies
    weight: float = 1.0                     # used when an endpoint emits >1 response variant
    delay_ms: Optional[int] = None          # simulate latency


class SchemaConfig(BaseModel):
    """Re-usable JSON Schema fragment with field-level value specs."""
    type: Literal["object", "array", "string", "number", "integer", "boolean"]
    properties: Dict[str, "FieldSpec"] = {}
    items: Optional["FieldSpec"] = None     # for arrays
    required: List[str] = []


class FieldSpec(BaseModel):
    """How to generate a single field value at render time."""
    type: str                               # OpenAPI type or platform type (VA32, N10, …)
    format: Optional[str] = None            # email, iban, uuid, date-time, …
    special_rule: Optional[str] = None      # routes through utils/helpers.py
    business_values: Optional[List[Any]] = None
    pattern: Optional[str] = None           # regex
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    nullable: bool = False
    null_rate: float = 0.0
    example: Optional[Any] = None           # canonical example for OpenAPI examples block


class ScenarioConfig(BaseModel):
    """Stateful sequence: request N changes the response of request N+1."""
    name: str
    states: List[StateTransition]


class MockSettings(BaseModel):
    base_path: str = ""
    default_latency_ms: int = 0
    error_injection_rate: float = 0.0       # global 5xx injection probability
    cors_enabled: bool = True
```

**Key design points:**

- `FieldSpec.special_rule` routes to `utils/helpers.py` exactly the same way
  `ColumnConfig.special_rules` does today — that's how the engine is shared.
- `ScenarioConfig` makes statefulness a first-class concept (snapshot-only
  thinking would be wrong for mocks).
- `body_template` lets users skip schema definition for raw payloads (XML,
  Protobuf JSON, anything weird) — Jinja-style placeholders that resolve
  through the same value generators.
- `priority` and `weight` give us WireMock-compatible request matching and
  response variant selection out of the box.

---

## 6. Configuration — sample `sdp-mock-v1` YAML

```yaml
config_format: sdp-mock-v1

info:
  title: Account Service Mocks
  version: 1.2.0

servers:
  - url: http://localhost:8080/api/v1

settings:
  default_latency_ms: 25
  error_injection_rate: 0.02
  cors_enabled: true

schemas:
  Account:
    type: object
    required: [iban, currency, balance]
    properties:
      iban:      { type: string, special_rule: IBAN, example: DE89370400440532013000 }
      currency:  { type: string, business_values: [EUR, USD, INR, GBP] }
      balance:   { type: number, minimum: 0, maximum: 1_000_000 }
      opened_at: { type: string, format: date-time }
      holder:    { type: string, special_rule: NAME }

  Error:
    type: object
    properties:
      code:    { type: string, business_values: [NOT_FOUND, FORBIDDEN, RATE_LIMITED] }
      message: { type: string }

endpoints:
  - name: list_accounts
    path: /accounts
    method: GET
    request:
      query_params:
        page:  { type: integer, minimum: 1, example: 1 }
        size:  { type: integer, minimum: 1, maximum: 100, example: 25 }
    responses:
      - status: 200
        headers:
          Content-Type: { type: string, example: application/json }
        body_schema:
          type: array
          items: { $ref: Account }

  - name: get_account_by_id
    path: /accounts/{iban}
    method: GET
    request:
      path_params:
        iban: { type: string, special_rule: IBAN }
    responses:
      - status: 200
        weight: 0.9
        body_schema: { $ref: Account }
      - status: 404
        weight: 0.1
        body_schema: { $ref: Error }

  - name: create_account
    path: /accounts
    method: POST
    request:
      headers:
        Idempotency-Key: { type: string, format: uuid }
      body_schema: { $ref: Account }
    responses:
      - status: 201
        body_schema: { $ref: Account }
      - status: 422
        body_schema: { $ref: Error }
        weight: 0.05

scenarios:
  - name: rate_limit_after_3_calls
    states:
      - on_match: { endpoint: list_accounts }
        after: 3
        next_response: { status: 429, body: { code: RATE_LIMITED } }
```

This is **its own file format** (`sdp-mock-v1`), not a section grafted onto
`sdp-yaml-v1`. The two are independently versionable.

---

## 7. What Carries Over — explicit module-by-module

### Shared verbatim (zero changes)

| Module | Why it works for the mocks track |
|---|---|
| `utils/helpers.py` | A realistic IBAN / email / NAME is the same value whether it lands in a Parquet cell or a JSON response body |
| `utils/mimesis_provider.py` | Same — locale-aware value generators are format-agnostic |
| `utils/rule_evaluator.py` | when/then maps cleanly to "if request header X, then response field Y"; derived expressions resolve placeholders in body templates |
| `llm/multi_provider.py` | Provider abstraction — useful for "synthesise a 429 error envelope", "draft a missing example for this schema", "expand a one-line endpoint description into a full ResponseTemplate" |
| `llm/client.py` (system prompt + cache_control) | Same |

### Shared with adapter (small wrapper)

| Module | What's needed |
|---|---|
| `utils/config_parser.py` | Add a sibling `MockConfigParser` that loads `sdp-mock-v1` YAML / OpenAPI / Postman. Existing `ConfigParser` untouched |
| Validation (`models/`) | New `MockConfig` pydantic models in `models/mock_models.py`; existing `config_models.py` untouched |

### Does **not** carry over (relational-only, intentionally)

| Module | Why not |
|---|---|
| `ml/relationship_signals.py`, `ml/relationship_classifier.py`, `ml/relationship_feedback_store.py`, `ml/relationship_inferrer.py` | All exist to answer "which child column references which parent column?" — a question that has no analogue in the API world |
| `llm/relationship_inferrer.py` | Same — specific to FK inference between tables |
| `models/config_models.py:RelationshipConfig` | Endpoints don't have FK-style references between them |
| `generators/data_generator.py` (FK resolution pass, SDV synthesizer) | A `GET /accounts/{id}` mock has no FK semantics; SDV's relational synthesis doesn't apply to HTTP |
| `utils/parquet_post_processor.py` (delta/SCD2) | Versioning concerns belong to the data track. Webhook-style "what changed" events are a separate idea (see §11) |
| `utils/data_validator.py` | FK-aware validator — replaced by JSON Schema validation in the mocks track |

### Net new modules

| New module | Purpose |
|---|---|
| `models/mock_models.py` | `MockConfig`, `EndpointConfig`, `RequestMatcher`, `ResponseTemplate`, `SchemaConfig`, `FieldSpec`, `ScenarioConfig` |
| `mocks/config_parser.py` | Parse `sdp-mock-v1` YAML/JSON; validate against schema |
| `mocks/openapi_importer.py` | OpenAPI 3.x → `MockConfig` |
| `mocks/postman_importer.py` | Postman v2.1 collection → `MockConfig` |
| `mocks/har_importer.py` | HAR capture → `MockConfig` |
| `mocks/template_engine.py` | Jinja-style placeholder resolver that calls into `utils/helpers.py` |
| `mocks/scenario_engine.py` | Stateful scenario evaluation (request log + state transitions) |
| `mocks/renderers/wiremock.py` | `MockConfig` → WireMock `mappings/` + `__files/` directory tree |
| `mocks/renderers/pact.py` | `MockConfig` → Pact v3 JSON files |
| `mocks/renderers/postman.py` | `MockConfig` → Postman collection v2.1 |
| `mocks/renderers/openapi_examples.py` | Inject `examples:` blocks into an existing OpenAPI spec |
| `mocks/renderers/json_fixture.py` | Standalone JSON / JSON Lines / XML response bodies |

---

## 8. Phases — re-anchored to ingest-first

The original plan had output abstraction as Phase 1. Rebooted: ingest the
spec first, validate the model, then build renderers.

### Phase A — Internal model & validation
*The foundation: define the data structures and parser.*

- Create `models/mock_models.py`
- Hand-author a `sdp-mock-v1` YAML loader in `mocks/config_parser.py`
- Round-trip tests (YAML → MockConfig → YAML)
- JSON Schema for IDE validation: `schemas/sdp_mock.schema.json`

**Effort:** Medium. Models + parser + tests, no behaviour beyond round-trip.

### Phase B — OpenAPI ingest
*Make the most-used input format work end to end.*

- `mocks/openapi_importer.py` — walk `paths`, `components.schemas`, `examples`
- Map OpenAPI types → `FieldSpec` (string/integer/number/array/object)
- Detect `format: email|date-time|uuid|iban` → `special_rule`
- Detect `enum` → `business_values`
- Pull `example:` and `examples:` blocks straight through
- CLI: `python main.py mock-init --from openapi.yaml --output mocks.yaml`

**Effort:** Medium. OpenAPI parsing has gotchas ($refs, allOf/oneOf,
discriminators), but a 90% solution lands quickly.

### Phase C — Template engine + value generation
*Bridge `FieldSpec` to actual bytes.*

- `mocks/template_engine.py` — render a `SchemaConfig` to a JSON document by
  calling `utils/helpers.py` for each leaf
- Honour `special_rule`, `business_values`, `pattern`, `minimum/maximum`,
  `nullable`, `null_rate`
- Jinja-style `{{ placeholder }}` resolution in `body_template` strings
- Deterministic mode via `--seed`

**Effort:** Low-medium. Engine already exists; this is the wiring layer.

### Phase D — WireMock renderer
*The headline output format.*

- `mocks/renderers/wiremock.py`
- For each `EndpointConfig`, emit one WireMock mapping JSON per response
  variant, weighted by `weight`
- Generate N concrete examples per endpoint (configurable via `examples:`)
- Honour `priority`, `delay_ms`, header matchers, query param matchers
- Output: WireMock-standard `mappings/*.json` + `__files/` directory
- CLI: `python main.py mock-render --config mocks.yaml --output stubs/ --format wiremock`

**Effort:** Medium-high. WireMock has a rich matcher language — getting the
mapping JSON right is more careful translation than novel logic.

### Phase E — Pact, Postman, OpenAPI-examples renderers
*Cover the contract-test and exploratory-test workflows.*

- `mocks/renderers/pact.py` — Pact v3 interaction files
- `mocks/renderers/postman.py` — Collection v2.1
- `mocks/renderers/openapi_examples.py` — round-trip a spec, populating empty
  `example:` slots from generated values

**Effort:** Medium each. Format-spec work; deterministic translations.

### Phase F — JSON / XML fixture renderer
*Standalone payload files, no HTTP wrapper.*

- `mocks/renderers/json_fixture.py` — write `<endpoint>_<status>.json` per
  endpoint+status, plus `<schema>.json` for reusable schemas

**Effort:** Low. Subset of the WireMock renderer's body work.

### Phase G — Scenarios & stateful stubs
*Make sequencing a first-class feature.*

- `mocks/scenario_engine.py` — request-log + state transitions
- WireMock has a `scenarioName` + `requiredScenarioState` mechanism we can
  target; Pact expresses provider state in interactions
- CLI: `python main.py mock-test --config mocks.yaml --port 8080` (run a
  local WireMock-style server, optional)

**Effort:** Medium-high. The state model is small; the integrations to each
tool's scenario primitives are the work.

### Phase H — LLM-assisted authoring
*Bring the AI features over for the new track.*

- `python main.py mock-enrich --spec partial.yaml --output enriched.yaml` —
  fills missing examples, error responses, and schema descriptions via the
  multi-provider abstraction
- New prompt template targeted at `MockConfig` shape
- Optional: suggest a full `MockConfig` from a one-paragraph endpoint
  description (`mock-init --description "Returns paginated transactions for an IBAN"`)

**Effort:** Low (prompt engineering on existing infra).

### Phase I — Postman / HAR ingest
*Reverse-engineer existing artefacts into `MockConfig`.*

- `mocks/postman_importer.py`
- `mocks/har_importer.py`
- Useful for legacy systems where the only available source of truth is a
  shared Postman collection or a captured browser session

**Effort:** Medium each. Format parsing + heuristics for inferring schemas
from concrete examples.

---

## 9. CLI shape

Separate command namespace (`mock-*`) so it doesn't muddy the data track:

```bash
# Ingest an OpenAPI spec → sdp-mock-v1 YAML
python main.py mock-init --from api/openapi.yaml --output mocks/accounts.yaml

# Render WireMock stubs from a mock config
python main.py mock-render --config mocks/accounts.yaml \
    --output stubs/accounts/ --format wiremock --examples 20 --seed 42

# Render multiple formats in one go
python main.py mock-render --config mocks/accounts.yaml \
    --output stubs/accounts/ --format wiremock,pact,postman

# Validate a mock config against the schema
python main.py mock-lint --config mocks/accounts.yaml

# LLM-enrich a partial spec (fill missing examples / error responses)
python main.py mock-enrich --config mocks/accounts.yaml \
    --output mocks/accounts_enriched.yaml

# Reverse: ingest a Postman collection or HAR file
python main.py mock-init --from postman_collection.json --output mocks/x.yaml
python main.py mock-init --from session.har --output mocks/y.yaml

# Run a local mock server (optional, Phase G)
python main.py mock-serve --config mocks/accounts.yaml --port 8080
```

The existing `generate`, `delta`, `scd2`, `lint`, `enrich`, `infer-config`,
`pii-scan`, `infer-relationships`, `record-feedback`, `collibra-import` data
commands are **untouched**.

---

## 10. What carries vs. doesn't — at a glance

| Component | Stays as-is | Adapter | Net new | Not used |
|---|:---:|:---:|:---:|:---:|
| `utils/helpers.py` | ✓ | | | |
| `utils/mimesis_provider.py` | ✓ | | | |
| `utils/rule_evaluator.py` | ✓ | | | |
| `llm/multi_provider.py` | ✓ | | | |
| `llm/client.py` | ✓ | | | |
| `models/config_models.py` (TableConfig etc.) | ✓ | | | |
| `models/mock_models.py` | | | ✓ | |
| `utils/config_parser.py` (`ConfigParser`) | ✓ | | | |
| `mocks/config_parser.py` (`MockConfigParser`) | | | ✓ | |
| `mocks/openapi_importer.py` | | | ✓ | |
| `mocks/template_engine.py` | | | ✓ | |
| `mocks/scenario_engine.py` | | | ✓ | |
| `mocks/renderers/*` | | | ✓ | |
| `ml/relationship_*.py` | | | | ✓ |
| `llm/relationship_inferrer.py` | | | | ✓ |
| `llm/schema_enricher.py` | | ✓ (sibling `mock_enricher.py`) | | |
| `generators/data_generator.py` | | | | ✓ |
| `utils/parquet_post_processor.py` | | | | ✓ |
| `utils/data_validator.py` | | | | ✓ (replaced by JSON Schema validation) |

---

## 11. Adjacent question — webhook events from delta

The data track produces delta/SCD2 outputs that *could* be re-shaped as
webhook event payloads (`{ "event": "ACCOUNT_UPDATED", "payload": {...} }`).
This is a small standalone feature that genuinely belongs to the data track,
not the mocks track — the input is a Parquet snapshot, the output is JSON
events derived from row-level changes. It's listed here only because the old
plan rolled it in. Recommended placement: a separate `--event-format` flag on
`delta`, kept distinct from `MockConfig`.

---

## 12. Implementation order & priorities

| Priority | Phase | Effort | Why |
|---|---|---|---|
| 1 | A — Internal model & validation | Medium | Prerequisite for everything |
| 2 | B — OpenAPI ingest | Medium | The main input most users will have |
| 3 | C — Template engine + value generation | Low-medium | The bridge from spec to bytes |
| 4 | D — WireMock renderer | Medium-high | Highest-demand output format |
| 5 | F — JSON / XML fixture renderer | Low | Falls out for free once C+D exist |
| 6 | E — Pact / Postman / OpenAPI-examples | Medium each | Niche but high-value |
| 7 | H — LLM-assisted authoring | Low | Polish on top of working foundation |
| 8 | G — Scenarios & stateful stubs | Medium-high | Needed for realism, not for first release |
| 9 | I — Postman / HAR ingest | Medium each | Reverse-engineering aids; useful but not blocking |

**Suggested first cut:** A → B → C → D → F. That's enough to take an
OpenAPI spec, generate WireMock stubs and JSON fixtures, and run a real
integration test. Everything past D is additive.

---

## 13. Testing strategy

Same patterns as the data track:

- **Unit tests per module**: pure functions on `MockConfig` (parsing,
  validation, type mapping)
- **Round-trip tests**: OpenAPI → `MockConfig` → WireMock → re-import via
  WireMock's own client → confirm the request matchers fire
- **Renderer fidelity tests**: render to WireMock JSON, hand-check a small
  golden fixture per response variant
- **Determinism tests**: same `--seed` produces byte-identical output
- **Reuse the `multi_provider` test pattern**: mock the LLM call, assert on
  the prompt + the parsed response structure

---

## 14. Summary

```
DATA TRACK (today, 8/10)              MOCKS TRACK (this plan, 0/10 → 8/10)
─────────────────────────────────     ──────────────────────────────────────
TableConfig + RelationshipConfig      MockConfig + EndpointConfig
sdp-yaml-v1 / sdp-json-v1             sdp-mock-v1 (new)

Inputs:                               Inputs:
  Excel / YAML / JSON / Collibra        OpenAPI / Postman / HAR / hand-YAML
  / sample data files

Outputs:                              Outputs:
  Parquet / Delta Lake / SCD2           WireMock mappings, Pact, Postman,
  (+ adjacent: JSON Lines, SQL          OpenAPI examples, JSON/XML fixtures
   seeds — small extension)

ML / LLM features:                    ML / LLM features:
  FK relationship inference,            Schema enrichment, example
  schema enrichment, multi-provider     synthesis, multi-provider
  (shared)                              (shared)
```

> The two tracks share the **engine** (value generation, rules, LLM provider).
> They have **independent config models, independent ingest paths, independent
> renderers**. Nothing on the mocks track changes anything on the data track.

---

*Plan rewritten: 2026-05-04 | Replaces the earlier `api:`-block-on-TableConfig
draft. Next step: Phase A — design and land `models/mock_models.py` plus a
hand-authored `sdp-mock-v1` round-trip parser.*
