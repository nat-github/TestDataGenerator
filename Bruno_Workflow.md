# Bruno + WireMock — Offline Workflow Guide

End-to-end recipe for trying the stubs/mocks track offline using:

- **Synthetic Data Platform** — generates realistic stub bodies
- **WireMock** — runs a local HTTP server that serves those stubs
- **Bruno** — open-source API client (alternative to Postman) that calls
  the local WireMock server

You'll go from a YAML/JSON OpenAPI spec to an interactive mock API that
Bruno can hit, in a single terminal session, with no cloud services and
no API keys.

---

## What you'll have when this is done

```
[ OpenAPI spec ]                              [ Bruno UI ]
       |                                            |
       | mock-init                                  | sends HTTP requests
       v                                            v
[ sdp-mock-v1 ]   mock-render   [ WireMock files ]  [ WireMock server ]
       |  ───────────────────>          ─────────>  ─────────>
                                                        |
                                                        | returns realistic JSON
                                                        v
                                                  [ Bruno collection ]
```

You'll see: realistic IBANs, UUIDs, emails, names, dates, ISBNs — all
generated locally and reproducibly with `--seed`, no fake-data service.

---

## Prerequisites

| Tool | Why | How to install |
|---|---|---|
| **Python 3.13** + Poetry | Runs the platform | Already on your machine if the rest of this repo works |
| **Java 11+** | WireMock standalone is a JAR | `java -version` to check |
| **WireMock standalone JAR** | The mock server | Download `wiremock-standalone-X.Y.Z.jar` from [WireMock's official page](https://wiremock.org/docs/running-standalone/) and drop it anywhere — e.g. `~/tools/wiremock.jar` |
| **Bruno** | The API client UI | Get it from the official Bruno site |

WireMock and Bruno both have official websites with download links — check
their docs for the latest version. No specific versions are pinned here.

---

## Step 1 — Pick an example spec

Three reference specs ship in `examples/openapi/`:

| File | Endpoints | Schemas | Highlights |
|---|---:|---:|---|
| `simple_books.yaml` | 2 | 2 | List + GET-by-id, no auth, ISBN regex |
| `medium_tasks.yaml` | 5 | 4 | Full CRUD, bearer auth, pagination, multi-status responses |
| `complex_payments.yaml` | 5 | 14 | Fintech-shaped: OAuth2, IBAN/BIC, idempotency, allOf, oneOf, webhooks |

Start with **simple_books** to verify the loop, then try the others.

---

## Step 2 — Convert the spec → sdp-mock-v1 config

```bash
poetry run python main.py mock-init \
  --from examples/openapi/simple_books.yaml \
  --output mocks/books.yaml
```

Expected output:

```
=== mock-init ===
  Source:    examples/openapi/simple_books.yaml
  Output:    mocks/books.yaml
  Endpoints: 2
  Schemas:   2
    GET    /books         -> [200]    (list_books)
    GET    /books/{id}    -> [200,404] (get_book_by_id)
```

You can inspect `mocks/books.yaml` and edit it — change `business_values`,
add `examples`, tweak `weight` on response variants, etc. Then re-render.
This is the layer you customise; the OpenAPI spec is just the seed.

---

## Step 3 — Render WireMock stubs + JSON fixtures

```bash
poetry run python main.py mock-render \
  --config mocks/books.yaml \
  --output stubs/books \
  --format wiremock,json \
  --examples 5 \
  --seed 42
```

Expected output:

```
  wiremock:  10 mapping(s) -> stubs/books/wiremock
  json:      5 fixture(s)  -> stubs/books/json

=== mock-render ===
  Total: 15 file(s)
```

Result tree:

```
stubs/books/
├── wiremock/
│   ├── mappings/
│   │   ├── list_books__200__00.json   (one mapping per example, per status)
│   │   ├── list_books__200__01.json
│   │   ├── ...
│   │   ├── get_book_by_id__200__00.json
│   │   ├── get_book_by_id__200__01.json
│   │   ├── ...
│   │   └── get_book_by_id__404__00.json
│   └── __files/                       (reserved for future bodyFileName feature)
└── json/
    ├── list_books__200.json           (raw response body, no HTTP wrapper)
    ├── get_book_by_id__200.json
    ├── get_book_by_id__404.json
    ├── requests/                      (request bodies, when defined)
    └── schemas/                       (one canonical example per reusable schema)
```

`--seed 42` makes this run byte-identical to any future re-run with the
same seed. Pass a different seed (or omit it) for fresh values.

### `--match-mode concrete` vs `--match-mode any`

| Mode | Result for `/books/{id}` |
|---|---|
| `concrete` (default) | `urlPath: /books/1825` (one mapping per concrete id from the renderer) — **best for snapshot tests** |
| `any` | `urlPathPattern: /books/[0-9]+` (one mapping per status, regex-matches any id) — **best for exploratory dev** |

Pick `any` when you want Bruno to be able to pass any id and still get a
response. Pick `concrete` when your tests assert on specific ids.

---

## Step 4 — Run WireMock standalone

In a separate terminal:

```bash
java -jar ~/tools/wiremock.jar \
  --root-dir stubs/books/wiremock \
  --port 8080 \
  --verbose
```

WireMock will print:

```
The WireMock server is started ...
port:                 8080
root directory:       stubs/books/wiremock
mappings loaded:      10
```

Sanity check from the same shell:

```bash
curl http://localhost:8080/books
```

You should see a JSON array of book objects with realistic titles and
ISBNs (the ones from the spec's `example:` block, since `simple_books.yaml`
hand-authored them).

Pick any concrete id from the `mappings/` filenames and try the detail
endpoint:

```bash
ls stubs/books/wiremock/mappings | grep get_book_by_id__200
# get_book_by_id__200__00.json   ...

# inspect to find the rendered id
cat stubs/books/wiremock/mappings/get_book_by_id__200__00.json | python -m json.tool | grep urlPath
# "urlPath": "/books/1825"

curl http://localhost:8080/books/1825
```

If WireMock is running with `--match-mode any`, any integer id will work:
`curl http://localhost:8080/books/42`.

---

## Step 5 — Hit the mock server from Bruno

### 5a. Import the OpenAPI spec into Bruno

Bruno can ingest OpenAPI specs directly:

1. Open Bruno
2. **Collection → New Collection** (or **Import Collection** if you prefer)
3. Choose **Import → OpenAPI v3 Spec** and pick
   `examples/openapi/simple_books.yaml`
4. Bruno generates a collection with one request per operation

### 5b. Point the Collection at the local WireMock

Bruno collections support **environments**. Create a `local` environment:

```
baseUrl = http://localhost:8080/api/v1
```

(The `simple_books.yaml` spec declares `servers: - url: http://localhost:8080/api/v1`,
so requests already use that base.)

### 5c. Send requests

| Bruno request | Calls |
|---|---|
| **List books** (`GET {{baseUrl}}/books`) | `GET http://localhost:8080/api/v1/books` |
| **Get book by id** (`GET {{baseUrl}}/books/{{id}}`) | `GET http://localhost:8080/api/v1/books/1825` (or any id from the rendered files in `--match-mode concrete`, or any integer in `--match-mode any`) |

You'll see the realistic JSON bodies WireMock loaded from the renderer.

---

## Step 6 — Try the medium and complex specs

### Medium — Tasks API

```bash
poetry run python main.py mock-init \
  --from examples/openapi/medium_tasks.yaml \
  --output mocks/tasks.yaml

poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks \
  --format wiremock,json \
  --examples 3 --seed 42 --match-mode any

java -jar ~/tools/wiremock.jar --root-dir stubs/tasks/wiremock --port 8081
```

In Bruno, create a request with the bearer token header (any value — WireMock
just checks presence):

```
GET http://localhost:8081/api/v2/tasks?page=1&size=25
Authorization: Bearer test-token
```

You'll get a paginated `TaskPage` response with realistic task titles,
UUIDs, emails, and ISO timestamps. Headers like `X-Total-Count` and
`X-RateLimit-Remaining` come back populated.

Try the `POST /tasks`:

```
POST http://localhost:8081/api/v2/tasks
Content-Type: application/json
Authorization: Bearer test-token

{ "title": "Review PR #42", "priority": 3 }
```

You'll get a 201 with the created task.

### Complex — Payments API

```bash
poetry run python main.py mock-init \
  --from examples/openapi/complex_payments.yaml \
  --output mocks/payments.yaml

poetry run python main.py mock-render \
  --config mocks/payments.yaml \
  --output stubs/payments \
  --format wiremock --examples 5 --seed 1 --match-mode any

java -jar ~/tools/wiremock.jar --root-dir stubs/payments/wiremock --port 8082
```

Now Bruno can hit:

```
GET http://localhost:8082/v1/accounts
GET http://localhost:8082/v1/accounts/DE89370400440532013000
GET http://localhost:8082/v1/accounts/DE89370400440532013000/transactions

POST http://localhost:8082/v1/payments
Content-Type: application/json
Idempotency-Key: 1cab09a7-3f9b-4d05-9b63-1d65f0e45b21
Authorization: Bearer dummy-token

{
  "amount":           { "value": "50.00", "currency": "EUR" },
  "source_iban":      "DE89370400440532013000",
  "destination_iban": "GB29NWBK60161331926819",
  "reference":        "Test transfer"
}
```

The `POST /payments` mapping has 4 response variants (201, 202, 422, 401, 429).
WireMock will pick by matcher precedence; you can edit individual mapping files
under `stubs/payments/wiremock/mappings/` to force a specific behaviour.

---

## Step 7 — Iterate

The `mocks/<name>.yaml` is your editable layer. Common tweaks:

```yaml
# Add a higher-quality example
schemas:
  Account:
    properties:
      iban:
        special_rule: IBAN
        example: DE89370400440532013000   # specific value for predictable tests

# Lower the chance of a 4xx variant
endpoints:
  - name: create_payment
    responses:
      - status: 201
        weight: 0.95
      - status: 422
        weight: 0.05

# Add network latency to test timeouts
settings:
  default_latency_ms: 250
```

Re-run `mock-render` and reload WireMock (`Ctrl-C` and restart, or call its
admin API to reset).

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `404 Not Found` from WireMock for a templated path | `--match-mode concrete` rendered specific ids that don't match what Bruno is sending. Use `--match-mode any` or look up a concrete id in the mapping filenames. |
| Bruno reports CORS errors | WireMock standalone has CORS off by default. Add `--enable-stub-cors` to the WireMock command, or set `cors_enabled: true` in your `mocks/*.yaml` settings (the renderer will be extended to honour this in a follow-up). |
| `mock-render` fails with "unresolved $ref" | The OpenAPI spec references a schema that's not in `components.schemas`. Inspect `mocks/<name>.yaml` and either define the missing schema or replace the `$ref` with an inline shape. |
| WireMock loads zero mappings | Pointed `--root-dir` at the wrong directory. It should be the directory **containing** `mappings/` and `__files/`, not `mappings/` itself. |
| Seeds give different output between runs | You changed the `mocks/<name>.yaml` between runs (so the input changed). Same seed + same input → identical output is the contract; same seed + different input → no guarantees. |

---

## More render formats (Phase E)

Beyond `wiremock` and `json`, the renderer supports three more:

```bash
# Pact v3 contracts — for consumer-driven contract testing
poetry run python main.py mock-render \
  --config mocks/tasks.yaml --output stubs/pact \
  --format pact --pact-consumer client-app --pact-provider tasks-svc

# Postman v2.1 collection — drag-and-drop into Postman or Bruno
poetry run python main.py mock-render \
  --config mocks/tasks.yaml --output stubs/postman \
  --format postman --examples 5

# Inject realistic example: blocks back into the source OpenAPI spec —
# useful for documentation tools (Redoc, Stoplight, Swagger UI)
poetry run python main.py mock-render \
  --config mocks/tasks.yaml --output stubs/oas-enriched \
  --format openapi-examples \
  --openapi-source examples/openapi/medium_tasks.yaml

# Or all formats at once
poetry run python main.py mock-render \
  --config mocks/tasks.yaml --output stubs/everything \
  --format wiremock,json,pact,postman --examples 3 --seed 42
```

Bruno can import the Postman collection directly:
**Collection → Import → Postman** and pick the `*.postman_collection.json`.

## Reverse importers (Phase I) — start from existing artefacts

When you already have a Postman collection or a HAR capture and you want
to convert it into a `sdp-mock-v1` config, `mock-init` auto-detects the
source format:

```bash
# Postman collection → sdp-mock-v1 (auto-detected)
poetry run python main.py mock-init \
  --from my-team-collection.json --output mocks/imported.yaml

# HAR capture from browser DevTools (Save All as HAR with content)
poetry run python main.py mock-init \
  --from session.har --output mocks/from-real-traffic.yaml

# Force a particular importer if auto-detection guesses wrong
poetry run python main.py mock-init \
  --from ambiguous.json --output mocks/x.yaml --source-type postman
```

The HAR importer is especially handy: capture a session against the real
API, run `mock-init`, and you have a starter `MockConfig` reflecting the
exact requests/responses your code makes. Edit, render, run WireMock,
and now your tests can run offline against that captured behaviour.

## Scenarios (Phase G) — stateful stubs

Add a `scenarios:` block to your `mocks/<name>.yaml` to model state changes:

```yaml
scenarios:
  - name: rate_limit_after_3_calls
    states:
      - on_match: { endpoint: list_tasks }
        after: 3
        next_response:
          status: 429
          body: { code: RATE_LIMITED, message: Slow down }

  - name: create_then_dup
    states:
      - on_match: { endpoint: create_task }
        after: 1
        next_response: { status: 409, body: { code: DUPLICATE } }
```

Re-render with WireMock format and the renderer will emit additional
mappings carrying `scenarioName` + `requiredScenarioState` + `newScenarioState`.
WireMock advances state automatically as each request matches; the third
call to `list_tasks` returns 429, subsequent calls to `create_task` return
409. No code changes — pure config.

## LLM-assisted authoring (Phase H) — `mock-enrich`

Once you have a `sdp-mock-v1` config, `mock-enrich` can fill in two
common gaps using whatever LLM you have configured (hosted Anthropic /
OpenAI / Groq, or local LM Studio / Ollama):

```bash
# Default: hosted Anthropic — needs ANTHROPIC_API_KEY
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml

# Local LM Studio — no key needed
SDP_LLM_PROVIDER=lm-studio \
SDP_LLM_MODEL="meta-llama-3.1-8b-instruct" \
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml

# Or set provider per-command
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml \
  --llm-provider ollama --llm-model llama3.2

# Skip one of the two passes
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml --output mocks/x.yaml \
  --no-draft-errors          # only fill examples
```

Two passes:
1. **fill_missing_examples** — for each `MockConfig.schemas` entry without
   an `example:`, ask the LLM to suggest one. Existing examples are
   preserved.
2. **draft_error_responses** — for each endpoint missing 4xx/5xx
   variants, draft realistic error envelopes (NOT_FOUND, UNAUTHORIZED,
   VALIDATION_FAILED, etc.) appropriate to the endpoint's domain.

Failures degrade gracefully: malformed LLM JSON is skipped with a warning,
provider exceptions are caught, and the config is returned unchanged.
The result diff is purely additive — the source config never loses data.

---

## One-page cheat sheet

```bash
# 1. Convert OpenAPI → editable mock config
poetry run python main.py mock-init \
  --from examples/openapi/medium_tasks.yaml \
  --output mocks/tasks.yaml

# 2. Validate (optional but useful)
poetry run python main.py mock-lint --config mocks/tasks.yaml

# 3. Render any combination of formats
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks \
  --format wiremock,json,pact,postman \
  --examples 5 \
  --seed 42 \
  --match-mode any \
  --pact-consumer client-app --pact-provider tasks-svc

# 4. Optional: enrich the config with LLM-suggested examples + error responses
SDP_LLM_PROVIDER=lm-studio poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml

# 5. Reverse: start from a Postman collection or HAR capture
poetry run python main.py mock-init \
  --from session.har --output mocks/from-real-traffic.yaml

# 6. Run WireMock locally
java -jar ~/tools/wiremock.jar \
  --root-dir stubs/tasks/wiremock \
  --port 8081 \
  --verbose

# 7. Send requests from Bruno against http://localhost:8081/...
```
