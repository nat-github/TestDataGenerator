# Master Commands - One Stop Execution Guide

This is a consolidated command hub for the Synthetic Data Platform.
It collates practical commands from `CLAUDE.md`, `Examples_Walkthrough.md`, `Bruno_Workflow.md`, `UI_Quickstart.md`, and `MCP_Integration.md`.

## 0) Setup and sanity

| What | Command |
|---|---|
| Install project deps | `poetry install` |
| Install UI extra | `poetry install --extras ui` |
| Install MCP extra | `poetry install --extras mcp` |
| Show CLI help | `poetry run python main.py --help` |

Optional runtime deps:

```bash
pip install azure-storage-blob
pip install boto3
pip install matplotlib
poetry install --extras mimesis
```

## 1) Data generation (snapshot)

| What | Command |
|---|---|
| Basic snapshot from Excel | `python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --default-records 1000` |
| Snapshot from JSON config | `python main.py generate --config config/sample_workflow.json --output output/sample_run --seed 42` |
| Snapshot from YAML workflow | `python main.py generate --config config/sample_workflow.yaml --output output/sample_run --seed 42` |
| Reproducible run (seeded) | `python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --seed 42` |
| Legacy mode (treated as generate) | `python main.py --config config/Acct_bkng.xlsx --output output/run_01` |

Inspect parquet output:

```bash
poetry run python readParquet.py output/run_01
```

## 2) Linting and config checks

| What | Command |
|---|---|
| Lint Excel/YAML/JSON generation config | `python main.py lint --config config/Acct_bkng.xlsx` |
| Lint mock config (`sdp-mock-v1`) | `python main.py mock-lint --config mocks/tasks.yaml` |

## 3) Relationship inference and feedback loop

| What | Command |
|---|---|
| Infer relationships during generate (ML default) | `python main.py generate --config config/bare.xlsx --output output/run_01 --infer-relationships` |
| Infer relationships during generate (LLM) | `python main.py generate --config config/bare.xlsx --output output/run_01 --infer-relationships --method llm --llm-confidence 0.7` |
| Standalone infer + ER output | `python main.py infer-relationships --config config/bare.xlsx --config-output config/inferred.yaml --er-output diagrams/inferred.mmd` |
| Record SME feedback | `python main.py record-feedback --inferred config/inferred.yaml --reviewed config/inferred.yaml.reviewed` |

## 4) LLM schema enrichment (data track)

| What | Command |
|---|---|
| Enrich bare schema config | `python main.py enrich --config config/bare.xlsx --output config/enriched.yaml --confidence 0.7` |

LM Studio local example:

```bash
SDP_LLM_PROVIDER=lm-studio \
SDP_LLM_MODEL="meta-llama-3.1-8b-instruct" \
python main.py enrich --config config/bare.xlsx --output config/enriched.yaml --confidence 0.6
```

## 5) CDC workflows

| What | Command |
|---|---|
| Delta between snapshots | `python main.py delta --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/delta_run` |
| Build SCD2 history | `python main.py scd2 --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/scd2_run` |

## 6) ER diagrams and cloud upload

| What | Command |
|---|---|
| Generate ER diagram(s) during snapshot | `python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --er-diagram --er-format mermaid dot` |
| Upload output to Azure | `python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --upload-to azure://my-container/prefix` |
| Upload output to S3 | `python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --upload-to s3://my-bucket/prefix` |

## 7) Collibra import

| What | Command |
|---|---|
| Import dataset config from Collibra | `python main.py collibra-import --dataset "Account Booking" --output config/from_collibra.yaml` |
| Import from specific domain | `python main.py collibra-import --dataset "Customer" --domain "Finance" --output config/customer.yaml` |

## 8) Mocks/stubs - core flow

### 8.1 Convert source artifact -> mock config (`mock-init`)

```bash
python main.py mock-init --from examples/openapi/medium_tasks.yaml --output mocks/tasks.yaml
python main.py mock-init --from session.har --output mocks/captured.yaml
python main.py mock-init --from collection.json --output mocks/postman.yaml
```

### 8.2 Render stubs/artifacts from mock config (`mock-render`)

```bash
python main.py mock-render --config mocks/tasks.yaml --output stubs/tasks \
  --format wiremock,json,pact,postman --examples 5 --seed 42 --match-mode any \
  --pact-consumer client-app --pact-provider tasks-svc
```

OpenAPI examples round-trip:

```bash
python main.py mock-render --config mocks/tasks.yaml --output stubs/oas \
  --format openapi-examples --openapi-source examples/openapi/medium_tasks.yaml
```

LLM assist for mocks config:

```bash
python main.py mock-enrich --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml
```

### 8.3 Full LLM + mocks/stubs execution flow (end-to-end)

```bash
# 0) Pick LLM provider/model (example: local LM Studio)
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
export SDP_LLM_BASE_URL=http://localhost:1234/v1

# 1) Import source API artifact -> editable mock config
python main.py mock-init --from examples/openapi/medium_tasks.yaml --output mocks/tasks.yaml

# 2) LLM-enrich mock config (fills missing examples + drafts 4xx/5xx)
python main.py mock-enrich --config mocks/tasks.yaml --output mocks/tasks_enriched.yaml

# 3) Validate enriched mock config
python main.py mock-lint --config mocks/tasks_enriched.yaml

# 4) Render all major stub/contract exports
python main.py mock-render --config mocks/tasks_enriched.yaml --output stubs/tasks \
  --format wiremock,json,pact,postman,openapi-examples \
  --openapi-source examples/openapi/medium_tasks.yaml \
  --pact-consumer client-app --pact-provider tasks-svc \
  --examples 5 --seed 42 --match-mode any

# 5) Run WireMock for live mock execution
java -jar ~/tools/wiremock.jar --root-dir stubs/tasks/wiremock --port 8081 --verbose
```

Quick verification:

```bash
curl http://localhost:8081/__admin/mappings
curl "http://localhost:8081/api/v2/tasks?page=1&size=25"
```

## 9) WireMock run commands (local server)

Simple books:

```bash
java -jar ~/tools/wiremock.jar --root-dir stubs/books/wiremock --port 8080 --verbose
```

Medium tasks:

```bash
java -jar ~/tools/wiremock.jar --root-dir stubs/tasks/wiremock --port 8081
```

Complex payments:

```bash
java -jar ~/tools/wiremock.jar --root-dir stubs/payments/wiremock --port 8082
```

Quick curl checks:

```bash
curl http://localhost:8080/books
curl http://localhost:8080/books/42
curl http://localhost:8080/__admin/mappings
```

## 10) Mocks demo commands from walkthrough

```bash
python main.py mock-init --from examples/openapi/simple_books.yaml --output mocks/books.yaml
python main.py mock-render --config mocks/books.yaml --output stubs/books --format wiremock,json --examples 5 --seed 42 --match-mode any

python main.py mock-render --config mocks/books.yaml --output stubs/all \
  --format wiremock,json,pact,postman,openapi-examples \
  --openapi-source examples/openapi/simple_books.yaml \
  --pact-consumer reader-app --pact-provider books-svc \
  --examples 3 --seed 1

python main.py mock-init --from examples/mocks/sample_postman_collection.json --output mocks/from_postman.yaml
python main.py mock-init --from examples/mocks/sample_session.har --output mocks/from_har.yaml
```

## 11) Streamlit UI

| What | Command |
|---|---|
| Run UI | `poetry run streamlit run ui/streamlit_app.py` |
| UI tests | `poetry run pytest tests/test_ui_streamlit.py -v` |

## 12) MCP server

| What | Command |
|---|---|
| Run MCP server (stdio) | `poetry run python -m mcp_server.server` |
| MCP tests | `poetry run pytest tests/test_mcp_server.py -v` |

## 13) Test commands

| What | Command |
|---|---|
| Run all tests | `poetry run pytest tests/ -v` |
| Single test example | `poetry run pytest tests/test_config_and_parquet_flows.py::test_name -v` |

## 14) Useful env vars (execution time)

LLM provider selection:

```bash
export SDP_LLM_PROVIDER=anthropic   # or openai, lm-studio, ollama, azure-openai, groq, together, openrouter
export SDP_LLM_MODEL=claude-sonnet-4-6
export SDP_LLM_BASE_URL=http://localhost:1234/v1
```

API keys (as needed by provider/workflow):

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
```

Cloud upload creds:

```bash
export AZURE_STORAGE_CONNECTION_STRING=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

## 15) Suggested daily command sequence (quickstart)

```bash
# 1) Validate input config
python main.py lint --config config/Acct_bkng.xlsx

# 2) Generate snapshot
python main.py generate --config config/Acct_bkng.xlsx --output output/run_01 --seed 42

# 3) Inspect output
poetry run python readParquet.py output/run_01
```

For mocks:

```bash
# 1) Import source API artifact
python main.py mock-init --from examples/openapi/simple_books.yaml --output mocks/books.yaml

# 2) Validate mock config
python main.py mock-lint --config mocks/books.yaml

# 3) Render stubs
python main.py mock-render --config mocks/books.yaml --output stubs/books --format wiremock,json --seed 42 --match-mode any

# 4) Run WireMock
java -jar ~/tools/wiremock.jar --root-dir stubs/books/wiremock --port 8080 --verbose
```

