# Synthetic Data Platform — One-Stop Commands Reference

Every command for every feature, using the bundled examples under
`examples/`. Copy-paste; no IDE required.

> **Convention:** all paths are relative to the repo root. Run from there.
> Output goes to `output/...` by default — that directory is git-ignored.

---

## Table of contents

- [0. One-time setup](#0-one-time-setup)
- [1. Data generation — single-table](#1-data-generation--single-table)
- [2. Data generation — multi-table with relationships](#2-data-generation--multi-table-with-relationships)
- [3. Rules and derived columns](#3-rules-and-derived-columns)
- [4. CDC — delta and SCD2](#4-cdc--delta-and-scd2)
- [5. Relationship inference (ML + LLM)](#5-relationship-inference-ml--llm)
- [6. LLM schema enrichment](#6-llm-schema-enrichment)
- [7. Validation — FK + Great Expectations + Quality reports](#7-validation--fk--great-expectations--quality-reports)
- [8. PII scanning](#8-pii-scanning)
- [9. Mocks — OpenAPI / Postman / HAR ingest](#9-mocks--openapi--postman--har-ingest)
- [10. Mocks — render WireMock / JSON / Pact / Postman / OpenAPI examples](#10-mocks--render-wiremock--json--pact--postman--openapi-examples)
- [11. Mocks — LLM-assisted authoring](#11-mocks--llm-assisted-authoring)
- [12. Cloud upload (Azure / S3)](#12-cloud-upload-azure--s3)
- [13. Streamlit UI](#13-streamlit-ui)
- [14. MCP server](#14-mcp-server)
- [15. Docker — every command above, in a container](#15-docker--every-command-above-in-a-container)
- [16. Inspect outputs](#16-inspect-outputs)
- [17. ER diagrams + Collibra](#17-er-diagrams--collibra)
- [18. End-to-end demo arc (10 minutes)](#18-end-to-end-demo-arc-10-minutes)

---

## 0. One-time setup

Pick **one** path below.

### Path A — Poetry (local Python)

```bash
# Install with every optional extra so the full feature set works
poetry install --extras "gx ui mcp mimesis"
```

### Path B — Docker

```bash
docker build -t sdp:latest .
# Then prefix every CLI command below with:
#   docker run --rm -v "$PWD:/work" sdp:latest
# and replace `examples/...` with `/work/examples/...`
# See section 15 for the recipes.
```

### Path C — docker-compose

```bash
docker compose build
# Use:
#   docker compose run --rm cli <subcommand> [args...]
#   docker compose up streamlit
```

---

## 1. Data generation — single-table

### 1.1 Simplest possible run (YAML)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/01_simple_users.yaml \
  --output output/01_simple_users \
  --default-records 200 \
  --seed 42
```

### 1.2 Same example, JSON config

```bash
poetry run python main.py generate \
  --config examples/configs/json/01_simple_users.json \
  --output output/01_simple_users_json \
  --default-records 200 --seed 42
```

### 1.3 Same example, Excel config

```bash
poetry run python main.py generate \
  --config examples/configs/xlsx/01_simple_users.xlsx \
  --output output/01_simple_users_xlsx \
  --default-records 200 --seed 42
```

### 1.4 Special-rules showcase (60+ rules)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/03_special_rules_showcase.yaml \
  --output output/03_special_rules \
  --default-records 50 --seed 1
```

### 1.5 Multi-locale (5 locales side by side)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/05_multi_locale.yaml \
  --output output/05_multi_locale \
  --default-records 100 --seed 3
```

### 1.6 Distributions + business values (statistical realism)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/04_distributions_and_business_values.yaml \
  --output output/04_distributions \
  --default-records 1000 --seed 11
```

### 1.7 PII-rich schema (for the scanner demo in §8)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/09_pii_columns.yaml \
  --output output/09_pii \
  --default-records 200 --seed 4
```

### 1.8 Per-table record counts

```bash
# Override the default for specific tables: --records <table>:<count>
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_custom_counts \
  --default-records 100 \
  --records customers:50 orders:300 order_items:1500 \
  --seed 7
```

### 1.9 Streaming generation (large runs)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/04_distributions_and_business_values.yaml \
  --output output/04_streaming \
  --default-records 500000 \
  --stream --chunk-size 100000 \
  --seed 11
```

---

## 2. Data generation — multi-table with relationships

### 2.1 E-commerce (3 tables, FKs preserved)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_ecommerce \
  --default-records 500 --seed 7
```

### 2.2 Same, with FK validation after generation

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_ecommerce_validated \
  --default-records 500 --seed 7 \
  --validate
```

### 2.3 Same, with ER diagram (Mermaid)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_ecommerce_with_er \
  --default-records 500 --seed 7 \
  --er-diagram --er-format mermaid
```

### 2.4 Same, with all three diagram formats

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_ecommerce_diagrams \
  --default-records 500 --seed 7 \
  --er-diagram --er-format mermaid dot png \
  --er-output output/02_ecommerce_diagrams/diagram
```

> `png` requires `pip install matplotlib`.

---

## 3. Rules and derived columns

### 3.1 When/then rules + derived columns demo

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/06_rules_and_derived.yaml \
  --output output/06_rules \
  --default-records 200 --seed 21
```

**What to check** in the output:
- `closure_date` is null where `status == ACTIVE`, fixed value where `status == CLOSED`
- `full_name` is `"{first_name} {last_name}"` for every row
- `overdraft_flag == 'Y'` only when balance < 0 AND status = ACTIVE

---

## 4. CDC — delta and SCD2

### 4.1 Delta workflow — 3 commands

```bash
# 1. Snapshot v1
poetry run python main.py generate \
  --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
  --output output/07_cdc/snap_v1 \
  --default-records 1000 --seed 1

# 2. Snapshot v2 (different seed → some inserts, updates, deletes)
poetry run python main.py generate \
  --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
  --output output/07_cdc/snap_v2 \
  --default-records 1000 --seed 2

# 3. Compute delta (Delta Lake table with _operation column)
poetry run python main.py delta \
  --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
  --previous output/07_cdc/snap_v1 \
  --current  output/07_cdc/snap_v2 \
  --output   output/07_cdc/delta
```

### 4.2 SCD2 workflow — 3 commands

```bash
# 1. Baseline
poetry run python main.py generate \
  --config examples/configs/yaml/08_scd2_history.yaml \
  --output output/08_scd2/snap_v1 \
  --default-records 500 --seed 1

# 2. Updated state
poetry run python main.py generate \
  --config examples/configs/yaml/08_scd2_history.yaml \
  --output output/08_scd2/snap_v2 \
  --default-records 500 --seed 2

# 3. Build SCD2 history (effective_from_ts / effective_to_ts / is_current / version_num)
poetry run python main.py scd2 \
  --config examples/configs/yaml/08_scd2_history.yaml \
  --previous output/08_scd2/snap_v1 \
  --current  output/08_scd2/snap_v2 \
  --output   output/08_scd2/history
```

---

## 5. Relationship inference (ML + LLM)

### 5.1 ML inferrer (free, deterministic — default)

```bash
poetry run python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_inferred.yaml \
  --er-output     output/11_inferred.mmd \
  --method ml
```

### 5.2 LLM inferrer (hosted Anthropic — needs `ANTHROPIC_API_KEY`)

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."

poetry run python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_llm.yaml \
  --method llm \
  --llm-confidence 0.7
```

### 5.3 LLM inferrer via local LM Studio (no API key)

```bash
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
# (Optional) export SDP_LLM_BASE_URL=http://localhost:1234/v1

poetry run python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_lmstudio.yaml \
  --method llm
```

### 5.4 Both methods (ML first, LLM only on candidates ML missed)

```bash
poetry run python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_both.yaml \
  --method both
```

### 5.5 Close the loop — feed SME edits back to the adaptive classifier

```bash
# 1. Open output/11_inferred.yaml in any editor.
# 2. Delete entries you disagree with; edit columns to correct.
# 3. Save as output/11_reviewed.yaml.
# 4. Record the deltas:

poetry run python main.py record-feedback \
  --inferred  output/11_inferred.yaml \
  --reviewed  output/11_reviewed.yaml
```

After ~30 feedback rows (with both classes), the adaptive classifier
activates and starts overriding the heuristic on future runs.

### 5.6 Generate WITH inference baked in (one shot)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --output output/11_with_inference \
  --default-records 100 \
  --infer-relationships --method ml \
  --validate
```

---

## 6. LLM schema enrichment

Bare config in → enriched YAML out (LLM suggests `business_values`,
`special_rules`, `data_type` corrections).

### 6.1 Hosted Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
poetry run python main.py enrich \
  --config examples/configs/yaml/10_bare_for_llm_enrichment.yaml \
  --output output/10_enriched.yaml \
  --confidence 0.7
```

### 6.2 Local LM Studio (no API key)

```bash
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
poetry run python main.py enrich \
  --config examples/configs/yaml/10_bare_for_llm_enrichment.yaml \
  --output output/10_enriched.yaml \
  --confidence 0.6
```

Diff the two files to see what was added:

```bash
diff examples/configs/yaml/10_bare_for_llm_enrichment.yaml output/10_enriched.yaml
```

---

## 7. Validation — FK + Great Expectations + Quality reports

### 7.1 FK integrity (built in)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/07_fk_validated \
  --default-records 500 --seed 7 \
  --validate
```

### 7.2 Great Expectations during generation

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/07_gx \
  --default-records 500 --seed 7 \
  --validate-with-gx --gx-fail-on-error
```

### 7.3 Great Expectations standalone (separate run)

```bash
poetry run python main.py validate-data \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --input  output/07_gx \
  --tolerance 0.5 \
  --report-json output/07_gx/validation.json \
  --verbose
```

### 7.4 Quality report — univariate-only (no source)

```bash
poetry run python main.py quality-report \
  --generated output/02_ecommerce \
  --output-html output/02_ecommerce/quality.html \
  --output-json output/02_ecommerce/quality.json
```

### 7.5 Quality report — fidelity vs source

```bash
# 1. Generate v1 — call this "source"
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/qual/source --default-records 500 --seed 1

# 2. Generate v2 — call this "synthetic"
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/qual/synthetic --default-records 500 --seed 2

# 3. Compute fidelity
poetry run python main.py quality-report \
  --generated output/qual/synthetic \
  --source    output/qual/source \
  --output-html output/qual/quality.html \
  --output-json output/qual/quality.json \
  --verbose
```

### 7.6 Quality report with privacy proxy (catches near-duplicates)

```bash
poetry run python main.py quality-report \
  --generated output/qual/synthetic \
  --source    output/qual/source \
  --privacy-threshold 1e-6 \
  --output-json output/qual/quality_priv.json
```

---

## 8. PII scanning

### 8.1 Scan generated data

```bash
# First generate the PII-rich example
poetry run python main.py generate \
  --config examples/configs/yaml/09_pii_columns.yaml \
  --output output/09_pii \
  --default-records 200 --seed 4

# Then scan it
poetry run python main.py pii-scan --input output/09_pii --verbose
```

---

## 9. Mocks — OpenAPI / Postman / HAR ingest

`mock-init` auto-detects the source format. Override with `--source-type`
if needed.

### 9.1 Simple OpenAPI spec → sdp-mock-v1

```bash
poetry run python main.py mock-init \
  --from   examples/openapi/simple_books.yaml \
  --output mocks/books.yaml
```

### 9.2 Medium spec (auth, pagination, multi-status responses)

```bash
poetry run python main.py mock-init \
  --from   examples/openapi/medium_tasks.yaml \
  --output mocks/tasks.yaml
```

### 9.3 Complex spec (allOf / oneOf / IBAN / OAuth / webhooks)

```bash
poetry run python main.py mock-init \
  --from   examples/openapi/complex_payments.yaml \
  --output mocks/payments.yaml
```

### 9.4 Reverse — Postman collection → sdp-mock-v1

```bash
poetry run python main.py mock-init \
  --from   examples/mocks/sample_postman_collection.json \
  --output mocks/from_postman.yaml
```

### 9.5 Reverse — HAR capture → sdp-mock-v1

```bash
poetry run python main.py mock-init \
  --from   examples/mocks/sample_session.har \
  --output mocks/from_har.yaml
```

### 9.6 Validate any sdp-mock-v1 config

```bash
poetry run python main.py mock-lint --config mocks/tasks.yaml
```

---

## 10. Mocks — render WireMock / JSON / Pact / Postman / OpenAPI examples

### 10.1 WireMock stub mappings

```bash
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks \
  --format wiremock \
  --examples 5 --seed 42 --match-mode any
```

### 10.2 JSON fixtures only

```bash
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks_fixtures \
  --format json \
  --examples 3 --seed 42
```

### 10.3 Pact contract files

```bash
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks_pact \
  --format pact \
  --pact-consumer reader-app \
  --pact-provider tasks-svc \
  --examples 3 --seed 42
```

### 10.4 Postman collection (importable into Postman / Bruno)

```bash
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks_postman \
  --format postman \
  --examples 5 --seed 42
```

### 10.5 OpenAPI spec round-trip with `example:` blocks injected

```bash
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks_oas \
  --format openapi-examples \
  --openapi-source examples/openapi/medium_tasks.yaml \
  --seed 42
```

### 10.6 Every format at once

```bash
poetry run python main.py mock-render \
  --config mocks/tasks.yaml \
  --output stubs/tasks_all \
  --format wiremock,json,pact,postman,openapi-examples \
  --openapi-source examples/openapi/medium_tasks.yaml \
  --pact-consumer reader-app --pact-provider tasks-svc \
  --examples 3 --seed 42 --match-mode any
```

### 10.7 Stateful scenarios (rate limit after 3 calls)

```bash
poetry run python main.py mock-render \
  --config examples/mocks/accounts_with_scenarios.yaml \
  --output stubs/accounts \
  --format wiremock --match-mode any --seed 42

# Run WireMock standalone (Java required)
java -jar ~/tools/wiremock.jar \
  --root-dir stubs/accounts/wiremock \
  --port 8080 --verbose

# Watch the rate limit kick in:
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 429!
```

---

## 11. Mocks — LLM-assisted authoring

### 11.1 Hosted Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml \
  --output mocks/tasks_enriched.yaml
```

### 11.2 Local LM Studio (no API key)

```bash
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml \
  --output mocks/tasks_enriched.yaml
```

### 11.3 Skip one of the two passes

```bash
# Only fill missing examples
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml \
  --output mocks/tasks_examples_only.yaml \
  --no-draft-errors

# Only draft 4xx/5xx error envelopes
poetry run python main.py mock-enrich \
  --config mocks/tasks.yaml \
  --output mocks/tasks_errors_only.yaml \
  --no-fill-examples
```

---

## 12. Cloud upload (Azure / S3)

> Credentials come from env vars only — never config or CLI args.

### 12.1 Azure Blob Storage

```bash
# Set credentials FIRST (Linux / macOS):
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
# OR:
export AZURE_STORAGE_ACCOUNT="myaccount"
export AZURE_STORAGE_KEY="your-key"

# Generate + upload in one shot:
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_for_azure \
  --default-records 500 --seed 7 \
  --upload-to azure://my-container/runs/2026-05-11
```

PowerShell equivalent:
```powershell
$env:AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
poetry run python main.py generate `
  --config examples/configs/yaml/02_ecommerce_relationships.yaml `
  --output output/02_for_azure `
  --default-records 500 --seed 7 `
  --upload-to azure://my-container/runs/2026-05-11
```

### 12.2 AWS S3

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="eu-west-1"

poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_for_s3 \
  --default-records 500 --seed 7 \
  --upload-to s3://my-bucket/synthetic/run_01
```

### 12.3 Upload an existing directory (delta / SCD2 / saved snapshot)

The Streamlit UI has a "different local directory" upload option for this.
For the CLI, generate into the target dir then re-run with the same output
+ `--upload-to`, or use the `upload_output` Python helper directly.

---

## 13. Streamlit UI

### 13.1 Launch

```bash
poetry install --extras ui     # one-time
poetry run streamlit run ui/streamlit_app.py
# Opens http://localhost:8501
```

### 13.2 Pages in the sidebar

- **streamlit_app** (entry) — Generate Data + Quality Report + Cloud Upload
- **API Mocks** — OpenAPI / Postman / HAR → WireMock / JSON / Pact / Postman / OpenAPI examples

### 13.3 With cloud-upload credentials pre-set (Azure)

```bash
export AZURE_STORAGE_CONNECTION_STRING="..."
poetry run streamlit run ui/streamlit_app.py
```

Then in the UI, generate as usual and use Section 7's "Upload to cloud."

---

## 14. MCP server

### 14.1 Run the server (stdio, default)

```bash
poetry install --extras mcp                 # one-time
poetry run python -m mcp_server.server
```

### 14.2 With LM Studio routing (so MCP tools that use an LLM also use LM Studio)

```bash
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
poetry run python -m mcp_server.server
```

### 14.3 Wire into LM Studio (`mcp.json`)

```json
{
  "mcpServers": {
    "synthetic-data-platform": {
      "command": "C:/Users/natar/TestDataGeneration/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/natar/TestDataGeneration",
      "env": {
        "SDP_LLM_PROVIDER": "lm-studio",
        "SDP_LLM_MODEL": "meta-llama-3.1-8b-instruct",
        "SDP_LLM_BASE_URL": "http://localhost:1234/v1"
      }
    }
  }
}
```

### 14.4 Wire into Claude Desktop

Same JSON, in `claude_desktop_config.json` (path varies by OS — see
`MCP_Integration.md` §8).

### 14.5 Try it (in any MCP-aware client chat)

> "List the synthetic-data examples this platform ships with."
>
> "Generate 500 rows from `examples/configs/yaml/02_ecommerce_relationships.yaml`
>  with seed 7 into `output/from_mcp`, then validate it with Great Expectations."
>
> "Diagnose the LLM connection."

---

## 15. Docker — every command above, in a container

Build once:

```bash
docker build -t sdp:latest .
```

The image is multi-purpose. Substitute every `poetry run python main.py <cmd>` above with:

```bash
docker run --rm -v "$PWD:/work" sdp:latest <cmd>
```

…and replace `examples/...` and `output/...` with `/work/examples/...` and `/work/output/...`.

### 15.1 Generate (Docker)

```bash
docker run --rm -v "$PWD:/work" sdp:latest \
  generate --config /work/examples/configs/yaml/01_simple_users.yaml \
           --output /work/output/01_docker --default-records 200 --seed 42
```

### 15.2 Streamlit UI (Docker)

```bash
# Linux / macOS
docker run --rm -p 8501:8501 -v "$PWD:/work" sdp:latest streamlit

# Windows PowerShell
docker run --rm -p 8501:8501 -v "${PWD}:/work" sdp:latest streamlit
```

Open http://localhost:8501.

### 15.3 MCP server (Docker)

```bash
docker run --rm -i -v "$PWD:/work" sdp:latest mcp
```

### 15.4 Quality report (Docker)

```bash
docker run --rm -v "$PWD:/work" sdp:latest \
  quality-report --generated /work/output/02_ecommerce \
                 --output-html /work/output/02_ecommerce/quality.html
```

### 15.5 Cloud upload from Docker (env vars passed through)

```bash
docker run --rm -v "$PWD:/work" \
  -e AZURE_STORAGE_CONNECTION_STRING="DefaultEndpoints..." \
  sdp:latest \
  generate --config /work/examples/configs/yaml/02_ecommerce_relationships.yaml \
           --output /work/output/from_docker \
           --default-records 500 --seed 7 \
           --upload-to azure://my-container/from-docker
```

### 15.6 docker-compose shorthand

```bash
# CLI as a one-shot
docker compose run --rm cli \
  generate --config /work/examples/configs/yaml/02_ecommerce_relationships.yaml \
           --output /work/output/from_compose \
           --default-records 500 --seed 7

# Streamlit on :8501
docker compose up streamlit
# Ctrl-C to stop
```

### 15.7 Drop into a shell inside the image

```bash
docker run --rm -it -v "$PWD:/work" sdp:latest shell
```

---

## 16. Inspect outputs

### 16.1 Read a Parquet directory

```bash
poetry run python readParquet.py output/02_ecommerce
```

### 16.2 Read a Delta Lake table

```bash
poetry run python -c "
from deltalake import DeltaTable
dt = DeltaTable('output/07_cdc/delta/account_balances')
df = dt.to_pandas()
print(df['_operation'].value_counts())
print(df.head())
"
```

### 16.3 Inspect SCD2 versions

```bash
poetry run python -c "
import pyarrow.parquet as pq
import pandas as pd
df = pq.read_table('output/08_scd2/history/customers.parquet').to_pandas()
print(df[['customer_id', 'version_num', 'is_current', 'effective_from_ts', 'effective_to_ts']].head(20))
print('Versions per customer:', df.groupby('customer_id').size().describe())
"
```

### 16.4 Quick column-by-column glance at a generated row

```bash
poetry run python -c "
import pyarrow.parquet as pq
df = pq.read_table('output/03_special_rules/identity_showcase.parquet').to_pandas()
print(df.iloc[0].to_string())
"
```

---

## 17. ER diagrams + Collibra

### 17.1 Mermaid ER diagram (no extra deps)

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_for_diagram \
  --default-records 100 --seed 7 \
  --er-diagram --er-format mermaid \
  --er-output output/02_for_diagram/diagram
```

### 17.2 Graphviz DOT + PNG

```bash
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_for_diagram \
  --default-records 100 --seed 7 \
  --er-diagram --er-format mermaid dot png \
  --er-output output/02_for_diagram/diagram
```

> PNG needs `pip install matplotlib`.

### 17.3 Standalone diagram via the inferrer (no generation)

```bash
poetry run python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_inferred.yaml \
  --er-output     output/11_inferred.mmd \
  --method ml
```

### 17.4 Collibra — pull a dataset definition into a YAML config

```bash
export COLLIBRA_BASE_URL=https://your-collibra.example.com
export COLLIBRA_USERNAME=you
export COLLIBRA_PASSWORD=...

poetry run python main.py collibra-import \
  --dataset "Account Booking" \
  --domain  "Finance" \
  --output  config/from_collibra.yaml
```

---

## 18. End-to-end demo arc (10 minutes)

Run these in order; each takes < 2 minutes.

```bash
# 1. Single table — basics (45s)
poetry run python main.py generate \
  --config examples/configs/yaml/01_simple_users.yaml \
  --output output/demo01 --default-records 200 --seed 42

# 2. Multi-table with FKs (1m)
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/demo02 --default-records 500 --seed 7 --validate

# 3. Special rules + locales (1m)
poetry run python main.py generate \
  --config examples/configs/yaml/03_special_rules_showcase.yaml \
  --output output/demo03 --default-records 50 --seed 1

# 4. Relationship inference — ML, free (2m)
poetry run python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/demo04_inferred.yaml \
  --er-output     output/demo04.mmd \
  --method ml

# 5. Rules + derived columns (1m)
poetry run python main.py generate \
  --config examples/configs/yaml/06_rules_and_derived.yaml \
  --output output/demo05 --default-records 200 --seed 21

# 6. CDC delta (1.5m) — three commands
poetry run python main.py generate --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
  --output output/demo06/snap_v1 --default-records 1000 --seed 1
poetry run python main.py generate --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
  --output output/demo06/snap_v2 --default-records 1000 --seed 2
poetry run python main.py delta --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
  --previous output/demo06/snap_v1 --current output/demo06/snap_v2 \
  --output   output/demo06/delta

# 7. Mocks: OpenAPI → WireMock + Postman (2m)
poetry run python main.py mock-init \
  --from examples/openapi/medium_tasks.yaml --output mocks/demo_tasks.yaml
poetry run python main.py mock-render \
  --config mocks/demo_tasks.yaml --output stubs/demo_tasks \
  --format wiremock,postman --examples 5 --seed 42 --match-mode any

# 8. Stateful scenarios (1m) — Java + WireMock JAR required
poetry run python main.py mock-render \
  --config examples/mocks/accounts_with_scenarios.yaml \
  --output stubs/demo_accounts --format wiremock --match-mode any --seed 42
# java -jar ~/tools/wiremock.jar --root-dir stubs/demo_accounts/wiremock --port 8080
# curl http://localhost:8080/api/v1/accounts (4 times — fourth call returns 429)

# 9. Validation (1m) — schema + statistical
poetry run python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/demo09 --default-records 500 --seed 7 \
  --validate --validate-with-gx
poetry run python main.py quality-report \
  --generated output/demo09 \
  --output-html output/demo09/quality.html

# 10. (Optional headline) MCP via LM Studio — open LM Studio, ask its model:
#   "List the synthetic-data examples this platform ships with."
#   "Generate 500 rows from the e-commerce one and validate it."
```

---

## Cheat sheet — flags you'll use most

| Flag | Where | What it does |
|---|---|---|
| `--seed N` | most subcommands | Byte-identical output across runs |
| `--default-records N` | `generate` | Per-table default (also `--records table:N`) |
| `--validate` | `generate` | Post-generation FK integrity check |
| `--validate-with-gx` | `generate` | Run Great Expectations |
| `--gx-fail-on-error` | `generate` | Exit non-zero on GX failure |
| `--er-diagram` + `--er-format` | `generate` | Emit ER diagram (mermaid / dot / png) |
| `--upload-to` | `generate` | Push output to `azure://...` or `s3://...` |
| `--stream` + `--chunk-size` | `generate` | Streaming generation for large runs |
| `--method` | `infer-relationships` | `ml` / `llm` / `both` |
| `--ml-confidence` / `--llm-confidence` | inference | Threshold filters |
| `--match-mode` | `mock-render` | `concrete` (literal paths) or `any` (regex) |
| `--format` | `mock-render` | `wiremock,json,pact,postman,openapi-examples` (comma-sep) |
| `--report-json` | `validate-data` | Machine-readable validation report |
| `--privacy-threshold` | `quality-report` | NN-distance for the privacy proxy |
| `--no-fill-examples` / `--no-draft-errors` | `mock-enrich` | Skip one of the two LLM passes |

---

## Env vars cheat sheet

| Env var | Used by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | LLM features (default provider) | Hosted Claude |
| `SDP_LLM_PROVIDER` | LLM features | `anthropic` / `openai` / `lm-studio` / `ollama` / `azure-openai` / `groq` / `together` / `openrouter` |
| `SDP_LLM_MODEL` | LLM features | Provider-specific model id |
| `SDP_LLM_BASE_URL` | LLM features | Override base URL (custom LM Studio port etc.) |
| `SDP_FEEDBACK_PATH` | ML inferrer | Override feedback-store path |
| `AZURE_STORAGE_CONNECTION_STRING` | cloud upload | Preferred Azure auth |
| `AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_KEY` | cloud upload | Azure auth (alternative) |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | cloud upload | S3 auth |
| `AWS_DEFAULT_REGION` | cloud upload | S3 region |
| `COLLIBRA_BASE_URL` + `COLLIBRA_USERNAME` + `COLLIBRA_PASSWORD` | `collibra-import` | Collibra auth |

---

*If a command in this doc doesn't work, the bundled example may have moved.
Run `python main.py --help` for the live argparse.
For deeper dives:* `Examples_Walkthrough.md` *(per-example narrative),*
`Bruno_Workflow.md` *(mocks-into-Bruno),* `Data_Validation.md`,
`Quality_Reports.md`, `MCP_Integration.md`, `Docker_Quickstart.md`.
