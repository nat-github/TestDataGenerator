# Examples Walkthrough — Demo Playbook

End-to-end demo script for the Synthetic Data Platform. Every feature has a
working example you can run, an expected output description, and a few
talking points to anchor a stakeholder demo.

> All examples live under `examples/configs/yaml/` (canonical), with mirrored
> JSON variants under `examples/configs/json/` and Excel variants under
> `examples/configs/xlsx/` (where Excel can represent the feature cleanly).
>
> The XLSX and JSON variants are auto-generated from the YAML by:
>     `poetry run python examples/configs/generate_variants.py`
> Source of truth = YAML.

---

## Format coverage matrix

| # | Example | YAML | JSON | XLSX | Why |
|---|---|:---:|:---:|:---:|---|
| 01 | Simple users (single table) | ✓ | ✓ | ✓ | Fits cleanly in all three. |
| 02 | E-commerce (3 tables, FKs) | ✓ | ✓ | ✓ | Composite keys still fit XLSX semicolon-list. |
| 03 | Special-rules showcase | ✓ | ✓ | ✓ | Pure column-level rules. |
| 04 | Distributions + business values | ✓ | ✓ | ✓ | Distributions live in `distribution:` block — XLSX has a column for it. |
| 05 | Multi-locale (5 tables) | ✓ | ✓ | ✓ | Same shape as 03, just more tables. |
| 06 | Rules + derived columns | ✓ | ✓ | — | when/then rules and derived expressions don't map to flat sheets. |
| 07 | CDC delta workflow | ✓ | ✓ | — | The unified `cdc:` block has nested fields. Use the legacy XLSX columns (`cdc_mode`, `cdc_track`) instead if Excel is required. |
| 08 | SCD2 history | ✓ | ✓ | — | Same as 07. |
| 09 | PII columns | ✓ | ✓ | ✓ | Same shape as 03. |
| 10 | Bare for LLM enrichment | ✓ | ✓ | ✓ | Deliberately sparse — perfect Excel. |
| 11 | Bare for relationship inference | ✓ | ✓ | — | Has composite PKs across tables; YAML/JSON cleaner here. |

---

## Example 01 — Simple single-table snapshot

**File:** `examples/configs/yaml/01_simple_users.yaml` (+ JSON, XLSX)

**Demonstrates:** the minimum viable config. Faker-backed special_rules,
business values as enum, regex generation, locale support.

```bash
python main.py generate \
  --config examples/configs/yaml/01_simple_users.yaml \
  --output output/01_simple_users \
  --default-records 200 --seed 42

# inspect
poetry run python readParquet.py output/01_simple_users
```

**Talking points:**
- One YAML, eight columns, 200 realistic rows in under a second.
- Each column type is generated differently: Faker handles names/emails/phones,
  regex generator handles `[A-Z]{2}` for country code, business values pick
  randomly from the enum, integer/decimal use min/max bounds.
- `--seed 42` makes this reproducible — re-running produces identical output.

---

## Example 02 — Multi-table with referential integrity

**File:** `02_ecommerce_relationships.yaml`

**Demonstrates:** FK relationships across three tables (customers → orders →
order_items), composite primary keys, post-generation FK resolution.

```bash
python main.py generate \
  --config examples/configs/yaml/02_ecommerce_relationships.yaml \
  --output output/02_ecommerce \
  --default-records 500 --seed 7 --validate

# --validate runs the post-generation FK validator
```

**Talking points:**
- Generated in topological order: customers first, then orders (with valid
  customer_ids), then order_items (with valid order_ids).
- `--validate` confirms every child FK references a real parent — the
  platform's referential integrity guarantee is verifiable.
- `customer_id` in `orders` only takes values that exist in `customers`.

---

## Example 03 — Special rules showcase

**File:** `03_special_rules_showcase.yaml`

**Demonstrates:** the breadth of built-in `special_rules` — banking IDs
(IBAN, SWIFT, US/UK/IN routing), national IDs (SSN, PAN, AADHAAR, CPF),
tax/VAT, network (IPv4/v6, MAC, UUID), healthcare (NHS), product codes
(EAN13, ISBN13), crypto (BTC, ETH), plus locale variants and custom regex.

```bash
python main.py generate \
  --config examples/configs/yaml/03_special_rules_showcase.yaml \
  --output output/03_special_rules \
  --default-records 50 --seed 1

# Open one row in pandas to scan column-by-column
poetry run python -c "
import pyarrow.parquet as pq
df = pq.read_table('output/03_special_rules/identity_showcase.parquet').to_pandas()
print(df.iloc[0].to_string())
"
```

**Talking points:**
- 60+ rules built in, each producing format-correct values out of the box.
- Locale suffix syntax: `NAME:de_DE`, `IBAN:DE` — 18 supported locales.
- `GLOBAL_NAME` rolls a random locale per row — useful for international
  test data without maintaining 18 separate columns.
- Add Mimesis (`poetry install --extras mimesis`) to unlock `MIMESIS_*` rules.

---

## Example 04 — Distributions and business values

**File:** `04_distributions_and_business_values.yaml`

**Demonstrates:** statistical distributions on numeric columns (normal for
amounts, uniform for ratings), enum business values, sparse columns via
`null_rate`.

```bash
python main.py generate \
  --config examples/configs/yaml/04_distributions_and_business_values.yaml \
  --output output/04_distributions \
  --default-records 1000 --seed 11

poetry run python -c "
import pyarrow.parquet as pq
df = pq.read_table('output/04_distributions/transactions.parquet').to_pandas()
print('amount mean:', df['amount'].mean(), 'std:', df['amount'].std())
print('memo nulls:', df['memo'].isna().mean())
print('currency distribution:', df['currency'].value_counts(normalize=True).to_dict())
"
```

**Talking points:**
- The amount column should center near €1200 with ~€450 stddev — that's the
  `distribution: { name: norm, params: [1200.0, 450.0] }` block kicking in.
- `memo` is null ~90% of the time — realistic for an optional field.
- This is what makes the data *plausible*, not just structurally correct.

---

## Example 05 — Multi-locale data

**File:** `05_multi_locale.yaml`

**Demonstrates:** five locales side-by-side (US, DE, FR, JP, IN) producing
locale-correct names, addresses, phones, banking IDs, national IDs.

```bash
python main.py generate \
  --config examples/configs/yaml/05_multi_locale.yaml \
  --output output/05_multi_locale \
  --default-records 100 --seed 3
```

**Talking points:**
- Compare the `name` columns across the five tables — they're written in the
  appropriate script (Latin / Japanese / Hindi-aware Latin).
- IBANs are country-correct: `DE...` for Germany, `FR...` for France.
- US users have SSNs; UK users would have NI numbers; Indian users have PAN +
  AADHAAR + IFSC. All format-validated.

---

## Example 06 — Rules and derived columns

**File:** `06_rules_and_derived.yaml`

**Demonstrates:** Layer A (when/then rules) and Layer B (derived columns)
running after FK resolution.

```bash
python main.py generate \
  --config examples/configs/yaml/06_rules_and_derived.yaml \
  --output output/06_rules \
  --default-records 200 --seed 21

poetry run python -c "
import pyarrow.parquet as pq
df = pq.read_table('output/06_rules/accounts.parquet').to_pandas()
# closure_date should be NaT where status=ACTIVE, set where status=CLOSED
print(df.groupby('status')['closure_date'].agg(['count', lambda s: s.isna().sum()]))
# full_name should be 'first_name last_name'
print(df[['first_name', 'last_name', 'full_name']].head())
# overdraft_flag composite condition
print(df.groupby('overdraft_flag')[['status', 'balance']].describe())
"
```

**Talking points:**
- `closure_date` is null for ACTIVE accounts, set to a fixed date for CLOSED,
  freely generated for PENDING — three branches, declarative.
- `full_name` is `{first_name} {last_name}` for every row — derived
  expressions resolve placeholders against other columns.
- `overdraft_flag` uses an `and` composite: only "Y" when balance < 0 AND
  status = ACTIVE — the rule engine supports nested `and`/`or` operators.

---

## Example 07 — CDC delta workflow

**File:** `07_cdc_delta_workflow.yaml`

**Demonstrates:** snapshot v1 → snapshot v2 → delta computation produces a
Delta Lake table with insert/update/delete operations.

```bash
# 1. Snapshot v1
python main.py generate --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
    --output output/07_cdc/snap_v1 --default-records 1000 --seed 1

# 2. Snapshot v2 (different seed → some rows added, some changed, some dropped)
python main.py generate --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
    --output output/07_cdc/snap_v2 --default-records 1000 --seed 2

# 3. Compute the delta
python main.py delta --config examples/configs/yaml/07_cdc_delta_workflow.yaml \
    --previous output/07_cdc/snap_v1 \
    --current  output/07_cdc/snap_v2 \
    --output   output/07_cdc/delta

# 4. Inspect with the deltalake reader
poetry run python -c "
from deltalake import DeltaTable
dt = DeltaTable('output/07_cdc/delta/account_balances')
df = dt.to_pandas()
print(df['_operation'].value_counts())
print(df.head())
"
```

**Talking points:**
- Output is a real Delta Lake table — Spark, Polars, deltalake-py, Trino,
  and Databricks can all read it.
- `_operation` column distinguishes I (insert), U (update), D (delete) rows.
- Partitioning by `posting_date` is preserved across delta runs.
- The business_key_columns (`account_id`) drives the I/U/D decision.

---

## Example 08 — SCD2 history

**File:** `08_scd2_history.yaml`

**Demonstrates:** SCD2 (slowly-changing dimension type 2) — when tracked
columns change, a new versioned row is created with effective dating.

```bash
# 1. Baseline
python main.py generate --config examples/configs/yaml/08_scd2_history.yaml \
    --output output/08_scd2/snap_v1 --default-records 500 --seed 1

# 2. Updated state
python main.py generate --config examples/configs/yaml/08_scd2_history.yaml \
    --output output/08_scd2/snap_v2 --default-records 500 --seed 2

# 3. Build SCD2 history
python main.py scd2 --config examples/configs/yaml/08_scd2_history.yaml \
    --previous output/08_scd2/snap_v1 \
    --current  output/08_scd2/snap_v2 \
    --output   output/08_scd2/history
```

**Talking points:**
- The `track:` list (`tier`, `address`) is the only thing that triggers a
  new version. A `name` change alone keeps the same version.
- Each row gets `effective_from_ts`, `effective_to_ts`, `is_current`,
  `version_num` — standard SCD2 effective-dating columns.
- Versioning happens per-business-key (`customer_id`) so you can replay any
  customer's full history with one query.

---

## Example 09 — PII detection

**File:** `09_pii_columns.yaml`

**Demonstrates:** the `pii-scan` command scans both column names and sample
values to identify columns holding sensitive information.

```bash
# 1. Generate the data
python main.py generate --config examples/configs/yaml/09_pii_columns.yaml \
    --output output/09_pii --default-records 200 --seed 4

# 2. Scan it
python main.py pii-scan --input output/09_pii --verbose
```

**Talking points:**
- The scanner classifies columns by category (PERSON, EMAIL, PHONE, ADDRESS,
  GOVERNMENT_ID, FINANCIAL, etc.).
- It uses both column-name heuristics and value pattern matching, so a
  column called `email_address` with values like `ada@example.com` is doubly
  confirmed.
- Compliance angle: for any dataset, you can answer "where's the PII?" in
  one command.

---

## Example 10 — LLM schema enrichment

**File:** `10_bare_for_llm_enrichment.yaml`

**Demonstrates:** an LLM (hosted Anthropic, OpenAI, or local LM Studio /
Ollama) suggests `business_values`, `special_rules`, and `data_type`
corrections for an otherwise-bare config.

```bash
# Hosted Anthropic (default — needs ANTHROPIC_API_KEY)
python main.py enrich \
  --config examples/configs/yaml/10_bare_for_llm_enrichment.yaml \
  --output output/10_enriched.yaml --confidence 0.7

# OR local LM Studio (no API key)
SDP_LLM_PROVIDER=lm-studio \
SDP_LLM_MODEL="meta-llama-3.1-8b-instruct" \
python main.py enrich \
  --config examples/configs/yaml/10_bare_for_llm_enrichment.yaml \
  --output output/10_enriched.yaml --confidence 0.6

# Diff input vs output
diff examples/configs/yaml/10_bare_for_llm_enrichment.yaml output/10_enriched.yaml
```

**Talking points:**
- Bare input → enriched output. Expected suggestions:
  - `country_code` → ISO list / COUNTRY_CODE special_rule
  - `currency_code` → currency enum
  - `email` → EMAIL
  - `iban`-shaped column → IBAN
  - `passport_no` → PASSPORT
  - `status` → likely an enum like ACTIVE/CLOSED
- The provider abstraction means you can demo this against a free local LLM
  without any API key.
- Suggestions below `--confidence` threshold are dropped — only the
  high-confidence ones are applied.

---

## Example 11 — Relationship inference (ML + LLM)

**File:** `11_bare_for_relationship_inference.yaml`

**Demonstrates:** four tables, no `relationships:` block. Both the ML
heuristic inferrer and the LLM inferrer should propose the same set of
foreign keys.

```bash
# ML inferrer (free, deterministic, default)
python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_inferred.yaml \
  --er-output     output/11_inferred.mmd \
  --method ml

# LLM inferrer
SDP_LLM_PROVIDER=lm-studio python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_llm.yaml --method llm

# Both — ML first, LLM fills gaps
python main.py infer-relationships \
  --config examples/configs/yaml/11_bare_for_relationship_inference.yaml \
  --config-output output/11_both.yaml --method both

# Closing the feedback loop:
# 1. Open output/11_inferred.yaml in your editor
# 2. Delete entries you disagree with, edit columns, leave correct ones alone
# 3. Save as output/11_reviewed.yaml
# 4. Feed your edits back into the adaptive classifier:
python main.py record-feedback \
  --inferred output/11_inferred.yaml \
  --reviewed output/11_reviewed.yaml
```

**Talking points:**
- The ML inferrer reports its four signals (name_similarity, value_subset,
  type_compatibility, pk_likeness) so reviewers can audit decisions.
- After ~30 feedback decisions, the adaptive classifier activates and starts
  overriding the heuristic. The system gets better with use.
- ER diagram (`output/11_inferred.mmd`) is Mermaid — paste into any Markdown
  renderer for a visual summary.

---

## Mocks track — try it offline

The mocks track has its own example specs at `examples/openapi/` (three
real-shape OpenAPI 3 specs) plus supporting fixtures at `examples/mocks/`.

### Mock 01 — Simple OpenAPI to WireMock + Bruno

```bash
# 1. Convert OpenAPI → editable sdp-mock-v1
python main.py mock-init \
  --from examples/openapi/simple_books.yaml \
  --output mocks/books.yaml

# 2. Render WireMock + JSON fixtures
python main.py mock-render \
  --config mocks/books.yaml \
  --output stubs/books \
  --format wiremock,json --examples 5 --seed 42 --match-mode any

# 3. Run WireMock standalone
java -jar ~/tools/wiremock.jar --root-dir stubs/books/wiremock --port 8080

# 4. From Bruno: import examples/openapi/simple_books.yaml as a collection,
#    point baseUrl at http://localhost:8080/api/v1, click Send.
```

### Mock 02 — Multiple output formats from one spec

```bash
python main.py mock-render \
  --config mocks/books.yaml \
  --output stubs/all \
  --format wiremock,json,pact,postman,openapi-examples \
  --openapi-source examples/openapi/simple_books.yaml \
  --pact-consumer reader-app --pact-provider books-svc \
  --examples 3 --seed 1
```

Output structure:
```
stubs/all/
├── wiremock/         # WireMock mappings
├── json/             # standalone JSON response/request fixtures
├── pact/             # Pact v3 contract files
├── postman/          # Postman v2.1 collection JSON
└── openapi-examples/ # source spec round-tripped with example: blocks injected
```

### Mock 03 — Stateful scenarios

**File:** `examples/mocks/accounts_with_scenarios.yaml`

```bash
python main.py mock-render \
  --config examples/mocks/accounts_with_scenarios.yaml \
  --output stubs/accounts \
  --format wiremock --examples 3 --seed 42 --match-mode any

java -jar ~/tools/wiremock.jar --root-dir stubs/accounts/wiremock --port 8080

# Demonstrate the scenario — fourth call rate-limits:
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 200
curl http://localhost:8080/api/v1/accounts  # 429!
```

### Mock 04 — Reverse import: Postman collection → mocks

```bash
# Auto-detected as Postman:
python main.py mock-init \
  --from examples/mocks/sample_postman_collection.json \
  --output mocks/from_postman.yaml

python main.py mock-render \
  --config mocks/from_postman.yaml \
  --output stubs/from_postman --format wiremock --seed 1
```

### Mock 05 — Reverse import: HAR capture → mocks

```bash
# Auto-detected as HAR:
python main.py mock-init \
  --from examples/mocks/sample_session.har \
  --output mocks/from_har.yaml

# Note how /v1/users/42 + /v1/users/137 collapsed into /v1/users/{id}
python main.py mock-render \
  --config mocks/from_har.yaml \
  --output stubs/from_har --format wiremock --match-mode any --seed 1
```

### Mock 06 — LLM-assisted authoring

```bash
# Locally via LM Studio:
SDP_LLM_PROVIDER=lm-studio \
SDP_LLM_MODEL="meta-llama-3.1-8b-instruct" \
python main.py mock-enrich \
  --config mocks/books.yaml \
  --output mocks/books_enriched.yaml

# Diff to see what the LLM filled in
diff mocks/books.yaml mocks/books_enriched.yaml
```

---

## Suggested 10-minute demo arc

For a stakeholder demo, this is the order I'd run things in:

1. **Example 01** (45 s) — "one config, real data, one second."
2. **Example 02 + `--validate`** (1 m) — "with relationships, the data stays
   referentially honest."
3. **Example 03** (1 m) — "60+ rules cover banking, IDs, network, healthcare."
4. **Example 11** (2 m) — "And the platform infers relationships *for* you,
   either heuristically (free) or with an LLM."
5. **Example 06** (1 m) — "And business rules. Status determines closure
   date. Derived columns compose values."
6. **Example 07** (1.5 m) — "Two snapshots, one delta, real Delta Lake."
7. **Mock 01 + Bruno** (2 m) — "Same platform produces API mocks. Bruno
   hits a real WireMock pointed at our generated stubs."
8. **Mock 03 scenarios** (1 m) — "Stateful: rate limit kicks in on the
   fourth call. Pure config."

Total ~10 minutes, every major capability shown working.
