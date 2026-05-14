# Data Generation Execution Guide (Parquet + Delta + SCD2 + AI/ML/LLM)

This is a terminal-first runbook for the **data generation track**.

- Repo root: `/Users/natarajankanakasabapathy/FECTECH/TestDataGenerator`
- Baseline config used here: `config/Acct_bkng.xlsx`
- Relationship-inference demo config: `config/Creditcard_no_rel.xlsx`

---

## 0) Memory Flow

**L -> G -> D -> S**

- **L**int config
- **G**enerate snapshots (`v1`, `v2`)
- **D**elta between snapshots
- **S**CD2 history from snapshots

For relationship inference + LLM:

**I -> R -> F**

- **I**nfer relationships
- **R**eview inferred YAML
- **F**eedback record (optional but recommended)

---

## 1) One-Time Setup

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry install
```

Optional extras (only if you need these features):

```bash
pip install matplotlib
poetry install --extras mimesis
```

---

## 2) Baseline Lint + Snapshot Generation

### Step 1: Lint config

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py lint --config config/Acct_bkng.xlsx
```

### Step 2: Generate snapshot v1

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/Acct_bkng.xlsx --output output/snap_v1 --default-records 1000 --seed 42
```

### Step 3: Generate snapshot v2 (different seed so data changes)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/Acct_bkng.xlsx --output output/snap_v2 --default-records 1000 --seed 43
```

### Step 4: Inspect parquet output (optional)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python readParquet.py output/snap_v1
poetry run python readParquet.py output/snap_v2
```

---

## 3) Parquet Delta Execution

### Step 1: Run delta between snapshots

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py delta --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/delta_run
```

### Step 2: (Optional) Delta for selected tables only

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py delta --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/delta_run_subset --tables df_cac_acg_entr df_cash_bookg
```

### Step 3: (Optional) Override partition column

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py delta --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/delta_run_partitioned --partition-column batch_partition_date
```

---

## 4) SCD2 Execution

### Step 1: Build SCD2 history

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py scd2 --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/scd2_run
```

### Step 2: Inspect SCD2 output (optional)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python readParquet.py output/scd2_run
```

Expected SCD2 metadata columns include fields like:
- `effective_from_ts`
- `effective_to_ts`
- `is_current`
- `version_num`

---

## 5) AI/ML Relationship Inference (No API key required)

### Step 1: Generate with inferred relationships (ML default)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/Creditcard_no_rel.xlsx --output output/infer_ml_generate --infer-relationships --method ml --ml-confidence 0.55 --seed 42
```

### Step 2: Standalone infer (reviewable YAML + ER diagram)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py infer-relationships --config config/Creditcard_no_rel.xlsx --config-output config/inferred_creditcard.yaml --er-output diagrams/inferred_creditcard.mmd
```

### Step 3: Review + edit inferred YAML

- Open `config/inferred_creditcard.yaml`
- Keep/remove/add relationships as SME review output
- Save reviewed file as: `config/inferred_creditcard.reviewed.yaml`

### Step 4: Record feedback for adaptive learning

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py record-feedback --inferred config/inferred_creditcard.yaml --reviewed config/inferred_creditcard.reviewed.yaml
```

---

## 6) LLM Relationship Inference + LLM Data Features

## 6.1 Export LM Studio environment

```bash
export SDP_LLM_PROVIDER="lm-studio"
export SDP_LLM_BASE_URL="http://localhost:1234/v1"
export SDP_LLM_MODEL="google/gemma-4-e4b"
```

## 6.2 Generate with LLM relationship inference

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/Creditcard_no_rel.xlsx --output output/infer_llm_generate --infer-relationships --method llm --llm-confidence 0.7 --seed 42
```

## 6.3 Generate with combined inference (ML first, LLM for misses)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/Creditcard_no_rel.xlsx --output output/infer_both_generate --infer-relationships --method both --ml-confidence 0.55 --llm-confidence 0.7 --seed 42
```

## 6.4 LLM schema enrichment (business values/rules/type hints)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py enrich --config config/Creditcard_no_rel.xlsx --output config/creditcard_enriched.yaml --confidence 0.7
```

## 6.5 Generate from enriched config

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/creditcard_enriched.yaml --output output/enriched_generate --seed 42
```

---

## 7) Recommended End-to-End Sequence (Copy/Paste)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry install

poetry run python main.py lint --config config/Acct_bkng.xlsx
poetry run python main.py generate --config config/Acct_bkng.xlsx --output output/snap_v1 --default-records 1000 --seed 42
poetry run python main.py generate --config config/Acct_bkng.xlsx --output output/snap_v2 --default-records 1000 --seed 43

poetry run python main.py delta --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/delta_run
poetry run python main.py scd2 --config config/Acct_bkng.xlsx --previous output/snap_v1 --current output/snap_v2 --output output/scd2_run

export SDP_LLM_PROVIDER="lm-studio"
export SDP_LLM_BASE_URL="http://localhost:1234/v1"
export SDP_LLM_MODEL="google/gemma-4-e4b"

poetry run python main.py infer-relationships --config config/Creditcard_no_rel.xlsx --config-output config/inferred_creditcard.yaml --er-output diagrams/inferred_creditcard.mmd
poetry run python main.py enrich --config config/Creditcard_no_rel.xlsx --output config/creditcard_enriched.yaml --confidence 0.7
poetry run python main.py generate --config config/creditcard_enriched.yaml --output output/enriched_generate --seed 42
```

---

## 8) ER Generation (Separate Section)

### 8.1 ER during standard `generate`

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py generate --config config/Acct_bkng.xlsx --output output/run_with_er --er-diagram --er-format mermaid dot
```

This writes ER artifacts for the run in Mermaid and DOT formats.

### 8.2 ER as PNG (optional)

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
pip install matplotlib
poetry run python main.py generate --config config/Acct_bkng.xlsx --output output/run_with_er_png --er-diagram --er-format png
```

### 8.3 ER from relationship inference workflow

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run python main.py infer-relationships --config config/Creditcard_no_rel.xlsx --config-output config/inferred_creditcard.yaml --er-output diagrams/inferred_creditcard.mmd
```

---

## 9) Troubleshooting

### `infer-relationships --method llm` fails with API-key/provider errors

- Confirm env vars are exported in the same terminal session:
  - `SDP_LLM_PROVIDER`
  - `SDP_LLM_BASE_URL`
  - `SDP_LLM_MODEL`
- Confirm LM Studio local server is running on `http://localhost:1234/v1`

### Delta/SCD2 command fails due missing snapshots

- Ensure `output/snap_v1` and `output/snap_v2` exist and contain parquet files.

### SCD2 produces no version changes

- Verify tracked columns are configured in workbook/YAML (`scd2_tracked_columns` or unified `cdc.track`).

### Low-quality inferred relationships

- Run `record-feedback` repeatedly with SME-reviewed files; classifier improves after enough labels.

---

## 10) Useful Test Commands

```bash
cd /Users/natarajankanakasabapathy/FECTECH/TestDataGenerator
poetry run pytest tests/test_rules_and_cdc.py -v
poetry run pytest tests/test_relationship_inferrer.py -v
poetry run pytest tests/test_cli_relationship_inference.py -v
poetry run pytest tests/test_config_and_parquet_flows.py -v
```
