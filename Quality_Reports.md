# Synthetic-Data Quality Reports

A statistical fidelity / privacy report on generated data. Two modes:

- **Univariate-only** (no source data) — per-column dtype, null rate, unique
  count, summary stats; per-table Pearson correlation matrix. Always
  available. Answers *"does this synthetic data look plausible on its own?"*
- **Fidelity vs source** (source data provided) — adds KS test for numeric
  columns, total-variation distance for categoricals, correlation-matrix
  delta, and a nearest-neighbour distance privacy proxy. Answers *"is this
  synthetic data faithful to the source distribution while not leaking
  individual rows?"*

This complements Great Expectations: GX validates *schema conformance*,
quality reports validate *statistical fidelity*. Different questions, both
worth answering.

---

## How to run

### CLI — standalone

```bash
# Univariate-only (no source)
python main.py quality-report \
  --generated output/run_01 \
  --output-html quality.html \
  --output-json quality.json

# Fidelity vs source
python main.py quality-report \
  --generated output/run_01 \
  --source     data/real_sample \
  --output-html quality.html \
  --output-json quality.json \
  --verbose

# With a privacy threshold (NN-distance proxy)
python main.py quality-report \
  --generated output/run_01 --source data/real_sample \
  --privacy-threshold 1e-6 \
  --output-json quality.json
```

### Streamlit UI

After generating data in the UI, click **Generate quality report** in the
"Quality report" section. Optionally upload Parquet/CSV files for source
data — the filename without extension must match the synthetic table name
(e.g. `users.parquet` for the `users` table).

### MCP

```
quality_report(
    generated_dir="output/run_01",
    source_dir="data/real_sample",     # optional
    privacy_threshold=0.0,             # optional
)
```

Returns a structured dict with `markdown_summary`, `overall_fidelity`, and
per-table metrics. Agents can summarise the result in natural language.

### From Python

```python
from validators.quality_report import quality_report, quality_report_from_paths

# In-memory
report = quality_report({"users": users_df}, source={"users": real_df})
print(report.to_markdown())
print("Overall fidelity:", report.overall_fidelity)

# From disk
report = quality_report_from_paths("output/run_01", "data/real_sample")
report.to_html()  # self-contained HTML page
```

---

## Metrics — what they mean

### Univariate (every column, always)

| Metric | Description |
|---|---|
| `dtype` | Pandas dtype after generation |
| `null_count` / `null_rate` | Number / fraction of nulls |
| `unique_count` | Distinct non-null values |
| `mean` / `std` / `median` / `min` / `max` | Numeric only |
| `top_values` | Top 10 categorical values with counts |

### Cross-column (per table)

| Metric | Description |
|---|---|
| `correlation_synthetic` | Pearson correlation matrix over numeric columns |
| `correlation_distance` | Frobenius norm of (synthetic_corr − source_corr), scaled to [0, 1]. **0 = identical, 1 = maximally different.** Only when source has the same numeric columns. |

### Distribution fidelity (per column, only with source)

| Metric | Numeric? | Categorical? | Description |
|---|:---:|:---:|---|
| `ks_statistic` | yes | — | Kolmogorov-Smirnov test statistic. **0 = identical, 1 = disjoint.** |
| `ks_pvalue` | yes | — | p-value. High p ⇒ same distribution (fail to reject H0). |
| `tv_distance` | — | yes | Total-variation distance over the categorical alphabet. **0 = identical, 1 = disjoint.** |
| `chi2_pvalue` | — | yes | Chi-square test p-value when expected counts are valid. |
| `distribution_score` | yes | yes | Composite **0..1** where 1 = perfect match. Numeric: `1 - ks_statistic`. Categorical: `1 - tv_distance`. |

### Privacy (per table, only with source)

| Metric | Description |
|---|---|
| `privacy_nn_too_close_rate` | Fraction of synthetic rows whose nearest source row (in standardised numeric space) is within `privacy_threshold`. **A *simple* membership-inference proxy** — high rate ⇒ synthetic data is suspiciously close to real rows. |

> **Caveat:** the NN proxy is a sanity check, not formal differential
> privacy. For strong guarantees, use a privacy-preserving generator (DP-SGD,
> PATE, etc.) and validate against published privacy bounds. The proxy
> catches the easy failures: copies, near-copies, and overfit synthesizers.

---

## How to read the score

| `overall_fidelity` | What it means |
|---|---|
| **≥ 0.85** | Marginal distributions match closely. Safe to use as a drop-in for the source on most analyses. |
| **0.6 – 0.85** | Acceptable for non-statistical use (functional tests, integration data) but not for downstream stats. |
| **< 0.6** | Significant divergence. Either: the synthesizer didn't fit well, or you're comparing distributions that legitimately differ (e.g. same schema, different time period). |

Always cross-check `correlation_distance`. Univariate fidelity is necessary
but not sufficient — two columns can have correct marginals but completely
different joint distributions.

---

## Sample report

```
============================================================
Synthetic Data Quality Report
============================================================
Overall fidelity score: 0.957 (1.0 = identical to source, 0.0 = disjoint)

  customers                       rows=     200  fidelity=0.962  privacy_too_close=0.0%
  orders                          rows=     500  fidelity=0.944  privacy_too_close=1.2%
  order_items                     rows=    1500  fidelity=0.965  privacy_too_close=0.0%
```

The `--verbose` flag dumps the full per-column markdown table.

---

## Performance

- Univariate-only: O(n) per column, scales linearly. Works fine on
  millions of rows.
- KS / chi-square: subsamples to **10,000 rows** (cap) for speed. Shouldn't
  meaningfully affect the result on smooth distributions.
- NN-distance privacy proxy: O(n × m × d) brute force (n synthetic, m
  source rows, d numeric columns). Subsamples to 10K × 10K. On larger
  datasets, expect ~1s per table.

If your tables are huge and the cap matters, file an issue — happy to add
proper k-d tree NN search.

---

## When NOT to compute fidelity

- **No source data exists.** Univariate-only mode still works and is
  useful — but you can't compute fidelity without a baseline.
- **Source data is leaked production data.** Don't ingest production data
  into a quality-report pipeline that writes its findings to disk —
  you've just created a different leak. Aggregate fidelity scores are
  safe; raw source data is not.
- **Schema is intentionally evolving.** If your synthetic data is
  structurally newer than your source (added columns, dropped columns),
  fidelity will look bad even if both are correct. Compare matched
  columns only.

---

## Implementation pointer

`validators/quality_report.py` is pure-function and depends only on
pandas / numpy / scipy (already in the base install). No optional extra
needed. ~22 unit tests cover every metric path including edge cases
(empty data, all-null columns, mismatched columns, exact-duplicate
privacy violations).

To extend with a new metric, add a field to `ColumnQualityMetrics` or
`TableQualityMetrics`, populate it in `_column_metrics` /
`_table_metrics`, and add a markdown row in `_render_markdown`. The
JSON / HTML / MCP outputs all flow from the same dataclasses.
