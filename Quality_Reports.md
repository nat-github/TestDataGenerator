# Synthetic-Data Quality Reports

A statistical fidelity / utility / bias / privacy report on generated data.
Two modes:

- **Univariate-only** (no source data) — per-column dtype, null rate, unique
  count, summary stats; per-table Pearson correlation matrix. Always
  available. Answers *"does this synthetic data look plausible on its own?"*
- **Full report** (source data provided) — adds KS test for numeric columns,
  total-variation distance for categoricals, correlation-matrix delta, a
  nearest-neighbour privacy proxy, a **TSTR utility score**, and **bias
  drift**.

This complements Great Expectations: GX validates *schema conformance*,
quality reports validate *statistical fidelity*. Different questions, both
worth answering.

## Four questions, four answers

Each measure catches failures the others cannot see:

| Measure | Question | Blind to |
|---|---|---|
| **Fidelity** | Does it *look* like the real data? | Whether the data is still usable |
| **Utility** | Can a model still *learn* from it? | Who is represented |
| **Bias** | Did generation skew groups or outcomes? | Individual-row leakage |
| **Privacy** | Is it too close to real rows? | Everything above |

The gap between fidelity and utility is the one that surprises people. A
generator can reproduce every marginal distribution perfectly while
destroying the relationships *between* columns — every histogram matches,
and a model trained on the result learns nothing:

```
Overall fidelity score: 0.893 (1.0 = identical to source, 0.0 = disjoint)
Overall utility (TSTR): 0.000 (1.0 = as useful as real data)
Bias flags: loans
```

Fidelity called that data 89% faithful. It is worthless for its purpose.

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

# Name the column to predict / measure outcomes against (repeatable).
# Auto-selected per table when omitted.
python main.py quality-report \
  --generated output/run_01 --source data/real_sample \
  --target loans=approved --target customers=tier

# Skip the slow bit — TSTR fits two models per table
python main.py quality-report \
  --generated output/run_01 --source data/real_sample --no-utility
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
from sdp.validators.quality_report import quality_report, quality_report_from_paths

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

### Utility — TSTR (per table, only with source)

**T**rain on **S**ynthetic, **T**est on **R**eal. A model is trained on the
synthetic data and scored on held-out *real* rows; an identical model
trained on *real* rows is scored on the same held-out set. The ratio of the
two says whether the synthetic data still carries the signal.

| Metric | Description |
|---|---|
| `target_column` | Column being predicted. Auto-selected (lowest-cardinality non-identifier categorical, else a numeric column) or set via `--target`. |
| `task` | `binary` / `multiclass` / `regression` |
| `metric` | `roc_auc` / `macro_f1` / `r2` |
| `score_real` | TRTR baseline — the model trained on real data |
| `score_synthetic` | TSTR — the model trained on synthetic data |
| `utility_ratio` | **1.0 = as useful as real data, 0.0 = useless.** `None` when the real baseline is itself near chance. |
| `verdict` | Plain-language reading of the ratio |

The ratio is computed on **chance-adjusted skill**, never on raw scores:

```
skill = (score − chance) / (1 − chance)     # clipped at 0
ratio = skill_synthetic / skill_real
```

| Metric | Chance level |
|---|---|
| `roc_auc` | 0.5 |
| `macro_f1` | 1/k (k = classes in the real test set) |
| `r2` | 0.0 |

Raw ratios flatter synthetic data badly. Two models at 0.55 and 0.60 AUC
give a respectable-looking 0.92 when both are barely better than a coin
toss; on skill that reads 0.50. The trap is worse for multiclass — four
classes at macro F1 0.233 and 0.237 are *both* at chance (1/4 = 0.25), and
a raw ratio calls that "excellent — as useful as real data".

When the real baseline is itself at or near chance, **no ratio is
reported** (`None`, with a note naming the chance level). Nothing was
learnable from the real data either, so there is no meaningful comparison
to make — and a number there would read as a verdict.

| `utility_ratio` | Verdict |
|---|---|
| **≥ 0.95** | Excellent — as useful as real data |
| **0.80 – 0.95** | Good — minor loss of signal |
| **0.50 – 0.80** | Degraded — noticeable loss of signal |
| **< 0.50** | Poor — downstream models learn little |

A single class in the synthetic target (mode collapse) scores **0.0** rather
than erroring: no model can be trained on it, so zero utility is the honest
answer.

### Bias (per categorical column, only with source)

Two distinct failures, because they occur independently — representation can
look perfect while outcomes are badly skewed.

| Metric | Description |
|---|---|
| `max_representation_shift` | Largest change in any group's share of the table. A status that is 8% of real rows but 2% of synthetic rows means tests barely exercise that path. |
| `flagged_groups` | Groups whose share moved ≥ 5 points (groups under 1% of source are reported but never flagged — too rare to compare stably) |
| `source_disparity` | Widest gap in positive-outcome rate between groups, in the **real** data |
| `synthetic_disparity` | The same gap in the **synthetic** data |
| `disparity_amplification` | `synthetic − source`. **Above zero means the generator widened the gap between groups** — it invented discrimination that was not in the original. |
| `verdict` | `faithful` / `representation drift` / `severe representation drift` / `outcome disparity amplified` |

The outcome column defaults to whatever the utility check predicted, so both
measures describe the same target.

> Bias output is **descriptive, not prescriptive**. It reports what changed
> between source and synthetic. Whether a given shift matters is a domain
> judgement — a deliberately balanced test set will flag as drifted, and
> that is correct behaviour, not a false positive.

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
Overall utility (TSTR): 0.912 (1.0 = as useful as real data)
Bias flags: orders

  customers        rows=     200  fidelity=0.962  utility=0.981  privacy_too_close=0.0%
  orders           rows=     500  fidelity=0.944  utility=0.843  privacy_too_close=1.2%  bias=1col
  order_items      rows=    1500  fidelity=0.965  utility=0.912  privacy_too_close=0.0%
```

`--verbose` dumps the full markdown, including the utility and bias
sections:

```
### Utility — can a model still learn from this data?

Predicting **`approved`** (binary, scored by `roc_auc`)

| Trained on | Score | Rows |
|---|---:|---:|
| Real data (baseline) | 0.9996 | 1,400 |
| Synthetic data | 0.4287 | 2,000 |

**Utility ratio: 0.000** — poor — downstream models learn little
*Both models scored on the same 600 held-out real rows.*

### Bias — representation and outcome drift

| Column | Verdict | Max share shift | Groups affected | Disparity amplification |
|---|---|---:|---|---:|
| region | severe representation drift | 34.8% | east, north, south | -1.1% |
```

---

## Performance

- Univariate-only: O(n) per column, scales linearly. Works fine on
  millions of rows.
- KS / chi-square: subsamples to **10,000 rows** (cap) for speed. Shouldn't
  meaningfully affect the result on smooth distributions.
- NN-distance privacy proxy: O(n × m × d) brute force (n synthetic, m
  source rows, d numeric columns). Subsamples to 10K × 10K. On larger
  datasets, expect ~1s per table.
- **TSTR utility: the most expensive metric** — it fits two random forests
  per table. Training caps at 20,000 rows and the test set at 10,000. Budget
  a few seconds per table; pass `--no-utility` when you want a quick pass.
- Bias: pure `value_counts` / `groupby`. Negligible.

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

Three modules, all in the base install — no optional extra needed:

| Module | Responsibility | Dependencies |
|---|---|---|
| `validators/quality_report.py` | Univariate + fidelity + privacy; composes the other two | pandas, numpy, scipy |
| `validators/utility.py` | TSTR utility | + scikit-learn |
| `validators/bias.py` | Representation drift, outcome disparity | pandas only |

Utility and bias live in their own modules rather than inside
`quality_report.py` so each stays readable and independently testable;
the report composes them and owns serialisation and rendering.

Both are computed defensively — a failure in either is caught, recorded as
a table note, and the rest of the report still returns. Losing the whole
report because one metric hit an odd dtype is not an acceptable trade.

Test coverage: ~22 tests in `tests/test_quality_report.py` for the
fidelity/privacy paths, 29 in `tests/test_utility_and_bias.py` for utility
and bias. The latter build datasets with a *known* answer — a learnable
signal that synthetic data either preserves or destroys, a group skew that
is either faithful or amplified — so the tests verify the metrics measure
what they claim rather than merely returning a number.

To extend with a new metric, add a field to `ColumnQualityMetrics` or
`TableQualityMetrics`, populate it in `_column_metrics` /
`_table_metrics`, and add a markdown row in `_render_markdown`. The
JSON / HTML / MCP outputs all flow from the same dataclasses. Note that
`@property` values are *not* captured by `asdict()` — add them explicitly
in `_serialise_*`, as `verdict`, `delta` and `ratio` do.
