# Differential Privacy (`dp-marginal`)

The only engine in this platform that makes a **provable** privacy claim.
Everything else — including the nearest-neighbour proxy in
`quality-report` — is a heuristic that catches copies and overfit models
but proves nothing.

```bash
python main.py generate --config config/loans.yaml --output out/ \
  --engine dp-marginal --epsilon 1.0 --privacy-report-json privacy.json
```

---

## What ε means

ε (epsilon) is the privacy budget. It bounds how much the output can change
when any single row is added or removed. **Lower is more private and less
accurate.**

| ε | Reading |
|---|---|
| 0.1 | Very strong. Rare categories will be visibly distorted. |
| 1.0 | Common default. Strong protection, modest distortion on typical tables. |
| 10 | Weak in the formal sense, though still far better than nothing. |
| ≥ 100 | Essentially a formality — do not present this as "private". |

## The mechanism

For each column with a publicly declared domain: build a histogram over
that domain, add Laplace noise at scale `1/ε_col`, clip negatives to zero,
renormalise, sample.

Adding or removing one row changes any histogram by at most 1, so L1
sensitivity is 1 and Laplace noise at scale `1/ε_col` gives `ε_col`-DP.
Sampling from the noisy histogram is post-processing and costs nothing
further.

---

## Why this platform can do DP honestly

The usual way to leak while claiming DP is to **derive the domain from the
data** — taking bin edges from the observed min/max, or the category list
from observed values. Both are non-private queries, and the ε you publish
afterwards is a fiction.

This platform does not have that problem, because **the config is the
public schema**. It already declares:

- `business_values` → the category set
- `min_value` / `max_value` → numeric bounds

Nothing about the domain is learned from the data, so the accounting holds.

**Columns without a declared domain are never measured against the data at
all.** They are generated from config rules alone — zero budget, zero
information learned. This is reported explicitly:

```json
"unmeasured_columns": [
  {"table": "loans", "column": "notes",
   "reason": "no declared domain — generated from config rules, no access to the data"}
]
```

Primary and foreign keys are excluded deliberately: they are identifiers,
not distributions. PKs must stay unique, FKs are overwritten by FK
resolution, so modelling either would be pointless as well as
privacy-relevant.

---

## Composition

**Within a table**, per-column histograms are sequential composition over
the same records, so their epsilons **add**. The budget is split evenly
across measured columns: `ε_col = ε / m`. Three measured columns at ε=3.0
means ε=1.0 each.

**Across tables**, the guarantee is stated **per table**.

> ### The limit worth reading twice
>
> ε protects the presence of any single row *within its own table*. It does
> **not** protect an entity appearing across several tables. One customer
> with fifty orders is not covered by a per-row guarantee on the orders
> table.
>
> Relational DP is an open research problem. Claiming a cross-table
> guarantee here would be the same kind of fiction as data-derived bins.

The report states this in every output rather than leaving it to be
assumed:

```json
"guarantee": "epsilon-differential privacy per row, within each table;
              does not cover entities spanning multiple tables"
```

---

## Seeding voids the guarantee

`--seed` makes the Laplace noise reproducible. An adversary who knows the
seed can subtract the noise and recover the true counts. The engine warns
when both are set:

```
! seeded run: the Laplace noise is reproducible, so an adversary who knows
  the seed can subtract it. Use an unseeded run when the privacy guarantee
  needs to hold.
```

Seeded DP runs are fine for testing the pipeline. They are not private.

---

## What it costs you

**Marginals are independent, so correlations between columns are not
preserved.** This is the standard DP baseline, and it is an honest trade:
correlation structure is exactly what identifies individuals.

Run `quality-report` against the output to see the cost in utility terms
before deciding the trade is worth it. Expect the TSTR utility score to
drop sharply — a model needs the relationships this engine discards.

### Where the noise actually bites

Laplace noise magnitude depends on ε, **not** on row count, so relative
error scales as 1/n. Privacy is cheap on large tables and expensive on
small ones and rare categories.

Measured at ε=0.5, true rare-category share 5%, 20 noise draws:

| Rows | Recovered share | Mean absolute error |
|---:|---:|---:|
| 50 | 0.043 | 0.030 |
| 200 | 0.049 | 0.009 |
| 1,000 | 0.049 | 0.003 |
| 10,000 | 0.049 | 0.003 |

At 50 rows the rare category carries ~60% relative error. By 1,000 rows it
is ~6%. **If your table is small or the categories you care about are rare,
DP will hurt — and that is the mechanism working, not failing.**

---

## An important caveat about *what* is being protected

DP protects the data the engine is **fitted on**. In the normal flow this
platform fits on sample rows generated from your config rules — which
contain no real data, making the guarantee technically valid but
**vacuous**: there is no private information present to protect.

The guarantee becomes meaningful when real data enters training, which
happens through **anchored generation** (a table's `source:` field feeds
real rows into fitting — see `Yaml_Config_Schema.md`).

Before you present ε to anyone, confirm real data is actually in the
training set. A correct ε over synthetic input protects nothing.

---

## Configuration

```yaml
run_settings:
  synthesizer_engine: dp-marginal
  engine_options: "epsilon=0.5;numeric_bins=30"
```

| Option | Default | Meaning |
|---|---|---|
| `epsilon` | 1.0 | Total budget per table |
| `numeric_bins` | 20 | Histogram bins for numeric columns. More bins capture shape better but spread the same budget thinner, so each bin is noisier. |

`--epsilon` is a first-class CLI flag rather than just another
`--engine-option`, because its value is a promise to a regulator, not a
tuning knob.

---

## Reading the privacy report

```python
gen = DataGenerator("config/loans.yaml", engine="dp-marginal")
...
gen.privacy_report
```

```json
{
  "epsilon_requested": 1.0,
  "per_table_epsilon": {"loans": 1.0},
  "guarantee": "epsilon-differential privacy per row, within each table; ...",
  "measured_columns": [
    {"table": "loans", "column": "status", "epsilon": 0.5, "domain": "categorical", "bins": 3},
    {"table": "loans", "column": "amount", "epsilon": 0.5, "domain": "numeric", "bins": 20}
  ],
  "unmeasured_columns": [...],
  "warnings": [...]
}
```

`privacy_report` is `None` for every other engine. That absence means "no
formal guarantee was made", which is the honest answer for engines that
make none.

---

## Implementation pointer

`sdp/synthesizers/dp_marginal.py`. Tests in `tests/test_dp_marginal.py`
(29) target the properties that make the guarantee real, not just the code
path: the domain comes from config and never from data; epsilons compose
correctly; noise is genuinely applied and lower ε means more of it;
undeclared columns never reflect the training data; the reported accounting
matches what was spent.
