# Synthesizer Engines

Generation engines are pluggable. The engine decides *how rows are
invented*; everything else — config parsing, primary keys, foreign-key
resolution, rules and derived columns, type coercion, Parquet export — is
owned by `DataGenerator` and is identical whichever engine you pick.

## Why this exists

The consistent finding across the synthetic-data literature is that **no
single method dominates**. Fidelity, utility and privacy trade off against
one another, and the ranking shifts with the dataset ([Systematic
Assessment of Tabular Data Synthesis](https://arxiv.org/abs/2402.06806)).
A platform hardcoded to one engine cannot act on that finding — it cannot
even measure it.

Pair this with `quality-report`, which now measures utility as well as
fidelity, and engine selection becomes an evidence-based decision instead
of a default nobody revisits.

---

## Built-in engines

| Name | What it is | Relationships | Speed |
|---|---|---|---|
| `sdv` *(default)* | SDV `HMASynthesizer` — hierarchical, multi-table | **Modelled natively** | Slow |
| `gaussian-copula` | SDV `GaussianCopulaSynthesizer`, per table | FK resolution only | Fast |
| `ctgan` | SDV `CTGANSynthesizer` — conditional GAN, per table | FK resolution only | Very slow (torch) |
| `tvae` | SDV `TVAESynthesizer` — variational autoencoder, per table | FK resolution only | Slow (torch) |
| `dp-marginal` | Laplace-noised marginals — **formal ε-DP guarantee** | FK resolution only | Fast |
| `rule-based` | Config rules only — regex, business values, Faker | FK resolution only | Instant |

`dp-marginal` is the only engine offering a *provable* privacy guarantee;
everything else, including the NN privacy proxy in `quality-report`, is a
heuristic. It also discards correlations between columns, which is a real
cost. See **[Differential_Privacy.md](Differential_Privacy.md)**.

```bash
python main.py generate --list-engines
```

### Relationships: the one thing to understand

Only `sdv` models cross-table structure. The single-table engines see each
table in isolation, so **correlations between tables are not preserved** —
only correlations within them.

Referential integrity is still guaranteed on every engine: FK resolution
runs after sampling and overwrites foreign-key columns with real parent
keys. What you lose is statistical relationships *across* tables, not
valid joins.

Pick `sdv` when cross-table structure matters. Pick a single-table engine
when per-table fidelity matters more, or when SDV's fit is too slow or
fails on your column types.

---

## Selecting an engine

Precedence: **CLI flag → config setting → default (`sdv`)**.

```bash
# CLI
python main.py generate --config config/acct.yaml --output out/ --engine gaussian-copula

# With engine-specific options (repeatable)
python main.py generate --config config/acct.yaml --output out/ \
  --engine ctgan --engine-option epochs=300 --engine-option batch_size=500
```

```yaml
# YAML config
run_settings:
  synthesizer_engine: gaussian-copula
  engine_options: "epochs=250"        # or a mapping
```

In an Excel `Run_Settings` sheet, use the same two setting names. Because a
cell cannot hold structured data, `engine_options` also accepts a
`key=value;key=value` string. Digit-only values are coerced to integers —
`epochs=300` arrives as `300`, not `"300"`.

```python
# Python
from sdp.generators.data_generator import DataGenerator

gen = DataGenerator("config/acct.yaml", seed=42,
                    engine="ctgan", engine_options={"epochs": 300})
```

---

## Cost reporting

Every engine records what it cost. The 417-model survey
([arXiv:2401.02524](https://arxiv.org/abs/2401.02524)) singles out the
neglect of training and computational cost as a gap in the literature — an
engine that is 3% more faithful and 40× slower is not obviously the better
choice, and you cannot have that argument without numbers.

```
Engine cost: sdv — fit 18.003s, sample 20.489s (300 training rows, 900 sampled)
```

```python
gen.engine_stats
# {'engine': 'sdv', 'fit_seconds': 18.003, 'sample_seconds': 20.489,
#  'total_seconds': 38.492, 'fit_rows': 300, 'sampled_rows': 900,
#  'retries': 0, 'notes': []}
```

Measured on `examples/configs/yaml/02_ecommerce_relationships.yaml`, 300
rows per table across 3 tables: `sdv` fit in 18.0s, `gaussian-copula` in
1.0s. Whether the 18× cost buys anything is exactly the question
`quality-report` now answers.

---

## Writing an engine

Implement two methods and register the class:

```python
from sdp.synthesizers.base import Synthesizer
from sdp.synthesizers.registry import register


class MyEngine(Synthesizer):
    name = "my-engine"
    description = "One line, shown by --list-engines"
    handles_relationships = False       # True only if you model cross-table structure

    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_dependency       # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, sample_data, metadata=None) -> bool:
        # sample_data: {table_name: DataFrame}, already sanitised
        # Return False on failure — never raise.
        with self._timed("fit"):
            ...
        self._fitted = True
        return True

    def sample(self, records_per_table):
        # Return {table_name: DataFrame} with at least the requested rows.
        with self._timed("sample"):
            ...


register("my-engine", "my_package.engines:MyEngine")
```

Contract notes:

- **`fit` must not raise.** A failed fit is an expected outcome on messy
  configs, not an exceptional one; returning False lets the caller fall
  back to config-rule generation. `sample` *should* raise on failure — by
  then the caller has committed to this engine, and a silent empty result
  would be worse.
- **Registration is by import path**, so adding an engine costs nothing at
  import time for runs that do not use it. Resolving `rule-based` must not
  drag torch in via the `ctgan` entry.
- **`is_available()` is checked before instantiation**, so a missing
  optional dependency produces a clear message rather than an ImportError
  mid-run.
- **`model` is identity-checked** by `DataGenerator` to decide whether the
  engine still owns the current model. Return the same object each call —
  do not rebuild a dict on the fly.
- Use `self._timed(...)` so your engine appears in cost reporting.
- Retry policy belongs to the **caller**, not the engine: regenerating
  sample data with more aggressive sanitisation needs config knowledge an
  engine does not have.

---

## Compatibility

The refactor is additive. `DataGenerator.synthesizer`,
`DataGenerator.is_fitted` and `train_synthesizer()` behave exactly as
before, and assigning a synthesizer double directly to
`generator.synthesizer` still works — `_sample_from_synthesizer` falls back
to the shared multi-table helper when no engine owns the model.

Two engine-specific limits:

- **Model artifact caching** (`save_model_artifact`) is SDV-only. The
  artifact format is a single pickled synthesizer plus an SDV-version
  sidecar; other engines hold one model per table. Caching is skipped with
  a log line rather than writing an unloadable artifact.
- `train_synthesizer()` returning **False is not always an error**. For
  `rule-based` it is the expected outcome — there is no model to fit — and
  generation proceeds from config rules.

---

## Implementation pointer

| File | Responsibility |
|---|---|
| `sdp/synthesizers/base.py` | `Synthesizer` ABC, `EngineStats`, shared `sample_multi_table` helper |
| `sdp/synthesizers/registry.py` | Lazy name → class registry, `create`, `describe` |
| `sdp/synthesizers/sdv_hma.py` | The default HMA engine |
| `sdp/synthesizers/single_table.py` | Gaussian copula / CTGAN / TVAE, per table |
| `sdp/synthesizers/rule_based.py` | Explicit no-model path |

Tests: `tests/test_synthesizer_engines.py` (33) — registry, sampling
helper, engine contract, `DataGenerator` wiring, CLI plumbing.
