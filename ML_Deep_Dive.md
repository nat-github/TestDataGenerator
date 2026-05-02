# ML Deep Dive — FDL Synthetic Data Platform

*From "what does that even mean?" to "I could explain this at a technical review."*

---

## How to Read This Document

Each section is split into three tiers. You don't have to read all three — stop when you have enough.

| Tier | Who it's for |
|---|---|
| **The Simple Picture** | Anyone. No code, no jargon. |
| **The Mechanics** | Someone comfortable reading configs and logs. |
| **Under the Hood** | Someone who reads Python or wants to modify the code. |

---

## Table of Contents

1. [Why ML in a Data Generator?](#1-why-ml-in-a-data-generator)
2. [Libraries Used — What and Why](#2-libraries-used--what-and-why)
3. [The Three ML Capabilities](#3-the-three-ml-capabilities)
4. [Feature 1 — Auto-Config: Reading Real Data to Write Your Config](#4-feature-1--auto-config-reading-real-data-to-write-your-config)
5. [Feature 2 — PII Detector: Finding Sensitive Columns](#5-feature-2--pii-detector-finding-sensitive-columns)
6. [Feature 3 — Distribution Fitter: Making Numbers Look Real](#6-feature-3--distribution-fitter-making-numbers-look-real)
7. [Performance Intelligence: Batch, Cache, Parallel](#7-performance-intelligence-batch-cache-parallel)
8. [How All Three Connect — The Full Pipeline](#8-how-all-three-connect--the-full-pipeline)
9. [Where Each Feature Lives in the Code](#9-where-each-feature-lives-in-the-code)
10. [CLI Commands Cheat Sheet](#10-cli-commands-cheat-sheet)
11. [Confidence Scores Explained](#11-confidence-scores-explained)

---

## 1. Why ML in a Data Generator?

### The Simple Picture

Imagine you have a spreadsheet with 50 columns of real banking data — account numbers, amounts, currencies, country codes. You want to generate synthetic test data from it. Without ML you would have to:

- Read every column name and guess what it contains
- Figure out whether it's a number, date, or text
- Notice that `ACCT_CCY` only ever contains three values: `EUR`, `USD`, `INR`
- Spot that `ACCT_ID` looks like an IBAN and shouldn't be random garbage
- See that `BOOKG_AMT_NMRC` follows a bell-curve distribution centred around a typical transaction amount

That's hours of manual analysis. The ML layer does it automatically in seconds.

### The Mechanics

"ML" here does not mean deep learning or neural networks. It means **statistical learning from data** — the program observes patterns in your real data and makes decisions based on what it finds. The three things it learns:

1. **Structure** — column types, uniqueness, which values appear
2. **Sensitivity** — does this column contain personal or financial data?
3. **Shape** — does this number follow a normal bell curve, a skewed curve, or a flat distribution?

### Under the Hood

The ML package sits in `ml/` and is composed of three independent modules. They share no state between each other — each does one job and returns a plain Python dict or list. This makes them testable, replaceable, and safely optional (if scipy is not installed, distribution fitting gracefully degrades to uniform random).

```
ml/
├── __init__.py            # empty package marker
├── auto_config.py         # Feature 1 — schema inference
├── pii_detector.py        # Feature 2 — PII detection
└── distribution_fitter.py # Feature 3 — statistical fitting
```

---

## 2. Libraries Used — What and Why

### The Simple Picture

The ML layer is built on four main tools. Think of them as specialist instruments:

- **pandas** — the data reader and analyst. Loads your file, tells you what type each column is, counts nulls, samples values.
- **numpy** — the fast calculator. Does math on thousands of numbers at once instead of one by one.
- **scipy** — the statistician. Knows the mathematical shapes of distributions and can fit them to your data.
- **Faker** — the impersonator. Knows how to make realistic fake names, emails, phone numbers, IBANs, and 60+ other formats.

### The Mechanics

| Library | Type | Used in | Role |
|---|---|---|---|
| `pandas` | Third-party | `auto_config.py`, `pii_detector.py`, `distribution_fitter.py` | Reads CSV/Parquet/Excel; introspects column dtypes; handles null values; samples rows for PII checking |
| `numpy` | Third-party | `distribution_fitter.py`, `helpers.py` | Vectorised numeric generation (`np.random.randint`, `np.random.uniform`); array clipping (`np.clip`); converts pandas Series to raw arrays for scipy |
| `scipy.stats` | Third-party *(optional)* | `distribution_fitter.py` | Fits statistical distributions using Maximum Likelihood Estimation; runs Kolmogorov-Smirnov test to select the best-fitting distribution; samples from the fitted distribution |
| `Faker` | Third-party | `helpers.py` (batch generation path) | Generates realistic fake values for special rules: `NAME`, `EMAIL`, `PHONE`, `IBAN`, `SWIFT`, `ADDRESS`, and 55+ others across 18 locales |
| `rstr` | Third-party | `helpers.py` (regex generation path) | Generates strings that match arbitrary user-defined regex patterns (e.g. `REGEX:\d{4}-[A-Z]{3}`); builds an internal AST from the pattern |
| `re` | Python stdlib | `pii_detector.py`, `helpers.py` | Compiles regex patterns for PII column-name matching and PII value-structure matching; cached at module load for performance |
| `threading` | Python stdlib | `data_generator.py` | `threading.RLock` protects shared PK state during parallel table generation |
| `concurrent.futures` | Python stdlib | `data_generator.py` | `ThreadPoolExecutor` runs independent table generations in parallel (up to 4 workers) |

### Under the Hood

**Why scipy is optional** — scipy is a large scientific library. Not every project needs it. The distribution fitter wraps the import in a lazy loader:

```python
def _get_scipy_stats():
    try:
        from scipy import stats
        return stats
    except ImportError:
        return None   # caller falls back to uniform random
```

If `scipy` is not installed, `_get_scipy_stats()` returns `None`. Every call site checks for `None` and degrades to uniform random generation. The platform never crashes — it just loses the statistical shaping capability.

**Why numpy over plain Python loops** — generating 1,000 random integers in Python:

```python
# Python loop — 1,000 function calls
[random.randint(min_val, max_val) for _ in range(1000)]

# numpy — one C-level call, result as contiguous memory array
np.random.randint(min_val, max_val + 1, size=1000)
```

The numpy version is typically 10–50× faster because it executes in compiled C, avoids Python object overhead, and benefits from CPU vectorisation (SIMD instructions).

**Why Faker and not just random strings** — a column called `CTPTY_NM` (counterparty name) could be filled with `Xk7pQ2mR`. But tests built on that data would fail to catch bugs that only appear with realistic data lengths, character sets, and formats. `Faker.name()` returns `"Marie Dupont"` — believable, the right length, the right character set, locale-aware.

**Why rstr** — when a config has `special_rules: REGEX:\d{3}-\d{2}-\d{4}` (a custom SSN format), rstr parses the regex into an AST and generates strings guaranteed to match. No other general-purpose library does this reliably for arbitrary regex patterns.

---

## 3. The Three ML Capabilities

### The Simple Picture

Think of three specialists who look at your data and each report something different:

```
Your Real Data File
       │
       ├──► Schema Detective   → "Here's a ready-to-use YAML config"
       │
       ├──► Privacy Inspector  → "These 4 columns contain sensitive data"
       │
       └──► Stats Analyst      → "This column follows a normal distribution,
                                  centre at 250, spread of 80"
```

### The Mechanics

| Capability | Input | Output | When it runs |
|---|---|---|---|
| Auto-Config | CSV / Parquet / Excel file | FDL YAML config file | `infer-config` command, or on-demand |
| PII Detector | Data file or existing config | Severity-ranked findings report | `pii-scan` command, or inside `infer-config` |
| Distribution Fitter | Numeric column values | `{name, params, ks_stat}` block | Inside `infer-config`, or during generation |

### Under the Hood

All three are triggered from two places:

1. **CLI** — `main.py` dispatches `infer-config` → `run_infer_config()` and `pii-scan` → `run_pii_scan()`
2. **Generation path** — `DataGenerator._generate_table_data()` calls `helpers.generate_column_batch()` which calls `DistributionFitter.sample()` when a `distribution` block exists in the column config

---

## 4. Feature 1 — Auto-Config: Reading Real Data to Write Your Config

### The Simple Picture

You have a CSV file of real transactions. You run one command:

```bash
python main.py infer-config --input data/transactions.csv --output config/transactions.yaml
```

Five seconds later you have a complete, working YAML config. What the tool figured out on its own:

- Column `ACCT_CCY` only has 3 values → it wrote `values: EUR;USD;INR`
- Column `ACCT_ID` looks like an IBAN → it wrote `special_rules: IBAN`
- Column `CARD_SEQ_NR` has numbers from 17 to 979 → it wrote `min: 17` and `max: 979`
- Column `YEAR_MONTH` follows a normal distribution → it wrote a `distribution:` block
- Column `TX_ACCT_SVCR_REF` is unique in every row → it marked `pk: true`

You just saved 2 hours of manual config writing.

### The Mechanics

The inferrer reads up to 5,000 rows of your file and makes six decisions per column, in order:

```
For each column:
│
├─ 1. What FDL type is it?        (N, VA18, DT, DC, ...)
├─ 2. Is it a primary key?        (100% unique AND 0% null)
├─ 3. What null rate does it have? (e.g. 26% of values are missing)
├─ 4. Low-cardinality values?     (≤30 unique → use as business values)
│      If not:
│      ├─ 5a. Numeric/date bounds? (min + max)
│      └─ 5b. Statistical distribution? (call Distribution Fitter)
└─ 6. PII? (call PII Detector → add special_rules + warning)
```

**Type inference decision table:**

| What pandas sees | FDL type assigned |
|---|---|
| Boolean | `A1` |
| Integer, max < 1,000,000 | `N` |
| Integer, max < 1,000,000,000 | `N19` |
| Integer, larger | `N38` |
| Float / decimal | `DC` |
| Datetime, all times midnight | `D` (date only) |
| Datetime, mixed times | `DT` |
| String, max length ≤ 1 | `VA1` |
| String, max length ≤ 3 | `VA3` |
| String, max length ≤ 18 | `VA18` |
| String, max length ≤ 50 | `VA50` |
| String, max length ≤ 256 | `VA256` |
| String, longer | `T` (text) |

**Low-cardinality rule** — values are captured as a business_values list if:
- Unique value count ≤ 30, AND
- Unique / total ≤ 10% (or unique count ≤ 10, regardless of fraction)

This prevents a column with 1,000 unique customer names from being mistakenly listed as an enum.

### Under the Hood

**Entry point:** `ml/auto_config.py` — class `AutoConfigInferrer`

```python
# Public API — you can call this from Python too, not just CLI
inferrer = AutoConfigInferrer(
    fit_distributions=True,   # call DistributionFitter for numeric cols
    scan_pii=True,            # call PIIDetector for each column
    max_business_values=30,   # enum threshold
    sample_size=5_000,        # max rows read from file
    pii_confidence=0.70,      # minimum confidence to emit PII warning
)
config_dict = inferrer.infer_from_file("data/transactions.csv")
```

**The `_infer_column` method is the decision tree:**

```python
def _infer_column(self, series, col_name, table_name, pk_candidates, pii_map):
    fdl_type  = _infer_fdl_type(series)          # step 1
    null_rate = _null_rate(series)               # step 3
    is_pk     = col_name in pk_candidates        # step 2

    cfg = {"name": col_name, "type": fdl_type}

    if is_pk:
        cfg["pk"] = True
    if null_rate > 0.001:
        cfg["null_rate"] = round(null_rate, 3)

    if not is_pk:
        bv = self._compute_business_values(series)   # step 4
        if bv:
            cfg["values"] = bv
        else:
            bounds = self._compute_bounds(series, fdl_type)  # step 5a
            if bounds:
                cfg.update(bounds)
            if self.fit_distributions:
                dist = self._try_fit_distribution(series, fdl_type)  # step 5b
                if dist:
                    cfg["distribution"] = dist

    if col_name in pii_map:                          # step 6
        cfg["special_rules"] = pii_map[col_name].suggested_rule
        cfg["_pii_note"] = "... review before committing"

    return cfg
```

**Important detail — integer bounds use `int()` not `float()`** to keep the YAML readable (`min: 17` not `min: 17.0`). This was a deliberate fix — numpy's `.min()` returns a numpy scalar which YAML serializes as a complex object unless explicitly converted.

**File format support:**

| Extension | How it's read | Notes |
|---|---|---|
| `.csv` | `pd.read_csv(path, nrows=sample_size)` | Sample size applied at read |
| `.parquet` | `pd.read_parquet(path)` then `df.sample()` | Full read, then subsample |
| `.xlsx` / `.xls` | `pd.read_excel(path, nrows=sample_size)` | Sample size applied at read |

---

## 5. Feature 2 — PII Detector: Finding Sensitive Columns

### The Simple Picture

PII stands for **Personally Identifiable Information** — names, account numbers, tax IDs, emails, phone numbers, passports, and similar data that could identify or harm a real person if leaked.

The PII Detector scans your data (or your config) and reports:

```
⚠️  PII scan — 4 finding(s)

🔴 HIGH (0)
🟡 MEDIUM (3)
  df_cac_acg_entr.ACCT_ID  [IBAN]  conf=95%  → add special_rules: IBAN
    ↳ 100% of sampled values match IBAN pattern
  df_cac_acg_entr.CTPTY_ACCT_ID_IBAN  [IBAN]  conf=98%
    ↳ column name matches /iban/
  df_cac_acg_entr.CTPTY_AGT_BIC  [SWIFT_BIC]  conf=92%  → add special_rules: SWIFT
    ↳ column name matches /swift|bic.?(code)?/
🔵 LOW (1)
  df_cac_acg_entr.CTPTY_NM_ACCRD_TO_ORGTR  [COMPANY]  conf=65%
    ↳ column name matches /company|employer|org.../
```

This tells you: three columns definitely contain financial identifiers, and one column probably contains company names. The tool then suggests exactly what `special_rules:` value to add to your config to generate realistic-but-synthetic replacements.

### The Mechanics

The detector runs two independent passes — like two different inspectors looking at the same column:

**Pass 1 — Name Inspector** (runs always, even without data)
Looks at the column name and matches it against 50+ known patterns. If the column is called `IBAN`, `acct_iban`, or `iban_number` — it's almost certainly an IBAN.

**Pass 2 — Value Inspector** (runs when actual data is available)
Takes up to 30 sample values from the column and tests them against structural patterns. An IBAN always starts with two letters, two digits, then up to 30 alphanumeric characters. If 90% of your sample matches that pattern, confidence is high.

**Severity levels:**

| Level | Meaning | Examples |
|---|---|---|
| 🔴 HIGH | Definitely regulated data — must be handled | SSN, Passport, Aadhaar, NHS Number, Driving Licence |
| 🟡 MEDIUM | Likely sensitive — should be reviewed | IBAN, SWIFT, Email, Phone, Tax ID, IP Address, Date of Birth |
| 🔵 LOW | Possibly sensitive — worth noting | Address, Company Name, Gender, Age, Salary, UUID |

**Confidence score explained** — a number from 0 to 1 (shown as %). It is the product of:
- The base confidence of the pattern rule (e.g. `iban` in name → 98%)
- The fraction of values that matched (e.g. 100% of 30 samples look like IBANs)

If the name match already gives HIGH severity confidence, value sampling is skipped (no need to do more work).

### Under the Hood

**Entry point:** `ml/pii_detector.py` — class `PIIDetector`

**The detection rules are compiled at module load:**

```python
_NAME_RULES = [
    # (regex_pattern, pii_type, suggested_rule, severity, base_confidence)
    (r"iban",                    "IBAN",      "IBAN",   "MEDIUM", 0.98),
    (r"swift|bic.?(code)?",      "SWIFT_BIC", "SWIFT",  "MEDIUM", 0.92),
    (r"ssn|social.?sec",         "SSN",       "SSN",    "HIGH",   0.95),
    (r"e.?mail|email.?addr",     "EMAIL",     "EMAIL",  "MEDIUM", 0.95),
    # ... 46 more rules
]

_COMPILED_NAME_RULES = [
    (re.compile(pat, re.IGNORECASE), pii_type, rule, severity, conf)
    for pat, pii_type, rule, severity, conf in _NAME_RULES
]
```

Compiling at module load means pattern matching is fast — no recompilation on every scan.

**The core `_scan_column` method — the two-pass logic:**

```python
def _scan_column(self, col_name, table_name, series):
    best = None
    best_conf = self.confidence_threshold   # default: 0.60

    # Pass 1: name patterns
    for pattern, pii_type, rule, severity, conf in _COMPILED_NAME_RULES:
        if pattern.search(col_name) and conf > best_conf:
            best_conf = conf
            best = PIIFinding(column=col_name, pii_type=pii_type, ...)

    # Pass 2: value sampling (skip if already HIGH severity from name)
    if series is not None and (best is None or best.severity != "HIGH"):
        sample = series.dropna().astype(str).head(30)
        for pattern, pii_type, rule, conf in _COMPILED_VALUE_RULES:
            hit_rate = sample.apply(lambda v: bool(pattern.match(v))).mean()
            adjusted = conf * hit_rate    # e.g. 0.95 * 1.0 = 0.95
            if adjusted > best_conf:
                best_conf = adjusted
                best = PIIFinding(...)   # value-based evidence

    return best   # None if nothing exceeded threshold
```

**Winner-takes-all** — only the highest-confidence finding per column is returned. If name matching finds an IBAN at 98%, we don't also return a lower-confidence value match — they describe the same thing.

**The `_pii_note` convention** — when auto-config writes a PII finding into the YAML, it adds:
```yaml
_pii_note: IBAN detected (conf=98%) — review and remove _pii_note before committing
```
This is a deliberate human gate. The tool won't silently add a rule without you seeing it. Once you've reviewed it, delete the `_pii_note:` line.

**Two scan modes:**

```python
# 1. Scan a data file (has both column names AND values)
detector.scan_dataframe(df, "my_table")  # runs both passes

# 2. Scan a config (column names only, no actual data)
detector.scan_config(tables_config_dict)  # runs Pass 1 only
```

---

## 6. Feature 3 — Distribution Fitter: Making Numbers Look Real

### The Simple Picture

Imagine your real data has transaction amounts ranging from €1 to €50,000. If you generate synthetic amounts by picking randomly between €1 and €50,000 (uniform random), your test data will look nothing like real data — it will have the same count of €1 transactions as €10,000 transactions. In reality, most transactions cluster around a typical amount with a long tail of large ones.

The Distribution Fitter measures the actual shape of your real data and makes the synthetic data follow the same shape.

```
Real data shape:          Uniform random (bad):      Fitted distribution (good):

  ████                      █ █ █ █ █ █ █ █ █           ████
  ██████                                                 ████████
  ████████                                               ██████████████
  ██████████████                                         ████████████████████
  ██████████████████████    ─────────────────────        ████████████████████
  ───────────────────────
  €1    €500   €5K   €50K   €1    €500   €5K   €50K     €1    €500   €5K   €50K
```

### The Mechanics

The fitter tries five mathematical distribution shapes against your data and picks the one that fits best. Think of these as different "templates" for how data can be shaped:

| Distribution | Shape | Typical real-world use |
|---|---|---|
| **Normal** (bell curve) | Symmetric, peak in middle | Heights, measurement errors, many natural phenomena |
| **Log-normal** | Right-skewed, peak near left | Income, transaction amounts, file sizes |
| **Exponential** | Falls off fast from zero | Wait times, inter-event times |
| **Gamma** | Flexible skewed shape | Loan durations, insurance claims |
| **Uniform** | Flat — all values equally likely | Fallback when nothing else fits |

It picks the winner using a **KS test** (Kolmogorov-Smirnov) — a statistical test that measures how different two distributions are. Lower KS stat = better fit.

The `ks_stat` value is stored in the config so you can see how confident the fit was:

```yaml
distribution:
  name: norm
  params: [202272.5, 179.9]   # [mean, std deviation]
  ks_stat: 0.137              # 0 = perfect fit, 1 = completely wrong shape
  data_min: 202012.0
  data_max: 202580.0
```

### Under the Hood

**Entry point:** `ml/distribution_fitter.py` — class `DistributionFitter`

**The fitting loop:**

```python
_CANDIDATES = [
    ("norm",    "scipy.stats.norm"),
    ("lognorm", "scipy.stats.lognorm"),
    ("expon",   "scipy.stats.expon"),
    ("gamma",   "scipy.stats.gamma"),
    ("uniform", "scipy.stats.uniform"),
]

def fit(self, series: pd.Series) -> Dict:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()

    # Safety check: need at least 30 data points to fit meaningfully
    if len(numeric) < 30:
        return self._uniform_fallback(numeric)

    stats = _get_scipy_stats()  # lazy import — graceful if scipy absent
    if stats is None:
        return self._uniform_fallback(numeric)

    arr = numeric.to_numpy(dtype=float)
    best_name, best_params, best_ks = "uniform", None, float("inf")

    for name, _ in _CANDIDATES:
        dist = getattr(stats, name)
        params = dist.fit(arr)               # Maximum Likelihood Estimation
        ks_stat, _ = stats.kstest(arr, name, args=params)
        if ks_stat < best_ks:               # lower = better fit
            best_ks, best_name, best_params = ks_stat, name, params

    return {
        "name":     best_name,
        "params":   [float(p) for p in best_params],  # plain Python for YAML safety
        "ks_stat":  float(best_ks),
        "data_min": float(arr.min()),
        "data_max": float(arr.max()),
    }
```

**Why `float(p)` explicitly?** — scipy's `dist.fit()` returns numpy scalar types. If you put those directly into a YAML dump, you get:

```yaml
# BAD — numpy scalar serialised as complex object
params: !!python/object/apply:numpy._core.multiarray.scalar ...
```

Explicit `float()` conversion produces clean YAML: `params: [202272.5, 179.9]`.

**The sampling path — how it's used during generation:**

```python
def sample(self, distribution: Dict, n: int, min_val=None, max_val=None) -> np.ndarray:
    dist_obj = getattr(stats, distribution["name"])
    values = dist_obj.rvs(*distribution["params"], size=n)  # vectorised numpy draw
    return np.clip(values, lo, hi)                           # clamp to observed bounds
```

`np.clip` is critical — a normal distribution has infinite tails. Without clipping, you'd occasionally generate a `YEAR_MONTH` of `1850` or `2500`. Clipping to `data_min`/`data_max` keeps values in the range actually observed in the source data.

**Where `params` comes from** — `dist.fit(arr)` uses Maximum Likelihood Estimation (MLE) to find the parameters that make the distribution most likely to have produced the observed data. For a normal distribution, `params` = `[loc, scale]` = `[mean, std]`. For log-normal, `params` = `[shape, loc, scale]`. The exact meaning depends on the scipy distribution — but you don't need to know: `dist_obj.rvs(*params)` unpacks them correctly.

---

## 7. Performance Intelligence: Batch, Cache, Parallel

### The Simple Picture

Generating 1,000 rows for 60 columns one-at-a-time is slow. Three optimisations make it fast:

1. **Batch generation** — generate all 1,000 values for a column in one numpy call, not 1,000 individual calls
2. **Regex cache** — compile a regex pattern once, reuse it thousands of times
3. **Parallel tables** — generate independent tables at the same time on separate threads

### The Mechanics

**Before optimisation:** For 1,000 rows × 60 columns = 60,000 individual function calls.

**After optimisation:**
- Most columns: 60 batch calls (one per column), each generating 1,000 values at once
- Regex patterns: compiled once, cached for all subsequent uses
- Reference tables (no FK dependencies): all generated simultaneously

**Speed impact (rough estimates):**

| Change | Typical speed improvement |
|---|---|
| Batch numeric generation (numpy) | 10–50× faster per column |
| Regex AST cache | 5–20× faster per unique pattern |
| Parallel table generation | Up to 4× for multi-table configs |

### Under the Hood

**Regex AST cache — `utils/helpers.py`:**

```python
class DataHelpers:
    _regex_ast_cache: Dict[str, tuple] = {}   # class-level = shared across all instances

    def generate_from_regex_rule(self, rule: str, max_length=None) -> str:
        pattern = self.extract_regex_pattern(rule)

        cache = self.__class__._regex_ast_cache
        if pattern not in cache:
            # First time: compile regex + build the generation AST
            cache[pattern] = (re.compile(pattern), self._parse_regex_pattern(pattern))
        compiled, ast = cache[pattern]   # subsequent calls: pure dict lookup
        ...
```

The `ast` here is not Python's `ast` module — it's the internal representation of the regex structure used by the `rstr` library to generate strings that match the pattern. Building this is expensive; caching it makes repeated generation of `REGEX:\d{4}-[A-Z]{3}` near-instant from the second call onward.

**Batch column generation — `utils/helpers.py`, method `generate_column_batch`:**

```python
def generate_column_batch(self, config: Dict, n: int) -> List[Any]:
    bv = config.get("business_values")
    if bv:
        bv_list = [v.strip() for v in bv.split(";")]
        return random.choices(bv_list, k=n)         # O(n) with replacement

    distribution = config.get("distribution")
    if distribution:
        from ml.distribution_fitter import DistributionFitter
        arr = DistributionFitter().sample(distribution, n)
        return arr.tolist()                          # numpy vectorised

    data_type = config.get("data_type", "")
    if data_type.startswith("N"):
        return np.random.randint(min_bound, max_cap+1, n).tolist()

    special = config.get("special_rules")
    if special:
        return [self.generate_special_value(special, data_type) for _ in range(n)]

    return [self.generate_sample_value(data_type, config) for _ in range(n)]
```

**Priority order:** business values → distribution → numpy numeric → special rule → fallback. The first match wins, so the fastest paths (business values, numpy) are checked first.

**Parallel table generation — `generators/data_generator.py`:**

```python
def _generate_tables_parallel(self, table_names, records_per_table):
    results = {}
    def _gen(name):
        return name, self._generate_table_data(
            self.tables_config[name], records_per_table[name], for_training=False
        )
    with ThreadPoolExecutor(max_workers=min(len(table_names), 4)) as pool:
        futures = {pool.submit(_gen, t): t for t in table_names}
        for future in as_completed(futures):
            name, df = future.result()
            results[name] = df
    return results
```

**Thread safety note:** PK generation is not parallelised — it uses `threading.RLock` (`_pk_lock`) to protect the shared PK state. Non-PK columns are inherently thread-safe because each column's numpy call is independent. Only independent (non-FK-dependent) reference tables are parallelised; tables that depend on other tables' FKs must wait.

---

## 8. How All Three Connect — The Full Pipeline

### The Simple Picture

When you run `infer-config`, the three ML features work together like a production line:

```
Your CSV/Parquet/Excel file
         │
         ▼
┌─────────────────────────────────┐
│     AutoConfigInferrer          │
│  reads up to 5,000 rows         │
│  loops over every column        │
└────────────┬────────────────────┘
             │  for each column
             ├──────────────────────────────────────────┐
             │                                          │
             ▼                                          ▼
┌────────────────────────┐              ┌───────────────────────────┐
│   Type + PK + BV       │              │      PIIDetector           │
│   inference            │              │  Pass 1: name patterns     │
│   (always runs)        │              │  Pass 2: value sampling    │
└────────────┬───────────┘              └───────────────┬───────────┘
             │ numeric col, not BV                      │ finding
             ▼                                          │
┌────────────────────────┐                              │
│  DistributionFitter    │                              │
│  tries 5 distributions │                              │
│  KS-test picks best    │                              │
└────────────┬───────────┘                              │
             │ distribution block                       │
             └──────────────────────────────────────────┘
                          │ both results
                          ▼
             ┌────────────────────────┐
             │  Final column config   │
             │  written to YAML       │
             └────────────────────────┘
```

Then later, when you **generate data** with that YAML:

```
YAML config (with distribution: block)
         │
         ▼
┌─────────────────────────────────┐
│   DataGenerator                 │
│   _generate_table_data()        │
│                                 │
│   PK columns  → per-row path    │  (must be unique)
│   Non-PK cols → batch path      │
│      └► helpers.generate_column_batch()
│           └► DistributionFitter.sample()  ← uses the fitted params
└─────────────────────────────────┘
```

### Under the Hood

**Integration point in `DataGenerator._generate_table_data()`:**

```python
for column in table_config.columns:
    if column.is_pk:
        # Per-row path — uniqueness guaranteed
        for i in range(actual_num_records):
            val = self._generate_enhanced_value(column, table_name, i, n)
            values.append(val)
    else:
        # Batch path — fast, uses ML distribution if present
        col_config = {
            "business_values": column.business_values,
            "special_rules":   column.special_rules,
            "data_type":       column.data_type,
            "min_value":       column.min_value,
            "max_value":       column.max_value,
            "column_name":     column.column_name,
            "max_length":      column.length,
            "distribution":    getattr(column, "distribution", None),  # ← from auto-config
        }
        values = self.helpers.generate_column_batch(col_config, actual_num_records)
```

The `distribution` field is added to `ColumnConfig` in `models/config_models.py`:
```python
class ColumnConfig(BaseModel):
    ...
    distribution: Optional[Dict[str, Any]] = None   # from auto-config or manual
    example_value: Optional[Any] = None             # wire mock forward compat
```

---

## 9. Where Each Feature Lives in the Code

### File Map

```
TestDataGeneration/
│
├── ml/                              ← ALL ML logic lives here
│   ├── __init__.py
│   ├── auto_config.py               ← Feature 1
│   │   ├── AutoConfigInferrer       class
│   │   ├── _infer_fdl_type()        function — dtype → FDL type string
│   │   ├── _null_rate()             function — fraction of NaN
│   │   └── _build_fdl_config()      function — wraps tables in FDL envelope
│   │
│   ├── pii_detector.py              ← Feature 2
│   │   ├── _NAME_RULES              list — 50+ column-name patterns
│   │   ├── _VALUE_RULES             list — 9 value-structure patterns
│   │   ├── _COMPILED_NAME_RULES     pre-compiled at module load
│   │   ├── PIIFinding               dataclass — one result per column
│   │   └── PIIDetector              class
│   │       ├── scan_dataframe()     both passes
│   │       ├── scan_config()        name-pass only (no data)
│   │       ├── format_report()      human-readable output
│   │       └── _scan_column()       core logic
│   │
│   └── distribution_fitter.py       ← Feature 3
│       ├── _CANDIDATES              list — 5 scipy distributions in order
│       ├── MIN_SAMPLES = 30         below this → uniform fallback
│       └── DistributionFitter       class
│           ├── fit()                → {name, params, ks_stat, data_min, data_max}
│           ├── sample()             → np.ndarray, clipped to bounds
│           └── _uniform_fallback()  → safe default when fitting not possible
│
├── utils/helpers.py
│   ├── _regex_ast_cache             class var — perf Feature (regex cache)
│   ├── generate_from_regex_rule()   uses cache
│   └── generate_column_batch()      perf Feature (batch generation)
│
├── generators/data_generator.py
│   ├── _bv_overflow_warned          set — deduplicates warning spam
│   ├── _pk_lock                     RLock — thread safety
│   ├── _generate_table_data()       calls batch path for non-PK
│   └── _generate_tables_parallel()  perf Feature (parallel tables)
│
├── models/config_models.py
│   └── ColumnConfig
│       ├── distribution             Optional[Dict] — carries fitter output
│       └── example_value            Optional[Any]  — wire mock compat
│
└── main.py
    ├── run_infer_config()           CLI handler for infer-config
    └── run_pii_scan()               CLI handler for pii-scan
```

---

## 10. CLI Commands Cheat Sheet

### Auto-Config (`infer-config`)

```bash
# Basic — infer everything
python main.py infer-config --input data/transactions.csv --output config/transactions.yaml

# Skip scipy fitting (faster, no distribution blocks)
python main.py infer-config --input data.parquet --output config.yaml --no-distributions

# Skip PII detection
python main.py infer-config --input data.xlsx --output config.yaml --no-pii-scan

# Lower PII threshold (catch more potential PII, more false positives)
python main.py infer-config --input data.csv --output config.yaml --pii-confidence 0.50

# Override the inferred table name
python main.py infer-config --input data.csv --output config.yaml --table-name my_table

# Read more rows for better accuracy (slower)
python main.py infer-config --input data.csv --output config.yaml --sample-size 20000

# See detailed logging
python main.py infer-config --input data.csv --output config.yaml --verbose
```

### PII Scan (`pii-scan`)

```bash
# Scan a data file (name + value analysis)
python main.py pii-scan --input data/customers.csv

# Scan an existing YAML config (name analysis only — no data values)
python main.py pii-scan --input config/transactions.yaml

# Lower threshold to catch lower-confidence findings
python main.py pii-scan --input data.csv --confidence 0.50

# Scan a Parquet file
python main.py pii-scan --input output/run_01/df_cac_acg_entr.parquet
```

**Output interpretation:**

```
⚠️  PII scan — 4 finding(s)

🔴 HIGH   → Action required. Real regulated data present.
🟡 MEDIUM → Review required. Likely sensitive.
🔵 LOW    → Informational. Worth noting.

Each finding shows:
  table.COLUMN  [PII_TYPE]  conf=XX%  → add special_rules: RULE
    ↳ evidence (why it was flagged)
```

### After `infer-config` — What to Do

1. Open the generated YAML file
2. Search for `_pii_note:` — these are the PII flags
3. For each: decide if the `special_rules:` suggestion is correct
4. Remove the `_pii_note:` line (it's a human review marker, not a valid config key)
5. Run: `python main.py generate --config config/transactions.yaml --output output/v1`

---

## 11. Confidence Scores Explained

Confidence scores appear everywhere in the ML output. Here's exactly what they mean.

### The Simple Picture

Confidence is a number from 0 to 1 (or 0% to 100%) that says "how sure is the system about this finding?" It is not a probability in a strict mathematical sense — it is a calibrated signal combining multiple evidence types.

### How Scores Are Calculated Per Feature

**PII Detector — name-based:**
Directly from the rule definition. If a column contains the word `iban`, the rule assigns `0.98`. This was manually calibrated — "iban" in a column name is almost always an IBAN.

**PII Detector — value-based:**
```
adjusted_confidence = base_rule_confidence × hit_rate
```
Example: IBAN value pattern has base `0.95`. If 20 out of 30 sampled values match the IBAN regex: `0.95 × (20/30) = 0.633`.

**Distribution Fitter — KS stat:**
Not a confidence score but an error measure. Lower is better. `ks_stat = 0.0` means perfect fit; `ks_stat = 1.0` means the distribution completely wrong. Values below `0.15` are generally considered good fits.

### What to Do With Different Score Ranges

| Confidence | Meaning | Action |
|---|---|---|
| ≥ 90% | Near-certain | Accept the suggestion, just review the special_rules value |
| 70–89% | Likely correct | Look at a few sample values yourself to confirm |
| 60–69% | Possible | Review carefully — column name is suggestive but values might be generic |
| < 60% | Below threshold | Not shown in output (filtered by `confidence_threshold`) |

**Tuning the threshold:**
- Use `--pii-confidence 0.85` if you want only high-confidence PII (fewer false positives)
- Use `--pii-confidence 0.50` if you want a broad sweep (more false positives, fewer misses)
- Default is `0.70` — a reasonable balance for financial data

---

## Quick Reference — The One-Page Version

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ML FEATURES — QUICK REFERENCE                         │
├────────────────┬────────────────────────────────┬───────────────────────────┤
│ FEATURE        │ WHAT IT DOES                   │ HOW TO USE                │
├────────────────┼────────────────────────────────┼───────────────────────────┤
│ Auto-Config    │ Reads real data file →          │ python main.py            │
│                │ generates FDL YAML config       │   infer-config            │
│                │ with types, PKs, values,        │   --input data.csv        │
│                │ bounds, distributions, PII      │   --output config.yaml    │
├────────────────┼────────────────────────────────┼───────────────────────────┤
│ PII Detector   │ Scans column names + values →   │ python main.py pii-scan   │
│                │ flags sensitive data with        │   --input data.csv        │
│                │ severity + confidence +          │ (also runs inside         │
│                │ suggested special_rules          │  infer-config by default) │
├────────────────┼────────────────────────────────┼───────────────────────────┤
│ Distribution   │ Fits scipy distribution to      │ Runs automatically inside │
│ Fitter         │ numeric columns → stores        │ infer-config. Also used   │
│                │ {name, params} so generation    │ by generate when config   │
│                │ follows the same statistical    │ has distribution: block   │
│                │ shape as real data              │                           │
├────────────────┼────────────────────────────────┼───────────────────────────┤
│ Batch Gen      │ numpy-vectorised column         │ Transparent — always on.  │
│ (perf)         │ generation instead of 1-by-1    │ ~10-50x faster per col    │
├────────────────┼────────────────────────────────┼───────────────────────────┤
│ Regex Cache    │ Compile regex + AST once,       │ Transparent — always on.  │
│ (perf)         │ reuse across all rows           │ ~5-20x faster per pattern │
├────────────────┼────────────────────────────────┼───────────────────────────┤
│ Parallel       │ ThreadPoolExecutor (max 4)      │ Transparent — kicks in    │
│ Tables (perf)  │ for independent ref tables      │ for multi-table configs   │
└────────────────┴────────────────────────────────┴───────────────────────────┘
```

---

*Document written for the FDL Synthetic Data Platform — ml/ package, utils/helpers.py, generators/data_generator.py*
