# Data Validation with Great Expectations

After generation, validate the output against expectations auto-derived from
your config — no hand-written GX suite required. Run a one-liner; get a
pass/fail summary plus exactly which rows broke which rule.

## Why

The platform produces realistic data, but "realistic" doesn't guarantee
"compliant with the schema you declared." Examples of what slips through
without validation:

- A regex `special_rule` accepts a value the downstream consumer rejects.
- An FK resolution surprise leaves a child column with values outside the
  parent's PK domain.
- An SDV training fallback produces a numeric distribution that drifts
  outside the declared `min_value` / `max_value`.
- A renaming refactor misses a column → the table is silently missing it.

Great Expectations (GX) catches all of these at the boundary: between
"data we just produced" and "data we hand to the next system."

## How it's wired in

```
ColumnConfig            ──► validators/gx_validator.derive_expectations_for_column()
TableConfig             ──► validators/gx_validator.derive_expectations_for_table()
                              │
                              ▼
                       List[(kind, column, kwargs)]
                              │
                              ▼
                       GX 1.x pandas validator
                              │
                              ▼
                       ValidationReport (per-table, per-expectation)
```

The mapping is deterministic and documented in
[validators/gx_validator.py](./validators/gx_validator.py):

| ColumnConfig field          | Auto-derived expectation                          |
|-----------------------------|---------------------------------------------------|
| `is_pk: true`               | `expect_column_values_to_be_unique` + `..._not_be_null` |
| `nullable: false`           | `expect_column_values_to_not_be_null`             |
| `business_values: "A;B;C"`  | `expect_column_values_to_be_in_set`               |
| `min_value` / `max_value`   | `expect_column_values_to_be_between`              |
| `length: N`                 | `expect_column_value_lengths_to_be_between(1, N)` |
| `special_rules: "REGEX:X"`  | `expect_column_values_to_match_regex(X)`          |
| `special_rules: "EMAIL"`    | `..._match_regex` with the canonical email shape  |
| `special_rules: "UUID"`     | `..._match_regex` with the UUID shape             |
| `special_rules: "IBAN"`     | `..._match_regex` with the IBAN shape             |
| `special_rules: "IPV4/6"`   | `..._match_regex` with the address shape          |
| `data_type: "N10"`          | `expect_column_values_to_be_in_type_list([int…])` |
| (every column)              | `column_exists` (custom)                          |
| `TableConfig.num_rows`      | `row_count_between(num_rows × (1 ± tolerance))`   |

`special_rules` we **don't** validate via regex (NAME, ADDRESS, COMPANY,
etc.) are intentionally omitted — they're too varied for a static pattern.
PII-format rules with deterministic shapes (EMAIL, UUID, IPv4/6, MAC, URL,
IBAN, SWIFT, BIC, SSN) all do get regex expectations.

## Install

```bash
poetry install --extras gx
```

GX is an optional dependency. The validator module imports it lazily and
exposes `validators.gx_validator.HAS_GX` so other code can degrade
gracefully when it's absent.

## Use — three entry points

### 1. Inline with `generate`

```bash
python main.py generate --config examples/configs/yaml/02_ecommerce_relationships.yaml \
    --output output/02 \
    --default-records 500 --seed 7 \
    --validate-with-gx
```

The validation runs after generation, before the function returns. Add
`--gx-fail-on-error` to make a failed expectation set a non-zero exit
code (useful in CI).

### 2. Standalone `validate-data` subcommand

```bash
python main.py validate-data \
    --config examples/configs/yaml/02_ecommerce_relationships.yaml \
    --input output/02 \
    --tolerance 0.5 \
    --report-json output/02_validation.json \
    --verbose
```

Use this when you want to validate data that was generated separately
(or from a different pipeline). The `--report-json` flag dumps the full
result as JSON for downstream dashboards.

### 3. From Python

```python
from sdp.utils.config_parser import ConfigParser
from sdp.validators.gx_validator import validate_tables, format_report

parser = ConfigParser("config/my.yaml")
parser.load_config()
tables = parser.parse_tables()

report = validate_tables(tables, output_dir="output/run_01", row_count_tolerance=0.5)
print(format_report(report, verbose=True))

# or feed in pre-loaded DataFrames (handy in tests):
report = validate_tables(tables, dataframes={"users": users_df, "orders": orders_df})
```

## What the report looks like

```
=== Great Expectations validation: PASS ===
Tables: 3, expectations: 41, passed: 41, failed: 0

  PASS  customers                       rows=     200  expectations=11/11
  PASS  orders                          rows=     500  expectations=15/15
  PASS  order_items                     rows=    1500  expectations=15/15
```

A failure shows the offending samples:

```
=== Great Expectations validation: FAIL ===
Tables: 1, expectations: 14, passed: 10, failed: 4

  FAIL  users                           rows=      10  expectations=10/14
         X   values_to_be_unique [user_id]
             unexpected=2 (20.0%)
             samples: [2, 2]
         X   values_to_match_regex [email]
             unexpected=5 (50.0%)
             samples: ['bad', 'bad', 'bad', 'bad', 'bad']
         X   values_to_be_in_set [status]
             unexpected=1 (10.0%)
             samples: ['UNKNOWN']
         X   values_to_be_between [age]
             unexpected=1 (10.0%)
             samples: [17]
```

`samples` shows up to 5 offending values so you can debug without going
back to the generator.

## Tuning

| Knob | Default | Notes |
|------|---------|-------|
| `--gx-tolerance` (CLI) / `row_count_tolerance` (Python) | `0.5` | Row-count variance allowed: `0.5` ⇒ ±50%. Pass `0` for an exact match (rarely useful — SDV produces ±N rows when fitting is loose). |
| `--gx-fail-on-error` | off | If set, `generate` exits non-zero on validation failure. Pair with CI to gate releases. |
| `--report-json <path>` (`validate-data` only) | none | Dumps the full report as JSON for downstream dashboards. |
| `--verbose` | off | Prints every passed expectation too, not just failures. |

## Extending — adding new derived expectations

Open `validators/gx_validator.py` and edit `derive_expectations_for_column`.
Each rule is a 3-tuple `(kind, column, kwargs)`. Add a new `kind` and
register it in `_build_gx_expectation` so the validator knows which GX
class to instantiate.

### Adding a new format regex

1. Add the format → regex mapping in `_FORMAT_REGEX`:

   ```python
   _FORMAT_REGEX["GSTIN"] = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{3}$"
   ```

2. The format will then auto-apply whenever `special_rules` starts with
   that token (with or without `:locale` suffix).

### Adding a custom expectation kind

```python
# in derive_expectations_for_column:
if col.column_name == "some_special_column":
    out.append(("freshness_check", col.column_name, {"max_age_hours": 24}))

# in _build_gx_expectation:
if kind == "freshness_check":
    return MyCustomExpectation(column=column, **kwargs)
```

## What's intentionally not validated

- **Cross-table referential integrity (FKs).** GX is a
  table-at-a-time validator. The platform's existing
  `utils/data_validator.py` (run via `generate --validate`) handles FK
  validation already. The GX validator complements that, it doesn't
  replace it.
- **Distribution shape.** We check bounds (`min_value`/`max_value`) and
  type, but not that the values follow the declared `distribution:` block
  (e.g. "is this normal-distributed with mean 1200?"). For
  distribution-shape validation, use SDV's own quality reports or
  `scipy.stats` directly on the output.
- **Free-text content.** `NAME`, `ADDRESS`, `COMPANY` produce realistic
  but unconstrained strings. Validating those would require an LLM judge,
  which is out of scope for a deterministic validator.

## Tests

```bash
poetry run pytest tests/test_gx_validator.py -v
```

The test file has two halves:

- **Pure-function tests** for `derive_expectations_*` (run without GX
  installed)
- **Integration tests** using the actual GX validator (auto-skipped when
  GX isn't on the path)

Together they cover every mapping rule plus the happy/sad paths through
the validator.
