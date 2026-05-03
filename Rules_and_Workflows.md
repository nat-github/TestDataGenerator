# Rules, Derived Columns & Workflows

This guide covers the **column-level conditional logic** and **declarative state-machine workflows** added to the Synthetic Data Platform. These features let you make generated data realistic by expressing the kind of business rules you'd see in production — without writing Python.

> **Status:** Layer A (rules) and Layer B (derived) are shipped. Layer C (state-machine workflows) is documented here but planned for v2.

## Why this exists

Random generation produces realistic *individual* values but loses the relationships between them. A `status = ACTIVE` row might still have a `closure_date`. A `unit_price = 50, quantity = 3` row might have `total_amount = 17`. Three layers fix this:

| Layer | What it does | Example | Status |
|---|---|---|---|
| **A** — when/then rules | Override a column's value based on other columns in the same row | "if status = CLOSED, closure_date is between 2020-01-01 and today" | ✅ v1 |
| **B** — derived columns | Compute a column deterministically from other columns | "full_name = {first} + ' ' + {last}" | ✅ v1 |
| **C** — workflows | State-machine over multiple status/timestamp columns per row | "Order: PLACED → SHIPPED → DELIVERED, with timestamps that monotonically increase" | 🚧 v2 |

All three layers run **after** SDV / fallback generation and **after** FK resolution, so your foreign keys and parent-child cardinalities are stable when the rules fire.

---

## Layer A — when / then rules

Attach a `rules:` list to any column. Each rule is `{ when: …, then: … }`. Rules evaluate top-to-bottom; the **last matching rule wins** (so put more specific rules later).

### Minimal example

```yaml
columns:
  - name: status
    data_type: VA10
    business_values: [ACTIVE, SUSPENDED, CLOSED]

  - name: closure_date
    data_type: D
    nullable: true
    rules:
      - when: { status: { eq: ACTIVE } }
        then: { set_null: true }
      - when: { status: { in: [SUSPENDED, CLOSED] } }
        then: { min: 2020-01-01, max: today }   # date range override
```

### `when:` operators

| Operator | Meaning | Example |
|---|---|---|
| `eq` | equals | `{ status: { eq: ACTIVE } }` |
| `ne` | not equals | `{ status: { ne: CLOSED } }` |
| `in` | value in list | `{ tier: { in: [GOLD, PLATINUM] } }` |
| `not_in` | value not in list | `{ region: { not_in: [APAC, EU] } }` |
| `gt`, `gte`, `lt`, `lte` | numeric / date compare | `{ balance: { gt: 1000 } }` |
| `between` | inclusive range | `{ amount: { between: [100, 500] } }` |
| `is_null` | column is null (`true` or `false`) | `{ closed_at: { is_null: true } }` |
| `not_null` | column is not null | `{ phone: { not_null: true } }` |
| `matches` | regex match | `{ email: { matches: "@example\\.com$" } }` |

### Composite conditions: `and` / `or`

```yaml
rules:
  - when:
      and:
        - { status: { eq: ACTIVE } }
        - { tier: { in: [GOLD, PLATINUM] } }
    then: { min: 10000, max: 1000000 }     # premium customers, big balances

  - when:
      or:
        - { country: { eq: US } }
        - { country: { eq: CA } }
    then: { special_rules: PHONE }           # NA phone format
```

### Shorthand

`{ status: ACTIVE }` is equivalent to `{ status: { eq: ACTIVE } }`. Use it for simple equality.

### `then:` actions

| Action | Effect |
|---|---|
| `value: <const>` | Set the column to a fixed value |
| `set_null: true` | Force the column to `null` |
| `min: <n>`, `max: <n>` | Resample numeric range for this row |
| `business_values: [a, b, c]` | Pick uniformly from this list |
| `null_rate: 0.3` | 30% chance of producing `null`, otherwise fall through to other actions |
| `special_rules: <name>` | Apply a special rule (e.g. `IBAN`, `EMAIL`) |
| `distribution: { name: lognormal, mean: 5, sigma: 1 }` | Distribution override |

You can combine actions: `{ then: { null_rate: 0.1, min: 100, max: 5000 } }` produces null 10% of the time, otherwise samples from [100, 5000].

### Real-world examples

**Closure date depends on status:**
```yaml
- name: closure_date
  data_type: D
  nullable: true
  rules:
    - when: { status: { eq: ACTIVE } }
      then: { set_null: true }
    - when: { status: { eq: CLOSED } }
      then: { min: 2020-01-01, max: today }
```

**Balance range depends on account type:**
```yaml
- name: balance
  data_type: DC(18,2)
  rules:
    - when: { account_type: { eq: SAVINGS } }
      then: { min: 100, max: 50000 }
    - when: { account_type: { eq: CHECKING } }
      then: { min: 0, max: 10000 }
    - when: { account_type: { eq: PREMIUM } }
      then: { min: 10000, max: 1000000, distribution: { name: lognormal } }
```

**Phone format follows country:**
```yaml
- name: phone_number
  data_type: VA20
  rules:
    - when: { country: { eq: US } }
      then: { special_rules: PHONE:en_US }
    - when: { country: { eq: DE } }
      then: { special_rules: PHONE:de_DE }
    - when: { country: { eq: IN } }
      then: { special_rules: PHONE:en_IN }
```

**Probabilistic transaction channel:**
```yaml
- name: channel
  data_type: VA10
  rules:
    - when: { transaction_type: { eq: WIRE } }
      then: { business_values: [SWIFT, SWIFT, SWIFT, SWIFT, SWIFT, SWIFT, SWIFT, SEPA, SEPA, SEPA] }
      # Repeating values produces a 70/30 weight without a real distribution lib.
```

---

## Layer B — derived columns

Set `derived: <expression>` on a column. The column is **computed** from others — random generation is skipped for it.

### Two modes

**Template mode** — substitute `{col}` placeholders with their values:
```yaml
- name: full_name
  data_type: VA64
  derived: "{first_name} {last_name}"
```

**Expression mode** — prefix with `=` to evaluate Python-like arithmetic:
```yaml
- name: total_amount
  data_type: DC(18,2)
  derived: "={unit_price} * {quantity}"

- name: net_amount
  data_type: DC(18,2)
  derived: "={gross_amount} - {tax_amount}"

- name: age_years
  data_type: N3
  derived: "=years_between({birth_date}, today())"
```

### Expression syntax

- **Arithmetic:** `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Comparison:** `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`
- **Boolean:** `and`, `or`, `not`
- **Conditional:** `=A if {cond} else B`
- **Built-in functions:**
  - `today()` — date object for today
  - `years_between(d1, d2)` — integer difference
  - `lower(s)`, `upper(s)`, `len(s)`
  - `concat(a, b, c, ...)` — null-safe string concat
  - `coalesce(a, b, c, ...)` — first non-null value
  - `int(x)`, `float(x)`, `str(x)`, `round(x, n)`, `abs(x)`, `min(...)`, `max(...)`

> **What's *not* allowed:** imports, attribute access, list/dict comprehensions, lambdas. The evaluator rejects anything outside the whitelist — this is intentional. If you need fancier logic, write it as a special_rule in Python and reference it.

### Dependency ordering

Derived columns are sorted topologically — you can chain them:
```yaml
- name: full_name
  derived: "{first_name} {last_name}"

- name: display_name
  derived: "=upper({full_name})"

- name: handle
  derived: "=lower({first_name})"
```
The platform computes `full_name` first, then `display_name` (which references it), in the right order automatically.

> **Cycles** (A depends on B depends on A) are detected and broken silently with a warning in the logs.

### Tolerance

If a derived expression references a column that doesn't exist (typo), the evaluator returns `None` rather than crashing the whole run. Check the logs (`rule_evaluator: ...` lines) when a derived column comes out unexpectedly empty.

---

## Layer C — declarative workflows (planned)

> **Not yet implemented.** Documented here so you can shape your data model around it.

For datasets where rows have a **lifecycle** (orders, claims, loans, accounts), use a state-machine workflow:

```yaml
workflows:
  - name: order_lifecycle
    table: orders
    state_column: status
    timestamps:                    # column → state that produces it
      placed_ts:    PLACED
      shipped_ts:   SHIPPED
      delivered_ts: DELIVERED
      cancelled_ts: CANCELLED
    transitions:
      - { from: PLACED,  to: SHIPPED,    probability: 0.95 }
      - { from: PLACED,  to: CANCELLED,  probability: 0.05 }
      - { from: SHIPPED, to: DELIVERED,  probability: 0.98 }
      - { from: SHIPPED, to: RETURNED,   probability: 0.02 }
```

The generator will:
1. For each row, pick a terminal state weighted by transition probabilities.
2. Back-fill timestamps for every state visited (monotonically increasing).
3. Leave timestamps for unvisited states as `null`.

This produces realistic distributions: ~95% delivered, ~5% cancelled, ~2% returned-after-shipping, etc.

Until v2 ships, you can fake a workflow using rules + derived columns:
- Generate `status` as a categorical with weighted business_values.
- Use rules to null out timestamps that don't apply for that status.
- Use derived columns to ensure timestamps are monotonic per row.

---

## Authoring formats

Rules and derived columns work the same in **YAML**, **JSON**, and **Excel**. See:
- `Yaml_Config_Schema.md` — full YAML reference
- `Json_Config_Schema.md` — full JSON reference (and JSON Schema file at `schemas/sdp_config.schema.json`)

### YAML — natural shape
```yaml
- name: closure_date
  data_type: D
  nullable: true
  rules:
    - when: { status: { eq: CLOSED } }
      then: { value: "2024-06-15" }
```

### JSON — same fields, JSON syntax
```json
{
  "name": "closure_date",
  "data_type": "D",
  "nullable": true,
  "rules": [
    { "when": { "status": { "eq": "CLOSED" } }, "then": { "value": "2024-06-15" } }
  ]
}
```

### Excel — `rules` cell holds a JSON-encoded list
In the `Columns` sheet, add an optional `rules` column. The cell value is a JSON array (the same shape as the YAML/JSON `rules:` field):

| table_name | column_name | data_type | rules |
|---|---|---|---|
| accounts | closure_date | D | `[{"when":{"status":{"eq":"CLOSED"}},"then":{"value":"2024-06-15"}}]` |

The same applies to `derived` — a single optional column on the `Columns` sheet:

| ... | column_name | data_type | derived |
|---|---|---|---|
| | full_name | VA64 | `{first_name} {last_name}` |

> **Tip:** for complex Excel rule sets, author them in YAML/JSON for review, then use `enrich` or a small script to project them into the Excel cells.

---

## Performance notes

- The rule evaluator runs row-by-row in Python after generation. For 1M rows × 30 rules per table, expect ~5-10 extra seconds.
- Derived expressions are parsed once per column and cached, so 1M rows × 5 derived columns add ~2 seconds.
- If you need higher throughput, batch-vectorise rules in NumPy — the engine is intentionally simple to keep the API stable; vectorisation can be added later without breaking configs.

## Debugging

When generated data doesn't match expectations:

1. **Run `python main.py lint --config <your.yaml>`** — validates the config and reports row/column errors.
2. **Add a `note:` to each rule** — appears in the verbose log so you can see which rule fired for which row.
3. **Inspect the parquet** with `poetry run python readParquet.py output/run_01` — derived columns should match their expression; rule-affected columns should respect the `when:` condition.
4. **Check `rule_evaluator: ...` log lines** — they fire on bad operators, expression parse errors, or cycles.

---

## Cheat sheet

```yaml
# 1. Conditional value
- name: closed_on
  rules:
    - when: { status: { eq: CLOSED } }
      then: { min: 2020-01-01, max: today }
    - when: { status: { ne: CLOSED } }
      then: { set_null: true }

# 2. Conditional range
- name: balance
  rules:
    - when: { tier: { in: [GOLD, PLATINUM] } }
      then: { min: 50000, max: 5000000 }

# 3. Derived string
- name: full_name
  derived: "{first} {last}"

# 4. Derived arithmetic
- name: total
  derived: "={qty} * {price}"

# 5. Derived with function
- name: age
  derived: "=years_between({dob}, today())"

# 6. Composite condition
- name: discount
  rules:
    - when:
        and:
          - { customer_type: { eq: B2B } }
          - { order_amount: { gt: 10000 } }
      then: { value: 0.15 }
    - when: { customer_type: { eq: B2C } }
      then: { value: 0.05 }
```
