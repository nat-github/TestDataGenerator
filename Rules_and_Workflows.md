# Rules, Derived Columns & Workflows

This guide covers the **column-level conditional logic** and **declarative state-machine workflows** added to the Synthetic Data Platform. These features let you make generated data realistic by expressing the kind of business rules you'd see in production — without writing Python.

> **Status:** all three layers are shipped. Layer C (state-machine workflows) is YAML/JSON only — Excel authoring is not yet supported.

## Why this exists

Random generation produces realistic *individual* values but loses the relationships between them. A `status = ACTIVE` row might still have a `closure_date`. A `unit_price = 50, quantity = 3` row might have `total_amount = 17`. Three layers fix this:

| Layer | What it does | Example | Status |
|---|---|---|---|
| **A** — when/then rules | Override a column's value based on other columns in the same row | "if status = CLOSED, closure_date is between 2020-01-01 and today" | ✅ v1 |
| **B** — derived columns | Compute a column deterministically from other columns | "full_name = {first} + ' ' + {last}" | ✅ v1 |
| **C** — workflows | State-machine over multiple status/timestamp columns per row | "Order: PLACED → SHIPPED → DELIVERED, with timestamps that monotonically increase" | ✅ v1 (YAML/JSON) |

All three layers run **after** SDV / fallback generation and **after** FK resolution, so your foreign keys and parent-child cardinalities are stable when the rules fire. Within that pass the order is **C → A → B**: a workflow assigns the lifecycle, then rules react to it, then derived columns compute from the result.

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

## Layer C — declarative workflows

> **Shipped.** YAML and JSON configs only — Excel authoring is not yet
> supported. Validated by `python main.py lint`.

For datasets where rows have a **lifecycle** (orders, claims, loans,
accounts), a workflow replaces independent column draws with a walk through
a state machine.

Without it, random generation produces a CANCELLED order that carries a
delivery timestamp, or a DELIVERED order that was never shipped. Those rows
are the ones that make a downstream test pass when it should fail.

```yaml
workflows:
  - name: order_lifecycle
    table: orders
    state_column: status
    start_state: PLACED                        # optional — inferred when unambiguous
    start_between: ["2024-01-01", "2024-12-31"]  # optional — window for the first timestamp
    step_hours: [2, 96]                        # optional — gap between consecutive states

    timestamps:                                # column → state that produces it
      placed_ts:    PLACED
      shipped_ts:   SHIPPED
      delivered_ts: DELIVERED
      cancelled_ts: CANCELLED
      returned_ts:  RETURNED

    transitions:
      - { from: PLACED,    to: SHIPPED,   probability: 0.93 }
      - { from: PLACED,    to: CANCELLED, probability: 0.07 }
      - { from: SHIPPED,   to: DELIVERED, probability: 0.97 }
      - { from: DELIVERED, to: RETURNED,  probability: 0.02 }
```

Runnable example: `examples/configs/yaml/12_workflow_lifecycle.yaml`.

### What it guarantees

For every row:

1. The path starts at `start_state` and follows **only declared transitions**.
2. Each visited state's timestamp column gets a value, **strictly increasing**
   along the path.
3. Columns for states **not** visited are `NULL`.
4. The `state_column` is set to the terminal state.

So `delivered_ts > shipped_ts > placed_ts` holds by construction, a
CANCELLED row has no `shipped_ts` or `delivered_ts`, and a DELIVERED row
always has a `shipped_ts`.

### Probabilities: the residual rule

Probabilities are read **per source state**. Where the outgoing
probabilities sum to less than 1, **the remainder is the chance of stopping
in that state**.

In the example above:

| State | Outgoing | Residual | Meaning |
|---|---|---|---|
| `PLACED` | 0.93 + 0.07 = 1.0 | 0 | always moves on |
| `SHIPPED` | 0.97 | 0.03 | 3% stay in transit |
| `DELIVERED` | 0.02 | 0.98 | 98% stay delivered |
| `CANCELLED` | none | 1.0 | terminal |

That is how a state stays non-terminal in the schema while still absorbing
most of its rows. A state with no outgoing transitions is terminal outright.

A measured run of the shipped example, 300 rows:

```
🔀 Workflow order_lifecycle → orders: CANCELLED=22, DELIVERED=270, RETURNED=3, SHIPPED=5
```

### Fields

| Field | Required | Meaning |
|---|:---:|---|
| `name` | ✅ | Identifier, used in logs and the generation report |
| `table` | ✅ | Table the workflow applies to |
| `state_column` | ✅ | Column that receives the terminal state. **Overwritten** — whatever generation produced is replaced |
| `transitions` | ✅ | List of `{from, to, probability}`. `probability` defaults to 1.0 |
| `timestamps` | — | `column: state` map. Omit for a state-only workflow |
| `start_state` | — | Inferred when exactly one state is never a transition target; **required otherwise** (e.g. any cyclic workflow) |
| `start_between` | — | `[min, max]` ISO dates for the first timestamp. Defaults to the last 365 days |
| `step_hours` | — | `[min, max]` gap between consecutive state timestamps. Defaults to `[1, 72]` |

### Cycles

Cycles are allowed — a claim can reopen, an account can be suspended and
reactivated:

```yaml
transitions:
  - { from: OPEN,   to: CLOSED, probability: 0.8 }
  - { from: CLOSED, to: OPEN,   probability: 0.3 }
```

Two consequences:

- `start_state` **must be declared**, because no state is a natural root.
- Walks are capped at 50 steps. Rows that hit the cap are counted and
  logged as `truncated_walks`; a revisited state's timestamp holds the
  **last** time it was entered.

### Ordering against Layers A and B

Layer C runs **first**, then Layer A rules, then Layer B derived columns.

That ordering is deliberate: rules and derived columns can react to the
state the workflow assigned. A rule keyed on `status == CANCELLED` sees the
workflow's verdict, not a random draw.

The corollary is that **a Layer A rule targeting the state column will
override the workflow.** If you have both, the rule wins — which is
occasionally what you want, and otherwise a bug worth knowing about.

### Validation

`python main.py lint` reports, as errors:

- a `state_column` or `timestamps` column absent from the table
- a `timestamps` entry pointing at a state no transition mentions
- outgoing probabilities summing above 1.0
- a `start_state` that cannot be inferred and was not declared
- `step_hours` that is not `[min, max]`

A workflow that fails to parse is skipped with a warning at load time and
reported as an error by `lint` — the same policy as `rules:`.

### Limitations

- **YAML/JSON only.** There is no `Workflows` Excel sheet yet.
- **One workflow per table.** Two workflows naming the same table both run,
  and the second overwrites the first's state column. Don't.
- **Timestamps are generated, not derived from existing values.** The
  workflow owns those columns entirely.
- **No per-state dwell distributions.** `step_hours` is a single uniform
  range for every transition.

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
