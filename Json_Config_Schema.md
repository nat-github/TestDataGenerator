# JSON Config Schema

The Synthetic Data Platform accepts the same configuration in three authoring formats: **YAML**, **JSON**, and **Excel**. This document is the canonical reference for the JSON form (`config_format: sdp-json-v1`).

JSON is a strict subset of YAML — every YAML config that uses simple values (no anchors, no multi-document streams) maps cleanly to JSON. The runtime treats both identically once parsed.

> **For a hands-on example see `config/sample_rules_and_cdc.json`.**
> **For IDE validation, point your editor at `schemas/sdp_config.schema.json`** (JSON Schema 2020-12).

---

## When to choose JSON

| Use JSON when... | Use YAML when... | Use Excel when... |
|---|---|---|
| Programmatic generation (CI/CD, scripts) | Hand-authored configs in PRs | Business / analyst authoring |
| Frontend UIs that produce config | Rich comments needed | Tabular review with stakeholders |
| Tooling that already speaks JSON | Most readable for humans | Last-mile edits before a run |

The JSON Schema file lets VS Code, IntelliJ, and most modern editors auto-complete and validate as you type.

---

## Top-level structure

```json
{
  "config_format": "sdp-json-v1",
  "config_version": "1.0",
  "run_settings": {
    "default_records_per_table": 1000,
    "seed": 42
  },
  "tables": [ ... ],
  "relationships": [ ... ]
}
```

| Key | Required | Notes |
|---|---|---|
| `config_format` | recommended | `"sdp-json-v1"`. `"fdl-json-v1"` accepted as legacy alias. |
| `config_version` | optional | Free-form version string, surfaced in logs. |
| `run_settings` | optional | Same shape as YAML / Excel `Run_Settings`. |
| `tables` | required | List of table objects. |
| `relationships` | optional | Explicit foreign-key relationships. Inferred from `is_fk` if omitted. |

---

## Table object

```json
{
  "name": "accounts",
  "description": "Customer accounts",
  "table_kind": "transactional",
  "rows": 10000,
  "primary_key_columns": ["account_id"],
  "business_key_columns": ["customer_id", "opened_date"],
  "cdc": {
    "mode": "scd2",
    "track": ["status", "balance", "tier"],
    "event_time": "last_modified_ts",
    "partition_by": ["region"]
  },
  "columns": [ ... ]
}
```

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Lower-case table identifier. |
| `description` | no | Free-form. |
| `table_kind` | no | `transactional` \| `dimension` \| `reference` \| `fact` \| `bridge`. |
| `rows` (alias `row_count`) | no | Number of rows to generate. Falls back to `default_records_per_table`. |
| `primary_key_columns` | no | List or `";"`-separated string. |
| `business_key_columns` | no | List or `";"`-separated string. |
| `cdc` | no | Unified change-data-capture block. See below. |
| `columns` | yes | List of column objects. |
| `active` | no | Boolean. Inactive tables are skipped. Default `true`. |
| `notes` | no | Free-form. |

### Legacy fields (still accepted)

These are kept for backward compatibility. Prefer the `cdc:` block in new configs.

| Legacy field | Replaced by |
|---|---|
| `delta_eligible`, `delta` | `cdc.mode == "delta"` (or `"scd2"`) |
| `scd2_enabled`, `scd2` | `cdc.mode == "scd2"` |
| `scd2_tracked_columns`, `track_changes` | `cdc.track` |
| `partition_enabled`, `partition_columns` | `cdc.partition_by` |
| `event_time_column` | `cdc.event_time` |
| `generation_mode` | `cdc.mode` |

The parser merges both; legacy fields take precedence when both are set, so migration is incremental.

---

## CDC block (unified change-data-capture)

```json
"cdc": {
  "mode": "scd2",
  "track": ["status", "balance"],
  "event_time": "last_modified_ts",
  "partition_by": ["region"]
}
```

| Field | Type | Notes |
|---|---|---|
| `mode` | `"snapshot"` \| `"delta"` \| `"scd2"` | Generation behaviour. `scd2` implies `delta`. |
| `track` | array of strings | Columns whose changes trigger a new SCD2 version. Empty when `mode != scd2`. |
| `event_time` | string | Column name used as the monotonic event-time / ordering key. |
| `partition_by` | array of strings | Partition columns. Empty disables partitioning. |

**Modes:**
- **`snapshot`** — full overwrite each run; no I/U/D rows; no effective dating.
- **`delta`** — each run produces inserts/updates/deletes against the previous snapshot, with an `operation_type` column.
- **`scd2`** — full effective-dated history with `effective_from_ts`, `effective_to_ts`, `is_current`, `version_num`. Implies delta tracking.

---

## Column object

```json
{
  "name": "balance",
  "data_type": "DC(18,2)",
  "min_value": 0,
  "max_value": 1000000,
  "nullable": false,
  "rules": [
    { "when": { "tier": { "in": ["GOLD", "PLATINUM"] } },
      "then": { "min": 50000, "max": 5000000 } }
  ]
}
```

### Standard fields

| Field | Notes |
|---|---|
| `name` (alias `column_name`) | Column identifier. |
| `data_type` (alias `type`) | `N10`, `VA64`, `DC(18,2)`, `D`, `DT`, `TS`, etc. |
| `is_pk` (alias `pk`) | Primary key column. |
| `is_fk` (alias `fk`) | Foreign key column. |
| `ref_table`, `ref_column` | FK target. |
| `business_values` (alias `values`) | List or `";"`-separated string. |
| `special_rules` | `IBAN`, `EMAIL`, `PHONE:de_DE`, `REGEX:\\d{4}`, etc. |
| `min_value` (alias `min`), `max_value` (alias `max`) | Numeric / date range. |
| `nullable` | Boolean. |
| `is_business_key_component` | Marks the column as part of the business key. |

### CDC-related fields (column-level)

| Field | Notes |
|---|---|
| `event_time` | This column is the table's event-time column. |
| `partition_role` | `partition_key` \| `event_time`. |
| `scd2_tracked` | Changes to this column produce a new SCD2 version. |

### Rules / derived (Layer A + B)

| Field | Notes |
|---|---|
| `rules` | List of `{ when, then }` rule objects. See `Rules_and_Workflows.md`. |
| `derived` | Expression string. Template (`"{first} {last}"`) or `=`-prefixed expression (`"={qty} * {price}"`). |

---

## Relationship object

```json
{
  "name": "accounts_customers",
  "source_table": "accounts",
  "source_columns": ["customer_id"],
  "target_table": "customers",
  "target_columns": ["customer_id"],
  "relationship_type": "many_to_one",
  "preserve_on_delta": true,
  "active": true
}
```

| Field | Required | Notes |
|---|---|---|
| `source_table`, `target_table` | yes | Lower-case names. |
| `source_columns`, `target_columns` | yes | Lists; lengths must match. Singular forms `source_column` / `target_column` accepted. |
| `relationship_type` (alias `cardinality`) | no | `one_to_one` \| `one_to_many` \| `many_to_one` \| `many_to_many`. |
| `preserve_on_delta` | no | Keep relationship intact across delta runs. |
| `name` | no | Optional label, useful in logs. |

If `relationships` is omitted, the parser auto-derives them from any `is_fk: true` column with `ref_table` + `ref_column`.

---

## Complete example

```json
{
  "config_format": "sdp-json-v1",
  "config_version": "1.0",
  "run_settings": {
    "default_records_per_table": 100,
    "seed": 42
  },
  "tables": [
    {
      "name": "customers",
      "rows": 50,
      "primary_key_columns": ["customer_id"],
      "cdc": { "mode": "snapshot" },
      "columns": [
        { "name": "customer_id", "data_type": "N10", "is_pk": true, "nullable": false },
        { "name": "first_name", "data_type": "VA32", "special_rules": "FIRST_NAME" },
        { "name": "last_name", "data_type": "VA32", "special_rules": "LAST_NAME" },
        { "name": "full_name", "data_type": "VA64", "derived": "{first_name} {last_name}" },
        { "name": "tier", "data_type": "VA10", "business_values": ["BRONZE", "SILVER", "GOLD", "PLATINUM"] }
      ]
    },
    {
      "name": "accounts",
      "rows": 200,
      "primary_key_columns": ["account_id"],
      "cdc": {
        "mode": "scd2",
        "track": ["status", "balance"],
        "event_time": "last_modified"
      },
      "columns": [
        { "name": "account_id", "data_type": "N12", "is_pk": true, "nullable": false },
        { "name": "customer_id", "data_type": "N10", "is_fk": true, "ref_table": "customers", "ref_column": "customer_id" },
        { "name": "status", "data_type": "VA10", "business_values": ["ACTIVE", "SUSPENDED", "CLOSED"] },
        {
          "name": "balance",
          "data_type": "DC(18,2)",
          "min_value": 0,
          "max_value": 100000,
          "rules": [
            { "when": { "status": { "eq": "CLOSED" } }, "then": { "value": 0 } }
          ]
        },
        {
          "name": "closure_date",
          "data_type": "D",
          "nullable": true,
          "rules": [
            { "when": { "status": { "eq": "ACTIVE" } }, "then": { "set_null": true } },
            { "when": { "status": { "in": ["CLOSED", "SUSPENDED"] } },
              "then": { "min": "2020-01-01", "max": "today" } }
          ]
        },
        { "name": "last_modified", "data_type": "TS" }
      ]
    }
  ],
  "relationships": [
    {
      "source_table": "accounts",
      "source_columns": ["customer_id"],
      "target_table": "customers",
      "target_columns": ["customer_id"],
      "relationship_type": "many_to_one"
    }
  ]
}
```

Run it with the same CLI as YAML / Excel:

```bash
python main.py generate --config config/sample_rules_and_cdc.json --output output/run_01
```

---

## IDE validation

Add this at the top of your JSON file (or configure your editor's JSON Schema mapping):

```json
{
  "$schema": "./schemas/sdp_config.schema.json",
  "config_format": "sdp-json-v1",
  ...
}
```

VS Code with the built-in JSON Language Server will give you autocompletion for every field, enum value, and operator name.

## Schema validation in `lint`

The same schema is enforced by `lint` for YAML and JSON configs:

```bash
python main.py lint --config config/orders.yaml                  # violations as warnings
python main.py lint --config config/orders.yaml --strict-schema  # violations as errors, exit 1
```

```
[WARN]  (sheet=schema, field=workflows[0]) schema: Additional properties are not allowed ('typo_field' was unexpected)
[WARN]  (sheet=schema, field=workflows[0].step_hours) schema: [1] is too short
[WARN]  (sheet=schema, field=workflows[0].transitions[0].probability) schema: 1.7 is greater than the maximum of 1
[WARN]  (sheet=schema, field=workflows[0].transitions[1]) schema: 'from' is a required property
```

### Why warnings by default

The parser is deliberately tolerant — an unrecognised key is skipped so one
bad rule cannot stop a run. That is right for generation and wrong for
authoring, where a mistyped `transitons:` silently does nothing.

Schema violations are therefore reported but non-fatal by default, so a
working config never breaks on an upgrade. Use `--strict-schema` in CI,
where you want the typo to fail the build.

**Excel configs are skipped** — the schema describes a YAML/JSON document,
and a workbook has no such document to validate. Excel configs still get
every semantic check `lint` performs.

Every config in `examples/configs/` is validated against this schema by the
test suite, so the shipped examples cannot drift from it.

---

## Conversion between formats

```bash
# YAML → JSON
python -c "import yaml, json, sys; print(json.dumps(yaml.safe_load(open(sys.argv[1])), indent=2, default=str))" config/Acct_bkng.yaml > config/Acct_bkng.json

# JSON → YAML
python -c "import yaml, json, sys; print(yaml.safe_dump(json.load(open(sys.argv[1])), sort_keys=False))" config/Acct_bkng.json > config/Acct_bkng.yaml
```

Both round-trip cleanly because they normalise to the same internal model.
