# YAML Config Schema

## Purpose

This document defines the **canonical runtime YAML format** for this project.

The goal is simple:

- allow YAML as an alternative input to Excel and JSON
- keep the **same generation core logic**
- normalize all three formats into the same internal configuration model

In other words:

- **Excel**, **YAML**, and **JSON** are three authoring formats
- the generator, validator, delta flow, and SCD2 flow remain the same

> **See also:**
> - `Json_Config_Schema.md` — JSON authoring (same fields, JSON syntax)
> - `Rules_and_Workflows.md` — column-level conditional rules and derived columns
> - `schemas/sdp_config.schema.json` — JSON Schema for IDE autocompletion / validation

---

## Design Principle

The runtime should treat both inputs the same way:

- Excel `Columns` sheet -> normalized internal config
- YAML `tables[].columns[]` -> normalized internal config

The downstream logic should not need to care whether the source was Excel or YAML.

---

## Recommended top-level YAML structure

```yaml
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 1000
  operation_column: operation_type

tables:
  - name: parent
    table_kind: dimension
    rows: 100
    business_key_columns: [id]
    primary_key_columns: [id]

    # Unified CDC block (preferred over scattered legacy fields)
    cdc:
      mode: snapshot           # snapshot | delta | scd2
      track: []                # SCD2-tracked columns; empty for non-scd2
      event_time: null         # column name used as monotonic ordering key
      partition_by: []         # partition key columns

    active: true
    notes: Example table
    columns:
      - name: id
        data_type: N10
        is_pk: true
        nullable: false
        is_business_key_component: true
      - name: status
        data_type: VA10
        business_values: A;B
        nullable: true
      # Layer A — same-row when/then rules (see Rules_and_Workflows.md)
      - name: closed_at
        data_type: D
        nullable: true
        rules:
          - when: { status: { eq: A } }
            then: { set_null: true }
          - when: { status: { eq: B } }
            then: { min: 2020-01-01, max: today }
      # Layer B — derived column (see Rules_and_Workflows.md)
      - name: status_label
        data_type: VA32
        derived: "=upper({status})"

relationships:
  - name: child_parent
    source_table: child
    source_columns: [parent_id]
    target_table: parent
    target_columns: [id]
    relationship_type: many_to_one
    preserve_on_delta: true
    active: true
    notes: Optional description
```

---

## Top-level sections

### `config_format`
Optional but recommended.

Recommended value:

```yaml
config_format: sdp-yaml-v1
```

This helps version the schema over time.

### `run_settings`
Optional.

Can be a mapping of setting name to value.

Example:

```yaml
run_settings:
  default_records_per_table: 1000
  save_model_artifact: true
  model_artifact_path: output/models
```

These are normalized into the same internal settings used by the Excel `Run_Settings` sheet.

### `tables`
Required for runtime YAML configs.

Each table contains:

- table metadata
- a `columns` list

### `relationships`
Optional.

If omitted, relationships can still be auto-derived from column-level FK definitions, similar to Excel legacy mode.

---

## Table object

Each item under `tables` may contain:

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Equivalent to Excel `table_name` |
| `table_kind` | no | e.g. `transactional`, `dimension`, `reference` |
| `description` | no | Optional description |
| `rows` (alias `row_count`) | no | Equivalent to Excel `row_count`. Ignored for anchor tables (see `source`). |
| `business_key_columns` | no | List or semicolon-separated string |
| `primary_key_columns` | no | List or semicolon-separated string |
| `cdc` | no | Unified change-data-capture block — see below |
| `source` | no | Path to an existing `.parquet`/`.csv` dataset — see *Anchored generation* below |
| `active` | no | Boolean, defaults to `true` |
| `notes` | no | Optional notes |
| `write_delta` | no | Boolean — when `generate --write-delta` runs, write **this table** as a Delta Lake table at `<output>/<table>/`. Tables without it stay as flat parquet. Read directly by the CLI; bypasses `config_parser`. |
| `delta_partition_col` | no | Override the Delta partition column for **this table only** (default: CLI flag `--delta-partition-col`, falls back to `BOOKING_TM`). Useful when different sources partition by different columns in the same run. Only meaningful with `write_delta: true`. |
| `columns` | yes | Column definitions |

### Anchored generation (`source:`)

When a table declares `source:`, it is **loaded verbatim** from a real dataset
instead of being generated. The other tables generate *around* it: their foreign
keys resolve against the anchor's real key values, and the anchor's real rows
also train the SDV synthesizer so generated tables mimic its distributions.

```yaml
tables:
  - name: customer
    source: data/real/customer.parquet   # loaded as-is — not generated
    primary_key_columns: [customer_id]
    columns:
      - { name: customer_id, data_type: N10, is_pk: true }
      - { name: full_name,   data_type: VA64 }
  - name: account
    rows: 5000
    columns:
      - { name: account_id,  data_type: N10, is_pk: true }
      - name: customer_id            # FK resolves against the REAL customer keys
        data_type: N10
        is_fk: true
        ref_table: customer
        ref_column: customer_id
```

Rules:
- Supported formats: `.parquet`, `.csv`. The path resolves as given, then
  relative to the config file's directory, then to the working directory.
- The source dataset **must contain every column** declared for the table;
  extra columns are dropped with a warning.
- An anchor table's `rows:` is ignored — its row count equals the source file's.
- An anchor table is exported alongside the generated tables, so the output
  folder is a complete, join-consistent set.

### `cdc:` block (recommended — replaces six legacy fields)

```yaml
cdc:
  mode: scd2                     # snapshot | delta | scd2
  track: [status, balance]       # SCD2 tracked columns
  event_time: last_modified_ts   # monotonic ordering key
  partition_by: [region]         # partition columns
```

| Field | Type | Notes |
|---|---|---|
| `mode` | `snapshot` \| `delta` \| `scd2` | Generation behaviour. `scd2` implies `delta`. |
| `track` | list / `;`-string | SCD2 tracked columns. |
| `event_time` | string | Column used as the event-time / ordering key. |
| `partition_by` | list / `;`-string | Partition key columns. |

### Legacy fields (still accepted)

The parser keeps these working for backward compatibility. New configs should prefer `cdc:`.

| Legacy | Replaced by |
|---|---|
| `generation_mode: snapshot/delta_ready/scd2_ready` | `cdc.mode` |
| `delta` / `delta_eligible` | `cdc.mode == delta` (or `scd2`) |
| `scd2` / `scd2_enabled` | `cdc.mode == scd2` |
| `track_changes` / `scd2_tracked_columns` | `cdc.track` |
| `partition_enabled`, `partition_columns` | `cdc.partition_by` |
| `event_time_column` | `cdc.event_time` |

---

## Column object

Each column under `tables[].columns[]` may contain:

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Equivalent to Excel `column_name` |
| `data_type` | yes | Same format as Excel, e.g. `N10`, `A34`, `DC(18,2)`, `DT` |
| `is_pk` | no | Boolean |
| `is_fk` | no | Boolean |
| `ref_table` | no | Required when FK is defined explicitly |
| `ref_column` | no | Required when FK is defined explicitly |
| `business_values` | no | Semicolon-separated values |
| `special_rules` | no | Same syntax as Excel `special_rules` |
| `min_value` | no | Numeric minimum |
| `max_value` | no | Numeric maximum |
| `nullable` | no | Boolean |
| `is_business_key_component` | no | Boolean |
| `event_time` | no | Boolean |
| `partition_role` | no | e.g. `partition_key`, `event_time` |
| `scd2_tracked` | no | Boolean |
| `rules` | no | List of `{when, then}` rules — Layer A. See `Rules_and_Workflows.md`. |
| `derived` | no | Expression string — Layer B. Template `"{first} {last}"` or `=`-prefixed expression `"={qty} * {price}"`. |

---

## Relationship object

Each item under `relationships` may contain:

| Field | Required | Notes |
|---|---|---|
| `name` | no | Optional relationship name |
| `source_table` | yes | Child/source table |
| `source_columns` | yes | List or semicolon-separated string |
| `target_table` | yes | Parent/target table |
| `target_columns` | yes | List or semicolon-separated string |
| `relationship_type` | no | Usually `many_to_one` |
| `preserve_on_delta` | no | Boolean |
| `active` | no | Boolean |
| `notes` | no | Optional notes |

---

## Normalization rules

The YAML loader should normalize into the same internal structures used by Excel parsing.

### Lists accepted as either:
- YAML list
- semicolon-separated string

Example:

```yaml
primary_key_columns: [ACCT_ID, NTRY_SEQ_NB]
```

or:

```yaml
primary_key_columns: ACCT_ID;NTRY_SEQ_NB
```

### Column inference helpers

The loader may infer some column flags from table metadata:

- if a column is listed in `primary_key_columns`, it may default `is_pk: true`
- if a column is listed in `business_key_columns`, it may default `is_business_key_component: true`
- if a column equals `event_time_column`, it may default `event_time: true`
- if a column appears in `scd2_tracked_columns`, it may default `scd2_tracked: true`

Even with inference, explicit column flags are preferred for clarity.

---

## Recommended usage guidance

### Use YAML when:
- you want version-control-friendly config
- you want easier review in PRs
- you want CI/CD or API integration
- you want to generate configs programmatically

### Use Excel when:
- business users need easy tabular authoring
- analysts are defining values/rules collaboratively
- the config is being drafted interactively

---

## Important note on sample YAML files

A documentation/sample YAML file may intentionally omit some columns for readability.

That is acceptable for:
- discussion
- design review
- illustrating the shape of the config

That is **not** ideal for runtime.

A runtime YAML config should fully define all columns needed for generation.

---

## Recommended next step

If YAML becomes a supported runtime input, keep this rule:

> Excel and YAML must both normalize to the same internal config contract before any generation logic runs.

