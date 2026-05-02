# YAML Config Schema

## Purpose

This document defines the **canonical runtime YAML format** for this project.

The goal is simple:

- allow YAML as an alternative input to Excel
- keep the **same generation core logic**
- normalize YAML into the same internal configuration model already used by the Excel parser

In other words:

- **Excel** and **YAML** are two authoring formats
- the generator, validator, delta flow, and SCD2 flow remain the same

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
    row_count: 100
    generation_mode: snapshot
    business_key_columns: [id]
    primary_key_columns: [id]
    partition_enabled: false
    partition_columns: []
    event_time_column: null
    scd2_enabled: false
    scd2_tracked_columns: []
    delta_eligible: true
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
| `row_count` | no | Equivalent to Excel `row_count` |
| `generation_mode` | no | e.g. `snapshot`, `delta_ready`, `scd2_ready` |
| `business_key_columns` | no | List or semicolon-separated string |
| `primary_key_columns` | no | List or semicolon-separated string |
| `partition_enabled` | no | Boolean |
| `partition_columns` | no | List or semicolon-separated string |
| `event_time_column` | no | Column name |
| `scd2_enabled` | no | Boolean |
| `scd2_tracked_columns` | no | List or semicolon-separated string |
| `delta_eligible` | no | Boolean |
| `active` | no | Boolean, defaults to `true` |
| `notes` | no | Optional notes |
| `columns` | yes | Column definitions |

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

