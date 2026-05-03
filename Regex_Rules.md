# Regex Rule Guide

## Purpose

This project supports **regex-driven synthetic values** through the `special_rules` column in the Excel `Columns` sheet.

Use this when you want generated values to follow a pattern such as:

- `AD123`
- `AB-2025-001`
- `NL12ABC3456`
- `INV-000123`
- `A1B2C3`

## Where to configure it

Use the **`special_rules`** column.

Recommended syntax:

```text
REGEX:<pattern>
```

Also supported:

```text
REGEX=<pattern>
```

## Combining regex with other special-rule directives

You can put **one value generator** and **optional modifiers** in the same `special_rules` cell.

Use this separator:

```text
;;
```

Examples:

```text
REGEX:ADR-\d{4}(-[A-Z]{2})?;;NULL_RATE=0.25
REGEX:(INV|CRN)-\d{6};;NULL_PCT=10
EMAIL;;NULL_RATE=0.05
```

Rules:

- one value-generating directive per cell
- optional modifiers may follow after `;;`
- currently supported modifiers are:
  - `NULL_RATE=<0..1>`
  - `NULL_PCT=<0..100>`
- `NULL_RATE` / `NULL_PCT` are not allowed on primary-key columns

Examples that are **not valid**:

```text
REGEX:AD\d{3};;EMAIL
REGEX:REL\d{4};;NULL_PCT=10   # invalid if the column is a PK
```

## Why `special_rules` and not `business_values`

- `business_values` is intended for **literal enumerated values** such as `A;B;C` or `ACTIVE;INACTIVE`.
- regex is a **generation rule**, not a literal list.
- therefore regex belongs in `special_rules`.

---

## Quick examples

| Goal | `special_rules` value | Example generated values |
|---|---|---|
| Prefix + 3 digits | `REGEX:AD\d{3}` | `AD123`, `AD004` |
| 2 letters + 4 digits | `REGEX:[A-Z]{2}\d{4}` | `AB1234`, `ZX0008` |
| Alternation | `REGEX:(AD|BC)\d{3}` | `AD123`, `BC456` |
| Invoice style | `REGEX:INV-\d{6}` | `INV-000123` |
| Country + digits + code | `REGEX:NL\d{2}[A-Z]{4}\d{4}` | `NL12ABCD3456` |
| Optional suffix | `REGEX:REF\d{4}(-[A-Z]{2})?` | `REF1234`, `REF1234-AB` |
| Mixed alphanumeric | `REGEX:[A-Z0-9]{8}` | `A1B2C3D4` |
| One of fixed codes | `REGEX:(NEW|ACT|CLS)` | `NEW`, `ACT` |
| Letter range | `REGEX:[A-F]{3}\d{2}` | `ABC12`, `FED09` |
| One or more digits after prefix | `REGEX:TXN\d+` | `TXN1`, `TXN482` |

---

## Supported regex features

This generator supports a practical subset of regex that works well for synthetic data generation.

### 1. Literal characters
```text
REGEX:AD123
```
Matches exactly:
- `AD123`

### 2. Character classes
```text
REGEX:[A-Z]{2}\d{3}
```
Supported examples:
- `[A-Z]`
- `[a-z]`
- `[0-9]`
- `[A-Z0-9]`
- `[A-F]`

### 3. Shorthand classes
Supported:
- `\d` = digit
- `\D` = non-digit from the generator's allowed character pool
- `\w` = word character `[A-Za-z0-9_]`
- `\W` = punctuation character from the generator's allowed punctuation pool
- `\s` = space
- `\S` = non-space alphanumeric

Examples:
```text
REGEX:AD\d{3}
REGEX:\w{8}
```

### 4. Quantifiers
Supported:
- `{n}` exactly n times
- `{n,m}` between n and m times
- `{n,}` at least n times
- `?` zero or one
- `*` zero or more
- `+` one or more

Examples:
```text
REGEX:AD\d{3}
REGEX:[A-Z]{2,4}\d{2}
REGEX:REF\d+
REGEX:ABC?
```

### 5. Alternation
```text
REGEX:(AD|BC)\d{3}
```
Example values:
- `AD123`
- `BC987`

### 6. Grouping
```text
REGEX:(NL|BE)\d{2}[A-Z]{4}\d{4}
```

### 7. Anchors
The generator accepts `^` and `$` and treats them as regex anchors.

Example:
```text
REGEX:^AD\d{3}$
```

---

## Unsupported regex features

The generator **does not** support advanced regex constructs such as:

- lookaheads / lookbehinds
  - `(?=...)`
  - `(?!...)`
  - `(?<=...)`
- backreferences
  - `\1`, `\2`
- named groups
- conditional expressions

Examples that are **not supported**:

```text
REGEX:(?=AD)AD\d{3}
REGEX:(AB)\1
```

These patterns will fail validation during configuration loading.

---

## Length behavior

Regex generation still respects the configured column length.

### Example
If the column data type is:

```text
A5
```

and the regex is:

```text
REGEX:AD\d{3}
```

this is valid because generated values are length 5.

If the regex is:

```text
REGEX:[A-Z]{10}
```

this is invalid for `A5`, and configuration validation will fail.

### Important note for `VA...`
`VA254` means:
- maximum length = 254
- not exact length = 254

So regex-generated values can be shorter, as long as they do not exceed the configured maximum.

---

## Excel examples

### Example 1: simple code column
| table_name | column_name | data_type | special_rules |
|---|---|---|---|
| orders | order_code | A5 | `REGEX:AD\d{3}` |

### Example 2: mixed document number
| table_name | column_name | data_type | special_rules |
|---|---|---|---|
| documents | doc_id | VA20 | `REGEX:(INV|CRN)-\d{6}` |

### Example 3: account-like code
| table_name | column_name | data_type | special_rules |
|---|---|---|---|
| accounts | account_ref | VA16 | `REGEX:[A-Z]{2}\d{2}[A-Z0-9]{8}` |

---

## Recommended patterns by use case

### Simple prefixed ID
```text
REGEX:CUS\d{6}
```

### Fixed business code
```text
REGEX:(NEW|ACT|SUS|CLS)
```

### Alphanumeric reference
```text
REGEX:[A-Z0-9]{10}
```

### Human-readable reference
```text
REGEX:REF-[A-Z]{3}-\d{4}
```

### Optional branch suffix
```text
REGEX:ACC\d{6}(-[A-Z]{2})?
```

---

## Validation behavior

Regex rules are validated during config loading.

Validation checks:
- pattern syntax is valid
- unsupported constructs are rejected
- minimum generated length does not exceed the configured column length

If invalid, the run will stop with a clear error message.

---

## Best practices

1. Use `special_rules` for regex generation.
2. Keep patterns readable and business-focused.
3. Match the regex length to the configured data type length.
4. Use `business_values` only for literal lists.
5. Prefer explicit formats like `AD\d{3}` over overly broad patterns like `.*`.

---

## Comparison: business values vs regex

### Use `business_values` for fixed lists
```text
ACTIVE;INACTIVE;PENDING
```

### Use `special_rules` with regex for patterned generation
```text
REGEX:AD\d{3}
```

---

## One-line summary

Use `special_rules = REGEX:<pattern>` whenever you want synthetic data to follow a reusable format such as prefixes, codes, numbers, and structured identifiers.

---

## Mimesis-backed rules (optional)

[Mimesis](https://mimesis.name/) is an alternative atomic-value generator that complements Faker — broader locale coverage, faster for some workloads. Install with:

```bash
poetry install --extras mimesis
```

Once installed, prefix any rule with `MIMESIS_` to route it through Mimesis instead of Faker. Plain `NAME`, `EMAIL`, etc. continue to use Faker — there is no behaviour change for existing configs.

### Supported `MIMESIS_*` rules

| Category | Rules |
|---|---|
| **People** | `MIMESIS_NAME`, `MIMESIS_FULL_NAME`, `MIMESIS_FIRST_NAME`, `MIMESIS_LAST_NAME`, `MIMESIS_USERNAME`, `MIMESIS_EMAIL`, `MIMESIS_PHONE`, `MIMESIS_GENDER`, `MIMESIS_TITLE`, `MIMESIS_OCCUPATION`, `MIMESIS_NATIONALITY` |
| **Address** | `MIMESIS_ADDRESS`, `MIMESIS_CITY`, `MIMESIS_STATE`, `MIMESIS_COUNTRY`, `MIMESIS_COUNTRY_CODE`, `MIMESIS_POSTCODE`, `MIMESIS_ZIP`, `MIMESIS_STREET`, `MIMESIS_STREET_NUMBER`, `MIMESIS_LATITUDE`, `MIMESIS_LONGITUDE` |
| **Finance / payment** | `MIMESIS_COMPANY`, `MIMESIS_CURRENCY`, `MIMESIS_PRICE`, `MIMESIS_STOCK_TICKER`, `MIMESIS_CREDIT_CARD`, `MIMESIS_CC_EXP`, `MIMESIS_CVV` |
| **Internet** | `MIMESIS_URL`, `MIMESIS_IPV4`, `MIMESIS_IPV6`, `MIMESIS_MAC`, `MIMESIS_USER_AGENT` |
| **Identifiers** | `MIMESIS_UUID`, `MIMESIS_TOKEN`, `MIMESIS_HASH` |
| **Text** | `MIMESIS_WORD`, `MIMESIS_SENTENCE`, `MIMESIS_TEXT`, `MIMESIS_QUOTE`, `MIMESIS_COLOR` |
| **Datetime** | `MIMESIS_DATE`, `MIMESIS_TIME`, `MIMESIS_DATETIME`, `MIMESIS_TIMEZONE` |

> IBAN, BIC, and SWIFT are **not** in Mimesis — keep using the existing Faker-backed `IBAN`, `BIC`, `SWIFT` rules for those.

### Locale suffix

The same `:<locale>` syntax used elsewhere works here:

```text
MIMESIS_FIRST_NAME:de_DE     # German first name
MIMESIS_CITY:fr_FR           # French city
MIMESIS_LAST_NAME:ja_JP      # Japanese surname
```

Faker-style locale codes (`de_DE`, `en_US`, `pt_BR`, `zh_CN`, `ru_RU`, `nl_NL`, etc.) are mapped automatically. Unknown locales fall back to English silently.

### Choosing between Faker and Mimesis

- **Stick with Faker** when you need IBAN / BIC / SWIFT, banking codes, or any of the 60+ existing locale-aware rules already in `helpers.py`.
- **Use Mimesis** when you need very broad locale coverage (e.g., Estonian names, Persian addresses), large-volume generation where speed matters, or rules Faker doesn't carry (`MIMESIS_STOCK_TICKER`, `MIMESIS_USER_AGENT`).
- **Mix both freely.** Different columns in the same table can use different generators.

