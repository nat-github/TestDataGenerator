# Pending Items

This document captures the pending items discussed for evolving the current utility into a stronger product/platform.

---

# 1. Already Completed

These items are **done** and therefore not pending anymore:

- regex support added to `special_rules`
- `;;` delimiter convention clarified
- merged-branch regression in `generators/data_generator.py` fixed
- SDV sample API compatibility fixed
- bigint/decimal parquet export issues fixed
- regex correctness restored on the **SDV generation path**
- test suite passing
- standard `generate` working
- `delta` working
- `scd2` working
- legacy CLI compatibility working
- `PRD_Roadmap.md` created

---

# 2. Immediate Engineering Pending Items

## A. Deterministic / Reproducible Generation
**Pending**
- add global seed support
- add per-table seed support
- ensure same config + same seed => same output

**Why important**
- critical for QA repeatability
- needed for enterprise trust

---

## B. Explainability / Generation Audit Report
**Pending**
- per-column generation trace:
  - SDV-generated
  - fallback-generated
  - regex-overwritten
  - business-values-driven
  - FK-reconciled
- structured generation summary artifact
- table-level and run-level conformance summary

**Why important**
- makes the utility auditable
- helps debugging and stakeholder trust

---

## C. Better Workbook Validation UX
**Pending**
- report errors with exact:
  - sheet name
  - row number
  - column name
- add config lint/check mode
- improve clarity of invalid regex / invalid datatype / bad relationship messages

**Why important**
- biggest usability gain for non-developers

---

## D. Remaining Technical Warnings / Cleanup
**Pending**
- migrate Pydantic v1 `@validator` usage to v2 style
- reduce/deal with `rdt` deprecation warnings
- clean metadata overwrite warning for `output/models/metadata.json`
- optional cleanup of static-analysis/code-quality warnings in `generators/data_generator.py`

**Why important**
- improves maintainability
- reduces future breakage risk

---

## E. Streaming / Performance Hardening
**Pending**
- strengthen chunked/stream generation path
- benchmark large workloads
- memory-safe export for huge datasets
- improve performance metrics collection

**Why important**
- needed before positioning as enterprise-scale

---

# 3. Product Roadmap Pending Items

These align with `PRD_Roadmap.md` and remain future work.

## Phase 1 Pending
- formal constraint fidelity layer as a named architecture layer
- explainability/reporting
- deterministic runs
- better validation UX
- packaging/stability hardening

---

## Phase 2 Pending
- conditional rules
  - e.g. country-specific formats
- cross-column dependency engine
- weighted business values
- temporal dependency rules
- uniqueness scopes beyond PK
- data quality/distribution profiles
- finance/domain packs
- enhanced delta/SCD2 scenario controls
- config templates / workbook wizard / schema import

---

## Phase 3 Pending
- mature streaming generation
- parallel generation/export
- formal benchmarking suite
- observability metrics
- CI/CD integration patterns

---

## Phase 4 Pending
- API layer
- config/version catalog
- governance workflow
- reusable generation recipes
- collaboration/shared template support
- dataset registry

---

## Phase 5 Pending
- policy-driven synthetic data
- scenario simulation
- multi-modal input support
- synthetic data assurance scoring
- marketplace/template ecosystem

---

# 4. GenAI / LLM Integration Pending Items

These are the strongest “smart platform” opportunities discussed.

## A. Natural Language to Config Generation
**Pending**
Support prompts like:

> Generate a banking transactions dataset with customer, account, and transaction tables, with NL IBANs, 1M rows, parquet output, delta-ready.

The system should be able to:
- generate workbook config
- suggest tables/columns
- suggest datatypes
- suggest relationships
- suggest regex/business rules

**Value**
- major usability breakthrough
- lowers onboarding effort drastically

---

## B. LLM-Assisted Workbook Completion
**Pending**
Given a partially filled workbook, use LLM assistance to:
- infer missing datatypes
- infer likely PK/FK columns
- suggest regex patterns
- suggest `business_values`
- suggest null rates
- identify bad or contradictory rules

**Value**
- makes workbook authoring much faster
- helps business analysts

---

## C. Domain-Aware Rule Recommendation Engine
**Pending**
Use LLM/GenAI to recommend rules by column meaning:
- `iban` -> IBAN rule
- `bic` -> BIC rule
- `country_code` -> ISO country values
- `txn_ref` -> regex-like business reference
- `customer_status` -> weighted status values

**Value**
- faster config building
- higher realism

---

## D. NL-to-Test-Scenario Generation
**Pending**
Prompt-driven scenario generation such as:
- create suspicious AML-like cash deposit patterns
- simulate monthly account booking changes
- generate SCD2 history for customer status transitions
- simulate late-arriving delta changes

**Value**
- extremely powerful for QA and data engineering use cases
- makes this more than a static data generator

---

## E. LLM-Based Config Validation Assistant
**Pending**
An assistant that explains config issues in plain English:
- what is wrong
- why it is wrong
- how to fix it
- optionally auto-fix proposals

**Value**
- huge usability improvement
- ideal for enterprise onboarding

---

## F. Semantic Schema Ingestion
**Pending**
Use LLMs to ingest:
- parquet schema
- SQL DDL
- API specs
- data dictionaries
- Excel column glossaries

Then propose:
- tables
- relationships
- types
- rules
- sample values

**Value**
- big accelerant for adoption

---

## G. Synthetic Dataset Copilot
**Pending**
Interactive assistant for questions like:
- why did this column become null?
- which rules were applied here?
- why was FK overwritten?
- which columns were repaired after SDV?
- what changed between snapshot and delta?

**Value**
- turns the tool into a smart platform
- highly differentiated

---

## H. Prompt-to-Regex / Prompt-to-Rule Support
**Pending**
User says:
- make invoice ids like `INV-000123`
- customer ids should look like `CUS` + 8 digits
- optional suffix for branch code

System generates:
- regex
- special rule
- null modifier suggestions

**Value**
- bridges business language to technical config

---

# 5. Recommended GenAI / LLM Priority Order

## Highest-value first
1. **LLM-assisted workbook completion**
2. **Natural language to config generation**
3. **LLM-based config validation assistant**
4. **Domain-aware rule recommendation**
5. **Prompt-to-regex / prompt-to-rule**

## Later
6. NL-to-scenario generation  
7. semantic schema ingestion  
8. synthetic dataset copilot  
9. policy-driven GenAI orchestration

---

# 6. Suggested “Next Build” Pending Set

## Near-term priority
1. deterministic seed support  
2. explainability report  
3. workbook lint/validation UX  
4. codebase cleanup / warnings  
5. config templates

## After that
6. cross-column rule engine  
7. schema import  
8. LLM-assisted config completion  
9. NL-to-config  
10. scenario generation

---

# 7. One-Line Summary

The biggest pending areas are:

- **trust and explainability**
- **better config authoring experience**
- **scalability**
- **platformization**
- and the biggest future differentiator is clearly **GenAI/LLM-assisted config, rule, and scenario generation**

