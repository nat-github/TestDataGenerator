# PRD-Level Roadmap

## Product Name
**Synthetic Data Platform**  
_Excel-configurable, relationship-aware, enterprise-grade synthetic test data generation for snapshot, delta, and SCD2 workflows._

---

# 1. Product Vision

Build the most practical **enterprise synthetic data generation platform** for structured data teams that need:

- schema-aware test data
- realistic and constrained values
- relationship integrity
- repeatable snapshot generation
- downstream-ready delta and SCD2 datasets
- low-friction configuration through business-friendly artifacts like Excel

### Vision Statement
Enable QA, data engineering, analytics, and platform teams to generate **trustworthy, production-shaped synthetic datasets** for complex relational and analytical pipelines without exposing sensitive data.

---

# 2. Problem Statement

Most synthetic data tools fail in one or more of these areas:

- they generate only flat/mock data
- they don’t preserve PK/FK relationships well
- they don’t support business-style constraints like regex, enumerations, null rates, and typed formats
- they stop at row generation and do not help with **delta** and **SCD2**
- they are too code-heavy for business/data analysts
- they are not reliable for enterprise parquet-driven lakehouse workflows

This product solves that by combining:

- workbook-driven schema/rule definition
- deterministic + model-driven generation
- relational integrity
- parquet export
- data lifecycle outputs like snapshot/delta/SCD2

---

# 3. Target Users

## Primary Users

### 1. Data Engineers
Need synthetic parquet data for ingestion, transformation, CDC, delta, and historical processing.

### 2. QA / Test Automation Teams
Need repeatable datasets for integration and regression testing.

### 3. Analytics / BI Engineers
Need production-shaped but non-sensitive datasets to validate semantic models and reports.

### 4. Platform / Data Product Teams
Need domain-aware sample datasets with referential integrity for demos, test environments, and onboarding.

## Secondary Users

### 5. Business Analysts / Domain SMEs
Need to define allowed values, regex formats, and table rules without writing code.

---

# 4. Current Product Strengths

The utility already has unusually strong foundations:

- **Excel/workbook-driven configuration**
- **PK/FK relationship handling**
- **regex-driven generation**
- **business values + special rules**
- **fallback + SDV hybrid generation**
- **parquet-first export**
- **delta generation**
- **SCD2 generation**
- **CLI-based execution**
- **constraint reconciliation after SDV generation**

This is a strong differentiator because many tools only do one of these well.

---

# 5. Product Positioning

## Positioning Statement
For enterprise data teams that need realistic, relational, downstream-ready synthetic data, this product provides a **configurable generation pipeline** that supports both **business-defined rules** and **model-based generation**, while producing not just snapshots, but also **delta and SCD2 outputs**.

## Strategic Position
This should evolve from:
- a useful internal utility

into:
- a **governed enterprise data generation product**

---

# 6. Product Goals

## Goals
1. Make synthetic data generation **trustworthy**
2. Make configuration **accessible**
3. Make output **pipeline-ready**
4. Make runs **repeatable and explainable**
5. Make the tool usable across **small test cases and large data volumes**
6. Make it extensible for more enterprise constraints and domains

## Non-Goals
1. Not a full-featured transactional database simulator
2. Not a replacement for privacy-preserving production masking tools
3. Not initially a no-code web application for all personas
4. Not initially focused on unstructured documents/media generation

---

# 7. Core Product Themes

The roadmap should be driven by 5 themes:

### Theme A — Trust & Constraint Fidelity
Ensure outputs actually obey schema, business, and regex rules.

### Theme B — Usability & Configuration Experience
Reduce setup friction and make workbook usage safer and easier.

### Theme C — Enterprise Data Lifecycle Support
Expand snapshot, delta, SCD2, and temporal generation capabilities.

### Theme D — Scale & Performance
Support bigger datasets and operational workloads.

### Theme E — Platformization & Governance
Move from CLI utility to team-grade product with observability, APIs, governance, and auditability.

---

# 8. Phased Roadmap

## Phase 1 — Product Hardening
### Timeline
**0–3 months**

### Objective
Make the current system reliable, explainable, and easy to adopt internally.

### Key Outcomes
- Stable and predictable generation
- Strong config validation
- Better reporting and trust in generated data
- Smooth onboarding for first internal teams

### Features
#### 1. Constraint Fidelity Layer
- formalize post-generation reconciliation
- guarantee regex, business values, null-rate, and type constraints
- ensure consistency across fallback and SDV paths

#### 2. Run Explainability
- generation report per table/column
- what was generated by:
  - SDV
  - fallback
  - business values
  - regex/special rules
  - FK reconciliation

#### 3. Better Validation UX
- workbook validation with sheet/row/column references
- clearer failure messages
- config lint mode

#### 4. Deterministic Runs
- global seed support
- per-table seed support
- replay same config + seed = same output

#### 5. Packaging & Stability
- reduce warnings/deprecations
- improve artifact handling
- standardize logging and errors

### Success Metrics
- >95% successful first-run completion for validated configs
- <5% manual intervention after config validation
- 100% pass rate for core regression suite
- adoption by 2–3 internal use cases

---

## Phase 2 — Enterprise Feature Expansion
### Timeline
**3–6 months**

### Objective
Expand beyond “good generator” into “enterprise-ready synthetic data engine.”

### Key Outcomes
- richer constraints
- stronger domain modeling
- better lifecycle realism
- wider data engineering use

### Features
#### 1. Advanced Constraint Engine
- conditional rules
  - e.g. if `country=NL`, IBAN must match NL format
- cross-column dependencies
- weighted business values
- temporal dependency rules
- uniqueness scopes beyond PK

#### 2. Data Quality Profiles
- per-column distribution profiles
- null distribution tuning
- skew and cardinality controls
- frequency weighting for categorical data

#### 3. Domain Packs
- finance starter pack
- party/customer models
- transaction models
- address/ID/reference templates

#### 4. Enhanced Delta / SCD2 Controls
- late arriving change scenarios
- configurable operation ratios
- effective dating controls
- event-time-driven snapshot evolution

#### 5. Config Experience V2
- config templates
- workbook generator/wizard
- schema import from parquet / metadata / SQL-like definitions

### Success Metrics
- 50% reduction in custom config effort for new domains
- 3–5 reusable domain templates
- >90% conformity for advanced rule coverage in benchmark datasets
- internal stakeholders can onboard new tables in <1 day

---

## Phase 3 — Scale & Operationalization
### Timeline
**6–9 months**

### Objective
Support large-scale generation and operational usage in engineering environments.

### Key Outcomes
- bigger data volumes
- improved runtime efficiency
- more reliable batch execution

### Features
#### 1. Streaming / Chunked Generation Maturity
- stable large-scale generation mode
- direct parquet streaming
- memory-safe pipeline for huge outputs

#### 2. Parallelization
- per-table parallel generation
- export pipeline concurrency
- large workload orchestration

#### 3. Benchmarking & Sizing
- standard benchmark suite
- throughput measurement
- config-based runtime estimation

#### 4. Observability
- structured run logs
- timing per stage
- records/sec, export metrics, reconciliation counts

#### 5. CI/CD Integration
- stable non-interactive modes
- config check in pipeline
- golden output test strategy

### Success Metrics
- 10M+ row generation for selected workloads
- predictable memory profile in streaming mode
- measurable SLA targets for common dataset sizes
- successful CI integration in internal pipelines

---

## Phase 4 — Platformization
### Timeline
**9–15 months**

### Objective
Turn the utility into a team-grade platform rather than only a CLI engine.

### Key Outcomes
- self-service usage
- governance
- collaboration
- reuse at scale

### Features
#### 1. API Layer
- REST/CLI service wrapper
- run creation, status, artifact retrieval
- config upload and validation endpoints

#### 2. Metadata & Catalog
- saved config versions
- reusable generation recipes
- dataset lineage metadata

#### 3. Governance
- approval workflow for configs
- rule pack versioning
- audit trail of generation runs

#### 4. Team Collaboration
- shared templates
- reusable table packs
- domain-specific configuration libraries

#### 5. Output Registry
- managed synthetic datasets
- labeled snapshots, deltas, SCD2 histories
- retention/versioning policy

### Success Metrics
- multiple teams using shared recipes
- reduced duplicate config creation
- full run traceability
- config reuse across business domains

---

## Phase 5 — Strategic Differentiation
### Timeline
**12–18+ months**

### Objective
Make the product category-defining in its niche.

### Key Outcomes
- strong differentiation from generic fake-data tools
- enterprise credibility
- broader product identity

### Features
#### 1. Policy-Driven Synthetic Data
- declarative policy engine
- compliance-oriented rule packs
- environment-specific generation policies

#### 2. Synthetic Data Simulation
- controlled change over time
- scenario generation
- stress/test-case simulation packs

#### 3. Multi-Modal Input Support
- schema ingestion from db/parquet/catalog
- optional UI-based config authoring
- code + config dual workflow

#### 4. Synthetic Data Assurance
- automatic conformance scoring
- realism scoring
- relationship integrity score
- rule-preservation score

#### 5. Marketplace / Pack Ecosystem
- reusable regex packs
- industry templates
- banking / insurance / fintech starter kits

### Success Metrics
- recognized internal standard for synthetic enterprise datasets
- 70% reuse of standardized configs/templates
- strong benchmarked differentiation vs generic tools
- path to external/open-source/product packaging if desired

---

# 9. Functional Requirements by Theme

## A. Constraint & Schema Fidelity
- preserve PK/FK integrity
- preserve regex/special rules
- preserve business values and null modifiers
- enforce type-specific constraints
- support composite key-aware behavior

## B. Configurability
- workbook remains first-class
- config linting
- config templates
- schema import support
- versioned rule sets

## C. Data Lifecycle
- snapshot generation
- delta generation
- SCD2 generation
- temporal scenario generation
- replayable change sequences

## D. Scale
- large-row generation
- chunked export
- efficient parquet writing
- better memory handling

## E. Trust & Visibility
- validation reports
- run reports
- audit trail
- conformance metrics

---

# 10. Success Metrics / KPIs

## Adoption Metrics
- number of internal teams using it
- number of configs created/reused
- repeat runs per config

## Quality Metrics
- rule conformance rate
- relationship validity rate
- export success rate
- config validation failure clarity score

## Efficiency Metrics
- time to onboard a new dataset
- time to generate X rows
- time to create snapshot + delta + scd2 outputs

## Product Maturity Metrics
- % of runs using deterministic seeds
- % of configs using reusable templates
- % reduction in manual test data creation

---

# 11. Risks and Mitigations

## Risk 1: Complexity grows too fast
**Mitigation**
- keep workbook UX simple
- separate “basic mode” and “advanced mode”

## Risk 2: SDV and rule engine diverge
**Mitigation**
- make reconciliation a formal architecture layer
- add conformance reporting

## Risk 3: Too much finance/domain coupling
**Mitigation**
- keep domain packs modular
- maintain a generic core engine

## Risk 4: Scale issues at enterprise volumes
**Mitigation**
- prioritize chunked generation early
- benchmark with realistic datasets

## Risk 5: Config quality becomes the bottleneck
**Mitigation**
- add validation, templates, and schema import

---

# 12. Recommended Prioritization

If sequencing for maximum impact:

## Highest priority now
1. **Constraint fidelity and explainability**
2. **Deterministic runs**
3. **Validation UX**
4. **Template-driven onboarding**
5. **Streaming/performance hardening**

## Next priority
6. cross-column rules  
7. scenario/time-based simulation  
8. API layer  
9. governance/versioning  
10. domain packs

---

# 13. Product Narrative for Stakeholders

### Internal one-liner
A synthetic data platform that generates **realistic, rule-compliant, relationship-aware parquet datasets**, including **snapshot, delta, and SCD2 outputs**, from a business-friendly workbook configuration.

### Executive version
This product reduces dependency on sensitive production data while accelerating QA, analytics validation, and data engineering delivery by producing enterprise-ready synthetic datasets with lifecycle-aware outputs.

---

# 14. Suggested Milestone Names

- **M1 — Trust the Output**
- **M2 — Model the Business**
- **M3 — Scale the Engine**
- **M4 — Productize the Platform**
- **M5 — Define the Category**

