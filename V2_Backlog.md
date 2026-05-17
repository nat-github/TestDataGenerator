# Synthetic Data Platform — V2 Backlog

State today: **V1 is MVP-ready**. Capability matrix is ~95% production-ready,
the remaining gaps are productisation polish (wheel, type checking,
benchmarks) plus features that should wait for real-user feedback.

This doc is the parking lot. Tick items as they ship; drop items that lose
relevance after feedback. Each entry has effort, dependencies, and a one-line
"why" so future-you doesn't have to re-derive the rationale.

> **Status legend:** `[ ]` queued · `[~]` in flight · `[x]` shipped · `[-]` dropped

---

## My honest pick — top 5 to do first

If you do nothing else from this doc, do these. They're cheapest-per-minute
and they unblock everything else.

| # | Item | Effort | Why first |
|---|---|---:|---|
| 1 | **Wheel package + console scripts** | 1 day | Unblocks `pip install`, Databricks, private PyPI. Cheapest productisation move. |
| 2 | **`ruff` + `mypy --strict` baseline** | half-day | Catches type drift before runtime. One config file, no behaviour change. |
| 3 | **Inline ER diagram in UI** | half-day | Visual confidence-builder for SME demos. The renderer exists; just embed. |
| 4 | **CDC (delta / SCD2) page in UI** | 1-2 days | Currently CLI-only. Big SME-demo win without new engine work. |
| 5 | **Drift detection (snapshot vs snapshot)** | 1 week | Composes with delta you already have. Closes "watch what changes over time." |

After these five, the rest of the backlog is "pick one when you have a
specific user need." Don't pre-emptively build breadth.

---

## Tier 1 — Productisation polish

Cheap, high-credibility moves that don't change the product, only how it's
shipped and reviewed.

### Wheel package + console scripts

- [ ] **Build the wheel** — `poetry build` already produces `dist/*.whl`; add
  `[project.scripts]` entries to `pyproject.toml` so `pip install sdp` creates:
  - `sdp` → `main:main` (CLI: `sdp generate ...` instead of `python main.py ...`)
  - `sdp-ui` → tiny `ui/cli.py` that spawns Streamlit on the bundled UI
  - `sdp-mcp` → `sdp.mcp_server.server:main`

  Effort: **1 day**. Dependencies: none. **Highest leverage item in this doc.**

- [ ] **Bundle examples as `package_data`** (or ship a separate `sdp-examples`
  wheel) so `pip install sdp` includes the 11 reference configs + 3 OpenAPI
  specs + 3 mock fixtures. Effort: **half-day**.

- [ ] **PyPI / private-index publishing** — automate `poetry publish` against
  whichever index your team uses. Effort: **half-day**.

- [ ] **Conda recipe** — only if a user asks. Effort: **1 day**.

- [ ] **Versioning strategy** — semver, with a `CHANGELOG.md`. Effort: **30 min**
  to set up; ongoing discipline.

### Code-quality tooling

- [ ] **`ruff` baseline** — drop in `ruff.toml` with sensible defaults; run
  once, fix or `# noqa` what shows up. Effort: **2-3 hours**.

- [ ] **`mypy --strict` baseline** — start with the strictest config that
  passes today; tighten over time. Effort: **half-day**.

- [ ] **Pre-commit hooks** — `pre-commit-config.yaml` with `ruff` + `mypy` +
  `end-of-file-fixer`. Effort: **1 hour**.

### Performance + observability

- [ ] **`pytest-benchmark` suite** — baseline numbers for: 10k-row generation,
  GX validation, mock rendering, quality report. Effort: **1 day**.
  Dependencies: should land *before* CI/CD so the baselines stick.

- [ ] **Memory profiler integration** — `memray` or `tracemalloc` snapshots
  during large runs. Effort: **half-day**. Only if generation OOMs at scale.

- [ ] **Structured logs (JSON)** — opt-in `--log-format json` flag for the
  CLI; simpler grep/Splunk ingestion. Effort: **half-day**.

### Distribution

- [ ] **Docker image — multi-arch** (amd64 + arm64) — currently
  amd64-implicit. Effort: **2 hours** if buildx is available.

- [ ] **Slim Docker image variant** without `mocks` and `ui` extras for
  servers that only generate data. Effort: **1 hour**.

- [ ] **Helm chart** for k8s-hosted deployments (Streamlit + MCP). Effort:
  **1-2 days**. Skip until someone asks.

### Output format extensions

- [ ] **CSV / XLSX export from `generate`** — today the generator only
  writes Parquet (chosen for typed-dtype preservation, size, and
  downstream Spark/Delta/Trino compatibility). Add `--output-format`
  accepting any combo of `parquet,csv,xlsx`; expose `export_to_csv()`
  and `export_to_excel()` on `DataGenerator`; surface the choice in the
  Streamlit UI's download section. **Why:** Excel-bound stakeholders,
  legacy ETL tools, and data scientists who paste into other systems
  need readable text formats. **Caveat:** CSV/XLSX lose dtype precision
  on round-trip — keep Parquet as the default, format flag opts in to
  the others alongside it. Effort: **1 day**.

### Intentionally deferred

- [-] **CI/CD via GitHub Actions** — explicitly off the table per platform
  owner direction. Note for future-you: when this comes back, all the Tier 1
  tooling above lands first so the workflow is just `ruff && mypy && pytest`.

---

## Tier 2 — UI enhancements

The Streamlit UI is sized for ≤ 10k rows + form-driven flows. Don't expand
beyond that. These items deepen what's there for the SME audience.

### Generate page

- [ ] **Inline ER diagram preview** — the Mermaid renderer already exists in
  `utils/er_diagram.py`. Embed via `st.markdown(...)` with a Mermaid-aware
  component (or `streamlit-mermaid`). Effort: **half-day**.

- [ ] **Distribution histograms in quality report** — currently we render
  numeric summary stats; add a `st.bar_chart` per numeric column showing the
  shape. Effort: **half-day**.

- [ ] **Side-by-side synth vs source distribution panel** — when a source is
  uploaded for the quality report, render two histograms side by side per
  column. Effort: **1 day**.

- [ ] **Streaming preview for larger runs** — `st.fragment` lets us update
  the preview every N rows. Effort: **1 day**. Only worth it if users push the
  10k cap regularly.

- [ ] **Cloud upload progress bar** — `st.progress` driven by the uploader
  callback. Effort: **2 hours**.

### New pages

- [ ] **CDC (delta / SCD2) page** — input two snapshot dirs (or generate them
  inline with two seeds), produce delta or SCD2 output, preview I/U/D rows
  inline. Effort: **1-2 days**.

- [ ] **LLM enrichment page** — upload a bare config, pick provider
  (Anthropic / LM Studio / Ollama), preview the suggested edits before
  applying. Effort: **1-2 days**.

- [ ] **SME relationship-review page** — load the YAML produced by
  `infer-relationships`, present each suggested FK with accept/reject
  buttons, write feedback back into the JSONL store. **High-value**: the
  current YAML-editing flow is the biggest SME friction point. Effort:
  **2 days**.

- [ ] **PII-scan page** — point at a parquet directory, view the scanner's
  classifications inline with severity counts. Effort: **half-day**.

- [ ] **Lint page (configs)** — drop a config in, see exact sheet/row/column
  errors with copy-friendly text. Effort: **half-day**.

### Cross-cutting

- [ ] **Run history (sqlite-backed)** — store config hash + seed + timestamps
  + summary metrics for each run. Lets users diff two runs. Effort: **1-2 days**.

- [ ] **Saved provider profiles** — instead of re-typing LM Studio model +
  base URL each time, save named profiles. Effort: **half-day**. Stored in
  `~/.sdp/ui_profiles.json`.

- [ ] **Dark mode toggle** — Streamlit honours system theme; just expose a
  switch. Effort: **15 min**.

- [ ] **Streamlit AppTest coverage for the new pages** — extend
  `tests/test_ui_streamlit.py` so the API Mocks page + CDC page (when built)
  load without exception. Effort: **3 hours**.

---

## Tier 3 — Mocks track depth

The mocks track is feature-complete for the 80% case. These are real but
narrow gaps for the remaining 20%.

### Renderer fidelity

- [ ] **Pact `matchingRules`** — current Pact output is exact-match. Add
  fuzzy matchers (regex, type-only, datetime format). Effort: **2-3 days**.
  Required for real Pact contract testing in production.

- [ ] **OpenAPI polymorphism** — `oneOf` / `anyOf` with `discriminator`
  in the importer. Currently we pick the first variant. Real fintech /
  e-commerce specs use this heavily. Effort: **3-4 days**.

- [ ] **External `$refs`** — currently $ref must point inside the same
  document. Effort: **1 day**.

- [ ] **OpenAPI security flow modelling** — capture OAuth2 / OIDC flow
  metadata, render mocks that demand the right header shape. Effort:
  **2 days**. Niche.

### Reverse importers

- [ ] **Postman cookie / form-data bodies** — currently we handle raw JSON;
  cookies and multipart need additional plumbing. Effort: **1 day**.

- [ ] **HAR auth-token scrubbing** — captured Authorization headers can leak
  real tokens. Add a flag to redact / regex-replace before saving the
  MockConfig. Effort: **half-day**. Compliance-relevant.

### Stateful scenarios

- [ ] **Branching scenarios** — current state machine is linear (after-N
  triggers). Add conditional branches based on request headers / body fields.
  Effort: **3-4 days**.

- [ ] **Time-of-day rules** — e.g. "between 09:00–17:00 UTC, return business
  hours; outside, return 503." Effort: **1 day**.

- [ ] **Cross-endpoint state** — endpoint A's response sets a flag; endpoint
  B's response depends on it. Effort: **1 week**.

### Format coverage

- [ ] **GraphQL support** — schema introspection → MockConfig; renderer for
  Apollo Mock Server / `graphql-tools`. Big slice of modern API surfaces.
  Effort: **1-2 weeks**.

- [ ] **AsyncAPI 2.x support** — event-driven systems blind spot today
  (Kafka / RabbitMQ stub messages). Effort: **1-2 weeks**.

- [ ] **gRPC / Protobuf** — niche. Skip until a real user asks. Effort:
  **1-2 weeks**.

### Tooling

- [ ] **`mock-serve` — built-in HTTP server** — small `aiohttp`-based server
  that serves a `MockConfig` directly, no WireMock dependency. Effort:
  **1-2 days**. Removes Java/WireMock install friction for casual users.

- [ ] **Mock fault injection** — random 5xx, slow responses, partial bodies.
  Effort: **1 day**.

- [ ] **Mock-config visual editor** — for SMEs editing scenarios without YAML.
  Effort: **1 week**. Skip until the YAML friction is real.

---

## Tier 4 — Data-science depth

Higher value for analytics-heavy users. Pick one when you have a concrete
"prove the synthetic data is faithful" or "watch how it shifts" need.

### Quality / fidelity

- [ ] **Drift detection (snapshot vs snapshot)** — feed v1 vs v2 of the same
  table, get a per-column drift score. Composes with delta/SCD2. Effort:
  **1 week**. **Top-5 pick — see above.**

- [ ] **Mutual information / cross-column dependency tests** — beyond Pearson
  correlation, capture non-linear relationships. Effort: **2-3 days**.

- [ ] **SDV's own quality reports surfaced** — SDV ships
  `sdmetrics.reports.single_table.QualityReport`; surface alongside ours.
  Effort: **half-day**. Nice corroboration of the in-house implementation.

- [ ] **Per-column distribution shape classifier** — auto-flag bimodal /
  long-tail / heavy-skew columns. Effort: **2 days**.

### Privacy

- [ ] **k-anonymity check** — ensure no quasi-identifier combination has < k
  matches in the synthetic set. Effort: **2 days**.

- [ ] **l-diversity / t-closeness checks** — stronger than k-anonymity.
  Effort: **2-3 days**.

- [ ] **Membership-inference attack scoring** — formal proxy beyond NN
  distance. Effort: **3-4 days**.

- [ ] **Differential-privacy synthesizer integration** — wire in a DP-aware
  generator (smartnoise-synth, ydata DP variants) as an optional alternative
  to SDV. Effort: **1-2 weeks**.

### New data shapes

- [ ] **Time-series table support** — TimeGAN / DoppelGANger via
  ydata-synthetic as a complementary backend (NOT a replacement for SDV).
  Effort: **2 weeks**. Skip unless a user has sequential-data needs.

- [ ] **Hierarchical / nested data** — e.g. JSON columns with sub-structure.
  Effort: **2 weeks**.

### Test-data subsetting

- [ ] **Real-DB referentially-consistent subsetting** — given production DB
  + a sample size, extract a coherent subset that preserves FK closure.
  Effort: **2 weeks**. Adjacent product, not the same as synthesis.

---

## Tier 5 — Infra / deployment / hosting

Only chase these when the platform has real users beyond your team.

### Databricks

- [ ] **Databricks notebook quickstart** — once the wheel ships, a one-page
  notebook showing `pip install`, then generate + validate from a
  Spark-friendly notebook. Effort: **half-day**.

- [ ] **Unity Catalog handlers** — pull schemas from Unity Catalog as
  `TableConfig` source, write outputs back as Unity-managed tables. Effort:
  **1 week**.

- [ ] **Spark-parallel generation** — distribute `generate_data` across
  cluster workers via PySpark UDFs. Big change to `data_generator.py`. Effort:
  **2-3 weeks**. Only worth it for million+ row workloads.

- [ ] **Databricks Apps deployment** — host the Streamlit UI as a Databricks
  App. Effort: **half-day** once the wheel exists.

### k8s / serverless

- [ ] **Helm chart** for Streamlit + MCP service deployment. Effort:
  **1-2 days**.

- [ ] **Lambda-style serverless wrapper** — `generate_data` as a stateless
  HTTP endpoint. Effort: **2-3 days**. Niche.

### Observability

- [ ] **OpenTelemetry traces** — span per CLI command / MCP tool / UI action.
  Effort: **2-3 days**.

- [ ] **Prometheus metrics endpoint** — generation count, latency, error
  rates. Effort: **1 day**.

- [ ] **Sentry / error reporting integration** — `--telemetry-on` opt-in
  flag. Effort: **half-day**.

### Multi-tenant / hosted

- [ ] **API auth (JWT / OAuth)** for the MCP server in HTTP transport mode.
  Effort: **1 week**.

- [ ] **Per-user run quotas + billing telemetry** — Effort: **2 weeks**.

- [ ] **Web auth for Streamlit UI** — `streamlit-authenticator` or similar.
  Effort: **half-day** for basic.

---

## Tier 6 — Strategic directions (defer until proven)

These are interesting but should not be pre-emptively built.

- [ ] **RAG layer for docs Q&A** — grounded "ask the platform anything"
  using the 17+ markdown docs. Effort: **3-5 days**. Defer until users
  actually ask the same question repeatedly.

- [ ] **Knowledge graph for cross-project schema-pattern reuse** — see
  earlier discussion. **Picks one slot with RAG above** — they solve
  similar problems, don't build both. Effort: **2-3 weeks**.

- [ ] **AI-assisted scenario authoring** — describe a scenario in English,
  LLM produces the YAML. Effort: **1 week**.

- [ ] **Schema diff + migration helper** — given two configs, produce a
  migration script. Effort: **1 week**.

- [ ] **Visual config builder in the UI** — form-driven config authoring
  for users who shouldn't touch YAML. Effort: **2 weeks**. Heavy investment.

- [ ] **OAuth flow simulation in mocks** — add OAuth2 dance to the mocks
  track. Effort: **1 week**. Easier in WireMock proxies than in our config
  model — that's why it's Tier 6.

---

## Won't-do (yet)

Explicit deferrals so they don't keep coming back as suggestions.

- [-] **CI/CD via GitHub Actions** — intentionally deferred per platform
  owner. Will revisit when team adoption broadens.

- [-] **Plugin system** — premature; the 60+ built-in special rules cover
  most real needs. Add only when a third party asks.

- [-] **gRPC mocks** — niche slice; revisit only with a concrete user.

- [-] **Streaming / SSE / WebSocket mocks** — same reasoning. Adds complexity
  for a small slice.

- [-] **Frontend rewrite (React / Vue)** — Streamlit is the right shape for
  the form-driven flows we have. Don't rewrite.

- [-] **Replacing SDV with another synthesizer** — SDV's relational support
  is the foundation. ydata is *complementary at best* (time-series only),
  not a replacement.

---

## Documentation backlog

Smaller items that improve the existing surface without adding capability.

- [ ] **Sphinx-style API reference** auto-generated from docstrings. Effort:
  **1 day**.

- [ ] **Architecture Decision Records (ADRs)** — `docs/adr/0001-parallel-tracks.md`
  etc. capturing the "why" behind major design choices. Effort: **2 hours
  per ADR**, write ~5.

- [ ] **Video screencast scripts** — 3-5 minute walkthrough scripts for
  each major capability (generate / mocks / quality report / MCP). Effort:
  **half-day per script**.

- [ ] **Migration guide** — for users moving from a hand-rolled fixture
  pipeline to this platform. Effort: **1 day**.

- [ ] **CONTRIBUTING.md** — once external contribution is in scope. Effort:
  **2 hours**.

---

## How to use this doc

1. Tick `[x]` when an item ships; reference the commit hash in the line.
2. Drop `[-]` items that lose relevance after user feedback (move them to
   Won't-do, with a one-line reason).
3. Move items between tiers as priorities shift; tier order = recommended
   pick order, not a hard rule.
4. Keep the **top-5 list** at the top in sync with what you're actually
   working on next — that's the artefact most worth glancing at.
5. New items added during feedback collection: drop them into the right
   tier with a line about who asked and what for.

---

*Last updated: 2026-05-11. Captured at the V1 freeze for feedback collection.*
