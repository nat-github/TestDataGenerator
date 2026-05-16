# ML Relationship Inference

A free, deterministic alternative to the LLM relationship inferrer. Detects
foreign-key relationships between tables using four interpretable signals plus
an adaptive feedback loop — no API keys, no per-run cost, no network call.

The LLM path (`llm/relationship_inferrer.py`) and the ML path
(`ml/relationship_inferrer.py`) implement the same `infer(...)` interface and
return the same `RelationshipConfig` shape, so callers can swap engines via
the `--method` flag.

---

## Quick start

```bash
# 1. Inspect a config and emit suggested relationships for SME review
python main.py infer-relationships \
    --config config/bare.xlsx \
    --config-output config/inferred.yaml \
    --er-output diagrams/inferred.mmd \
    --method ml

# 1b. Use schema-aware knowledge-graph disambiguation on lookup-heavy configs.
# In this mode the YAML output defaults to a simple shape: tables + relationships.
python main.py infer-relationships \
    --config config/Creditcard_no_rel.xlsx \
    --config-output output/creditcard_kg.yaml \
    --method ml \
    --ml-mode knowledge-graph

# 2. SME edits config/inferred.yaml — deletes wrong entries, adds missing ones

# 3. Feed the diff back so the model learns
python main.py record-feedback \
    --inferred config/inferred.yaml \
    --reviewed config/inferred.yaml.reviewed
```

Running inference alone does not mutate the model. After enough
review-and-record rounds (~30 labelled examples with both classes represented),
an adaptive logistic-regression classifier activates and starts overriding the
heuristic on future runs.

---

## Pipeline

For every `(child_table.col, parent_table.col)` candidate pair the engine:

1. **Type-compatibility gate.** `N*` (numeric), `VA*`/`A*` (string), `D`/`DT`/`TS`
   (datetime) families. Cross-family pairs are dropped immediately.
2. **Compute four signals** (each ∈ [0, 1]):
   - `type_compatibility` — 1.0 if the family matches, else 0.
   - `name_similarity` — token-aware: tokenises snake/camel case, drops
     stopwords (`id`, `key`, `code`), then combines Levenshtein and token
     overlap. Falls back to raw string ratio when stopword stripping leaves
     nothing (so `id ↔ id` still scores 1.0).
   - `value_subset` — fraction of distinct child values present in the parent
     set (requires `sample_data`; skipped otherwise). Normalises ints/floats/
     strings so `[1.0, 2.0]` matches `[1, 2, 3]`.
   - `pk_likeness` — declared-PK shortcut returns 1.0; otherwise a blend of
     uniqueness and cardinality on the parent column.
3. **Heuristic baseline.** Weighted sum of the above (weights live in
   `ml.relationship_signals.DEFAULT_WEIGHTS`). `type_compatibility == 0` short-
   circuits the score to 0 — the gate doubles as a multiplier.
4. **Pattern-memory adjustment.** Looks up the candidate in the feedback store:
   - Exact-match accept history → `+0.15`
   - Exact-match reject history → `−0.40`
   - Column-name-only history (cross-project memory) → `+0.07`
5. **Optional classifier override.** When the classifier is fitted, the final
   confidence becomes `0.7 * classifier_p + 0.3 * (heuristic + memory)`. The
   30 % heuristic anchor keeps early-fitted models from collapsing to noise.
6. **Threshold filter.** Drop anything below `confidence_threshold`
   (default 0.55, configurable via `--ml-confidence`).
7. **Dedupe per child column.** When several PK candidates compete for the
   same child column, only the highest-confidence one survives.

### Knowledge-graph mode

`ml/relationship_knowledge_graph.py` adds an opt-in schema-aware layer on top of
the baseline ML inferrer. It builds a real `networkx.DiGraph` over the schema —
**table**, **column** and **entity** nodes, plus **candidate_fk** edges — and
reasons over its structure. It is aimed at cases where many reference tables
share generic PK names such as `CODE`, making pure name similarity too noisy.

Four reliability mechanisms run on top of the heuristic baseline:

1. **Inclusion dependency (primary, fully domain-agnostic).** When sample data
   is supplied, a real FK requires the child's value set to sit inside the
   parent's. A candidate with value containment below `0.30` is dropped outright
   regardless of how well the names match. This is the strongest, most generic
   FK signal and the data-led scoring path is led by it.
2. **Hub / degree prior.** A parent key referenced by many child columns is a
   genuine dimension and earns a small (≤ 0.03) tie-breaking boost.
3. **Global one-parent assignment.** Each child FK column commits to exactly one
   parent — the highest-scoring — resolving generic `CODE`/`ID` ambiguity.
4. **FK-cycle resolution.** Cycles among distinct tables are almost always
   wrong; the weakest edge in each cycle is dropped (self-references are kept).

Supporting structural signals: table/descriptor semantic similarity, FK priors
from `is_fk`, lookup-table bonuses, generic-key and ambiguity penalties, and
datatype length compatibility. All `kg_*` signals (including `kg_value_inclusion`
and `kg_hub_bonus`) are written into `inference_signals` for auditability.

**Genericity.** The graph machinery is domain-agnostic. The only domain-specific
knowledge — abbreviations and stopwords used for name matching — lives in a
pluggable `SemanticProfile` (`ml/semantic_profile.py`): a `GENERIC` layer of
universal abbreviations plus an optional bundled `BANKING` layer. Point
`SDP_SEMANTIC_PROFILE` at a YAML file to extend or replace the vocabulary for a
different domain without touching inference code.

The implementation is additive: `--ml-mode standard` keeps the original ML
behaviour, while `--ml-mode knowledge-graph` opts into the enhanced ranking.
With no sample data the knowledge-graph mode falls back to the name/structure
path and behaves as before.

---

## Modules

| Module | Role |
|---|---|
| `ml/relationship_signals.py` | Pure-function signal computers + `combine()` |
| `ml/relationship_feedback_store.py` | JSONL append-only store, pattern lookups, training-data extraction |
| `ml/relationship_classifier.py` | Logistic-regression classifier with cold-start refusal |
| `ml/relationship_inferrer.py` | Orchestrator — same interface as `llm.RelationshipInferrer` |

`ml/relationship_inferrer.MLRelationshipInferrer.infer(...)` returns an
`MLInferenceResult` with: `relationships`, `reasons`, `skipped_reasons`,
`input_tables`, `classifier_fitted`, `classifier_examples`.

---

## CLI

### `infer-relationships`

```bash
python main.py infer-relationships \
    --config <path>                     # .xlsx | .yaml | .yml | .json
    --config-output <path>              # YAML written for review or direct use
    [--method ml|llm|both]              # default: ml
    [--ml-confidence 0.55]
    [--ml-mode standard|knowledge-graph]
    [--llm-confidence 0.7]
    [--simple-yaml]                     # force minimal tables+relationships YAML
    [--review-yaml]                     # force richer review YAML with metadata
    [--feedback-store <path>]           # default: ml_feedback/relationship_feedback.jsonl
    [--er-output <path>]                # .mmd|.dot|.png — drawn from existing + inferred
    [--sample-data <dir>]               # parquet/CSV files for value-subset signal
```

`--method both` runs the ML inferrer first, then asks the LLM only about
candidates ML didn't cover. Useful when you want the cost ceiling of ML with
the recall of the LLM on novel schemas.

YAML output defaults:

- `--ml-mode standard` → richer review YAML
- `--ml-mode knowledge-graph` → simple YAML by default
- `--review-yaml` overrides the knowledge-graph default when SME review metadata
  is needed

### `record-feedback`

```bash
python main.py record-feedback \
    --inferred <path>                   # the YAML produced by infer-relationships
    --reviewed <path>                   # same file after SME edits
    [--feedback-store <path>]
```

Diff strategy:

- Entries marked `inferred_by_ml` or `inferred_by_llm` and present in both
  files → recorded as **accept**.
- Entries marked `inferred_by_ml`/`inferred_by_llm` and missing from the
  reviewed file → recorded as **reject**.
- Entries in the reviewed file the inferrer never proposed → recorded as
  **SME-added** (with empty signals so they only feed pattern-memory, not
  classifier training).

This is the point where the system learns. `record-feedback` writes accepted,
rejected, and SME-added examples to the feedback store; future ML runs reuse
that history immediately via pattern memory and later via the classifier once
training thresholds are met.

### `generate --infer-relationships`

The `generate` subcommand has supported `--infer-relationships` for the LLM
path; it now respects `--method`, `--ml-confidence`, and `--feedback-store`
the same way `infer-relationships` does. The default method is `ml`, which
flips the long-standing default away from the LLM (free is the new free).

---

## Feedback store

JSONL file, one entry per line. Default location:
`<project_root>/ml_feedback/relationship_feedback.jsonl`. Override with the
`SDP_FEEDBACK_PATH` env var or `--feedback-store` flag.

```json
{
  "source_table": "orders",
  "source_column": "customer_id",
  "target_table": "customers",
  "target_column": "customer_id",
  "accepted": true,
  "signals": {"name_similarity": 1.0, "type_compatibility": 1.0,
              "value_subset": 1.0, "pk_likeness": 1.0},
  "predicted_confidence": 0.82,
  "timestamp": 1714780800.0,
  "note": "recorded via record-feedback"
}
```

Malformed lines are silently skipped — append-only durability matters more
than strict validation here.

---

## Classifier activation

Logistic regression on the fixed feature vector
(`type_compatibility, name_similarity, value_subset, pk_likeness,
pattern_memory_score`). Refuses to train when:

- fewer than `MIN_TRAINING_EXAMPLES = 30` total entries, or
- fewer than `MIN_PER_CLASS = 5` accepts or rejects.

`MLInferenceResult.classifier_fitted` and `classifier_examples` surface the
state on every run, so callers can tell when learning has kicked in.

---

## Tuning

| Knob | Where | Notes |
|---|---|---|
| Default weights | `DEFAULT_WEIGHTS` in `ml.relationship_signals` | Adjust the heuristic balance |
| Pattern memory boost / penalty | `PATTERN_*` consts in `ml.relationship_inferrer` | Boost +0.15, penalty −0.40 by default |
| Classifier blend ratio | `0.7 * classifier_p + 0.3 * heuristic` in `_final_confidence` | Lower the 0.7 if the classifier overfits |
| Activation thresholds | `MIN_TRAINING_EXAMPLES` / `MIN_PER_CLASS` in `ml.relationship_classifier` | Raise to delay activation on noisy data |

---

## Tests

| File | Coverage |
|---|---|
| `tests/test_relationship_signals.py` | Each signal in isolation — typing, names, value subset, PK-likeness, weighted combine |
| `tests/test_relationship_feedback_and_classifier.py` | JSONL persistence, pattern lookups, classifier cold-start + activation + retraining |
| `tests/test_relationship_inferrer.py` | End-to-end: two/three-table chains, ambiguity dedup, threshold, pattern-memory boost, classifier activation |
| `tests/test_cli_relationship_inference.py` | `infer-relationships` + `record-feedback` CLI dispatch and round-trip |
| `tests/test_relationship_knowledge_graph.py` | Knowledge-graph disambiguation on synthetic lookup schemas and `config/Creditcard_no_rel.xlsx` |
