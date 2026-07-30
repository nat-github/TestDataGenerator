# Platform Direction — open decisions

Working notes from the architecture sessions. This captures **decisions not
yet made** and **context that is not derivable from the code**. Everything
already built is described in `CLAUDE.md` and the git history.

Last updated: 2026-07-31. Branch: `feature/ClaudeCodeupdate` @ `370db85`.

---

## 1. The open question: multiple flavours, one platform

The proposal is to extend beyond relationship-aware tabular data into:

- **ML model testing** — stress sets: rare-slice amplification, controlled
  distribution shift, counterfactual pairs, boundary rows
- **LLM evaluation sets** — prompts, multi-turn conversations, tool calls,
  rubrics
- **LLM fine-tuning data** — instruction/response pairs

### The framing that was agreed

These are **flavours of one platform, not one generator**. A flavour may
share nothing internally with the tabular track, and that is fine — forced
reuse across dissimilar domains produces the worst abstractions.

What makes it one platform is the **experience**, not the internals. Four
things must be common:

| | Meaning |
|---|---|
| **One noun** | Today: "a config describes a dataset." For LLM evals: "a config describes a case set." If the shared noun cannot be stated in one sentence, it is a monorepo, not a platform. |
| **The same verbs** | `generate`, `lint`, `evaluate`, `export` — a user who learns one flavour can drive the next. This is the entire value of the umbrella. |
| **One quality vocabulary** | Tabular: fidelity / utility / bias / privacy. LLM evals: roughly coverage / difficulty / contamination / leakage. The report *shape* and the discipline carry over even when the metrics do not. |
| **One governance story** | Provenance, PII handling, what a privacy claim actually means. Every flavour will have its own version of the DP trap (ε over config-generated data protects nothing) and users need one place that tells them the truth. |

### Structural implication (not yet built)

A `kind:` discriminator, in the style of Kubernetes `apiVersion`/`kind`:

```yaml
config_format: sdp-yaml-v1
kind: tabular          # | llm-eval | ml-stress | finetune
```

- `sdp/domains/<kind>/` — each owning its models, generators, renderers
- Shared: CLI verb dispatch, config loading, schema validation, report shell
- **Not** shared: generation cores
- JSON Schema: `oneOf` on `kind`, so each flavour gets its own *closed*
  schema and its own IDE autocompletion
- Per-flavour extras: `sdp[llm-eval]`, `sdp[tabular]` — this is also the
  forcing function that finally makes `sdv`/`torch` optional

### Recommended sequencing

1. **ML model testing** — barely an extension. Layer A rules and Layer C
   workflows already express most stress-set intent declaratively. Highest
   value, lowest risk, best test of whether the "one platform" thesis holds.
2. **LLM eval sets** — the mocks track already turns a spec into structured
   cases with stateful scenarios; a multi-turn conversation is a state
   machine over turns. The genuinely new part is *semantic* validity, which
   needs an LLM-as-judge in the **evaluation** layer, not the generation
   layer.
3. **Fine-tuning data — last, or not in-house.** Note the irony: the
   DeepMind paper that inspired this direction
   ([arXiv:2404.07503](https://arxiv.org/html/2404.07503)) is the one
   warning hardest about it — factuality, alignment ambiguity, and
   benchmark contamination where synthetic rephrasings leak into training
   and token-level decontamination stops working. The failure mode is a
   model that degrades undetectably. Crowded space, mature external tools.
   Prefer to consume.

### Next concrete step (not started)

Write a two-page **platform charter** defining the noun, the verbs, the
quality vocabulary and the governance story — *before* any flavour code.
Any flavour that satisfies the charter earns its place; any that cannot is
a separate product wearing the same logo. That document is worth more than
the first flavour's code, because it is what stops flavour four from being
incoherent.

---

## 2. Rollout readiness — the blockers no code review scores

A separate thread: the platform was assessed for **company-wide rollout**.
Code quality is now reasonable (~7.5/10 on an external review). These are
the things that would actually sink a rollout, and none of them appear on a
code-quality scale:

- **CI has never run on this work.** The branch is pushed and merged but no
  pipeline has executed against it.
- **One maintainer.** No CODEOWNERS, no contribution guide, no second
  person who understands the architecture.
- **No release process.** `version = "0.3.0"` in `pyproject.toml`, no tags,
  no changelog, no deprecation policy.
- **No on-ramp.** 30+ root-level markdown docs, a 2.6 KB README. Excellent
  reference material, no starting point. This becomes the support queue.
- **Unproven at scale.** Everything measured has been hundreds to thousands
  of rows. There is a `OneBillion_SavingsBooking.xlsx` in `config/` and no
  evidence of a run at that size.

### Recommendation on record

**Pilot with 2–3 teams for a quarter before going wider.** The pilot exists
mainly to find the next never-executed-code-path bug — three shipped
commands (MCP generation, `pii-scan`, `collibra-import`) had *never worked*
and were found by inspection, not by users. That class of bug is cheap to
find with 3 teams and expensive with 30.

Gates before wider rollout: CI green on every commit; pick the supported
surface and mark the rest experimental; one real README; tag a version and
write a changelog; one honest scale test; a second maintainer reading code.

---

## 3. Deferred engineering items

Explicitly scoped out, with reasons — not forgotten:

| Item | Why deferred |
|---|---|
| `torch` / `sdv` optional | Needs `data_generator` decoupled from module-level `sdv` imports. Touches the core path; a botched job costs more than the packaging win. Becomes natural if per-flavour extras land. |
| `helpers.py` (~1,900 lines) | Now has 34 unit tests but is untouched structurally. The last monolith. |
| Remaining 104 broad `except Exception` | **Deliberate.** See the "Where `except Exception` is correct" section in `CLAUDE.md` — MCP's 13 are the tool error contract, third-party boundaries raise undocumented types, and the per-cell Arrow handlers already name their exceptions. Reviews keep re-flagging the raw count; that section exists to stop it. |

---

## 4. Reference — what the papers actually said

The direction was inspired by three papers. Two of them are **not about
relational tabular data**, which is worth remembering before importing
their recommendations wholesale:

- [arXiv:2401.02524](https://arxiv.org/abs/2401.02524) — surveys 417 models
  and states plainly that *computer vision dominates*. Its taxonomy mostly
  does not transfer. Its useful points: metric fragmentation, and that
  training/compute cost is neglected (which is why engines report cost).
- [arXiv:2404.07503](https://arxiv.org/html/2404.07503) (DeepMind) — about
  synthetic data **for training AI models**. Directly relevant to the
  fine-tuning flavour; also the strongest warning against it.
- [Electronics 13:3509](https://www.mdpi.com/2079-9292/13/17/3509) —
  Goyal & Mahmoud. Closest fit. Its six open challenges drove the
  evaluation and DP work already shipped. Names *"user-friendly tools and
  frameworks"* as future work — which is what this platform is.
- [arXiv:2402.06806](https://arxiv.org/abs/2402.06806) — not on the
  original list but the most useful of all for a tabular platform:
  the fidelity / utility / privacy triad this codebase now implements.
