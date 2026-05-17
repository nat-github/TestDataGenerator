# Streamlit UI — Quickstart

A simple, browser-based interface for the data-generation track. Sized for
small runs (≤ 10,000 rows per table) — for larger runs, batches, CDC, mocks,
or LLM features, use the CLI.

---

## Install

```bash
# One-time: install the optional `ui` extra (adds Streamlit ~150 MB)
poetry install --extras ui
```

## Run

```bash
poetry run streamlit run sdp/ui/streamlit_app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).
Open it in your browser.

---

## What you get

The page has five sections, top to bottom:

1. **Pick a config** — choose one of the 11 bundled examples (under
   `examples/configs/`) or upload your own XLSX / YAML / JSON.
2. **Settings** — records per table (capped at 10,000), random seed.
3. **Actions** — `Generate` (runs the synthesizer) or `Lint config`
   (validates without generating).
4. **Preview** — tabbed view of the generated tables, first 100 rows each.
5. **Download** — bundles every Parquet output into a single ZIP file.

The whole flow is in-process — no subprocess, no separate API server, no
state outside what Streamlit's session model holds.

---

## When to drop back to the CLI

The UI deliberately covers the **first thing a new user wants to try** and
nothing more. For everything else:

| Need | Use |
|---|---|
| > 10,000 rows per table | `python main.py generate --default-records N` |
| Multi-table batch with custom counts | `python main.py generate --records table:count …` |
| CDC delta / SCD2 history | `python main.py delta` / `python main.py scd2` |
| LLM relationship inference | `python main.py infer-relationships` |
| LLM schema enrichment | `python main.py enrich` |
| API mocks (WireMock / Pact / Postman) | `python main.py mock-init` + `mock-render` |
| PII scanning | `python main.py pii-scan` |
| Cloud upload | `python main.py generate --upload-to azure://…` |

Every command is documented in `Examples_Walkthrough.md` with copy-pasteable
examples.

---

## Roadmap

The UI is intentionally minimal at v1. Likely additions, in order:

- **v2** — inline ER diagram preview (Mermaid; the renderer already exists).
- **v3** — separate page for `lint` only, with column-level error pinpointing.
- **v4** — separate page for the mocks track (`mock-init` + `mock-render` +
  rendered-JSON preview).
- **v5** — LLM enrichment panel with provider picker (Anthropic / OpenAI /
  LM Studio / Ollama / …).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `poetry install --extras ui` first. |
| `streamlit: command not found` | Use `poetry run streamlit run …` (or activate the venv). |
| The Generate button is greyed out | No config picked — choose a bundled example or upload one. |
| Generation hangs on > 5,000 rows | Streamlit reruns the script on each interaction; long synchronous jobs can feel laggy. The 10,000-row cap should keep things snappy on most machines. If you need bigger, use the CLI. |
| The page says "module 'pyarrow' has no attribute …" | The optional `ui` extra is meant to be additive; ensure the base install is also up to date: `poetry install`. |

---

## Tests

Streamlit's built-in `AppTest` harness runs the app inline without a
browser, fast enough for CI:

```bash
poetry run pytest tests/test_ui_streamlit.py -v
```

Seven smoke tests cover load, widget presence, and the 10k cap. New UI
features should add an AppTest case alongside.
