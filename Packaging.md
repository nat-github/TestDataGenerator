# Packaging

The platform is a single installable Python package, `sdp`, distributed as a
standard wheel.

## Layout

Every module lives under the `sdp/` namespace:

```
sdp/
├── __init__.py          # version + SyntheticDataPlatform re-export
├── cli.py               # CLI implementation (console script: `sdp`)
├── sdk.py               # programmatic SDK facade
├── api/                 # REST API (FastAPI)  — console script: `sdp-api`
├── generators/          # data generation orchestrator
├── utils/               # config parsing, helpers, post-processing
├── models/              # pydantic config + mock models
├── llm/                 # LLM providers + inference/enrichment
├── ml/                  # ML relationship inference + knowledge graph
├── mocks/               # API stubs/mocks track
├── validators/          # Great Expectations + quality reports
├── mcp_server/          # MCP server
└── ui/                  # Streamlit UI
```

A root `main.py` shim is kept so `python main.py ...` and `from main import ...`
keep working; the canonical entry point is the installed `sdp` console script.

## Build the wheel

```bash
poetry install            # dev environment
poetry build              # → dist/synthetic_data_platform-<version>-py3-none-any.whl
                          #   dist/synthetic_data_platform-<version>.tar.gz
```

The wheel version is `[tool.poetry] version` in `pyproject.toml`
(kept in sync with `sdp.__version__`).

## Install

```bash
# Core
pip install dist/synthetic_data_platform-*.whl

# With optional extras
pip install "dist/synthetic_data_platform-*.whl[api]"
pip install "dist/synthetic_data_platform-*.whl[api,ui,gx]"
```

| Extra | Enables | Pulls in |
|---|---|---|
| `api` | REST API (`sdp-api`) | fastapi, uvicorn, python-multipart |
| `ui` | Streamlit UI | streamlit |
| `mcp` | MCP server | mcp |
| `gx` | Great Expectations validation | great-expectations |
| `mimesis` | `MIMESIS_*` special rules | mimesis |

## Console scripts

The wheel installs two commands onto `PATH`:

| Command | Entry point | Purpose |
|---|---|---|
| `sdp` | `sdp.cli:main` | Full CLI (`generate`, `delta`, `scd2`, `lint`, ...) |
| `sdp-api` | `sdp.api.__main__:main` | Launch the REST API via uvicorn |

```bash
sdp generate --config config/Acct_bkng.xlsx --output output/run_01 --seed 42
sdp-api --host 0.0.0.0 --port 8000
```

## Verify a build

```bash
poetry build
python -m zipfile -l dist/synthetic_data_platform-*.whl   # inspect wheel contents

# Smoke test in a throwaway environment
python -m venv /tmp/sdp-check && /tmp/sdp-check/bin/pip install dist/*.whl
/tmp/sdp-check/bin/sdp --help
```

## Publishing

Out of scope for the current setup — `poetry build` produces the artefacts in
`dist/` for manual distribution. To publish later: `poetry publish` (configure a
PyPI token first with `poetry config pypi-token.pypi <token>`).
