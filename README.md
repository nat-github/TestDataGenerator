# Synthetic Data Platform (`sdp`)

[![CI](https://github.com/nat-github/TestDataGenerator/actions/workflows/ci.yml/badge.svg)](https://github.com/nat-github/TestDataGenerator/actions/workflows/ci.yml)

Relationship-aware synthetic data generation. Produces realistic, referentially
consistent Parquet data — plus delta (CDC) and SCD2 history — from Excel, YAML
or JSON configs. Built on SDV (Synthetic Data Vault) with an automatic
rule-based fallback. Also generates API stubs/mocks (WireMock, Pact, Postman,
OpenAPI examples).

The platform can be driven four ways:

| Interface | Entry point | Install |
|---|---|---|
| **CLI** | `sdp ...` (or `python main.py ...`) | core |
| **Python SDK** | `from sdp import SyntheticDataPlatform` | core |
| **REST API** | `sdp-api` / `uvicorn sdp.api:app` | `--extras api` |
| **MCP server** | `python -m sdp.mcp_server.server` | `--extras mcp` |

## Install

```bash
# From source (Poetry)
poetry install

# Build a wheel
poetry build            # → dist/synthetic_data_platform-<ver>-py3-none-any.whl

# Install the wheel elsewhere
pip install dist/synthetic_data_platform-*.whl
pip install "synthetic-data-platform[api] @ ./dist/synthetic_data_platform-*.whl"
```

Optional extras: `api`, `ui`, `mcp`, `gx`, `mimesis`.

## CLI

```bash
sdp generate --config config/Acct_bkng.xlsx --output output/run_01 --seed 42
sdp lint --config config/Acct_bkng.xlsx
sdp infer-relationships --config config/bare.xlsx --config-output config/inferred.yaml
```

`python main.py ...` remains supported for backward compatibility.

## Python SDK

```python
from sdp import SyntheticDataPlatform

sdp = SyntheticDataPlatform()
result = sdp.generate(config="config/Acct_bkng.xlsx", output="output/run_01", seed=42)
print(result.tables.keys(), result.total_records)

frames = result.frames          # dict[str, pandas.DataFrame] — in-memory access
sdp.lint("config/Acct_bkng.xlsx")
```

See `SDK_and_API.md` for the full surface.

## REST API

```bash
pip install "synthetic-data-platform[api]"
sdp-api --host 0.0.0.0 --port 8000      # or: uvicorn sdp.api:app
```

Interactive docs at `http://localhost:8000/docs`. See `SDK_and_API.md`.

## Documentation

- `Usage.md` — full CLI guide
- `SDK_and_API.md` — Python SDK and REST API reference
- `Yaml_Config_Schema.md` / `Json_Config_Schema.md` — config formats
- `ML_Relationship_Inference.md` — relationship inference
- `Data_Contract_Testing.md` — verify data against a contract; detect breaking changes
- `Packaging.md` — building and distributing the wheel

## Tests

```bash
poetry run pytest tests/ -v
```
