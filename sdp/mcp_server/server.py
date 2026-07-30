"""MCP server exposing Synthetic Data Platform capabilities.

Built on the official MCP Python SDK's `FastMCP` helper for terseness:

    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("synthetic-data-platform")

    @mcp.tool()
    def generate_data(...): ...

Run with:
    poetry run python -m mcp_server.server

Or, plumbed through Claude Desktop / Claude Code via the `mcpServers` block
in their settings JSON — see MCP_Integration.md for copy-pasteable configs.

Tools exposed (every one a thin wrapper over an existing platform function):

  - generate_data           Generate Parquet from a config
  - lint_config             Validate a config without generating
  - infer_relationships     Run the ML/LLM relationship inferrer
  - mock_init               OpenAPI / Postman / HAR → sdp-mock-v1 YAML
  - mock_render             sdp-mock-v1 → WireMock / JSON / Pact / Postman
  - list_examples           Enumerate the bundled example configs

Resources exposed:

  - sdp://example/<name>    Read a bundled example config

Following the MCP convention, tool descriptions are written for an *agent*
audience (Claude) — they explain when to call the tool, not just what it
does.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable when launched as `python -m sdp.mcp_server.server`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - guidance for the install
    sys.stderr.write(
        "MCP SDK not installed. Run: poetry install --extras mcp\n"
    )
    raise

logger = logging.getLogger("sdp.mcp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "configs"
OPENAPI_EXAMPLES_DIR = REPO_ROOT / "examples" / "openapi"
MOCKS_EXAMPLES_DIR = REPO_ROOT / "examples" / "mocks"

# Hard cap so an agent can't accidentally request a billion-row generation
# (the kind of mistake that costs real money in CI). Mirrors the UI cap.
MAX_ROWS_PER_TABLE = 10_000


mcp = FastMCP("synthetic-data-platform")


# ---------------------------------------------------------------------------
# Tools — data track
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_data(
    config_path: str,
    output_dir: str,
    default_records: int = 200,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Generate synthetic Parquet data from an Excel/YAML/JSON config.

    Use this when the user wants test data for a tabular schema. The
    `config_path` should point to a workbook or YAML/JSON file describing
    tables and columns. The result is one Parquet file per table written
    under `output_dir`.

    Hard cap: 10,000 rows per table. For larger runs, instruct the user
    to invoke `python main.py generate` directly.
    """
    from sdp.generators.data_generator import DataGenerator
    from sdp.utils.config_parser import ConfigParser

    cap = min(int(default_records), MAX_ROWS_PER_TABLE)
    cfg_path = Path(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = ConfigParser(str(cfg_path))
    if not parser.load_config():
        return {"ok": False, "error": "Failed to load config"}
    tables = parser.parse_tables()
    parser.parse_relationships()

    # DataGenerator takes the config *path* and parses it itself; the seed is
    # a constructor argument, not a per-call one.
    gen = DataGenerator(str(cfg_path), seed=seed)
    if not gen.load_configuration():
        return {"ok": False, "error": "Failed to load configuration"}

    records_config = {
        name: cap for name, cfg in tables.items() if cfg.active
    }

    try:
        gen.create_sdv_metadata()
        gen.train_synthesizer(sample_size=min(cap, 200))
        gen.generate_data(records_config)
        gen.export_to_parquet(str(out_dir))
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }

    parquet_files = sorted(p.name for p in out_dir.glob("*.parquet"))
    return {
        "ok": True,
        "output_dir": str(out_dir.resolve()),
        "tables": parquet_files,
        "rows_per_table": cap,
        "seed": seed,
        "capped_at": MAX_ROWS_PER_TABLE,
    }


@mcp.tool()
def lint_config(config_path: str) -> Dict[str, Any]:
    """Validate a synthetic-data config without generating any data.

    Use this before `generate_data` when the user is uncertain a config is
    valid, or after editing one. Returns a per-table summary plus any
    parser errors with row/column context.
    """
    from sdp.utils.config_parser import ConfigParser

    parser = ConfigParser(config_path)
    try:
        if not parser.load_config():
            return {"ok": False, "errors": ["Failed to load config"], "tables": []}
        tables = parser.parse_tables()
        rels = parser.parse_relationships() or []
        return {
            "ok": True,
            "tables": [
                {"name": n, "columns": len(t.columns), "rows": t.num_rows or 0}
                for n, t in tables.items()
            ],
            "relationships": len(rels),
        }
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


@mcp.tool()
def infer_relationships(
    config_path: str,
    config_output: str,
    method: str = "ml",
    ml_confidence: float = 0.55,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Infer foreign-key relationships from a config that doesn't declare them.

    Use this when the user has tables but no relationships block, and wants
    the platform to suggest FKs. `method` is "ml" (free, deterministic),
    "llm" (any LLM provider — see below), or "both".

    LLM provider configuration (passed through to llm.multi_provider):
      - llm_provider: "anthropic" (default), "openai", "lm-studio", "ollama",
        "azure-openai", "groq", "together", "openrouter"
      - llm_model: provider-specific model identifier
      - llm_base_url: override base URL (e.g. http://localhost:1234/v1 for
        a custom LM Studio port)

    When omitted, these read from SDP_LLM_PROVIDER / SDP_LLM_MODEL /
    SDP_LLM_BASE_URL env vars, then fall back to the hosted Anthropic
    default (which requires ANTHROPIC_API_KEY).

    The output YAML is annotated with confidence + signals. The user can
    edit it and feed accept/reject decisions back via record-feedback.
    """
    from sdp.utils.config_parser import ConfigParser
    from sdp.ml.relationship_inferrer import MLRelationshipInferrer

    parser = ConfigParser(config_path)
    if not parser.load_config():
        return {"ok": False, "error": "Failed to load config"}
    tables = parser.parse_tables()
    existing = parser.parse_relationships() or []

    inferred: List[Any] = []
    if method in ("ml", "both"):
        ml = MLRelationshipInferrer(confidence_threshold=ml_confidence)
        result = ml.infer(tables, existing_relationships=existing)
        inferred.extend(result.relationships)

    if method in ("llm", "both"):
        try:
            from sdp.llm.relationship_inferrer import RelationshipInferrer
            llm = RelationshipInferrer(
                confidence_threshold=0.7,
                provider=llm_provider,
                model=llm_model,
                base_url=llm_base_url,
            )
            llm_result = llm.infer(tables, existing_relationships=existing)
            seen = {(r.source_table, r.source_column, r.target_table, r.target_column) for r in inferred}
            for r in llm_result.relationships:
                key = (r.source_table, r.source_column, r.target_table, r.target_column)
                if key not in seen:
                    inferred.append(r)
        except Exception as exc:
            return {"ok": False, "error": f"LLM inference failed: {exc}"}

    # Run the existing CLI helper to write the reviewable YAML
    from sdp.cli import _write_reviewable_yaml
    out_path = Path(config_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_reviewable_yaml(out_path, existing, inferred)

    return {
        "ok": True,
        "config_output": str(out_path.resolve()),
        "method": method,
        "existing_relationships": len(existing),
        "inferred_relationships": len(inferred),
        "inferred": [
            {
                "source": f"{r.source_table}.{r.source_column}",
                "target": f"{r.target_table}.{r.target_column}",
                "confidence": r.ml_confidence if r.inferred_by_ml else r.llm_confidence,
                "kind": "ML" if r.inferred_by_ml else "LLM",
            }
            for r in inferred
        ],
    }


# ---------------------------------------------------------------------------
# Tools — mocks track
# ---------------------------------------------------------------------------


@mcp.tool()
def mock_init(
    source: str,
    output: str,
    source_type: str = "auto",
) -> Dict[str, Any]:
    """Convert an OpenAPI spec / Postman collection / HAR capture into a sdp-mock-v1 YAML.

    Use this when the user has any of those API artefacts and wants to
    generate stubs from them. Auto-detects the source format; pass
    `source_type` of "openapi", "postman", or "har" to override.
    """
    from sdp.cli import _detect_mock_source_type
    from sdp.mocks.config_parser import dump_mock_config

    detected = _detect_mock_source_type(source, source_type)
    if detected == "auto":
        return {"ok": False, "error": "Could not detect source format; pass source_type explicitly."}

    try:
        if detected == "openapi":
            from sdp.mocks.openapi_importer import import_openapi
            cfg = import_openapi(source)
        elif detected == "postman":
            from sdp.mocks.postman_importer import import_postman
            cfg = import_postman(source)
        elif detected == "har":
            from sdp.mocks.har_importer import import_har
            cfg = import_har(source)
        else:
            return {"ok": False, "error": f"Unknown source_type: {detected}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_mock_config(cfg, out_path)

    return {
        "ok": True,
        "source_type": detected,
        "output": str(out_path.resolve()),
        "endpoints": [
            {"name": ep.name, "method": ep.method, "path": ep.path,
             "statuses": [r.status for r in ep.responses]}
            for ep in cfg.endpoints
        ],
        "schemas": list(cfg.schemas.keys()),
    }


@mcp.tool()
def mock_render(
    config_path: str,
    output_dir: str,
    formats: str = "wiremock,json",
    examples: int = 3,
    seed: Optional[int] = 42,
    match_mode: str = "concrete",
    pact_consumer: str = "consumer",
    pact_provider: str = "provider",
    openapi_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Render mocks (WireMock / JSON / Pact / Postman / OpenAPI examples) from a sdp-mock-v1 config.

    Use this after `mock_init` (or once the user already has a sdp-mock-v1
    YAML). `formats` is a comma-separated list of any of: wiremock, json,
    pact, postman, openapi-examples. The openapi-examples format requires
    `openapi_source` to point at the original spec.
    """
    from sdp.mocks.config_parser import load_mock_config

    try:
        cfg = load_mock_config(config_path)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmts = [f.strip().lower() for f in formats.split(",") if f.strip()]
    written: Dict[str, int] = {}

    for fmt in fmts:
        target = out_dir / fmt
        try:
            if fmt == "wiremock":
                from sdp.mocks.renderers.wiremock import render_wiremock
                files = render_wiremock(cfg, target, seed=seed,
                                        examples_per_endpoint=examples,
                                        match_mode=match_mode)
            elif fmt in ("json", "json-fixture", "fixture"):
                from sdp.mocks.renderers.json_fixture import render_json_fixtures
                files = render_json_fixtures(cfg, target, seed=seed)
            elif fmt == "pact":
                from sdp.mocks.renderers.pact import render_pact
                files = render_pact(cfg, target, consumer=pact_consumer,
                                    provider=pact_provider, seed=seed,
                                    examples_per_endpoint=examples)
            elif fmt == "postman":
                from sdp.mocks.renderers.postman import render_postman
                files = render_postman(cfg, target, seed=seed,
                                       examples_per_endpoint=examples)
            elif fmt in ("openapi-examples", "openapi"):
                if not openapi_source:
                    return {"ok": False, "error": "openapi-examples requires openapi_source"}
                from sdp.mocks.renderers.openapi_examples import render_openapi_examples
                files = render_openapi_examples(cfg, target,
                                                source_spec=openapi_source,
                                                seed=seed)
            else:
                return {"ok": False, "error": f"Unknown format: {fmt}"}
        except Exception as exc:
            return {"ok": False, "error": f"{fmt} renderer failed: {exc}"}
        written[fmt] = len(files)

    return {
        "ok": True,
        "output_dir": str(out_dir.resolve()),
        "files_written": written,
        "total_files": sum(written.values()),
    }


# ---------------------------------------------------------------------------
# Tools — discoverability
# ---------------------------------------------------------------------------


@mcp.tool()
def mock_enrich(
    config_path: str,
    output: str,
    fill_missing_examples: bool = True,
    draft_error_responses: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Use an LLM to fill missing schema examples and draft 4xx/5xx responses.

    Routes through llm.multi_provider, so any supported backend works:
    Anthropic (default), OpenAI, LM Studio, Ollama, Azure OpenAI, Groq,
    Together, OpenRouter. When `llm_provider` etc. are omitted, the
    SDP_LLM_PROVIDER / SDP_LLM_MODEL / SDP_LLM_BASE_URL env vars are
    honoured.
    """
    try:
        from sdp.mocks.config_parser import load_mock_config, dump_mock_config
        from sdp.mocks.llm_enricher import enrich
    except Exception as exc:
        return {"ok": False, "error": f"mocks track unavailable: {exc}"}

    try:
        cfg = load_mock_config(config_path)
        enriched, result = enrich(
            cfg,
            fill_missing_examples=fill_missing_examples,
            draft_error_responses=draft_error_responses,
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
        )
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_mock_config(enriched, out_path)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "output": str(out_path.resolve()),
        "examples_added": result.examples_added,
        "error_responses_added": result.error_responses_added,
        "schemas_touched": list(set(result.schemas_touched)),
        "endpoints_touched": list(set(result.endpoints_touched)),
        "warnings": result.warnings,
    }


@mcp.tool()
def llm_diagnose(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a tiny ping to the configured LLM and report what came back.

    Use this as a connectivity sanity check before invoking
    infer_relationships(method="llm") or mock_enrich. Confirms the
    provider/model/base_url combination resolves and the model responds.

    Returns the resolved provider config, the model's reply, and
    elapsed milliseconds. On failure returns {"ok": false, "error": ...}.
    """
    import time
    try:
        from sdp.llm.multi_provider import chat, resolve_config
        cfg = resolve_config(provider=provider, model=model, base_url=base_url)
        started = time.perf_counter()
        reply = chat(
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            provider=provider,
            model=model,
            base_url=base_url,
            max_tokens=20,
            temperature=0.0,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "provider": cfg.provider.name,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "api_key_set": bool(cfg.api_key),
            "reply": reply.strip()[:200],
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
def validate_data(
    config_path: str,
    output_dir: str,
    tolerance: float = 0.5,
) -> Dict[str, Any]:
    """Run Great Expectations validation on generated Parquet output.

    Use this after generate_data to verify the output complies with the
    constraints declared in the config (PK uniqueness, business_values,
    min/max, special_rules with deterministic shapes, row counts).

    Requires `poetry install --extras gx`.
    """
    try:
        from sdp.validators.gx_validator import HAS_GX, validate_tables, format_report
    except Exception as exc:
        return {"ok": False, "error": f"validators unavailable: {exc}"}
    if not HAS_GX:
        return {
            "ok": False,
            "error": "Great Expectations not installed. Run: poetry install --extras gx",
        }

    from sdp.utils.config_parser import ConfigParser
    parser = ConfigParser(config_path)
    if not parser.load_config():
        return {"ok": False, "error": "Failed to load config"}
    tables = parser.parse_tables()
    parser.parse_relationships()

    try:
        report = validate_tables(
            tables,
            output_dir=output_dir,
            row_count_tolerance=tolerance,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": report.success,
        "summary": format_report(report, verbose=False),
        "total_expectations": report.total_expectations,
        "total_passed": report.total_passed,
        "total_failed": report.total_failed,
        "tables": {
            name: {
                "row_count": t.row_count,
                "passed": t.passed_expectations,
                "failed": t.failed_expectations,
                "error": t.error,
            }
            for name, t in report.tables.items()
        },
    }


@mcp.tool()
def quality_report(
    generated_dir: str,
    source_dir: Optional[str] = None,
    privacy_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Statistical quality / fidelity / privacy report on generated Parquet output.

    Two modes:

      - **Univariate-only** (no source_dir): per-column dtype, null rate,
        unique count, summary stats, plus a Pearson correlation matrix.
        Use this to answer "does this synthetic data look plausible on
        its own?"

      - **Fidelity vs source** (source_dir provided): adds KS test for
        numeric columns, total-variation distance for categoricals,
        correlation-matrix delta, and a nearest-neighbour distance
        privacy proxy. Use this to answer "is this synthetic data
        faithful to the source distribution while not leaking individual
        rows?"

    Returns the structured report. For human-readable output, the
    response includes a `markdown_summary` field. The full per-column
    metrics live under `tables[<table_name>].columns`.
    """
    try:
        from sdp.validators.quality_report import quality_report_from_paths
    except Exception as exc:
        return {"ok": False, "error": f"validators unavailable: {exc}"}
    try:
        report = quality_report_from_paths(
            synthetic_dir=generated_dir,
            source_dir=source_dir,
            privacy_threshold=privacy_threshold,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    payload = report.to_dict()
    payload["ok"] = True
    payload["markdown_summary"] = report.to_markdown(max_columns_shown=30)
    return payload


@mcp.tool()
def list_examples() -> Dict[str, Any]:
    """List the bundled example configs that ship with the platform.

    Use this to discover what's available before suggesting a config to the
    user. Returns separate buckets for the data-track examples (yaml / json /
    xlsx variants) and the mocks-track examples (openapi specs and reverse-
    import fixtures).
    """
    data_yaml = sorted(p.name for p in (EXAMPLES_DIR / "yaml").glob("*.yaml")) if (EXAMPLES_DIR / "yaml").exists() else []
    data_json = sorted(p.name for p in (EXAMPLES_DIR / "json").glob("*.json")) if (EXAMPLES_DIR / "json").exists() else []
    data_xlsx = sorted(p.name for p in (EXAMPLES_DIR / "xlsx").glob("*.xlsx")) if (EXAMPLES_DIR / "xlsx").exists() else []

    openapi_specs = sorted(p.name for p in OPENAPI_EXAMPLES_DIR.glob("*.yaml")) if OPENAPI_EXAMPLES_DIR.exists() else []
    mocks_fixtures = sorted(p.name for p in MOCKS_EXAMPLES_DIR.glob("*")) if MOCKS_EXAMPLES_DIR.exists() else []

    return {
        "data_track": {
            "yaml": data_yaml,
            "json": data_json,
            "xlsx": data_xlsx,
            "directory": str(EXAMPLES_DIR.resolve()),
        },
        "mocks_track": {
            "openapi": openapi_specs,
            "fixtures": mocks_fixtures,
            "openapi_dir": str(OPENAPI_EXAMPLES_DIR.resolve()),
            "fixtures_dir": str(MOCKS_EXAMPLES_DIR.resolve()),
        },
    }


# ---------------------------------------------------------------------------
# Resources — let agents read example contents directly
# ---------------------------------------------------------------------------


@mcp.resource("sdp://example/{name}")
def read_example(name: str) -> str:
    """Read a bundled example config by filename (e.g. `01_simple_users.yaml`)."""
    candidates: List[Path] = []
    for sub in ("yaml", "json", "xlsx"):
        candidates.append(EXAMPLES_DIR / sub / name)
    candidates.append(OPENAPI_EXAMPLES_DIR / name)
    candidates.append(MOCKS_EXAMPLES_DIR / name)

    for p in candidates:
        if p.exists() and p.is_file():
            if p.suffix == ".xlsx":
                return f"<binary XLSX file at {p}; size={p.stat().st_size} bytes>"
            return p.read_text(encoding="utf-8")

    raise FileNotFoundError(f"No example named {name!r} found under examples/")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    transport = os.environ.get("SDP_MCP_TRANSPORT", "stdio")
    logger.info("Starting Synthetic Data Platform MCP server (transport=%s)", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
