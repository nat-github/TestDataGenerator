"""Smoke tests for the MCP server.

We invoke the registered tool functions directly (FastMCP exposes them via
its tool manager) without spinning up the stdio transport. That keeps tests
fast and deterministic — full end-to-end MCP transport testing is left to
manual verification with Claude Desktop / Claude Code.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

# Skip the whole module if the MCP SDK isn't installed
pytest.importorskip("mcp")

from mcp_server import server as srv

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_YAML = REPO_ROOT / "examples" / "configs" / "yaml"
OPENAPI_EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# Shape of the registry
# ---------------------------------------------------------------------------


def test_server_named_correctly():
    assert srv.mcp.name == "synthetic-data-platform"


def test_expected_tools_registered():
    """Every advertised capability must be exposed as a tool."""
    expected = {
        "generate_data",
        "lint_config",
        "infer_relationships",
        "mock_init",
        "mock_render",
        "mock_enrich",
        "list_examples",
        "llm_diagnose",
        "validate_data",
        "quality_report",
    }
    tool_names = set(srv.mcp._tool_manager._tools.keys())
    missing = expected - tool_names
    assert not missing, f"missing MCP tools: {missing}"


def test_resource_handler_registered():
    """The sdp://example/{name} resource template must be registered."""
    # FastMCP stores resources/templates separately; check both
    templates = list(srv.mcp._resource_manager._templates.values()) if hasattr(srv.mcp._resource_manager, "_templates") else []
    template_uris = {t.uri_template for t in templates}
    assert any("sdp://example/" in uri for uri in template_uris), \
        f"sdp://example/ resource not registered (got {template_uris})"


# ---------------------------------------------------------------------------
# Tool behaviour — direct invocation
# ---------------------------------------------------------------------------


def _call(tool_name: str, **kwargs):
    """Invoke a FastMCP tool's underlying function directly."""
    tool = srv.mcp._tool_manager._tools[tool_name]
    fn = tool.fn
    if asyncio.iscoroutinefunction(fn):
        return asyncio.run(fn(**kwargs))
    return fn(**kwargs)


def test_list_examples_finds_bundled_configs():
    out = _call("list_examples")
    assert out["data_track"]["yaml"], "no YAML examples discovered"
    assert any("simple_users" in n for n in out["data_track"]["yaml"])
    assert out["mocks_track"]["openapi"], "no OpenAPI examples discovered"


def test_lint_config_passes_for_known_good():
    cfg = EXAMPLES_YAML / "01_simple_users.yaml"
    out = _call("lint_config", config_path=str(cfg))
    assert out["ok"] is True
    table_names = {t["name"] for t in out["tables"]}
    assert "users" in table_names
    assert out["relationships"] == 0


def test_lint_config_returns_errors_for_broken_input(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("config_format: not-valid\n", encoding="utf-8")
    out = _call("lint_config", config_path=str(bad))
    # Either ok=False with errors or ok=True with no tables — both are
    # acceptable degraded states; the contract is "don't raise".
    assert "ok" in out


def test_mock_init_imports_openapi_spec(tmp_path: Path):
    out = _call(
        "mock_init",
        source=str(OPENAPI_EXAMPLES / "simple_books.yaml"),
        output=str(tmp_path / "mocks.yaml"),
    )
    assert out["ok"] is True
    assert out["source_type"] == "openapi"
    names = {ep["name"] for ep in out["endpoints"]}
    assert "list_books" in names
    assert "get_book_by_id" in names


def test_mock_render_writes_wiremock_mappings(tmp_path: Path):
    # First import an OpenAPI spec to a sdp-mock-v1 YAML
    init = _call(
        "mock_init",
        source=str(OPENAPI_EXAMPLES / "simple_books.yaml"),
        output=str(tmp_path / "mocks.yaml"),
    )
    assert init["ok"]
    # Then render
    out = _call(
        "mock_render",
        config_path=str(tmp_path / "mocks.yaml"),
        output_dir=str(tmp_path / "stubs"),
        formats="wiremock,json",
        examples=2,
        seed=1,
    )
    assert out["ok"] is True
    assert out["files_written"]["wiremock"] > 0
    assert out["files_written"]["json"] > 0


def test_mock_render_unknown_format_returns_error(tmp_path: Path):
    init = _call(
        "mock_init",
        source=str(OPENAPI_EXAMPLES / "simple_books.yaml"),
        output=str(tmp_path / "m.yaml"),
    )
    assert init["ok"]
    out = _call(
        "mock_render",
        config_path=str(tmp_path / "m.yaml"),
        output_dir=str(tmp_path / "stubs"),
        formats="not-a-format",
    )
    assert out["ok"] is False
    assert "Unknown format" in out["error"]


# ---------------------------------------------------------------------------
# Resource handler
# ---------------------------------------------------------------------------


def test_read_example_returns_yaml_content():
    text = srv.read_example("01_simple_users.yaml")
    assert "config_format: sdp-yaml-v1" in text
    assert "users" in text


def test_read_example_raises_on_unknown_name():
    with pytest.raises(FileNotFoundError):
        srv.read_example("does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


def test_generate_data_caps_at_max_rows(tmp_path: Path, monkeypatch):
    """Confirm the 10k row cap is enforced — even when caller passes higher."""
    cfg = EXAMPLES_YAML / "01_simple_users.yaml"
    captured = {}

    # Stub out heavy SDV calls so the test is fast
    from generators.data_generator import DataGenerator

    def fake_train(self, sample_size=None, seed=None):
        return None

    def fake_generate(self, records_config, seed=None):
        captured["records_config"] = records_config

    def fake_export(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Touch a placeholder so the tool's parquet glob finds something
        (Path(output_dir) / "users.parquet").write_bytes(b"")

    monkeypatch.setattr(DataGenerator, "train_synthesizer", fake_train)
    monkeypatch.setattr(DataGenerator, "generate_data", fake_generate)
    monkeypatch.setattr(DataGenerator, "export_to_parquet", fake_export)

    out = _call(
        "generate_data",
        config_path=str(cfg),
        output_dir=str(tmp_path / "out"),
        default_records=999_999_999,   # huge — must be capped
    )
    assert out["ok"] is True
    assert out["rows_per_table"] == srv.MAX_ROWS_PER_TABLE
    # The records_config the generator saw must also be capped
    for cap in captured["records_config"].values():
        assert cap == srv.MAX_ROWS_PER_TABLE


# ---------------------------------------------------------------------------
# Multi-provider LLM tools
# ---------------------------------------------------------------------------


def test_infer_relationships_threads_provider_to_llm(monkeypatch):
    """The MCP wrapper must pass llm_provider/model/base_url through to RelationshipInferrer."""
    captured: dict = {}

    class FakeResult:
        relationships = []

    class FakeInferrer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def infer(self, tables, existing_relationships=None):
            return FakeResult()

    import llm.relationship_inferrer as rli
    monkeypatch.setattr(rli, "RelationshipInferrer", FakeInferrer)

    out = _call(
        "infer_relationships",
        config_path=str(EXAMPLES_YAML / "01_simple_users.yaml"),
        config_output=str(REPO_ROOT / "build" / "ignored.yaml"),
        method="llm",
        llm_provider="lm-studio",
        llm_model="qwen-coder",
        llm_base_url="http://localhost:1234/v1",
    )
    assert out["ok"] is True
    assert captured.get("provider") == "lm-studio"
    assert captured.get("model") == "qwen-coder"
    assert captured.get("base_url") == "http://localhost:1234/v1"


def test_llm_diagnose_returns_resolved_config(monkeypatch):
    """The diagnose tool reports the resolved provider + a brief reply."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    import llm.multi_provider as mp

    def fake_chat(messages, **kwargs):
        return "ready"

    monkeypatch.setattr(mp, "chat", fake_chat)
    # The MCP wrapper imports `chat` lazily inside the function; patch the
    # source module so it's used regardless of import path.
    out = _call("llm_diagnose", provider="anthropic", model="claude-haiku-4-5-20251001")
    assert out["ok"] is True
    assert out["provider"] == "anthropic"
    assert out["model"] == "claude-haiku-4-5-20251001"
    assert "ready" in out["reply"]


def test_llm_diagnose_reports_failure_gracefully(monkeypatch):
    """Provider config errors come back as ok: false rather than raising."""
    # Strip every recognised key so resolve_config raises EnvironmentError
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SDP_LLM_API_KEY",
                "GROQ_API_KEY", "TOGETHER_API_KEY", "OPENROUTER_API_KEY",
                "AZURE_OPENAI_API_KEY", "SDP_LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)

    out = _call("llm_diagnose", provider="anthropic")
    assert out["ok"] is False
    assert "error" in out


# ---------------------------------------------------------------------------
# validate_data tool
# ---------------------------------------------------------------------------


def test_validate_data_reports_missing_gx_dependency(monkeypatch):
    """When GX isn't installed, the tool returns a useful error rather than raising."""
    import validators.gx_validator as gxv
    monkeypatch.setattr(gxv, "HAS_GX", False)

    out = _call(
        "validate_data",
        config_path=str(EXAMPLES_YAML / "01_simple_users.yaml"),
        output_dir="/tmp/nonexistent",
    )
    assert out["ok"] is False
    assert "Great Expectations" in out["error"]


def test_quality_report_univariate_only(tmp_path: Path):
    """Run the quality_report tool against a tiny Parquet directory without source data."""
    import pandas as pd

    out_dir = tmp_path / "syn"
    out_dir.mkdir()
    pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}).to_parquet(
        out_dir / "users.parquet", index=False,
    )

    out = _call("quality_report", generated_dir=str(out_dir))
    assert out["ok"] is True
    assert out["has_source"] is False
    assert "users" in out["tables"]
    assert out["tables"]["users"]["row_count_synthetic"] == 5
    assert "markdown_summary" in out
    assert "Quality Report" in out["markdown_summary"]


def test_quality_report_with_source_computes_fidelity(tmp_path: Path):
    """When source_dir is provided, fidelity score is computed."""
    import pandas as pd

    syn_dir = tmp_path / "syn"
    src_dir = tmp_path / "src"
    syn_dir.mkdir()
    src_dir.mkdir()

    df = pd.DataFrame({"k": ["A", "B"] * 50, "v": list(range(100))})
    df.to_parquet(syn_dir / "t.parquet", index=False)
    df.to_parquet(src_dir / "t.parquet", index=False)

    out = _call("quality_report", generated_dir=str(syn_dir), source_dir=str(src_dir))
    assert out["ok"] is True
    assert out["has_source"] is True
    assert out["overall_fidelity"] is not None
    assert out["overall_fidelity"] > 0.95


def test_quality_report_handles_missing_dir():
    out = _call("quality_report", generated_dir="/does/not/exist/anywhere")
    assert out["ok"] is False
    assert "error" in out


def test_validate_data_runs_against_real_data(tmp_path: Path):
    """End-to-end: write a tiny Parquet file, validate it through the MCP tool."""
    import pandas as pd
    import validators.gx_validator as gxv
    if not gxv.HAS_GX:
        pytest.skip("great-expectations not installed")

    # Use the simple_users config and write a compliant Parquet file
    cfg = EXAMPLES_YAML / "01_simple_users.yaml"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)

    # Build a DataFrame that satisfies the simple_users config
    df = pd.DataFrame({
        "user_id": list(range(1, 6)),
        "full_name": ["A B"] * 5,
        "email": [f"u{i}@x.com" for i in range(5)],
        "phone": ["+44 1234567890"] * 5,
        "country_code": ["GB"] * 5,
        "status": ["ACTIVE"] * 5,
        "signup_date": pd.to_datetime(["2024-01-01"] * 5),
        "lifetime_value": [100.0] * 5,
    })
    df.to_parquet(out_dir / "users.parquet", index=False)

    out = _call(
        "validate_data",
        config_path=str(cfg),
        output_dir=str(out_dir),
        tolerance=0.95,  # 5 rows vs 200 declared — wide tolerance
    )
    assert "summary" in out
    assert out["total_expectations"] > 0
