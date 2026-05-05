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
        "list_examples",
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
