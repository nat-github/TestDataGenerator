"""CLI tests for the Phase E / H / I additions to the mocks track.

Covers:
  - mock-render: pact, postman, openapi-examples formats
  - mock-init: source-type auto-detection across openapi / postman / har
  - mock-enrich: argparse plumbing + dispatch (LLM call mocked)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from main import main, parse_arguments

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# Argparse — new flags
# ---------------------------------------------------------------------------


def test_argparse_mock_render_accepts_pact_postman_flags():
    args = parse_arguments([
        "mock-render",
        "--config", "m.yaml",
        "--output", "out/",
        "--format", "pact,postman",
        "--pact-consumer", "client",
        "--pact-provider", "server",
    ])
    assert args.format == "pact,postman"
    assert args.pact_consumer == "client"
    assert args.pact_provider == "server"


def test_argparse_mock_render_accepts_openapi_examples_args():
    args = parse_arguments([
        "mock-render",
        "--config", "m.yaml",
        "--output", "out/",
        "--format", "openapi-examples",
        "--openapi-source", "spec.yaml",
        "--openapi-overwrite",
    ])
    assert args.openapi_source == "spec.yaml"
    assert args.openapi_overwrite is True


def test_argparse_mock_init_source_type_flag_defaults_to_auto():
    args = parse_arguments([
        "mock-init",
        "--from", "x.yaml",
        "--output", "y.yaml",
    ])
    assert args.source_type == "auto"


def test_argparse_mock_enrich_requires_config_and_output():
    args = parse_arguments([
        "mock-enrich",
        "--config", "m.yaml",
        "--output", "enriched.yaml",
    ])
    assert args.command == "mock-enrich"
    assert args.no_fill_examples is False
    assert args.no_draft_errors is False


def test_argparse_mock_enrich_passes_provider_overrides():
    args = parse_arguments([
        "mock-enrich",
        "--config", "m.yaml",
        "--output", "enriched.yaml",
        "--llm-provider", "lm-studio",
        "--llm-model", "qwen-coder",
        "--llm-base-url", "http://localhost:1234/v1",
    ])
    assert args.llm_provider == "lm-studio"
    assert args.llm_model == "qwen-coder"
    assert args.llm_base_url == "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# mock-render — Phase E formats end-to-end
# ---------------------------------------------------------------------------


def _init_mocks(tmp_path: Path, spec: str = "simple_books.yaml") -> Path:
    out = tmp_path / "mocks.yaml"
    main(["mock-init", "--from", str(EXAMPLES / spec), "--output", str(out)])
    return out


def test_mock_render_writes_pact_contract(tmp_path: Path):
    cfg = _init_mocks(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render", "--config", str(cfg), "--output", str(out_dir),
        "--format", "pact", "--pact-consumer", "reader", "--pact-provider", "books",
        "--examples", "2", "--seed", "1",
    ])
    assert rc == 0
    files = list((out_dir / "pact").iterdir())
    assert files
    pact = json.loads(files[0].read_text(encoding="utf-8"))
    assert pact["consumer"]["name"] == "reader"
    assert pact["provider"]["name"] == "books"
    assert pact["interactions"]


def test_mock_render_writes_postman_collection(tmp_path: Path):
    cfg = _init_mocks(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render", "--config", str(cfg), "--output", str(out_dir),
        "--format", "postman", "--examples", "2", "--seed", "1",
    ])
    assert rc == 0
    files = list((out_dir / "postman").iterdir())
    assert files
    coll = json.loads(files[0].read_text(encoding="utf-8"))
    assert "item" in coll
    assert "v2.1.0" in coll["info"]["schema"]


def test_mock_render_openapi_examples_requires_source(tmp_path: Path):
    """Missing --openapi-source should produce an error (no files written)."""
    cfg = _init_mocks(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render", "--config", str(cfg), "--output", str(out_dir),
        "--format", "openapi-examples", "--seed", "1",
    ])
    # rc is 1 because no files were written for the requested format
    assert rc == 1
    assert not (out_dir / "openapi-examples").exists() or not list((out_dir / "openapi-examples").iterdir())


def test_mock_render_openapi_examples_writes_enriched_spec(tmp_path: Path):
    cfg = _init_mocks(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render", "--config", str(cfg), "--output", str(out_dir),
        "--format", "openapi-examples",
        "--openapi-source", str(EXAMPLES / "simple_books.yaml"),
        "--seed", "1",
    ])
    assert rc == 0
    files = list((out_dir / "openapi-examples").iterdir())
    assert files
    enriched = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert "example" in enriched["components"]["schemas"]["Book"]


def test_mock_render_combined_formats(tmp_path: Path):
    cfg = _init_mocks(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render", "--config", str(cfg), "--output", str(out_dir),
        "--format", "wiremock,json,pact,postman",
        "--seed", "1", "--examples", "1",
    ])
    assert rc == 0
    for sub in ("wiremock", "json", "pact", "postman"):
        assert (out_dir / sub).exists(), f"missing {sub}/ output"


# ---------------------------------------------------------------------------
# mock-init — source-type detection
# ---------------------------------------------------------------------------


def _write_postman_collection(path: Path) -> None:
    path.write_text(json.dumps({
        "info": {"name": "X", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
        "item": [{
            "name": "Get",
            "request": {"method": "GET", "url": {"path": ["x"]}},
            "response": [{"code": 200, "body": "{}"}],
        }],
    }), encoding="utf-8")


def _write_har_capture(path: Path) -> None:
    path.write_text(json.dumps({
        "log": {
            "version": "1.2",
            "creator": {"name": "test"},
            "entries": [{
                "request": {"method": "GET", "url": "https://api.example.com/x", "headers": [], "queryString": []},
                "response": {"status": 200, "headers": [], "content": {"mimeType": "application/json", "text": "{}"}},
            }],
        }
    }), encoding="utf-8")


def test_mock_init_auto_detects_postman(tmp_path: Path, capsys):
    src = tmp_path / "collection.json"
    _write_postman_collection(src)
    rc = main(["mock-init", "--from", str(src), "--output", str(tmp_path / "m.yaml")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Source type: postman" in out


def test_mock_init_auto_detects_har(tmp_path: Path, capsys):
    src = tmp_path / "capture.har"
    _write_har_capture(src)
    rc = main(["mock-init", "--from", str(src), "--output", str(tmp_path / "m.yaml")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Source type: har" in out


def test_mock_init_explicit_source_type_overrides_detection(tmp_path: Path, capsys):
    """Force-treat a Postman file as a HAR file → fail clean."""
    src = tmp_path / "collection.json"
    _write_postman_collection(src)
    rc = main([
        "mock-init", "--from", str(src), "--output", str(tmp_path / "m.yaml"),
        "--source-type", "har",
    ])
    assert rc == 1


def test_mock_init_openapi_unchanged(tmp_path: Path, capsys):
    rc = main([
        "mock-init", "--from", str(EXAMPLES / "simple_books.yaml"),
        "--output", str(tmp_path / "m.yaml"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Source type: openapi" in out


# ---------------------------------------------------------------------------
# mock-enrich — argparse + dispatch (LLM call mocked)
# ---------------------------------------------------------------------------


def test_mock_enrich_runs_dispatch_with_mocked_llm(tmp_path: Path, monkeypatch):
    """Patch llm_chat to return a canned suggestion; assert the enriched YAML lands."""
    cfg = _init_mocks(tmp_path)

    suggestion = json.dumps([
        {"name": "Book", "example": {"id": 1, "title": "x", "author": "y"}},
        {"name": "NotFoundError", "example": {"code": "NOT_FOUND", "message": "x"}},
    ])
    error_suggestion = json.dumps([
        {"status": 500, "body": {"code": "INTERNAL", "message": "Server error"}},
    ])

    from mocks import llm_enricher
    responses = [suggestion, error_suggestion, error_suggestion]

    def fake_chat(**kwargs):
        return responses.pop(0) if responses else ""

    monkeypatch.setattr(llm_enricher, "llm_chat", fake_chat)

    out_path = tmp_path / "enriched.yaml"
    rc = main(["mock-enrich", "--config", str(cfg), "--output", str(out_path)])
    assert rc == 0
    enriched = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert enriched["schemas"]["Book"]["example"]["title"] == "x"


def test_mock_enrich_no_fill_examples_skips_first_pass(tmp_path: Path, monkeypatch):
    cfg = _init_mocks(tmp_path)

    from mocks import llm_enricher
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return json.dumps([])  # empty suggestions

    monkeypatch.setattr(llm_enricher, "llm_chat", fake_chat)

    rc = main([
        "mock-enrich", "--config", str(cfg),
        "--output", str(tmp_path / "out.yaml"),
        "--no-fill-examples",
    ])
    assert rc == 0
    # When fill-examples is skipped, the only LLM calls should be from
    # the error-drafting pass — and there are 2 endpoints in simple_books
    # so we expect at most 2 calls.
    assert len(calls) <= 2


def test_mock_enrich_no_draft_errors_skips_second_pass(tmp_path: Path, monkeypatch):
    cfg = _init_mocks(tmp_path)

    from mocks import llm_enricher
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return json.dumps([])

    monkeypatch.setattr(llm_enricher, "llm_chat", fake_chat)

    rc = main([
        "mock-enrich", "--config", str(cfg),
        "--output", str(tmp_path / "out.yaml"),
        "--no-draft-errors",
    ])
    assert rc == 0
    # Without draft-errors, LLM calls only from fill-examples (one batch)
    assert len(calls) == 1
