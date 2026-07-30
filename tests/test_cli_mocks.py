"""Tests for the mock-* CLI subcommands.

Exercises argparse plumbing + run_mock_init / run_mock_render / run_mock_lint
against the example OpenAPI specs. No subprocess — calls main([...]) directly.
"""
from __future__ import annotations

import json
from pathlib import Path


from main import main, parse_arguments

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------


def test_argparse_mock_init_requires_from_and_output():
    args = parse_arguments([
        "mock-init",
        "--from", "spec.yaml",
        "--output", "out.yaml",
    ])
    assert args.command == "mock-init"
    assert args.source == "spec.yaml"
    assert args.output == "out.yaml"


def test_argparse_mock_render_defaults_to_wiremock():
    args = parse_arguments([
        "mock-render",
        "--config", "mocks.yaml",
        "--output", "stubs/",
    ])
    assert args.format == "wiremock"
    assert args.match_mode == "concrete"
    assert args.examples is None
    assert args.seed is None


def test_argparse_mock_render_accepts_multiple_formats():
    args = parse_arguments([
        "mock-render",
        "--config", "mocks.yaml",
        "--output", "stubs/",
        "--format", "wiremock,json",
        "--examples", "5",
        "--seed", "42",
        "--match-mode", "any",
    ])
    assert args.format == "wiremock,json"
    assert args.examples == 5
    assert args.seed == 42
    assert args.match_mode == "any"


def test_argparse_mock_lint_requires_config():
    args = parse_arguments(["mock-lint", "--config", "mocks.yaml"])
    assert args.command == "mock-lint"


# ---------------------------------------------------------------------------
# mock-init
# ---------------------------------------------------------------------------


def test_mock_init_simple_books_writes_yaml(tmp_path: Path, capsys):
    out = tmp_path / "mocks.yaml"
    rc = main([
        "mock-init",
        "--from", str(EXAMPLES / "simple_books.yaml"),
        "--output", str(out),
    ])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "config_format: sdp-mock-v1" in text
    assert "list_books" in text
    assert "get_book_by_id" in text
    captured = capsys.readouterr().out
    # Output is column-aligned, so just assert that the count-of-2 message landed
    assert "Endpoints:" in captured and " 2\n" in captured


def test_mock_init_complex_payments(tmp_path: Path):
    out = tmp_path / "mocks.yaml"
    rc = main([
        "mock-init",
        "--from", str(EXAMPLES / "complex_payments.yaml"),
        "--output", str(out),
    ])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    for required in ["list_accounts", "create_payment", "get_customer", "Account", "Money"]:
        assert required in text


def test_mock_init_returns_nonzero_on_bad_input(tmp_path: Path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("title: not an openapi doc", encoding="utf-8")
    rc = main([
        "mock-init",
        "--from", str(bad),
        "--output", str(tmp_path / "out.yaml"),
    ])
    assert rc == 1


# ---------------------------------------------------------------------------
# mock-render
# ---------------------------------------------------------------------------


def _init_mocks_from_simple(tmp_path: Path) -> Path:
    """Round-trip simple_books → sdp-mock-v1 to give us a config to render."""
    out = tmp_path / "mocks.yaml"
    main([
        "mock-init",
        "--from", str(EXAMPLES / "simple_books.yaml"),
        "--output", str(out),
    ])
    return out


def test_mock_render_writes_wiremock_mappings(tmp_path: Path):
    cfg = _init_mocks_from_simple(tmp_path)
    out_dir = tmp_path / "stubs"
    rc = main([
        "mock-render",
        "--config", str(cfg),
        "--output", str(out_dir),
        "--format", "wiremock",
        "--examples", "2",
        "--seed", "7",
    ])
    assert rc == 0
    mappings = list((out_dir / "wiremock" / "mappings").iterdir())
    assert mappings
    # Each is valid JSON with the WireMock shape
    for p in mappings:
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert "request" in doc and "response" in doc


def test_mock_render_writes_json_fixtures(tmp_path: Path):
    cfg = _init_mocks_from_simple(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render",
        "--config", str(cfg),
        "--output", str(out_dir),
        "--format", "json",
        "--seed", "7",
    ])
    assert rc == 0
    files = list((out_dir / "json").iterdir())
    assert any(p.name.startswith("list_books") for p in files if p.is_file())


def test_mock_render_dual_format_writes_both(tmp_path: Path):
    cfg = _init_mocks_from_simple(tmp_path)
    out_dir = tmp_path / "out"
    rc = main([
        "mock-render",
        "--config", str(cfg),
        "--output", str(out_dir),
        "--format", "wiremock,json",
        "--examples", "1",
        "--seed", "1",
    ])
    assert rc == 0
    assert (out_dir / "wiremock" / "mappings").exists()
    assert (out_dir / "json").exists()


def test_mock_render_seed_is_deterministic(tmp_path: Path):
    cfg = _init_mocks_from_simple(tmp_path)
    a = tmp_path / "a"
    b = tmp_path / "b"
    main(["mock-render", "--config", str(cfg), "--output", str(a),
          "--format", "wiremock", "--examples", "2", "--seed", "99"])
    main(["mock-render", "--config", str(cfg), "--output", str(b),
          "--format", "wiremock", "--examples", "2", "--seed", "99"])
    files_a = sorted(p.name for p in (a / "wiremock" / "mappings").iterdir())
    files_b = sorted(p.name for p in (b / "wiremock" / "mappings").iterdir())
    assert files_a == files_b
    for fname in files_a:
        assert (a / "wiremock" / "mappings" / fname).read_bytes() \
            == (b / "wiremock" / "mappings" / fname).read_bytes()


# ---------------------------------------------------------------------------
# mock-lint
# ---------------------------------------------------------------------------


def test_mock_lint_passes_for_valid_config(tmp_path: Path, capsys):
    cfg = _init_mocks_from_simple(tmp_path)
    rc = main(["mock-lint", "--config", str(cfg)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "OK" in captured


def test_mock_lint_fails_for_broken_config(tmp_path: Path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("config_format: not-the-right-version", encoding="utf-8")
    rc = main(["mock-lint", "--config", str(bad)])
    assert rc == 1
    captured = capsys.readouterr().out
    assert "FAIL" in captured
    assert "ERROR" in captured
