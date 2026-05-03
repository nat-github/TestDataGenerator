"""Tests for the WireMock + JSON fixture renderers."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mocks.openapi_importer import import_openapi
from mocks.renderers.json_fixture import render_json_fixtures
from mocks.renderers.wiremock import render_wiremock

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# WireMock renderer
# ---------------------------------------------------------------------------


def _read_mappings(mappings_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(mappings_dir.iterdir())]


def test_wiremock_creates_mappings_and_files_dirs(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    written = render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=2)
    assert (tmp_path / "mappings").is_dir()
    assert (tmp_path / "__files").is_dir()
    assert written
    for p in written:
        assert p.name.endswith(".json")


def test_wiremock_concrete_path_substitutes_template_params(tmp_path: Path):
    """In concrete mode, /books/{id} should become /books/<concrete-int>."""
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=2, match_mode="concrete")
    mappings = _read_mappings(tmp_path / "mappings")
    paths = [m["request"]["urlPath"] for m in mappings if "urlPath" in m["request"]]
    detail_paths = [p for p in paths if p.startswith("/books/") and p != "/books"]
    assert detail_paths
    for p in detail_paths:
        # /books/<digits-or-anything-but-not-a-template>
        assert "{" not in p
        # detail path should have a non-empty id segment
        assert re.match(r"^/books/[^/]+$", p), p


def test_wiremock_any_mode_uses_url_path_pattern(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=5, match_mode="any")
    mappings = _read_mappings(tmp_path / "mappings")
    detail_mappings = [m for m in mappings if "urlPathPattern" in m["request"]]
    assert detail_mappings
    for m in detail_mappings:
        # Pattern should contain a regex chunk where {id} was
        assert "{" not in m["request"]["urlPathPattern"]
        assert "[" in m["request"]["urlPathPattern"] or "+" in m["request"]["urlPathPattern"]


def test_wiremock_any_mode_collapses_examples(tmp_path: Path):
    """In any-mode, only one mapping per (endpoint, status) should exist."""
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=5, match_mode="any")
    mappings = _read_mappings(tmp_path / "mappings")
    # We have 2 endpoints; list_books has 200, get_book_by_id has 200+404 = 3 mappings
    assert len(mappings) == 3


def test_wiremock_method_and_status_propagate(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    mappings = _read_mappings(tmp_path / "mappings")
    methods = {m["request"]["method"] for m in mappings}
    statuses = {m["response"]["status"] for m in mappings}
    assert methods == {"GET", "POST", "PUT", "DELETE"}
    assert statuses & {200, 201, 401, 404, 400}


def test_wiremock_response_includes_jsonBody_when_schema_present(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    mappings = _read_mappings(tmp_path / "mappings")
    for m in mappings:
        if 200 <= m["response"]["status"] < 300:
            assert "jsonBody" in m["response"]


def test_wiremock_metadata_block_present(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    mappings = _read_mappings(tmp_path / "mappings")
    for m in mappings:
        assert "metadata" in m
        assert "sdp" in m["metadata"]
        assert "endpoint" in m["metadata"]["sdp"]


def test_wiremock_complex_payment_idempotency_header_matched(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    mappings = _read_mappings(tmp_path / "mappings")
    create_payment = [m for m in mappings if m["metadata"]["sdp"]["endpoint"] == "create_payment"]
    assert create_payment
    for m in create_payment:
        if m["request"]["method"] == "POST":
            assert "headers" in m["request"]
            assert "Idempotency-Key" in m["request"]["headers"]


def test_wiremock_seed_is_deterministic(tmp_path: Path):
    """Same seed → same files. Different seed → different files."""
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    render_wiremock(cfg, a, seed=42, examples_per_endpoint=2)
    render_wiremock(cfg, b, seed=42, examples_per_endpoint=2)
    render_wiremock(cfg, c, seed=43, examples_per_endpoint=2)

    for fname in [p.name for p in (a / "mappings").iterdir()]:
        assert (a / "mappings" / fname).read_bytes() == (b / "mappings" / fname).read_bytes()

    # At least one file must differ between seed=42 and seed=43
    differences = 0
    for fname in [p.name for p in (a / "mappings").iterdir()]:
        if (a / "mappings" / fname).read_bytes() != (c / "mappings" / fname).read_bytes():
            differences += 1
    assert differences > 0


# ---------------------------------------------------------------------------
# JSON fixture renderer
# ---------------------------------------------------------------------------


def test_json_fixtures_simple_books_writes_response_per_status(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    written = render_json_fixtures(cfg, tmp_path, seed=1)
    names = {p.name for p in written}
    assert "list_books__200.json" in names
    assert "get_book_by_id__200.json" in names
    assert "get_book_by_id__404.json" in names


def test_json_fixtures_writes_request_bodies(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    render_json_fixtures(cfg, tmp_path, seed=1)
    requests_dir = tmp_path / "requests"
    assert requests_dir.is_dir()
    files = {p.name for p in requests_dir.iterdir()}
    assert "create_task.json" in files
    assert "replace_task.json" in files


def test_json_fixtures_writes_schema_canonical_examples(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    render_json_fixtures(cfg, tmp_path, seed=1)
    schemas_dir = tmp_path / "schemas"
    assert schemas_dir.is_dir()
    files = {p.name for p in schemas_dir.iterdir()}
    for required in ["Account.json", "Money.json", "Transaction.json"]:
        assert required in files


def test_json_fixtures_payload_is_valid_json(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    render_json_fixtures(cfg, tmp_path, seed=1)
    for path in tmp_path.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))  # raises on malformed JSON


def test_json_fixtures_seed_is_deterministic(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    a = tmp_path / "a"
    b = tmp_path / "b"
    render_json_fixtures(cfg, a, seed=99)
    render_json_fixtures(cfg, b, seed=99)
    for fname in [p.name for p in a.iterdir() if p.is_file()]:
        assert (a / fname).read_bytes() == (b / fname).read_bytes()
