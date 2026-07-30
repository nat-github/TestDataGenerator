"""Tests for Pact / Postman / OpenAPI-examples renderers (Phase E)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from sdp.mocks.openapi_importer import import_openapi
from sdp.mocks.renderers.openapi_examples import enrich_in_place, render_openapi_examples
from sdp.mocks.renderers.pact import render_pact
from sdp.mocks.renderers.postman import render_postman

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# Pact renderer
# ---------------------------------------------------------------------------


def test_pact_writes_single_contract_file(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    written = render_pact(
        cfg, tmp_path,
        consumer="reader-app",
        provider="books-service",
        seed=1, examples_per_endpoint=2,
    )
    assert len(written) == 1
    assert written[0].name == "reader-app-books-service.json"


def test_pact_contract_has_required_top_level_fields(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_pact(cfg, tmp_path, consumer="c", provider="p", seed=1)[0]
    pact = json.loads(out.read_text(encoding="utf-8"))
    assert pact["consumer"] == {"name": "c"}
    assert pact["provider"] == {"name": "p"}
    assert "interactions" in pact
    assert pact["metadata"]["pactSpecification"]["version"] == "3.0.0"
    assert pact["interactions"]


def test_pact_interaction_shape(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    out = render_pact(cfg, tmp_path, consumer="c", provider="p", seed=2,
                     examples_per_endpoint=1)[0]
    pact = json.loads(out.read_text(encoding="utf-8"))
    for interaction in pact["interactions"]:
        assert "description" in interaction
        assert "request" in interaction
        assert "response" in interaction
        assert "method" in interaction["request"]
        assert "path" in interaction["request"]
        assert "status" in interaction["response"]


def test_pact_can_skip_error_responses(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    out_with_errors = render_pact(
        cfg, tmp_path / "with",
        consumer="c", provider="p", seed=1,
        examples_per_endpoint=1, include_error_responses=True,
    )[0]
    out_no_errors = render_pact(
        cfg, tmp_path / "without",
        consumer="c", provider="p", seed=1,
        examples_per_endpoint=1, include_error_responses=False,
    )[0]
    a = json.loads(out_with_errors.read_text(encoding="utf-8"))
    b = json.loads(out_no_errors.read_text(encoding="utf-8"))
    a_statuses = {i["response"]["status"] for i in a["interactions"]}
    b_statuses = {i["response"]["status"] for i in b["interactions"]}
    assert any(s >= 400 for s in a_statuses)
    assert all(s < 400 for s in b_statuses)


def test_pact_path_params_rendered_to_concrete_values(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_pact(cfg, tmp_path, consumer="c", provider="p", seed=1,
                     examples_per_endpoint=1)[0]
    pact = json.loads(out.read_text(encoding="utf-8"))
    detail_paths = [
        i["request"]["path"] for i in pact["interactions"]
        if "/" in i["request"]["path"] and i["request"]["path"] != "/books"
    ]
    for path in detail_paths:
        # No template braces should remain
        assert "{" not in path
        assert "}" not in path


def test_pact_complex_payment_includes_idempotency_header(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    out = render_pact(cfg, tmp_path, consumer="c", provider="p", seed=1,
                     examples_per_endpoint=1)[0]
    pact = json.loads(out.read_text(encoding="utf-8"))
    create_payment = [
        i for i in pact["interactions"]
        if i["request"]["method"] == "POST" and i["request"]["path"].endswith("/payments")
    ]
    assert create_payment
    for i in create_payment:
        assert "headers" in i["request"]
        assert "Idempotency-Key" in i["request"]["headers"]


# ---------------------------------------------------------------------------
# Postman renderer
# ---------------------------------------------------------------------------


def test_postman_writes_single_collection(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    written = render_postman(cfg, tmp_path, seed=1, examples_per_endpoint=2)
    assert len(written) == 1
    assert written[0].name.endswith(".postman_collection.json")


def test_postman_collection_has_v2_1_schema(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_postman(cfg, tmp_path, seed=1)[0]
    coll = json.loads(out.read_text(encoding="utf-8"))
    assert "v2.1.0" in coll["info"]["schema"]
    assert coll["info"]["name"] == "Simple Books API"
    assert "item" in coll
    assert "variable" in coll
    base_url_var = next(v for v in coll["variable"] if v["key"] == "baseUrl")
    assert base_url_var["value"]


def test_postman_groups_endpoints_by_tag(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    out = render_postman(cfg, tmp_path, seed=1, examples_per_endpoint=1)[0]
    coll = json.loads(out.read_text(encoding="utf-8"))
    # medium_tasks has two tags: 'tasks' and 'collections'
    folder_names = {it["name"] for it in coll["item"] if "item" in it}
    assert "tasks" in folder_names or "collections" in folder_names


def test_postman_request_uses_baseUrl_variable(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_postman(cfg, tmp_path, seed=1)[0]
    coll = json.loads(out.read_text(encoding="utf-8"))

    def _walk_items(items):
        for it in items:
            if "request" in it:
                yield it
            elif "item" in it:
                yield from _walk_items(it["item"])

    items = list(_walk_items(coll["item"]))
    assert items
    for it in items:
        url = it["request"]["url"]
        assert "{{baseUrl}}" in url["raw"]


def test_postman_path_params_become_colon_segments(tmp_path: Path):
    """Postman convention: /books/{id} → /books/:id"""
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_postman(cfg, tmp_path, seed=1)[0]
    coll = json.loads(out.read_text(encoding="utf-8"))

    def _walk_items(items):
        for it in items:
            if "request" in it:
                yield it
            elif "item" in it:
                yield from _walk_items(it["item"])

    detail = next(
        it for it in _walk_items(coll["item"])
        if it["name"] == "get_book_by_id"
    )
    assert ":id" in "/".join(detail["request"]["url"]["path"])


def test_postman_saves_example_responses(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_postman(cfg, tmp_path, seed=1, examples_per_endpoint=3)[0]
    coll = json.loads(out.read_text(encoding="utf-8"))

    def _walk_items(items):
        for it in items:
            if "request" in it:
                yield it
            elif "item" in it:
                yield from _walk_items(it["item"])

    for it in _walk_items(coll["item"]):
        # Every endpoint should have at least one saved response
        assert it["response"]
        for r in it["response"]:
            assert "code" in r
            assert "status" in r
            assert "body" in r


def test_postman_post_includes_request_body(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    out = render_postman(cfg, tmp_path, seed=1, examples_per_endpoint=1)[0]
    coll = json.loads(out.read_text(encoding="utf-8"))

    def _walk_items(items):
        for it in items:
            if "request" in it:
                yield it
            elif "item" in it:
                yield from _walk_items(it["item"])

    create_task = next(it for it in _walk_items(coll["item"]) if it["name"] == "create_task")
    assert create_task["request"]["method"] == "POST"
    assert create_task["request"]["body"]["mode"] == "raw"
    body_raw = create_task["request"]["body"]["raw"]
    parsed = json.loads(body_raw)
    assert "title" in parsed


# ---------------------------------------------------------------------------
# OpenAPI examples enricher
# ---------------------------------------------------------------------------


def test_openapi_enricher_adds_example_to_named_schemas(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_openapi_examples(
        cfg, tmp_path,
        source_spec=EXAMPLES / "simple_books.yaml",
        seed=1,
    )[0]
    enriched = yaml.safe_load(out.read_text(encoding="utf-8"))
    schemas = enriched["components"]["schemas"]
    assert "example" in schemas["Book"]
    assert "id" in schemas["Book"]["example"]
    assert "title" in schemas["Book"]["example"]


def test_openapi_enricher_adds_example_to_response(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    out = render_openapi_examples(
        cfg, tmp_path,
        source_spec=EXAMPLES / "simple_books.yaml",
        seed=1,
    )[0]
    enriched = yaml.safe_load(out.read_text(encoding="utf-8"))
    get_by_id = enriched["paths"]["/books/{id}"]["get"]
    response_200 = get_by_id["responses"]["200"]
    media = response_200["content"]["application/json"]
    assert "example" in media


def test_openapi_enricher_preserves_existing_examples_by_default():
    """Authored examples in the source spec must not be overwritten."""
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    spec = yaml.safe_load((EXAMPLES / "simple_books.yaml").read_text(encoding="utf-8"))
    # The simple_books spec has an authored example for list_books 200
    original_example = (
        spec["paths"]["/books"]["get"]["responses"]["200"]["content"]
        ["application/json"]["example"]
    )
    enriched = enrich_in_place(spec=spec, config=cfg, seed=999)
    new_example = (
        enriched["paths"]["/books"]["get"]["responses"]["200"]["content"]
        ["application/json"]["example"]
    )
    assert new_example == original_example


def test_openapi_enricher_can_overwrite_when_asked():
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    spec = yaml.safe_load((EXAMPLES / "simple_books.yaml").read_text(encoding="utf-8"))
    original = spec["paths"]["/books"]["get"]["responses"]["200"]["content"]["application/json"]["example"]
    enriched = enrich_in_place(spec=spec, config=cfg, seed=999, overwrite_existing=True)
    new = enriched["paths"]["/books"]["get"]["responses"]["200"]["content"]["application/json"]["example"]
    assert new != original


def test_openapi_enricher_handles_complex_spec(tmp_path: Path):
    """The complex spec has $refs, allOf, oneOf, multiple servers."""
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    out = render_openapi_examples(
        cfg, tmp_path,
        source_spec=EXAMPLES / "complex_payments.yaml",
        seed=1,
    )[0]
    enriched = yaml.safe_load(out.read_text(encoding="utf-8"))
    # Money, Account, Payment should all have examples
    for schema_name in ["Money", "Account", "Payment"]:
        assert "example" in enriched["components"]["schemas"][schema_name], schema_name
    # Original servers and security blocks preserved
    assert enriched["servers"]
    assert "securitySchemes" in enriched["components"]


def test_openapi_enricher_round_trip_is_valid_yaml(tmp_path: Path):
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    out = render_openapi_examples(
        cfg, tmp_path,
        source_spec=EXAMPLES / "complex_payments.yaml",
        seed=1,
    )[0]
    # Must parse cleanly as YAML
    enriched = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert enriched["openapi"].startswith("3.")
    assert "paths" in enriched
    assert "components" in enriched
