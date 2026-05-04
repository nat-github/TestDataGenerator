"""Tests for the Postman and HAR reverse importers (Phase I)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mocks.har_importer import HARImportError, import_har_str
from mocks.postman_importer import PostmanImportError, import_postman_str


# ---------------------------------------------------------------------------
# Postman importer
# ---------------------------------------------------------------------------


def _postman_collection() -> Dict[str, Any]:
    return {
        "info": {
            "name": "Acme API",
            "version": "1.2.3",
            "description": "A sample collection",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [{"key": "baseUrl", "value": "http://localhost:8080/api"}],
        "item": [
            {
                "name": "Users",
                "item": [
                    {
                        "name": "Get user",
                        "request": {
                            "method": "GET",
                            "url": {
                                "raw": "{{baseUrl}}/users/:id",
                                "host": ["{{baseUrl}}"],
                                "path": ["users", ":id"],
                                "variable": [{"key": "id", "value": "42", "description": "User id"}],
                                "query": [{"key": "include", "value": "profile"}],
                            },
                            "header": [{"key": "Authorization", "value": "Bearer {{token}}"}],
                        },
                        "response": [
                            {
                                "name": "OK",
                                "code": 200,
                                "header": [{"key": "Content-Type", "value": "application/json"}],
                                "body": json.dumps({"id": 42, "name": "Ada"}),
                            },
                            {
                                "name": "Not found",
                                "code": 404,
                                "body": json.dumps({"code": "NOT_FOUND"}),
                            },
                        ],
                    },
                    {
                        "name": "Create user",
                        "request": {
                            "method": "POST",
                            "url": {"path": ["users"]},
                            "header": [{"key": "Content-Type", "value": "application/json"}],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps({"name": "Ada", "email": "ada@example.com"}),
                            },
                        },
                        "response": [
                            {"code": 201, "body": json.dumps({"id": 1, "name": "Ada"})},
                        ],
                    },
                ],
            },
        ],
    }


def test_postman_rejects_non_collection_doc():
    with pytest.raises(PostmanImportError, match="missing top-level"):
        import_postman_str(json.dumps({"foo": "bar"}))


def test_postman_rejects_invalid_json():
    with pytest.raises(PostmanImportError, match="invalid JSON"):
        import_postman_str("{not json")


def test_postman_imports_basic_collection():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    assert cfg.info.title == "Acme API"
    assert cfg.info.version == "1.2.3"
    assert len(cfg.endpoints) == 2
    names = [ep.name for ep in cfg.endpoints]
    assert "get_user" in names
    assert "create_user" in names


def test_postman_substitutes_baseUrl_in_servers():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    assert cfg.servers
    assert cfg.servers[0].url == "http://localhost:8080/api"


def test_postman_path_variables_become_template_braces():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    get_user = next(ep for ep in cfg.endpoints if ep.name == "get_user")
    assert "{id}" in get_user.path
    # The variable's `value` should land as the example
    assert get_user.request.path_params["id"].example == "42"
    assert get_user.request.path_params["id"].description == "User id"


def test_postman_query_params_imported():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    get_user = next(ep for ep in cfg.endpoints if ep.name == "get_user")
    assert "include" in get_user.request.query_params
    assert get_user.request.query_params["include"].example == "profile"


def test_postman_header_with_unresolved_var_kept_as_literal():
    """`Bearer {{token}}` — token isn't defined, so the literal stays."""
    cfg = import_postman_str(json.dumps(_postman_collection()))
    get_user = next(ep for ep in cfg.endpoints if ep.name == "get_user")
    auth = get_user.request.headers.get("Authorization")
    assert auth is not None
    assert "Bearer" in str(auth.example)


def test_postman_request_body_parsed_as_json():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    create = next(ep for ep in cfg.endpoints if ep.name == "create_user")
    assert create.request.body_schema is not None
    assert create.request.body_schema.example == {"name": "Ada", "email": "ada@example.com"}
    assert create.request.body_schema.type == "object"


def test_postman_saved_responses_become_response_templates():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    get_user = next(ep for ep in cfg.endpoints if ep.name == "get_user")
    statuses = {r.status for r in get_user.responses}
    assert statuses == {200, 404}
    # 2xx-first ordering
    assert get_user.responses[0].status == 200


def test_postman_folder_name_becomes_tag():
    cfg = import_postman_str(json.dumps(_postman_collection()))
    for ep in cfg.endpoints:
        assert ep.tags == ["Users"]


def test_postman_endpoint_names_are_unique_after_collision():
    coll = _postman_collection()
    # Add a second "Get user" item under a different folder
    coll["item"].append({
        "name": "Other",
        "item": [{
            "name": "Get user",
            "request": {"method": "GET", "url": {"path": ["other", "get"]}},
            "response": [{"code": 200, "body": ""}],
        }],
    })
    cfg = import_postman_str(json.dumps(coll))
    names = [ep.name for ep in cfg.endpoints]
    assert names.count("get_user") == 1  # collision avoided
    assert "get_user_2" in names


def test_postman_endpoint_with_no_responses_gets_synthetic_200():
    coll = {
        "info": {"name": "x", "version": "1"},
        "item": [{
            "name": "ping",
            "request": {"method": "GET", "url": {"path": ["ping"]}},
            "response": [],
        }],
    }
    cfg = import_postman_str(json.dumps(coll))
    assert cfg.endpoints[0].responses
    assert cfg.endpoints[0].responses[0].status == 200


# ---------------------------------------------------------------------------
# HAR importer
# ---------------------------------------------------------------------------


def _har(entries: List[Dict[str, Any]]) -> str:
    return json.dumps({
        "log": {
            "version": "1.2",
            "creator": {"name": "test", "version": "0.0.1"},
            "entries": entries,
        }
    })


def _entry(method: str, url: str, status: int = 200, *,
           query: List[Dict[str, str]] = None,
           response_body: Any = None,
           response_mime: str = "application/json",
           request_body: Any = None,
           request_mime: str = "application/json") -> Dict[str, Any]:
    return {
        "request": {
            "method": method,
            "url": url,
            "queryString": query or [],
            "headers": [{"name": "Content-Type", "value": request_mime}] if request_body else [],
            "postData": (
                {"mimeType": request_mime, "text": json.dumps(request_body)}
                if request_body is not None else {}
            ),
        },
        "response": {
            "status": status,
            "headers": [{"name": "Content-Type", "value": response_mime}],
            "content": (
                {"mimeType": response_mime, "text": json.dumps(response_body)}
                if response_body is not None else {}
            ),
        },
    }


def test_har_rejects_invalid_json():
    with pytest.raises(HARImportError, match="invalid JSON"):
        import_har_str("not json")


def test_har_rejects_doc_without_log_block():
    with pytest.raises(HARImportError, match="missing top-level `log`"):
        import_har_str(json.dumps({"foo": "bar"}))


def test_har_imports_single_entry():
    raw = _har([
        _entry("GET", "https://api.example.com/v1/users", status=200,
               response_body=[{"id": 1}, {"id": 2}]),
    ])
    cfg = import_har_str(raw)
    assert len(cfg.endpoints) == 1
    ep = cfg.endpoints[0]
    assert ep.method == "GET"
    assert ep.path == "/v1/users"


def test_har_groups_by_method_and_template_path():
    """Two entries with different ids should collapse to one endpoint with `{id}`."""
    raw = _har([
        _entry("GET", "https://api.example.com/users/1", status=200,
               response_body={"id": 1, "name": "Ada"}),
        _entry("GET", "https://api.example.com/users/2", status=200,
               response_body={"id": 2, "name": "Lin"}),
        _entry("GET", "https://api.example.com/users/9999", status=404,
               response_body={"code": "NOT_FOUND"}),
    ])
    cfg = import_har_str(raw)
    assert len(cfg.endpoints) == 1
    ep = cfg.endpoints[0]
    assert "{id}" in ep.path
    statuses = {r.status for r in ep.responses}
    assert statuses == {200, 404}


def test_har_uuid_segments_are_templated():
    raw = _har([
        _entry("GET", "https://api.example.com/orders/7c0a3b9a-0b09-4b30-bd75-a4dde9a5b7e6",
               status=200, response_body={"id": "..."}),
    ])
    cfg = import_har_str(raw)
    assert len(cfg.endpoints) == 1
    assert "{uuid}" in cfg.endpoints[0].path


def test_har_servers_derived_from_distinct_origins():
    raw = _har([
        _entry("GET", "https://api.a.com/x", status=200, response_body={}),
        _entry("GET", "https://api.b.com/x", status=200, response_body={}),
        _entry("GET", "https://api.a.com/y", status=200, response_body={}),
    ])
    cfg = import_har_str(raw)
    server_urls = {s.url for s in cfg.servers}
    assert server_urls == {"https://api.a.com", "https://api.b.com"}


def test_har_query_params_aggregated_across_entries():
    raw = _har([
        _entry("GET", "https://api.example.com/search",
               query=[{"name": "q", "value": "ada"}],
               status=200, response_body={"hits": []}),
        _entry("GET", "https://api.example.com/search",
               query=[{"name": "page", "value": "2"}],
               status=200, response_body={"hits": []}),
    ])
    cfg = import_har_str(raw)
    ep = cfg.endpoints[0]
    assert "q" in ep.request.query_params
    assert "page" in ep.request.query_params


def test_har_volatile_headers_are_filtered():
    """Date/Server/Connection/etc. should not pollute the response template."""
    raw = json.dumps({
        "log": {
            "creator": {"name": "test"},
            "entries": [{
                "request": {
                    "method": "GET",
                    "url": "https://api.example.com/x",
                    "headers": [
                        {"name": "Host", "value": "api.example.com"},
                        {"name": "X-Trace-Id", "value": "abc"},
                    ],
                    "queryString": [],
                },
                "response": {
                    "status": 200,
                    "headers": [
                        {"name": "Date", "value": "Wed, 1 Jan 2026 00:00:00 GMT"},
                        {"name": "Server", "value": "nginx"},
                        {"name": "X-RateLimit-Remaining", "value": "99"},
                    ],
                    "content": {"mimeType": "application/json", "text": "{}"},
                },
            }],
        }
    })
    cfg = import_har_str(raw)
    ep = cfg.endpoints[0]
    # Volatile headers stripped
    assert "Date" not in ep.responses[0].headers
    assert "Server" not in ep.responses[0].headers
    # Useful header retained
    assert "X-RateLimit-Remaining" in ep.responses[0].headers
    # Volatile request headers (Host) stripped
    assert "Host" not in ep.request.headers
    assert "X-Trace-Id" in ep.request.headers


def test_har_request_body_captured_for_post():
    raw = _har([
        _entry("POST", "https://api.example.com/users",
               status=201,
               request_body={"name": "Ada"},
               response_body={"id": 1, "name": "Ada"}),
    ])
    cfg = import_har_str(raw)
    ep = cfg.endpoints[0]
    assert ep.request.body_schema is not None
    assert ep.request.body_schema.example == {"name": "Ada"}


def test_har_endpoint_with_no_response_body_gets_minimal_200():
    raw = _har([
        _entry("DELETE", "https://api.example.com/users/1", status=204),
    ])
    cfg = import_har_str(raw)
    ep = cfg.endpoints[0]
    statuses = {r.status for r in ep.responses}
    assert 204 in statuses


def test_har_path_id_param_typed_as_integer_when_named_id():
    raw = _har([
        _entry("GET", "https://api.example.com/things/42", status=200, response_body={"id": 42}),
    ])
    cfg = import_har_str(raw)
    ep = cfg.endpoints[0]
    assert "{id}" in ep.path
    spec = ep.request.path_params["id"]
    assert spec.type == "integer"
    assert spec.example == "42"
