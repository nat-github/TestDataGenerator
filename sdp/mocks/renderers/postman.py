"""Postman v2.1 Collection renderer — `MockConfig` → importable Postman JSON.

Output: ``<title>.postman_collection.json``. Folder structure follows
the OpenAPI tags found on each endpoint (one folder per tag, ungrouped
endpoints land at the collection root). Each request gets a saved
example response per response variant so users can inspect realistic
shapes without firing the request.

Postman Collection v2.1 schema:
    https://schema.postman.com/json/collection/v2.1.0/collection.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sdp.mocks.template_engine import TemplateEngine
from sdp.models.mock_models import EndpointConfig, MockConfig, ResponseTemplate

logger = logging.getLogger(__name__)


SCHEMA_URL = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"


def render_postman(
    config: MockConfig,
    output_dir: Path,
    *,
    seed: Optional[int] = None,
    examples_per_endpoint: Optional[int] = None,
    base_url_var: str = "baseUrl",
) -> List[Path]:
    """Write a single Postman v2.1 collection JSON.

    Parameters
    ----------
    base_url_var:
        Name of the Postman variable used for the server base URL.
        Defaults to ``baseUrl``. Becomes ``{{baseUrl}}`` in URLs and a
        collection-level variable consumers can override per environment.
    examples_per_endpoint:
        How many saved example responses per endpoint. The first 2xx
        response always becomes the *primary* example; the rest are
        attached as `response[]` entries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = TemplateEngine(config, seed=seed)
    n_examples = examples_per_endpoint or config.settings.default_examples_count

    # Group endpoints by their first tag (or "default")
    folders: Dict[str, List[Dict[str, Any]]] = {}
    for endpoint in config.endpoints:
        item = _build_request_item(
            endpoint=endpoint,
            engine=engine,
            base_url_var=base_url_var,
            examples_count=n_examples,
            base_path=config.settings.base_path or "",
        )
        tag = (endpoint.tags or ["default"])[0]
        folders.setdefault(tag, []).append(item)

    # Build the collection items list
    items: List[Dict[str, Any]] = []
    for tag in sorted(folders):
        if tag == "default":
            items.extend(folders[tag])
        else:
            items.append({"name": tag, "item": folders[tag]})

    # Default base URL — first declared server, falling back to localhost
    default_base = (
        config.servers[0].url if config.servers else "http://localhost:8080"
    )

    collection: Dict[str, Any] = {
        "info": {
            "name": config.info.title,
            "description": config.info.description or "",
            "schema": SCHEMA_URL,
            "_postman_id": _stable_uuid(config.info.title),
        },
        "item": items,
        "variable": [
            {"key": base_url_var, "value": default_base, "type": "string"},
        ],
    }
    if config.info.version:
        collection["info"]["version"] = {
            "raw": config.info.version,
            "major": _safe_int(config.info.version.split(".")[0] if "." in config.info.version else "1"),
        }

    out_path = output_dir / f"{_safe(config.info.title)}.postman_collection.json"
    out_path.write_text(json.dumps(collection, indent=2, default=str), encoding="utf-8")

    logger.info("postman: wrote %d folder(s)/request(s) to %s", len(items), out_path)
    return [out_path]


# ---------------------------------------------------------------------------
# One request item
# ---------------------------------------------------------------------------


def _build_request_item(
    *,
    endpoint: EndpointConfig,
    engine: TemplateEngine,
    base_url_var: str,
    examples_count: int,
    base_path: str,
) -> Dict[str, Any]:
    full_path = base_path + endpoint.path
    raw_url, host_segs, path_segs, query = _split_url(
        path=full_path,
        endpoint=endpoint,
        engine=engine,
        base_url_var=base_url_var,
    )

    # Headers (collection-level — user-editable)
    headers = []
    if endpoint.request.headers:
        for name, field in endpoint.request.headers.items():
            headers.append({
                "key": name,
                "value": str(engine.render_field(field)),
                "type": "text",
            })
    if endpoint.request.body_schema is not None and not any(h["key"].lower() == "content-type" for h in headers):
        headers.append({
            "key": "Content-Type",
            "value": endpoint.request.content_type or "application/json",
            "type": "text",
        })

    # Body (only when one is declared)
    body: Optional[Dict[str, Any]] = None
    if endpoint.request.body_schema is not None:
        rendered = engine.render_field(endpoint.request.body_schema)
        body = {
            "mode": "raw",
            "raw": json.dumps(rendered, indent=2, default=str),
            "options": {"raw": {"language": "json"}},
        }

    request_block: Dict[str, Any] = {
        "method": endpoint.method,
        "header": headers,
        "url": {
            "raw": raw_url,
            "host": host_segs,
            "path": path_segs,
        },
        "description": endpoint.description or endpoint.summary,
    }
    if query:
        request_block["url"]["query"] = query
    if body:
        request_block["body"] = body

    # Saved example responses — one entry per response variant; for 2xx
    # responses we generate up to N concrete examples to give the user a
    # feel for value variation.
    saved_responses: List[Dict[str, Any]] = []
    for response in endpoint.responses:
        count = (
            response.examples_count
            or (examples_count if response.status < 400 else 1)
        )
        for example_index in range(count):
            saved_responses.append(_build_saved_response(
                endpoint=endpoint,
                response=response,
                example_index=example_index,
                engine=engine,
                request_block=request_block,
            ))

    item: Dict[str, Any] = {
        "name": endpoint.name,
        "request": request_block,
        "response": saved_responses,
    }
    return item


def _split_url(
    *,
    path: str,
    endpoint: EndpointConfig,
    engine: TemplateEngine,
    base_url_var: str,
) -> tuple[str, List[str], List[str], List[Dict[str, str]]]:
    """Postman wants URLs split into {raw, host[], path[], query[]}."""
    rendered_path = path
    for name, field in endpoint.request.path_params.items():
        rendered_path = rendered_path.replace("{" + name + "}", f":{name}")
    # Convert leading "/" stripped + path-segment array
    path_segs = [seg for seg in rendered_path.split("/") if seg]

    query = []
    for name, field in endpoint.request.query_params.items():
        query.append({
            "key": name,
            "value": str(engine.render_field(field)),
        })

    raw_url = "{{" + base_url_var + "}}" + rendered_path
    if query:
        raw_url += "?" + "&".join(f"{q['key']}={q['value']}" for q in query)

    host_segs = ["{{" + base_url_var + "}}"]
    return raw_url, host_segs, path_segs, query


def _build_saved_response(
    *,
    endpoint: EndpointConfig,
    response: ResponseTemplate,
    example_index: int,
    engine: TemplateEngine,
    request_block: Dict[str, Any],
) -> Dict[str, Any]:
    name = f"{endpoint.name} {response.status}"
    if example_index > 0:
        name = f"{name} #{example_index + 1}"

    body = ""
    if response.body_template is not None:
        body = engine._render_template_string(response.body_template)
    elif response.body_schema is not None:
        body = json.dumps(engine.render_field(response.body_schema), indent=2, default=str)

    headers = []
    if "Content-Type" not in response.headers and (response.body_schema or response.body_template):
        headers.append({
            "key": "Content-Type",
            "value": response.body_content_type,
        })
    for hname, hfield in response.headers.items():
        headers.append({
            "key": hname,
            "value": str(engine.render_field(hfield)),
        })

    return {
        "name": name,
        "originalRequest": {
            "method": request_block["method"],
            "header": request_block.get("header", []),
            "url": request_block["url"],
        },
        "status": _status_text(response.status),
        "code": response.status,
        "_postman_previewlanguage": "json",
        "header": headers,
        "cookie": [],
        "body": body,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HTTP_STATUS_TEXT = {
    200: "OK", 201: "Created", 202: "Accepted", 204: "No Content",
    301: "Moved Permanently", 302: "Found", 304: "Not Modified",
    400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
    404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
    422: "Unprocessable Entity", 429: "Too Many Requests",
    500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable",
}


def _status_text(code: int) -> str:
    return _HTTP_STATUS_TEXT.get(code, "Unknown")


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "collection"


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return 1


def _stable_uuid(seed: str) -> str:
    """Generate a deterministic UUID-shaped string from a seed (so re-runs
    don't churn the `_postman_id` field)."""
    import hashlib
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
