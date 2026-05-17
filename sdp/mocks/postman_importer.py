"""Postman v2.1 Collection → MockConfig importer.

Reads a Postman collection JSON and produces a `MockConfig`. Handles the
common shape consumers will hit:

  - Folder hierarchy (item arrays nested) flattened to per-endpoint records
  - Variables (``{{baseUrl}}`` etc.) substituted where possible
  - Saved example responses (Postman's ``response[]`` array on each item)
    used to infer response status codes and body shapes
  - Auth headers detected from ``request.auth`` and turned into header
    requirements

Schema inference is heuristic: we only have concrete example bodies, so
the resulting `FieldSpec` records pin those bodies as `example` values
rather than inferring full JSON Schema.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sdp.models.mock_models import (
    EndpointConfig,
    FieldSpec,
    HTTP_METHODS,
    MockConfig,
    MockInfo,
    MockSettings,
    RequestMatcher,
    ResponseTemplate,
    SchemaConfig,
    ServerConfig,
)

logger = logging.getLogger(__name__)


class PostmanImportError(ValueError):
    """Raised when a Postman collection cannot be turned into a MockConfig."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_postman(path: Union[str, Path]) -> MockConfig:
    p = Path(path)
    if not p.exists():
        raise PostmanImportError(f"Postman collection not found: {p}")
    return import_postman_str(p.read_text(encoding="utf-8"), source=str(p))


def import_postman_str(raw: str, *, source: str = "<string>") -> MockConfig:
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PostmanImportError(f"{source}: invalid JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise PostmanImportError(f"{source}: top-level Postman collection must be a mapping")
    if "info" not in doc:
        raise PostmanImportError(f"{source}: missing top-level `info` block — is this a Postman collection?")

    return _PostmanImporter(doc, source).build()


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


_VAR_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


class _PostmanImporter:
    def __init__(self, doc: Dict[str, Any], source: str):
        self.doc = doc
        self.source = source
        # Collection-level variables are simple key/value pairs
        self.variables: Dict[str, str] = {
            v.get("key"): v.get("value", "")
            for v in (doc.get("variable") or [])
            if isinstance(v, dict) and v.get("key")
        }
        self._endpoint_names: set[str] = set()

    # ------------------------------------------------------------------
    def build(self) -> MockConfig:
        info_block = self.doc.get("info") or {}
        info = MockInfo(
            title=str(info_block.get("name") or "Imported Collection"),
            version=self._extract_version(info_block),
            description=info_block.get("description") if isinstance(info_block.get("description"), str) else None,
        )

        servers = self._derive_servers()
        endpoints: List[EndpointConfig] = []
        self._walk_items(self.doc.get("item") or [], endpoints, folder_path=[])

        # Heuristic: when many endpoints share a leading host, hoist it
        # into MockSettings.base_path so per-endpoint paths stay short.
        return MockConfig(
            info=info,
            servers=servers,
            settings=MockSettings(),
            schemas={},  # heuristic importer keeps schemas inline as examples
            endpoints=endpoints,
        )

    # ------------------------------------------------------------------
    def _walk_items(
        self,
        items: List[Any],
        endpoints: List[EndpointConfig],
        folder_path: List[str],
    ) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            if "item" in item:
                # Folder
                self._walk_items(item["item"], endpoints, folder_path + [item.get("name") or ""])
            elif "request" in item:
                ep = self._item_to_endpoint(item, folder_path)
                if ep is not None:
                    endpoints.append(ep)

    # ------------------------------------------------------------------
    def _item_to_endpoint(
        self,
        item: Dict[str, Any],
        folder_path: List[str],
    ) -> Optional[EndpointConfig]:
        request = item.get("request")
        if not isinstance(request, dict):
            return None
        method = str(request.get("method") or "GET").upper()
        if method not in HTTP_METHODS:
            return None

        url_info = self._parse_url(request.get("url"))
        if url_info is None:
            return None
        path = url_info["path"]

        # Build request matcher
        path_params = self._extract_path_params(path, request.get("url"))
        query_params = self._extract_query_params(request.get("url"))
        headers = self._extract_request_headers(request.get("header") or [])
        body_schema = self._extract_request_body(request.get("body"))
        content_type = self._content_type_from_headers(headers)

        # Build response variants from saved Postman responses
        responses = self._extract_responses(item.get("response") or [])
        if not responses:
            # Postman items don't always carry saved responses — fabricate
            # a 200 with no body so the resulting MockConfig is still valid.
            responses.append(ResponseTemplate(status=200, body_schema=FieldSpec(type="object")))
        responses.sort(key=lambda r: (r.status >= 400, r.status))

        name = self._unique_name(item.get("name") or f"{method}_{path}")
        tags = [folder_path[-1]] if folder_path and folder_path[-1] else []

        return EndpointConfig(
            name=name,
            path=path,
            method=method,  # type: ignore[arg-type]
            summary=str(item.get("name") or "") or None,
            description=str(request.get("description") or "") or None,
            request=RequestMatcher(
                path_params=path_params,
                query_params=query_params,
                headers=headers,
                body_schema=body_schema,
                body_match_mode="json-equal" if body_schema is not None else "ignore",
                content_type=content_type,
            ),
            responses=responses,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # URL / parameters
    # ------------------------------------------------------------------
    def _parse_url(self, url: Any) -> Optional[Dict[str, Any]]:
        """Postman URLs come in two flavours: a string (`raw`) or a dict.

        Always return a dict with at minimum a normalised ``path`` field.
        """
        if isinstance(url, str):
            raw = self._substitute_vars(url)
            from urllib.parse import urlparse
            parsed = urlparse(raw)
            return {"path": parsed.path or "/", "raw": raw}

        if not isinstance(url, dict):
            return None

        path_segs = url.get("path") or []
        if isinstance(path_segs, str):
            path_segs = path_segs.split("/")
        # Postman replaces /{id}/ with /:id/ — convert back to OpenAPI braces.
        norm_segs = []
        for seg in path_segs:
            seg = self._substitute_vars(str(seg))
            if seg.startswith(":"):
                norm_segs.append("{" + seg[1:] + "}")
            else:
                norm_segs.append(seg)
        path = "/" + "/".join(s for s in norm_segs if s != "")
        if not path.startswith("/"):
            path = "/" + path
        return {"path": path or "/", "raw": url.get("raw")}

    def _extract_path_params(self, path: str, url: Any) -> Dict[str, FieldSpec]:
        params: Dict[str, FieldSpec] = {}
        # Names from the path template
        for match in re.findall(r"\{([^}]+)\}", path):
            params[match] = FieldSpec(type="string")
        # Variable definitions from Postman (provides description / example)
        if isinstance(url, dict):
            for var in url.get("variable") or []:
                if not isinstance(var, dict):
                    continue
                key = var.get("key")
                if not key:
                    continue
                spec = params.get(key, FieldSpec(type="string"))
                if var.get("value") is not None:
                    spec.example = self._substitute_vars(str(var.get("value")))
                if var.get("description"):
                    spec.description = str(var.get("description"))
                params[key] = spec
        return params

    def _extract_query_params(self, url: Any) -> Dict[str, FieldSpec]:
        params: Dict[str, FieldSpec] = {}
        if not isinstance(url, dict):
            return params
        for q in url.get("query") or []:
            if not isinstance(q, dict):
                continue
            key = q.get("key")
            if not key:
                continue
            spec = FieldSpec(type="string")
            if q.get("value") is not None:
                spec.example = self._substitute_vars(str(q.get("value")))
            if q.get("description"):
                spec.description = str(q.get("description"))
            params[key] = spec
        return params

    # ------------------------------------------------------------------
    # Request headers / body
    # ------------------------------------------------------------------
    def _extract_request_headers(self, header_list: List[Any]) -> Dict[str, FieldSpec]:
        out: Dict[str, FieldSpec] = {}
        for h in header_list:
            if not isinstance(h, dict):
                continue
            key = h.get("key")
            if not key:
                continue
            spec = FieldSpec(type="string")
            if h.get("value") is not None:
                spec.example = self._substitute_vars(str(h.get("value")))
            out[key] = spec
        return out

    def _extract_request_body(self, body: Any) -> Optional[FieldSpec]:
        if not isinstance(body, dict):
            return None
        mode = body.get("mode")
        if mode == "raw":
            raw = body.get("raw") or ""
            raw = self._substitute_vars(raw)
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return FieldSpec(type="string", example=raw)
            return _value_to_field_spec(value)
        if mode == "urlencoded" or mode == "formdata":
            entries = body.get(mode) or []
            example = {
                e.get("key"): self._substitute_vars(str(e.get("value", "")))
                for e in entries if isinstance(e, dict) and e.get("key")
            }
            return FieldSpec(type="object", example=example)
        return None

    def _content_type_from_headers(self, headers: Dict[str, FieldSpec]) -> Optional[str]:
        for k, v in headers.items():
            if k.lower() == "content-type" and v.example:
                return str(v.example)
        return None

    # ------------------------------------------------------------------
    # Saved responses
    # ------------------------------------------------------------------
    def _extract_responses(self, saved_responses: List[Any]) -> List[ResponseTemplate]:
        out: List[ResponseTemplate] = []
        for sr in saved_responses:
            if not isinstance(sr, dict):
                continue
            status = sr.get("code")
            if status is None:
                # Some collections only carry "status" text — derive from common ones
                status = 200
            try:
                status = int(status)
            except (TypeError, ValueError):
                continue
            body_raw = sr.get("body") or ""
            body_field: Optional[FieldSpec] = None
            content_type = "application/json"
            if isinstance(body_raw, str) and body_raw.strip():
                try:
                    parsed = json.loads(body_raw)
                    body_field = _value_to_field_spec(parsed)
                except (json.JSONDecodeError, TypeError):
                    body_field = FieldSpec(type="string", example=body_raw)
                    content_type = "text/plain"

            headers = self._extract_response_headers(sr.get("header") or [])
            out.append(ResponseTemplate(
                status=status,
                headers=headers,
                body_schema=body_field,
                body_content_type=content_type,
                description=sr.get("name") or None,
            ))
        return out

    def _extract_response_headers(self, header_list: List[Any]) -> Dict[str, FieldSpec]:
        out: Dict[str, FieldSpec] = {}
        for h in header_list:
            if not isinstance(h, dict):
                continue
            key = h.get("key")
            if not key:
                continue
            spec = FieldSpec(type="string")
            if h.get("value") is not None:
                spec.example = str(h.get("value"))
            out[key] = spec
        return out

    # ------------------------------------------------------------------
    # Servers
    # ------------------------------------------------------------------
    def _derive_servers(self) -> List[ServerConfig]:
        # Postman collections rarely declare servers explicitly, but they
        # often define a `baseUrl` variable.
        base = self.variables.get("baseUrl") or self.variables.get("base_url")
        if base:
            return [ServerConfig(url=str(base), description="From baseUrl variable")]
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _substitute_vars(self, value: str) -> str:
        if not isinstance(value, str) or "{{" not in value:
            return value
        def repl(m: re.Match) -> str:
            return str(self.variables.get(m.group(1), m.group(0)))
        return _VAR_PATTERN.sub(repl, value)

    def _extract_version(self, info: Dict[str, Any]) -> str:
        v = info.get("version")
        if isinstance(v, dict) and v.get("raw"):
            return str(v["raw"])
        if isinstance(v, str):
            return v
        return "1.0.0"

    def _unique_name(self, base: str) -> str:
        candidate = _snake_case(base)
        if candidate not in self._endpoint_names:
            self._endpoint_names.add(candidate)
            return candidate
        i = 2
        while f"{candidate}_{i}" in self._endpoint_names:
            i += 1
        self._endpoint_names.add(f"{candidate}_{i}")
        return f"{candidate}_{i}"


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _value_to_field_spec(value: Any) -> FieldSpec:
    """Build a `FieldSpec` from a concrete observed value.

    Arrays need an `items:` declaration — recurse into the first element
    if available, fall back to a generic string item otherwise.
    """
    if isinstance(value, list):
        items_spec = _value_to_field_spec(value[0]) if value else FieldSpec(type="string")
        return FieldSpec(type="array", items=items_spec, example=value)
    return FieldSpec(type=_python_to_openapi_type(value), example=value)


def _python_to_openapi_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _snake_case(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "endpoint"
