"""HAR (HTTP Archive) → MockConfig importer.

HAR files are produced by browsers (DevTools "Save All as HAR with content")
and HTTP proxies (mitmproxy, Charles). They capture real request/response
pairs — perfect raw material for replay-style mocks.

Strategy:
  - Group entries by (method, normalised path)
  - For each group, take the first response as the canonical 200 (and any
    additional distinct status codes as additional ResponseTemplates)
  - Path templating: paths with high-cardinality segments (UUIDs, ints)
    are replaced with `{param}` placeholders so e.g. `/users/123` and
    `/users/456` collapse to a single endpoint `/users/{id}`

Limitations:
  - We don't try to detect query parameter intent — every observed query
    key becomes a query param spec
  - Authentication tokens captured in headers are recorded as-is, which
    means re-rendering the mock will literally serve those tokens. Strip
    them from your HAR before importing if that matters.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from models.mock_models import (
    EndpointConfig,
    FieldSpec,
    HTTP_METHODS,
    MockConfig,
    MockInfo,
    MockSettings,
    RequestMatcher,
    ResponseTemplate,
    ServerConfig,
)

logger = logging.getLogger(__name__)


class HARImportError(ValueError):
    """Raised when a HAR file cannot be turned into a MockConfig."""


# Heuristic patterns: path segments that look like ids and should be
# templated rather than treated as literal.
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_INT_RE = re.compile(r"^\d+$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{12,}$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_har(path: Union[str, Path]) -> MockConfig:
    p = Path(path)
    if not p.exists():
        raise HARImportError(f"HAR file not found: {p}")
    return import_har_str(p.read_text(encoding="utf-8"), source=str(p))


def import_har_str(raw: str, *, source: str = "<string>") -> MockConfig:
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HARImportError(f"{source}: invalid JSON: {exc}") from exc

    if not isinstance(doc, dict) or "log" not in doc:
        raise HARImportError(f"{source}: missing top-level `log` block — is this a HAR file?")

    return _HARImporter(doc, source).build()


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class _HARImporter:
    def __init__(self, doc: Dict[str, Any], source: str):
        self.doc = doc
        self.source = source
        log = doc.get("log") or {}
        self.entries: List[Dict[str, Any]] = log.get("entries") or []
        self.creator: Dict[str, Any] = log.get("creator") or {}
        self._endpoint_names: set[str] = set()

    # ------------------------------------------------------------------
    def build(self) -> MockConfig:
        servers = self._derive_servers()

        # Step 1: collect (method, raw_path, query, headers, response) tuples
        observations: List[Tuple[str, str, Dict[str, Any]]] = []
        for entry in self.entries:
            obs = self._entry_to_observation(entry)
            if obs is not None:
                observations.append(obs)

        # Step 2: cluster by (method, templated_path)
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for method, raw_path, payload in observations:
            templated, params = _template_path(raw_path)
            groups[(method, templated)].append({**payload, "params": params})

        # Step 3: build one EndpointConfig per group
        endpoints: List[EndpointConfig] = []
        for (method, path), payloads in groups.items():
            ep = self._group_to_endpoint(method, path, payloads)
            if ep is not None:
                endpoints.append(ep)

        info = MockInfo(
            title=str(self.creator.get("name") or "Imported HAR"),
            version=str(self.creator.get("version") or "1.0.0"),
            description=f"Imported from {len(self.entries)} HAR entry/entries",
        )

        return MockConfig(
            info=info,
            servers=servers,
            settings=MockSettings(),
            schemas={},
            endpoints=endpoints,
        )

    # ------------------------------------------------------------------
    def _entry_to_observation(self, entry: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        if not isinstance(entry, dict):
            return None
        request = entry.get("request") or {}
        response = entry.get("response") or {}
        if not request or not response:
            return None

        method = str(request.get("method") or "").upper()
        if method not in HTTP_METHODS:
            return None

        url = str(request.get("url") or "")
        if not url:
            return None
        from urllib.parse import urlparse
        parsed = urlparse(url)
        raw_path = parsed.path or "/"

        return (method, raw_path, {
            "query": _kvlist_to_dict(request.get("queryString") or []),
            "request_headers": _kvlist_to_dict(request.get("headers") or []),
            "request_body": _post_data_to_value(request.get("postData") or {}),
            "request_content_type": _header_value(request.get("headers") or [], "content-type"),
            "status": int(response.get("status") or 200),
            "response_headers": _kvlist_to_dict(response.get("headers") or []),
            "response_body": _content_to_value(response.get("content") or {}),
            "response_content_type": (response.get("content") or {}).get("mimeType") or "application/json",
        })

    # ------------------------------------------------------------------
    def _group_to_endpoint(
        self,
        method: str,
        path: str,
        payloads: List[Dict[str, Any]],
    ) -> Optional[EndpointConfig]:
        first = payloads[0]
        path_param_names = re.findall(r"\{([^}]+)\}", path)
        path_params = {
            name: FieldSpec(type=("integer" if name.endswith("id") else "string"))
            for name in path_param_names
        }
        # Pick the most common param value as the example
        for name in path_param_names:
            samples = [p["params"].get(name) for p in payloads if p.get("params", {}).get(name)]
            if samples:
                path_params[name].example = samples[0]

        # Aggregate query params + request headers from union
        query_params: Dict[str, FieldSpec] = {}
        for p in payloads:
            for q_name, q_value in (p.get("query") or {}).items():
                if q_name not in query_params:
                    query_params[q_name] = FieldSpec(type="string", example=q_value)

        request_headers: Dict[str, FieldSpec] = {}
        for p in payloads:
            for h_name, h_value in (p.get("request_headers") or {}).items():
                if h_name.lower() in {"host", "user-agent", "accept", "accept-encoding",
                                      "accept-language", "connection", "cookie",
                                      "referer", "origin"}:
                    continue
                if h_name not in request_headers:
                    request_headers[h_name] = FieldSpec(type="string", example=h_value)

        # Body — first payload that had one
        body_field: Optional[FieldSpec] = None
        for p in payloads:
            if p.get("request_body") is not None:
                body_field = _value_to_field_spec(p["request_body"])
                break

        # Responses: one variant per distinct status code
        responses_by_status: Dict[int, ResponseTemplate] = {}
        for p in payloads:
            status = p["status"]
            if status in responses_by_status:
                continue
            body = p.get("response_body")
            body_spec = _value_to_field_spec(body) if body is not None else None
            responses_by_status[status] = ResponseTemplate(
                status=status,
                headers={
                    k: FieldSpec(type="string", example=v)
                    for k, v in (p.get("response_headers") or {}).items()
                    if k.lower() not in {"date", "server", "content-length", "transfer-encoding",
                                          "connection", "keep-alive"}
                },
                body_schema=body_spec,
                body_content_type=p.get("response_content_type") or "application/json",
            )

        responses = sorted(responses_by_status.values(),
                           key=lambda r: (r.status >= 400, r.status))
        if not responses:
            responses = [ResponseTemplate(status=200, body_schema=FieldSpec(type="object"))]

        name_root = path.strip("/").replace("/", "_") or "root"
        name = self._unique_name(f"{method.lower()}_{name_root}")

        return EndpointConfig(
            name=name,
            path=path,
            method=method,  # type: ignore[arg-type]
            request=RequestMatcher(
                path_params=path_params,
                query_params=query_params,
                headers=request_headers,
                body_schema=body_field,
                body_match_mode="json-equal" if body_field is not None else "ignore",
                content_type=first.get("request_content_type"),
            ),
            responses=responses,
        )

    # ------------------------------------------------------------------
    def _derive_servers(self) -> List[ServerConfig]:
        seen: Dict[str, str] = {}
        for entry in self.entries:
            url = ((entry.get("request") or {}).get("url") or "")
            if not url:
                continue
            from urllib.parse import urlparse
            p = urlparse(url)
            base = f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else None
            if base and base not in seen:
                seen[base] = base
        return [ServerConfig(url=u) for u in seen]

    # ------------------------------------------------------------------
    def _unique_name(self, base: str) -> str:
        candidate = re.sub(r"\W+", "_", base).strip("_").lower() or "endpoint"
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


def _template_path(path: str) -> Tuple[str, Dict[str, str]]:
    """Replace dynamic segments with `{param}` placeholders.

    Detection rules: integer ids → ``{id}``; UUIDs → ``{uuid}``; long hex
    strings → ``{hash}``. Returns ``(templated_path, captured_param_values)``.
    """
    segments = path.split("/")
    out_segments: List[str] = []
    params: Dict[str, str] = {}
    counters: Dict[str, int] = {"id": 0, "uuid": 0, "hash": 0}

    for seg in segments:
        if not seg:
            out_segments.append(seg)
            continue
        if _UUID_RE.match(seg):
            name = _next_param_name("uuid", counters)
            params[name] = seg
            out_segments.append("{" + name + "}")
        elif _INT_RE.match(seg):
            name = _next_param_name("id", counters)
            params[name] = seg
            out_segments.append("{" + name + "}")
        elif _HEX_RE.match(seg):
            name = _next_param_name("hash", counters)
            params[name] = seg
            out_segments.append("{" + name + "}")
        else:
            out_segments.append(seg)

    return "/".join(out_segments), params


def _next_param_name(kind: str, counters: Dict[str, int]) -> str:
    if counters[kind] == 0:
        counters[kind] += 1
        return kind
    counters[kind] += 1
    return f"{kind}{counters[kind] - 1}"


def _kvlist_to_dict(items: List[Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if not name:
            continue
        out[name] = str(it.get("value", ""))
    return out


def _header_value(headers: List[Any], name: str) -> Optional[str]:
    for h in headers:
        if isinstance(h, dict) and (h.get("name") or "").lower() == name.lower():
            return str(h.get("value") or "")
    return None


def _post_data_to_value(post_data: Dict[str, Any]) -> Any:
    text = post_data.get("text")
    if not text:
        return None
    mime = (post_data.get("mimeType") or "").lower()
    if "json" in mime:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return text


def _content_to_value(content: Dict[str, Any]) -> Any:
    text = content.get("text")
    if not text:
        return None
    mime = (content.get("mimeType") or "").lower()
    if "json" in mime:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return text


def _value_to_field_spec(value: Any) -> FieldSpec:
    """Build a `FieldSpec` from a concrete observed value.

    Arrays must declare `items`; when the array has at least one element we
    recurse into the first to type the items, otherwise fall back to a
    generic string item.
    """
    if isinstance(value, list):
        items_spec = (
            _value_to_field_spec(value[0]) if value else FieldSpec(type="string")
        )
        return FieldSpec(type="array", items=items_spec, example=value)
    return FieldSpec(type=_py_type(value), example=value)


def _py_type(value: Any) -> str:
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
