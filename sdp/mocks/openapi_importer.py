"""OpenAPI 3.x → MockConfig importer.

Walks an OpenAPI document and produces a fully-validated `MockConfig`. The
goal is a 90% solution that covers the cases real-world specs use most:
flat object schemas, $ref to components.schemas, enum values, common
formats (email, uuid, date-time, iban), path/query/header parameters,
multiple response status codes, and `example` / `examples` blocks.

Out of scope for the first cut (tracked as TODOs):
  - oneOf/anyOf/allOf composition (we currently flatten allOf and pick
    the first variant of one/any-of)
  - discriminator-based polymorphism
  - external $refs (cross-file)
  - security schemes (OAuth2 flows, OpenID Connect) — request matchers
    capture the header/query but flow definitions are not yet modelled
  - server variables and templated server URLs
  - callbacks / webhooks
  - parameter `style` and `explode` for query serialisation

Usage:
    from sdp.mocks.openapi_importer import import_openapi
    cfg = import_openapi("api/openapi.yaml")
    # cfg is a MockConfig — dump_mock_config to write it to disk
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

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


# ---------------------------------------------------------------------------
# Format hint mapping — OpenAPI `format:` → platform `special_rule`
# ---------------------------------------------------------------------------


_FORMAT_TO_SPECIAL_RULE: Dict[str, str] = {
    "email": "EMAIL",
    "uuid": "UUID",
    "uri": "URL",
    "url": "URL",
    "ipv4": "IPV4",
    "ipv6": "IPV6",
    "iban": "IBAN",
    "bic": "SWIFT",
    "swift": "SWIFT",
    # Datetime formats are passed through as `format` so renderers can decide
    # whether to call helpers.faker_date_time / .date / similar.
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OpenAPIImportError(ValueError):
    """Raised when an OpenAPI document cannot be turned into a MockConfig."""


def import_openapi(path: Union[str, Path]) -> MockConfig:
    """Read an OpenAPI 3.x YAML or JSON spec and return a `MockConfig`."""
    p = Path(path)
    if not p.exists():
        raise OpenAPIImportError(f"OpenAPI spec not found: {p}")
    return import_openapi_str(p.read_text(encoding="utf-8"), source=str(p))


def import_openapi_str(raw: str, *, source: str = "<string>") -> MockConfig:
    """Parse an OpenAPI 3.x string into a `MockConfig`."""
    try:
        spec = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise OpenAPIImportError(f"{source}: invalid YAML/JSON: {exc}") from exc

    if not isinstance(spec, dict):
        raise OpenAPIImportError(f"{source}: top-level OpenAPI document must be a mapping")

    version = str(spec.get("openapi") or spec.get("swagger") or "").strip()
    if not version:
        raise OpenAPIImportError(
            f"{source}: missing `openapi:` or `swagger:` version key — is this really an OpenAPI doc?"
        )
    if not (version.startswith("3.") or version.startswith("2.")):
        logger.warning("Unrecognised OpenAPI version %r — attempting import anyway", version)

    importer = _Importer(spec, source)
    return importer.build()


# ---------------------------------------------------------------------------
# Importer — does the walk
# ---------------------------------------------------------------------------


_REF_PATTERN = re.compile(r"^#/components/schemas/(?P<name>[A-Za-z0-9_.\-]+)$")


class _Importer:
    """Stateful walker — collects schemas first, then endpoints."""

    def __init__(self, spec: Dict[str, Any], source: str):
        self.spec = spec
        self.source = source
        self.components = (spec.get("components") or {})
        self.component_schemas: Dict[str, Any] = self.components.get("schemas") or {}
        self.component_parameters: Dict[str, Any] = self.components.get("parameters") or {}
        self.schemas: Dict[str, SchemaConfig] = {}
        self.endpoints: List[EndpointConfig] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def build(self) -> MockConfig:
        # Phase 1: convert all named schemas first so refs resolve.
        for name, raw in self.component_schemas.items():
            self.schemas[name] = self._schema_to_config(raw, schema_name=name)

        # Phase 2: walk paths.
        paths = self.spec.get("paths") or {}
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            self._walk_path_item(path, path_item)

        info = self._build_info()
        servers = self._build_servers()
        return MockConfig(
            info=info,
            servers=servers,
            settings=MockSettings(),
            schemas=self.schemas,
            endpoints=self.endpoints,
        )

    # ------------------------------------------------------------------
    # info / servers
    # ------------------------------------------------------------------
    def _build_info(self) -> MockInfo:
        info = self.spec.get("info") or {}
        return MockInfo(
            title=str(info.get("title") or "Imported API"),
            version=str(info.get("version") or "1.0.0"),
            description=info.get("description"),
            contact_email=(info.get("contact") or {}).get("email") if isinstance(info.get("contact"), dict) else None,
        )

    def _build_servers(self) -> List[ServerConfig]:
        out: List[ServerConfig] = []
        for server in (self.spec.get("servers") or []):
            if isinstance(server, dict) and server.get("url"):
                out.append(ServerConfig(url=str(server["url"]), description=server.get("description")))
        return out

    # ------------------------------------------------------------------
    # Paths and operations
    # ------------------------------------------------------------------
    def _walk_path_item(self, path: str, path_item: Dict[str, Any]) -> None:
        path_level_params = path_item.get("parameters") or []
        for method in [m.lower() for m in HTTP_METHODS]:
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue
            try:
                ep = self._operation_to_endpoint(path, method.upper(), op, path_level_params)
                self.endpoints.append(ep)
            except Exception as exc:
                logger.warning(
                    "Skipping %s %s — could not import: %s", method.upper(), path, exc
                )

    def _operation_to_endpoint(
        self,
        path: str,
        method: str,
        op: Dict[str, Any],
        path_level_params: List[Any],
    ) -> EndpointConfig:
        op_id = op.get("operationId") or self._auto_op_id(method, path)
        # OpenAPI operationId is the most reliable name source. We sanitise to
        # snake_case + ensure uniqueness across the importer's lifetime.
        name = self._unique_name(self._snake_case(op_id))

        # Combine path-level + operation-level parameters
        all_params = list(path_level_params) + list(op.get("parameters") or [])
        request = self._build_request_matcher(all_params, op.get("requestBody"))

        responses = self._build_responses(op.get("responses") or {})
        if not responses:
            # Endpoints without any response are valid in OpenAPI but useless
            # as mocks — fabricate a minimal 200.
            responses.append(ResponseTemplate(
                status=200,
                body_schema=FieldSpec(type="object"),
                description="(synthesised: spec defined no responses)",
            ))

        tags = [str(t) for t in (op.get("tags") or []) if isinstance(t, (str, int))]

        return EndpointConfig(
            name=name,
            path=path,
            method=method,
            summary=op.get("summary"),
            description=op.get("description"),
            request=request,
            responses=responses,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Request matcher: parameters + body
    # ------------------------------------------------------------------
    def _build_request_matcher(
        self,
        parameters: List[Any],
        request_body: Optional[Dict[str, Any]],
    ) -> RequestMatcher:
        path_params: Dict[str, FieldSpec] = {}
        query_params: Dict[str, FieldSpec] = {}
        headers: Dict[str, FieldSpec] = {}

        for raw in parameters:
            param = self._resolve_parameter(raw)
            if not param:
                continue
            name = param.get("name")
            if not name:
                continue
            location = param.get("in")
            schema = param.get("schema") or {}
            field = self._schema_to_fieldspec(schema)
            if param.get("example") is not None:
                field.example = param["example"]
            if param.get("description"):
                field.description = param["description"]

            if location == "path":
                path_params[name] = field
            elif location == "query":
                query_params[name] = field
            elif location == "header":
                headers[name] = field
            # cookies are intentionally skipped — out of scope for first cut

        body_schema: Optional[FieldSpec] = None
        content_type: Optional[str] = None
        body_match_mode: str = "ignore"
        if request_body and isinstance(request_body, dict):
            content = request_body.get("content") or {}
            for ct, media in content.items():
                if isinstance(media, dict) and media.get("schema"):
                    body_schema = self._schema_to_fieldspec(media["schema"])
                    content_type = ct
                    body_match_mode = "json-equal" if "json" in ct.lower() else "ignore"
                    break  # take the first content type — usually application/json

        return RequestMatcher(
            path_params=path_params,
            query_params=query_params,
            headers=headers,
            body_schema=body_schema,
            body_match_mode=body_match_mode,  # type: ignore[arg-type]
            content_type=content_type,
        )

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------
    def _build_responses(self, responses: Dict[str, Any]) -> List[ResponseTemplate]:
        out: List[ResponseTemplate] = []
        for status_str, raw in responses.items():
            if not isinstance(raw, dict):
                continue
            try:
                status = int(status_str) if status_str != "default" else 200
            except ValueError:
                continue  # skip wildcard codes like '2XX' for now

            response = self._resolve_ref_or_inline(raw, kind="responses")
            if not isinstance(response, dict):
                continue

            headers = self._build_response_headers(response.get("headers") or {})
            body_schema, body_content_type = self._first_response_body(response.get("content") or {})

            out.append(ResponseTemplate(
                status=status,
                headers=headers,
                body_schema=body_schema,
                body_content_type=body_content_type or "application/json",
                description=response.get("description"),
            ))
        # Sort 2xx first so the first variant is the happy path
        out.sort(key=lambda r: (r.status >= 400, r.status))
        return out

    def _build_response_headers(self, headers_raw: Dict[str, Any]) -> Dict[str, FieldSpec]:
        out: Dict[str, FieldSpec] = {}
        for name, h in headers_raw.items():
            if not isinstance(h, dict):
                continue
            schema = h.get("schema") or {}
            field = self._schema_to_fieldspec(schema)
            if h.get("example") is not None:
                field.example = h["example"]
            if h.get("description"):
                field.description = h["description"]
            out[name] = field
        return out

    def _first_response_body(
        self, content: Dict[str, Any]
    ) -> Tuple[Optional[FieldSpec], Optional[str]]:
        for ct, media in content.items():
            if not isinstance(media, dict):
                continue
            schema = media.get("schema")
            if not schema:
                continue
            field = self._schema_to_fieldspec(schema)
            # Pull example/examples through if present
            if media.get("example") is not None:
                field.example = media["example"]
            elif isinstance(media.get("examples"), dict):
                examples = []
                for ex in media["examples"].values():
                    if isinstance(ex, dict) and ex.get("value") is not None:
                        examples.append(ex["value"])
                if examples:
                    field.examples = examples
                    field.example = examples[0]
            return field, ct
        return None, None

    # ------------------------------------------------------------------
    # Schema → FieldSpec / SchemaConfig
    # ------------------------------------------------------------------
    def _schema_to_fieldspec(self, schema: Dict[str, Any]) -> FieldSpec:
        """Convert an inline OpenAPI schema to a FieldSpec (used at field level)."""
        if not isinstance(schema, dict):
            return FieldSpec(type="string")

        # $ref → just record the target name; renderer will follow it
        ref = schema.get("$ref")
        if ref:
            target = self._ref_to_name(ref)
            if target:
                return FieldSpec(ref=target)
            logger.warning("Unrecognised $ref shape %r — treating as opaque string", ref)
            return FieldSpec(type="string")

        # allOf: merge into one
        if schema.get("allOf"):
            merged = self._merge_allOf(schema["allOf"])
            return self._schema_to_fieldspec(merged)

        # oneOf / anyOf: take the first variant for now (TODO)
        for keyword in ("oneOf", "anyOf"):
            if schema.get(keyword):
                first = schema[keyword][0]
                return self._schema_to_fieldspec(first)

        otype = schema.get("type")
        oformat = schema.get("format")

        # Inline object
        if otype == "object" or "properties" in schema:
            props = {
                name: self._schema_to_fieldspec(child)
                for name, child in (schema.get("properties") or {}).items()
            }
            return FieldSpec(
                type="object",
                properties=props,
                required=list(schema.get("required") or []),
                description=schema.get("description"),
                example=schema.get("example"),
            )

        # Inline array
        if otype == "array":
            items_schema = schema.get("items") or {"type": "string"}
            return FieldSpec(
                type="array",
                items=self._schema_to_fieldspec(items_schema),
                description=schema.get("description"),
                example=schema.get("example"),
                min_length=schema.get("minItems"),
                max_length=schema.get("maxItems"),
            )

        # Primitives
        spec = FieldSpec(
            type=otype or "string",
            format=oformat,
            description=schema.get("description"),
            example=schema.get("example"),
            pattern=schema.get("pattern"),
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            min_length=schema.get("minLength"),
            max_length=schema.get("maxLength"),
            nullable=bool(schema.get("nullable", False)),
        )
        if schema.get("enum"):
            spec.business_values = list(schema["enum"])
        if oformat and oformat in _FORMAT_TO_SPECIAL_RULE:
            spec.special_rule = _FORMAT_TO_SPECIAL_RULE[oformat]
        return spec

    def _schema_to_config(self, schema: Dict[str, Any], schema_name: str) -> SchemaConfig:
        """Convert a named OpenAPI schema to a SchemaConfig (lives in MockConfig.schemas)."""
        if not isinstance(schema, dict):
            return SchemaConfig(type="object")

        if schema.get("allOf"):
            schema = self._merge_allOf(schema["allOf"])

        otype = schema.get("type")
        if otype == "array":
            items_schema = schema.get("items") or {"type": "string"}
            return SchemaConfig(
                type="array",
                items=self._schema_to_fieldspec(items_schema),
                description=schema.get("description"),
                example=schema.get("example"),
            )

        # Default to object — most OpenAPI named schemas are objects.
        properties = {
            name: self._schema_to_fieldspec(child)
            for name, child in (schema.get("properties") or {}).items()
        }
        return SchemaConfig(
            type="object" if (otype is None or otype == "object") else otype,  # type: ignore[arg-type]
            properties=properties,
            required=list(schema.get("required") or []),
            description=schema.get("description"),
            example=schema.get("example"),
        )

    # ------------------------------------------------------------------
    # $ref resolution helpers
    # ------------------------------------------------------------------
    def _ref_to_name(self, ref: str) -> Optional[str]:
        m = _REF_PATTERN.match(ref)
        return m.group("name") if m else None

    def _resolve_parameter(self, raw: Any) -> Optional[Dict[str, Any]]:
        """Parameters can be `$ref: '#/components/parameters/Foo'` — resolve once."""
        if not isinstance(raw, dict):
            return None
        ref = raw.get("$ref")
        if not ref:
            return raw
        if ref.startswith("#/components/parameters/"):
            name = ref.split("/")[-1]
            return self.component_parameters.get(name)
        return None

    def _resolve_ref_or_inline(self, raw: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
        ref = raw.get("$ref")
        if not ref:
            return raw
        prefix = f"#/components/{kind}/"
        if ref.startswith(prefix):
            name = ref[len(prefix):]
            return (self.components.get(kind) or {}).get(name)
        return None

    def _merge_allOf(self, parts: List[Any]) -> Dict[str, Any]:
        """Naïve allOf merge: union of properties + required, last-wins on type."""
        merged: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for part in parts:
            if isinstance(part, dict) and part.get("$ref"):
                # Resolve a $ref into a copy of the target schema
                name = self._ref_to_name(part["$ref"])
                if name and name in self.component_schemas:
                    part = self.component_schemas[name]
            if not isinstance(part, dict):
                continue
            if part.get("type"):
                merged["type"] = part["type"]
            merged["properties"].update(part.get("properties") or {})
            for r in (part.get("required") or []):
                if r not in merged["required"]:
                    merged["required"].append(r)
        return merged

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------
    def _auto_op_id(self, method: str, path: str) -> str:
        slug = re.sub(r"[{}/]+", "_", path).strip("_") or "root"
        return f"{method.lower()}_{slug}"

    @staticmethod
    def _snake_case(name: str) -> str:
        # CamelCase → camel_case + collapse non-ident chars to underscores
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
        return s or "endpoint"

    def _unique_name(self, base: str) -> str:
        existing = {ep.name for ep in self.endpoints}
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"
