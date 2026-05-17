"""Pydantic models for the stubs/mocks track (`sdp-mock-v1`).

Sibling to `models/config_models.py`. Rooted in HTTP/OpenAPI vocabulary
(paths × operations × request matchers × response templates) rather than
relational tables. The two model families never reference each other and
are independently versionable.

Top-level type is `MockConfig`. The hierarchy:

    MockConfig
    ├── info, servers, settings           (metadata)
    ├── schemas: Dict[str, SchemaConfig]  (re-usable request/response shapes)
    ├── endpoints: List[EndpointConfig]   (one HTTP operation each)
    │     ├── request: RequestMatcher
    │     │     ├── path_params, query_params, headers (each Dict[str, FieldSpec])
    │     │     └── body_schema: SchemaConfig | $ref
    │     └── responses: List[ResponseTemplate]
    │           ├── status, headers, weight, delay_ms
    │           └── body_schema: SchemaConfig | $ref
    └── scenarios: List[ScenarioConfig]   (stateful sequences)
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Field-level value spec — leaf node of every schema
# ---------------------------------------------------------------------------


class FieldSpec(BaseModel):
    """How to generate a single field value at render time.

    Bridges OpenAPI's type system with this platform's value generators
    (`utils/helpers.py`, `utils/mimesis_provider.py`). At least one of
    `type`, `ref`, or `business_values` must be set in practice.
    """

    # --- structural ---
    type: Optional[str] = None
        # OpenAPI primitive ("string", "integer", "number", "boolean", "array",
        # "object") or a platform type ("VA32", "N10", "DC", …). The renderer
        # accepts either vocabulary.
    format: Optional[str] = None
        # OpenAPI format hint: "email", "date-time", "uuid", "uri", "iban", …
        # Routed through utils/helpers.py when it matches a known special rule.
    ref: Optional[str] = None
        # "$ref"-style pointer into MockConfig.schemas. Mutually exclusive
        # with the structural fields above. Stored without the leading "#/" —
        # just the schema name (e.g. "Account").
    nullable: bool = False
    null_rate: float = 0.0          # 0.0–1.0; ignored when nullable is False

    # --- value generation hints (mirror ColumnConfig) ---
    special_rule: Optional[str] = None   # e.g. "IBAN", "EMAIL", "REGEX:[A-Z]{2}\\d{4}"
    business_values: Optional[List[Any]] = None  # enum
    pattern: Optional[str] = None        # raw regex
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    # --- nested shapes ---
    items: Optional["FieldSpec"] = None  # for arrays
    properties: Optional[Dict[str, "FieldSpec"]] = None  # for inline objects
    required: List[str] = Field(default_factory=list)

    # --- examples ---
    example: Optional[Any] = None
    examples: List[Any] = Field(default_factory=list)

    # --- description (human + LLM-friendly) ---
    description: Optional[str] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "FieldSpec":
        if self.ref and (self.type or self.properties or self.items):
            raise ValueError(
                "FieldSpec cannot mix `ref` with `type`/`properties`/`items` — "
                "a $ref replaces the inline shape."
            )
        if self.type == "array" and self.items is None and self.ref is None:
            raise ValueError("array FieldSpec must declare `items`")
        if self.null_rate and not self.nullable:
            # Non-fatal: silently coerce. Authors often forget to flip nullable.
            self.nullable = True
        if self.null_rate and not 0.0 <= self.null_rate <= 1.0:
            raise ValueError(f"null_rate must be in [0.0, 1.0]; got {self.null_rate}")
        return self


# ---------------------------------------------------------------------------
# Reusable schema fragments (lives in MockConfig.schemas)
# ---------------------------------------------------------------------------


class SchemaConfig(BaseModel):
    """A re-usable request/response body shape.

    Equivalent of a single entry under OpenAPI's `components.schemas`. Note
    that simple schemas can also be expressed inline inside a FieldSpec —
    `SchemaConfig` exists so common shapes can be named once and `$ref`-ed.
    """
    type: Literal["object", "array", "string", "number", "integer", "boolean"] = "object"
    properties: Dict[str, FieldSpec] = Field(default_factory=dict)
    items: Optional[FieldSpec] = None       # for type=array
    required: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    example: Optional[Any] = None

    # OpenAPI composition keywords. We accept them on input but the template
    # engine currently picks the first variant for `oneOf`/`anyOf` and merges
    # `allOf` into a flat object. Real polymorphism is a future feature.
    one_of: List["SchemaConfig"] = Field(default_factory=list, alias="oneOf")
    any_of: List["SchemaConfig"] = Field(default_factory=list, alias="anyOf")
    all_of: List["SchemaConfig"] = Field(default_factory=list, alias="allOf")
    discriminator: Optional[str] = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate_required_subset(self) -> "SchemaConfig":
        if self.type == "object" and self.required:
            unknown = [r for r in self.required if r not in self.properties]
            if unknown:
                raise ValueError(
                    f"required[] references unknown properties: {unknown}"
                )
        if self.type == "array" and self.items is None and not self.one_of and not self.any_of:
            raise ValueError("array SchemaConfig must declare `items`")
        return self


# ---------------------------------------------------------------------------
# Request matching
# ---------------------------------------------------------------------------


class RequestMatcher(BaseModel):
    """How an incoming HTTP request is matched to this endpoint.

    Mirrors WireMock's request matcher fields and Pact's request specification
    closely enough that renderers can translate without much loss.
    """
    path_params: Dict[str, FieldSpec] = Field(default_factory=dict)
        # /{id} → FieldSpec(type=string, special_rule=IBAN, …)
    query_params: Dict[str, FieldSpec] = Field(default_factory=dict)
    headers: Dict[str, FieldSpec] = Field(default_factory=dict)
    body_schema: Optional[FieldSpec] = None
        # Inline schema OR FieldSpec(ref="Account")
    body_match_mode: Literal["exact", "json-equal", "json-path", "regex", "ignore"] = "json-equal"
    content_type: Optional[str] = None       # e.g. "application/json"

    @model_validator(mode="after")
    def _validate_body_match_mode(self) -> "RequestMatcher":
        if self.body_schema is None and self.body_match_mode != "ignore":
            # Authors often only set body_schema when there *is* a body.
            # Default to ignore in that case so request matching still works.
            self.body_match_mode = "ignore"
        return self


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------


class ResponseTemplate(BaseModel):
    """One response variant for an endpoint (200, 404, 500, …).

    Multiple ResponseTemplates per endpoint allow modelling the realistic
    case where an endpoint sometimes returns 200, sometimes 404, sometimes
    429. The `weight` controls the proportion in random-rendering mode.
    """
    status: int = 200
    headers: Dict[str, FieldSpec] = Field(default_factory=dict)
    body_schema: Optional[FieldSpec] = None
    body_template: Optional[str] = None
        # Raw Jinja-style template string. Used for non-JSON responses
        # (XML, Protobuf JSON, anything weird). Mutually exclusive with body_schema.
    body_content_type: str = "application/json"
    weight: float = 1.0                       # relative probability when rendering randomly
    delay_ms: Optional[int] = None
    description: Optional[str] = None         # for docs / OpenAPI examples
    examples_count: Optional[int] = None
        # Override the endpoint-level examples count for this specific variant.
        # Useful for "render 20 happy paths but only 2 error paths".

    @model_validator(mode="after")
    def _validate_one_body_source(self) -> "ResponseTemplate":
        if self.body_schema is not None and self.body_template is not None:
            raise ValueError(
                "ResponseTemplate cannot have both body_schema and body_template — pick one."
            )
        if not 100 <= self.status <= 599:
            raise ValueError(f"HTTP status must be in [100, 599]; got {self.status}")
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative; got {self.weight}")
        return self


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


class EndpointConfig(BaseModel):
    """One HTTP operation on one path."""
    name: str
        # Human-readable id, used as a filename stem and a scenario identifier.
        # Convention: snake_case, action-first ("list_accounts", "get_account_by_id").
    path: str
        # OpenAPI-style path with curly-brace template params: /accounts/{iban}/transactions
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    summary: Optional[str] = None
    description: Optional[str] = None
    request: RequestMatcher = Field(default_factory=RequestMatcher)
    responses: List[ResponseTemplate] = Field(default_factory=list)
    examples_count: int = 5
        # How many concrete example stubs to render per response variant.
    priority: int = 5
        # WireMock priority for overlap resolution. Lower = higher priority.
        # Use < 5 for catch-all rules, > 5 for specific overrides.
    tags: List[str] = Field(default_factory=list)
        # OpenAPI tags — used for grouping in renderer output.

    @field_validator("path")
    @classmethod
    def _path_must_start_with_slash(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(f"path must start with '/'; got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_responses_present(self) -> "EndpointConfig":
        if not self.responses:
            # An endpoint with no responses is useless; renderers would crash.
            raise ValueError(
                f"endpoint {self.name!r} must declare at least one ResponseTemplate"
            )
        return self


# ---------------------------------------------------------------------------
# Scenarios — stateful sequences
# ---------------------------------------------------------------------------


class StateTransition(BaseModel):
    """One step in a scenario state machine.

    Example: after 3 calls to `list_accounts`, switch to a 429 response.
    """
    on_match: Dict[str, Any] = Field(default_factory=dict)
        # { "endpoint": "list_accounts" } and/or { "header": {...} }
    after: int = 1
        # Trigger after the Nth matching request.
    next_response: Dict[str, Any] = Field(default_factory=dict)
        # Inline override of status/body to use after the trigger fires.
    requires_state: Optional[str] = None
    sets_state: Optional[str] = None
        # Named state markers, mirroring WireMock's scenarioName mechanism.


class ScenarioConfig(BaseModel):
    """A named stateful sequence (request N changes the response of N+1)."""
    name: str
    description: Optional[str] = None
    states: List[StateTransition] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """One server entry — base URL plus optional description."""
    url: str
    description: Optional[str] = None


class MockInfo(BaseModel):
    """OpenAPI-style info block."""
    title: str = "Mock API"
    version: str = "1.0.0"
    description: Optional[str] = None
    contact_email: Optional[str] = None


class MockSettings(BaseModel):
    """Global render-time settings."""
    base_path: str = ""
        # Prepended to every endpoint path at render time. Useful when one
        # MockConfig is hosted under multiple service prefixes.
    default_latency_ms: int = 0
    default_examples_count: int = 5
        # Used when EndpointConfig.examples_count is not overridden.
    error_injection_rate: float = 0.0
        # Probability that the renderer adds a random 5xx variant per endpoint.
    cors_enabled: bool = True
    seed: Optional[int] = None
        # Optional render-time seed (CLI --seed wins over this).


class MockConfig(BaseModel):
    """Top-level container for a `sdp-mock-v1` document."""
    config_format: Literal["sdp-mock-v1"] = "sdp-mock-v1"
    info: MockInfo = Field(default_factory=MockInfo)
    servers: List[ServerConfig] = Field(default_factory=list)
    settings: MockSettings = Field(default_factory=MockSettings)
    schemas: Dict[str, SchemaConfig] = Field(default_factory=dict)
    endpoints: List[EndpointConfig] = Field(default_factory=list)
    scenarios: List[ScenarioConfig] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience lookups
    # ------------------------------------------------------------------
    def get_endpoint(self, name: str) -> Optional[EndpointConfig]:
        return next((e for e in self.endpoints if e.name == name), None)

    def get_schema(self, name: str) -> Optional[SchemaConfig]:
        return self.schemas.get(name)

    def resolve_ref(self, field: FieldSpec) -> Optional[SchemaConfig]:
        """Follow a FieldSpec.ref to the corresponding SchemaConfig (one hop)."""
        if not field.ref:
            return None
        return self.schemas.get(field.ref)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate_refs(self) -> "MockConfig":
        """Collect every $ref in the document and confirm the target exists.

        We only validate *named* refs against `schemas`. Inline FieldSpec /
        SchemaConfig nesting is already validated by the inner models.
        """
        unresolved: List[str] = []

        def _walk_field(field: FieldSpec, path: str) -> None:
            if field.ref and field.ref not in self.schemas:
                unresolved.append(f"{path} -> ref '{field.ref}'")
            if field.items:
                _walk_field(field.items, f"{path}[]")
            for prop_name, prop in (field.properties or {}).items():
                _walk_field(prop, f"{path}.{prop_name}")

        def _walk_schema(schema: SchemaConfig, path: str) -> None:
            for prop_name, field in schema.properties.items():
                _walk_field(field, f"{path}.{prop_name}")
            if schema.items:
                _walk_field(schema.items, f"{path}[]")

        for schema_name, schema in self.schemas.items():
            _walk_schema(schema, f"schemas.{schema_name}")

        for ep in self.endpoints:
            for param_name, field in ep.request.path_params.items():
                _walk_field(field, f"{ep.name}.request.path_params.{param_name}")
            for param_name, field in ep.request.query_params.items():
                _walk_field(field, f"{ep.name}.request.query_params.{param_name}")
            if ep.request.body_schema:
                _walk_field(ep.request.body_schema, f"{ep.name}.request.body")
            for i, resp in enumerate(ep.responses):
                if resp.body_schema:
                    _walk_field(resp.body_schema, f"{ep.name}.responses[{i}].body")

        if unresolved:
            raise ValueError(
                "MockConfig has unresolved $refs:\n  - " + "\n  - ".join(unresolved)
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_endpoint_names(self) -> "MockConfig":
        seen: set[str] = set()
        for ep in self.endpoints:
            if ep.name in seen:
                raise ValueError(f"duplicate endpoint name: {ep.name!r}")
            seen.add(ep.name)
        return self


# Forward reference resolution for self-referential FieldSpec
FieldSpec.model_rebuild()
SchemaConfig.model_rebuild()
