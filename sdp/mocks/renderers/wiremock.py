"""WireMock renderer — `MockConfig` → WireMock-compatible stub mappings.

Output structure (mirrors what WireMock standalone expects):

    output/
      mappings/
        <endpoint_name>__<status>__<index>.json   (one mapping per example)
      __files/                                    (currently empty;
                                                   reserved for the future
                                                   bodyFileName feature)

Each generated mapping is a self-contained WireMock stub:

    {
      "name": "...",
      "priority": 5,
      "request": { "method": "GET", "urlPath": "/accounts/DE89..." },
      "response": {
        "status": 200,
        "headers": { "Content-Type": "application/json" },
        "jsonBody": { ... }
      }
    }

Path templates (`/accounts/{iban}`) become a fixed `urlPath` per example
where the path parameter is rendered to a concrete value via the template
engine. This means N examples = N separate stubs, each with a different
concrete path. That's how WireMock prefers it — request matching is
literal by default.

For the "match anything" mode, set `--match-mode any` at the CLI; the
renderer then emits one stub per (endpoint, status) variant using
`urlPathPattern` with a regex placeholder for path parameters.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sdp.mocks.scenario_engine import ScenarioPlan, ScenarioStep, compile_scenarios
from sdp.mocks.template_engine import TemplateEngine
from sdp.models.mock_models import (
    EndpointConfig,
    FieldSpec,
    MockConfig,
    ResponseTemplate,
)

logger = logging.getLogger(__name__)


# Maps OpenAPI primitive types to a WireMock urlPathPattern regex chunk.
_PATH_PARAM_REGEX_FOR_TYPE: Dict[str, str] = {
    "integer": r"[0-9]+",
    "number": r"-?[0-9]+(\.[0-9]+)?",
    "string": r"[^/]+",
    "boolean": r"true|false",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_wiremock(
    config: MockConfig,
    output_dir: Path,
    *,
    match_mode: str = "concrete",       # "concrete" | "any"
    examples_per_endpoint: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Path]:
    """Write WireMock mappings into ``output_dir/mappings/``.

    Parameters
    ----------
    match_mode:
        ``"concrete"`` — emit one mapping per example; path params are
        rendered to literal values. Best for snapshot-style tests.
        ``"any"`` — emit one mapping per (endpoint, status); path params
        become regex placeholders so any IBAN/UUID/etc. matches the same
        stub. Best for development-style mocks.
    examples_per_endpoint:
        Override `EndpointConfig.examples_count`. Useful for the CLI
        ``--examples`` flag.
    """
    output_dir = Path(output_dir)
    mappings_dir = output_dir / "mappings"
    files_dir = output_dir / "__files"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    engine = TemplateEngine(config, seed=seed)
    plan = compile_scenarios(config)
    written: List[Path] = []

    # Scenario steps render alongside the regular mappings (one extra mapping
    # per step). They're tagged with the scenario name so consumers can
    # filter or replay them through WireMock's admin API.
    for step in plan.steps:
        mapping = _build_scenario_mapping(
            config=config,
            step=step,
            engine=engine,
            match_mode=match_mode,
        )
        if mapping is None:
            continue
        fname = _safe_filename(
            f"scenario__{step.scenario_name}__{step.required_state}__to__{step.new_state}.json"
        )
        out_path = mappings_dir / fname
        out_path.write_text(json.dumps(mapping, indent=2, default=str), encoding="utf-8")
        written.append(out_path)

    for endpoint in config.endpoints:
        n_examples = examples_per_endpoint or endpoint.examples_count or 5
        for resp_index, response in enumerate(endpoint.responses):
            count = response.examples_count or (
                n_examples if response.status < 400 else max(1, n_examples // 5)
            )
            for example_index in range(count):
                mapping = _build_mapping(
                    config=config,
                    endpoint=endpoint,
                    response=response,
                    response_index=resp_index,
                    example_index=example_index,
                    engine=engine,
                    match_mode=match_mode,
                )
                if mapping is None:
                    continue
                fname = _safe_filename(
                    f"{endpoint.name}__{response.status}__{example_index:02d}.json"
                )
                out_path = mappings_dir / fname
                out_path.write_text(
                    json.dumps(mapping, indent=2, default=str),
                    encoding="utf-8",
                )
                written.append(out_path)
                # `match_mode=any` collapses examples into a single regex stub
                if match_mode == "any":
                    break

    logger.info(
        "wiremock: wrote %d mapping(s) to %s (mode=%s)",
        len(written), mappings_dir, match_mode,
    )
    return written


# ---------------------------------------------------------------------------
# One mapping
# ---------------------------------------------------------------------------


def _build_mapping(
    *,
    config: MockConfig,
    endpoint: EndpointConfig,
    response: ResponseTemplate,
    response_index: int,
    example_index: int,
    engine: TemplateEngine,
    match_mode: str,
) -> Optional[Dict[str, Any]]:
    base = config.settings.base_path or ""
    full_path = base + endpoint.path

    # Path parameters: substitute (concrete) or convert to regex (any-mode)
    rendered_path, url_field = _resolve_path(
        full_path, endpoint, engine, match_mode,
    )

    request_block: Dict[str, Any] = {
        "method": endpoint.method,
        url_field: rendered_path,
    }

    # Query parameter matchers
    if endpoint.request.query_params:
        request_block["queryParameters"] = {
            name: {"matches": ".*"} for name in endpoint.request.query_params
        }

    # Header matchers — only require explicitly listed headers
    if endpoint.request.headers:
        request_block["headers"] = {
            name: {"matches": ".+"} for name in endpoint.request.headers
        }

    # Request body matcher (only for non-GET/DELETE typically, but trust the spec)
    if endpoint.request.body_schema is not None:
        if endpoint.request.body_match_mode == "json-equal":
            # Use a permissive matcher so consumers can post any compliant body
            request_block["bodyPatterns"] = [{"matchesJsonPath": "$"}]

    # Response block
    response_block: Dict[str, Any] = {
        "status": response.status,
        "headers": _render_headers(response, endpoint, engine),
    }
    if response.delay_ms is not None or config.settings.default_latency_ms:
        response_block["fixedDelayMilliseconds"] = (
            response.delay_ms if response.delay_ms is not None
            else config.settings.default_latency_ms
        )
    body = _render_body(response, engine)
    if body is not None:
        if isinstance(body, (dict, list)):
            response_block["jsonBody"] = body
        else:
            response_block["body"] = str(body)

    name = f"{endpoint.name} {response.status} #{example_index + 1}"

    mapping: Dict[str, Any] = {
        "name": name,
        "priority": endpoint.priority,
        "request": request_block,
        "response": response_block,
    }

    # Tag with metadata so debugging & inspection are easier
    mapping["metadata"] = {
        "sdp": {
            "endpoint": endpoint.name,
            "tags": endpoint.tags,
            "status": response.status,
            "example_index": example_index,
        }
    }
    return mapping


def _resolve_path(
    full_path: str,
    endpoint: EndpointConfig,
    engine: TemplateEngine,
    match_mode: str,
) -> tuple[str, str]:
    """Return ``(rendered_path, url_field)``.

    ``url_field`` is either ``"urlPath"`` (concrete) or ``"urlPathPattern"``
    (regex). Concrete mode replaces ``{x}`` with a generated value. Regex
    mode replaces ``{x}`` with a regex chunk based on the path-param's type.
    """
    if "{" not in full_path:
        return full_path, "urlPath"

    if match_mode == "concrete":
        rendered = full_path
        for name, field in endpoint.request.path_params.items():
            value = engine.render_field(field)
            rendered = rendered.replace("{" + name + "}", str(value))
        # If any unspecified params remain (rare), drop in a wildcard
        rendered = re.sub(r"\{[^}]+\}", "any", rendered)
        return rendered, "urlPath"

    # any-mode: build a regex
    pattern = full_path
    for name, field in endpoint.request.path_params.items():
        regex = _PATH_PARAM_REGEX_FOR_TYPE.get(
            (field.type or "string").lower(), r"[^/]+"
        )
        pattern = pattern.replace("{" + name + "}", regex)
    pattern = re.sub(r"\{[^}]+\}", r"[^/]+", pattern)
    return pattern, "urlPathPattern"


def _render_body(response: ResponseTemplate, engine: TemplateEngine) -> Any:
    if response.body_template is not None:
        return engine._render_template_string(response.body_template)
    if response.body_schema is None:
        return None
    return engine.render_field(response.body_schema)


def _render_headers(
    response: ResponseTemplate,
    endpoint: EndpointConfig,
    engine: TemplateEngine,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "Content-Type" not in response.headers and response.body_content_type:
        out["Content-Type"] = response.body_content_type
    for name, field in response.headers.items():
        value = engine.render_field(field)
        if value is not None:
            out[name] = str(value)
    return out


# ---------------------------------------------------------------------------
# Scenario mappings
# ---------------------------------------------------------------------------


def _build_scenario_mapping(
    *,
    config: MockConfig,
    step: ScenarioStep,
    engine: TemplateEngine,
    match_mode: str,
) -> Optional[Dict[str, Any]]:
    """Translate a compiled `ScenarioStep` into a WireMock mapping JSON.

    The mapping carries ``scenarioName`` + ``requiredScenarioState`` + ``newScenarioState``
    so WireMock advances its internal scenario state machine on each match.
    """
    base = config.settings.base_path or ""

    # If the step targets a specific endpoint, use that endpoint's path/method;
    # otherwise emit a wildcard request matcher.
    request_block: Dict[str, Any]
    if step.endpoint_name:
        endpoint = config.get_endpoint(step.endpoint_name)
        if endpoint is None:
            return None
        full_path = base + endpoint.path
        rendered_path, url_field = _resolve_path(
            full_path, endpoint, engine, match_mode,
        )
        request_block = {
            "method": endpoint.method,
            url_field: rendered_path,
        }
    else:
        # Catch-all scenario step
        request_block = {"method": "ANY", "urlPathPattern": ".*"}

    # Response — either the override declared in the transition, or the
    # endpoint's default 2xx response.
    if step.response_override:
        response_block = _build_override_response(step.response_override)
    elif step.endpoint_name:
        endpoint = config.get_endpoint(step.endpoint_name)
        default = endpoint.responses[0] if endpoint and endpoint.responses else None
        if default is None:
            return None
        response_block = {
            "status": default.status,
            "headers": _render_headers(default, endpoint, engine),
        }
        body = _render_body(default, engine)
        if body is not None:
            if isinstance(body, (dict, list)):
                response_block["jsonBody"] = body
            else:
                response_block["body"] = str(body)
    else:
        response_block = {"status": 200}

    # Higher priority than the "regular" mappings so scenario steps win
    # when both could match (priority is "lower number = higher precedence"
    # in WireMock).
    priority = 1

    return {
        "name": f"scenario:{step.scenario_name}:{step.required_state}->{step.new_state}",
        "priority": priority,
        "scenarioName": step.scenario_name,
        "requiredScenarioState": step.required_state,
        "newScenarioState": step.new_state,
        "request": request_block,
        "response": response_block,
        "metadata": {
            "sdp": {
                "scenario": step.scenario_name,
                "endpoint": step.endpoint_name,
                "after": step.after,
                "requires_state": step.requires_state,
                "sets_state": step.sets_state,
            }
        },
    }


def _build_override_response(override: Dict[str, Any]) -> Dict[str, Any]:
    """Translate `StateTransition.next_response` into a WireMock response block."""
    resp: Dict[str, Any] = {"status": int(override.get("status", 200))}
    headers = override.get("headers")
    if headers:
        resp["headers"] = {str(k): str(v) for k, v in headers.items()}
    body = override.get("body")
    if body is not None:
        if isinstance(body, (dict, list)):
            resp["jsonBody"] = body
        else:
            resp["body"] = str(body)
    delay = override.get("delay_ms")
    if delay is not None:
        resp["fixedDelayMilliseconds"] = int(delay)
    return resp


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", name).strip("_")
