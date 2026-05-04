"""Pact v3 renderer — `MockConfig` → consumer-driven contract files.

Pact contracts describe an interaction between a named *consumer* and a
named *provider*. Each interaction captures one expected request/response
pair, optionally tagged with a `providerState` precondition.

Output: one ``<consumer>-<provider>.json`` file per render run, holding
all interactions for the MockConfig. Pact tooling typically expects one
file per (consumer, provider) pair, so we produce that single file.

Pact spec: https://github.com/pact-foundation/pact-specification

Limitations of this first cut (vs. the full spec):
  - We don't emit `matchingRules` — the values rendered are concrete, so
    consumers see exact-match expectations. Add matching rules in a
    follow-up if you need fuzzy matching at the contract level.
  - We collapse all `ScenarioConfig.states` of an endpoint into a single
    `providerStates` array per interaction (best-effort mapping).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mocks.template_engine import TemplateEngine
from models.mock_models import (
    EndpointConfig,
    MockConfig,
    ResponseTemplate,
    ScenarioConfig,
)

logger = logging.getLogger(__name__)


PACT_SPEC_VERSION = "3.0.0"


def render_pact(
    config: MockConfig,
    output_dir: Path,
    *,
    consumer: str = "consumer",
    provider: str = "provider",
    seed: Optional[int] = None,
    examples_per_endpoint: Optional[int] = None,
    include_error_responses: bool = True,
) -> List[Path]:
    """Write one Pact v3 contract file under ``output_dir``.

    Parameters
    ----------
    consumer, provider:
        Names that go into the ``consumer.name`` / ``provider.name`` fields.
        Pact tooling uses these to file the contract.
    examples_per_endpoint:
        How many distinct interactions to generate per (endpoint, status)
        pair. Each interaction is rendered with its own concrete values, so
        if you set this to 3, you get 3 interactions per endpoint+status.
    include_error_responses:
        If True (default), emit interactions for 4xx/5xx variants too.
        Some teams prefer happy-path-only contracts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = TemplateEngine(config, seed=seed)
    scenarios_by_endpoint = _index_scenarios(config.scenarios)

    interactions: List[Dict[str, Any]] = []
    base_path = config.settings.base_path or ""
    n_examples_default = examples_per_endpoint or config.settings.default_examples_count

    for endpoint in config.endpoints:
        for response in endpoint.responses:
            if not include_error_responses and response.status >= 400:
                continue
            count = response.examples_count or n_examples_default
            for example_index in range(count):
                interactions.append(_build_interaction(
                    endpoint=endpoint,
                    response=response,
                    example_index=example_index,
                    engine=engine,
                    base_path=base_path,
                    provider_states=scenarios_by_endpoint.get(endpoint.name, []),
                ))

    pact: Dict[str, Any] = {
        "consumer": {"name": consumer},
        "provider": {"name": provider},
        "interactions": interactions,
        "metadata": {
            "pactSpecification": {"version": PACT_SPEC_VERSION},
            "client": {"name": "synthetic-data-platform", "version": "0.1.0"},
        },
    }

    out_path = output_dir / f"{_safe(consumer)}-{_safe(provider)}.json"
    out_path.write_text(json.dumps(pact, indent=2, default=str), encoding="utf-8")

    logger.info(
        "pact: wrote %d interaction(s) to %s",
        len(interactions), out_path,
    )
    return [out_path]


# ---------------------------------------------------------------------------
# One interaction
# ---------------------------------------------------------------------------


def _build_interaction(
    *,
    endpoint: EndpointConfig,
    response: ResponseTemplate,
    example_index: int,
    engine: TemplateEngine,
    base_path: str,
    provider_states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    description = (
        endpoint.summary
        or endpoint.description
        or f"{endpoint.method} {endpoint.path} → {response.status}"
    )
    if example_index > 0:
        description = f"{description} (example {example_index + 1})"

    # ---- request ----
    rendered_path = base_path + endpoint.path
    for name, field in endpoint.request.path_params.items():
        rendered_path = rendered_path.replace(
            "{" + name + "}", str(engine.render_field(field))
        )

    request_block: Dict[str, Any] = {
        "method": endpoint.method,
        "path": rendered_path,
    }
    if endpoint.request.query_params:
        request_block["query"] = {
            name: [str(engine.render_field(field))]
            for name, field in endpoint.request.query_params.items()
        }
    if endpoint.request.headers:
        request_block["headers"] = {
            name: str(engine.render_field(field))
            for name, field in endpoint.request.headers.items()
        }
    if endpoint.request.body_schema is not None:
        request_block["body"] = engine.render_field(endpoint.request.body_schema)
        if endpoint.request.content_type:
            request_block.setdefault("headers", {})
            request_block["headers"]["Content-Type"] = endpoint.request.content_type

    # ---- response ----
    response_block: Dict[str, Any] = {"status": response.status}
    headers = {}
    if response.body_schema is not None or response.body_template is not None:
        headers["Content-Type"] = response.body_content_type
    for name, field in response.headers.items():
        headers[name] = str(engine.render_field(field))
    if headers:
        response_block["headers"] = headers
    if response.body_template is not None:
        response_block["body"] = engine._render_template_string(response.body_template)
    elif response.body_schema is not None:
        response_block["body"] = engine.render_field(response.body_schema)

    interaction: Dict[str, Any] = {
        "description": description,
        "request": request_block,
        "response": response_block,
    }
    if provider_states:
        interaction["providerStates"] = provider_states
    return interaction


def _index_scenarios(scenarios: List[ScenarioConfig]) -> Dict[str, List[Dict[str, Any]]]:
    """Map endpoint name → list of provider-state entries.

    A scenario's `state.on_match.endpoint` lets us attach the scenario as a
    provider-state precondition to that endpoint's interactions.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for scn in scenarios:
        for state in scn.states:
            ep_name = state.on_match.get("endpoint")
            if not ep_name:
                continue
            entry = {"name": scn.name}
            if state.requires_state:
                entry["params"] = {"required_state": state.requires_state}
            out.setdefault(ep_name, []).append(entry)
    return out


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"
