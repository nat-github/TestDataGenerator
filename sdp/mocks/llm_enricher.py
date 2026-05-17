"""LLM-assisted enrichment for `MockConfig` documents.

Uses the unified `llm/multi_provider.py` so any backend works — hosted
Anthropic / OpenAI / Groq, or local LM Studio / Ollama. The enricher is
purely additive: it never deletes or replaces values the user authored.

Two enrichment modes:

  1. ``fill_missing_examples`` — for each FieldSpec / SchemaConfig that has
     no `example`, ask the LLM to suggest a plausible one.
  2. ``draft_error_responses`` — for each endpoint that lacks 4xx/5xx
     variants, ask the LLM to draft realistic error envelopes.

Both modes round-trip the full MockConfig — they take a config and return
an enriched config. Failures degrade gracefully: if the LLM returns
malformed JSON or no response at all, the config is returned unchanged
with a warning logged.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sdp.llm.multi_provider import chat as llm_chat
from sdp.models.mock_models import (
    EndpointConfig,
    FieldSpec,
    MockConfig,
    ResponseTemplate,
    SchemaConfig,
)

logger = logging.getLogger(__name__)


# Default error-response statuses to draft when none exist.
_DEFAULT_ERROR_STATUSES = (400, 401, 404, 422, 500)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """Diagnostic record returned by ``enrich(...)``."""
    examples_added: int = 0
    error_responses_added: int = 0
    schemas_touched: List[str] = field(default_factory=list)
    endpoints_touched: List[str] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich(
    config: MockConfig,
    *,
    fill_missing_examples: bool = True,
    draft_error_responses: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[MockConfig, EnrichmentResult]:
    """Run the requested enrichment passes against ``config``.

    Returns a *new* MockConfig (the input is not mutated) plus an
    `EnrichmentResult` summarising what changed. LLM provider config is
    threaded through to ``llm.multi_provider.chat``.
    """
    cfg_dict = config.model_dump(mode="python", by_alias=True)
    result = EnrichmentResult()

    if fill_missing_examples:
        _fill_missing_examples(cfg_dict, result, provider, model, base_url, api_key)

    if draft_error_responses:
        _draft_error_responses(cfg_dict, result, provider, model, base_url, api_key)

    enriched = MockConfig.model_validate(cfg_dict)
    return enriched, result


# ---------------------------------------------------------------------------
# Pass 1 — fill missing examples
# ---------------------------------------------------------------------------


_FILL_SYSTEM = (
    "You are a JSON Schema example author. You receive a list of schema "
    "fragments lacking concrete `example` values, and you return realistic "
    "examples for each. Always reply with valid JSON. Be conservative: do "
    "not invent fields, only example values for the fields shown."
)


def _fill_missing_examples(
    cfg_dict: Dict[str, Any],
    result: EnrichmentResult,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """Walk MockConfig schemas; for each schema with no example, request one."""
    schemas = cfg_dict.get("schemas") or {}
    targets: List[Dict[str, Any]] = []
    for name, schema in schemas.items():
        if not isinstance(schema, dict):
            continue
        if schema.get("example") is None:
            targets.append({
                "name": name,
                "schema": _trim_schema_for_prompt(schema),
            })

    if not targets:
        return

    # We batch up to 8 schemas per prompt to keep token use bounded.
    for batch in _chunked(targets, 8):
        prompt = _build_fill_prompt(batch)
        try:
            raw = llm_chat(
                messages=[{"role": "user", "content": prompt}],
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                system=_FILL_SYSTEM,
                max_tokens=2048,
                temperature=0.3,
            )
        except Exception as exc:
            result.warnings.append(f"fill_missing_examples LLM call failed: {exc}")
            continue

        result.raw_responses.append(raw)
        suggestions = _parse_suggestions(raw)
        for suggestion in suggestions:
            schema_name = suggestion.get("name")
            example_value = suggestion.get("example")
            if not schema_name or example_value is None:
                continue
            if schema_name not in schemas:
                continue
            if schemas[schema_name].get("example") is not None:
                continue  # never overwrite
            schemas[schema_name]["example"] = example_value
            result.examples_added += 1
            result.schemas_touched.append(schema_name)


def _build_fill_prompt(batch: List[Dict[str, Any]]) -> str:
    return f"""\
For each schema below, return a realistic concrete `example` object that
satisfies the schema. Reply ONLY with a JSON array of {{"name": ..., "example": ...}}
entries — no prose, no markdown fences.

Schemas:
{json.dumps(batch, indent=2, default=str)}
"""


# ---------------------------------------------------------------------------
# Pass 2 — draft error responses
# ---------------------------------------------------------------------------


_ERROR_SYSTEM = (
    "You are an API design assistant. You receive an endpoint description "
    "and you return a JSON list of plausible error responses (status code "
    "+ JSON body). Use realistic error codes that match the endpoint's "
    "domain (e.g. NOT_FOUND for 404, VALIDATION_FAILED for 422)."
)


def _draft_error_responses(
    cfg_dict: Dict[str, Any],
    result: EnrichmentResult,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """For each endpoint missing 4xx/5xx variants, ask the LLM to draft them."""
    endpoints = cfg_dict.get("endpoints") or []
    for endpoint in endpoints:
        existing_statuses = {
            int(r.get("status", 200)) for r in (endpoint.get("responses") or [])
        }
        missing = [s for s in _DEFAULT_ERROR_STATUSES if s not in existing_statuses]
        if not missing:
            continue

        prompt = _build_error_prompt(endpoint, missing)
        try:
            raw = llm_chat(
                messages=[{"role": "user", "content": prompt}],
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                system=_ERROR_SYSTEM,
                max_tokens=1500,
                temperature=0.3,
            )
        except Exception as exc:
            result.warnings.append(
                f"draft_error_responses LLM call failed for "
                f"{endpoint.get('name', '?')}: {exc}"
            )
            continue

        result.raw_responses.append(raw)
        suggested = _parse_suggestions(raw)
        for entry in suggested:
            try:
                status = int(entry.get("status"))
            except (TypeError, ValueError):
                continue
            if status < 400 or status >= 600:
                continue
            if status in existing_statuses:
                continue
            body = entry.get("body")
            if body is None:
                continue
            new_response: Dict[str, Any] = {
                "status": status,
                "body_content_type": "application/json",
                "body_schema": _body_to_field_spec(body),
                "weight": 0.05,
                "description": entry.get("description") or f"Synthesised {status} response",
            }
            endpoint["responses"].append(new_response)
            existing_statuses.add(status)
            result.error_responses_added += 1
            result.endpoints_touched.append(endpoint.get("name", "?"))


def _build_error_prompt(endpoint: Dict[str, Any], missing: List[int]) -> str:
    return f"""\
Endpoint:
  name:        {endpoint.get('name')}
  method:      {endpoint.get('method')}
  path:        {endpoint.get('path')}
  summary:     {endpoint.get('summary') or ''}
  description: {endpoint.get('description') or ''}

Existing responses: {[r.get('status') for r in endpoint.get('responses') or []]}
Missing statuses to draft: {missing}

Return ONLY a JSON array of entries, one per status code, shaped like:
  [{{"status": 404, "body": {{"code": "NOT_FOUND", "message": "..."}}, "description": "..."}}, ...]
Do NOT include status codes that are already declared. Skip any status that
genuinely doesn't make sense for this endpoint (e.g. 422 on a GET).
No prose, no markdown fences.
"""


def _body_to_field_spec(body: Any) -> Dict[str, Any]:
    """Convert a concrete JSON body into a FieldSpec-compatible dict.

    We don't try to infer JSON Schema from the body — instead we pin the
    body as a literal `example` so the renderer reproduces it verbatim.
    That's the cheapest correct thing.
    """
    if isinstance(body, dict):
        return {"type": "object", "example": body}
    if isinstance(body, list):
        return {"type": "array", "items": {"type": "object"}, "example": body}
    return {"type": "string", "example": str(body)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trim_schema_for_prompt(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop noisy keys before sending to the LLM."""
    keep = ("type", "properties", "required", "items", "description", "enum",
            "format", "minimum", "maximum", "pattern")
    return {k: v for k, v in schema.items() if k in keep}


def _chunked(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_suggestions(raw: str) -> List[Dict[str, Any]]:
    """Strip markdown fences and parse a JSON array, tolerating loose output."""
    if not raw:
        return []
    cleaned = _FENCE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models prefix with prose; try to grab the first array
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            logger.warning("LLM returned no JSON array; raw=%r", raw[:300])
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse suggestions: %s", exc)
            return []
    if isinstance(parsed, dict):
        # Some models wrap the array in {"suggestions": [...]}
        for key in ("suggestions", "results", "items", "data"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        return []
    if isinstance(parsed, list):
        return parsed
    return []
