"""Tests for `mocks/llm_enricher.py`.

LLM calls are mocked via monkeypatching `llm.multi_provider.chat`.
Tests cover both enrichment passes plus failure modes (malformed JSON,
LLM call exceptions, empty responses).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from mocks import llm_enricher
from mocks.llm_enricher import enrich
from models.mock_models import (
    EndpointConfig,
    FieldSpec,
    MockConfig,
    ResponseTemplate,
    SchemaConfig,
)


# ---------------------------------------------------------------------------
# Fixture: a tiny MockConfig with deliberate gaps for the enricher to fill
# ---------------------------------------------------------------------------


def _config_with_gaps() -> MockConfig:
    return MockConfig(
        schemas={
            "Account": SchemaConfig(
                type="object",
                required=["iban", "currency"],
                properties={
                    "iban": FieldSpec(type="string"),
                    "currency": FieldSpec(type="string"),
                },
                # No example — enricher should fill this in.
            ),
            "Customer": SchemaConfig(
                type="object",
                properties={
                    "id": FieldSpec(type="string", format="uuid"),
                    "name": FieldSpec(type="string"),
                },
                # No example — also missing.
            ),
        },
        endpoints=[
            EndpointConfig(
                name="get_account",
                path="/accounts/{iban}",
                method="GET",
                request={"path_params": {"iban": FieldSpec(type="string")}},
                responses=[
                    ResponseTemplate(status=200, body_schema=FieldSpec(ref="Account")),
                    # Note: no error responses — enricher should draft some.
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------


class _ChatRecorder:
    """Records every llm_chat invocation and returns scripted responses."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return ""
        return self._responses.pop(0)


@pytest.fixture
def fake_chat(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake llm_chat — call .install(responses) per test."""
    holder = {"recorder": None}

    def install(responses: List[str]) -> _ChatRecorder:
        recorder = _ChatRecorder(responses)
        monkeypatch.setattr(llm_enricher, "llm_chat", recorder)
        holder["recorder"] = recorder
        return recorder

    return install


# ---------------------------------------------------------------------------
# Pass 1 — fill_missing_examples
# ---------------------------------------------------------------------------


def test_fill_missing_examples_populates_schema_example(fake_chat):
    cfg = _config_with_gaps()
    suggestion = json.dumps([
        {"name": "Account", "example": {"iban": "DE89370400440532013000", "currency": "EUR"}},
        {"name": "Customer", "example": {"id": "7c0a3b9a-…", "name": "Acme"}},
    ])
    fake_chat([suggestion])

    enriched, result = enrich(
        cfg,
        fill_missing_examples=True,
        draft_error_responses=False,
    )
    assert result.examples_added == 2
    assert "Account" in result.schemas_touched
    assert enriched.schemas["Account"].example["iban"].startswith("DE89")
    assert enriched.schemas["Customer"].example["name"] == "Acme"


def test_fill_does_not_overwrite_authored_examples(fake_chat):
    cfg = _config_with_gaps()
    cfg.schemas["Account"].example = {"iban": "AUTHORED"}
    suggestion = json.dumps([
        {"name": "Account", "example": {"iban": "FROM-LLM"}},
        {"name": "Customer", "example": {"id": "abc", "name": "x"}},
    ])
    fake_chat([suggestion])

    enriched, result = enrich(
        cfg,
        fill_missing_examples=True,
        draft_error_responses=False,
    )
    # Account had an authored example — must be preserved
    assert enriched.schemas["Account"].example == {"iban": "AUTHORED"}
    # Customer was missing — gets the LLM-suggested value
    assert enriched.schemas["Customer"].example == {"id": "abc", "name": "x"}
    assert result.examples_added == 1


def test_fill_handles_markdown_fenced_response(fake_chat):
    """Some local models wrap JSON in ``` fences — must be stripped."""
    cfg = _config_with_gaps()
    fenced = (
        "```json\n"
        + json.dumps([{"name": "Account", "example": {"iban": "x", "currency": "EUR"}}])
        + "\n```"
    )
    fake_chat([fenced])

    enriched, _ = enrich(cfg, fill_missing_examples=True, draft_error_responses=False)
    assert enriched.schemas["Account"].example == {"iban": "x", "currency": "EUR"}


def test_fill_tolerates_invalid_json_response(fake_chat):
    cfg = _config_with_gaps()
    fake_chat(["this is not JSON at all"])
    enriched, result = enrich(cfg, fill_missing_examples=True, draft_error_responses=False)
    # Nothing applied; config returned unchanged
    assert result.examples_added == 0
    assert enriched.schemas["Account"].example is None


def test_fill_tolerates_llm_exception(fake_chat, monkeypatch):
    cfg = _config_with_gaps()

    def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_enricher, "llm_chat", boom)

    enriched, result = enrich(cfg, fill_missing_examples=True, draft_error_responses=False)
    assert result.examples_added == 0
    assert any("LLM call failed" in w for w in result.warnings)
    # Config preserved
    assert enriched.schemas["Account"].example is None


def test_fill_unwraps_dict_with_suggestions_key(fake_chat):
    """Some models reply with `{"suggestions": [...]}` instead of a bare array."""
    cfg = _config_with_gaps()
    wrapped = json.dumps({
        "suggestions": [
            {"name": "Account", "example": {"iban": "x", "currency": "USD"}}
        ]
    })
    fake_chat([wrapped])

    enriched, _ = enrich(cfg, fill_missing_examples=True, draft_error_responses=False)
    assert enriched.schemas["Account"].example == {"iban": "x", "currency": "USD"}


def test_fill_skips_when_all_schemas_have_examples(fake_chat):
    cfg = _config_with_gaps()
    cfg.schemas["Account"].example = {"iban": "a", "currency": "b"}
    cfg.schemas["Customer"].example = {"id": "c", "name": "d"}
    recorder = fake_chat([])

    enriched, result = enrich(cfg, fill_missing_examples=True, draft_error_responses=False)
    # No LLM calls when nothing's missing
    assert recorder.calls == []
    assert result.examples_added == 0


# ---------------------------------------------------------------------------
# Pass 2 — draft_error_responses
# ---------------------------------------------------------------------------


def test_draft_error_responses_adds_4xx_variants(fake_chat):
    cfg = _config_with_gaps()
    suggestion = json.dumps([
        {"status": 404, "body": {"code": "NOT_FOUND", "message": "Account does not exist"}},
        {"status": 401, "body": {"code": "UNAUTHORIZED", "message": "Missing token"}},
    ])
    fake_chat([suggestion])

    enriched, result = enrich(cfg, fill_missing_examples=False, draft_error_responses=True)
    statuses = {r.status for r in enriched.endpoints[0].responses}
    assert 404 in statuses
    assert 401 in statuses
    assert result.error_responses_added == 2


def test_draft_does_not_duplicate_existing_status_codes(fake_chat):
    """If the LLM suggests a status that's already declared, skip it."""
    cfg = _config_with_gaps()
    cfg.endpoints[0].responses.append(
        ResponseTemplate(status=404, body_schema=FieldSpec(type="object"))
    )
    suggestion = json.dumps([
        {"status": 404, "body": {"code": "NOT_FOUND"}},
        {"status": 401, "body": {"code": "UNAUTHORIZED"}},
    ])
    fake_chat([suggestion])

    enriched, result = enrich(cfg, fill_missing_examples=False, draft_error_responses=True)
    statuses = [r.status for r in enriched.endpoints[0].responses]
    # 404 remains a single entry; 401 added
    assert statuses.count(404) == 1
    assert 401 in statuses
    assert result.error_responses_added == 1


def test_draft_rejects_non_4xx_5xx_suggestions(fake_chat):
    cfg = _config_with_gaps()
    suggestion = json.dumps([
        {"status": 200, "body": {"a": "b"}},   # ignored — 2xx is not "error"
        {"status": 599, "body": {"c": "d"}},   # accepted — within [400, 599]
        {"status": 999, "body": {"e": "f"}},   # ignored — out of valid HTTP range
    ])
    fake_chat([suggestion])

    enriched, result = enrich(cfg, fill_missing_examples=False, draft_error_responses=True)
    new_statuses = {r.status for r in enriched.endpoints[0].responses}
    assert 599 in new_statuses
    assert 999 not in new_statuses
    assert 200 in new_statuses  # the original 200 is still there
    # Only the 599 entry was added
    assert result.error_responses_added == 1


def test_draft_skips_endpoints_with_full_error_set(fake_chat):
    """An endpoint already declaring all default error statuses should not call the LLM."""
    cfg = _config_with_gaps()
    for status in (400, 401, 404, 422, 500):
        cfg.endpoints[0].responses.append(
            ResponseTemplate(status=status, body_schema=FieldSpec(type="object"))
        )
    recorder = fake_chat([])

    enriched, result = enrich(cfg, fill_missing_examples=False, draft_error_responses=True)
    assert recorder.calls == []
    assert result.error_responses_added == 0


# ---------------------------------------------------------------------------
# Combined run — both passes
# ---------------------------------------------------------------------------


def test_enrich_runs_both_passes_and_returns_full_result(fake_chat):
    cfg = _config_with_gaps()
    fill = json.dumps([
        {"name": "Account", "example": {"iban": "x", "currency": "EUR"}},
        {"name": "Customer", "example": {"id": "y", "name": "z"}},
    ])
    errors = json.dumps([
        {"status": 404, "body": {"code": "NOT_FOUND"}},
    ])
    fake_chat([fill, errors])

    enriched, result = enrich(cfg)
    assert result.examples_added == 2
    assert result.error_responses_added == 1
    assert "Account" in result.schemas_touched
    assert "get_account" in result.endpoints_touched
    # Original config left untouched
    assert cfg.schemas["Account"].example is None


def test_provider_kwargs_threaded_through(fake_chat):
    cfg = _config_with_gaps()
    recorder = fake_chat([json.dumps([])])

    enrich(
        cfg,
        fill_missing_examples=True,
        draft_error_responses=False,
        provider="lm-studio",
        model="local-model",
        base_url="http://localhost:1234/v1",
        api_key=None,
    )
    assert recorder.calls
    call = recorder.calls[0]
    assert call["provider"] == "lm-studio"
    assert call["model"] == "local-model"
    assert call["base_url"] == "http://localhost:1234/v1"
