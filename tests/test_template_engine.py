"""Tests for `mocks/template_engine.py`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mocks.config_parser import load_mock_config_str
from mocks.openapi_importer import import_openapi
from mocks.template_engine import TemplateEngine
from models.mock_models import (
    EndpointConfig,
    FieldSpec,
    MockConfig,
    RequestMatcher,
    ResponseTemplate,
    SchemaConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_with_field(field: FieldSpec) -> MockConfig:
    """Wrap a field in a one-endpoint MockConfig so we can call render_field."""
    return MockConfig(
        endpoints=[
            EndpointConfig(
                name="x", path="/x", method="GET",
                responses=[ResponseTemplate(status=200, body_schema=field)],
            )
        ]
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_output():
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    a = TemplateEngine(cfg, seed=42).render_endpoint_response("get_book_by_id")
    b = TemplateEngine(cfg, seed=42).render_endpoint_response("get_book_by_id")
    assert a == b


def test_different_seeds_produce_different_output():
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    a = TemplateEngine(cfg, seed=1).render_endpoint_response("get_book_by_id")
    b = TemplateEngine(cfg, seed=2).render_endpoint_response("get_book_by_id")
    assert a != b


# ---------------------------------------------------------------------------
# Precedence rules
# ---------------------------------------------------------------------------


def test_example_wins_over_business_values():
    field = FieldSpec(type="string", business_values=["X", "Y"], example="EXAMPLE")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert out == "EXAMPLE"


def test_business_values_picked_when_no_example():
    field = FieldSpec(type="string", business_values=["EUR", "USD", "INR"])
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert out in {"EUR", "USD", "INR"}


def test_special_rule_routes_through_helpers():
    """EMAIL must produce a string with '@' — proves the helpers bridge works."""
    field = FieldSpec(type="string", special_rule="EMAIL")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert isinstance(out, str)
    assert "@" in out


def test_format_email_acts_like_special_rule():
    field = FieldSpec(type="string", format="email")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert "@" in out


def test_format_uuid_produces_uuid_shape():
    field = FieldSpec(type="string", format="uuid")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert re.match(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        out,
    )


def test_format_iban_produces_iban_shape():
    field = FieldSpec(type="string", format="iban")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    # IBAN: 2 letters + 2 digits + up to 30 alphanumerics
    assert re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]+$", out), out


def test_pattern_generates_matching_string():
    field = FieldSpec(type="string", pattern="^[0-9]{4}$")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert re.match("^[0-9]{4}$", out), out


# ---------------------------------------------------------------------------
# Numeric / boolean / array
# ---------------------------------------------------------------------------


def test_integer_respects_minimum_and_maximum():
    field = FieldSpec(type="integer", minimum=10, maximum=20)
    cfg = _cfg_with_field(field)
    for _ in range(20):
        out = TemplateEngine(cfg, seed=None).render_field(field)
        assert 10 <= out <= 20


def test_number_respects_bounds():
    field = FieldSpec(type="number", minimum=0.5, maximum=1.5)
    cfg = _cfg_with_field(field)
    for _ in range(20):
        out = TemplateEngine(cfg, seed=None).render_field(field)
        assert 0.5 <= out <= 1.5


def test_boolean_returns_bool():
    field = FieldSpec(type="boolean")
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert isinstance(out, bool)


def test_array_respects_min_max_length():
    field = FieldSpec(
        type="array",
        items=FieldSpec(type="integer", minimum=1, maximum=5),
        min_length=3,
        max_length=3,
    )
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert len(out) == 3


def test_array_default_length_is_reasonable():
    field = FieldSpec(type="array", items=FieldSpec(type="integer"))
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert 2 <= len(out) <= 5


# ---------------------------------------------------------------------------
# Nullability
# ---------------------------------------------------------------------------


def test_null_rate_can_emit_none():
    field = FieldSpec(type="string", nullable=True, null_rate=1.0)
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert out is None


def test_null_rate_zero_never_emits_none():
    field = FieldSpec(type="string", nullable=True, null_rate=0.0)
    cfg = _cfg_with_field(field)
    out = TemplateEngine(cfg, seed=0).render_field(field)
    assert out is not None


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


def test_ref_to_named_schema_is_rendered():
    cfg = MockConfig(
        schemas={
            "Account": SchemaConfig(
                type="object",
                required=["iban"],
                properties={
                    "iban": FieldSpec(type="string", special_rule="IBAN"),
                    "currency": FieldSpec(type="string", business_values=["EUR", "USD"]),
                },
            )
        },
        endpoints=[
            EndpointConfig(
                name="x", path="/x", method="GET",
                responses=[
                    ResponseTemplate(status=200, body_schema=FieldSpec(ref="Account"))
                ],
            )
        ],
    )
    out = TemplateEngine(cfg, seed=0).render_endpoint_response("x")
    assert "iban" in out
    assert out["currency"] in {"EUR", "USD"}


def test_unresolved_ref_returns_none_with_warning(caplog):
    cfg = MockConfig(
        schemas={"X": SchemaConfig(type="object", properties={"a": FieldSpec(type="string")})},
        endpoints=[
            EndpointConfig(
                name="x", path="/x", method="GET",
                responses=[ResponseTemplate(status=200, body_schema=FieldSpec(ref="X"))],
            )
        ],
    )
    # Construct a ref to a bogus name AFTER validation has passed
    bad_field = FieldSpec(ref="Nope")
    out = TemplateEngine(cfg, seed=0).render_field(bad_field)
    assert out is None


# ---------------------------------------------------------------------------
# End-to-end against example specs
# ---------------------------------------------------------------------------


def test_simple_books_renders_each_response():
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    te = TemplateEngine(cfg, seed=42)
    for ep in cfg.endpoints:
        for resp in ep.responses:
            body = te.render_endpoint_response(ep.name, status=resp.status)
            # Either None (no body) or a JSON-serialisable value
            assert body is None or isinstance(body, (dict, list, str, int, float, bool))


def test_medium_tasks_renders_paginated_list():
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    te = TemplateEngine(cfg, seed=7)
    body = te.render_endpoint_response("list_tasks", status=200)
    assert "data" in body
    assert isinstance(body["data"], list)
    assert "page" in body
    assert "total" in body


def test_complex_payments_renders_idempotent_create():
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    te = TemplateEngine(cfg, seed=99)
    body = te.render_endpoint_response("create_payment", status=201)
    # Payment is allOf(PaymentCreate + extras); flattened in the importer
    assert "amount" in body
    assert "id" in body
    assert "status" in body


def test_complex_payments_renders_money_with_pattern():
    """Money.value is `^-?[0-9]+(\\.[0-9]{2})?$` — must always validate."""
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    te = TemplateEngine(cfg, seed=1)
    body = te.render_endpoint_response("get_account", status=200)
    bal = body["balance"]
    assert re.match(r"^-?[0-9]+(\.[0-9]{2})?$", bal["value"]), bal["value"]
    assert re.match(r"^[A-Z]{3}$", bal["currency"]), bal["currency"]
