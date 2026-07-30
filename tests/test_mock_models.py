"""Tests for `models/mock_models.py` and `mocks/config_parser.py`.

Phase A coverage:
  - Pydantic shape validation (FieldSpec, SchemaConfig, RequestMatcher,
    ResponseTemplate, EndpointConfig, MockConfig)
  - YAML and JSON round-trip via load_mock_config_str + dump_mock_config
  - $ref resolution (named refs into MockConfig.schemas must exist)
  - Cross-field validators (mutually exclusive body_schema/body_template,
    HTTP method whitelist, path-must-start-with-slash, status code range,
    duplicate endpoint names, required[] subset of properties)
  - lint_mock_config soft warnings
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sdp.mocks.config_parser import (
    MockConfigError,
    dump_mock_config,
    lint_mock_config,
    load_mock_config,
    load_mock_config_str,
)
from sdp.models.mock_models import (
    EndpointConfig,
    FieldSpec,
    RequestMatcher,
    ResponseTemplate,
    SchemaConfig,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid documents
# ---------------------------------------------------------------------------


def _minimal_yaml() -> str:
    return """
config_format: sdp-mock-v1
info: {title: Demo, version: 0.1.0}
schemas:
  Account:
    type: object
    required: [iban]
    properties:
      iban: {type: string, special_rule: IBAN}
      currency: {type: string, business_values: [EUR, USD, INR]}
endpoints:
  - name: get_account
    path: /accounts/{iban}
    method: GET
    request:
      path_params:
        iban: {type: string, special_rule: IBAN}
    responses:
      - status: 200
        body_schema: {ref: Account}
      - status: 404
        weight: 0.1
        body_schema:
          type: object
          properties:
            code: {type: string, business_values: [NOT_FOUND]}
"""


def _minimal_dict() -> dict:
    return yaml.safe_load(_minimal_yaml())


# ---------------------------------------------------------------------------
# FieldSpec — leaf validation
# ---------------------------------------------------------------------------


def test_field_spec_array_requires_items():
    with pytest.raises(ValueError, match="array FieldSpec must declare `items`"):
        FieldSpec(type="array")


def test_field_spec_ref_excludes_inline_shape():
    with pytest.raises(ValueError, match="cannot mix `ref`"):
        FieldSpec(ref="Account", type="object", properties={"x": FieldSpec(type="string")})


def test_field_spec_null_rate_implies_nullable():
    f = FieldSpec(type="string", null_rate=0.1)
    assert f.nullable is True


def test_field_spec_null_rate_range():
    with pytest.raises(ValueError, match="null_rate"):
        FieldSpec(type="string", null_rate=1.5)


def test_field_spec_business_values_accepted():
    f = FieldSpec(type="string", business_values=["EUR", "USD"])
    assert f.business_values == ["EUR", "USD"]


# ---------------------------------------------------------------------------
# SchemaConfig — composition + required[] consistency
# ---------------------------------------------------------------------------


def test_schema_required_must_be_subset_of_properties():
    with pytest.raises(ValueError, match="references unknown properties"):
        SchemaConfig(
            type="object",
            properties={"a": FieldSpec(type="string")},
            required=["a", "b"],
        )


def test_schema_array_needs_items():
    with pytest.raises(ValueError, match="array SchemaConfig must declare `items`"):
        SchemaConfig(type="array")


def test_schema_oneOf_alias_accepted():
    """Authors using OpenAPI vocabulary should still load — `oneOf` aliases `one_of`."""
    s = SchemaConfig.model_validate({"type": "object", "oneOf": [
        {"type": "object", "properties": {"a": {"type": "string"}}},
    ]})
    assert len(s.one_of) == 1


# ---------------------------------------------------------------------------
# RequestMatcher / ResponseTemplate
# ---------------------------------------------------------------------------


def test_request_body_match_mode_defaults_to_ignore_when_no_body():
    rm = RequestMatcher()
    assert rm.body_match_mode == "ignore"


def test_response_rejects_both_body_schema_and_template():
    with pytest.raises(ValueError, match="cannot have both body_schema and body_template"):
        ResponseTemplate(
            status=200,
            body_schema=FieldSpec(type="object"),
            body_template="<xml/>",
        )


def test_response_rejects_invalid_status_code():
    with pytest.raises(ValueError, match=r"status must be in \[100, 599\]"):
        ResponseTemplate(status=999)


def test_response_weight_must_be_non_negative():
    with pytest.raises(ValueError, match="weight"):
        ResponseTemplate(status=200, weight=-0.1)


# ---------------------------------------------------------------------------
# EndpointConfig
# ---------------------------------------------------------------------------


def test_endpoint_path_must_start_with_slash():
    with pytest.raises(ValueError, match="path must start with"):
        EndpointConfig(
            name="x", path="accounts", method="GET",
            responses=[ResponseTemplate(status=200)],
        )


def test_endpoint_must_have_at_least_one_response():
    with pytest.raises(ValueError, match="must declare at least one"):
        EndpointConfig(name="x", path="/x", method="GET", responses=[])


def test_endpoint_rejects_unknown_method():
    with pytest.raises(ValueError):
        EndpointConfig(
            name="x", path="/x", method="FETCH",  # type: ignore[arg-type]
            responses=[ResponseTemplate(status=200)],
        )


# ---------------------------------------------------------------------------
# MockConfig — top-level cross-validation
# ---------------------------------------------------------------------------


def test_minimal_mock_config_round_trips_yaml(tmp_path: Path):
    cfg = load_mock_config_str(_minimal_yaml())
    out = tmp_path / "out.yaml"
    dump_mock_config(cfg, out)
    cfg2 = load_mock_config(out)
    assert cfg2.endpoints[0].name == "get_account"
    assert "Account" in cfg2.schemas


def test_minimal_mock_config_round_trips_json(tmp_path: Path):
    cfg = load_mock_config_str(_minimal_yaml())
    out = tmp_path / "out.json"
    dump_mock_config(cfg, out)
    raw = out.read_text(encoding="utf-8")
    parsed = json.loads(raw)  # confirm it's actually JSON, not YAML
    assert parsed["config_format"] == "sdp-mock-v1"
    cfg2 = load_mock_config(out)
    assert cfg2.endpoints[0].name == "get_account"


def test_unresolved_ref_is_rejected():
    bad = _minimal_dict()
    bad["endpoints"][0]["responses"][0]["body_schema"] = {"ref": "DoesNotExist"}
    with pytest.raises(MockConfigError, match="unresolved \\$refs"):
        load_mock_config_str(yaml.safe_dump(bad))


def test_duplicate_endpoint_names_rejected():
    bad = _minimal_dict()
    dup = json.loads(json.dumps(bad["endpoints"][0]))  # deep copy
    bad["endpoints"].append(dup)
    with pytest.raises(MockConfigError, match="duplicate endpoint name"):
        load_mock_config_str(yaml.safe_dump(bad))


def test_unsupported_config_format_rejected():
    bad = _minimal_dict()
    bad["config_format"] = "something-else"
    with pytest.raises(MockConfigError, match="unsupported config_format"):
        load_mock_config_str(yaml.safe_dump(bad))


def test_empty_document_rejected():
    with pytest.raises(MockConfigError, match="empty"):
        load_mock_config_str("")


def test_top_level_must_be_mapping():
    with pytest.raises(MockConfigError, match="must be a mapping"):
        load_mock_config_str("- a list at the top")


# ---------------------------------------------------------------------------
# $ref walking — nested across path_params, query_params, body schemas, items
# ---------------------------------------------------------------------------


def test_ref_inside_array_items_is_resolved():
    """An array body whose items use a ref should still validate."""
    yaml_doc = """
config_format: sdp-mock-v1
schemas:
  Item:
    type: object
    properties:
      sku: {type: string}
endpoints:
  - name: list_items
    path: /items
    method: GET
    responses:
      - status: 200
        body_schema:
          type: array
          items: {ref: Item}
"""
    cfg = load_mock_config_str(yaml_doc)
    assert cfg.endpoints[0].responses[0].body_schema.items.ref == "Item"


def test_ref_inside_query_param_is_resolved():
    yaml_doc = """
config_format: sdp-mock-v1
schemas:
  Filter:
    type: object
    properties:
      q: {type: string}
endpoints:
  - name: search
    path: /search
    method: GET
    request:
      query_params:
        filter: {ref: Filter}
    responses:
      - status: 200
        body_schema: {type: object}
"""
    cfg = load_mock_config_str(yaml_doc)
    assert cfg.endpoints[0].request.query_params["filter"].ref == "Filter"


# ---------------------------------------------------------------------------
# Lint reports
# ---------------------------------------------------------------------------


def test_lint_returns_ok_on_minimal_doc(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text(_minimal_yaml(), encoding="utf-8")
    report = lint_mock_config(p)
    assert report["ok"] is True
    assert report["endpoints"] == 1
    assert report["schemas"] == 1


def test_lint_warns_on_missing_2xx(tmp_path: Path):
    """A document with only 4xx responses should lint with a warning."""
    p = tmp_path / "m.yaml"
    p.write_text(
        """
config_format: sdp-mock-v1
endpoints:
  - name: x
    path: /x
    method: GET
    responses:
      - status: 404
        body_schema: {type: object}
""",
        encoding="utf-8",
    )
    report = lint_mock_config(p)
    assert report["ok"] is True
    assert any("no 2xx" in w for w in report["warnings"])


def test_lint_collects_errors_without_raising(tmp_path: Path):
    """Lint should never propagate the validation exception."""
    p = tmp_path / "broken.yaml"
    p.write_text("config_format: sdp-mock-v1\nendpoints: [{name: x, path: x, method: GET, responses: []}]",
                 encoding="utf-8")
    report = lint_mock_config(p)
    assert report["ok"] is False
    assert report["errors"]


def test_lint_handles_missing_file():
    report = lint_mock_config("does/not/exist.yaml")
    assert report["ok"] is False
    assert any("not found" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Convenience lookups on MockConfig
# ---------------------------------------------------------------------------


def test_get_endpoint_and_schema_lookups():
    cfg = load_mock_config_str(_minimal_yaml())
    assert cfg.get_endpoint("get_account") is not None
    assert cfg.get_endpoint("nope") is None
    assert cfg.get_schema("Account") is not None
    assert cfg.get_schema("nope") is None


def test_resolve_ref_returns_target_schema():
    cfg = load_mock_config_str(_minimal_yaml())
    field = cfg.endpoints[0].responses[0].body_schema
    target = cfg.resolve_ref(field)
    assert target is not None
    assert "iban" in target.properties
