"""Tests for `mocks/openapi_importer.py`.

Exercises the importer against three real example specs (simple/medium/
complex) plus targeted unit tests for individual conversion paths
($ref, allOf, oneOf, parameters, response headers, format hints).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mocks.openapi_importer import (
    OpenAPIImportError,
    import_openapi,
    import_openapi_str,
)
from models.mock_models import MockConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples" / "openapi"


# ---------------------------------------------------------------------------
# Targeted unit tests
# ---------------------------------------------------------------------------


def test_rejects_non_openapi_doc():
    with pytest.raises(OpenAPIImportError, match="missing"):
        import_openapi_str("title: not an openapi doc\nversion: 1.0.0")


def test_rejects_top_level_list():
    with pytest.raises(OpenAPIImportError, match="must be a mapping"):
        import_openapi_str("- a list\n- of items")


def test_format_email_maps_to_special_rule():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: getUser
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                type: object
                properties:
                  email: {type: string, format: email}
"""
    cfg = import_openapi_str(spec)
    body = cfg.endpoints[0].responses[0].body_schema
    assert body.properties["email"].special_rule == "EMAIL"
    assert body.properties["email"].format == "email"


def test_format_iban_maps_to_special_rule():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /a:
    get:
      operationId: getAccount
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                type: object
                properties:
                  iban: {type: string, format: iban}
"""
    cfg = import_openapi_str(spec)
    assert cfg.endpoints[0].responses[0].body_schema.properties["iban"].special_rule == "IBAN"


def test_enum_becomes_business_values():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /s:
    get:
      operationId: getStatus
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                type: object
                properties:
                  status: {type: string, enum: [active, frozen, closed]}
"""
    cfg = import_openapi_str(spec)
    field = cfg.endpoints[0].responses[0].body_schema.properties["status"]
    assert field.business_values == ["active", "frozen", "closed"]


def test_ref_to_components_schemas_is_resolved():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /a:
    get:
      operationId: getA
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Item"
components:
  schemas:
    Item:
      type: object
      properties:
        id: {type: integer}
"""
    cfg = import_openapi_str(spec)
    body = cfg.endpoints[0].responses[0].body_schema
    assert body.ref == "Item"
    assert "Item" in cfg.schemas


def test_parameters_are_split_by_location():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u/{id}:
    get:
      operationId: getU
      parameters:
        - {in: path, name: id, required: true, schema: {type: string}}
        - {in: query, name: include, schema: {type: string}}
        - {in: header, name: X-Trace, schema: {type: string}}
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema: {type: object}
"""
    cfg = import_openapi_str(spec)
    req = cfg.endpoints[0].request
    assert "id" in req.path_params
    assert "include" in req.query_params
    assert "X-Trace" in req.headers


def test_path_level_parameters_inherited_by_operations():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u/{id}:
    parameters:
      - {in: path, name: id, required: true, schema: {type: string, format: uuid}}
    get:
      operationId: getU
      responses:
        "200":
          description: ok
          content: {application/json: {schema: {type: object}}}
    delete:
      operationId: deleteU
      responses:
        "204":
          description: gone
"""
    cfg = import_openapi_str(spec)
    for ep in cfg.endpoints:
        assert "id" in ep.request.path_params, f"{ep.name} did not inherit path-level id param"


def test_parameter_ref_is_resolved():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: listU
      parameters:
        - $ref: "#/components/parameters/PageParam"
      responses:
        "200":
          description: ok
          content: {application/json: {schema: {type: object}}}
components:
  parameters:
    PageParam:
      in: query
      name: page
      schema: {type: integer, minimum: 1}
"""
    cfg = import_openapi_str(spec)
    assert "page" in cfg.endpoints[0].request.query_params


def test_response_ref_is_resolved():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: getU
      responses:
        "200":
          description: ok
          content: {application/json: {schema: {type: object}}}
        "404":
          $ref: "#/components/responses/NotFound"
components:
  responses:
    NotFound:
      description: not found
      content:
        application/json:
          schema:
            type: object
            properties:
              code: {type: string}
"""
    cfg = import_openapi_str(spec)
    statuses = [r.status for r in cfg.endpoints[0].responses]
    assert 404 in statuses


def test_allOf_is_flattened_into_one_object():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u/{id}:
    get:
      operationId: getU
      parameters:
        - {in: path, name: id, required: true, schema: {type: string}}
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/CustomerWithKyc"
components:
  schemas:
    Customer:
      type: object
      required: [id, name]
      properties:
        id: {type: string, format: uuid}
        name: {type: string}
    KycInfo:
      type: object
      required: [status]
      properties:
        status: {type: string, enum: [verified, pending]}
    CustomerWithKyc:
      allOf:
        - $ref: "#/components/schemas/Customer"
        - $ref: "#/components/schemas/KycInfo"
"""
    cfg = import_openapi_str(spec)
    merged = cfg.schemas["CustomerWithKyc"]
    assert "id" in merged.properties
    assert "name" in merged.properties
    assert "status" in merged.properties
    assert "id" in merged.required
    assert "status" in merged.required


def test_oneOf_picks_first_variant():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /e:
    get:
      operationId: getEvent
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema:
                oneOf:
                  - {type: object, properties: {kind: {type: string, enum: [a]}}}
                  - {type: object, properties: {kind: {type: string, enum: [b]}}}
"""
    cfg = import_openapi_str(spec)
    body = cfg.endpoints[0].responses[0].body_schema
    assert "kind" in body.properties
    assert body.properties["kind"].business_values == ["a"]


def test_response_headers_imported():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: listU
      responses:
        "200":
          description: ok
          headers:
            X-Total-Count:
              schema: {type: integer}
              example: 42
            X-RateLimit-Remaining:
              schema: {type: integer}
          content: {application/json: {schema: {type: object}}}
"""
    cfg = import_openapi_str(spec)
    headers = cfg.endpoints[0].responses[0].headers
    assert "X-Total-Count" in headers
    assert headers["X-Total-Count"].example == 42


def test_responses_sorted_2xx_first():
    """Renderers should see 2xx before 4xx so the happy path is the first variant."""
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: getU
      responses:
        "404": {description: not found, content: {application/json: {schema: {type: object}}}}
        "200": {description: ok, content: {application/json: {schema: {type: object}}}}
        "500": {description: oops, content: {application/json: {schema: {type: object}}}}
"""
    cfg = import_openapi_str(spec)
    statuses = [r.status for r in cfg.endpoints[0].responses]
    assert statuses == [200, 404, 500]


def test_examples_block_pulls_first_value_through():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: listU
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema: {type: object, properties: {id: {type: integer}}}
              examples:
                first:
                  value: {id: 1}
                second:
                  value: {id: 2}
"""
    cfg = import_openapi_str(spec)
    body = cfg.endpoints[0].responses[0].body_schema
    assert body.examples == [{"id": 1}, {"id": 2}]
    assert body.example == {"id": 1}


def test_endpoint_without_operationId_gets_synthetic_name():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /things/{id}/sub:
    get:
      responses:
        "200": {description: ok, content: {application/json: {schema: {type: object}}}}
"""
    cfg = import_openapi_str(spec)
    assert cfg.endpoints[0].name  # any non-empty string
    assert cfg.endpoints[0].name.startswith("get_")


def test_synthetic_2xx_added_when_no_responses_declared():
    spec = """
openapi: 3.0.3
info: {title: x, version: 1.0.0}
paths:
  /u:
    get:
      operationId: getU
      responses: {}
"""
    cfg = import_openapi_str(spec)
    assert cfg.endpoints[0].responses
    assert cfg.endpoints[0].responses[0].status == 200


# ---------------------------------------------------------------------------
# Integration: import each example spec end to end
# ---------------------------------------------------------------------------


def test_simple_books_imports_cleanly():
    cfg = import_openapi(EXAMPLES / "simple_books.yaml")
    assert isinstance(cfg, MockConfig)
    assert cfg.info.title == "Simple Books API"
    names = [ep.name for ep in cfg.endpoints]
    assert "list_books" in names
    assert "get_book_by_id" in names
    assert "Book" in cfg.schemas
    book = cfg.schemas["Book"]
    assert "id" in book.properties
    assert book.properties["isbn"].pattern is not None


def test_medium_tasks_imports_cleanly():
    cfg = import_openapi(EXAMPLES / "medium_tasks.yaml")
    assert cfg.info.title == "Tasks API"
    names = [ep.name for ep in cfg.endpoints]
    assert "list_tasks" in names
    assert "create_task" in names
    assert "get_task" in names
    assert "replace_task" in names
    assert "delete_task" in names

    list_tasks = cfg.get_endpoint("list_tasks")
    assert "page" in list_tasks.request.query_params
    assert "size" in list_tasks.request.query_params
    assert "X-Total-Count" in list_tasks.responses[0].headers
    assert list_tasks.responses[0].body_schema.ref == "TaskPage"

    create_task = cfg.get_endpoint("create_task")
    assert create_task.request.body_schema.ref == "TaskCreate"
    statuses = [r.status for r in create_task.responses]
    assert 201 in statuses and 400 in statuses and 401 in statuses


def test_complex_payments_imports_cleanly():
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    assert cfg.info.title == "Payments Platform API"
    names = [ep.name for ep in cfg.endpoints]
    for required in ["list_accounts", "get_account", "list_transactions",
                     "create_payment", "get_customer"]:
        assert required in names

    create_payment = cfg.get_endpoint("create_payment")
    assert "Idempotency-Key" in create_payment.request.headers
    statuses = [r.status for r in create_payment.responses]
    assert 201 in statuses and 202 in statuses and 422 in statuses

    # allOf composition: CustomerWithKyc must contain both Customer and KycInfo properties
    cwk = cfg.schemas["CustomerWithKyc"]
    assert "id" in cwk.properties        # from Customer
    assert "status" in cwk.properties    # from KycInfo

    # IBAN format must produce special_rule
    accounts = cfg.schemas["Account"]
    assert accounts.properties["iban"].special_rule == "IBAN"


def test_complex_spec_round_trips_through_dump(tmp_path: Path):
    """Import → dump → re-load yields the same number of endpoints/schemas."""
    from mocks.config_parser import dump_mock_config, load_mock_config
    cfg = import_openapi(EXAMPLES / "complex_payments.yaml")
    out = tmp_path / "complex.yaml"
    dump_mock_config(cfg, out)
    cfg2 = load_mock_config(out)
    assert len(cfg2.endpoints) == len(cfg.endpoints)
    assert set(cfg2.schemas.keys()) == set(cfg.schemas.keys())
