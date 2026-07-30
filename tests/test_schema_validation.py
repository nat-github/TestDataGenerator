"""Tests for JSON Schema validation of YAML/JSON configs.

The schema existed for IDE autocompletion but nothing enforced it, so a
config could be schema-invalid and still load — the parser is deliberately
tolerant and silently skips what it does not understand. That is right for
generation and wrong for authoring, which is the gap these tests pin.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from sdp.utils.schema_validator import (
    SchemaUnavailable,
    SchemaViolation,
    load_document,
    load_schema,
    schema_supported,
    validate_config_file,
    validate_document,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "configs"


MINIMAL = {
    "config_format": "sdp-yaml-v1",
    "tables": [{
        "name": "orders",
        "columns": [{"name": "order_id", "data_type": "N38", "is_pk": True},
                    {"name": "status", "data_type": "VA16"}],
    }],
}


def _with_workflow(**overrides):
    workflow = {
        "name": "order_lifecycle",
        "table": "orders",
        "state_column": "status",
        "timestamps": {"placed_ts": "PLACED"},
        "transitions": [{"from": "PLACED", "to": "SHIPPED", "probability": 0.9}],
    }
    workflow.update(overrides)
    return {**MINIMAL, "workflows": [workflow]}


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def test_schema_loads_and_is_draft_2020_12():
    schema = load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert "workflows" in schema["properties"]


def test_missing_schema_file_raises_clearly(tmp_path):
    with pytest.raises(SchemaUnavailable, match="not found"):
        load_schema(tmp_path / "nope.json")


def test_malformed_schema_file_raises_clearly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaUnavailable, match="not valid JSON"):
        load_schema(bad)


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("cfg.yaml", True), ("cfg.yml", True), ("cfg.json", True),
    ("cfg.xlsx", False), ("cfg.xls", False), ("cfg", False),
])
def test_schema_supported_by_suffix(name, expected):
    """Excel has no document to validate — the schema describes YAML/JSON."""
    assert schema_supported(name) is expected


def test_excel_config_yields_no_violations(tmp_path):
    workbook = tmp_path / "cfg.xlsx"
    workbook.write_bytes(b"not really xlsx")
    assert validate_config_file(str(workbook)) == []


# ---------------------------------------------------------------------------
# Valid documents
# ---------------------------------------------------------------------------


def test_minimal_config_is_valid():
    assert validate_document(MINIMAL) == []


def test_config_with_a_workflow_is_valid():
    assert validate_document(_with_workflow()) == []


def test_workflow_optional_fields_are_valid():
    document = _with_workflow(
        start_state="PLACED",
        start_between=["2024-01-01", "2024-12-31"],
        step_hours=[2, 96],
        note="lifecycle",
    )
    assert validate_document(document) == []


def test_transition_probability_is_optional():
    document = _with_workflow(transitions=[{"from": "A", "to": "B"}])
    assert validate_document(document) == []


# ---------------------------------------------------------------------------
# Workflow violations — the block the schema previously did not cover
# ---------------------------------------------------------------------------


def test_missing_required_workflow_field_is_reported():
    document = _with_workflow()
    del document["workflows"][0]["state_column"]
    violations = validate_document(document)
    assert any("state_column" in v.message for v in violations)


def test_probability_above_one_is_reported():
    document = _with_workflow(
        transitions=[{"from": "A", "to": "B", "probability": 1.7}]
    )
    violations = validate_document(document)
    assert any("maximum" in v.message for v in violations)
    assert any("probability" in v.path for v in violations)


def test_transition_missing_from_is_reported():
    document = _with_workflow(transitions=[{"to": "B"}])
    assert any("'from' is a required property" in v.message
               for v in validate_document(document))


def test_empty_transition_list_is_reported():
    """A workflow with no transitions cannot walk anywhere."""
    document = _with_workflow(transitions=[])
    assert any("non-empty" in v.message.lower()
               for v in validate_document(document))


def test_unknown_workflow_key_is_reported():
    """Typos are the main thing schema validation buys — `transitons:` would
    otherwise be silently ignored."""
    document = _with_workflow(typo_field="oops")
    assert any("typo_field" in v.message for v in validate_document(document))


def test_step_hours_must_be_a_pair():
    assert any("short" in v.message.lower()
               for v in validate_document(_with_workflow(step_hours=[1])))
    assert any("long" in v.message.lower()
               for v in validate_document(_with_workflow(step_hours=[1, 2, 3])))


def test_start_between_must_be_a_pair():
    assert validate_document(_with_workflow(start_between=["2024-01-01"]))


def test_timestamps_values_must_be_strings():
    document = _with_workflow(timestamps={"placed_ts": 5})
    assert any("placed_ts" in v.path for v in validate_document(document))


# ---------------------------------------------------------------------------
# Violations elsewhere in the document
# ---------------------------------------------------------------------------


def test_bad_config_format_is_reported():
    document = {**MINIMAL, "config_format": "not-a-format"}
    assert any("config_format" in v.path for v in validate_document(document))


def test_table_without_columns_is_reported():
    document = {"tables": [{"name": "orders"}]}
    assert any("columns" in v.message for v in validate_document(document))


def test_unknown_cdc_key_is_reported():
    document = {**MINIMAL}
    document["tables"] = [{**MINIMAL["tables"][0], "cdc": {"mode": "delta", "typo": 1}}]
    assert any("typo" in v.message for v in validate_document(document))


def test_invalid_cdc_mode_is_reported():
    document = {**MINIMAL}
    document["tables"] = [{**MINIMAL["tables"][0], "cdc": {"mode": "sideways"}}]
    assert any("mode" in v.path for v in validate_document(document))


def test_all_violations_are_returned_not_just_the_first():
    """Three typos should produce three messages."""
    document = _with_workflow(typo_field="a", step_hours=[1])
    document["config_format"] = "nope"
    assert len(validate_document(document)) >= 3


def test_violation_order_is_deterministic():
    """Sorted by (path, message) so CI diffs are stable between runs."""
    document = _with_workflow(typo_field="a", step_hours=[1])
    first = [(v.path, v.message) for v in validate_document(document)]
    second = [(v.path, v.message) for v in validate_document(document)]
    assert first == second
    assert first == sorted(first)


def test_violation_path_locates_the_problem():
    document = _with_workflow(
        transitions=[{"from": "A", "to": "B"}, {"from": "C", "to": "D", "probability": 9}]
    )
    violation = next(v for v in validate_document(document) if "probability" in v.path)
    assert violation.path == "workflows[0].transitions[1].probability"
    assert "workflows[0].transitions[1].probability" in str(violation)


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------


def test_validate_config_file_on_yaml(tmp_path):
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_with_workflow()), encoding="utf-8")
    assert validate_config_file(str(path)) == []


def test_validate_config_file_on_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_with_workflow()), encoding="utf-8")
    assert validate_config_file(str(path)) == []


def test_unparseable_file_is_reported_not_raised(tmp_path):
    """lint should show a parse failure like any other problem."""
    path = tmp_path / "cfg.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    violations = validate_config_file(str(path))
    assert violations and "could not parse" in violations[0].message


def test_empty_file_is_reported(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("", encoding="utf-8")
    violations = validate_config_file(str(path))
    assert violations and "empty" in violations[0].message


def test_load_document_reads_yaml_and_json(tmp_path):
    import yaml

    y = tmp_path / "a.yaml"
    y.write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")
    assert load_document(str(y)) == {"a": 1}

    j = tmp_path / "b.json"
    j.write_text(json.dumps({"b": 2}), encoding="utf-8")
    assert load_document(str(j)) == {"b": 2}


# ---------------------------------------------------------------------------
# Every shipped example must satisfy its own schema
# ---------------------------------------------------------------------------


def _example_configs():
    return sorted(
        [*(EXAMPLES / "yaml").glob("*.yaml"), *(EXAMPLES / "json").glob("*.json")]
    )


def test_there_are_example_configs_to_check():
    assert _example_configs(), "no example configs found"


@pytest.mark.parametrize("config", _example_configs(), ids=lambda p: p.name)
def test_shipped_example_matches_the_schema(config):
    """A shipped example that violates the schema teaches the wrong shape."""
    violations = validate_config_file(str(config))
    assert violations == [], "\n".join(str(v) for v in violations)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_lint_exposes_strict_schema_flag():
    from sdp.cli import build_parser

    args = build_parser().parse_args(["lint", "--config", "c.yaml", "--strict-schema"])
    assert args.strict_schema is True


def test_strict_schema_defaults_off():
    from sdp.cli import build_parser

    args = build_parser().parse_args(["lint", "--config", "c.yaml"])
    assert args.strict_schema is False


def test_schema_issues_are_warnings_by_default(tmp_path):
    import yaml

    from sdp.cli_commands.config_tools import _schema_issues

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_with_workflow(typo_field="oops")), encoding="utf-8")

    issues = _schema_issues(str(path), strict=False)
    assert issues and all(i.level == "warning" for i in issues)


def test_strict_schema_promotes_them_to_errors(tmp_path):
    import yaml

    from sdp.cli_commands.config_tools import _schema_issues

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_with_workflow(typo_field="oops")), encoding="utf-8")

    issues = _schema_issues(str(path), strict=True)
    assert issues and all(i.level == "error" for i in issues)


def test_schema_issues_carry_the_document_path(tmp_path):
    import yaml

    from sdp.cli_commands.config_tools import _schema_issues

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_with_workflow(step_hours=[1])), encoding="utf-8")

    issue = next(i for i in _schema_issues(str(path), strict=False)
                 if "step_hours" in (i.field_name or ""))
    assert issue.sheet == "schema"


def test_schema_issues_empty_for_excel(tmp_path):
    from sdp.cli_commands.config_tools import _schema_issues

    workbook = tmp_path / "cfg.xlsx"
    workbook.write_bytes(b"stub")
    assert _schema_issues(str(workbook), strict=True) == []
