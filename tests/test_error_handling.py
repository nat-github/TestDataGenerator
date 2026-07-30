"""Tests for error handling after the broad-`except` audit.

The rule applied across the codebase:

- **Name the exceptions that can actually occur.** ``except Exception`` hides
  genuine bugs (a typo raising ``NameError`` looked identical to a malformed
  value).
- **Every swallowed failure must leave a trace** — a log line, a note, or a
  returned error — unless it runs per-cell in a hot loop, where a million log
  lines would be worse than silence.

These tests pin both halves: the degraded path still works, *and* it says
something on the way through.
"""
from __future__ import annotations

import ast
import logging
import pathlib

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config parsing degrades loudly
# ---------------------------------------------------------------------------


def test_invalid_rule_is_skipped_with_a_warning(caplog):
    """A malformed rule used to vanish silently.

    ``when`` must be a mapping; an int fails validation.
    """
    from sdp.utils.config_parser import ConfigParser

    with caplog.at_level(logging.WARNING):
        rules = ConfigParser._parse_rules_field([{"when": 123}])

    assert rules is None
    assert "Skipping invalid rule" in caplog.text


def test_one_bad_rule_does_not_discard_the_good_ones(caplog):
    """The handler must skip the offender, not abandon the list."""
    from sdp.utils.config_parser import ConfigParser

    with caplog.at_level(logging.WARNING):
        rules = ConfigParser._parse_rules_field([
            {"when": {"status": "A"}, "then": {"set_value": "X"}},
            {"when": 123},                       # invalid
        ])

    assert rules is not None and len(rules) == 1
    assert "Skipping invalid rule" in caplog.text


def test_valid_rules_parse_without_warnings(caplog):
    from sdp.utils.config_parser import ConfigParser

    with caplog.at_level(logging.WARNING):
        rules = ConfigParser._parse_rules_field(
            [{"when": {"status": "A"}, "then": {"set_value": "X"}}]
        )

    assert rules is not None and len(rules) == 1
    assert "Skipping invalid rule" not in caplog.text


def test_invalid_cdc_block_warns_before_falling_back(caplog):
    """Silently defaulting turns a bad cdc block into a snapshot table, which
    looks like the config was honoured."""
    from sdp.utils.config_parser import ConfigParser

    with caplog.at_level(logging.WARNING):
        result = ConfigParser._expand_cdc_block({"mode": ["not", "a", "string"]})

    assert result                                  # still returns usable defaults
    assert "Invalid cdc block" in caplog.text


def test_empty_cdc_block_is_not_an_error(caplog):
    from sdp.utils.config_parser import ConfigParser

    with caplog.at_level(logging.WARNING):
        assert ConfigParser._expand_cdc_block({}) == {}
    assert "Invalid cdc block" not in caplog.text


# ---------------------------------------------------------------------------
# Generation degrades loudly
# ---------------------------------------------------------------------------


def test_unparseable_null_rate_warns(caplog, tmp_path):
    """A null_rate that will not parse is a config error, not a normal state."""
    from sdp.generators.data_generator import DataGenerator

    config = tmp_path / "c.yaml"
    config.write_text(
        "config_format: sdp-yaml-v1\ntables:\n  - name: t\n    active: true\n"
        "    num_rows: 2\n    columns:\n      - name: c\n        data_type: VA3\n",
        encoding="utf-8",
    )
    gen = DataGenerator(str(config))

    class _Col:
        column_name = "c"
        nullable = True
        null_rate = "not-a-number"
        special_rules = None

    with caplog.at_level(logging.WARNING):
        rate = gen._get_null_probability(_Col())

    assert rate == 0.0                             # falls through to the default
    assert "null_rate" in caplog.text


def test_valid_null_rate_is_honoured(tmp_path):
    from sdp.generators.data_generator import DataGenerator

    config = tmp_path / "c.yaml"
    config.write_text(
        "config_format: sdp-yaml-v1\ntables:\n  - name: t\n    active: true\n"
        "    num_rows: 2\n    columns:\n      - name: c\n        data_type: VA3\n",
        encoding="utf-8",
    )
    gen = DataGenerator(str(config))

    class _Col:
        column_name = "c"
        nullable = True
        null_rate = 0.25
        special_rules = None

    assert gen._get_null_probability(_Col()) == pytest.approx(0.25)


def test_not_nullable_beats_any_null_rate(tmp_path):
    from sdp.generators.data_generator import DataGenerator

    config = tmp_path / "c.yaml"
    config.write_text(
        "config_format: sdp-yaml-v1\ntables:\n  - name: t\n    active: true\n"
        "    num_rows: 2\n    columns:\n      - name: c\n        data_type: VA3\n",
        encoding="utf-8",
    )
    gen = DataGenerator(str(config))

    class _Col:
        column_name = "c"
        nullable = False
        null_rate = 0.9
        special_rules = None

    assert gen._get_null_probability(_Col()) == 0.0


# ---------------------------------------------------------------------------
# Helpers degrade loudly
# ---------------------------------------------------------------------------


def test_unknown_faker_locale_warns_and_falls_back(caplog):
    """Returning US data for a config that asked for something else is a
    decision worth announcing."""
    from sdp.utils.helpers import DataHelpers

    helpers = DataHelpers()
    with caplog.at_level(logging.WARNING):
        faker = helpers._get_locale_faker("zz_ZZ_not_a_locale")

    assert faker is not None
    assert "locale" in caplog.text.lower()


def test_known_locale_does_not_warn(caplog):
    from sdp.utils.helpers import DataHelpers

    helpers = DataHelpers()
    with caplog.at_level(logging.WARNING):
        assert helpers._get_locale_faker("de_DE") is not None
    assert "falling back to en_US" not in caplog.text


def test_bad_distribution_warns_and_still_generates(caplog):
    """Generation must not stop because a distribution descriptor is wrong.

    A non-dict descriptor raises ``AttributeError`` inside the fitter. That
    type was missing from the narrowed handler when it was first written —
    exactly the failure mode narrowing can introduce — so it is pinned here.
    """
    from sdp.utils.helpers import DataHelpers

    helpers = DataHelpers()
    with caplog.at_level(logging.WARNING):
        values = helpers.generate_column_batch(
            {
                "data_type": "N38", "min_value": 1, "max_value": 10,
                "distribution": "not-a-dict",
            },
            n=5,
        )

    assert len(values) == 5
    assert all(1 <= int(v) <= 10 for v in values)
    assert "Distribution sampling failed" in caplog.text


def test_unknown_distribution_name_is_handled_by_the_fitter(caplog):
    """The fitter falls back internally for an unknown name, so no warning
    should be emitted — the handler is for genuine failures only."""
    from sdp.utils.helpers import DataHelpers

    with caplog.at_level(logging.WARNING):
        values = DataHelpers().generate_column_batch(
            {
                "data_type": "N38", "min_value": 1, "max_value": 10,
                "distribution": {"name": "not_a_real_distribution", "params": []},
            },
            n=5,
        )

    assert len(values) == 5
    assert "Distribution sampling failed" not in caplog.text


# ---------------------------------------------------------------------------
# Hot loops stay quiet — deliberately
# ---------------------------------------------------------------------------


def test_arrow_coercion_handles_junk_without_logging(caplog):
    """Per-cell handlers must not log: one bad column would otherwise emit a
    line per row. They coerce to null and move on."""
    from sdp.generators._arrow_export import ArrowExportMixin

    series = pd.Series(["1.23", "not-a-number", None, "inf", "4.56"])
    with caplog.at_level(logging.DEBUG):
        array = ArrowExportMixin._to_arrow_dc(series, precision=10, scale=2)

    values = array.to_pylist()
    assert values[0] is not None and values[4] is not None
    assert values[1] is None and values[2] is None and values[3] is None
    assert caplog.text == ""                       # silence is the requirement


def test_per_cell_coercion_handlers_name_their_exceptions():
    """Inside the coercion helpers, an unexpected exception type is a real
    bug and must propagate rather than silently becoming a null.

    ``export_to_parquet`` is excluded: its broad handler logs and re-raises,
    which is the correct shape for a top-level boundary.
    """
    source = (REPO_ROOT / "sdp" / "generators" / "_arrow_export.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name == "export_to_parquet":
            continue
        for handler in ast.walk(node):
            if (isinstance(handler, ast.ExceptHandler) and handler.type
                    and ast.unparse(handler.type) == "Exception"):
                offenders.append(f"{node.name}:{handler.lineno}")

    assert not offenders, f"broad handler in a per-cell path: {offenders}"


def test_export_boundary_logs_and_reraises():
    """The one broad handler that should exist: a top-level boundary that
    reports the failure and lets it propagate."""
    source = (REPO_ROOT / "sdp" / "generators" / "_arrow_export.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    export = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "export_to_parquet")
    broad = [h for h in ast.walk(export)
             if isinstance(h, ast.ExceptHandler) and h.type
             and ast.unparse(h.type) == "Exception"]

    assert broad, "export_to_parquet should have a reporting boundary"
    for handler in broad:
        body = "\n".join(ast.unparse(s) for s in handler.body)
        assert "raise" in body, "the export boundary must re-raise, not swallow"
        assert "log" in body.lower(), "the export boundary must report before re-raising"


# ---------------------------------------------------------------------------
# Codebase-wide guard
# ---------------------------------------------------------------------------


def _broad_handlers_without_signal():
    """Broad handlers that neither log, re-raise, nor return an error."""
    offenders = []
    for path in (REPO_ROOT / "sdp").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not (node.type and ast.unparse(node.type) in ("Exception", "BaseException")):
                continue
            body = "\n".join(ast.unparse(s) for s in node.body).lower()
            if not any(k in body for k in
                       ("log", "warn", "raise", "return", "append",
                        "st.error", "st.warning", "print(")):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return offenders


def test_no_new_silent_broad_handlers():
    """Ratchet: 5 remain, all per-cell hot loops in Arrow/type coercion where
    logging would be worse than silence. Fix one, lower the number — never
    raise it.
    """
    offenders = _broad_handlers_without_signal()
    assert len(offenders) <= 5, (
        f"new silent broad handler(s) introduced: {offenders}"
    )


def test_no_bare_except_anywhere():
    """`except:` also catches KeyboardInterrupt and SystemExit."""
    offenders = []
    for path in (REPO_ROOT / "sdp").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"bare except found: {offenders}"


# ---------------------------------------------------------------------------
# Handlers that were narrowed — and the ones deliberately left broad
# ---------------------------------------------------------------------------


def test_config_readers_survive_a_malformed_json_config(tmp_path, caplog):
    """These re-read the config file outside ConfigParser. Narrowing them to
    named exceptions must not turn a bad file into a crash."""
    from sdp.services.generation import (
        _read_delta_partition_overrides,
        _read_delta_table_selection,
        _read_versions_per_key,
    )

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert _read_delta_table_selection(str(broken)) is None
        assert _read_versions_per_key(str(broken)) == {}
        assert _read_delta_partition_overrides(str(broken)) == {}
    assert "Could not read" in caplog.text


def test_config_readers_survive_a_malformed_yaml_config(tmp_path, caplog):
    from sdp.services.generation import _read_delta_table_selection

    broken = tmp_path / "broken.yaml"
    broken.write_text("tables: [unclosed\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert _read_delta_table_selection(str(broken)) is None


def test_config_readers_survive_a_wrong_shaped_config(tmp_path):
    """`tables:` holding strings rather than mappings raises AttributeError
    inside the loop — which the narrowed tuple must still catch."""
    from sdp.services.generation import _read_versions_per_key

    odd = tmp_path / "odd.yaml"
    odd.write_text("tables:\n  - just-a-string\n", encoding="utf-8")
    assert _read_versions_per_key(str(odd)) == {}


def test_config_readers_handle_a_missing_file(tmp_path):
    from sdp.services.generation import _read_delta_table_selection

    assert _read_delta_table_selection(str(tmp_path / "gone.yaml")) is None


def test_json_config_path_does_not_hit_an_unbound_yaml_name(tmp_path):
    """Regression: `import yaml` used to live inside the .yaml branch, so a
    .json config left the name unbound — and the narrowed handler references
    yaml.YAMLError, which would then raise UnboundLocalError."""
    import json as _json

    from sdp.services.generation import _read_delta_table_selection

    good = tmp_path / "c.json"
    good.write_text(_json.dumps({"tables": [{"name": "t", "write_delta": True}]}),
                    encoding="utf-8")
    assert _read_delta_table_selection(str(good)) == ["t"]


def test_mcp_broad_handlers_are_the_tool_error_channel():
    """Documented exemption: every broad handler in the MCP server returns an
    error result. A tool that raises gives the agent a transport error rather
    than a message it can act on, so these must not be narrowed.
    """
    source = (REPO_ROOT / "sdp" / "mcp_server" / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ExceptHandler) and node.type
                and ast.unparse(node.type) == "Exception"):
            continue
        body = "\n".join(ast.unparse(s) for s in node.body)
        if not ("return" in body and ("ok" in body or "error" in body)):
            offenders.append(node.lineno)

    assert not offenders, (
        f"MCP handlers at {offenders} neither narrow nor return an error result"
    )


def test_import_guards_are_narrowed_to_import_error():
    """An import guard catching Exception hides a NameError in the module it
    is importing, reporting it as 'could not import'."""
    for module in ("sdp/cli_commands/quality.py",):
        tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = "\n".join(ast.unparse(s) for s in node.body)
            if "Could not import" in body:
                assert node.type and "ImportError" in ast.unparse(node.type), (
                    f"{module}:{node.lineno} import guard is still broad"
                )
