"""Tests for Layer C — declarative lifecycle workflows.

The point of a workflow is that certain rows become *impossible*: a
CANCELLED order with a delivery timestamp, a DELIVERED order that was never
shipped, a return that predates the delivery. These tests assert those
impossibilities rather than just checking the code runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.models.config_models import WorkflowConfig, WorkflowTransition
from sdp.utils.workflow_engine import (
    MAX_STEPS,
    WorkflowError,
    apply_workflow,
    apply_workflows,
    validate_workflow,
    walk,
)


def _rng(seed: int = 42):
    return np.random.default_rng(seed)


def _order_workflow(**overrides) -> WorkflowConfig:
    base = dict(
        name="order_lifecycle",
        table="orders",
        state_column="status",
        start_state="PLACED",
        timestamps={
            "placed_ts": "PLACED",
            "shipped_ts": "SHIPPED",
            "delivered_ts": "DELIVERED",
            "cancelled_ts": "CANCELLED",
        },
        transitions=[
            {"from": "PLACED", "to": "SHIPPED", "probability": 0.9},
            {"from": "PLACED", "to": "CANCELLED", "probability": 0.1},
            {"from": "SHIPPED", "to": "DELIVERED", "probability": 1.0},
        ],
    )
    base.update(overrides)
    return WorkflowConfig(**base)


def _orders(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": range(n),
        "status": ["WHATEVER"] * n,
        "placed_ts": pd.NaT,
        "shipped_ts": pd.NaT,
        "delivered_ts": pd.NaT,
        "cancelled_ts": pd.NaT,
    })


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_transition_accepts_the_natural_from_to_shape():
    """`from` is a Python keyword; configs must still read naturally."""
    transition = WorkflowTransition(**{"from": "A", "to": "B", "probability": 0.5})
    assert transition.from_state == "A"
    assert transition.to_state == "B"


def test_transition_probability_is_bounded():
    with pytest.raises(ValueError):
        WorkflowTransition(**{"from": "A", "to": "B", "probability": 1.5})
    with pytest.raises(ValueError):
        WorkflowTransition(**{"from": "A", "to": "B", "probability": -0.1})


def test_states_are_listed_in_first_seen_order():
    workflow = _order_workflow()
    assert workflow.states == ["PLACED", "SHIPPED", "CANCELLED", "DELIVERED"]


def test_start_state_is_inferred_when_unambiguous():
    workflow = _order_workflow(start_state=None)
    assert workflow.resolve_start_state() == "PLACED"


def test_start_state_cannot_be_inferred_from_a_cycle():
    workflow = WorkflowConfig(
        name="w", table="t", state_column="s",
        transitions=[{"from": "A", "to": "B"}, {"from": "B", "to": "A"}],
    )
    assert workflow.resolve_start_state() is None


def test_declared_start_state_wins_over_inference():
    workflow = _order_workflow(start_state="SHIPPED")
    assert workflow.resolve_start_state() == "SHIPPED"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_workflow_reports_no_problems():
    columns = ["order_id", "status", "placed_ts", "shipped_ts",
               "delivered_ts", "cancelled_ts"]
    assert validate_workflow(_order_workflow(), columns) == []


def test_missing_state_column_is_reported():
    problems = validate_workflow(_order_workflow(), ["order_id", "placed_ts"])
    assert any("state_column" in p for p in problems)


def test_missing_timestamp_column_is_reported():
    problems = validate_workflow(_order_workflow(), ["order_id", "status"])
    assert any("placed_ts" in p for p in problems)


def test_timestamp_mapped_to_an_unknown_state_is_reported():
    workflow = _order_workflow(timestamps={"placed_ts": "NO_SUCH_STATE"})
    problems = validate_workflow(workflow, ["status", "placed_ts"])
    assert any("no transition mentions" in p for p in problems)


def test_probabilities_over_one_are_reported():
    workflow = _order_workflow(transitions=[
        {"from": "A", "to": "B", "probability": 0.8},
        {"from": "A", "to": "C", "probability": 0.5},
    ], start_state="A", timestamps={})
    problems = validate_workflow(workflow, ["status"])
    assert any("above 1.0" in p for p in problems)


def test_uninferable_start_state_is_reported():
    workflow = WorkflowConfig(
        name="w", table="t", state_column="status",
        transitions=[{"from": "A", "to": "B"}, {"from": "B", "to": "A"}],
    )
    problems = validate_workflow(workflow, ["status"])
    assert any("start_state" in p for p in problems)


def test_no_transitions_is_reported():
    workflow = WorkflowConfig(name="w", table="t", state_column="status")
    assert any("no transitions" in p for p in validate_workflow(workflow, ["status"]))


def test_bad_step_hours_is_reported():
    workflow = _order_workflow(step_hours=[10, 1])
    assert any("step_hours" in p for p in validate_workflow(
        workflow, ["status", "placed_ts", "shipped_ts", "delivered_ts", "cancelled_ts"]))


# ---------------------------------------------------------------------------
# Walking the machine
# ---------------------------------------------------------------------------


def test_walk_always_starts_at_the_start_state():
    workflow = _order_workflow()
    rng = _rng()
    for _ in range(50):
        path, _truncated = walk(workflow, rng)
        assert path[0] == "PLACED"


def test_walk_only_follows_declared_transitions():
    """No path may contain a step the config does not permit."""
    workflow = _order_workflow()
    legal = {(t.from_state, t.to_state) for t in workflow.transitions}
    rng = _rng()

    for _ in range(200):
        path, _ = walk(workflow, rng)
        for a, b in zip(path, path[1:]):
            assert (a, b) in legal, f"illegal transition {a} -> {b}"


def test_walk_terminates_at_a_state_with_no_outgoing_edges():
    workflow = _order_workflow()
    rng = _rng()
    terminals = {walk(workflow, rng)[0][-1] for _ in range(200)}
    assert terminals <= {"DELIVERED", "CANCELLED"}


def test_residual_probability_means_stop_here():
    """Outgoing probabilities summing below 1 leave the remainder as the
    chance of stopping — the mechanism the example config relies on."""
    workflow = WorkflowConfig(
        name="w", table="t", state_column="s", start_state="A",
        transitions=[{"from": "A", "to": "B", "probability": 0.5}],
    )
    rng = _rng(7)
    endings = [walk(workflow, rng)[0][-1] for _ in range(400)]
    stopped_at_a = endings.count("A")
    assert 150 < stopped_at_a < 250, stopped_at_a      # ~50%


def test_probabilities_are_respected():
    workflow = _order_workflow()
    rng = _rng(3)
    endings = [walk(workflow, rng)[0][-1] for _ in range(1000)]
    cancelled = endings.count("CANCELLED") / 1000
    assert 0.06 < cancelled < 0.14, cancelled          # configured 0.10


def test_cyclic_workflow_is_capped_and_flagged():
    """A reopenable lifecycle must not loop forever."""
    workflow = WorkflowConfig(
        name="w", table="t", state_column="s", start_state="OPEN",
        transitions=[
            {"from": "OPEN", "to": "CLOSED", "probability": 1.0},
            {"from": "CLOSED", "to": "OPEN", "probability": 1.0},
        ],
    )
    path, truncated = walk(workflow, _rng())
    assert truncated is True
    assert len(path) == MAX_STEPS + 1


def test_walk_without_a_resolvable_start_raises():
    workflow = WorkflowConfig(
        name="w", table="t", state_column="s",
        transitions=[{"from": "A", "to": "B"}, {"from": "B", "to": "A"}],
    )
    with pytest.raises(WorkflowError, match="start_state"):
        walk(workflow, _rng())


# ---------------------------------------------------------------------------
# Applying to a table — the invariants that matter
# ---------------------------------------------------------------------------


def test_unvisited_states_have_null_timestamps():
    """The headline invariant: a cancelled order has no delivery time."""
    out, _ = apply_workflow(_orders(300), _order_workflow(), _rng())

    cancelled = out[out["status"] == "CANCELLED"]
    assert cancelled["delivered_ts"].isna().all()
    assert cancelled["shipped_ts"].isna().all()

    delivered = out[out["status"] == "DELIVERED"]
    assert delivered["cancelled_ts"].isna().all()


def test_visited_states_all_have_timestamps():
    out, _ = apply_workflow(_orders(300), _order_workflow(), _rng())

    assert out["placed_ts"].notna().all()              # every path starts here
    delivered = out[out["status"] == "DELIVERED"]
    assert delivered["shipped_ts"].notna().all()       # reached via SHIPPED
    assert delivered["delivered_ts"].notna().all()


def test_timestamps_increase_along_the_path():
    out, _ = apply_workflow(_orders(300), _order_workflow(), _rng())

    shipped = out[out["shipped_ts"].notna()]
    assert (shipped["placed_ts"] < shipped["shipped_ts"]).all()

    delivered = out[out["delivered_ts"].notna()]
    assert (delivered["shipped_ts"] < delivered["delivered_ts"]).all()

    cancelled = out[out["cancelled_ts"].notna()]
    assert (cancelled["placed_ts"] < cancelled["cancelled_ts"]).all()


def test_state_column_is_overwritten():
    """Whatever generation produced for the state column is replaced."""
    out, _ = apply_workflow(_orders(50), _order_workflow(), _rng())
    assert "WHATEVER" not in set(out["status"])
    assert set(out["status"]) <= {"PLACED", "SHIPPED", "DELIVERED", "CANCELLED"}


def test_unrelated_columns_are_untouched():
    frame = _orders(40)
    frame["order_total"] = range(40)
    out, _ = apply_workflow(frame, _order_workflow(), _rng())
    assert list(out["order_total"]) == list(range(40))
    assert list(out["order_id"]) == list(range(40))


def test_row_count_is_preserved():
    out, _ = apply_workflow(_orders(137), _order_workflow(), _rng())
    assert len(out) == 137


def test_start_between_bounds_the_first_timestamp():
    workflow = _order_workflow(start_between=["2024-03-01", "2024-03-31"])
    out, _ = apply_workflow(_orders(100), workflow, _rng())

    assert out["placed_ts"].min() >= pd.Timestamp("2024-03-01")
    assert out["placed_ts"].min() <= pd.Timestamp("2024-04-01")


def test_step_hours_bounds_the_gap():
    workflow = _order_workflow(step_hours=[24, 25])
    out, _ = apply_workflow(_orders(100), workflow, _rng())

    shipped = out[out["shipped_ts"].notna()]
    gaps = (shipped["shipped_ts"] - shipped["placed_ts"]).dt.total_seconds() / 3600
    assert (gaps >= 24).all() and (gaps <= 25.01).all()


def test_invalid_start_between_falls_back_with_a_warning(caplog):
    import logging

    workflow = _order_workflow(start_between=["not-a-date", "also-not"])
    with caplog.at_level(logging.WARNING):
        out, _ = apply_workflow(_orders(20), workflow, _rng())

    assert out["placed_ts"].notna().all()
    assert "start_between" in caplog.text


def test_missing_column_raises_rather_than_corrupting():
    frame = pd.DataFrame({"order_id": [1, 2], "status": ["X", "Y"]})
    with pytest.raises(WorkflowError, match="no column"):
        apply_workflow(frame, _order_workflow(), _rng())


def test_empty_table_is_a_no_op():
    empty = _orders(0)
    out, stats = apply_workflow(empty, _order_workflow(), _rng())
    assert len(out) == 0
    assert stats.rows == 0


def test_stats_report_terminal_distribution():
    _, stats = apply_workflow(_orders(200), _order_workflow(), _rng())
    assert stats.rows == 200
    assert sum(stats.terminal_states.values()) == 200
    assert set(stats.terminal_states) <= {"DELIVERED", "CANCELLED"}
    assert stats.to_dict()["name"] == "order_lifecycle"


def test_same_seed_reproduces_the_same_lifecycles():
    a, _ = apply_workflow(_orders(100), _order_workflow(), _rng(11))
    b, _ = apply_workflow(_orders(100), _order_workflow(), _rng(11))
    pd.testing.assert_frame_equal(a, b)


def test_date_only_columns_stay_date_only():
    """A D column must not come back with a time component."""
    frame = _orders(50)
    frame["placed_ts"] = pd.Timestamp("2024-01-01")     # midnight → date-like
    out, _ = apply_workflow(frame, _order_workflow(), _rng())
    stamps = out["placed_ts"].dropna()
    assert (stamps.dt.normalize() == stamps).all()


# ---------------------------------------------------------------------------
# apply_workflows over a dataset
# ---------------------------------------------------------------------------


def test_apply_workflows_skips_absent_tables(caplog):
    import logging

    data = {"other": _orders(5)}
    with caplog.at_level(logging.WARNING):
        out, stats = apply_workflows(data, [_order_workflow()], _rng())

    assert stats == []
    assert "was not generated" in caplog.text
    assert set(out) == {"other"}


def test_apply_workflows_handles_multiple_workflows():
    data = {"orders": _orders(50), "claims": _orders(50)}
    claims = _order_workflow(name="claim_lifecycle", table="claims")

    _, stats = apply_workflows(data, [_order_workflow(), claims], _rng())
    assert {s.table for s in stats} == {"orders", "claims"}


# ---------------------------------------------------------------------------
# Config parsing + end-to-end
# ---------------------------------------------------------------------------


WORKFLOW_YAML = """
config_format: sdp-yaml-v1
tables:
  - name: orders
    active: true
    num_rows: 120
    columns:
      - name: order_id
        data_type: N38
        is_primary_key: true
      - name: status
        data_type: VA16
        business_values: "PLACED;SHIPPED;DELIVERED;CANCELLED"
      - name: placed_ts
        data_type: DT
      - name: shipped_ts
        data_type: DT
      - name: delivered_ts
        data_type: DT
      - name: cancelled_ts
        data_type: DT
workflows:
  - name: order_lifecycle
    table: orders
    state_column: status
    start_state: PLACED
    timestamps:
      placed_ts: PLACED
      shipped_ts: SHIPPED
      delivered_ts: DELIVERED
      cancelled_ts: CANCELLED
    transitions:
      - { from: PLACED, to: SHIPPED, probability: 0.9 }
      - { from: PLACED, to: CANCELLED, probability: 0.1 }
      - { from: SHIPPED, to: DELIVERED, probability: 1.0 }
"""


@pytest.fixture
def workflow_config(tmp_path):
    path = tmp_path / "wf.yaml"
    path.write_text(WORKFLOW_YAML.strip(), encoding="utf-8")
    return str(path)


def test_workflows_parse_from_yaml(workflow_config):
    from sdp.utils.config_parser import ConfigParser

    parser = ConfigParser(workflow_config)
    assert parser.load_config() is True
    assert len(parser.workflows) == 1
    assert parser.workflows[0].name == "order_lifecycle"
    assert parser.workflows[0].transitions[0].from_state == "PLACED"


def test_invalid_workflow_is_skipped_with_a_warning(tmp_path, caplog):
    import logging

    from sdp.utils.config_parser import ConfigParser

    path = tmp_path / "bad.yaml"
    path.write_text(
        WORKFLOW_YAML.strip() + "\n  - name: broken\n    transitions: 5\n",
        encoding="utf-8",
    )
    parser = ConfigParser(str(path))
    with caplog.at_level(logging.WARNING):
        parser.load_config()

    assert [w.name for w in parser.workflows] == ["order_lifecycle"]
    assert "Skipping invalid workflow" in caplog.text


def test_config_without_workflows_has_an_empty_list(tmp_path):
    from sdp.utils.config_parser import ConfigParser

    path = tmp_path / "plain.yaml"
    path.write_text(
        "config_format: sdp-yaml-v1\ntables:\n  - name: t\n    active: true\n"
        "    num_rows: 2\n    columns:\n      - name: c\n        data_type: VA3\n",
        encoding="utf-8",
    )
    parser = ConfigParser(str(path))
    parser.load_config()
    assert parser.workflows == []


def test_end_to_end_generation_applies_the_workflow(workflow_config, tmp_path):
    """The invariants must survive the full pipeline, not just the engine."""
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(
        config=workflow_config, output=str(tmp_path / "out"),
        default_records=120, seed=42, engine="rule-based",
    ))
    assert outcome.ok, outcome.error

    orders = pd.read_parquet(tmp_path / "out" / "orders.parquet")
    assert len(orders) == 120
    assert set(orders["status"]) <= {"PLACED", "SHIPPED", "DELIVERED", "CANCELLED"}

    cancelled = orders[orders["status"] == "CANCELLED"]
    assert cancelled["delivered_ts"].isna().all()
    delivered = orders[orders["status"] == "DELIVERED"]
    assert delivered["shipped_ts"].notna().all()
    assert (delivered["shipped_ts"] < delivered["delivered_ts"]).all()


def test_workflow_stats_appear_on_the_generation_report(workflow_config, tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(
        config=workflow_config, output=str(tmp_path / "out"),
        default_records=60, seed=1, engine="rule-based",
    ))
    workflows = outcome.report.get("workflows", [])
    assert workflows and workflows[0]["name"] == "order_lifecycle"
    assert sum(workflows[0]["terminal_states"].values()) == 60


def test_shipped_example_config_is_valid():
    """The example shipped in examples/configs/yaml must actually work."""
    import pathlib

    from sdp.utils.config_parser import ConfigParser
    from sdp.utils.workflow_engine import validate_workflow

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "examples" / "configs" / "yaml" / "12_workflow_lifecycle.yaml")
    parser = ConfigParser(str(path))
    assert parser.load_config() is True
    tables = parser.parse_tables()

    assert parser.workflows, "example declares no workflows"
    for workflow in parser.workflows:
        columns = [c.column_name for c in tables[workflow.table].columns]
        assert validate_workflow(workflow, columns) == []
