"""Tests for the scenario engine + WireMock renderer's scenario blocks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdp.mocks.renderers.wiremock import render_wiremock
from sdp.mocks.scenario_engine import (
    STARTED_STATE,
    ScenarioPlan,
    ScenarioStep,
    compile_scenarios,
)
from sdp.models.mock_models import (
    EndpointConfig,
    FieldSpec,
    MockConfig,
    ResponseTemplate,
    ScenarioConfig,
    StateTransition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _basic_endpoint(name: str = "list_things", path: str = "/things") -> EndpointConfig:
    return EndpointConfig(
        name=name,
        path=path,
        method="GET",
        responses=[ResponseTemplate(status=200, body_schema=FieldSpec(type="object"))],
    )


def _config_with_scenario(*scenarios: ScenarioConfig) -> MockConfig:
    return MockConfig(
        endpoints=[_basic_endpoint()],
        scenarios=list(scenarios),
    )


# ---------------------------------------------------------------------------
# compile_scenarios — pure logic
# ---------------------------------------------------------------------------


def test_compile_empty_scenarios_returns_empty_plan():
    cfg = MockConfig(endpoints=[_basic_endpoint()], scenarios=[])
    plan = compile_scenarios(cfg)
    assert plan.steps == []


def test_first_transition_starts_from_started_state():
    """The first transition in a scenario should require the WireMock-default `Started` state."""
    scn = ScenarioConfig(
        name="rate_limit",
        states=[
            StateTransition(
                on_match={"endpoint": "list_things"},
                after=1,
                next_response={"status": 429, "body": {"code": "RATE_LIMITED"}},
            )
        ],
    )
    plan = compile_scenarios(_config_with_scenario(scn))
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.required_state == STARTED_STATE
    assert step.scenario_name == "rate_limit"
    assert step.endpoint_name == "list_things"
    assert step.response_override == {"status": 429, "body": {"code": "RATE_LIMITED"}}


def test_after_n_emits_n_steps_with_chained_states():
    """`after: 3` should emit 3 steps so WireMock counts requests via state transitions."""
    scn = ScenarioConfig(
        name="rate_limit",
        states=[
            StateTransition(
                on_match={"endpoint": "list_things"},
                after=3,
                next_response={"status": 429},
            )
        ],
    )
    plan = compile_scenarios(_config_with_scenario(scn))
    assert len(plan.steps) == 3
    assert plan.steps[0].required_state == STARTED_STATE
    # Each step's new_state should be the next step's required_state
    for a, b in zip(plan.steps, plan.steps[1:]):
        assert a.new_state == b.required_state
    # Only the last step carries the response override
    assert plan.steps[0].response_override is None
    assert plan.steps[1].response_override is None
    assert plan.steps[2].response_override == {"status": 429}


def test_multi_transition_chain_states_correctly():
    scn = ScenarioConfig(
        name="lifecycle",
        states=[
            StateTransition(on_match={"endpoint": "list_things"}, after=1,
                            next_response={"status": 201}),
            StateTransition(on_match={"endpoint": "list_things"}, after=1,
                            next_response={"status": 409}),
        ],
    )
    plan = compile_scenarios(_config_with_scenario(scn))
    assert len(plan.steps) == 2
    # Step 0 starts at "Started", step 1 starts where step 0 ends
    assert plan.steps[0].required_state == STARTED_STATE
    assert plan.steps[0].new_state == plan.steps[1].required_state


def test_explicit_requires_state_overrides_chain():
    scn = ScenarioConfig(
        name="branching",
        states=[
            StateTransition(
                on_match={"endpoint": "list_things"},
                after=1,
                requires_state="custom_start",
                sets_state="custom_end",
                next_response={"status": 200},
            ),
        ],
    )
    plan = compile_scenarios(_config_with_scenario(scn))
    step = plan.steps[0]
    assert step.required_state == "custom_start"
    assert step.new_state == "custom_end"


def test_plan_lookup_by_endpoint():
    scn = ScenarioConfig(
        name="x",
        states=[StateTransition(on_match={"endpoint": "list_things"}, after=1,
                                next_response={"status": 200})],
    )
    plan = compile_scenarios(_config_with_scenario(scn))
    assert plan.has_endpoint("list_things")
    assert not plan.has_endpoint("create_thing")


# ---------------------------------------------------------------------------
# WireMock renderer with scenarios
# ---------------------------------------------------------------------------


def test_wiremock_emits_scenario_mapping_for_simple_scenario(tmp_path: Path):
    cfg = MockConfig(
        endpoints=[_basic_endpoint()],
        scenarios=[
            ScenarioConfig(
                name="rate_limit",
                states=[
                    StateTransition(
                        on_match={"endpoint": "list_things"},
                        after=1,
                        next_response={"status": 429, "body": {"code": "RATE_LIMITED"}},
                    )
                ],
            )
        ],
    )
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)

    mappings = list((tmp_path / "mappings").iterdir())
    scenario_mappings = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in mappings if p.name.startswith("scenario__")
    ]
    assert scenario_mappings, "no scenario mapping written"
    m = scenario_mappings[0]
    assert m["scenarioName"] == "rate_limit"
    assert m["requiredScenarioState"] == STARTED_STATE
    assert m["response"]["status"] == 429
    assert m["response"]["jsonBody"] == {"code": "RATE_LIMITED"}


def test_wiremock_after_3_creates_chain_of_3_mappings(tmp_path: Path):
    cfg = MockConfig(
        endpoints=[_basic_endpoint()],
        scenarios=[
            ScenarioConfig(
                name="rate_limit",
                states=[
                    StateTransition(
                        on_match={"endpoint": "list_things"},
                        after=3,
                        next_response={"status": 429},
                    )
                ],
            )
        ],
    )
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)

    scenario_mappings = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (tmp_path / "mappings").iterdir()
        if p.name.startswith("scenario__")
    ]
    assert len(scenario_mappings) == 3
    requireds = {m["requiredScenarioState"] for m in scenario_mappings}
    news = {m["newScenarioState"] for m in scenario_mappings}
    # The chain must start at Started and end somewhere else
    assert STARTED_STATE in requireds
    # Every internal state appears as both a required and a new state — that's
    # what makes it a chain. Only the head (Started) and the tail differ.
    overlap = requireds & news
    assert len(overlap) >= 1, f"chain not connected: requireds={requireds} news={news}"


def test_wiremock_scenario_metadata_block_present(tmp_path: Path):
    cfg = MockConfig(
        endpoints=[_basic_endpoint()],
        scenarios=[
            ScenarioConfig(
                name="x",
                states=[StateTransition(
                    on_match={"endpoint": "list_things"}, after=1,
                    next_response={"status": 200},
                )],
            )
        ],
    )
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    scenario_mappings = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (tmp_path / "mappings").iterdir()
        if p.name.startswith("scenario__")
    ]
    for m in scenario_mappings:
        assert m["metadata"]["sdp"]["scenario"] == "x"
        assert m["metadata"]["sdp"]["endpoint"] == "list_things"


def test_wiremock_scenario_priority_higher_than_default(tmp_path: Path):
    """Scenario mappings must outrank ordinary mappings so they win on overlap."""
    cfg = MockConfig(
        endpoints=[_basic_endpoint()],
        scenarios=[
            ScenarioConfig(
                name="x",
                states=[StateTransition(
                    on_match={"endpoint": "list_things"}, after=1,
                    next_response={"status": 503},
                )],
            )
        ],
    )
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)

    scenario_mappings = []
    regular_mappings = []
    for p in (tmp_path / "mappings").iterdir():
        m = json.loads(p.read_text(encoding="utf-8"))
        (scenario_mappings if "scenarioName" in m else regular_mappings).append(m)

    assert scenario_mappings and regular_mappings
    # Lower priority number = higher precedence in WireMock
    for s in scenario_mappings:
        for r in regular_mappings:
            assert s["priority"] < r["priority"]


def test_wiremock_renders_normal_mappings_when_no_scenarios(tmp_path: Path):
    """Sanity check: scenarios are additive — no scenarios → no scenario files."""
    cfg = MockConfig(endpoints=[_basic_endpoint()])
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    scenario_files = [
        p for p in (tmp_path / "mappings").iterdir()
        if p.name.startswith("scenario__")
    ]
    assert scenario_files == []


def test_wiremock_scenario_with_explicit_path_method(tmp_path: Path):
    """Scenario step targeting an endpoint must inherit that endpoint's method+path."""
    cfg = MockConfig(
        endpoints=[
            EndpointConfig(
                name="create_thing",
                path="/things",
                method="POST",
                responses=[ResponseTemplate(status=201)],
            )
        ],
        scenarios=[
            ScenarioConfig(
                name="dup_after_one",
                states=[StateTransition(
                    on_match={"endpoint": "create_thing"}, after=1,
                    next_response={"status": 409, "body": {"code": "DUPLICATE"}},
                )],
            )
        ],
    )
    render_wiremock(cfg, tmp_path, seed=1, examples_per_endpoint=1)
    scenario_mappings = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (tmp_path / "mappings").iterdir()
        if p.name.startswith("scenario__")
    ]
    assert scenario_mappings
    m = scenario_mappings[0]
    assert m["request"]["method"] == "POST"
    assert m["request"]["urlPath"].endswith("/things") or m["request"].get("urlPathPattern", "").endswith("/things")
