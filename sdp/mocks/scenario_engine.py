"""Scenario engine — compile `ScenarioConfig` into renderer-friendly state plans.

A scenario is a stateful sequence of expected behaviours: e.g. "after 3
calls to `list_accounts`, the next call should return 429 RATE_LIMITED".

This module is *renderer-agnostic*: it produces a flat list of
``ScenarioStep`` records that any renderer can map to its native state
mechanism:

  - WireMock: ``scenarioName`` + ``requiredScenarioState`` + ``newScenarioState``
  - Pact: ``providerStates``
  - Custom servers: a request log + state machine

The compilation is intentionally simple: each transition in a scenario
becomes a step that fires once when its prerequisite state and counter
match. Complex branching is out of scope for now (tracked as TODO).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sdp.models.mock_models import MockConfig, ScenarioConfig, StateTransition


# WireMock's reserved "starting state" name. Mappings without a
# `requiredScenarioState` implicitly accept this state.
STARTED_STATE = "Started"


@dataclass
class ScenarioStep:
    """One compiled step in a scenario.

    Attributes
    ----------
    scenario_name:
        Stable identifier shared by all steps belonging to the same
        ``ScenarioConfig``. Renderers usually emit this directly into
        WireMock's ``scenarioName`` or Pact's ``providerStates[].name``.
    endpoint_name:
        The endpoint this step targets. None means "any endpoint".
    required_state:
        The state the system must be in for this step to fire.
    new_state:
        The state to transition to after firing.
    after:
        How many matching requests must have happened before this step
        fires. WireMock can express this directly via repeated state
        transitions (Started → step_1 → step_2 → …).
    response_override:
        Optional inline response override (status, body, …) used by the
        WireMock renderer when a transition replaces the default response.
    requires_state, sets_state:
        Pass-through from the source ``StateTransition`` for renderers
        that want richer semantics.
    """
    scenario_name: str
    endpoint_name: Optional[str]
    required_state: str
    new_state: str
    after: int = 1
    response_override: Optional[Dict[str, Any]] = None
    requires_state: Optional[str] = None
    sets_state: Optional[str] = None


@dataclass
class ScenarioPlan:
    """Compiled view of every scenario in a MockConfig."""
    steps: List[ScenarioStep] = field(default_factory=list)

    def steps_for_endpoint(self, endpoint_name: str) -> List[ScenarioStep]:
        """Return scenario steps that target a specific endpoint, in order."""
        return [
            s for s in self.steps
            if s.endpoint_name is None or s.endpoint_name == endpoint_name
        ]

    def has_endpoint(self, endpoint_name: str) -> bool:
        return any(
            s.endpoint_name == endpoint_name for s in self.steps
        )


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def compile_scenarios(config: MockConfig) -> ScenarioPlan:
    """Walk every ScenarioConfig and produce a flat list of `ScenarioStep`."""
    plan = ScenarioPlan()
    for scenario in config.scenarios:
        for index, transition in enumerate(scenario.states):
            plan.steps.extend(_compile_transition(scenario, transition, index))
    return plan


def _compile_transition(
    scenario: ScenarioConfig,
    transition: StateTransition,
    index: int,
) -> List[ScenarioStep]:
    """Turn one StateTransition into one or more ScenarioStep entries.

    Most transitions become a single step. The ``after`` field is honoured
    by emitting (after - 1) "no-op" intermediate steps that just bump the
    state counter — that's how WireMock counts repeated calls.
    """
    endpoint_name = transition.on_match.get("endpoint")

    # Build the required-state / new-state chain
    base_required = transition.requires_state or (
        STARTED_STATE if index == 0 else f"{scenario.name}__step_{index}"
    )
    base_new = transition.sets_state or f"{scenario.name}__step_{index + 1}"

    steps: List[ScenarioStep] = []

    # Intermediate "counter" steps when after > 1
    for i in range(1, transition.after):
        intermediate_required = (
            base_required if i == 1 else f"{scenario.name}__counter_{index}_{i - 1}"
        )
        intermediate_new = f"{scenario.name}__counter_{index}_{i}"
        steps.append(ScenarioStep(
            scenario_name=scenario.name,
            endpoint_name=endpoint_name,
            required_state=intermediate_required,
            new_state=intermediate_new,
            after=1,
            response_override=None,  # use the default response for counter ticks
            requires_state=transition.requires_state,
            sets_state=None,
        ))

    # The actual fire step — uses the response_override (if any)
    final_required = (
        f"{scenario.name}__counter_{index}_{transition.after - 1}"
        if transition.after > 1 else base_required
    )
    steps.append(ScenarioStep(
        scenario_name=scenario.name,
        endpoint_name=endpoint_name,
        required_state=final_required,
        new_state=base_new,
        after=transition.after,
        response_override=transition.next_response or None,
        requires_state=transition.requires_state,
        sets_state=transition.sets_state,
    ))

    return steps
