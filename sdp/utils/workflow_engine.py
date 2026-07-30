"""Layer C — declarative lifecycle workflows.

Layers A (when/then rules) and B (derived columns) fix *within-row*
consistency. Neither can express a **lifecycle**: that an order reaches
DELIVERED only by passing through SHIPPED, that its delivery timestamp is
necessarily later than its ship timestamp, and that a CANCELLED order has
no delivery timestamp at all.

Independent column draws produce exactly those contradictions, and they are
the ones that make a downstream test pass when it should fail.

How a row is generated
----------------------
1. Start at the workflow's start state.
2. Repeatedly pick one outgoing transition, weighted by ``probability``.
   **If the outgoing probabilities sum to less than 1, the remainder is the
   chance of stopping here** — that is how a state becomes non-terminal but
   still absorbing for some rows. A state with no outgoing transitions is
   terminal.
3. The visited states are the row's path.
4. Every ``timestamps`` column whose state is on the path gets a value,
   strictly increasing along the path. Columns for unvisited states are set
   to NULL.

Cycles are permitted (a claim can reopen), so the walk is capped at
``MAX_STEPS`` to keep a cyclic definition from looping forever.

Ordering
--------
Workflows run **before** Layers A and B, so rules and derived columns see
the final state value and can build on it. This module operates on plain
dicts and a DataFrame, like ``rule_evaluator``, so the stub/mock track can
reuse it later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sdp.models.config_models import WorkflowConfig

logger = logging.getLogger(__name__)

#: A cyclic workflow could otherwise walk forever.
MAX_STEPS = 50

#: Default first-timestamp window when the config gives none.
_DEFAULT_START_DAYS_BACK = 365


class WorkflowError(ValueError):
    """A workflow definition that cannot be executed as written."""


@dataclass
class WorkflowStats:
    """What a workflow actually produced — for logging and tests."""
    name: str
    table: str
    rows: int = 0
    terminal_states: Dict[str, int] = field(default_factory=dict)
    truncated_walks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "rows": self.rows,
            "terminal_states": dict(self.terminal_states),
            "truncated_walks": self.truncated_walks,
        }


def validate_workflow(workflow: WorkflowConfig, columns: Sequence[str]) -> List[str]:
    """Check a workflow against a table's columns.

    Returns human-readable problems. Empty list means it is executable.
    Called by ``lint`` so a broken workflow is caught before generation
    rather than producing quietly wrong data.
    """
    problems: List[str] = []

    if not workflow.transitions:
        problems.append(f"workflow {workflow.name!r} has no transitions")

    if workflow.state_column not in columns:
        problems.append(
            f"workflow {workflow.name!r}: state_column {workflow.state_column!r} "
            f"is not a column of {workflow.table!r}"
        )

    for column, state in workflow.timestamps.items():
        if column not in columns:
            problems.append(
                f"workflow {workflow.name!r}: timestamp column {column!r} "
                f"is not a column of {workflow.table!r}"
            )
        if state not in workflow.states:
            problems.append(
                f"workflow {workflow.name!r}: timestamp column {column!r} maps to "
                f"state {state!r}, which no transition mentions"
            )

    start = workflow.resolve_start_state()
    if start is None:
        problems.append(
            f"workflow {workflow.name!r}: cannot infer start_state — declare it "
            f"explicitly (no single state is absent from every transition target)"
        )
    elif workflow.start_state and workflow.start_state not in workflow.states:
        problems.append(
            f"workflow {workflow.name!r}: start_state {workflow.start_state!r} "
            f"appears in no transition"
        )

    for state in workflow.states:
        outgoing = [t for t in workflow.transitions if t.from_state == state]
        total = sum(t.probability for t in outgoing)
        if total > 1.0 + 1e-9:
            problems.append(
                f"workflow {workflow.name!r}: transitions out of {state!r} sum to "
                f"{total:.3f}, which is above 1.0"
            )

    if len(workflow.step_hours) != 2 or workflow.step_hours[0] > workflow.step_hours[1]:
        problems.append(
            f"workflow {workflow.name!r}: step_hours must be [min, max] with min <= max"
        )

    return problems


def walk(workflow: WorkflowConfig, rng: Any) -> Tuple[List[str], bool]:
    """Walk one row through the state machine.

    Returns ``(path, truncated)``. ``truncated`` is True when the walk hit
    ``MAX_STEPS``, which only happens on a cyclic definition.
    """
    start = workflow.resolve_start_state()
    if start is None:
        raise WorkflowError(
            f"workflow {workflow.name!r}: cannot infer start_state — declare it explicitly"
        )

    outgoing: Dict[str, List] = {}
    for transition in workflow.transitions:
        outgoing.setdefault(transition.from_state, []).append(transition)

    path = [start]
    state = start
    for _ in range(MAX_STEPS):
        options = outgoing.get(state) or []
        if not options:
            return path, False                       # terminal state

        roll = float(rng.random())
        cumulative = 0.0
        for transition in options:
            cumulative += transition.probability
            if roll < cumulative:
                state = transition.to_state
                path.append(state)
                break
        else:
            # The roll landed in the residual probability mass: the row stops
            # here. This is what makes `probability` sums below 1.0 meaningful.
            return path, False

    return path, True


def _timestamps_for_path(
    workflow: WorkflowConfig,
    path: Sequence[str],
    rng: Any,
) -> Dict[str, Optional[datetime]]:
    """Assign an increasing timestamp to each visited state's column."""
    state_to_columns: Dict[str, List[str]] = {}
    for column, state in workflow.timestamps.items():
        state_to_columns.setdefault(state, []).append(column)

    values: Dict[str, Optional[datetime]] = {c: None for c in workflow.timestamps}
    if not workflow.timestamps:
        return values

    current = _start_datetime(workflow, rng)
    low, high = float(workflow.step_hours[0]), float(workflow.step_hours[1])

    for index, state in enumerate(path):
        if index > 0:
            gap = low if high <= low else float(rng.uniform(low, high))
            current = current + timedelta(hours=gap)
        for column in state_to_columns.get(state, []):
            # A state revisited by a cycle overwrites with the later time,
            # which is the correct reading of "when did this last happen".
            values[column] = current

    return values


def _start_datetime(workflow: WorkflowConfig, rng: Any) -> datetime:
    """First timestamp, drawn from ``start_between`` or the last year."""
    if workflow.start_between and len(workflow.start_between) == 2:
        try:
            low = datetime.fromisoformat(str(workflow.start_between[0]))
            high = datetime.fromisoformat(str(workflow.start_between[1]))
            if high > low:
                span = (high - low).total_seconds()
                return low + timedelta(seconds=float(rng.uniform(0, span)))
            return low
        except (TypeError, ValueError) as exc:
            logger.warning(
                "workflow %s: start_between %r is not a pair of ISO dates (%s) — "
                "using the default window", workflow.name, workflow.start_between, exc,
            )

    end = datetime.now().replace(microsecond=0)
    start = end - timedelta(days=_DEFAULT_START_DAYS_BACK)
    span = (end - start).total_seconds()
    return start + timedelta(seconds=float(rng.uniform(0, span)))


def apply_workflow(
    df: "object",                                   # pandas.DataFrame
    workflow: WorkflowConfig,
    rng: Any,
) -> Tuple["object", WorkflowStats]:
    """Apply one workflow to a table, returning the frame and its stats.

    The state column and every mapped timestamp column are overwritten:
    whatever generation produced for them is replaced by a consistent
    lifecycle. Columns not named by the workflow are untouched.
    """
    import pandas as pd

    stats = WorkflowStats(name=workflow.name, table=workflow.table, rows=len(df))
    if df.empty:
        return df, stats

    missing = [c for c in [workflow.state_column, *workflow.timestamps]
               if c not in df.columns]
    if missing:
        raise WorkflowError(
            f"workflow {workflow.name!r}: table {workflow.table!r} has no "
            f"column(s) {', '.join(missing)}"
        )

    out = df.copy()
    states: List[str] = []
    stamp_columns: Dict[str, List[Optional[datetime]]] = {
        c: [] for c in workflow.timestamps
    }

    for _ in range(len(out)):
        path, truncated = walk(workflow, rng)
        if truncated:
            stats.truncated_walks += 1
        terminal = path[-1]
        states.append(terminal)
        stats.terminal_states[terminal] = stats.terminal_states.get(terminal, 0) + 1

        for column, value in _timestamps_for_path(workflow, path, rng).items():
            stamp_columns[column].append(value)

    out[workflow.state_column] = states
    for column, values in stamp_columns.items():
        # Preserve the column's original dtype family where the source was a
        # date rather than a datetime.
        series = pd.Series(values, index=out.index, dtype="datetime64[ns]")
        if _is_date_only(df[column]):
            series = series.dt.normalize()
        out[column] = series

    if stats.truncated_walks:
        logger.warning(
            "workflow %s: %d walk(s) hit the %d-step cap — the definition is "
            "cyclic and some rows were cut short",
            workflow.name, stats.truncated_walks, MAX_STEPS,
        )

    return out, stats


def _is_date_only(series: "object") -> bool:
    """Whether a column holds dates rather than datetimes."""
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        return bool((non_null.dt.normalize() == non_null).all())
    return False


def apply_workflows(
    data: Dict[str, "object"],
    workflows: Sequence[WorkflowConfig],
    rng: Any,
) -> Tuple[Dict[str, "object"], List[WorkflowStats]]:
    """Apply every workflow to its table.

    A workflow naming an absent table is skipped with a warning rather than
    failing the run — the table may simply be inactive in this config.
    """
    all_stats: List[WorkflowStats] = []
    for workflow in workflows:
        if workflow.table not in data:
            logger.warning(
                "workflow %s targets table %r, which was not generated — skipping",
                workflow.name, workflow.table,
            )
            continue
        data[workflow.table], stats = apply_workflow(
            data[workflow.table], workflow, rng,
        )
        all_stats.append(stats)
        logger.info(
            "workflow %s applied to %s: %s",
            workflow.name, workflow.table,
            ", ".join(f"{state}={count}"
                      for state, count in sorted(stats.terminal_states.items())),
        )
    return data, all_stats
