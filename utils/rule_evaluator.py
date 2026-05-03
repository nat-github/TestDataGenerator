"""Rule evaluator for Layer A (when/then) and Layer B (derived) column logic.

Operates on plain dicts (one row at a time), so the same engine can later be
reused by the wire-mock / stub-renderer track to apply business logic to
response payloads. No DataFrame / Pandas dependency lives in this module.

Public API:
    evaluate_when(condition, row)              -> bool
    apply_when_then(rule, row)                 -> row mutated in place
    evaluate_derived(expression, row)          -> Any
    topo_sort_derived(columns)                 -> List[ColumnConfig]
    apply_to_dataframe(df, table_config)       -> pd.DataFrame
"""
from __future__ import annotations

import ast
import datetime as _dt
import logging
import operator
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from models.config_models import ColumnConfig, RuleConfig, TableConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer A — when/then evaluation
# ---------------------------------------------------------------------------

_NULL_SENTINEL = object()


def _coerce_for_compare(left: Any, right: Any) -> Tuple[Any, Any]:
    """Make a best-effort numeric comparison work even when one side is a
    string and the other a number (common in YAML / JSON authoring)."""
    if left is None or right is None:
        return left, right
    if isinstance(left, bool) or isinstance(right, bool):
        return left, right
    if isinstance(left, (int, float)) and isinstance(right, str):
        try:
            return left, float(right)
        except ValueError:
            return left, right
    if isinstance(right, (int, float)) and isinstance(left, str):
        try:
            return float(left), right
        except ValueError:
            return left, right
    return left, right


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        try:
            return v != v  # NaN check
        except Exception:
            return False
    return False


_BIN_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def _check_atom(column_value: Any, op: str, target: Any) -> bool:
    """Evaluate a single operator against a column value."""
    if op in _BIN_OPS:
        a, b = _coerce_for_compare(column_value, target)
        try:
            return bool(_BIN_OPS[op](a, b))
        except TypeError:
            return False
    if op == "in":
        if isinstance(target, (list, tuple, set)):
            return column_value in target
        return False
    if op == "not_in":
        if isinstance(target, (list, tuple, set)):
            return column_value not in target
        return True
    if op == "between":
        if isinstance(target, (list, tuple)) and len(target) == 2:
            lo, hi = target
            a, _ = _coerce_for_compare(column_value, lo)
            _, b = _coerce_for_compare(a, hi)
            try:
                return lo <= a <= hi if not isinstance(a, str) else lo <= column_value <= hi
            except TypeError:
                return False
        return False
    if op == "is_null":
        result = _is_null(column_value)
        return result if target else not result
    if op == "not_null":
        result = not _is_null(column_value)
        return result if target else not result
    if op == "matches":
        if column_value is None or not isinstance(target, str):
            return False
        try:
            return re.search(target, str(column_value)) is not None
        except re.error:
            return False
    logger.warning("rule_evaluator: unknown operator %r", op)
    return False


def evaluate_when(condition: Any, row: Dict[str, Any]) -> bool:
    """Recursive evaluator for `when:` blocks.

    A condition is a dict whose keys are either column names (mapped to an
    operator dict) or composite operators `and` / `or` whose values are lists
    of further conditions. An empty / missing condition matches everything.
    """
    if not condition:
        return True
    if not isinstance(condition, dict):
        return False

    for key, value in condition.items():
        if key == "and":
            if not isinstance(value, list) or not all(evaluate_when(c, row) for c in value):
                return False
            continue
        if key == "or":
            if not isinstance(value, list) or not any(evaluate_when(c, row) for c in value):
                return False
            continue
        # column-name key — value should be {op: target} or a literal for shorthand eq
        column_value = row.get(key)
        if isinstance(value, dict):
            for op, target in value.items():
                if not _check_atom(column_value, op, target):
                    return False
        else:
            # shorthand — `{ status: ACTIVE }` means `{ status: { eq: ACTIVE } }`
            if not _check_atom(column_value, "eq", value):
                return False
    return True


def apply_when_then(
    rule: RuleConfig,
    column_name: str,
    row: Dict[str, Any],
    column_config: Optional[ColumnConfig] = None,
    helpers: Any = None,
) -> bool:
    """If the rule's `when:` matches, apply the `then:` action to `row[column_name]`.

    Returns True when the rule fired (regardless of whether it changed anything).
    Mutates `row` in place.
    """
    if not evaluate_when(rule.when, row):
        return False

    action = rule.then

    # Explicit null wins over everything else.
    if action.set_null:
        row[column_name] = None
        return True

    # Constant value override.
    if action.value is not None:
        row[column_name] = action.value
        return True

    # Re-sample using overridden constraints. Requires helpers + column_config
    # to know the data type. Falls back to no-op if helpers are unavailable
    # (e.g. when the evaluator is invoked from a stub-payload context).
    if helpers is None or column_config is None:
        return True

    try:
        new_value = _resample_value(action, column_config, helpers)
        if new_value is not _NULL_SENTINEL:
            row[column_name] = new_value
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("rule_evaluator: resample failed for %s: %s", column_name, exc)

    return True


def _resample_value(action: Any, column_config: ColumnConfig, helpers: Any) -> Any:
    """Generate a single value honoring the action's constraint overrides.

    Tries (in order):
      1. null_rate roll
      2. business_values pick
      3. min/max numeric range using base data type as a hint
      4. helpers.generate_value_for_column (if present — future hook)
      5. _NULL_SENTINEL meaning "leave the original value"
    """
    import random

    if action.null_rate is not None:
        try:
            if random.random() < float(action.null_rate):
                return None
        except (TypeError, ValueError):
            pass

    # Business values list takes priority — pick uniformly.
    if action.business_values is not None:
        choices = action.business_values
        if isinstance(choices, str):
            choices = [c.strip() for c in choices.replace(',', ';').split(';') if c.strip()]
        if isinstance(choices, (list, tuple)) and choices:
            return random.choice(list(choices))

    # Numeric range override — choose int vs float from the column's data_type.
    if action.min is not None or action.max is not None:
        try:
            lo = float(action.min) if action.min is not None else None
            hi = float(action.max) if action.max is not None else None
            if lo is not None and hi is not None and lo <= hi:
                base = (column_config.data_type or '').upper()
                if base.startswith('N') or base.startswith('I'):
                    return random.randint(int(lo), int(hi))
                return round(random.uniform(lo, hi), 2)
        except (TypeError, ValueError):
            pass

    # Future hook: if helpers exposes a single-value generator, prefer it.
    fn = getattr(helpers, 'generate_value_for_column', None)
    if callable(fn):
        try:
            cc = column_config.model_copy(update={
                k: v for k, v in {
                    'min_value': action.min,
                    'max_value': action.max,
                    'distribution': action.distribution,
                    'business_values': action.business_values,
                    'special_rules': action.special_rules,
                }.items() if v is not None
            })
            return fn(cc)
        except Exception:
            return _NULL_SENTINEL

    return _NULL_SENTINEL


# ---------------------------------------------------------------------------
# Layer B — derived expressions
# ---------------------------------------------------------------------------

# Whitelist of AST nodes safe to evaluate. Anything else raises.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.IfExp, ast.Call, ast.Tuple, ast.List, ast.Subscript, ast.Index, ast.Slice,
    ast.JoinedStr, ast.FormattedValue,
)


def _today() -> _dt.date:
    return _dt.date.today()


def _years_between(d1: Any, d2: Any) -> Optional[int]:
    a = _to_date(d1)
    b = _to_date(d2)
    if a is None or b is None:
        return None
    diff = abs(b.year - a.year)
    if (b.month, b.day) < (a.month, a.day):
        diff -= 1
    return diff


def _to_date(v: Any) -> Optional[_dt.date]:
    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return _dt.datetime.strptime(v, fmt).date()
            except ValueError:
                continue
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime().date()
    return None


def _coalesce(*values: Any) -> Any:
    for v in values:
        if not _is_null(v):
            return v
    return None


def _safe_concat(*values: Any) -> str:
    return "".join("" if v is None else str(v) for v in values)


# Read-only function registry for derived expressions.
_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "today": _today,
    "years_between": _years_between,
    "lower": lambda s: ("" if s is None else str(s)).lower(),
    "upper": lambda s: ("" if s is None else str(s)).upper(),
    "concat": _safe_concat,
    "coalesce": _coalesce,
    "len": lambda s: 0 if s is None else len(s),
    "int": lambda x: int(x) if x is not None and x != "" else None,
    "float": lambda x: float(x) if x is not None and x != "" else None,
    "str": lambda x: "" if x is None else str(x),
    "round": lambda x, n=0: None if x is None else round(x, n),
    "abs": lambda x: None if x is None else abs(x),
    "min": min,
    "max": max,
}


def _eval_node(node: ast.AST, row: Dict[str, Any]) -> Any:
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"Disallowed expression construct: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, row)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _FUNCTIONS:
            return _FUNCTIONS[node.id]
        if node.id in row:
            return row[node.id]
        # Unknown bare name: treat as None so simple typos don't crash a run.
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, row)
        right = _eval_node(node.right, row)
        op = node.op
        if isinstance(op, ast.Add): return left + right
        if isinstance(op, ast.Sub): return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div): return left / right
        if isinstance(op, ast.FloorDiv): return left // right
        if isinstance(op, ast.Mod): return left % right
        if isinstance(op, ast.Pow): return left ** right
        raise ValueError(f"Unsupported binop: {type(op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, row)
        if isinstance(node.op, ast.USub): return -operand
        if isinstance(node.op, ast.UAdd): return +operand
        if isinstance(node.op, ast.Not): return not operand
        raise ValueError(f"Unsupported unaryop: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, row) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, row)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, row)
            ok = _check_compare(op, left, right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        cond = _eval_node(node.test, row)
        return _eval_node(node.body if cond else node.orelse, row)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        args = [_eval_node(a, row) for a in node.args]
        kwargs = {kw.arg: _eval_node(kw.value, row) for kw in node.keywords}
        return func(*args, **kwargs)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(e, row) for e in node.elts]
    if isinstance(node, ast.JoinedStr):
        return "".join(str(_eval_node(v, row)) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return _eval_node(node.value, row)
    raise ValueError(f"Unsupported node: {type(node).__name__}")


def _check_compare(op: ast.AST, left: Any, right: Any) -> bool:
    if isinstance(op, ast.Eq): return left == right
    if isinstance(op, ast.NotEq): return left != right
    if isinstance(op, ast.Lt): return left < right
    if isinstance(op, ast.LtE): return left <= right
    if isinstance(op, ast.Gt): return left > right
    if isinstance(op, ast.GtE): return left >= right
    if isinstance(op, ast.In): return left in right
    if isinstance(op, ast.NotIn): return left not in right
    raise ValueError(f"Unsupported compare op: {type(op).__name__}")


_TEMPLATE_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def evaluate_derived(expression: str, row: Dict[str, Any]) -> Any:
    """Evaluate a derived-column expression.

    Two modes:
      * Template:    "{first_name} {last_name}" -> str.format-like substitution
      * Expression:  "={qty} * {price}"        -> safe AST eval after substituting

    For expression mode the leading `=` is stripped, then `{col}` placeholders
    are rewritten to bare column names so the AST evaluator can resolve them.
    """
    if expression is None:
        return None
    expr = str(expression).strip()
    if not expr:
        return None

    if expr.startswith("="):
        # Replace `{col}` placeholders with bare names so they parse as Name nodes.
        body = _TEMPLATE_PATTERN.sub(r"\1", expr[1:].strip())
        try:
            tree = ast.parse(body, mode="eval")
        except SyntaxError as exc:
            logger.warning("rule_evaluator: bad expression %r: %s", expression, exc)
            return None
        try:
            return _eval_node(tree, row)
        except Exception as exc:
            logger.debug("rule_evaluator: eval failed for %r: %s", expression, exc)
            return None

    # Template mode — simple substitution, missing keys become empty.
    def _sub(match: re.Match) -> str:
        key = match.group(1)
        v = row.get(key)
        return "" if v is None else str(v)

    return _TEMPLATE_PATTERN.sub(_sub, expr)


# ---------------------------------------------------------------------------
# Topological sort of derived columns
# ---------------------------------------------------------------------------


def topo_sort_derived(columns: List[ColumnConfig]) -> List[ColumnConfig]:
    """Return derived columns in dependency order. Non-derived columns
    are not returned (caller already has them generated)."""
    derived = [c for c in columns if c.derived]
    if not derived:
        return []

    name_to_col = {c.column_name: c for c in derived}
    deps: Dict[str, List[str]] = {}
    for c in derived:
        refs = _TEMPLATE_PATTERN.findall(str(c.derived))
        deps[c.column_name] = [r for r in refs if r in name_to_col]

    ordered: List[ColumnConfig] = []
    visiting: set = set()
    visited: set = set()

    def visit(name: str):
        if name in visited:
            return
        if name in visiting:
            # cycle — break by treating as already visited; logged once
            logger.warning("rule_evaluator: derived cycle around %r", name)
            return
        visiting.add(name)
        for dep in deps.get(name, []):
            visit(dep)
        visiting.discard(name)
        visited.add(name)
        ordered.append(name_to_col[name])

    for c in derived:
        visit(c.column_name)
    return ordered


# ---------------------------------------------------------------------------
# DataFrame integration — entry point used by data_generator
# ---------------------------------------------------------------------------


def apply_to_dataframe(
    df: pd.DataFrame,
    table_config: TableConfig,
    helpers: Any = None,
) -> pd.DataFrame:
    """Apply Layer A (rules) and Layer B (derived) to every row of `df`.

    Returns a new DataFrame; original is not mutated. No-op when the table
    has no rules / derived columns. Designed to run after FK resolution.
    """
    if df is None or df.empty:
        return df

    has_rules = any(c.rules for c in table_config.columns)
    derived_order = topo_sort_derived(table_config.columns)

    if not has_rules and not derived_order:
        return df

    columns_by_name = {c.column_name: c for c in table_config.columns}

    out = df.copy()
    records = out.to_dict(orient='records')

    for row in records:
        # Layer A — rules first, so derived columns see post-rule values.
        if has_rules:
            for col in table_config.columns:
                if not col.rules:
                    continue
                for rule in col.rules:
                    apply_when_then(rule, col.column_name, row,
                                    column_config=col, helpers=helpers)
        # Layer B — derived columns in topo order.
        for dcol in derived_order:
            row[dcol.column_name] = evaluate_derived(dcol.derived, row)

    return pd.DataFrame.from_records(records, columns=out.columns)
