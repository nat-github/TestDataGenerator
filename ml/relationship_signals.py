"""Pure-function signal computers used by the ML relationship inferrer.

Each signal returns a score in [0, 1]. They're separated from the orchestrator
so they can be unit-tested independently and reweighted per-deployment.

Four signals:

  type_compatibility(left, right) -> 0/1   (gate, not weighted)
  name_similarity(left, right)    -> [0, 1]
  value_subset(left_values, right_values) -> [0, 1]
  pk_likeness(values, total_rows) -> [0, 1]

Optional fifth helper:

  combine(signals, weights) -> [0, 1]   (default weighted sum)
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Signal 1 — Type compatibility (gate)
# ---------------------------------------------------------------------------

# Map data-type prefixes to broad type families. FK pairs must share a family.
_TYPE_FAMILIES: Dict[str, str] = {
    "N": "numeric",   # N10, N19, N38
    "I": "numeric",   # int variants
    "D": "datetime",  # D, DT, DC ...  see exception below
    "T": "datetime",  # TS
    "V": "string",    # VA1, VA10, VA256
    "A": "string",    # A1, A34
    "S": "string",
    "C": "string",    # char
}


def _family(data_type: Optional[str]) -> Optional[str]:
    """Return the type family for a data-type string, or None if unknown."""
    if not data_type:
        return None
    s = str(data_type).strip().upper()
    if not s:
        return None
    # Special case: DC = decimal numeric, not datetime
    if s.startswith("DC"):
        return "numeric"
    return _TYPE_FAMILIES.get(s[0])


def type_compatibility(left_type: Optional[str], right_type: Optional[str]) -> float:
    """Return 1.0 if both columns belong to the same type family, else 0.0.

    Acts as a gate — incompatible types contribute 0 to confidence regardless
    of name similarity or value subset.
    """
    lf = _family(left_type)
    rf = _family(right_type)
    if lf is None or rf is None:
        return 0.0
    return 1.0 if lf == rf else 0.0


# ---------------------------------------------------------------------------
# Signal 2 — Name similarity
# ---------------------------------------------------------------------------

# Tokens that don't carry semantic weight in column names.
_STOPWORDS = {"id", "key", "code", "num", "no", "nbr", "nb", "fk", "pk", "ref"}

_TOKEN_SPLIT = re.compile(r"[_\W]+|(?<=[a-z])(?=[A-Z])")


def _tokenize(name: str) -> list[str]:
    """Split a column name into lowercase tokens.

    Handles snake_case, kebab-case, camelCase, and PascalCase uniformly.
    """
    if not name:
        return []
    raw = _TOKEN_SPLIT.split(str(name))
    return [t.lower() for t in raw if t]


def _levenshtein(a: str, b: str) -> int:
    """Iterative DP Levenshtein distance. O(len(a) * len(b)) time, O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[-1]


def _string_ratio(a: str, b: str) -> float:
    """1 - normalized Levenshtein. Both empty -> 1.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = _levenshtein(a.lower(), b.lower())
    return 1.0 - (d / max(len(a), len(b)))


def name_similarity(left_name: str, right_name: str) -> float:
    """Combine token-level Jaccard similarity with character-level Levenshtein.

    The result is the weighted max — token Jaccard captures `customer_id` ↔
    `cust_id` (shared `id`, partial `customer`/`cust`), Levenshtein captures
    typos and abbreviations like `qty` ↔ `quantity`.

    Stopwords (`id`, `key`, `code`, ...) are removed before token comparison
    so that `customer_id` ↔ `order_id` doesn't get a free 0.5 bump just from
    sharing `id`.
    """
    if not left_name or not right_name:
        return 0.0

    a_tokens = [t for t in _tokenize(left_name) if t not in _STOPWORDS]
    b_tokens = [t for t in _tokenize(right_name) if t not in _STOPWORDS]

    if not a_tokens and not b_tokens:
        # Both names were entirely stopwords (e.g. `id` ↔ `id`). Defer to
        # raw string ratio — that scores 1.0 for identical, less for typos.
        return _string_ratio(left_name, right_name)

    a_set, b_set = set(a_tokens), set(b_tokens)
    if a_set or b_set:
        token_score = len(a_set & b_set) / max(1, len(a_set | b_set))
    else:
        token_score = 0.0

    # Per-token best-match — handles abbreviations (cust ~ customer)
    fuzzy_token_score = 0.0
    if a_tokens and b_tokens:
        per_token = [
            max(_string_ratio(at, bt) for bt in b_tokens) for at in a_tokens
        ]
        fuzzy_token_score = sum(per_token) / len(per_token)

    char_score = _string_ratio(left_name, right_name)

    return max(token_score, fuzzy_token_score, char_score)


# ---------------------------------------------------------------------------
# Signal 3 — Value subset / containment
# ---------------------------------------------------------------------------


def value_subset(child_values: Iterable, parent_values: Iterable) -> float:
    """Fraction of distinct child values that exist in the parent's distinct set.

    Returns 1.0 when every child value has a matching parent — strong FK signal.
    Returns 0.0 when there's no overlap. Empty child or empty parent -> 0.0
    (no information either way).

    Handles type coercion gracefully: integers stored as floats by Pandas, or
    integer strings vs ints, are normalised to a comparable form so that
    `123` and `123.0` and `"123"` all match.
    """
    def _norm(v):
        if v is None:
            return None
        if isinstance(v, float):
            if v != v:  # NaN
                return None
            if v.is_integer():
                return int(v)
            return v
        if isinstance(v, bool):
            return v
        return v

    child_set = {_norm(v) for v in child_values if _norm(v) is not None}
    parent_set = {_norm(v) for v in parent_values if _norm(v) is not None}
    if not child_set or not parent_set:
        return 0.0

    # Try string-coerced comparison too — protects against int/str type drift.
    direct = len(child_set & parent_set) / len(child_set)
    if direct == 1.0:
        return 1.0

    child_str = {str(v) for v in child_set}
    parent_str = {str(v) for v in parent_set}
    coerced = len(child_str & parent_str) / len(child_str)
    return max(direct, coerced)


# ---------------------------------------------------------------------------
# Signal 4 — PK-likeness (uniqueness x cardinality)
# ---------------------------------------------------------------------------


def pk_likeness(
    values: Sequence,
    total_rows: Optional[int] = None,
    is_pk_declared: bool = False,
) -> float:
    """Score how "primary-key-like" a column's values are.

    The parent side of a FK relationship should have unique high-cardinality
    values. This signal computes:

        unique_ratio  = distinct(values) / total_rows
        cardinality_score = min(distinct / 100, 1.0)
        score = 0.6 * unique_ratio + 0.4 * cardinality_score

    A declared PK gets a fast-path 1.0 since the schema already tells us it's
    a unique identifier.
    """
    if is_pk_declared:
        return 1.0
    n = total_rows if total_rows is not None else len(list(values))
    if not n:
        return 0.0
    distinct = len({v for v in values if v is not None})
    if distinct == 0:
        return 0.0
    unique_ratio = distinct / n
    cardinality_score = min(distinct / 100.0, 1.0)
    return round(0.6 * unique_ratio + 0.4 * cardinality_score, 4)


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

# Default weights, derived empirically. Subject to change as feedback
# accumulates and the learned classifier overrides this with its own weights.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "name_similarity": 0.40,
    "value_subset":    0.40,
    "pk_likeness":     0.20,
}


def combine(signals: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Weighted sum of named signals.

    `type_compatibility` is treated as a hard gate — if it's present and 0,
    the combined score is 0 regardless of other signals.
    """
    weights = weights or DEFAULT_WEIGHTS
    if signals.get("type_compatibility", 1.0) == 0.0:
        return 0.0
    score = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        if name in signals:
            score += weight * float(signals[name])
            total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(min(score / total_weight, 1.0), 4)
