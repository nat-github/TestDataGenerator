"""Breaking-change detection between two contract versions.

A *breaking change* is one that will break existing consumers — a removed
column, a narrowed type, a shrunk enum. An *additive* change is safe. A
*review* change is ambiguous (a rename looks like a drop + add) and needs a
human. This is a pure structural comparison — no Great Expectations, no data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sdp.contracts.model import ChangeClass, ContractChange, ContractDiff

# String-type capacity ladders — used to tell a narrowing from a widening.
_TYPE_FAMILIES: Dict[str, Dict[str, int]] = {
    "varchar": {"VA1": 1, "VA3": 3, "VA256": 256},
    "alpha": {"A1": 1, "A5": 5},
}


def _type_family(data_type: Optional[str]) -> Optional[str]:
    dt = (data_type or "").strip().upper()
    for family, members in _TYPE_FAMILIES.items():
        if dt in members:
            return family
    return None


def _type_capacity(data_type: Optional[str]) -> Optional[int]:
    dt = (data_type or "").strip().upper()
    for members in _TYPE_FAMILIES.values():
        if dt in members:
            return members[dt]
    return None


def _business_values(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw if str(v).strip()]
    return [v.strip() for v in str(raw).replace(",", ";").split(";") if v.strip()]


def _columns_by_name(table_cfg: Any) -> Dict[str, Any]:
    return {c.column_name: c for c in getattr(table_cfg, "columns", [])}


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _diff_column(table: str, old_col: Any, new_col: Any) -> List[ContractChange]:
    """Compare one column across versions."""
    changes: List[ContractChange] = []
    target = f"{table}.{old_col.column_name}"

    # --- data type ---
    old_dt = (old_col.data_type or "").upper()
    new_dt = (new_col.data_type or "").upper()
    if old_dt != new_dt:
        old_cap, new_cap = _type_capacity(old_dt), _type_capacity(new_dt)
        same_family = _type_family(old_dt) and _type_family(old_dt) == _type_family(new_dt)
        if same_family and old_cap is not None and new_cap is not None:
            if new_cap < old_cap:
                changes.append(ContractChange(
                    "type_narrowed", target, ChangeClass.BREAKING,
                    f"type narrowed {old_dt} -> {new_dt}",
                ))
            else:
                changes.append(ContractChange(
                    "type_widened", target, ChangeClass.ADDITIVE,
                    f"type widened {old_dt} -> {new_dt}",
                ))
        else:
            changes.append(ContractChange(
                "type_changed", target, ChangeClass.REVIEW,
                f"type changed {old_dt} -> {new_dt} — verify consumer compatibility",
            ))

    # --- primary key ---
    if bool(old_col.is_pk) != bool(new_col.is_pk):
        changes.append(ContractChange(
            "primary_key_changed", target, ChangeClass.BREAKING,
            f"is_pk {old_col.is_pk} -> {new_col.is_pk}",
        ))

    # --- nullability ---
    if bool(old_col.nullable) != bool(new_col.nullable):
        if old_col.nullable and not new_col.nullable:
            changes.append(ContractChange(
                "nullability_tightened", target, ChangeClass.REVIEW,
                "column became NOT NULL — safe only if data already has no nulls",
            ))
        else:
            changes.append(ContractChange(
                "nullability_relaxed", target, ChangeClass.ADDITIVE,
                "column now allows NULLs",
            ))

    # --- business values (enum) ---
    old_bv, new_bv = set(_business_values(old_col.business_values)), set(_business_values(new_col.business_values))
    if old_bv or new_bv:
        removed = old_bv - new_bv
        added = new_bv - old_bv
        if removed:
            changes.append(ContractChange(
                "enum_shrunk", target, ChangeClass.BREAKING,
                f"allowed values removed: {sorted(removed)}",
            ))
        if added:
            changes.append(ContractChange(
                "enum_expanded", target, ChangeClass.ADDITIVE,
                f"allowed values added: {sorted(added)}",
            ))

    # --- numeric range ---
    old_min, new_min = _as_float(old_col.min_value), _as_float(new_col.min_value)
    old_max, new_max = _as_float(old_col.max_value), _as_float(new_col.max_value)
    if (new_min is not None and (old_min is None or new_min > old_min)) or \
       (new_max is not None and (old_max is None or new_max < old_max)):
        changes.append(ContractChange(
            "range_narrowed", target, ChangeClass.BREAKING,
            f"value range tightened ([{old_min}, {old_max}] -> [{new_min}, {new_max}])",
        ))
    elif (old_min is not None and (new_min is None or new_min < old_min)) or \
         (old_max is not None and (new_max is None or new_max > old_max)):
        changes.append(ContractChange(
            "range_widened", target, ChangeClass.ADDITIVE,
            f"value range widened ([{old_min}, {old_max}] -> [{new_min}, {new_max}])",
        ))

    # --- length ---
    old_len, new_len = old_col.length, new_col.length
    if old_len != new_len:
        if new_len is not None and (old_len is None or new_len < old_len):
            changes.append(ContractChange(
                "length_reduced", target, ChangeClass.BREAKING,
                f"max length reduced ({old_len} -> {new_len})",
            ))
        else:
            changes.append(ContractChange(
                "length_increased", target, ChangeClass.ADDITIVE,
                f"max length increased ({old_len} -> {new_len})",
            ))

    return changes


def diff_contracts(
    old_tables: Dict[str, Any],
    new_tables: Dict[str, Any],
    *,
    old_name: str = "old",
    new_name: str = "new",
) -> ContractDiff:
    """Compare two contracts (``Dict[str, TableConfig]``) and classify changes.

    Returns a :class:`~sdp.contracts.model.ContractDiff` listing every change as
    breaking, additive, or review.
    """
    diff = ContractDiff(old_name=old_name, new_name=new_name)

    old_names, new_names = set(old_tables), set(new_tables)

    for removed in sorted(old_names - new_names):
        diff.changes.append(ContractChange(
            "table_removed", removed, ChangeClass.BREAKING,
            "table removed from the contract",
        ))
    for added in sorted(new_names - old_names):
        diff.changes.append(ContractChange(
            "table_added", added, ChangeClass.ADDITIVE,
            "new table added to the contract",
        ))

    for table in sorted(old_names & new_names):
        old_cols = _columns_by_name(old_tables[table])
        new_cols = _columns_by_name(new_tables[table])

        for removed in sorted(set(old_cols) - set(new_cols)):
            diff.changes.append(ContractChange(
                "column_removed", f"{table}.{removed}", ChangeClass.BREAKING,
                "column removed from the contract",
            ))
        for added in sorted(set(new_cols) - set(old_cols)):
            col = new_cols[added]
            if not col.nullable:
                diff.changes.append(ContractChange(
                    "required_column_added", f"{table}.{added}", ChangeClass.BREAKING,
                    "new NOT NULL column added — existing producers won't populate it",
                ))
            else:
                diff.changes.append(ContractChange(
                    "column_added", f"{table}.{added}", ChangeClass.ADDITIVE,
                    "new optional column added",
                ))
        for common in sorted(set(old_cols) & set(new_cols)):
            diff.changes.extend(_diff_column(table, old_cols[common], new_cols[common]))

    return diff


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

_CLASS_MARK = {
    ChangeClass.BREAKING: "BREAKING",
    ChangeClass.ADDITIVE: "additive",
    ChangeClass.REVIEW: "review",
}


def format_contract_diff(diff: ContractDiff) -> str:
    """Render a :class:`ContractDiff` as a human-readable text block."""
    lines = []
    lines.append("=" * 64)
    lines.append(f"Contract diff — {diff.old_name}  ->  {diff.new_name}")
    lines.append("=" * 64)
    lines.append(
        f"  breaking={len(diff.breaking)}  additive={len(diff.additive)}  "
        f"review={len(diff.review)}"
    )
    if not diff.changes:
        lines.append("  no changes — contracts are identical")
        lines.append("")
        return "\n".join(lines)
    lines.append("")
    # Breaking first — that's what matters.
    for group in (ChangeClass.BREAKING, ChangeClass.REVIEW, ChangeClass.ADDITIVE):
        items = [c for c in diff.changes if c.classification is group]
        for c in items:
            lines.append(f"  [{_CLASS_MARK[group]:>8}] {c.target} — {c.kind}: {c.detail}")
    lines.append("")
    if diff.has_breaking:
        lines.append(f"RESULT: {len(diff.breaking)} breaking change(s) — consumers will break.")
    else:
        lines.append("RESULT: no breaking changes — safe to publish.")
    lines.append("")
    return "\n".join(lines)
