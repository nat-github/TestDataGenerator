"""Bias and fairness drift between source and synthetic data.

Synthetic data inherits the skew of whatever it was trained on — and can
make it worse. The literature (Goyal & Mahmoud 2024, §6) lists bias
amplification as one of the six open challenges in the field, and it is
the one with regulatory teeth for financial data.

Two distinct questions are answered here, because they fail differently:

**Representation drift** — is each group present in the same proportion
as in the real data? A status that is 8% of real rows but 2% of synthetic
rows means downstream tests barely exercise that path.

**Outcome disparity amplification** — given some outcome column, does the
gap in outcome rates *between* groups widen in synthetic data? Source
data where group A is approved 60% of the time and group B 40% has a
20-point disparity. If synthetic data stretches that to 35 points, the
generator has invented discrimination that was not in the original.

Representation can look perfect while outcomes are badly skewed, so
neither measure substitutes for the other.

Descriptive, not prescriptive: this reports what changed. Whether a shift
matters is a domain judgement.

pandas only — no optional dependencies, no model fitting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Categoricals with more levels than this are identifiers or free text,
# not groups worth measuring parity across.
_MAX_GROUPS = 50

# Representation shift (in share, 0..1) at which a group is flagged.
_SHIFT_FLAG = 0.05

# Groups thinner than this in the source are too rare for their synthetic
# share to be meaningful — reported, but never flagged.
_MIN_GROUP_SHARE = 0.01


@dataclass
class GroupRepresentation:
    """How one group's share of the table changed."""
    group: str
    source_share: float
    synthetic_share: float

    @property
    def delta(self) -> float:
        """Percentage-point change. Positive = over-represented."""
        return self.synthetic_share - self.source_share

    @property
    def ratio(self) -> Optional[float]:
        """Synthetic share ÷ source share. 1.0 = faithful."""
        if self.source_share <= 0:
            return None
        return self.synthetic_share / self.source_share


@dataclass
class ColumnBiasMetrics:
    """Bias measures for one categorical column."""
    column: str
    groups: List[GroupRepresentation] = field(default_factory=list)

    # Outcome parity — only when an outcome column was supplied.
    outcome_column: Optional[str] = None
    source_disparity: Optional[float] = None
    synthetic_disparity: Optional[float] = None
    disparity_amplification: Optional[float] = None

    notes: List[str] = field(default_factory=list)

    @property
    def max_representation_shift(self) -> float:
        """Largest absolute share change across groups."""
        return max((abs(g.delta) for g in self.groups), default=0.0)

    @property
    def flagged_groups(self) -> List[str]:
        """Groups whose share moved enough to matter, excluding ones too
        rare in the source for the comparison to be stable."""
        return [
            g.group for g in self.groups
            if abs(g.delta) >= _SHIFT_FLAG and g.source_share >= _MIN_GROUP_SHARE
        ]

    @property
    def verdict(self) -> str:
        """Plain-language reading, worst signal wins."""
        amp = self.disparity_amplification
        if amp is not None and amp >= 0.10:
            return "outcome disparity amplified"
        shift = self.max_representation_shift
        if shift >= 0.20:
            return "severe representation drift"
        if self.flagged_groups:
            return "representation drift"
        if amp is not None and amp <= -0.10:
            return "outcome disparity reduced"
        return "faithful"


def compute_bias(
    syn_df: "object",                    # pandas.DataFrame
    src_df: "object",                    # pandas.DataFrame
    *,
    outcome_column: Optional[str] = None,
    max_columns: int = 20,
) -> List[ColumnBiasMetrics]:
    """Compare group representation (and optionally outcome parity) between
    source and synthetic data.

    Parameters
    ----------
    syn_df, src_df:
        Synthetic and source frames for the same table.
    outcome_column:
        Column whose rate is compared across groups. Typically the same
        target the utility check predicts. Skipped when absent or when it
        is not two-valued.
    max_columns:
        Cap on categorical columns examined, keeping wide tables fast.
        Lowest-cardinality columns are examined first.
    """
    import pandas as pd

    candidates = _group_columns(syn_df, src_df, outcome_column)
    if not candidates:
        return []

    positive = None
    if outcome_column and outcome_column in src_df.columns and outcome_column in syn_df.columns:
        positive = _positive_value(src_df[outcome_column])
        if positive is None:
            logger.debug("bias: outcome '%s' is not two-valued — parity skipped",
                         outcome_column)

    out: List[ColumnBiasMetrics] = []
    for col in candidates[:max_columns]:
        try:
            out.append(_column_bias(col, syn_df, src_df, outcome_column, positive))
        except Exception as exc:                 # pragma: no cover - defensive
            logger.warning("bias: %s — %s", col, exc)
            out.append(ColumnBiasMetrics(
                column=col,
                notes=[f"bias failed: {type(exc).__name__}: {exc}"],
            ))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _group_columns(
    syn_df: "object", src_df: "object", outcome_column: Optional[str],
) -> List[str]:
    """Shared categorical columns usable as groupings, lowest cardinality
    first so the most interpretable groupings are reported when capped."""
    import pandas as pd

    scored: List[tuple] = []
    for col in syn_df.columns:
        if col == outcome_column or col not in src_df.columns:
            continue
        s = src_df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        clean = s.dropna()
        if len(clean) == 0:
            continue
        nunique = int(clean.nunique())
        if nunique < 2 or nunique > _MAX_GROUPS:
            continue
        # Continuous numerics are not groups; discrete codes are.
        if pd.api.types.is_numeric_dtype(s) and nunique > _MAX_GROUPS // 2:
            continue
        scored.append((nunique, col))

    return [c for _, c in sorted(scored, key=lambda t: (t[0], t[1]))]


def _column_bias(
    col: str,
    syn_df: "object",
    src_df: "object",
    outcome_column: Optional[str],
    positive: Optional[str],
) -> ColumnBiasMetrics:
    import pandas as pd

    metrics = ColumnBiasMetrics(column=col)

    src_share = _shares(src_df[col])
    syn_share = _shares(syn_df[col])
    for group in sorted(set(src_share) | set(syn_share)):
        metrics.groups.append(GroupRepresentation(
            group=group,
            source_share=src_share.get(group, 0.0),
            synthetic_share=syn_share.get(group, 0.0),
        ))

    missing = [g.group for g in metrics.groups
               if g.synthetic_share == 0.0 and g.source_share >= _MIN_GROUP_SHARE]
    if missing:
        metrics.notes.append(
            f"absent from synthetic data: {', '.join(missing[:5])}"
            + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        )

    invented = [g.group for g in metrics.groups if g.source_share == 0.0]
    if invented:
        metrics.notes.append(
            f"not present in source data: {', '.join(invented[:5])}"
            + (f" (+{len(invented) - 5} more)" if len(invented) > 5 else "")
        )

    if outcome_column and positive is not None:
        _populate_outcome_parity(metrics, col, syn_df, src_df,
                                 outcome_column, positive)

    return metrics


def _populate_outcome_parity(
    metrics: ColumnBiasMetrics,
    col: str,
    syn_df: "object",
    src_df: "object",
    outcome_column: str,
    positive: str,
) -> None:
    """Disparity = widest gap in positive-outcome rate between groups.

    Amplification is the synthetic gap minus the source gap: above zero
    means the generator widened the difference between groups.
    """
    src_rates = _positive_rates(src_df, col, outcome_column, positive)
    syn_rates = _positive_rates(syn_df, col, outcome_column, positive)

    # Compare like with like — only groups measurable on both sides.
    common = sorted(set(src_rates) & set(syn_rates))
    if len(common) < 2:
        metrics.notes.append(
            "fewer than two groups present in both datasets — parity skipped"
        )
        return

    metrics.outcome_column = outcome_column
    src_values = [src_rates[g] for g in common]
    syn_values = [syn_rates[g] for g in common]
    metrics.source_disparity = max(src_values) - min(src_values)
    metrics.synthetic_disparity = max(syn_values) - min(syn_values)
    metrics.disparity_amplification = (
        metrics.synthetic_disparity - metrics.source_disparity
    )


def _shares(series: "object") -> Dict[str, float]:
    """Value → share of non-null rows."""
    clean = series.dropna().astype(str)
    total = len(clean)
    if total == 0:
        return {}
    return {str(k): v / total for k, v in clean.value_counts().items()}


def _positive_rates(
    df: "object", group_col: str, outcome_col: str, positive: str,
) -> Dict[str, float]:
    """Group → rate of the positive outcome, for groups with enough rows."""
    subset = df[[group_col, outcome_col]].dropna()
    if len(subset) == 0:
        return {}

    groups = subset[group_col].astype(str)
    outcomes = subset[outcome_col].astype(str) == positive

    rates: Dict[str, float] = {}
    for group, mask in outcomes.groupby(groups):
        # A rate over a handful of rows is noise, not disparity.
        if len(mask) >= 10:
            rates[str(group)] = float(mask.mean())
    return rates


def _positive_value(series: "object") -> Optional[str]:
    """The value treated as the positive outcome, for two-valued columns.

    Prefers a conventional affirmative label; otherwise takes the rarer
    value, which is the one disparity analysis usually cares about.
    """
    clean = series.dropna().astype(str)
    values = list(clean.unique())
    if len(values) != 2:
        return None

    affirmative = {"1", "true", "yes", "y", "approved", "active", "success", "paid"}
    for v in values:
        if v.strip().lower() in affirmative:
            return v

    counts = clean.value_counts()
    return str(counts.index[-1])
