"""Synthetic-data quality reports.

Two modes:

  - **Univariate-only** (no source data): per-column dtype, null rate,
    unique count, summary stats; per-table correlation matrix; row counts.
    Always available. Useful for "does this synthetic data look plausible
    on its own?"

  - **Fidelity vs source** (source data provided): adds KS test for
    numeric columns, chi-square for categoricals, total-variation
    distance, correlation-matrix delta, and a nearest-neighbour distance
    privacy proxy. Useful for "is this synthetic data faithful to the
    source distribution while not leaking individual rows?"

    Also adds two measures that fidelity alone cannot answer:

    - **utility** (``validators/utility.py``) — TSTR: train a model on
      the synthetic data, test it on real data, compare against the same
      model trained on real data. Answers "can this data still do the
      job?", which faithful-looking data can still fail.
    - **bias** (``validators/bias.py``) — group representation drift and
      outcome disparity amplification. Answers "did generation skew who
      is represented, or widen the gap between groups?"

Inputs are pandas DataFrames keyed by table name. The report is a
nested dataclass with `to_dict()` and `to_markdown()` methods. Renderers
in the Streamlit UI and MCP server use those.

No optional dependencies. Uses pandas + numpy + scipy (already in the
base install). HTML output piggybacks on pandas' to_html.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from sdp.validators.bias import ColumnBiasMetrics, compute_bias
from sdp.validators.utility import UtilityMetrics, compute_utility

logger = logging.getLogger(__name__)


# Sample cap for expensive metrics (KS, NN distance). Beyond this we sample
# uniformly to keep the report fast on million-row tables.
_SAMPLE_CAP = 10_000

# Top-N values to record for a categorical column.
_TOP_N = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ColumnQualityMetrics:
    """Per-column metrics. Numeric vs categorical fields are populated
    based on the column's inferred dtype family."""
    column: str
    dtype: str
    is_numeric: bool
    row_count: int
    null_count: int
    null_rate: float
    unique_count: int

    # Numeric-only
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None

    # Categorical-only — top values with counts (capped at _TOP_N)
    top_values: Optional[List[Tuple[str, int]]] = None

    # Fidelity (when source data is supplied)
    ks_statistic: Optional[float] = None       # numeric: 0=identical, 1=disjoint
    ks_pvalue: Optional[float] = None          # numeric: high p ⇒ same distribution
    chi2_pvalue: Optional[float] = None        # categorical: high p ⇒ same distribution
    tv_distance: Optional[float] = None        # categorical: total-variation distance, 0..1
    distribution_score: Optional[float] = None # 1=identical, 0=disjoint (composite)

    # Notes / warnings — short human-readable strings
    notes: List[str] = field(default_factory=list)


@dataclass
class TableQualityMetrics:
    """Per-table aggregate metrics."""
    table_name: str
    row_count_synthetic: int
    row_count_source: Optional[int]
    columns: List[ColumnQualityMetrics]

    # Pearson correlation matrices — numeric columns only.
    correlation_synthetic: Optional[List[List[float]]] = None
    correlation_columns: Optional[List[str]] = None

    # Distance between synthetic and source correlation matrices (Frobenius
    # norm of the delta, scaled to [0, 1] by max possible). Only present
    # when source data has overlapping numeric columns.
    correlation_distance: Optional[float] = None

    # Privacy proxy: fraction of synthetic rows whose nearest-neighbour
    # distance to the source set is below a threshold (suspicious if high).
    privacy_nn_too_close_rate: Optional[float] = None

    # Aggregate fidelity score: average of per-column distribution_score.
    # 1.0 = perfect match, 0.0 = totally different. None when no source.
    fidelity_score: Optional[float] = None

    # Utility (TSTR) — is the data still usable for a downstream task?
    # None when there is no source, no usable target, or sklearn is absent.
    utility: Optional[UtilityMetrics] = None

    # Bias — per-column representation drift and outcome disparity.
    # Empty when there is no source or no categorical grouping columns.
    bias: List[ColumnBiasMetrics] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)

    @property
    def has_source(self) -> bool:
        return self.row_count_source is not None

    @property
    def utility_ratio(self) -> Optional[float]:
        return self.utility.utility_ratio if self.utility else None

    @property
    def biased_columns(self) -> List[str]:
        """Columns whose representation drifted or whose outcome gap widened."""
        return [b.column for b in self.bias if b.verdict != "faithful"]


@dataclass
class QualityReport:
    """Top-level report — one entry per table."""
    has_source: bool
    tables: Dict[str, TableQualityMetrics]

    @property
    def overall_fidelity(self) -> Optional[float]:
        """Average fidelity across tables that have a source. None if
        there's no source for any table."""
        scores = [t.fidelity_score for t in self.tables.values()
                  if t.fidelity_score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def overall_utility(self) -> Optional[float]:
        """Average TSTR ratio across tables where it was computable."""
        ratios = [t.utility_ratio for t in self.tables.values()
                  if t.utility_ratio is not None]
        if not ratios:
            return None
        return sum(ratios) / len(ratios)

    @property
    def bias_flagged_tables(self) -> List[str]:
        """Tables with at least one column showing bias drift."""
        return [name for name, t in self.tables.items() if t.biased_columns]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_source": self.has_source,
            "overall_fidelity": self.overall_fidelity,
            "overall_utility": self.overall_utility,
            "bias_flagged_tables": self.bias_flagged_tables,
            "tables": {
                name: _serialise_table(t) for name, t in self.tables.items()
            },
        }

    def to_markdown(self, *, max_columns_shown: int = 30) -> str:
        return _render_markdown(self, max_columns_shown=max_columns_shown)

    def to_html(self) -> str:
        """Self-contained HTML page rendering the report."""
        return _render_html(self)


# ---------------------------------------------------------------------------
# Public API — entry points
# ---------------------------------------------------------------------------


def quality_report(
    synthetic: Mapping[str, "object"],   # pandas.DataFrame
    *,
    source: Optional[Mapping[str, "object"]] = None,
    privacy_threshold: float = 0.0,
    with_utility: bool = True,
    with_bias: bool = True,
    targets: Optional[Mapping[str, str]] = None,
) -> QualityReport:
    """Compute a quality report for the given synthetic data.

    Parameters
    ----------
    synthetic:
        Dict ``table_name → DataFrame`` for the generated data.
    source:
        Optional dict ``table_name → DataFrame`` for the source data.
        When provided, fidelity metrics are computed for every overlapping
        table.
    privacy_threshold:
        Distance below which a synthetic row is considered "too close" to
        a source row for the privacy proxy. ``0.0`` flags exact duplicates
        only. Pass a small positive value (e.g. 1e-6) to also flag
        floating-point near-duplicates.
    with_utility:
        Compute the TSTR utility score. Requires source data and fits two
        models per table — by far the most expensive part of the report,
        so it can be turned off for quick runs.
    with_bias:
        Compute representation drift and outcome disparity. Cheap.
    targets:
        Optional ``table_name → column`` map naming the column to predict
        (utility) and measure outcomes against (bias). Auto-selected per
        table when not given.
    """
    has_source = source is not None
    tables: Dict[str, TableQualityMetrics] = {}

    for table_name, syn_df in synthetic.items():
        src_df = (source or {}).get(table_name)
        try:
            tables[table_name] = _table_metrics(
                table_name, syn_df, src_df,
                privacy_threshold=privacy_threshold,
                with_utility=with_utility,
                with_bias=with_bias,
                target=(targets or {}).get(table_name),
            )
        except Exception as exc:
            logger.warning("quality_report: %s — %s", table_name, exc)
            tables[table_name] = TableQualityMetrics(
                table_name=table_name,
                row_count_synthetic=0,
                row_count_source=None,
                columns=[],
                notes=[f"error: {type(exc).__name__}: {exc}"],
            )

    return QualityReport(has_source=has_source, tables=tables)


def quality_report_from_paths(
    synthetic_dir: Union[str, Path],
    source_dir: Optional[Union[str, Path]] = None,
    *,
    privacy_threshold: float = 0.0,
    with_utility: bool = True,
    with_bias: bool = True,
    targets: Optional[Mapping[str, str]] = None,
) -> QualityReport:
    """Like ``quality_report`` but reads Parquet directories from disk."""
    syn_dfs = _load_parquet_dir(Path(synthetic_dir))
    src_dfs = _load_parquet_dir(Path(source_dir)) if source_dir else None
    return quality_report(syn_dfs, source=src_dfs,
                          privacy_threshold=privacy_threshold,
                          with_utility=with_utility,
                          with_bias=with_bias,
                          targets=targets)


# ---------------------------------------------------------------------------
# Per-table computation
# ---------------------------------------------------------------------------


def _table_metrics(
    table_name: str,
    syn_df: "object",   # pandas.DataFrame
    src_df: Optional["object"],
    *,
    privacy_threshold: float,
    with_utility: bool = True,
    with_bias: bool = True,
    target: Optional[str] = None,
) -> TableQualityMetrics:

    cols: List[ColumnQualityMetrics] = []
    for col_name in syn_df.columns:
        cols.append(_column_metrics(col_name, syn_df[col_name],
                                    src_df[col_name] if src_df is not None and col_name in src_df.columns else None))

    notes: List[str] = []
    correlation_syn, correlation_columns = _correlation_matrix(syn_df)
    correlation_distance = None
    privacy_too_close = None
    fidelity_score = None
    utility: Optional[UtilityMetrics] = None
    bias: List[ColumnBiasMetrics] = []

    if src_df is not None:
        if not isinstance(src_df, type(syn_df)):
            notes.append("source dataframe type mismatch")

        # correlation delta
        if correlation_syn is not None:
            src_corr, src_cols = _correlation_matrix(src_df)
            if src_corr is not None and src_cols == correlation_columns:
                correlation_distance = _matrix_distance(correlation_syn, src_corr)
            else:
                notes.append("numeric column sets differ — correlation distance skipped")

        # privacy NN distance (numeric columns only, sampled)
        privacy_too_close = _privacy_nn_too_close_rate(
            syn_df, src_df, threshold=privacy_threshold,
        )

        # aggregate fidelity score: mean of per-column distribution_scores
        scores = [c.distribution_score for c in cols if c.distribution_score is not None]
        if scores:
            fidelity_score = sum(scores) / len(scores)

        # Utility (TSTR) — expensive, so it is the one metric that is
        # optional. Failures degrade to a note rather than losing the
        # rest of the report.
        if with_utility:
            try:
                utility = compute_utility(syn_df, src_df, target=target)
                if utility is None:
                    notes.append("no usable prediction target — utility skipped")
            except Exception as exc:
                logger.warning("utility: %s — %s", table_name, exc)
                notes.append(f"utility failed: {type(exc).__name__}: {exc}")

        # Bias — the outcome column is whatever utility predicted, so the
        # two measures describe the same target.
        if with_bias:
            outcome = target or (utility.target_column if utility else None)
            try:
                bias = compute_bias(syn_df, src_df, outcome_column=outcome)
            except Exception as exc:
                logger.warning("bias: %s — %s", table_name, exc)
                notes.append(f"bias failed: {type(exc).__name__}: {exc}")

    return TableQualityMetrics(
        table_name=table_name,
        row_count_synthetic=len(syn_df),
        row_count_source=len(src_df) if src_df is not None else None,
        columns=cols,
        correlation_synthetic=correlation_syn,
        correlation_columns=correlation_columns,
        correlation_distance=correlation_distance,
        privacy_nn_too_close_rate=privacy_too_close,
        fidelity_score=fidelity_score,
        utility=utility,
        bias=bias,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Per-column computation
# ---------------------------------------------------------------------------


def _column_metrics(name: str, syn: "object", src: Optional["object"] = None) -> ColumnQualityMetrics:
    import pandas as pd

    is_numeric = pd.api.types.is_numeric_dtype(syn)
    row_count = len(syn)
    null_count = int(syn.isna().sum())
    null_rate = null_count / row_count if row_count else 0.0
    unique_count = int(syn.nunique(dropna=True))

    cm = ColumnQualityMetrics(
        column=name,
        dtype=str(syn.dtype),
        is_numeric=bool(is_numeric),
        row_count=row_count,
        null_count=null_count,
        null_rate=null_rate,
        unique_count=unique_count,
    )

    if is_numeric:
        clean = syn.dropna()
        if len(clean) > 0:
            cm.mean = float(clean.mean())
            cm.std = float(clean.std()) if len(clean) > 1 else 0.0
            cm.min = float(clean.min())
            cm.max = float(clean.max())
            cm.median = float(clean.median())
    else:
        # Categorical / string: top-N values
        try:
            counts = syn.dropna().astype(str).value_counts().head(_TOP_N)
            cm.top_values = [(str(v), int(c)) for v, c in counts.items()]
        except Exception as exc:
            cm.notes.append(f"top_values failed: {exc}")

    # Fidelity vs source
    if src is not None:
        try:
            _populate_fidelity(cm, syn, src)
        except Exception as exc:
            cm.notes.append(f"fidelity failed: {exc}")

    return cm


def _populate_fidelity(cm: ColumnQualityMetrics, syn: "object", src: "object") -> None:
    """Add KS / chi-square / TV distance / distribution_score in place."""

    syn_clean = syn.dropna()
    src_clean = src.dropna()
    if len(syn_clean) == 0 or len(src_clean) == 0:
        cm.notes.append("empty after dropna — fidelity skipped")
        return

    if cm.is_numeric:
        # Subsample for speed
        syn_sample = _sample(syn_clean, _SAMPLE_CAP)
        src_sample = _sample(src_clean, _SAMPLE_CAP)
        try:
            from scipy.stats import ks_2samp
            ks = ks_2samp(syn_sample, src_sample)
            cm.ks_statistic = float(ks.statistic)
            cm.ks_pvalue = float(ks.pvalue)
            # Distribution score: 1 - KS statistic (KS stat is in [0, 1])
            cm.distribution_score = max(0.0, 1.0 - cm.ks_statistic)
        except ImportError:
            cm.notes.append("scipy not available — KS skipped")
    else:
        # Categorical: align value sets, compute TV distance and chi-square
        syn_counts = syn_clean.astype(str).value_counts()
        src_counts = src_clean.astype(str).value_counts()
        all_values = list(set(syn_counts.index) | set(src_counts.index))
        if not all_values:
            cm.notes.append("no observed values — fidelity skipped")
            return

        syn_total = float(syn_counts.sum())
        src_total = float(src_counts.sum())
        tv = 0.0
        for v in all_values:
            p = float(syn_counts.get(v, 0)) / syn_total if syn_total else 0.0
            q = float(src_counts.get(v, 0)) / src_total if src_total else 0.0
            tv += abs(p - q)
        tv /= 2  # canonical TV-distance is 1/2 sum of abs diffs
        cm.tv_distance = float(tv)
        cm.distribution_score = max(0.0, 1.0 - tv)

        # Chi-square (best-effort — only when expected counts are nonzero)
        try:
            from scipy.stats import chisquare
            # Project both onto the union, expected = source proportions × syn_total
            observed = []
            expected = []
            for v in all_values:
                obs = float(syn_counts.get(v, 0))
                exp = float(src_counts.get(v, 0)) / src_total * syn_total if src_total else 0.0
                if exp > 0:
                    observed.append(obs)
                    expected.append(exp)
            if len(observed) >= 2 and sum(observed) > 0:
                # scipy requires expected sums to match observed sums; rescale
                obs_sum = sum(observed)
                exp_sum = sum(expected)
                if exp_sum > 0:
                    expected = [e * obs_sum / exp_sum for e in expected]
                    chi = chisquare(f_obs=observed, f_exp=expected)
                    cm.chi2_pvalue = float(chi.pvalue)
        except ImportError:
            cm.notes.append("scipy not available — chi-square skipped")
        except Exception as exc:
            cm.notes.append(f"chi-square failed: {exc}")


# ---------------------------------------------------------------------------
# Cross-column metrics
# ---------------------------------------------------------------------------


def _correlation_matrix(df: "object") -> Tuple[Optional[List[List[float]]], Optional[List[str]]]:
    """Return Pearson correlation matrix over numeric columns only."""
    import pandas as pd

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return None, None

    corr = df[numeric_cols].corr()
    matrix = [[_safe_float(v) for v in row] for row in corr.values.tolist()]
    return matrix, list(corr.columns)


def _matrix_distance(a: List[List[float]], b: List[List[float]]) -> float:
    """Frobenius norm of (a - b), scaled to [0, 1] by max possible (which is
    2 × n for an n×n correlation matrix in [-1, 1])."""
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    sq = 0.0
    for i in range(n):
        for j in range(n):
            d = a[i][j] - b[i][j]
            sq += d * d
    fro = math.sqrt(sq)
    # Theoretical max — every pair off by 2 (range [-1, 1])
    max_fro = 2.0 * n
    return min(1.0, fro / max_fro) if max_fro else 0.0


def _privacy_nn_too_close_rate(
    syn_df: "object",
    src_df: "object",
    *,
    threshold: float,
) -> Optional[float]:
    """Fraction of synthetic rows whose nearest-neighbour distance to the
    source set is ≤ threshold. A simple membership-inference proxy: high
    rate ⇒ synthetic rows are suspiciously close to real rows."""
    import pandas as pd
    import numpy as np

    numeric_cols = [c for c in syn_df.columns
                    if pd.api.types.is_numeric_dtype(syn_df[c])
                    and c in src_df.columns
                    and pd.api.types.is_numeric_dtype(src_df[c])]
    if not numeric_cols:
        return None

    syn = syn_df[numeric_cols].dropna()
    src = src_df[numeric_cols].dropna()
    if len(syn) == 0 or len(src) == 0:
        return None

    # Sample both for speed
    syn = _sample_df(syn, _SAMPLE_CAP)
    src = _sample_df(src, _SAMPLE_CAP)

    # Standardise to comparable scales
    means = src.mean()
    stds = src.std().replace(0, 1)
    syn_std = (syn - means) / stds
    src_std = (src - means) / stds

    # Brute-force NN — fine at sample-cap scale; ~10k × 10k × dim multiplications
    syn_arr = syn_std.values.astype(float)
    src_arr = src_std.values.astype(float)
    too_close = 0
    for row in syn_arr:
        diffs = src_arr - row
        dist_sq = np.einsum('ij,ij->i', diffs, diffs)
        if dist_sq.min() <= threshold * threshold:
            too_close += 1
    return too_close / len(syn_arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(series: "object", n: int) -> "object":
    """Subsample a Series uniformly without replacement, capped at n."""
    if len(series) <= n:
        return series
    return series.sample(n=n, random_state=42)


def _sample_df(df: "object", n: int) -> "object":
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


def _safe_float(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def _load_parquet_dir(path: Path) -> Dict[str, "object"]:
    import pyarrow.parquet as pq

    out: Dict[str, "object"] = {}
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    for parquet_file in sorted(path.glob("*.parquet")):
        out[parquet_file.stem] = pq.read_table(parquet_file).to_pandas()
    return out


def _serialise_table(t: TableQualityMetrics) -> Dict[str, Any]:
    return {
        "table_name": t.table_name,
        "row_count_synthetic": t.row_count_synthetic,
        "row_count_source": t.row_count_source,
        "fidelity_score": t.fidelity_score,
        "correlation_distance": t.correlation_distance,
        "privacy_nn_too_close_rate": t.privacy_nn_too_close_rate,
        "utility": _serialise_utility(t.utility),
        "bias": [_serialise_bias(b) for b in t.bias],
        "notes": t.notes,
        "columns": [asdict(c) for c in t.columns],
        "correlation_synthetic": t.correlation_synthetic,
        "correlation_columns": t.correlation_columns,
    }


def _serialise_utility(u: Optional[UtilityMetrics]) -> Optional[Dict[str, Any]]:
    if u is None:
        return None
    data = asdict(u)
    data["verdict"] = u.verdict          # property — not captured by asdict
    return data


def _serialise_bias(b: ColumnBiasMetrics) -> Dict[str, Any]:
    return {
        "column": b.column,
        "max_representation_shift": b.max_representation_shift,
        "flagged_groups": b.flagged_groups,
        "verdict": b.verdict,
        "outcome_column": b.outcome_column,
        "source_disparity": b.source_disparity,
        "synthetic_disparity": b.synthetic_disparity,
        "disparity_amplification": b.disparity_amplification,
        "notes": b.notes,
        "groups": [
            {
                "group": g.group,
                "source_share": g.source_share,
                "synthetic_share": g.synthetic_share,
                "delta": g.delta,
                "ratio": g.ratio,
            }
            for g in b.groups
        ],
    }


def _render_markdown(report: QualityReport, *, max_columns_shown: int) -> str:
    lines: List[str] = []
    lines.append("# Synthetic Data Quality Report")
    lines.append("")
    if report.overall_fidelity is not None:
        lines.append(f"**Overall fidelity score:** `{report.overall_fidelity:.3f}` "
                     "(1.0 = identical to source, 0.0 = disjoint)")
    elif report.has_source:
        lines.append("**Overall fidelity score:** *not computable (insufficient data)*")
    else:
        lines.append("*No source data provided — univariate-only report.*")

    if report.overall_utility is not None:
        lines.append(f"**Overall utility (TSTR):** `{report.overall_utility:.3f}` "
                     "(1.0 = models trained on synthetic data do as well as on real data)")
    if report.bias_flagged_tables:
        lines.append("**Bias flags:** " +
                     ", ".join(f"`{n}`" for n in report.bias_flagged_tables))
    lines.append("")

    for name, t in report.tables.items():
        lines.append(f"## Table: `{name}`")
        lines.append("")
        lines.append(f"- Synthetic rows: **{t.row_count_synthetic:,}**")
        if t.row_count_source is not None:
            lines.append(f"- Source rows: **{t.row_count_source:,}**")
        if t.fidelity_score is not None:
            lines.append(f"- Fidelity score: **{t.fidelity_score:.3f}**")
        if t.correlation_distance is not None:
            lines.append(f"- Correlation-matrix distance: **{t.correlation_distance:.3f}** (lower is better)")
        if t.privacy_nn_too_close_rate is not None:
            lines.append(
                f"- Privacy NN too-close rate: **{t.privacy_nn_too_close_rate:.3%}** "
                "(rows uncomfortably close to a source row)"
            )
        for note in t.notes:
            lines.append(f"- *Note:* {note}")
        lines.append("")

        _append_utility_section(lines, t)
        _append_bias_section(lines, t)

        # Column table
        lines.append("| Column | Dtype | Null rate | Unique | Mean | Std | Top values | Score |")
        lines.append("|---|---|---:|---:|---:|---:|---|---:|")
        for c in t.columns[:max_columns_shown]:
            mean = f"{c.mean:.2f}" if c.mean is not None else "—"
            std = f"{c.std:.2f}" if c.std is not None else "—"
            top = ", ".join(f"{v}({n})" for v, n in (c.top_values or [])[:3]) or "—"
            score = f"{c.distribution_score:.3f}" if c.distribution_score is not None else "—"
            lines.append(
                f"| {c.column} | `{c.dtype}` | {c.null_rate:.1%} | "
                f"{c.unique_count} | {mean} | {std} | {top} | {score} |"
            )
        if len(t.columns) > max_columns_shown:
            lines.append(f"| ... and {len(t.columns) - max_columns_shown} more columns |||||||| |")
        lines.append("")
    return "\n".join(lines)


def _append_utility_section(lines: List[str], t: TableQualityMetrics) -> None:
    u = t.utility
    if u is None:
        return

    lines.append("### Utility — can a model still learn from this data?")
    lines.append("")
    lines.append(f"Predicting **`{u.target_column}`** ({u.task}, scored by `{u.metric}`)")
    lines.append("")
    lines.append("| Trained on | Score | Rows |")
    lines.append("|---|---:|---:|")
    real = f"{u.score_real:.4f}" if u.score_real is not None else "—"
    syn = f"{u.score_synthetic:.4f}" if u.score_synthetic is not None else "—"
    lines.append(f"| Real data (baseline) | {real} | {u.n_train_real:,} |")
    lines.append(f"| Synthetic data | {syn} | {u.n_train_synthetic:,} |")
    lines.append("")
    if u.utility_ratio is not None:
        lines.append(f"**Utility ratio: {u.utility_ratio:.3f}** — {u.verdict}")
    else:
        lines.append(f"**Utility ratio: not computable** — {u.verdict}")
    lines.append(f"*Both models scored on the same {u.n_test:,} held-out real rows.*")
    for note in u.notes:
        lines.append(f"- *Note:* {note}")
    lines.append("")


def _append_bias_section(lines: List[str], t: TableQualityMetrics) -> None:
    if not t.bias:
        return
    # Only worth a section when something actually moved.
    interesting = [b for b in t.bias if b.verdict != "faithful"]
    if not interesting:
        lines.append("### Bias — no representation drift detected")
        lines.append("")
        return

    lines.append("### Bias — representation and outcome drift")
    lines.append("")
    lines.append("| Column | Verdict | Max share shift | Groups affected | Disparity amplification |")
    lines.append("|---|---|---:|---|---:|")
    for b in interesting:
        groups = ", ".join(b.flagged_groups[:3]) or "—"
        if len(b.flagged_groups) > 3:
            groups += f" (+{len(b.flagged_groups) - 3})"
        amp = (f"{b.disparity_amplification:+.1%}"
               if b.disparity_amplification is not None else "—")
        lines.append(
            f"| {b.column} | {b.verdict} | {b.max_representation_shift:.1%} | "
            f"{groups} | {amp} |"
        )
    lines.append("")
    for b in interesting:
        for note in b.notes:
            lines.append(f"- *{b.column}:* {note}")
    lines.append("")


def _render_html(report: QualityReport) -> str:
    """Lightweight self-contained HTML — same content as the markdown view,
    but inside a <style>-tagged page that opens in a browser."""
    md = _render_markdown(report, max_columns_shown=200)
    # Tiny markdown-ish converter for our subset (headers, bold, code, table)
    body = (
        md.replace("&", "&amp;")
          .replace("<", "&lt;")
          .replace(">", "&gt;")
    )
    # We need to keep the markdown-ish formatting readable but escaped — for
    # simplicity, wrap in <pre> so the report renders verbatim with formatting
    # preserved. Good enough for "save → open in browser → read".
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Synthetic Data Quality Report</title>"
        "<style>"
        "body{font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "max-width:980px;margin:2rem auto;padding:0 1rem;color:#222}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;"
        "padding:1rem;border-radius:6px;font:13px/1.5 ui-monospace,monospace}"
        "</style>"
        f"<pre>{body}</pre>"
    )
