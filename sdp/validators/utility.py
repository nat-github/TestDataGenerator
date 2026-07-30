"""Utility evaluation for synthetic data — TSTR.

Fidelity asks *"does the synthetic data look like the real data?"*.
Utility asks the question users actually care about: *"can I still do my
job with it?"* The standard answer is **TSTR** — Train on Synthetic, Test
on Real — measured against a **TRTR** (Train on Real, Test on Real)
baseline scored on the *same* held-out real test set, so the two numbers
are directly comparable.

    utility_ratio ≈ 1.0   synthetic data is as useful as real data
    utility_ratio ≈ 0.0   a model trained on it learnt nothing usable

Only computable when source data is supplied: TRTR needs real rows to
train on, and both scores need a real test set.

Scoring per task
----------------
=================  ==========  ====================
Task               Metric      Chance level
=================  ==========  ====================
binary class.      ROC AUC     0.5
multiclass class.  macro F1    1/k (k = classes)
regression         R²          0.0
=================  ==========  ====================

The ratio is always computed on **chance-adjusted skill**::

    skill = (score - chance) / (1 - chance)      # clipped at 0
    ratio = skill_synthetic / skill_real

Raw ratios flatter synthetic data badly. Two models scoring 0.55 and 0.60
AUC give a respectable-looking 0.92 when in truth both are barely better
than a coin toss; on skill that reads 0.10/0.20 = 0.50, the honest number.
The same trap is worse for multiclass: four classes at macro F1 0.233 and
0.237 are *both* at chance (1/4 = 0.25), and a raw ratio calls that
"excellent — as useful as real data".

When the real baseline is itself at or near chance, no ratio is
meaningful — nothing was learnable from the real data either — so the
ratio is left as ``None`` with a note rather than reporting a figure that
looks like a verdict.

Uses scikit-learn, already a base dependency. Imports are deferred so
importing this module stays cheap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Row caps — TSTR is the most expensive metric in the report.
_TRAIN_CAP = 20_000
_TEST_CAP = 10_000

# A column whose values are nearly all distinct is an identifier, not a
# target or a useful feature.
_ID_UNIQUE_RATIO = 0.9

# Categorical target must have at least 2 and at most this many classes.
_MAX_TARGET_CLASSES = 20

# Categoricals with more levels than this are dropped as features — one-hot
# encoding them adds noise and cost without signal.
_MAX_FEATURE_LEVELS = 50

# Below this, the real baseline is too weak for a ratio to mean anything.
_MIN_BASELINE = 0.05


@dataclass
class UtilityMetrics:
    """TSTR result for one table."""
    target_column: str
    task: str                    # "binary" | "multiclass" | "regression"
    metric: str                  # "roc_auc" | "macro_f1" | "r2"

    score_real: Optional[float] = None       # TRTR — the baseline
    score_synthetic: Optional[float] = None  # TSTR
    utility_ratio: Optional[float] = None    # 1.0 = as useful as real data

    n_train_real: int = 0
    n_train_synthetic: int = 0
    n_test: int = 0
    feature_columns: List[str] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Plain-language reading of the ratio."""
        r = self.utility_ratio
        if r is None:
            return "not computable"
        if r >= 0.95:
            return "excellent — as useful as real data"
        if r >= 0.80:
            return "good — minor loss of signal"
        if r >= 0.50:
            return "degraded — noticeable loss of signal"
        return "poor — downstream models learn little"


def compute_utility(
    syn_df: "object",                    # pandas.DataFrame
    src_df: "object",                    # pandas.DataFrame
    *,
    target: Optional[str] = None,
    random_state: int = 42,
) -> Optional[UtilityMetrics]:
    """Run TSTR for one table.

    Parameters
    ----------
    syn_df, src_df:
        Synthetic and source frames for the same table.
    target:
        Column to predict. Auto-selected when omitted — see
        ``select_target_column``. Returns ``None`` if no usable target
        exists (e.g. every column is an identifier).
    random_state:
        Seeds the split and the model, so repeated runs agree.
    """

    shared = [c for c in syn_df.columns if c in src_df.columns]
    if not shared:
        return None

    if target is None:
        target = select_target_column(src_df, shared)
    if target is None or target not in shared:
        return None

    task, metric = _classify_task(src_df[target])
    if task is None:
        return None

    feature_cols = _select_features(src_df, shared, target)
    result = UtilityMetrics(
        target_column=target, task=task, metric=metric,
        feature_columns=feature_cols,
    )
    if not feature_cols:
        result.notes.append("no usable feature columns — utility skipped")
        return result

    try:
        _run_tstr(result, syn_df, src_df, target, feature_cols,
                  task, metric, random_state)
    except Exception as exc:                     # pragma: no cover - defensive
        logger.warning("utility: %s — %s", target, exc)
        result.notes.append(f"utility failed: {type(exc).__name__}: {exc}")

    return result


def select_target_column(df: "object", candidates: List[str]) -> Optional[str]:
    """Pick a column worth predicting.

    Prefers a low-cardinality categorical (a status/flag/category column —
    the kind of thing people actually build models against), falling back
    to a numeric column. Identifier-like columns are excluded. Selection is
    deterministic so reports are reproducible.
    """
    import pandas as pd

    n = len(df)
    if n == 0:
        return None

    categorical: List[Tuple[int, str]] = []
    numeric: List[Tuple[int, str]] = []

    for col in candidates:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        nunique = int(s.nunique())
        if nunique < 2:
            continue
        if nunique / max(n, 1) > _ID_UNIQUE_RATIO:
            continue                              # identifier
        if pd.api.types.is_numeric_dtype(s):
            if nunique <= _MAX_TARGET_CLASSES:
                categorical.append((nunique, col))  # discrete numeric = classes
            else:
                numeric.append((nunique, col))
        elif pd.api.types.is_datetime64_any_dtype(s):
            continue
        elif nunique <= _MAX_TARGET_CLASSES:
            categorical.append((nunique, col))

    # Lowest cardinality first, then name — deterministic.
    if categorical:
        return sorted(categorical, key=lambda t: (t[0], t[1]))[0][1]
    if numeric:
        return sorted(numeric, key=lambda t: t[1])[0][1]
    return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _classify_task(series: "object") -> Tuple[Optional[str], str]:
    import pandas as pd

    s = series.dropna()
    if len(s) == 0:
        return None, ""
    nunique = int(s.nunique())
    if nunique < 2:
        return None, ""

    if pd.api.types.is_numeric_dtype(s) and nunique > _MAX_TARGET_CLASSES:
        return "regression", "r2"
    if nunique == 2:
        return "binary", "roc_auc"
    if nunique <= _MAX_TARGET_CLASSES:
        return "multiclass", "macro_f1"
    return None, ""


def _select_features(df: "object", shared: List[str], target: str) -> List[str]:
    """Everything shared except the target, identifiers and unusable types."""
    import pandas as pd

    n = len(df)
    out: List[str] = []
    for col in shared:
        if col == target:
            continue
        s = df[col]
        clean = s.dropna()
        if len(clean) == 0:
            continue
        nunique = int(clean.nunique())
        if nunique < 2:
            continue                              # constant — no signal
        if pd.api.types.is_numeric_dtype(s):
            if nunique / max(n, 1) > _ID_UNIQUE_RATIO and _looks_like_id(col):
                continue
            out.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            out.append(col)                       # encoded as epoch seconds
        elif nunique <= _MAX_FEATURE_LEVELS:
            out.append(col)
    return out


def _looks_like_id(col: str) -> bool:
    lowered = col.lower()
    return lowered.endswith("_id") or lowered.endswith("_key") or lowered in {"id", "key"}


def _run_tstr(
    result: UtilityMetrics,
    syn_df: "object",
    src_df: "object",
    target: str,
    feature_cols: List[str],
    task: str,
    metric: str,
    random_state: int,
) -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Rows where the target is missing teach nothing.
    src = src_df[feature_cols + [target]].dropna(subset=[target])
    syn = syn_df[feature_cols + [target]].dropna(subset=[target])
    if len(src) < 20:
        result.notes.append(f"only {len(src)} source rows — too few for a train/test split")
        return
    if len(syn) < 10:
        result.notes.append(f"only {len(syn)} synthetic rows — too few to train on")
        return

    is_classification = task in ("binary", "multiclass")

    # Encoders are fitted on real ∪ synthetic so both frames map into the
    # same feature space; a category seen only in synthetic data still gets
    # a column rather than silently shifting the encoding.
    levels = _build_levels(src, syn, feature_cols)
    fill = _build_numeric_fill(src, feature_cols)

    y_src = src[target]
    stratify = y_src if is_classification and y_src.value_counts().min() >= 2 else None
    try:
        src_train, src_test = train_test_split(
            src, test_size=0.3, random_state=random_state, stratify=stratify,
        )
    except ValueError as exc:
        result.notes.append(f"train/test split failed: {exc}")
        return

    src_train = _cap(src_train, _TRAIN_CAP, random_state)
    src_test = _cap(src_test, _TEST_CAP, random_state)
    syn_train = _cap(syn, _TRAIN_CAP, random_state)

    x_test = _encode(src_test, feature_cols, levels, fill)
    y_test = src_test[target]

    if is_classification:
        # Score against the label space of the real test set — that is what
        # a downstream consumer would face.
        classes = sorted(pd.unique(y_test.astype(str)))
        if len(classes) < 2:
            result.notes.append("real test set collapsed to one class — utility skipped")
            return
    else:
        classes = []

    result.n_train_real = len(src_train)
    result.n_train_synthetic = len(syn_train)
    result.n_test = len(src_test)

    score_real = _fit_and_score(
        src_train, x_test, y_test, feature_cols, target, levels, fill,
        task, metric, classes, random_state, result, "real",
    )
    score_syn = _fit_and_score(
        syn_train, x_test, y_test, feature_cols, target, levels, fill,
        task, metric, classes, random_state, result, "synthetic",
    )

    result.score_real = score_real
    result.score_synthetic = score_syn
    result.utility_ratio = _ratio(
        score_real, score_syn, metric, max(len(classes), 2), result,
    )


def _fit_and_score(
    train_df: "object",
    x_test: "object",
    y_test: "object",
    feature_cols: List[str],
    target: str,
    levels: Dict[str, List[str]],
    fill: Dict[str, float],
    task: str,
    metric: str,
    classes: List[str],
    random_state: int,
    result: UtilityMetrics,
    label: str,
) -> Optional[float]:
    """Fit one model on ``train_df`` and score it on the shared real test set."""
    import pandas as pd

    x_train = _encode(train_df, feature_cols, levels, fill)
    y_train = train_df[target]

    if task == "regression":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        model = RandomForestRegressor(
            n_estimators=60, random_state=random_state, n_jobs=-1,
        )
        model.fit(x_train, pd.to_numeric(y_train, errors="coerce").fillna(0.0))
        preds = model.predict(x_test)
        return float(r2_score(pd.to_numeric(y_test, errors="coerce").fillna(0.0), preds))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    y_train_s = y_train.astype(str)
    if y_train_s.nunique() < 2:
        # Mode collapse: the generator produced a single class, so no
        # classifier can be trained. Zero utility is the honest score.
        result.notes.append(
            f"{label} training data has only one class for '{target}' — "
            "a model cannot be trained on it"
        )
        return 0.0

    model = RandomForestClassifier(
        n_estimators=60, random_state=random_state, n_jobs=-1,
    )
    model.fit(x_train, y_train_s)
    y_test_s = y_test.astype(str)

    if metric == "roc_auc":
        proba = model.predict_proba(x_test)
        positive = classes[-1]
        if positive not in list(model.classes_):
            # Trained model never saw the positive class — it cannot rank it.
            result.notes.append(
                f"{label} training data never contained '{positive}' — "
                "AUC falls back to chance"
            )
            return 0.5
        idx = list(model.classes_).index(positive)
        try:
            return float(roc_auc_score((y_test_s == positive).astype(int), proba[:, idx]))
        except ValueError as exc:
            result.notes.append(f"{label} roc_auc failed: {exc}")
            return None

    preds = model.predict(x_test)
    return float(f1_score(y_test_s, preds, average="macro", zero_division=0))


def _chance_level(metric: str, n_classes: int) -> float:
    """The score a model that learnt nothing would get.

    Getting this wrong is how a utility metric ends up flattering useless
    data: for macro F1 the floor is 1/k, not 0, so four classes at 0.233
    look respectable against a 0.05 threshold while being pure chance.
    """
    if metric == "roc_auc":
        return 0.5
    if metric == "macro_f1":
        return 1.0 / n_classes if n_classes > 1 else 0.0
    return 0.0                                   # r2: 0 is the null model


def _ratio(
    score_real: Optional[float],
    score_syn: Optional[float],
    metric: str,
    n_classes: int,
    result: UtilityMetrics,
) -> Optional[float]:
    """TSTR / TRTR on chance-adjusted skill. See the module docstring."""
    if score_real is None or score_syn is None:
        return None

    chance = _chance_level(metric, n_classes)
    span = 1.0 - chance
    if span <= 0:                                # pragma: no cover - defensive
        return None

    real = max(0.0, (score_real - chance) / span)
    syn = max(0.0, (score_syn - chance) / span)

    if real <= _MIN_BASELINE:
        result.notes.append(
            f"the real-data baseline scores {score_real:.4f} against a chance "
            f"level of {chance:.4f} — nothing was learnable from the real data "
            "either, so the synthetic data cannot be judged on this target"
        )
        return None
    return round(min(syn / real, 2.0), 4)


def _build_levels(
    src: "object", syn: "object", feature_cols: List[str],
) -> Dict[str, List[str]]:
    """Category levels per non-numeric feature, from real ∪ synthetic."""
    import pandas as pd

    levels: Dict[str, List[str]] = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(src[col]) or pd.api.types.is_datetime64_any_dtype(src[col]):
            continue
        values = set(src[col].dropna().astype(str).unique())
        if col in syn.columns:
            values |= set(syn[col].dropna().astype(str).unique())
        levels[col] = sorted(values)
    return levels


def _build_numeric_fill(src: "object", feature_cols: List[str]) -> Dict[str, float]:
    """Median of each numeric feature in the real data, used for imputation."""
    import pandas as pd

    fill: Dict[str, float] = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(src[col]):
            median = src[col].median()
            fill[col] = float(median) if pd.notna(median) else 0.0
    return fill


def _encode(
    df: "object",
    feature_cols: List[str],
    levels: Dict[str, List[str]],
    fill: Dict[str, float],
) -> "object":
    """Frame → dense float matrix.

    Numerics are imputed with the real-data median, datetimes become epoch
    seconds, and categoricals are ordinal-encoded against the shared level
    list (unseen values → -1).
    """
    import numpy as np
    import pandas as pd

    columns: List["object"] = []
    for col in feature_cols:
        s = df[col] if col in df.columns else pd.Series([np.nan] * len(df))
        if col in levels:
            mapping = {v: i for i, v in enumerate(levels[col])}
            columns.append(s.astype(str).map(mapping).fillna(-1).astype(float).values)
        elif pd.api.types.is_datetime64_any_dtype(s):
            epoch = pd.to_datetime(s, errors="coerce").astype("int64") / 1e9
            columns.append(epoch.replace([np.inf, -np.inf], np.nan).fillna(0.0).values)
        else:
            numeric = pd.to_numeric(s, errors="coerce")
            columns.append(numeric.fillna(fill.get(col, 0.0)).astype(float).values)

    if not columns:
        return np.empty((len(df), 0), dtype=float)
    return np.column_stack(columns)


def _cap(df: "object", n: int, random_state: int) -> "object":
    return df if len(df) <= n else df.sample(n=n, random_state=random_state)
