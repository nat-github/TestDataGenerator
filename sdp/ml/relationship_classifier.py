"""Adaptive classifier that learns to predict accept/reject from signals.

Cold-start safe: when there are fewer than `MIN_TRAINING_EXAMPLES` examples
or the classes are imbalanced, the classifier refuses to predict and the
caller falls back to the heuristic weighted-sum confidence. Once enough
labelled feedback accumulates, the classifier overrides the heuristic with
its learned probability.

This is intentionally a **lightweight model** — logistic regression on a
small fixed feature vector. The design goal is interpretable, fast (millis
per fit on hundreds of examples), and dependency-free beyond scikit-learn,
which is already a project dependency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum number of feedback rows required to train. Below this, predict() returns None.
MIN_TRAINING_EXAMPLES = 30

# Minimum count for each class (accept/reject) so we don't train on unbalanced data.
MIN_PER_CLASS = 5

# Fixed feature order — keeps inference deterministic even if signal keys
# arrive in a different order.
FEATURE_ORDER: List[str] = [
    "type_compatibility",
    "name_similarity",
    "value_subset",
    "pk_likeness",
    "pattern_memory_score",
]


@dataclass
class TrainingReport:
    """Diagnostic info returned by `fit()`."""
    fitted: bool
    n_examples: int
    n_accepted: int
    n_rejected: int
    coefs: Optional[Dict[str, float]] = None
    intercept: Optional[float] = None
    reason: Optional[str] = None


class RelationshipClassifier:
    """Optional classifier — see module docstring."""

    def __init__(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: List[Dict[str, float]], y: List[int]) -> TrainingReport:
        """Train on accumulated feedback.

        Returns a TrainingReport so the caller can decide whether to prefer
        the classifier or stick with the heuristic for this run.
        """
        n = len(y)
        n_pos = sum(1 for v in y if v == 1)
        n_neg = n - n_pos

        if n < MIN_TRAINING_EXAMPLES:
            return TrainingReport(
                fitted=False, n_examples=n, n_accepted=n_pos, n_rejected=n_neg,
                reason=f"need at least {MIN_TRAINING_EXAMPLES} examples (have {n})",
            )
        if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
            return TrainingReport(
                fitted=False, n_examples=n, n_accepted=n_pos, n_rejected=n_neg,
                reason=f"need at least {MIN_PER_CLASS} examples per class",
            )

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover — sklearn is in deps
            return TrainingReport(
                fitted=False, n_examples=n, n_accepted=n_pos, n_rejected=n_neg,
                reason=f"sklearn unavailable: {exc}",
            )

        X_matrix = [[float(row.get(feat, 0.0)) for feat in FEATURE_ORDER] for row in X]

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # protect against label skew
            solver="lbfgs",
        )
        try:
            model.fit(X_matrix, y)
        except Exception as exc:
            logger.warning("classifier: fit failed: %s", exc)
            return TrainingReport(
                fitted=False, n_examples=n, n_accepted=n_pos, n_rejected=n_neg,
                reason=f"fit failed: {exc}",
            )

        self._model = model
        coefs = dict(zip(FEATURE_ORDER, model.coef_[0].tolist()))
        return TrainingReport(
            fitted=True,
            n_examples=n,
            n_accepted=n_pos,
            n_rejected=n_neg,
            coefs=coefs,
            intercept=float(model.intercept_[0]),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, signals: Dict[str, float]) -> Optional[float]:
        """Return P(accepted | signals) or None if the classifier isn't fitted."""
        if self._model is None:
            return None
        x = [[float(signals.get(feat, 0.0)) for feat in FEATURE_ORDER]]
        try:
            proba = self._model.predict_proba(x)[0]
        except Exception as exc:
            logger.warning("classifier: predict failed: %s", exc)
            return None
        # predict_proba returns [P(class=0), P(class=1)] in label order
        classes = list(getattr(self._model, "classes_", [0, 1]))
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
