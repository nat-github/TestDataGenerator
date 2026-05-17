"""Statistical distribution fitting for numeric columns.

Fits the best scipy distribution to a pandas Series, stores the named
parameters, and provides a vectorised sampler. Used at two points:

  1. AutoConfigInferrer — annotates inferred columns with a `distribution`
     block so the config carries statistical shape information.
  2. DataHelpers.generate_column_batch — samples from the fitted
     distribution instead of uniform random, producing statistically
     realistic values for the fallback generation path.

scipy.stats is already available as a transitive SDV/scikit-learn dep.
Fails gracefully to uniform sampling when scipy is absent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Candidates in order of preference (simpler / fewer params first)
_CANDIDATES: List[Tuple[str, str]] = [
    ("norm",    "scipy.stats.norm"),
    ("lognorm", "scipy.stats.lognorm"),
    ("expon",   "scipy.stats.expon"),
    ("gamma",   "scipy.stats.gamma"),
    ("uniform", "scipy.stats.uniform"),
]

MIN_SAMPLES = 30   # below this threshold skip fitting, use uniform


def _get_scipy_stats():
    try:
        from scipy import stats
        return stats
    except ImportError:
        return None


class DistributionFitter:
    """Fit + sample statistical distributions for numeric columns."""

    def fit(self, series: pd.Series) -> Dict[str, Any]:
        """Return the best-fitting distribution descriptor for *series*.

        Returns a dict::

            {
                "name":     "<scipy dist name>",
                "params":   [<positional params for scipy rv.rvs()>],
                "ks_stat":  <KS test statistic — lower is better>,
                "data_min": <observed min>,
                "data_max": <observed max>,
            }
        """
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric) < MIN_SAMPLES:
            return self._uniform_fallback(numeric)

        stats = _get_scipy_stats()
        if stats is None:
            return self._uniform_fallback(numeric)

        arr = numeric.to_numpy(dtype=float)
        best_name = "uniform"
        best_params: Optional[tuple] = None
        best_ks = float("inf")

        for name, _ in _CANDIDATES:
            dist = getattr(stats, name, None)
            if dist is None:
                continue
            try:
                params = dist.fit(arr)
                ks_stat, _ = stats.kstest(arr, name, args=params)
                if ks_stat < best_ks:
                    best_ks = ks_stat
                    best_name = name
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            return self._uniform_fallback(numeric)

        return {
            "name":     best_name,
            "params":   [float(p) for p in best_params],
            "ks_stat":  float(best_ks),
            "data_min": float(arr.min()),
            "data_max": float(arr.max()),
        }

    def sample(
        self,
        distribution: Dict[str, Any],
        n: int,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> np.ndarray:
        """Sample *n* values from a distribution descriptor, clipped to [min_val, max_val]."""
        stats = _get_scipy_stats()
        name = distribution.get("name", "uniform")
        params = distribution.get("params")

        lo = min_val if min_val is not None else distribution.get("data_min", 0.0)
        hi = max_val if max_val is not None else distribution.get("data_max", 1.0)

        if stats is None or not params:
            return np.random.uniform(lo, hi, n)

        dist_obj = getattr(stats, name, None)
        if dist_obj is None:
            return np.random.uniform(lo, hi, n)

        try:
            values = dist_obj.rvs(*params, size=n)
        except Exception:
            return np.random.uniform(lo, hi, n)

        return np.clip(values, lo, hi)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _uniform_fallback(numeric: pd.Series) -> Dict[str, Any]:
        lo = float(numeric.min()) if len(numeric) > 0 else 0.0
        hi = float(numeric.max()) if len(numeric) > 0 else 1.0
        span = max(hi - lo, 1.0)
        return {
            "name":     "uniform",
            "params":   [lo, span],
            "ks_stat":  0.0,
            "data_min": lo,
            "data_max": hi,
        }
