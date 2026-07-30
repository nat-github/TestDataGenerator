"""Differentially private marginal engine.

The only engine here that offers a *formal* privacy guarantee. Everything
else in the platform — including the nearest-neighbour proxy in
``validators/quality_report.py`` — is a heuristic: it catches copies and
overfit models but proves nothing. This engine proves something specific,
and refuses to overstate what.

The mechanism
-------------
For each column with a **publicly declared domain**, build a histogram over
that domain, add Laplace noise, clip negatives to zero, renormalise, and
sample. Adding or removing one row changes any histogram by at most 1, so
L1 sensitivity is 1 and Laplace noise at scale ``1/ε_col`` gives ``ε_col``
-differential privacy. Sampling from the noisy histogram is post-processing
and costs nothing further.

Why this platform can do DP honestly
------------------------------------
The classic way to leak while claiming DP is to derive the domain from the
data — taking bin edges from the observed min/max, or the category list
from observed values. Both are non-private queries, and the published ε is
then a fiction.

This platform does not have that problem, because its configs already
declare the domain: ``business_values`` gives the category set, and
``min_value`` / ``max_value`` give numeric bounds. **The config is the
public schema.** Columns without a declared domain are never measured
against the data at all — they are generated from config rules alone, which
consumes zero budget and learns nothing.

Composition
-----------
Within a table, the per-column histograms are sequential composition over
the same records, so their epsilons **add**. The budget is therefore split
evenly across the measured columns of each table: ``ε_col = ε / m``.

Across tables the guarantee is stated **per table**: ε protects the
presence of any single row *within its own table*. It does not protect an
entity that appears in several tables — one customer with fifty orders is
not covered by a per-row guarantee on the orders table. Relational DP is an
open research problem, and claiming otherwise would be the same kind of
fiction as data-derived bins.

What you give up
----------------
Marginals are independent, so **correlations between columns are not
preserved**. This is the standard DP baseline and it is an honest trade:
correlation structure is exactly what leaks individuals. Use
``quality-report`` to see the cost in utility terms before deciding the
trade is worth it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sdp.synthesizers.base import Synthesizer

logger = logging.getLogger(__name__)

#: Sentinel bucket so a column's null rate is privatised too, rather than
#: being read off the data in the clear.
NULL_TOKEN = "__NULL__"


@dataclass
class ColumnDomain:
    """A column's publicly declared value space.

    Built from the config, never from the data — see the module docstring.
    """
    kind: str                                   # "categorical" | "numeric"
    values: Optional[List[Any]] = None          # categorical
    low: Optional[float] = None                 # numeric
    high: Optional[float] = None                # numeric
    integer: bool = False

    def is_valid(self) -> bool:
        if self.kind == "categorical":
            return bool(self.values)
        if self.kind == "numeric":
            return (
                self.low is not None and self.high is not None
                and math.isfinite(self.low) and math.isfinite(self.high)
                and self.high > self.low
            )
        return False


@dataclass
class ColumnBudget:
    """What was spent on one column, and on what."""
    table: str
    column: str
    epsilon: float
    domain_kind: str
    bins: int
    measured: bool = True
    reason: str = ""


@dataclass
class PrivacyReport:
    """The accounting, so the guarantee can be checked rather than trusted."""
    epsilon_requested: float = 0.0
    per_table_epsilon: Dict[str, float] = field(default_factory=dict)
    columns: List[ColumnBudget] = field(default_factory=list)
    unmeasured: List[ColumnBudget] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon_requested": self.epsilon_requested,
            "per_table_epsilon": self.per_table_epsilon,
            "guarantee": (
                "epsilon-differential privacy per row, within each table; "
                "does not cover entities spanning multiple tables"
            ),
            "measured_columns": [
                {
                    "table": c.table, "column": c.column,
                    "epsilon": round(c.epsilon, 6),
                    "domain": c.domain_kind, "bins": c.bins,
                }
                for c in self.columns
            ],
            "unmeasured_columns": [
                {"table": c.table, "column": c.column, "reason": c.reason}
                for c in self.unmeasured
            ],
            "warnings": self.warnings,
        }


class DPMarginalEngine(Synthesizer):
    """Laplace-noised marginals over config-declared domains.

    Options
    -------
    ``epsilon``
        Total privacy budget per table. Default 1.0. Lower is more private
        and less accurate.
    ``numeric_bins``
        Histogram bins for numeric columns. Default 20. More bins capture
        shape better but spread the same budget thinner, so each bin is
        noisier.
    ``domains``
        ``{table: {column: ColumnDomain}}``. Injected by ``DataGenerator``
        from the config.
    ``frame_factory``
        ``(table, n) -> DataFrame`` producing config-only rows. Used for
        every column this engine does not privatise, so those columns never
        touch the training data.
    """

    name = "dp-marginal"
    description = "Differentially private marginals (Laplace) over config-declared domains"
    handles_relationships = False

    DEFAULT_EPSILON = 1.0
    DEFAULT_BINS = 20

    def __init__(self, *, seed: Optional[int] = None, **options: Any) -> None:
        super().__init__(seed=seed, **options)
        self.epsilon = float(options.get("epsilon", self.DEFAULT_EPSILON))
        self.numeric_bins = int(options.get("numeric_bins", self.DEFAULT_BINS))
        self._domains: Dict[str, Dict[str, ColumnDomain]] = options.get("domains") or {}
        self._frame_factory: Optional[Callable[[str, int], Any]] = options.get("frame_factory")

        # table → column → (bin_labels, probabilities)
        self._marginals: Dict[str, Dict[str, Tuple[List[Any], List[float]]]] = {}
        self.privacy = PrivacyReport(epsilon_requested=self.epsilon)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.numeric_bins < 2:
            raise ValueError("numeric_bins must be >= 2")

    @classmethod
    def is_available(cls) -> bool:
        return True                                  # numpy/pandas only

    @property
    def model(self) -> Any:
        return self._marginals or None

    # -- fit --------------------------------------------------------------

    def fit(
        self,
        sample_data: Dict[str, "object"],
        metadata: Optional[Any] = None,
    ) -> bool:
        import numpy as np

        rng = self._rng()
        if self.seed is not None:
            self.privacy.warnings.append(
                "seeded run: the Laplace noise is reproducible, so an adversary "
                "who knows the seed can subtract it. Use an unseeded run when "
                "the privacy guarantee needs to hold."
            )

        if not self._domains:
            self._note(
                "no column domains supplied — nothing can be privatised safely"
            )
            return False

        self._marginals = {}
        try:
            with self._timed("fit"):
                for table_name, df in sample_data.items():
                    self._fit_table(table_name, df, rng)
        except Exception as exc:
            self._note(f"fit failed: {exc}")
            self._marginals = {}
            self._fitted = False
            return False

        if not any(self._marginals.values()):
            self._note(
                "no column had a config-declared domain — a DP run would "
                "measure nothing. Declare business_values or min/max."
            )
            self._fitted = False
            return False

        self.stats.fit_rows = sum(len(df) for df in sample_data.values())
        measured = len(self.privacy.columns)
        self._note(
            f"fitted {measured} column(s) under ε={self.epsilon} per table "
            f"({len(self.privacy.unmeasured)} generated from config only)"
        )
        self._fitted = True
        return True

    def _fit_table(self, table_name: str, df: "object", rng: Any) -> None:
        domains = self._domains.get(table_name, {})

        # Only columns with a usable public domain are measured; the split
        # must be known before any measurement so the budget is fixed
        # independently of the data.
        measurable = [
            col for col in df.columns
            if col in domains and domains[col].is_valid()
        ]
        for col in df.columns:
            if col not in measurable:
                domain = domains.get(col)
                self.privacy.unmeasured.append(ColumnBudget(
                    table=table_name, column=str(col), epsilon=0.0,
                    domain_kind=domain.kind if domain else "none", bins=0,
                    measured=False,
                    reason=(
                        "no declared domain — generated from config rules, "
                        "no access to the data"
                    ),
                ))

        if not measurable:
            self._marginals[table_name] = {}
            self.privacy.per_table_epsilon[table_name] = 0.0
            return

        # Sequential composition within a table: epsilons add.
        epsilon_per_column = self.epsilon / len(measurable)
        self.privacy.per_table_epsilon[table_name] = self.epsilon

        table_marginals: Dict[str, Tuple[List[Any], List[float]]] = {}
        for col in measurable:
            domain = domains[col]
            labels, probabilities, bins = self._noisy_marginal(
                df[col], domain, epsilon_per_column, rng,
            )
            table_marginals[col] = (labels, probabilities)
            self.privacy.columns.append(ColumnBudget(
                table=table_name, column=str(col),
                epsilon=epsilon_per_column,
                domain_kind=domain.kind, bins=bins,
            ))

        self._marginals[table_name] = table_marginals

    def _noisy_marginal(
        self,
        series: "object",
        domain: ColumnDomain,
        epsilon: float,
        rng: Any,
    ) -> Tuple[List[Any], List[float], int]:
        """Histogram → Laplace noise → clip → renormalise.

        L1 sensitivity is 1: one row moves exactly one bin count by one, so
        Laplace at scale ``1/epsilon`` gives ``epsilon``-DP.
        """
        import numpy as np
        import pandas as pd

        if domain.kind == "categorical":
            labels: List[Any] = list(domain.values or []) + [NULL_TOKEN]
            as_str = {str(v): i for i, v in enumerate(domain.values or [])}
            counts = np.zeros(len(labels), dtype=float)
            for value in series:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    counts[-1] += 1
                    continue
                idx = as_str.get(str(value))
                # Values outside the declared domain are dropped. The rule is
                # data-independent, so this costs no privacy.
                if idx is not None:
                    counts[idx] += 1
        else:
            edges = np.linspace(domain.low, domain.high, self.numeric_bins + 1)
            labels = [(float(edges[i]), float(edges[i + 1]))
                      for i in range(self.numeric_bins)]
            labels.append(NULL_TOKEN)
            numeric = pd.to_numeric(pd.Series(list(series)), errors="coerce")
            counts = np.zeros(len(labels), dtype=float)
            counts[-1] = float(numeric.isna().sum())
            clean = numeric.dropna()
            if len(clean) > 0:
                # Clipping to the declared bounds is data-independent.
                clipped = clean.clip(domain.low, domain.high)
                hist, _ = np.histogram(clipped, bins=edges)
                counts[:-1] = hist.astype(float)

        noisy = counts + rng.laplace(loc=0.0, scale=1.0 / epsilon, size=len(counts))
        noisy = np.clip(noisy, 0.0, None)           # post-processing

        total = noisy.sum()
        if total <= 0:
            # Every bin was noised below zero — fall back to the uniform
            # distribution over the public domain, which uses no data.
            probabilities = np.full(len(noisy), 1.0 / len(noisy))
        else:
            probabilities = noisy / total

        return labels, [float(p) for p in probabilities], len(labels) - 1

    # -- sample -----------------------------------------------------------

    def sample(self, records_per_table: Dict[str, int]) -> Dict[str, "object"]:
        import numpy as np
        import pandas as pd

        if self._frame_factory is None:
            raise ValueError(
                "dp-marginal requires a frame_factory to generate the columns "
                "it does not privatise"
            )

        rng = self._rng()
        out: Dict[str, "object"] = {}
        with self._timed("sample"):
            for table_name, count in records_per_table.items():
                # Config-only rows first: this supplies primary keys, foreign
                # keys and every column without a declared domain, none of
                # which touch the training data.
                frame = self._frame_factory(table_name, count)
                if frame is None:
                    raise ValueError(f"dp-marginal: no frame produced for {table_name}")
                frame = frame.head(count).reset_index(drop=True)

                # Every measured column is written, whether or not the factory
                # produced it. Skipping absent ones would silently drop a
                # column the engine had already spent privacy budget on.
                for col, (labels, probabilities) in self._marginals.get(table_name, {}).items():
                    domain = self._domains.get(table_name, {}).get(col)
                    frame[col] = self._draw(labels, probabilities, count, domain, rng)

                out[table_name] = frame

        self.stats.sampled_rows = sum(len(df) for df in out.values())
        return out

    def _draw(
        self,
        labels: List[Any],
        probabilities: List[float],
        count: int,
        domain: Optional[ColumnDomain],
        rng: Any,
    ) -> List[Any]:
        """Draw from the noisy histogram — pure post-processing."""
        import numpy as np

        picks = rng.choice(len(labels), size=count, p=probabilities)
        values: List[Any] = []
        for index in picks:
            label = labels[index]
            if label == NULL_TOKEN:
                values.append(None)
            elif isinstance(label, tuple):
                # Uniform within the bin; the bin edges are public.
                low, high = label
                value = rng.uniform(low, high)
                values.append(int(round(value)) if domain and domain.integer else float(value))
            else:
                values.append(label)
        return values

    # -- reporting --------------------------------------------------------

    def privacy_report(self) -> Dict[str, Any]:
        """The privacy accounting for this fit."""
        return self.privacy.to_dict()

    def _rng(self) -> Any:
        import numpy as np

        return np.random.default_rng(self.seed)
