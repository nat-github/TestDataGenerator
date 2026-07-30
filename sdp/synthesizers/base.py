"""The synthesizer plugin interface.

Every generation engine — SDV's HMA, a per-table GAN, the rule-based
fallback — implements the same two-method contract:

    fit(sample_data, metadata)  → did training succeed?
    sample(records_per_table)   → {table_name: DataFrame}

Why this exists
---------------
The consistent finding across the synthetic-data literature is that **no
single method dominates**. Fidelity, utility and privacy trade off against
one another, and the ranking shifts with the dataset (Systematic Assessment
of Tabular Data Synthesis, arXiv:2402.06806). A platform hardcoded to one
engine cannot act on that finding — it cannot even measure it.

``DataGenerator`` keeps everything that depends on *config semantics*:
metadata construction, primary keys, foreign-key resolution, type coercion,
rules and derived columns, Parquet export. An engine only has to fit a
model and draw rows from it. That seam is what makes engines swappable
without each one re-implementing the platform.

Engines report their own cost
-----------------------------
``EngineStats`` records fit/sample wall-clock and row counts. The 417-model
survey (arXiv:2401.02524) singles out the neglect of training and
computational cost as a gap in the literature; an engine that is 3% more
faithful and 40× slower is not obviously the better choice, and you cannot
have that argument without the numbers.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """What an engine cost to run."""
    engine: str = ""
    fit_seconds: float = 0.0
    sample_seconds: float = 0.0
    fit_rows: int = 0
    sampled_rows: int = 0
    retries: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.sample_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "fit_seconds": round(self.fit_seconds, 3),
            "sample_seconds": round(self.sample_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "fit_rows": self.fit_rows,
            "sampled_rows": self.sampled_rows,
            "retries": self.retries,
            "notes": self.notes,
        }


class Synthesizer(ABC):
    """Base class for generation engines.

    Subclasses set ``name`` and implement ``fit`` and ``sample``. Keep
    ``fit`` total: return ``False`` on failure rather than raising, so the
    caller can fall back to another engine — a failed fit is an expected
    outcome on messy configs, not an exceptional one.
    """

    #: Registry key, e.g. ``"sdv"``. Set by every concrete subclass.
    name: ClassVar[str] = ""

    #: One-line description shown by ``sdp generate --list-engines``.
    description: ClassVar[str] = ""

    #: Whether the engine learns relationships between tables itself. When
    #: False, the caller's FK resolution is solely responsible for
    #: referential integrity.
    handles_relationships: ClassVar[bool] = False

    def __init__(self, *, seed: Optional[int] = None, **options: Any) -> None:
        self.seed = seed
        self.options = options
        self.stats = EngineStats(engine=self.name)
        self._fitted = False

    # -- capability -------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether this engine's dependencies are importable.

        Checked before instantiation so a missing optional package produces
        a clear message rather than an ImportError mid-run.
        """
        return True

    # -- contract ---------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        sample_data: Dict[str, "object"],     # table → pandas.DataFrame
        metadata: Optional[Any] = None,
    ) -> bool:
        """Train on sample data. Returns success; must not raise."""

    @abstractmethod
    def sample(self, records_per_table: Dict[str, int]) -> Dict[str, "object"]:
        """Draw the requested number of rows per table.

        Raises on failure — by this point the caller has committed to this
        engine, so a silent empty result would be worse than an exception.
        """

    # -- state ------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def model(self) -> Any:
        """The underlying model object, or None.

        Exposed so callers can persist it (artifact caching) or inspect it.
        Engines with no single model object return None.
        """
        return None

    # -- helpers for subclasses -------------------------------------------

    @contextmanager
    def _timed(self, phase: str) -> Iterator[None]:
        """Record wall-clock for ``fit`` or ``sample`` into ``stats``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            if phase == "fit":
                self.stats.fit_seconds += elapsed
            else:
                self.stats.sample_seconds += elapsed

    def _note(self, message: str) -> None:
        self.stats.notes.append(message)
        logger.info("[%s] %s", self.name, message)

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        state = "fitted" if self._fitted else "unfitted"
        return f"<{type(self).__name__} name={self.name!r} {state}>"


def sample_multi_table(
    model: Any,
    records_per_table: Dict[str, int],
    fitted_sample_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, "object"]:
    """Draw rows from an SDV-style multi-table model and trim to size.

    Split out of ``DataGenerator`` so the HMA engine and any directly
    injected synthesizer go through identical logic.

    Older SDV multi-table synthesizers accept ``scale`` rather than
    ``num_rows``. When ``num_rows`` is rejected we translate the request
    into the largest ratio of requested-to-fitted rows, which over-samples
    every table, then trim. Over-sampling is deliberate: too many rows can
    be cut, too few cannot.
    """
    if model is None:
        raise ValueError("Synthesizer is not initialized")

    try:
        sampled = model.sample(num_rows=records_per_table)
    except TypeError as exc:
        if "num_rows" not in str(exc):
            raise

        ratios = []
        for table_name, requested in records_per_table.items():
            fitted = (fitted_sample_sizes or {}).get(table_name)
            if fitted:
                ratios.append(requested / max(1, fitted))

        scale = max(ratios) if ratios else 1.0
        sampled = model.sample(scale=max(scale, 1e-6))

    if not isinstance(sampled, dict):
        raise ValueError("Synthesizer returned a non-dictionary result")

    trimmed: Dict[str, "object"] = {}
    for table_name, requested in records_per_table.items():
        table_df = sampled.get(table_name)
        if table_df is None or table_df.empty:
            raise ValueError(f"Synthesizer returned no rows for table {table_name}")
        if len(table_df) < requested:
            raise ValueError(
                f"Synthesizer returned only {len(table_df)} rows for table "
                f"{table_name}; expected at least {requested}"
            )
        trimmed[table_name] = table_df.head(requested).reset_index(drop=True)

    return trimmed
