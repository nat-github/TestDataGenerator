"""SDV HMA engine — the platform's historical default.

``HMASynthesizer`` is hierarchical: it models parent/child tables jointly
and reproduces cross-table structure natively, which is why it is the
default for relational configs. It is also the most expensive engine here,
and its fit can fail outright on awkward column types — hence
``fit`` returning False rather than raising, so the caller can fall back.

Retry policy deliberately lives with the caller, not here. Regenerating
sample data with more aggressive sanitisation needs config knowledge this
class does not have, and *when to give up* is a caller decision.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sdp.synthesizers.base import Synthesizer, sample_multi_table

logger = logging.getLogger(__name__)


class HMAEngine(Synthesizer):
    """SDV ``HMASynthesizer`` — multi-table, relationship-aware."""

    name = "sdv"
    description = "SDV HMASynthesizer — hierarchical, models table relationships natively"
    handles_relationships = True

    #: Historical default, kept so behaviour is unchanged for existing configs.
    DEFAULT_LOCALES = ["nl_NL"]

    def __init__(self, *, seed: Optional[int] = None, **options: Any) -> None:
        super().__init__(seed=seed, **options)
        self._model: Any = None
        self._fitted_sample_sizes: Dict[str, int] = {}

    @classmethod
    def is_available(cls) -> bool:
        try:
            from sdv.multi_table import HMASynthesizer  # noqa: F401
            return True
        except Exception:                            # pragma: no cover
            return False

    @property
    def model(self) -> Any:
        return self._model

    @property
    def fitted_sample_sizes(self) -> Dict[str, int]:
        """Rows per table used for training — needed to translate a row
        request into a ``scale`` on SDV versions without ``num_rows``."""
        return dict(self._fitted_sample_sizes)

    def adopt(self, model: Any, fitted_sample_sizes: Optional[Dict[str, int]] = None) -> None:
        """Take ownership of an already-fitted model.

        Used by the artifact cache, which restores a pickled synthesizer
        instead of retraining.
        """
        self._model = model
        self._fitted_sample_sizes = dict(fitted_sample_sizes or {})
        self._fitted = model is not None
        if self._fitted:
            self._note("adopted a pre-fitted model (no training performed)")

    def fit(
        self,
        sample_data: Dict[str, "object"],
        metadata: Optional[Any] = None,
    ) -> bool:
        if metadata is None:
            self._note("no metadata supplied — cannot build an HMA synthesizer")
            return False

        from sdv.multi_table import HMASynthesizer

        locales = self.options.get("locales", self.DEFAULT_LOCALES)
        verbose = bool(self.options.get("verbose", True))

        try:
            with self._timed("fit"):
                self._model = HMASynthesizer(
                    metadata=metadata, verbose=verbose, locales=locales,
                )
                self._model.fit(sample_data)
        except Exception as exc:
            self._note(f"fit failed: {exc}")
            self._model = None
            self._fitted = False
            return False

        self._fitted_sample_sizes = {
            table: len(df) for table, df in sample_data.items()
        }
        self.stats.fit_rows = sum(self._fitted_sample_sizes.values())
        self._fitted = True
        return True

    def sample(self, records_per_table: Dict[str, int]) -> Dict[str, "object"]:
        with self._timed("sample"):
            out = sample_multi_table(
                self._model, records_per_table, self._fitted_sample_sizes,
            )
        self.stats.sampled_rows = sum(len(df) for df in out.values())
        return out
