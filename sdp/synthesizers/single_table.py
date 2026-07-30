"""Per-table engines: Gaussian copula, CTGAN, TVAE.

These are SDV's *single-table* synthesizers, fitted once per table and
sampled independently. They cover the statistical and deep-learning
families the literature treats as the main alternatives to a hierarchical
model — copulas, GANs and VAEs.

Relationships
-------------
None of these engines model cross-table structure: each sees one table in
isolation. ``handles_relationships`` is therefore False, and referential
integrity comes entirely from the caller's FK resolution, which runs after
sampling and overwrites foreign-key columns with real parent keys.

The practical consequence is that a *child* table's FK columns are noise
until FK resolution runs, and correlations *between* tables are not
preserved — only within them. When cross-table structure matters, use the
``sdv`` HMA engine. When single-table fidelity matters more (which the
benchmarks suggest is often the case for wide tables), these usually win.

Cost
----
CTGAN and TVAE train a neural network per table. On CPU that is minutes,
not seconds, for a table of any width. ``epochs`` defaults to 100 rather
than SDV's 300 — a deliberate choice for a platform whose common case is
test-data generation rather than research-grade fidelity. Override it
explicitly when you want SDV's default::

    generate --engine ctgan --engine-option epochs=300
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Optional, Tuple

from sdp.synthesizers.base import Synthesizer

logger = logging.getLogger(__name__)


class _SingleTableEngine(Synthesizer):
    """Shared machinery: fit one synthesizer per table, sample per table."""

    #: ``(module, class)`` of the SDV synthesizer to instantiate.
    SYNTH: ClassVar[Tuple[str, str]] = ("", "")

    #: Engine-specific constructor defaults, overridable via options.
    DEFAULTS: ClassVar[Dict[str, Any]] = {}

    handles_relationships = False

    def __init__(self, *, seed: Optional[int] = None, **options: Any) -> None:
        super().__init__(seed=seed, **options)
        self._models: Dict[str, Any] = {}
        self._fitted_sample_sizes: Dict[str, int] = {}

    # -- capability -------------------------------------------------------

    @classmethod
    def _load_class(cls) -> Any:
        import importlib

        module_name, class_name = cls.SYNTH
        return getattr(importlib.import_module(module_name), class_name)

    @classmethod
    def is_available(cls) -> bool:
        try:
            cls._load_class()
            return True
        except Exception:                            # pragma: no cover
            return False

    @property
    def model(self) -> Any:
        """No single model — one per table, so this is the mapping itself.

        Returns the live dict rather than a copy: callers identity-check
        this against the model they were handed to decide whether this
        engine still owns it.
        """
        return self._models or None

    # -- contract ---------------------------------------------------------

    def fit(
        self,
        sample_data: Dict[str, "object"],
        metadata: Optional[Any] = None,
    ) -> bool:
        synth_cls = self._load_class()
        kwargs = {**self.DEFAULTS, **{
            k: v for k, v in self.options.items()
            if k not in {"locales", "verbose"}
        }}

        self._models = {}
        try:
            with self._timed("fit"):
                for table_name, df in sample_data.items():
                    table_meta = self._table_metadata(metadata, table_name, df)
                    if table_meta is None:
                        self._note(f"no metadata for table {table_name!r}")
                        return False
                    model = synth_cls(metadata=table_meta, **kwargs)
                    model.fit(df)
                    self._models[table_name] = model
                    self._fitted_sample_sizes[table_name] = len(df)
        except Exception as exc:
            self._note(f"fit failed: {exc}")
            self._models = {}
            self._fitted = False
            return False

        if not self._models:
            self._note("no tables were fitted")
            self._fitted = False
            return False

        self.stats.fit_rows = sum(self._fitted_sample_sizes.values())
        self._fitted = True
        return True

    def sample(self, records_per_table: Dict[str, int]) -> Dict[str, "object"]:
        out: Dict[str, "object"] = {}
        with self._timed("sample"):
            for table_name, count in records_per_table.items():
                model = self._models.get(table_name)
                if model is None:
                    raise ValueError(
                        f"{self.name}: no fitted model for table {table_name}"
                    )
                df = model.sample(num_rows=count)
                if df is None or len(df) < count:
                    raise ValueError(
                        f"{self.name}: got {0 if df is None else len(df)} rows "
                        f"for table {table_name}; expected {count}"
                    )
                out[table_name] = df.head(count).reset_index(drop=True)

        self.stats.sampled_rows = sum(len(df) for df in out.values())
        return out

    # -- internals --------------------------------------------------------

    @staticmethod
    def _table_metadata(metadata: Any, table_name: str, df: "object") -> Any:
        """Single-table metadata for ``table_name``.

        Prefers slicing the config-derived multi-table metadata, since that
        carries the declared sdtypes. Falls back to detecting from the frame
        when the metadata has no such table — detection is a guess, but a
        working guess beats refusing to run.
        """
        if metadata is not None:
            try:
                return metadata.get_table_metadata(table_name)
            except Exception as exc:
                logger.debug("get_table_metadata(%s) failed: %s", table_name, exc)

        try:
            from sdv.metadata import Metadata

            return Metadata.detect_from_dataframe(data=df, table_name=table_name)
        except Exception as exc:                     # pragma: no cover
            logger.debug("detect_from_dataframe(%s) failed: %s", table_name, exc)
            return None


class GaussianCopulaEngine(_SingleTableEngine):
    """Fast statistical baseline — fits marginals plus a copula."""

    name = "gaussian-copula"
    description = "SDV GaussianCopulaSynthesizer — fast statistical baseline, per table"
    SYNTH = ("sdv.single_table", "GaussianCopulaSynthesizer")


class CTGANEngine(_SingleTableEngine):
    """Conditional tabular GAN. Strong on mixed types; expensive."""

    name = "ctgan"
    description = "SDV CTGANSynthesizer — conditional GAN, per table (slow, needs torch)"
    SYNTH = ("sdv.single_table", "CTGANSynthesizer")
    DEFAULTS = {"epochs": 100, "verbose": False}


class TVAEEngine(_SingleTableEngine):
    """Tabular VAE. Usually trains faster than CTGAN, smoother output."""

    name = "tvae"
    description = "SDV TVAESynthesizer — variational autoencoder, per table (needs torch)"
    SYNTH = ("sdv.single_table", "TVAESynthesizer")
    DEFAULTS = {"epochs": 100}
