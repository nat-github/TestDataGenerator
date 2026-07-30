"""Anchored generation — tables loaded verbatim from real source data.

Extracted from ``data_generator.py`` as a mixin. ``DataGenerator`` inherits
it, so ``self`` resolves exactly as before — this is a pure move, not a
behaviour change. The split exists so each concern can be read and tested
without loading a 2,400-line class.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import string
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, localcontext, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sdp.models.config_models import TableConfig, RelationshipConfig

logger = logging.getLogger(__name__)


class AnchorMixin:
    """Anchored generation — tables loaded verbatim from real source data."""

    def _resolve_source_path(self, source: str) -> Path:
        """Resolve a table's `source:` path — as given, then relative to the
        config file's directory, then relative to the working directory."""
        candidates = [Path(source)]
        config_dir = Path(self.config_file).resolve().parent
        candidates.append(config_dir / source)
        candidates.append(Path.cwd() / source)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # Nothing matched — return the most informative candidate for the error.
        return candidates[0]

    def _load_anchor_tables(self) -> None:
        """Load every table that declares a `source:` dataset into ``anchor_data``.

        Supported formats: ``.parquet`` and ``.csv``. The loaded data must
        contain all columns declared for the table in the config; extra columns
        are dropped with a warning. Anchor tables are never generated — they are
        used verbatim and also feed SDV training.
        """
        self.anchor_data = {}
        for table_name, table_config in self.tables_config.items():
            source = getattr(table_config, "source", None)
            if not source:
                continue

            path = self._resolve_source_path(str(source))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Anchor table '{table_name}': source dataset not found — '{source}'"
                )

            suffix = path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            elif suffix == ".csv":
                df = pd.read_csv(path)
            else:
                raise ValueError(
                    f"Anchor table '{table_name}': unsupported source format '{suffix}' "
                    f"— use .parquet or .csv"
                )

            if df.empty:
                raise ValueError(f"Anchor table '{table_name}': source dataset '{path}' has no rows")

            configured = [c.column_name for c in table_config.columns]
            missing = [c for c in configured if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Anchor table '{table_name}': source dataset is missing configured "
                    f"column(s) {missing}. The source file must contain every configured column."
                )
            extra = [c for c in df.columns if c not in configured]
            if extra:
                self.logger.warning(
                    f"⚠️ Anchor table '{table_name}': dropping {len(extra)} unconfigured "
                    f"column(s) from source data: {extra}"
                )
            # Keep configured columns, in configured order.
            self.anchor_data[table_name] = df[configured].reset_index(drop=True)
            self.logger.info(
                f"📌 Anchor table '{table_name}': loaded {len(df)} real row(s) from {path}"
            )

    def _is_anchor_table(self, table_name: str) -> bool:
        """True when the table's data comes from a real `source:` dataset."""
        return table_name in self.anchor_data

    def _inject_anchor_tables(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Replace any anchor tables in ``data`` with their real (verbatim) rows.

        Called before foreign-key resolution so generated child tables resolve
        their FKs against the anchor's real key values.
        """
        for table_name, anchor_df in self.anchor_data.items():
            data[table_name] = anchor_df.copy()
        return data
