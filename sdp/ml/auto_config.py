"""Auto-config: infer a YAML config from a sample data file.

Reads a CSV, Parquet, or Excel file containing *real* data and produces a
ready-to-use YAML config that can be passed directly to:

    python main.py generate --config <output.yaml>

What gets inferred per column
──────────────────────────────
  • Data type (N, N19, N38, DC, D, DT, TS, VA*, A*, T)
  • Primary key candidates   (100% unique, 0% null)
  • Low-cardinality values   → `values:` (business_values)
  • Numeric/date bounds      → `min:` / `max:`
  • Statistical distribution → `distribution:` block (scipy fit)
  • PII / sensitive columns  → `special_rules:` suggestion + warning

CLI usage:
    python main.py infer-config --input data/customers.csv --output config/customers.yaml
    python main.py infer-config --input data/orders.parquet --output config/orders.yaml

Python API:
    from sdp.ml.auto_config import AutoConfigInferrer
    cfg = AutoConfigInferrer().infer_from_file("data/customers.csv")
    # cfg is a plain dict — pass to yaml.dump()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_BV_UNIQUE   = 30     # max unique values to capture as business_values
_BV_MAX_FRACTION = 0.10   # skip BV if unique/total > this AND unique > 10
_MIN_DIST_ROWS   = 50     # minimum non-null rows to attempt dist fitting
_SAMPLE_SIZE     = 5_000  # max rows read from a file for inference


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------
def _infer_col_type(series: pd.Series) -> str:
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "A1"

    if pd.api.types.is_integer_dtype(dtype):
        clean = series.dropna()
        max_abs = int(clean.abs().max()) if len(clean) else 0
        if max_abs < 10**6:
            return "N"
        if max_abs < 10**9:
            return "N19"
        return "N38"

    if pd.api.types.is_float_dtype(dtype):
        return "DC"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        clean = series.dropna()
        if len(clean) > 0:
            try:
                if (clean.dt.hour == 0).all() and (clean.dt.minute == 0).all():
                    return "D"
            except AttributeError:
                pass
        return "DT"

    # String / object — infer from observed max length
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return "VA256"
    max_len = int(clean.str.len().max())
    if max_len <= 1:   return "VA1"
    if max_len <= 3:   return "VA3"
    if max_len <= 18:  return "VA18"
    if max_len <= 50:  return "VA50"
    if max_len <= 256: return "VA256"
    return "T"


class AutoConfigInferrer:
    """
    Infer a YAML config from one or more sample data files.

    Parameters
    ──────────
    fit_distributions
        Fit scipy distributions to numeric columns (adds ``distribution:``
        block). Requires scipy (already a transitive dep via SDV).
    scan_pii
        Run PII detector and suggest ``special_rules:`` for flagged columns.
    max_business_values
        Max unique values to store as ``values:`` (low-cardinality).
    sample_size
        Max rows to read from each file for inference.
    pii_confidence
        Minimum confidence score to emit a PII suggestion.
    """

    def __init__(
        self,
        fit_distributions: bool = True,
        scan_pii: bool = True,
        max_business_values: int = _MAX_BV_UNIQUE,
        sample_size: int = _SAMPLE_SIZE,
        pii_confidence: float = 0.70,
    ):
        self.fit_distributions    = fit_distributions
        self.scan_pii             = scan_pii
        self.max_business_values  = max_business_values
        self.sample_size          = sample_size
        self.pii_confidence       = pii_confidence
        self._pii_detector        = None
        self._dist_fitter         = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer_from_file(
        self,
        path: str,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read a CSV / Parquet / Excel file and return a YAML config dict."""
        p = Path(path)
        suffix = p.suffix.lower()
        logger.info(f"Reading {p.name} ...")

        if suffix == ".csv":
            df = pd.read_csv(p, nrows=self.sample_size)
        elif suffix == ".parquet":
            df = pd.read_parquet(p)
            if len(df) > self.sample_size:
                df = df.sample(self.sample_size, random_state=42)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p, nrows=self.sample_size)
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                "Supported: .csv, .parquet, .xlsx, .xls"
            )

        tname = table_name or p.stem.lower().replace(" ", "_").replace("-", "_")
        table_cfg = self.infer_from_dataframe(df, tname)
        return _build_config([table_cfg])

    def infer_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> Dict[str, Any]:
        """Return a single-table dict (one entry in the `tables` list)."""
        logger.info(
            f"Inferring schema for '{table_name}' "
            f"({len(df)} rows × {len(df.columns)} cols)"
        )

        pk_candidates = self._detect_pk_candidates(df)

        pii_map: Dict[str, Any] = {}
        if self.scan_pii:
            from sdp.ml.pii_detector import PIIDetector
            detector = PIIDetector(confidence_threshold=self.pii_confidence)
            findings = detector.scan_dataframe(df, table_name)
            pii_map = {f.column: f for f in findings}
            if findings:
                logger.info(
                    f"  PII detector flagged {len(findings)} column(s): "
                    + ", ".join(f.column for f in findings)
                )

        columns = [
            self._infer_column(df[col], col, table_name, pk_candidates, pii_map)
            for col in df.columns
        ]

        return {
            "name":    table_name,
            "rows":    max(len(df), 1000),
            "columns": columns,
        }

    # ------------------------------------------------------------------
    # Column inference
    # ------------------------------------------------------------------
    def _infer_column(
        self,
        series:        pd.Series,
        col_name:      str,
        table_name:    str,
        pk_candidates: List[str],
        pii_map:       Dict,
    ) -> Dict[str, Any]:
        col_type  = _infer_col_type(series)
        null_rate = _null_rate(series)
        is_pk     = col_name in pk_candidates

        cfg: Dict[str, Any] = {
            "name": col_name.upper(),
            "type": col_type,
        }

        if is_pk:
            cfg["pk"] = True

        if null_rate > 0.001:
            cfg["null_rate"] = round(null_rate, 3)

        # Business values (low-cardinality, non-PK)
        if not is_pk:
            bv = self._compute_business_values(series)
            if bv:
                cfg["values"] = bv
            else:
                # Numeric / date bounds
                bounds = self._compute_bounds(series, col_type)
                if bounds:
                    cfg.update(bounds)

                # Distribution fitting
                if self.fit_distributions:
                    dist = self._try_fit_distribution(series, col_type)
                    if dist:
                        cfg["distribution"] = dist

        # PII suggestion (overrides distribution — if PII detected, use rule)
        if col_name in pii_map:
            finding = pii_map[col_name]
            if finding.suggested_rule:
                cfg["special_rules"] = finding.suggested_rule
                cfg.pop("distribution", None)   # rule-based is more precise
                cfg["_pii_note"] = (
                    f"{finding.pii_type} detected (conf={finding.confidence:.0%}) "
                    f"— review and remove _pii_note before committing"
                )
                logger.warning(
                    f"  ⚠️  {table_name}.{col_name}: likely {finding.pii_type} "
                    f"({finding.confidence:.0%}) → special_rules: {finding.suggested_rule}"
                )

        return cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_pk_candidates(self, df: pd.DataFrame) -> List[str]:
        return [
            col for col in df.columns
            if df[col].notna().all() and df[col].nunique() == len(df)
        ]

    def _compute_business_values(self, series: pd.Series) -> Optional[str]:
        clean = series.dropna()
        n_unique = clean.nunique()
        if n_unique == 0 or n_unique > self.max_business_values:
            return None
        fraction = n_unique / max(len(clean), 1)
        if fraction > _BV_MAX_FRACTION and n_unique > 10:
            return None
        return ";".join(str(v) for v in clean.unique())

    def _compute_bounds(
        self, series: pd.Series, col_type: str
    ) -> Optional[Dict[str, Any]]:
        if col_type.startswith("N") or col_type == "DC":
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 2:
                mn, mx = float(numeric.min()), float(numeric.max())
                # Use int for integer types so YAML is clean
                if col_type.startswith("N"):
                    return {"min": int(mn), "max": int(mx)}
                return {"min": round(mn, 4), "max": round(mx, 4)}
        if col_type in ("D", "DT", "TS"):
            dt = pd.to_datetime(series, errors="coerce").dropna()
            if len(dt) >= 2:
                fmt = "%Y-%m-%d" if col_type == "D" else "%Y-%m-%d %H:%M:%S"
                return {"min": dt.min().strftime(fmt), "max": dt.max().strftime(fmt)}
        return None

    def _try_fit_distribution(
        self, series: pd.Series, col_type: str
    ) -> Optional[Dict]:
        if not (col_type.startswith("N") or col_type == "DC"):
            return None
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) < _MIN_DIST_ROWS:
            return None
        try:
            if self._dist_fitter is None:
                from sdp.ml.distribution_fitter import DistributionFitter
                self._dist_fitter = DistributionFitter()
            return self._dist_fitter.fit(numeric)
        except Exception as exc:
            logger.debug(f"Distribution fit failed for column: {exc}")
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _null_rate(series: pd.Series) -> float:
    return 0.0 if len(series) == 0 else float(series.isna().sum()) / len(series)


def _build_config(table_configs: List[Dict]) -> Dict[str, Any]:
    return {
        "config_format": "sdp-yaml-v1",
        "run_settings": {
            "default_records_per_table": 1000,
        },
        "tables": table_configs,
    }
