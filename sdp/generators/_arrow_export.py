"""Arrow type casting and Parquet export.

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


class ArrowExportMixin:
    """Arrow type casting and Parquet export.

    Note on error handling in this module: the coercion helpers run **once
    per cell**, so a table of a million rows enters these handlers a million
    times. Logging inside them is not an option — a single malformed column
    would produce a million log lines and dominate the run.

    The handlers therefore name the exceptions that can actually occur
    (``InvalidOperation``, ``ValueError``, ``TypeError``, ``OverflowError``)
    and coerce the offending cell to ``None`` without comment. An unexpected
    exception type now propagates instead of being silently swallowed, which
    is the behaviour change: a genuine bug surfaces, a bad value does not.
    """

    @staticmethod
    def _to_arrow_dc(series: pd.Series, precision: int, scale: int) -> pa.Array:
        """
        Convert a Series to Arrow Decimal(precision, scale), raising local context precision
        to avoid decimal.InvalidOperation on large coefficients.
        """
        exp = Decimal(1).scaleb(-scale)  # exponent 10^-scale

        dec_vals: list[Decimal | None] = []
        with localcontext() as ctx:
            # safe margin above requested precision
            ctx.prec = max(precision + 2, 40)

            for v in series:
                # Nulls
                if pd.isna(v):
                    dec_vals.append(None)
                    continue

                s = str(v).strip().replace(',', '').replace('_', '')
                sl = s.lower()
                if sl in {'nan', 'inf', '+inf', '-inf'}:
                    dec_vals.append(None)
                    continue

                # Build Decimal (prefer string to avoid float artifacts)
                try:
                    d0 = Decimal(s)
                except (InvalidOperation, ValueError, TypeError):
                    try:
                        fv = float(s)
                        if not (float('-inf') < fv < float('inf')):
                            dec_vals.append(None)
                            continue
                        d0 = Decimal(str(fv))
                    except (InvalidOperation, ValueError, TypeError, OverflowError):
                        dec_vals.append(None)
                        continue

                # >>> Your fixed try/except block <<<
                try:
                    dq = d0.quantize(exp, rounding=ROUND_HALF_UP)
                except InvalidOperation:
                    # Integral fallback then apply scale
                    try:
                        dq = (
                            d0.to_integral_value(rounding=ROUND_HALF_UP)
                            .quantize(exp, rounding=ROUND_HALF_UP)
                        )
                    except (InvalidOperation, OverflowError, ValueError):
                        dec_vals.append(None)
                        continue

                dec_vals.append(dq)

        pa_type = pa.decimal128(precision, scale) if precision <= 38 else pa.decimal256(precision, scale)
        return pa.array(dec_vals, type=pa_type)

    @staticmethod
    def _count_digits_int_str(x: str) -> int:
        """Count integer digits in a numeric string (ignoring sign and fractional part)."""
        s = x.strip()
        if s.startswith('-') or s.startswith('+'):
            s = s[1:]
        if '.' in s:
            s = s.split('.', 1)[0]
        # Remove thousands separators/underscores if any
        s = s.replace(',', '').replace('_', '')
        return len(s) if s.isdigit() else 0

    @staticmethod
    def _to_arrow_bigint(series: pd.Series) -> pa.Array:

        dec_vals: list[Decimal | None] = []
        max_digits = 1
        for v in series:
            if pd.isna(v):
                dec_vals.append(None);
                continue
            s = str(v).strip().replace(',', '').replace('_', '')
            l = s.lower()
            if l in {'nan', 'inf', '+inf', '-inf'}:
                dec_vals.append(None);
                continue
            try:
                d = Decimal(s)  # exact (string-based)
            except (InvalidOperation, ValueError, TypeError):
                # fallback: float -> str -> Decimal, still reject non-finite
                try:
                    fv = float(s)
                    if not (float('-inf') < fv < float('inf')):
                        dec_vals.append(None);
                        continue
                    d = Decimal(str(fv))
                except (InvalidOperation, ValueError, TypeError, OverflowError):
                    dec_vals.append(None);
                    continue
            # Round to integer with HALF_UP (scale=0)
            di = d.to_integral_value(rounding=ROUND_HALF_UP)
            dec_vals.append(di)
            # Update max digits in integer part
            formatted = format(di, 'f').lstrip('+-').replace('.', '').lstrip('0')
            max_digits = max(max_digits, len(formatted) or 1)

        # Choose Arrow decimal type (scale=0)
        if max_digits <= 38:
            pa_type = pa.decimal128(max_digits, 0)
        else:
            pa_type = pa.decimal256(max_digits,
                                    0)  # up to 76 digits  [7](https://www.52spain.com/d/117547-a-complete-guide-to-spanish-bank-account-iban-format-avoid-transfer-hassles)
        string_vals = [None if value is None else format(value, 'f') for value in dec_vals]
        return pa.array(string_vals, type=pa.string()).cast(pa_type)

    def export_to_parquet(self, output_dir: str = "output"):
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Ensure FKs are resolved before export
            self._resolve_foreign_keys()

            files_exported = 0
            total_records = 0

            for table_name, data in self.generated_data.items():
                if data.empty:
                    self.logger.warning(f"⚠️ Table {table_name} is empty, skipping export")
                    continue

                table_cfg = self.tables_config[table_name]
                export_file = output_path / f"{table_name}.parquet"

                arrays: List[pa.Array] = []
                names: List[str] = []

                for column in table_cfg.columns:
                    col = column.column_name
                    if col not in data.columns:
                        continue

                    s = data[col]
                    base_type, _, precision, parsed_scale = self.config_parser.parse_data_type_details(column.data_type)

                    if base_type in ["DT", "TS"]:
                        iso = self._to_iso_datetime_strings(s)
                        arr_str = pa.array(iso, type=pa.string())
                        arr_ts_naive = pc.strptime(arr_str, format="%Y-%m-%d %H:%M:%S", unit="us", error_is_null=True)
                        arr_ts = arr_ts_naive.cast(pa.timestamp('us', tz='UTC'))
                        arrays.append(arr_ts)
                        names.append(col)

                    elif base_type == "D":
                        dt_series = pd.to_datetime(s, format="mixed", errors="coerce").dt.date
                        arr_date = pa.array(dt_series, type=pa.date32(), from_pandas=True)
                        arrays.append(arr_date)
                        names.append(col)

                    elif base_type == "T":
                        arr = pa.array(s.astype("string"))
                        arrays.append(arr)
                        names.append(col)

                    elif base_type == "NS":
                        arr = pa.array(s.astype("string"))
                        arrays.append(arr)
                        names.append(col)





                    elif base_type == "N":

                        # Existing numeric export, but robust for N38 (≥20 integer digits)

                        # 1) Parse declared length (if provided)

                        _bt, declared_len, _prec, _sc = self.config_parser.parse_data_type_details(column.data_type)

                        try:

                            declared_len = int(declared_len) if declared_len else None

                        except (TypeError, ValueError):

                            declared_len = None

                        # 2) Detect oversize beyond 64-bit (either by declared_len or observed digits)

                        oversize_64 = False

                        if declared_len and declared_len > 19:

                            oversize_64 = True

                        else:

                            for v in s.dropna():

                                digits = self._count_digits_int_str(str(v))

                                if digits > 19:
                                    oversize_64 = True

                                    break

                        if oversize_64:

                            # N38 (or similar): export as Decimal with scale=0 (exact integers)

                            arr = self._to_arrow_bigint(s)

                            arrays.append(arr);
                            names.append(col)

                        else:

                            # Normal int64 path — build Arrow array from Python ints (avoid pandas UInt64 cast)

                            vals_num = pd.to_numeric(s, errors="coerce")

                            int_list = [None if pd.isna(v) else int(v) for v in vals_num]

                            INT64_MIN = np.iinfo(
                                np.int64).min  # dtype, not string  [5](https://en.wikipedia.org/wiki/International_Bank_Account_Number)

                            INT64_MAX = np.iinfo(np.int64).max

                            min_val = vals_num.min(skipna=True)

                            max_val = vals_num.max(skipna=True)

                            if pd.isna(min_val) or pd.isna(max_val):

                                arr = pa.array(int_list, type=pa.int64())

                            elif min_val >= INT64_MIN and max_val <= INT64_MAX:

                                arr = pa.array(int_list, type=pa.int64())

                            else:

                                # Strictly non-negative & within uint64? use uint64; otherwise fallback to decimal(precision<=38)

                                UINT64_MAX = np.iinfo(np.uint64).max

                                if min_val >= 0 and max_val <= UINT64_MAX:

                                    arr = pa.array(int_list, type=pa.uint64())

                                else:

                                    # Extremely rare; ensure exact integer via Decimal128 up to 38 digits

                                    arr = self._to_arrow_bigint(s)

                            arrays.append(arr);
                            names.append(col)


                    elif base_type == "DC":
                        precision = int(precision or 18)
                        scale = int(getattr(column, "scale", None) or (parsed_scale or 2))
                        q = Decimal("1." + "0" * scale)
                        dec_vals = [
                            (None if pd.isna(v) else Decimal(str(v)).quantize(q, rounding=ROUND_HALF_UP)) for v in s
                        ]
                        arr = pa.array(dec_vals, type=pa.decimal128(precision, scale))
                        arrays.append(arr)
                        names.append(col)

                    else:
                        arr = pa.array(s.astype("string"))
                        arrays.append(arr)
                        names.append(col)

                table = pa.Table.from_arrays(arrays, names=names)
                pq.write_table(table, export_file)
                files_exported += 1
                total_records += len(data)

                self.logger.info(f"💾 Exported {table_name}.parquet ({len(data)} records)")

            self.logger.info(f"✅ Successfully exported {files_exported} files with {total_records} total records")
        except Exception as e:
            self.logger.error(f"❌ Error exporting data: {e}")
            raise
