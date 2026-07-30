"""Primary-key generation and uniqueness repair.

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


class PrimaryKeyMixin:
    """Primary-key generation and uniqueness repair."""

    def _initialize_pk_tracking(self):
        """Initialize primary key tracking for all tables"""
        for table_name, table_config in self.tables_config.items():
            pk_columns = [col for col in table_config.columns if col.is_pk]
            for pk_col in pk_columns:
                self.pk_sequences[(table_name, pk_col.column_name)] = 1
                self.used_pk_values[(table_name, pk_col.column_name)] = set()

    def _generate_unique_primary_key(self, column, table_name: str, index: int, num_records: int) -> Any:
        """
        GUARANTEED UNIQUE primary key generation
        Now HONORS special_rules (e.g., NL_IBAN, BBAN) for PKs as well.
        """
        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)
        pk_key = (table_name, column.column_name)

        # 0) If PK has explicit special_rules, generate via helpers and ensure uniqueness
        special = getattr(column, "special_rules", None)
        if special and not pd.isna(special):
            max_length = self._resolve_generator_max_length(base_type, length)
            for attempt in range(200):
                val = self.helpers.generate_special_value(
                    special,
                    column.data_type,
                    column_name=column.column_name,
                    max_length=max_length,
                )
                # enforce uniqueness for PK
                if val not in self.used_pk_values[pk_key]:
                    self.used_pk_values[pk_key].add(val)
                    return val
            # If we somehow collide, fall through to sequential

        # 1) Business values logic (existing)
        business_values = self.helpers.parse_business_values(column.business_values)
        if business_values:
            if index < len(business_values):
                pk_value = business_values[index]
                if pk_value in self.used_pk_values[pk_key]:
                    self.logger.warning(
                        f"⚠️ Business value duplicate detected for {table_name}.{column.column_name}, using sequential")
                    return self._generate_sequential_pk(column, table_name, index, num_records)
                self.used_pk_values[pk_key].add(pk_value)
                return pk_value
            else:
                warn_key = (table_name, column.column_name)
                if warn_key not in self._bv_overflow_warned:
                    self._bv_overflow_warned.add(warn_key)
                    self.logger.warning(
                        f"⚠️ More records requested than business values for {table_name}.{column.column_name}, generating sequential")
                return self._generate_sequential_pk(column, table_name, index, num_records)

        # 2) Default sequential PK (existing)
        return self._generate_sequential_pk(column, table_name, index, num_records)

    def _generate_sequential_pk(self, column, table_name: str, index: int, num_records: int) -> Any:
        """
        Generate guaranteed unique sequential primary key values
        IGNORES data type length constraints for PK uniqueness
        """
        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)
        pk_key = (table_name, column.column_name)

        # Calculate what the value SHOULD be based on sequence
        seq_val = self.pk_sequences[pk_key] + index

        # Generate based on data type, but IGNORE length constraints for uniqueness
        if base_type == "N":
            # For numeric PK, just use the sequence value regardless of length
            # If it exceeds N3 length, we still use it to maintain uniqueness
            pk_value = seq_val

            # Only apply min constraint, ignore max for uniqueness
            min_val = 1  # PKs usually start from 1
            if pk_value < min_val:
                pk_value = min_val

        elif base_type == "NS":
            # For numeric string, convert to string but don't truncate
            pk_value = str(seq_val)
            # Only apply zero-padding up to original length, but don't truncate
            if length and len(pk_value) < length:
                pk_value = pk_value.zfill(length)
            # If longer than specified length, keep it as-is for uniqueness

        elif base_type == "A":
            # For alphabetic, generate beyond specified length if needed
            if length and seq_val <= (26 ** length):
                # Within original capacity - generate normally
                pk_value = self._generate_alphabetic_sequence(seq_val, length)
            else:
                # Beyond capacity - use extended format
                base_val = self._generate_alphabetic_sequence(seq_val % (26 ** min(length or 4, 4)),
                                                              min(length or 4, 4))
                pk_value = f"{base_val}_{seq_val}"

        elif base_type == "AN":
            # For alphanumeric, similar approach
            if length and seq_val <= (36 ** length):
                pk_value = self._generate_alphanumeric_sequence(seq_val, length)
            else:
                base_val = self._generate_alphanumeric_sequence(seq_val % (36 ** min(length or 4, 4)),
                                                                min(length or 4, 4))
                pk_value = f"{base_val}_{seq_val}"

        else:
            # Fallback for other types (DC, D, DT, TS, VA)
            pk_value = f"{table_name}_{column.column_name}_{seq_val}"

        # CRITICAL: Track used values to guarantee uniqueness
        max_attempts = 100
        attempt = 0
        final_value = pk_value

        while attempt < max_attempts:
            if final_value not in self.used_pk_values[pk_key]:
                self.used_pk_values[pk_key].add(final_value)
                self.pk_sequences[pk_key] = max(self.pk_sequences[pk_key], seq_val)
                return final_value

            # If collision, modify the value
            attempt += 1
            if base_type == "N":
                final_value = seq_val + (attempt * num_records)
            elif base_type in ["NS", "A", "AN"]:
                final_value = f"{pk_value}_{attempt}"
            else:
                final_value = f"{pk_value}_DUP{attempt}"

        # Ultimate fallback - should never happen
        final_value = f"PK_{uuid.uuid4().hex[:16]}"
        self.used_pk_values[pk_key].add(final_value)
        return final_value

    def _generate_alphabetic_sequence(self, seq_val: int, length: int) -> str:
        """Generate alphabetic sequence (A, B, ..., Z, AA, AB, ...)"""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        n = seq_val

        while n > 0:
            n -= 1
            result = chars[n % 26] + result
            n //= 26

        # Pad or truncate to desired length
        if len(result) < length:
            result = result.rjust(length, 'A')
        elif len(result) > length:
            # For PK uniqueness, we return the full value even if longer
            pass  # Keep the full value

        return result

    def _generate_alphanumeric_sequence(self, seq_val: int, length: int) -> str:
        """Generate alphanumeric sequence (0-9, A-Z)"""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        n = seq_val

        while n > 0:
            result = chars[n % 36] + result
            n //= 36

        if not result:  # Handle zero case
            result = "0"

        # Pad or truncate to desired length
        if len(result) < length:
            result = result.zfill(length)
        elif len(result) > length:
            # For PK uniqueness, we return the full value even if longer
            pass  # Keep the full value

        return result

    def _validate_and_fix_pk_uniqueness(self, df: pd.DataFrame, table_config: TableConfig) -> pd.DataFrame:
        """Validate and guarantee PRIMARY KEY uniqueness in the generated data.

        Single primary key  → the column itself must be unique.
        Composite primary key → only the *combination* of members must be
        unique; individual members may (and routinely do) repeat — e.g. one
        ``order_id`` shared across many ``line_no`` rows. Members are never
        validated individually, and a foreign-key member is never rewritten.
        """
        pk_columns = [c for c in table_config.columns
                      if c.is_pk and c.column_name in df.columns]
        if not pk_columns:
            return df
        if len(pk_columns) == 1:
            return self._fix_single_pk(df, table_config, pk_columns[0])
        return self._fix_composite_pk(df, table_config, pk_columns)

    def _fix_single_pk(self, df: pd.DataFrame, table_config: TableConfig, pk_col) -> pd.DataFrame:
        """Guarantee a single-column primary key holds unique values."""
        column_name = pk_col.column_name
        duplicate_mask = df.duplicated(subset=[column_name], keep=False)
        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count == 0:
            unique_count = df[column_name].nunique()
            self.logger.info(
                f"✅ PK uniqueness verified for {table_config.name}.{column_name} "
                f"({unique_count}/{len(df)} unique)")
            return df

        self.logger.warning(
            f"⚠️ Found {duplicate_count} duplicate PK values in "
            f"{table_config.name}.{column_name}, fixing...")

        unique_values: Set[Any] = set()
        new_values: List[Any] = []
        for idx, value in enumerate(df[column_name]):
            if value in unique_values or duplicate_mask.iloc[idx]:
                new_value = self._generate_unique_pk_value(
                    pk_col, table_config.name, idx, len(df), existing_values=unique_values)
                new_values.append(new_value)
                unique_values.add(new_value)
            else:
                new_values.append(value)
                unique_values.add(value)
        df[column_name] = new_values

        final_duplicates = int(df.duplicated(subset=[column_name], keep=False).sum())
        if final_duplicates == 0:
            self.logger.info(f"✅ Fixed all PK duplicates for {table_config.name}.{column_name}")
        else:
            self.logger.error(f"❌ Still have {final_duplicates} PK duplicates after fix!")
        return df

    def _fix_composite_pk(self, df: pd.DataFrame, table_config: TableConfig, pk_cols: List) -> pd.DataFrame:
        """Guarantee a composite primary key is unique *as a combination*.

        Only the tuple of members is deduplicated. To preserve referential
        integrity a foreign-key member is never rewritten — a non-FK member is
        perturbed instead (falling back to the last member only if every member
        is itself a foreign key).
        """
        pk_names = [c.column_name for c in pk_cols]
        label = f"{table_config.name}.({','.join(pk_names)})"

        dup_rows = int(df.duplicated(subset=pk_names, keep=False).sum())
        if dup_rows == 0:
            unique_combos = len(df.drop_duplicates(subset=pk_names))
            self.logger.info(
                f"✅ Composite PK uniqueness verified for {label} "
                f"({unique_combos}/{len(df)} unique combinations)")
            return df

        self.logger.warning(
            f"⚠️ Found {dup_rows} row(s) with duplicate composite-PK "
            f"combinations in {label}, fixing...")

        # Perturb a non-FK member so foreign keys are never rewritten.
        non_fk = [c for c in pk_cols if not c.is_fk]
        perturb = (non_fk or pk_cols)[-1]
        p_pos = pk_names.index(perturb.column_name)

        rows = df[pk_names].values.tolist()
        seen: Set[tuple] = set()
        used_member: Set[Any] = set(df[perturb.column_name].tolist())
        for idx, row in enumerate(rows):
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                continue
            # Duplicate combination — regenerate the perturb member until the
            # full tuple is unique.
            for _ in range(2000):
                candidate = self._generate_unique_pk_value(
                    perturb, table_config.name, idx, len(df), existing_values=used_member)
                new_row = list(row)
                new_row[p_pos] = candidate
                new_key = tuple(new_row)
                if new_key not in seen:
                    rows[idx] = new_row
                    seen.add(new_key)
                    used_member.add(candidate)
                    break

        for col_pos, name in enumerate(pk_names):
            df[name] = [r[col_pos] for r in rows]

        final = int(df.duplicated(subset=pk_names, keep=False).sum())
        if final == 0:
            self.logger.info(f"✅ Fixed all composite-PK duplicates for {label}")
        else:
            self.logger.error(f"❌ Still have {final} composite-PK duplicates after fix!")
        return df

    def _generate_unique_pk_value(self, column, table_name: str, index: int, num_records: int,
                                  existing_values: Set[Any]) -> Any:
        """
        Generate a unique PK value that doesn't exist in existing_values
        """
        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)
        pk_key = (table_name, column.column_name)

        max_attempts = 100
        for attempt in range(max_attempts):
            # Use sequential generation but with offset to ensure uniqueness
            seq_val = self.pk_sequences[pk_key] + index + num_records + attempt

            if base_type == "N":
                pk_value = seq_val
            elif base_type == "NS":
                pk_value = str(seq_val)
                if length and len(pk_value) < length:
                    pk_value = pk_value.zfill(length)
            elif base_type == "A":
                pk_value = self._generate_alphabetic_sequence(seq_val, length or 4)
            elif base_type == "AN":
                pk_value = self._generate_alphanumeric_sequence(seq_val, length or 4)
            else:
                pk_value = f"{table_name}_{column.column_name}_{seq_val}_FIXED"

            if pk_value not in existing_values:
                return pk_value

        # Ultimate fallback
        return f"FALLBACK_{uuid.uuid4().hex[:16]}"

    def _calculate_max_unique_values(self, column) -> int:
        """Calculate maximum possible unique values for any data type"""
        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)

        if not length:
            return float('inf')  # No length constraint

        if base_type == "N":
            return (10 ** length) - 1
        elif base_type == "NS":
            return 10 ** length
        elif base_type == "A":
            return 26 ** length
        elif base_type == "AN":
            return 36 ** length
        else:
            return float('inf')
