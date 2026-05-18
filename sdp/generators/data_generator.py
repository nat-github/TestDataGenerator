# -*- coding: utf-8 -*-
"""
data_generator.py - FIXED PK UNIQUENESS FOR ALL DATA TYPES

Key fix: Remove data type length constraints for Primary Keys to guarantee uniqueness
"""

from __future__ import annotations

import logging
import random
import re
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP , localcontext ,InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer

from sdp.models.config_models import TableConfig, RelationshipConfig
from sdp.utils.config_parser import ConfigParser
from sdp.utils.helpers import DataHelpers
from sdp.utils.rule_evaluator import apply_to_dataframe as apply_rules_to_dataframe


class DataGenerator:
    SAFE_DATETIME_MIN = pd.Timestamp("1900-01-01 00:00:00")
    SAFE_DATETIME_MAX = pd.Timestamp("2262-04-11 23:47:16")

    def __init__(self, config_file: str, seed: Optional[int] = None):
        self.config_file = config_file
        self.config_parser = ConfigParser(config_file)
        self.helpers = DataHelpers()
        self.metadata: Optional[Metadata] = None
        self.synthesizer: Optional[HMASynthesizer] = None

        self.generated_data: Dict[str, pd.DataFrame] = {}
        # Anchored generation: real datasets loaded verbatim from `source:` files,
        # keyed by table name. These tables are not generated — other tables
        # generate around them and resolve foreign keys against their real keys.
        self.anchor_data: Dict[str, pd.DataFrame] = {}
        self.tables_config: Dict[str, TableConfig] = {}
        self.relationships: List[RelationshipConfig] = []
        self.sdv_relationship_groups: List[List[RelationshipConfig]] = []
        self._fitted_sample_sizes: Dict[str, int] = {}
        self.is_fitted = False
        self.seed = seed
        self._column_audit: Dict[str, Dict[str, Any]] = {}
        self.logger = self._setup_logging()

        # Enhanced PK tracking - track used values to prevent duplicates
        self.used_pk_values: Dict[Tuple[str, str], Set[Any]] = defaultdict(set)
        self.pk_sequences: Dict[Tuple[str, str], int] = defaultdict(int)
        self._bv_overflow_warned: Set[Tuple[str, str]] = set()  # suppress repeated BV-overflow warnings
        self._pk_lock = threading.RLock()  # guards shared PK state during parallel table generation

        self._apply_seed(seed)

    _SEMANTIC_KEYWORDS: frozenset = frozenset({
        "name", "email", "phone", "address", "city", "country", "zip", "postal",
        "iban", "bban", "bsn", "kvk", "postcode", "street", "gender", "dob",
        "birth", "company", "description", "remark", "note",
    })

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger(__name__)

    def _apply_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        try:
            from faker import Faker as _Faker
            _Faker.seed(seed)
        except Exception:
            pass

    def _apply_table_seed(self, table_name: str) -> None:
        if self.seed is None:
            return
        derived = (self.seed + hash(table_name)) & 0xFFFFFFFF
        import random as _random
        _random.seed(derived)
        np.random.seed(derived)

    # ---------------------------------------------------------------------
    # Configuration & Metadata
    # ---------------------------------------------------------------------
    def load_configuration(self) -> bool:
        try:
            if not self.config_parser.load_config():
                return False
            self.tables_config = self.config_parser.parse_tables()
            self.relationships = self.config_parser.parse_relationships()
            if not self.config_parser.validate_config():
                return False
            self.logger.info("✅ Configuration loaded successfully")

            # Anchored generation: load any `source:` datasets verbatim.
            self._load_anchor_tables()

            # Initialize PK sequences and tracking
            self._initialize_pk_tracking()
            return True
        except Exception as e:
            self.logger.error(f"❌ Error loading configuration: {e}")
            return False

    def _initialize_pk_tracking(self):
        """Initialize primary key tracking for all tables"""
        for table_name, table_config in self.tables_config.items():
            pk_columns = [col for col in table_config.columns if col.is_pk]
            for pk_col in pk_columns:
                self.pk_sequences[(table_name, pk_col.column_name)] = 1
                self.used_pk_values[(table_name, pk_col.column_name)] = set()

    # ---------------------------------------------------------------------
    # Anchored generation — load real datasets declared via `source:`
    # ---------------------------------------------------------------------
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
                except Exception:
                    try:
                        fv = float(s)
                        if not (float('-inf') < fv < float('inf')):
                            dec_vals.append(None)
                            continue
                        d0 = Decimal(str(fv))
                    except Exception:
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
                    except Exception:
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
                d = Decimal(s)  # exact (string-based)  [3](https://www.ibantest.com/en/iban-structure/france)
            except Exception:
                # fallback: float -> str -> Decimal, still reject non-finite
                try:
                    fv = float(s)
                    if not (float('-inf') < fv < float('inf')):
                        dec_vals.append(None);
                        continue
                    d = Decimal(str(fv))
                except Exception:
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

    def create_sdv_metadata(self) -> Metadata:
        self.metadata = Metadata()
        self.sdv_relationship_groups = []
        primary_keys_by_table: Dict[str, str] = {}

        # Add tables
        for table_name in self.tables_config.keys():
            self.metadata.add_table(table_name=table_name)

        # Add columns & PKs
        for table_name, table_config in self.tables_config.items():
            pk = self._resolve_sdv_primary_key(table_config)
            for column in table_config.columns:
                business_values = self.helpers.parse_business_values(column.business_values)
                col_meta = self._enhanced_sdv_type_mapping(column, business_values)

                if column.column_name == pk or column.is_pk:
                    col_meta = {"sdtype": "id"}
                elif column.is_fk:
                    col_meta = {"sdtype": "id"}

                try:
                    self.metadata.add_column(
                        table_name=table_name,
                        column_name=column.column_name,
                        **col_meta,
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not add column {table_name}.{column.column_name}: {e}")
                    self.metadata.add_column(
                        table_name=table_name,
                        column_name=column.column_name,
                        sdtype="categorical"
                    )

            if pk:
                try:
                    self.metadata.set_primary_key(table_name=table_name, column_name=pk)
                    primary_keys_by_table[table_name] = pk
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not set primary key for {table_name}: {e}")

        # Add relationships
        added = 0
        for relationship_group in self._iter_relationship_groups():
            if len(relationship_group) != 1:
                exemplar = relationship_group[0]
                self.logger.info(
                    f"ℹ️ Skipping composite relationship for SDV metadata: {exemplar.source_table} → {exemplar.target_table}"
                )
                continue

            rel = relationship_group[0]
            expected_parent_pk = primary_keys_by_table.get(rel.target_table)
            if expected_parent_pk != rel.target_column:
                self.logger.info(
                    f"ℹ️ Skipping unsupported SDV relationship {rel.source_table}.{rel.source_column} → "
                    f"{rel.target_table}.{rel.target_column}; parent primary key is {expected_parent_pk!r}"
                )
                continue

            try:
                self.metadata.add_relationship(
                    parent_table_name=rel.target_table,
                    parent_primary_key=rel.target_column,
                    child_table_name=rel.source_table,
                    child_foreign_key=rel.source_column,
                )
                self.sdv_relationship_groups.append(relationship_group)
                added += 1
                self.logger.info(
                    f"✅ Added relationship: {rel.source_table}.{rel.source_column} → {rel.target_table}.{rel.target_column}")
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Could not add relationship {rel.source_table}.{rel.source_column} → {rel.target_table}.{rel.target_column}: {e}")

        # Validate metadata
        try:
            self.metadata.validate()
            self.logger.info(f"✅ SDV metadata validated - {added} relationships")
        except Exception as e:
            self.logger.error(f"❌ SDV metadata validation failed: {e}")

        return self.metadata

    def _resolve_sdv_primary_key(self, table_config: TableConfig) -> Optional[str]:
        valid_columns = {column.column_name for column in table_config.columns}
        explicit_primary_keys = [column for column in table_config.primary_key_columns if column in valid_columns]
        derived_primary_keys = [column.column_name for column in table_config.columns if column.is_pk]
        business_keys = [column for column in table_config.business_key_columns if column in valid_columns]

        candidates = explicit_primary_keys or derived_primary_keys or business_keys
        deduplicated_candidates = list(dict.fromkeys(candidates))

        if len(deduplicated_candidates) == 1:
            return deduplicated_candidates[0]

        if len(deduplicated_candidates) > 1:
            self.logger.info(
                f"ℹ️ Table {table_config.name} uses composite keys {deduplicated_candidates}; SDV metadata only supports single-column primary keys, so relationships for this table will be skipped"
            )

        return None

    def _iter_relationship_groups(self) -> List[List[RelationshipConfig]]:
        grouped: Dict[Tuple[str, str, str], List[RelationshipConfig]] = {}

        for relationship in self.relationships:
            base_name = relationship.name or (
                f"{relationship.source_table}.{relationship.source_column}->{relationship.target_table}.{relationship.target_column}"
            )
            if "#" in base_name:
                base_name = base_name.rsplit("#", 1)[0]

            key = (relationship.source_table, relationship.target_table, base_name)
            grouped.setdefault(key, []).append(relationship)

        return list(grouped.values())

    def _enhanced_sdv_type_mapping(self, column, business_values: Optional[List[str]]) -> Dict[str, Any]:
        """Enhanced SDV type mapping"""
        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)
        name_lower = column.column_name.lower()

        mapping = {'sdtype': 'categorical'}

        if business_values:
            mapping['sdtype'] = 'categorical'
        elif base_type in ['D', 'DT', 'TS']:
            mapping['sdtype'] = 'datetime'
        elif column.is_pk or column.is_fk or name_lower == 'id' or name_lower.endswith('_id') or name_lower.endswith('uuid'):
            mapping['sdtype'] = 'id'
        elif base_type in ['N', 'DC']:
            mapping['sdtype'] = 'numerical'
        else:
            mapping['sdtype'] = 'text'

        return mapping

    # ---------------------------------------------------------------------
    # FIXED: Guaranteed Unique Primary Key Generation - NO LENGTH CONSTRAINTS
    # ---------------------------------------------------------------------

    @staticmethod
    def _resolve_generator_max_length(base_type: str, length: Optional[int]) -> Optional[int]:
        if base_type in {"A", "AN", "NS", "VA"}:
            return int(length) if length is not None else None
        return None

    # --- In _generate_unique_primary_key(...), add special_rule handling up-front ---
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
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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

    # ---------------------------------------------------------------------
    # FIXED: Record Count Logic - NO REDUCTION FOR DATA TYPE CAPACITY
    # ---------------------------------------------------------------------
    def _calculate_actual_record_count(self, table_config: TableConfig, requested_records: int) -> int:
        """Calculate actual records - NO REDUCTION for PK data type capacity"""
        pk_columns = [col for col in table_config.columns if col.is_pk]

        # Business-value cap only makes sense for single-column PKs.
        # For composite PKs each component column may repeat values across rows.
        if len(pk_columns) != 1:
            return requested_records

        for pk_col in pk_columns:
            # 1. Business values constraint (still enforced)
            business_values = self.helpers.parse_business_values(pk_col.business_values)
            if business_values:
                max_by_business = len(business_values)
                if requested_records > max_by_business:
                    self.logger.warning(
                        f"🔑 PK {pk_col.column_name} has {max_by_business} business values, "
                        f"reducing from {requested_records} to {max_by_business} records"
                    )
                    return max_by_business

            # 2. DATA TYPE CAPACITY - NO LONGER ENFORCED FOR UNIQUENESS
            base_type, length, precision, scale = self.config_parser.parse_data_type_details(pk_col.data_type)
            max_by_datatype = self._calculate_max_unique_values(pk_col)

            if requested_records > max_by_datatype:
                self.logger.info(
                    f"📏 PK {pk_col.column_name} with {pk_col.data_type} normally supports {max_by_datatype:,} unique values, "
                    f"but generating {requested_records} records by extending beyond data type length"
                )
            # NO RETURN - continue with requested records

        # Always return requested records - we'll make it work!
        return requested_records

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

    # ---------------------------------------------------------------------
    # Enhanced Table Data Generation with PK Uniqueness Validation
    # ---------------------------------------------------------------------
    def _generate_table_data(self, table_config: TableConfig, num_records: int,
                             for_training: bool = False) -> pd.DataFrame:
        """Generate table data with GUARANTEED PK uniqueness.

        Non-PK columns use the vectorised batch path (numpy / Faker list-comp)
        for a significant speed-up on large record counts.  PK columns keep the
        per-row uniqueness-checked path.
        """
        data: Dict[str, List[Any]] = {}
        actual_num_records = self._calculate_actual_record_count(table_config, num_records)

        if for_training:
            with self._pk_lock:
                for col in table_config.columns:
                    if col.is_pk:
                        pk_key = (table_config.name, col.column_name)
                        self.used_pk_values[pk_key] = set()
                        self.pk_sequences[pk_key] = 1

        self.logger.info(
            f"📊 Generating {actual_num_records} records for {table_config.name} "
            f"(requested: {num_records})")

        for column in table_config.columns:
            null_p = self._get_null_probability(column)

            if column.is_pk:
                # Per-row path — uniqueness must be checked row-by-row
                values: List[Any] = []
                for i in range(actual_num_records):
                    val = self._generate_enhanced_value(
                        column, table_config.name, i, actual_num_records)
                    values.append(val)
            else:
                # Batch path — numpy-vectorised where possible
                col_config = {
                    "business_values": column.business_values,
                    "special_rules":   column.special_rules,
                    "data_type":       column.data_type,
                    "min_value":       column.min_value,
                    "max_value":       column.max_value,
                    "column_name":     column.column_name,
                    "max_length":      column.length,
                    "distribution":    getattr(column, "distribution", None),
                }
                values = self.helpers.generate_column_batch(col_config, actual_num_records)

                # Apply null mask in one pass
                if null_p > 0:
                    null_mask = np.random.random(actual_num_records) < null_p
                    values = [None if null_mask[i] else v for i, v in enumerate(values)]

            data[column.column_name] = values

        df = pd.DataFrame(data)
        df = self._validate_and_fix_pk_uniqueness(df, table_config)
        df = self._apply_data_type_constraints(df, table_config)
        return df

    def _validate_and_fix_pk_uniqueness(self, df: pd.DataFrame, table_config: TableConfig) -> pd.DataFrame:
        """
        Validate and guarantee PRIMARY KEY uniqueness in the generated data
        """
        pk_columns = [col for col in table_config.columns if col.is_pk]

        for pk_col in pk_columns:
            if pk_col.column_name in df.columns:
                # Check for duplicates
                duplicate_mask = df.duplicated(subset=[pk_col.column_name], keep=False)
                duplicate_count = duplicate_mask.sum()

                if duplicate_count > 0:
                    self.logger.warning(
                        f"⚠️ Found {duplicate_count} duplicate PK values in {table_config.name}.{pk_col.column_name}, fixing...")

                    # Fix duplicates by generating new unique values
                    unique_values = set()
                    new_values = []

                    for idx, value in enumerate(df[pk_col.column_name]):
                        if value in unique_values or duplicate_mask.iloc[idx]:
                            # Generate new unique value for duplicate
                            new_value = self._generate_unique_pk_value(pk_col, table_config.name, idx, len(df),
                                                                       existing_values=unique_values)
                            new_values.append(new_value)
                            unique_values.add(new_value)
                        else:
                            new_values.append(value)
                            unique_values.add(value)

                    df[pk_col.column_name] = new_values

                    # Verify fix
                    final_duplicates = df.duplicated(subset=[pk_col.column_name], keep=False).sum()
                    if final_duplicates == 0:
                        self.logger.info(f"✅ Fixed all PK duplicates for {table_config.name}.{pk_col.column_name}")
                    else:
                        self.logger.error(f"❌ Still have {final_duplicates} PK duplicates after fix!")

                else:
                    unique_count = df[pk_col.column_name].nunique()
                    self.logger.info(
                        f"✅ PK uniqueness verified for {table_config.name}.{pk_col.column_name} ({unique_count}/{len(df)} unique)")

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

    # ---------------------------------------------------------------------
    # Enhanced Value Generation
    # ---------------------------------------------------------------------
    def _generate_enhanced_value(self, column, table_name: str, index: int = 0, num_records: int = 0) -> Any:
        """Generate value with guaranteed PK uniqueness and business values logic"""
        if column.is_pk:
            return self._generate_unique_primary_key(column, table_name, index, num_records)

        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)
        max_length = self._resolve_generator_max_length(base_type, length)

        # Business values first (for non-PK columns)
        business_values = self.helpers.parse_business_values(column.business_values)
        business_values = self._sanitize_business_values(business_values, base_type)
        if business_values:
            return random.choice(business_values)

        # Special rules
        if column.special_rules and not pd.isna(column.special_rules):
            special_value = self.helpers.generate_special_value(
                column.special_rules,
                column.data_type,
                column_name=column.column_name,
                max_length=max_length,
            )
            if special_value is not None:
                return special_value

        if base_type in {"VA", "A", "AN", "NS", "T"}:
            semantic_value = self.helpers.generate_realistic_dutch_data(
                column.column_name,
                column.data_type,
                max_length=length,
                min_val=column.min_value,
                max_val=column.max_value,
            )
            if semantic_value is not None:
                return semantic_value

        # Type-based generation (for non-PK columns, respect length constraints)
        if base_type == "DC" and precision and scale:
            return self._generate_decimal_value(precision, scale, column.min_value, column.max_value)
        elif base_type == "NS":
            actual_len = length if length else 15
            return self._generate_numeric_string(actual_len)
        elif base_type == "AN":
            actual_len = length if length else 20
            return self._generate_alphanumeric_string(actual_len)
        elif base_type == "N":
            actual_len = length if length else 10
            return self._generate_numeric_value(actual_len, column.min_value, column.max_value)
        elif base_type == "A":
            actual_len = length if length else 10
            return self._generate_alphabetic_string(actual_len)
        elif base_type == "VA":
            actual_len = length if length else 50
            return self._generate_variable_string(actual_len)
        elif base_type in ["D", "DT", "TS"]:
            return self.helpers.generate_sample_value(base_type, {
                "business_values": business_values,
                "special_rules": column.special_rules,
                "column_name": column.column_name,
                "max_length": max_length,
                "min_value": column.min_value,
                "max_value": column.max_value,
            })
        else:
            return self.helpers.generate_realistic_dutch_data(column.column_name, base_type)

    # ---------------------------------------------------------------------
    # Core Generation Methods
    # ---------------------------------------------------------------------
    def _generate_decimal_value(self, precision: int, scale: int, min_val: Optional[float],
                                max_val: Optional[float]) -> float:
        if min_val is None or pd.isna(min_val):
            min_val = Decimal("0.0")
        else:
            min_val = Decimal(str(min_val))

        if max_val is None or pd.isna(max_val):
            max_integer = 10 ** (precision - scale) - 1
            max_val = Decimal(str(max_integer)) + (Decimal("1") - (Decimal("10") ** -scale))
        else:
            max_val = Decimal(str(max_val))

        rnd = random.uniform(float(min_val), float(max_val))
        dec = Decimal(str(rnd))
        return float(dec.quantize(Decimal("1." + "0" * int(scale)), rounding=ROUND_HALF_UP))

    def _generate_numeric_string(self, length: int) -> str:
        if length <= 0:
            return ""
        if length == 1:
            return str(random.randint(0, 9))
        first = str(random.randint(1, 9))
        rest = "".join(str(random.randint(0, 9)) for _ in range(length - 1))
        return first + rest

    def _generate_alphanumeric_string(self, length: int) -> str:
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(random.choices(chars, k=max(0, length)))

    def _generate_alphabetic_string(self, length: int) -> str:
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(random.choices(chars, k=max(0, length)))

    def _generate_numeric_value(self, length: int, min_val: Optional[float], max_val: Optional[float]) -> int:
        if min_val is None or pd.isna(min_val):
            min_val = 10 ** (max(1, length) - 1)
        if max_val is None or pd.isna(max_val):
            max_val = (10 ** max(1, length)) - 1
        return random.randint(int(min_val), int(max_val))

    def _generate_variable_string(self, length: int) -> str:
        if length <= 0:
            return ""
        if length <= 12:
            return self.helpers.faker.word()[:length]
        if length <= 50:
            target = random.randint(6, length)
            return self.helpers.faker.sentence(nb_words=random.randint(2, 6))[:target].strip()

        target = random.randint(20, min(length, 140))
        return self.helpers.faker.text(max_nb_chars=target).strip()

    # ---------------------------------------------------------------------
    # NULL Generation Helpers
    # ---------------------------------------------------------------------
    def _parse_null_rate_from_rules(self, rules: Optional[str]) -> Optional[float]:
        if not rules or pd.isna(rules):
            return None
        try:
            return self.helpers.get_special_rule_null_probability(rules)
        except ValueError:
            return None

    def _get_null_probability(self, column) -> float:
        explicit = getattr(column, "null_rate", None)
        if explicit is not None and not pd.isna(explicit):
            try:
                return max(0.0, min(1.0, float(explicit)))
            except Exception:
                pass
        parsed = self._parse_null_rate_from_rules(getattr(column, "special_rules", None))
        if parsed is not None:
            return parsed
        return 0.0

    def _column_has_value_generation_rule(self, column) -> bool:
        business_values = self.helpers.parse_business_values(getattr(column, "business_values", None))
        if business_values:
            return True

        try:
            value_rule = self.helpers.get_special_rule_value_directive(getattr(column, "special_rules", None))
        except ValueError:
            return False

        return bool(value_rule)

    def _is_semantic_text_column(self, column) -> bool:
        """True for text columns where semantic name/company/address generation overrides SDV junk."""
        base_type, _, _, _ = self.config_parser.parse_data_type_details(column.data_type)
        if base_type not in {"VA", "A", "AN"}:
            return False
        col = column.column_name.lower()
        return (
            self.helpers._looks_like_name_field(col)
            or self.helpers._looks_like_company_field(col)
            or any(k in col for k in [
                "iban", "bban", "bic", "swift", "phone", "email",
                "address", "adres", "ctry", "cty_code", "country",
                "currency", "curr", "ccy",
            ])
        )

    def _reconcile_sdv_constraints(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Reapply workbook-driven rules after SDV sampling so constrained columns remain valid."""
        reconciled: Dict[str, pd.DataFrame] = {}
        self._initialize_pk_tracking()

        reconciled_columns = 0
        for table_name, df in data.items():
            table_config = self.tables_config.get(table_name)
            if table_config is None:
                reconciled[table_name] = df.copy()
                continue

            table_df = df.copy().reset_index(drop=True)
            row_count = len(table_df)

            for column in table_config.columns:
                column_name = column.column_name
                if column_name not in table_df.columns:
                    continue
                if column.is_fk:
                    continue

                # SDV emits opaque string ids ("sdv-id-XXXX") for primary keys,
                # which ignore the column's declared data type and get wiped to
                # NaN when the typed cast runs on export. Regenerate every PK as a
                # clean, type-correct, unique sequence so the declared type holds.
                if column.is_pk:
                    table_df[column_name] = [
                        self._generate_unique_primary_key(column, table_name, index, row_count)
                        for index in range(row_count)
                    ]
                    reconciled_columns += 1
                    continue

                has_value_rule = self._column_has_value_generation_rule(column)
                is_semantic = not has_value_rule and self._is_semantic_text_column(column)
                null_probability = 0.0 if column.is_pk else self._get_null_probability(column)

                if not has_value_rule and not is_semantic and null_probability <= 0.0:
                    continue

                existing_values = table_df[column_name].tolist()
                regenerated_values: List[Any] = []
                changed = False

                for index in range(row_count):
                    if null_probability > 0.0 and random.random() < null_probability:
                        regenerated_values.append(None)
                        if existing_values[index] is not None:
                            changed = True
                        continue

                    if has_value_rule or is_semantic:
                        regenerated_value = self._generate_enhanced_value(column, table_name, index, row_count)
                        regenerated_values.append(regenerated_value)
                        if regenerated_value != existing_values[index]:
                            changed = True
                    else:
                        regenerated_values.append(existing_values[index])

                if changed:
                    table_df[column_name] = regenerated_values
                    reconciled_columns += 1

            table_df = self._validate_and_fix_pk_uniqueness(table_df, table_config)
            table_df = self._apply_data_type_constraints(table_df, table_config)
            reconciled[table_name] = table_df

        if reconciled_columns:
            self.logger.info(f"🩹 Reconciled {reconciled_columns} constrained columns after SDV sampling")

        return reconciled

    # ---------------------------------------------------------------------
    # Data Type Constraints & Sanitization
    # ---------------------------------------------------------------------
    def _apply_data_type_constraints(self, df: pd.DataFrame, table_config: TableConfig) -> pd.DataFrame:
        for column in table_config.columns:
            if column.column_name not in df.columns:
                continue

            base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)

            # For PK columns, skip length constraints (already handled in generation)
            if column.is_pk:
                continue

            # Apply constraints only for non-PK columns
            if base_type == "DC" and (scale is not None):
                s = int(scale)
                numeric_series = pd.to_numeric(df[column.column_name], errors="coerce")
                df[column.column_name] = numeric_series.round(s)

            elif base_type == "NS" and length:
                L = int(length)
                df[column.column_name] = df[column.column_name].astype("string")
                df[column.column_name] = df[column.column_name].str.zfill(L)

            elif base_type == "AN" and length:
                L = int(length)
                df[column.column_name] = df[column.column_name].astype("string").str.rstrip().str[:L]

            elif base_type == "A" and length:
                L = int(length)
                df[column.column_name] = df[column.column_name].astype("string").str.rstrip().str[:L]

            elif base_type == "N" and length:
                max_val = 10 ** int(length) - 1

                numeric_series = pd.to_numeric(df[column.column_name], errors="coerce")
                df[column.column_name] = numeric_series.apply(
                    lambda value: min(int(value), max_val) if pd.notna(value) else value
                )

        return df

    def _sanitize_business_values(self, values: Optional[List[Any]], data_type: str) -> List[Any]:
        if not values:
            return []
        if data_type not in ["D", "DT", "TS"]:
            return values

        cleaned: List[str] = []
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                cleaned.append(s)

        coerced: List[Any] = []
        for s in cleaned:
            ts = self._coerce_safe_timestamp(s, normalize=(data_type == "D"))
            if ts is not None:
                coerced.append(ts)
        return coerced

    def _coerce_safe_timestamp(self, value: Any, normalize: bool = False) -> Optional[pd.Timestamp]:
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return None

        if pd.isna(parsed):
            return None

        ts = pd.Timestamp(parsed)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)

        if ts < self.SAFE_DATETIME_MIN or ts > self.SAFE_DATETIME_MAX:
            return None

        return ts.normalize() if normalize else ts

    def _coerce_identifier_series(
        self,
        series: pd.Series,
        table_name: str,
        column_name: str,
        enforce_unique: bool = False,
    ) -> pd.Series:
        coerced = pd.Series(pd.NA, index=series.index, dtype="string")
        seen_values: Dict[str, int] = {}

        non_null_mask = series.notna()
        if non_null_mask.any():
            for row_index, raw_value in series.loc[non_null_mask].items():
                text_value = str(raw_value)
                if enforce_unique:
                    duplicate_count = seen_values.get(text_value, 0)
                    seen_values[text_value] = duplicate_count + 1
                    if duplicate_count:
                        text_value = f"{text_value}__{duplicate_count + 1}"
                coerced.at[row_index] = text_value

        missing_mask = coerced.isna()
        if missing_mask.any():
            for position, row_index in enumerate(coerced.index[missing_mask], start=1):
                coerced.at[row_index] = f"{table_name}_{column_name}_{position}"

        return coerced

    def _coerce_datetime_series(self, series: pd.Series, date_only: bool = False, aggressive: bool = False) -> pd.Series:
        coerced = pd.to_datetime(series, errors="coerce")

        if getattr(coerced.dt, "tz", None) is not None:
            coerced = coerced.dt.tz_convert(None)

        lower_bound = self.SAFE_DATETIME_MIN.normalize() if date_only else self.SAFE_DATETIME_MIN
        upper_bound = self.SAFE_DATETIME_MAX.normalize() if date_only else self.SAFE_DATETIME_MAX
        coerced = coerced.clip(lower=lower_bound, upper=upper_bound)

        if date_only:
            coerced = coerced.dt.normalize()

        if aggressive and coerced.isna().any():
            fallback = pd.Timestamp("2024-01-01 00:00:00")
            if date_only:
                fallback = fallback.normalize()
            coerced = coerced.fillna(fallback)

        return coerced

    def _sanitize_sample_data_for_sdv(
        self,
        sample_data: Dict[str, pd.DataFrame],
        aggressive: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        sanitized: Dict[str, pd.DataFrame] = {}

        for table_name, df in sample_data.items():
            table_config = self.tables_config.get(table_name)
            if table_config is None:
                sanitized[table_name] = df.copy()
                continue

            clean_df = df.copy()
            sdv_primary_key = self._resolve_sdv_primary_key(table_config)
            for column in table_config.columns:
                column_name = column.column_name
                if column_name not in clean_df.columns:
                    continue

                base_type, _, _, _ = self.config_parser.parse_data_type_details(column.data_type)
                series = clean_df[column_name]

                if base_type in ["D", "DT", "TS"]:
                    clean_df[column_name] = self._coerce_datetime_series(
                        series,
                        date_only=(base_type == "D"),
                        aggressive=aggressive,
                    )
                elif column_name == sdv_primary_key or column.is_pk or column.is_fk:
                    clean_df[column_name] = self._coerce_identifier_series(
                        series,
                        table_name,
                        column_name,
                        enforce_unique=(column_name == sdv_primary_key),
                    )
                elif base_type in ["N", "DC"]:
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    if aggressive and numeric_series.isna().any():
                        fill_value = numeric_series.dropna().median() if not numeric_series.dropna().empty else 0
                        numeric_series = numeric_series.fillna(fill_value)
                    clean_df[column_name] = numeric_series
                else:
                    string_series = pd.Series(pd.NA, index=series.index, dtype="string")
                    non_null_mask = series.notna()
                    if non_null_mask.any():
                        string_series.loc[non_null_mask] = series.loc[non_null_mask].astype("string")
                    if aggressive and string_series.isna().any():
                        missing_mask = string_series.isna()
                        for position, row_index in enumerate(string_series.index[missing_mask], start=1):
                            string_series.at[row_index] = f"{table_name}_{column_name}_{position}"
                    clean_df[column_name] = string_series

            sanitized[table_name] = clean_df

        return self._enforce_relationships_in_sample(sanitized)

    # ---------------------------------------------------------------------
    # SDV Training & Data Generation
    # ---------------------------------------------------------------------
    def train_synthesizer(self, sample_size: int = 200) -> bool:
        try:
            if self.metadata is None:
                self.logger.error("❌ Metadata not created. Call create_sdv_metadata() first.")
                return False

            configured_sample_size = self.config_parser.get_setting('synthesizer_sample_size', sample_size)
            try:
                sample_size = int(configured_sample_size)
            except (TypeError, ValueError):
                sample_size = sample_size

            self.logger.info("🧑‍🤖 Initializing HMA Synthesizer...")
            self._apply_seed(self.seed)

            self.synthesizer = HMASynthesizer(
                metadata=self.metadata,
                verbose=True,
                locales=['nl_NL']
            )

            # Generate high-quality sample data
            sample_sizes = {t: min(sample_size, 100) for t in self.tables_config.keys()}
            sample_data = self._sanitize_sample_data_for_sdv(
                self._generate_high_quality_sample_data(sample_sizes)
            )

            self.logger.info("🛠️ Fitting synthesizer with enhanced sample data...")
            fitted_sample_data = sample_data

            try:
                self.synthesizer.fit(sample_data)
            except Exception as first_error:
                self.logger.warning(f"⚠️ Initial synthesizer fit attempt failed: {first_error}")
                retry_sample_sizes = {t: max(25, min(sample_size, 75)) for t in self.tables_config.keys()}
                retry_sample_data = self._sanitize_sample_data_for_sdv(
                    self._generate_high_quality_sample_data(retry_sample_sizes),
                    aggressive=True,
                )
                self.synthesizer = HMASynthesizer(
                    metadata=self.metadata,
                    verbose=True,
                    locales=['nl_NL']
                )
                self.logger.info("🔁 Retrying synthesizer fit with aggressively sanitized sample data...")
                self.synthesizer.fit(retry_sample_data)
                fitted_sample_data = retry_sample_data

            self._fitted_sample_sizes = {
                table_name: len(df)
                for table_name, df in fitted_sample_data.items()
            }
            self.is_fitted = True

            self.logger.info("✅ SDV synthesizer trained and fitted successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error training synthesizer: {e}")
            self.logger.info("🔄 Continuing with enhanced fallback generation...")
            self.is_fitted = False
            return False

    def save_model_artifacts(self, output_dir: Optional[str] = None) -> Optional[Path]:
        """Save fitted synthesizer artifacts when enabled in workbook settings."""
        if not self.is_fitted or self.synthesizer is None or self.metadata is None:
            return None

        save_enabled = self.config_parser.get_setting('save_model_artifact', False)
        if not bool(save_enabled):
            return None

        target_dir = output_dir or self.config_parser.get_setting('model_artifact_path', 'output/models')
        artifact_dir = Path(target_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = artifact_dir / 'metadata.json'
        model_file = artifact_dir / 'hma_synthesizer.pkl'

        try:
            self.metadata.save_to_json(filepath=str(metadata_file))
        except Exception as exc:
            self.logger.warning(f"⚠️ Could not save metadata artifact: {exc}")

        try:
            self.synthesizer.save(filepath=str(model_file))
            self.logger.info(f"💾 Saved synthesizer artifact to {model_file}")
            return artifact_dir
        except Exception as exc:
            self.logger.warning(f"⚠️ Could not save synthesizer artifact: {exc}")
            return artifact_dir if metadata_file.exists() else None

    # Real-data rows fed into SDV training for an anchor table are capped at
    # this size to keep HMASynthesizer.fit fast while still learning distributions.
    _ANCHOR_TRAIN_CAP = 500

    def _generate_high_quality_sample_data(self, sample_sizes: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Generate high-quality sample data for SDV training.

        Anchor tables (declared via ``source:``) contribute their *real* rows so
        the synthesizer learns the actual distributions, not synthetic ones.
        """
        sample_data = {}

        # Generate all tables first
        for table_name, num_records in sample_sizes.items():
            if table_name not in self.tables_config:
                continue

            if self._is_anchor_table(table_name):
                # Train SDV on the real anchor rows (capped for fit speed).
                sample_data[table_name] = (
                    self.anchor_data[table_name].head(self._ANCHOR_TRAIN_CAP).reset_index(drop=True)
                )
                continue

            table_config = self.tables_config[table_name]
            data = self._generate_table_data(table_config, num_records, for_training=True)
            sample_data[table_name] = data

        # Enforce relationships in sample data
        return self._enforce_relationships_in_sample(sample_data)

    def _is_one_to_one_fk(
        self,
        exemplar: RelationshipConfig,
        child_table: str,
        source_columns: List[str],
    ) -> bool:
        """True when the FK must hold *unique* values — a one-to-one relationship.

        That happens when the child FK column(s) are exactly the child table's
        primary key (so the FK value cannot repeat), or when the relationship
        explicitly declares ``relationship_type`` one_to_one.
        """
        declared = str(getattr(exemplar, "relationship_type", "") or "").strip().lower()
        if declared in ("one_to_one", "one-to-one", "1:1"):
            return True
        cfg = self.tables_config.get(child_table)
        if cfg is None:
            return False
        pk_cols = {c.column_name for c in cfg.columns if getattr(c, "is_pk", False)}
        # One-to-one only when the FK columns *are* the whole primary key.
        return bool(pk_cols) and pk_cols == set(source_columns)

    def _sample_fk_values(
        self,
        parent_values: List[Any],
        n: int,
        one_to_one: bool,
        label: str,
    ) -> List[Any]:
        """Pick ``n`` FK values from the parent key pool.

        ``one_to_one`` → sample *without* replacement, so each child row gets a
        distinct parent key and the child primary key stays unique. Otherwise
        sample *with* replacement (one parent row, many child rows).
        """
        if not one_to_one:
            return random.choices(parent_values, k=n)
        distinct = list(dict.fromkeys(parent_values))   # unique, order-preserving
        if len(distinct) >= n:
            return random.sample(distinct, k=n)
        # Not enough distinct parent keys for a true 1:1 — a config-size mismatch.
        # Use every distinct key, top up with repeats, and warn loudly.
        self.logger.warning(
            f"⚠️ One-to-one FK {label}: parent has only {len(distinct)} distinct "
            f"key(s) for {n} child rows — {n - len(distinct)} row(s) cannot be unique."
        )
        pool = distinct + random.choices(distinct, k=n - len(distinct))
        random.shuffle(pool)
        return pool

    def _apply_relationship_group(
        self,
        data: Dict[str, pd.DataFrame],
        relationship_group: List[RelationshipConfig],
    ) -> None:
        exemplar = relationship_group[0]
        parent_table = exemplar.target_table
        child_table = exemplar.source_table

        if parent_table not in data or child_table not in data:
            return

        # Anchor tables hold real data — never rewrite their foreign keys.
        if self._is_anchor_table(child_table):
            return

        parent_df = data[parent_table]
        child_df = data[child_table]
        source_columns = [relationship.source_column for relationship in relationship_group]
        target_columns = [relationship.target_column for relationship in relationship_group]

        if any(column not in parent_df.columns for column in target_columns):
            return
        if any(column not in child_df.columns for column in source_columns):
            return

        one_to_one = self._is_one_to_one_fk(exemplar, child_table, source_columns)

        if len(relationship_group) == 1:
            valid_parent_values = parent_df[target_columns[0]].dropna().tolist()
            if valid_parent_values:
                child_df[source_columns[0]] = self._sample_fk_values(
                    valid_parent_values, len(child_df), one_to_one,
                    f"{child_table}.{source_columns[0]}",
                )
                data[child_table] = child_df
            return

        parent_pairs = parent_df[target_columns].dropna().drop_duplicates()
        if parent_pairs.empty:
            return

        # One-to-one composite FK: each child row needs a distinct parent
        # key-combination — sample without replacement when there are enough.
        replace = True
        if one_to_one and len(parent_pairs) >= len(child_df):
            replace = False
        elif one_to_one:
            self.logger.warning(
                f"⚠️ One-to-one FK {child_table}.{tuple(source_columns)}: parent has "
                f"only {len(parent_pairs)} distinct key combinations for "
                f"{len(child_df)} child rows — uniqueness cannot be fully honoured."
            )
        sampled_parent_rows = parent_pairs.sample(n=len(child_df), replace=replace).reset_index(drop=True)
        child_df = child_df.copy()
        for source_column, target_column in zip(source_columns, target_columns):
            child_df[source_column] = sampled_parent_rows[target_column].tolist()
        data[child_table] = child_df

    def _enforce_relationships_in_sample(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enforce relationships in sample data for better SDV learning"""
        for relationship_group in self.sdv_relationship_groups:
            self._apply_relationship_group(sample_data, relationship_group)

        return sample_data

    def _sample_from_synthesizer(self, records_per_table: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        if self.synthesizer is None:
            raise ValueError("Synthesizer is not initialized")

        try:
            sampled = self.synthesizer.sample(num_rows=records_per_table)
        except TypeError as exc:
            if "num_rows" not in str(exc):
                raise

            requested_ratios = []
            for table_name, requested_count in records_per_table.items():
                fitted_count = self._fitted_sample_sizes.get(table_name)
                if fitted_count:
                    requested_ratios.append(requested_count / max(1, fitted_count))

            scale = max(requested_ratios) if requested_ratios else 1.0
            scale = max(scale, 1e-6)
            sampled = self.synthesizer.sample(scale=scale)

        if not isinstance(sampled, dict):
            raise ValueError("SDV synthesizer returned a non-dictionary result")

        trimmed: Dict[str, pd.DataFrame] = {}
        for table_name, requested_count in records_per_table.items():
            table_df = sampled.get(table_name)
            if table_df is None or table_df.empty:
                raise ValueError(f"SDV returned no rows for table {table_name}")
            if len(table_df) < requested_count:
                raise ValueError(
                    f"SDV returned only {len(table_df)} rows for table {table_name}; expected at least {requested_count}"
                )
            trimmed[table_name] = table_df.head(requested_count).reset_index(drop=True)

        return trimmed

    def generate_data(self, records_per_table: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """
        Enhanced data generation with guaranteed relationship integrity
        """
        try:
            # Try SDV generation first if fitted
            if self.is_fitted and self.synthesizer:
                self.logger.info("🎲 Generating data using trained SDV synthesizer...")
                try:
                    # Anchor tables are replaced verbatim afterwards — ask SDV
                    # for just one throwaway row so it never blocks generation.
                    sdv_counts = {
                        t: (1 if self._is_anchor_table(t) else n)
                        for t, n in records_per_table.items()
                    }
                    synthetic_data = self._sample_from_synthesizer(sdv_counts)
                    synthetic_data = self._reconcile_sdv_constraints(synthetic_data)
                    synthetic_data = self._inject_anchor_tables(synthetic_data)

                    # Validate and enforce relationships in SDV data
                    if self._validate_sdv_data(synthetic_data):
                        synthetic_data = self._enforce_all_relationships(synthetic_data)
                        self.generated_data = synthetic_data
                        self._resolve_foreign_keys()  # Final FK resolution
                        self._apply_rules_and_derived()  # Layer A + Layer B
                        self._build_column_audit(sdv_used=True)
                        total_records = sum(len(df) for df in self.generated_data.values())
                        self.logger.info(f"✅ SDV data generation completed: {total_records} total records")
                        return self.generated_data
                    else:
                        self.logger.warning("⚠️ SDV data validation failed, using enhanced fallback...")

                except Exception as sdv_error:
                    self.logger.warning(f"⚠️ SDV generation failed: {sdv_error}, using enhanced fallback...")

            # Enhanced fallback generation with relationship guarantees
            self.logger.info("🎲 Generating data using enhanced fallback method...")
            synthetic_data = self._generate_fallback_data_with_relationships(records_per_table)
            self.generated_data = synthetic_data

            total_records = sum(len(df) for df in self.generated_data.values())
            if total_records == 0:
                self.logger.error("❌ Generated data is empty!")
                raise ValueError("No data was generated")

            self._apply_rules_and_derived()  # Layer A + Layer B
            self._build_column_audit(sdv_used=False)
            self.logger.info(f"✅ Fallback data generation completed: {total_records} total records")
            return self.generated_data

        except Exception as e:
            self.logger.error(f"❌ Error generating data: {e}")
            raise

    def _generate_fallback_data_with_relationships(self, records_per_table: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Enhanced fallback data generation with relationship guarantees.

        Reference (parent) tables that have no FK dependencies are generated in
        parallel using a thread pool.  Child tables are generated sequentially
        afterwards so FK resolution can access parent data.
        """
        synthetic_data: Dict[str, pd.DataFrame] = {}
        self._initialize_pk_tracking()

        # Anchor tables are loaded verbatim — never generated.
        gen_counts = {
            t: n for t, n in records_per_table.items() if not self._is_anchor_table(t)
        }

        reference_tables = self._identify_reference_tables()
        ref_in_config = [t for t in reference_tables if t in gen_counts]

        # Parallel generation for independent reference tables
        if len(ref_in_config) > 1:
            parallel_results = self._generate_tables_parallel(ref_in_config, gen_counts)
            synthetic_data.update(parallel_results)
        elif ref_in_config:
            t = ref_in_config[0]
            synthetic_data[t] = self._generate_table_data(
                self.tables_config[t], gen_counts[t], for_training=False)

        # Sequential generation for child tables (depend on parent data)
        for table_name, num_records in gen_counts.items():
            if table_name not in reference_tables:
                synthetic_data[table_name] = self._generate_table_data(
                    self.tables_config[table_name], num_records, for_training=False)

        # Inject real anchor data before FK resolution so children reference it.
        synthetic_data = self._inject_anchor_tables(synthetic_data)
        synthetic_data = self._enforce_all_relationships(synthetic_data)
        return synthetic_data

    def _generate_tables_parallel(
        self,
        table_names: List[str],
        records_per_table: Dict[str, int],
    ) -> Dict[str, pd.DataFrame]:
        """Generate independent tables concurrently (up to 4 workers)."""
        results: Dict[str, pd.DataFrame] = {}
        max_workers = min(len(table_names), 4)

        def _gen(name: str):
            return name, self._generate_table_data(
                self.tables_config[name], records_per_table[name], for_training=False)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_gen, t): t for t in table_names}
            for future in as_completed(futures):
                name, df = future.result()
                results[name] = df

        return results

    def _identify_reference_tables(self) -> List[str]:
        """Identify parent/reference tables"""
        referenced_tables = set(rel.target_table for rel in self.relationships)
        referencing_tables = set(rel.source_table for rel in self.relationships)

        reference_tables = list(referenced_tables - referencing_tables)

        for table_name, table_config in self.tables_config.items():
            if not any(col.is_fk for col in table_config.columns):
                if table_name not in reference_tables:
                    reference_tables.append(table_name)

        self.logger.info(f"📋 Identified reference tables: {reference_tables}")
        return reference_tables

    def _enforce_all_relationships(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enforce ALL relationships in data"""
        for _ in range(3):
            for relationship_group in self._iter_relationship_groups():
                self._apply_relationship_group(data, relationship_group)

        return data

    def _validate_sdv_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate SDV generated data quality"""
        if not data:
            return False

        for table_name, df in data.items():
            if df.empty:
                self.logger.warning(f"⚠️ SDV generated empty table: {table_name}")
                return False

        return True

    # ---------------------------------------------------------------------
    # Layer A (rules) + Layer B (derived) post-generation pass
    # ---------------------------------------------------------------------
    def _apply_rules_and_derived(self) -> None:
        """Apply when/then rules and resolve derived columns for every table.

        No-op for tables with no rules and no derived columns. Runs after FK
        resolution so cross-row references are stable.
        """
        if not self.generated_data:
            return
        any_applied = False
        for table_name, df in self.generated_data.items():
            tc = self.tables_config.get(table_name)
            if tc is None:
                continue
            has_rules = any(c.rules for c in tc.columns)
            has_derived = any(c.derived for c in tc.columns)
            if not has_rules and not has_derived:
                continue
            self.generated_data[table_name] = apply_rules_to_dataframe(
                df, tc, helpers=self.helpers
            )
            any_applied = True
        if any_applied:
            self.logger.info("🧮 Applied row-level rules and derived columns")

    # ---------------------------------------------------------------------
    # Foreign Key Resolution
    # ---------------------------------------------------------------------
    def _resolve_foreign_keys(self) -> None:
        """Resolve foreign key relationships in generated_data"""
        if not self.relationships:
            return

        self.logger.info("🔗 Resolving foreign key relationships...")
        resolved_count = 0

        for relationship_group in self._iter_relationship_groups():
            exemplar = relationship_group[0]
            src_t = exemplar.source_table
            tgt_t = exemplar.target_table

            if src_t not in self.generated_data or tgt_t not in self.generated_data:
                continue

            src_df = self.generated_data[src_t]
            original_dtypes = {
                relationship.source_column: src_df[relationship.source_column].dtype
                for relationship in relationship_group
                if relationship.source_column in src_df.columns
            }
            before_frame = src_df[[column for column in original_dtypes]].copy() if original_dtypes else pd.DataFrame()

            self._apply_relationship_group(self.generated_data, relationship_group)
            src_df = self.generated_data[src_t]

            for source_column, original_dtype in original_dtypes.items():
                try:
                    src_df[source_column] = src_df[source_column].astype(original_dtype)
                except (ValueError, TypeError):
                    pass

            if not before_frame.empty and not before_frame.equals(src_df[list(original_dtypes)]):
                resolved_count += 1
                source_columns = ", ".join(rel.source_column for rel in relationship_group)
                target_columns = ", ".join(rel.target_column for rel in relationship_group)
                self.logger.info(f" ✅ Resolved FK: {src_t}.{source_columns} → {tgt_t}.{target_columns}")

        self.logger.info(f"✅ Resolved {resolved_count} foreign key relationships")

    # ---------------------------------------------------------------------
    # Export Methods
    # ---------------------------------------------------------------------
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

                        except Exception:

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

    def _to_iso_datetime_strings(self, series: pd.Series) -> List[Optional[str]]:
        out: List[Optional[str]] = []
        for v in series:
            if pd.isna(v):
                out.append(None)
                continue
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                ts = pd.Timestamp(v)
                out.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
                continue
            if isinstance(v, str):
                s = v.strip()
                out.append(s if s else None)
                continue
            ts = pd.to_datetime(v, errors="coerce")
            if pd.isna(ts):
                out.append(str(v))
            else:
                out.append(pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"))
        return out

    # ---------------------------------------------------------------------
    # Column Audit
    # ---------------------------------------------------------------------
    def _build_column_audit(self, sdv_used: bool) -> None:
        self._column_audit = {}
        for table_name, table_config in self.tables_config.items():
            table_audit: Dict[str, Any] = {}
            for column in table_config.columns:
                if sdv_used and not column.is_pk and not column.is_fk:
                    path = "sdv"
                elif column.is_pk:
                    path = "pk-sequence"
                elif column.is_fk:
                    path = "fk-resolved"
                elif self.helpers.parse_business_values(column.business_values):
                    path = "enum"
                elif column.special_rules and not pd.isna(column.special_rules):
                    col_lower = column.column_name.lower()
                    if any(kw in col_lower for kw in self._SEMANTIC_KEYWORDS):
                        path = "semantic"
                    else:
                        path = "regex"
                elif column.special_rules and not pd.isna(column.special_rules):
                    path = "faker"
                else:
                    col_lower = column.column_name.lower()
                    if any(kw in col_lower for kw in self._SEMANTIC_KEYWORDS):
                        path = "semantic"
                    else:
                        path = "range"
                table_audit[column.column_name] = path
            self._column_audit[table_name] = table_audit

    # ---------------------------------------------------------------------
    # Report & Debug
    # ---------------------------------------------------------------------
    def get_generation_report(self) -> Dict[str, Any]:
        if not self.generated_data:
            return {
                "tables_generated": [],
                "total_records": 0,
                "relationships_configured": len(self.relationships),
                "table_record_counts": {},
                "synthesizer_fitted": self.is_fitted,
                "seed": self.seed,
                "column_audit": self._column_audit,
                "generation_path_summary": {},
                "status": "NO_DATA_GENERATED",
            }

        counts: Dict[str, int] = {t: (0 if df.empty else len(df)) for t, df in self.generated_data.items()}
        total = sum(counts.values())

        path_summary: Dict[str, int] = {}
        for table_audit in self._column_audit.values():
            for path in table_audit.values():
                path_summary[path] = path_summary.get(path, 0) + 1

        return {
            "tables_generated": list(self.generated_data.keys()),
            "total_records": total,
            "relationships_configured": len(self.relationships),
            "table_record_counts": counts,
            "synthesizer_fitted": self.is_fitted,
            "seed": self.seed,
            "column_audit": self._column_audit,
            "generation_path_summary": path_summary,
            "status": "SUCCESS" if total > 0 else "FAILED",
        }
