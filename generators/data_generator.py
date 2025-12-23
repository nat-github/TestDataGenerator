# -*- coding: utf-8 -*-
"""
data_generator.py - FIXED PK UNIQUENESS FOR ALL DATA TYPES

Key fix: Remove data type length constraints for Primary Keys to guarantee uniqueness
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer

from models.config_models import TableConfig, RelationshipConfig
from utils.config_parser import ConfigParser
from utils.helpers import DataHelpers


class DataGenerator:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_parser = ConfigParser(config_file)
        self.helpers = DataHelpers()
        self.metadata: Optional[MultiTableMetadata] = None
        self.synthesizer: Optional[HMASynthesizer] = None

        self.generated_data: Dict[str, pd.DataFrame] = {}
        self.tables_config: Dict[str, TableConfig] = {}
        self.relationships: List[RelationshipConfig] = []
        self.is_fitted = False
        self.logger = self._setup_logging()

        # Enhanced PK tracking - track used values to prevent duplicates
        self.used_pk_values: Dict[Tuple[str, str], Set[Any]] = defaultdict(set)
        self.pk_sequences: Dict[Tuple[str, str], int] = defaultdict(int)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger(__name__)

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

    def create_sdv_metadata(self) -> MultiTableMetadata:
        self.metadata = MultiTableMetadata()

        # Add tables
        for table_name in self.tables_config.keys():
            self.metadata.add_table(table_name=table_name)

        # Add columns & PKs
        for table_name, table_config in self.tables_config.items():
            pk: Optional[str] = None
            for column in table_config.columns:
                business_values = self.helpers.parse_business_values(column.business_values)
                col_meta = self._enhanced_sdv_type_mapping(column, business_values)

                if column.is_pk:
                    col_meta["sdtype"] = "id"
                    pk = column.column_name
                if column.is_fk:
                    col_meta["sdtype"] = "id"

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
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not set primary key for {table_name}: {e}")

        # Add relationships
        added = 0
        for rel in self.relationships:
            try:
                self.metadata.add_relationship(
                    parent_table_name=rel.target_table,
                    parent_primary_key=rel.target_column,
                    child_table_name=rel.source_table,
                    child_foreign_key=rel.source_column,
                )
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

    def _enhanced_sdv_type_mapping(self, column, business_values: Optional[List[str]]) -> Dict[str, Any]:
        """Enhanced SDV type mapping"""
        base_type, length, precision, scale = self.config_parser.parse_data_type_details(column.data_type)

        mapping = {'sdtype': 'categorical'}

        if base_type in ['N', 'DC']:
            mapping['sdtype'] = 'numerical'
        elif base_type in ['D', 'DT', 'TS']:
            mapping['sdtype'] = 'datetime'
        elif any(keyword in column.column_name.lower() for keyword in ['id', 'code', 'key', 'num', 'nbr', 'seq']):
            mapping['sdtype'] = 'id'
        elif business_values:
            mapping['sdtype'] = 'categorical'
            mapping['order_by'] = business_values
        else:
            mapping['sdtype'] = 'text'

        return mapping

    # ---------------------------------------------------------------------
    # FIXED: Guaranteed Unique Primary Key Generation - NO LENGTH CONSTRAINTS
    # ---------------------------------------------------------------------

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
            for attempt in range(200):
                val = self.helpers.generate_special_value(special, base_type)
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
        """Generate table data with GUARANTEED PK uniqueness"""
        data: Dict[str, List[Any]] = {}

        # Calculate actual record count (with business values logic only)
        actual_num_records = self._calculate_actual_record_count(table_config, num_records)

        # Reset PK tracking for this table if training
        if for_training:
            for col in table_config.columns:
                if col.is_pk:
                    pk_key = (table_config.name, col.column_name)
                    self.used_pk_values[pk_key] = set()
                    self.pk_sequences[pk_key] = 1

        self.logger.info(
            f"📊 Generating {actual_num_records} records for {table_config.name} (requested: {num_records})")

        # Generate each column with uniqueness guarantee
        for column in table_config.columns:
            values = []
            null_p = self._get_null_probability(column)

            for i in range(actual_num_records):
                if random.random() < null_p and not column.is_pk:  # Never null PKs
                    values.append(None)
                else:
                    val = self._generate_enhanced_value(column, table_config.name, i, actual_num_records)
                    values.append(val)

            data[column.column_name] = values

        df = pd.DataFrame(data)

        # CRITICAL: Validate PK uniqueness before returning
        df = self._validate_and_fix_pk_uniqueness(df, table_config)

        # Apply constraints (except length constraints for PKs)
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

        # Business values first (for non-PK columns)
        business_values = self.helpers.parse_business_values(column.business_values)
        business_values = self._sanitize_business_values(business_values, base_type)
        if business_values:
            return random.choice(business_values)

        # Special rules
        if column.special_rules and not pd.isna(column.special_rules):
            special_value = self.helpers.generate_special_value(column.special_rules, base_type)
            if special_value is not None:
                return special_value

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
                "special_rules": column.special_rules
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
        if length <= 10:
            return self.helpers.faker.word()[:length]
        elif length <= 50:
            return self.helpers.faker.text(max_nb_chars=length)
        return self.helpers.faker.paragraph(nb_sentences=3)[:length]

    # ---------------------------------------------------------------------
    # NULL Generation Helpers
    # ---------------------------------------------------------------------
    def _parse_null_rate_from_rules(self, rules: Optional[str]) -> Optional[float]:
        if not rules or pd.isna(rules):
            return None
        s = str(rules).upper()
        m = re.search(r"NULL_RATE\s*=\s*([0-1]?(?:\.\d+)?)", s)
        if m:
            try:
                v = float(m.group(1))
                return max(0.0, min(1.0, v))
            except Exception:
                pass
        m = re.search(r"NULL_PCT\s*=\s*(\d+(?:\.\d+)?)", s)
        if m:
            try:
                pct = float(m.group(1))
                return max(0.0, min(1.0, pct / 100.0))
            except Exception:
                pass
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
                df[column.column_name] = df[column.column_name].apply(
                    lambda x: round(float(x), s) if pd.notna(x) else x
                )

            elif base_type == "NS" and length:
                L = int(length)
                df[column.column_name] = df[column.column_name].astype("string")
                df[column.column_name] = df[column.column_name].str.zfill(L)

            elif base_type == "AN" and length:
                L = int(length)
                df[column.column_name] = df[column.column_name].astype("string").str.ljust(L).str[:L]

            elif base_type == "A" and length:
                L = int(length)
                df[column.column_name] = df[column.column_name].astype("string").str.ljust(L).str[:L]

            elif base_type == "N" and length:
                max_val = 10 ** int(length) - 1

                def _cap(v):
                    if pd.isna(v):
                        return v
                    try:
                        iv = int(v)
                    except Exception:
                        return v
                    return min(iv, max_val)

                df[column.column_name] = df[column.column_name].apply(_cap)

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
            ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                coerced.append(s)
            else:
                if data_type == "D":
                    ts = pd.Timestamp(ts).normalize()
                coerced.append(pd.Timestamp(ts))
        return coerced

    # ---------------------------------------------------------------------
    # SDV Training & Data Generation
    # ---------------------------------------------------------------------
    def train_synthesizer(self, sample_size: int = 200) -> bool:
        try:
            if self.metadata is None:
                self.logger.error("❌ Metadata not created. Call create_sdv_metadata() first.")
                return False

            self.logger.info("🧑‍🤖 Initializing HMA Synthesizer...")

            self.synthesizer = HMASynthesizer(
                metadata=self.metadata,
                verbose=True,
                locales=['nl_NL']
            )

            # Generate high-quality sample data
            sample_sizes = {t: min(sample_size, 100) for t in self.tables_config.keys()}
            sample_data = self._generate_high_quality_sample_data(sample_sizes)

            self.logger.info("🛠️ Fitting synthesizer with enhanced sample data...")

            self.synthesizer.fit(sample_data)
            self.is_fitted = True

            self.logger.info("✅ SDV synthesizer trained and fitted successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error training synthesizer: {e}")
            self.logger.info("🔄 Continuing with enhanced fallback generation...")
            self.is_fitted = False
            return False

    def _generate_high_quality_sample_data(self, sample_sizes: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Generate high-quality sample data for SDV training"""
        sample_data = {}

        # Generate all tables first
        for table_name, num_records in sample_sizes.items():
            if table_name not in self.tables_config:
                continue

            table_config = self.tables_config[table_name]
            data = self._generate_table_data(table_config, num_records, for_training=True)
            sample_data[table_name] = data

        # Enforce relationships in sample data
        return self._enforce_relationships_in_sample(sample_data)

    def _enforce_relationships_in_sample(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enforce relationships in sample data for better SDV learning"""
        for relationship in self.relationships:
            parent_table = relationship.target_table
            child_table = relationship.source_table

            if parent_table in sample_data and child_table in sample_data:
                parent_df = sample_data[parent_table]
                child_df = sample_data[child_table]

                if (relationship.target_column in parent_df.columns and
                        relationship.source_column in child_df.columns):

                    valid_parent_values = parent_df[relationship.target_column].dropna().unique()

                    if len(valid_parent_values) > 0:
                        child_df[relationship.source_column] = random.choices(
                            valid_parent_values.tolist(),
                            k=len(child_df)
                        )
                        sample_data[child_table] = child_df

        return sample_data

    def generate_data(self, records_per_table: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """
        Enhanced data generation with guaranteed relationship integrity
        """
        try:
            # Try SDV generation first if fitted
            if self.is_fitted and self.synthesizer:
                self.logger.info("🎲 Generating data using trained SDV synthesizer...")
                try:
                    synthetic_data = self.synthesizer.sample(num_rows=records_per_table)

                    # Validate and enforce relationships in SDV data
                    if self._validate_sdv_data(synthetic_data):
                        synthetic_data = self._enforce_all_relationships(synthetic_data)
                        self.generated_data = synthetic_data
                        self._resolve_foreign_keys()  # Final FK resolution
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

            self.logger.info(f"✅ Fallback data generation completed: {total_records} total records")
            return self.generated_data

        except Exception as e:
            self.logger.error(f"❌ Error generating data: {e}")
            raise

    def _generate_fallback_data_with_relationships(self, records_per_table: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Enhanced fallback data generation with relationship guarantees"""
        synthetic_data: Dict[str, pd.DataFrame] = {}

        # Reset all PK sequences
        self._initialize_pk_tracking()

        # Generate reference tables first
        reference_tables = self._identify_reference_tables()
        for table_name in reference_tables:
            if table_name in records_per_table:
                table_config = self.tables_config[table_name]
                df = self._generate_table_data(table_config, records_per_table[table_name], for_training=False)
                synthetic_data[table_name] = df

        # Generate child tables
        for table_name, num_records in records_per_table.items():
            if table_name not in reference_tables:  # Skip already generated tables
                table_config = self.tables_config[table_name]
                df = self._generate_table_data(table_config, num_records, for_training=False)
                synthetic_data[table_name] = df

        # Enforce all relationships
        synthetic_data = self._enforce_all_relationships(synthetic_data)

        return synthetic_data

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
            for relationship in self.relationships:
                parent_table = relationship.target_table
                child_table = relationship.source_table

                if parent_table in data and child_table in data:
                    parent_df = data[parent_table]
                    child_df = data[child_table]

                    if (relationship.target_column in parent_df.columns and
                            relationship.source_column in child_df.columns):

                        valid_parent_values = parent_df[relationship.target_column].dropna().unique()

                        if len(valid_parent_values) > 0:
                            child_df[relationship.source_column] = random.choices(
                                valid_parent_values.tolist(),
                                k=len(child_df)
                            )
                            data[child_table] = child_df

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
    # Foreign Key Resolution
    # ---------------------------------------------------------------------
    def _resolve_foreign_keys(self) -> None:
        """Resolve foreign key relationships in generated_data"""
        if not self.relationships:
            return

        self.logger.info("🔗 Resolving foreign key relationships...")
        resolved_count = 0

        for relationship in self.relationships:
            src_t = relationship.source_table
            tgt_t = relationship.target_table

            if src_t in self.generated_data and tgt_t in self.generated_data:
                src_df = self.generated_data[src_t]
                tgt_df = self.generated_data[tgt_t]

                if (relationship.source_column in src_df.columns) and (relationship.target_column in tgt_df.columns):
                    valid = tgt_df[relationship.target_column].dropna().unique().tolist()
                    if valid:
                        original_dtype = src_df[relationship.source_column].dtype
                        src_df[relationship.source_column] = random.choices(valid, k=len(src_df))
                        try:
                            src_df[relationship.source_column] = src_df[relationship.source_column].astype(
                                original_dtype)
                        except (ValueError, TypeError):
                            pass

                        self.generated_data[src_t] = src_df
                        resolved_count += 1
                        self.logger.info(
                            f" ✅ Resolved FK: {src_t}.{relationship.source_column} → {tgt_t}.{relationship.target_column}")

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
                        arr_ts = pc.assume_timezone(arr_ts_naive, "UTC")
                        arrays.append(arr_ts)
                        names.append(col)

                    elif base_type == "D":
                        dt_series = pd.to_datetime(s, errors="coerce").dt.date
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
                        vals = pd.to_numeric(s, errors="coerce")
                        has_negative = pd.notna(vals) & (vals < 0)
                        if has_negative.any():
                            vals = vals.astype("Int64")
                            arr = pa.array(vals, type=pa.int64())
                        else:
                            max_val = vals.max(skipna=True)
                            if pd.isna(max_val):
                                vals = vals.astype("Int64")
                                arr = pa.array(vals, type=pa.int64())
                            elif max_val > np.iinfo("int64").max:
                                vals = vals.astype("UInt64")
                                arr = pa.array(vals, type=pa.uint64())
                            else:
                                vals = vals.astype("Int64")
                                arr = pa.array(vals, type=pa.int64())
                        arrays.append(arr)
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
                "status": "NO_DATA_GENERATED",
            }

        counts: Dict[str, int] = {t: (0 if df.empty else len(df)) for t, df in self.generated_data.items()}
        total = sum(counts.values())

        return {
            "tables_generated": list(self.generated_data.keys()),
            "total_records": total,
            "relationships_configured": len(self.relationships),
            "table_record_counts": counts,
            "synthesizer_fitted": self.is_fitted,
            "status": "SUCCESS" if total > 0 else "FAILED",
        }
