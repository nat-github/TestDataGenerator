# -*- coding: utf-8 -*-
"""
data_generator.py - FIXED PK UNIQUENESS FOR ALL DATA TYPES

Key fix: Remove data type length constraints for Primary Keys to guarantee uniqueness
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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

from sdp.generators._anchors import AnchorMixin
from sdp.generators._arrow_export import ArrowExportMixin
from sdp.generators._model_cache import ModelCacheMixin
from sdp.generators._primary_keys import PrimaryKeyMixin
from sdp.generators._relationships import RelationshipMixin
from sdp.generators._sdv_metadata import SDVMetadataMixin
from sdp.models.config_models import TableConfig, RelationshipConfig
from sdp.utils.config_parser import ConfigParser
from sdp.utils.helpers import DataHelpers
from sdp.utils.rule_evaluator import apply_to_dataframe as apply_rules_to_dataframe


class DataGenerator(
    AnchorMixin,
    PrimaryKeyMixin,
    SDVMetadataMixin,
    RelationshipMixin,
    ModelCacheMixin,
    ArrowExportMixin,
):
    """Orchestrates generation: config → metadata → train → generate → export.

    The concerns below are inherited rather than defined here — each was
    lifted verbatim into its own module so this class stays readable:

    ==================== ============================================
    Mixin                Owns
    ==================== ============================================
    ``AnchorMixin``      ``source:`` tables loaded from real data
    ``PrimaryKeyMixin``  PK generation and uniqueness repair
    ``SDVMetadataMixin`` SDV metadata + training-sample sanitisation
    ``RelationshipMixin``FK resolution and referential integrity
    ``ModelCacheMixin``  fingerprint-keyed artifact caching
    ``ArrowExportMixin`` Arrow casting and Parquet export
    ==================== ============================================

    What remains here is the orchestration itself plus per-column value
    generation, rules/derived columns, and the fallback path.
    """

    SAFE_DATETIME_MIN = pd.Timestamp("1900-01-01 00:00:00")
    SAFE_DATETIME_MAX = pd.Timestamp("2262-04-11 23:47:16")

    def __init__(self, config_file: str, seed: Optional[int] = None,
                 engine: Optional[str] = None,
                 engine_options: Optional[Dict[str, Any]] = None):
        self.config_file = config_file
        self.config_parser = ConfigParser(config_file)
        self.helpers = DataHelpers()
        self.metadata: Optional[Metadata] = None
        self.synthesizer: Optional[HMASynthesizer] = None

        # Generation engine. `engine=` wins over the config's
        # `synthesizer_engine` setting, which wins over the registry default.
        # `self.synthesizer` remains the underlying model object so existing
        # callers (and the artifact cache) keep working unchanged.
        self._engine_override = engine
        self._engine_options_override = dict(engine_options) if engine_options else None
        self._engine: Optional["object"] = None

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
        except ImportError:
            # Faker is a core dependency, but seeding it is best-effort: a
            # build without it still generates, just not reproducibly for
            # Faker-backed columns.
            self.logger.debug("Faker unavailable — Faker-backed columns will not be seeded")

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


    # ---------------------------------------------------------------------
    # Anchored generation — load real datasets declared via `source:`
    # ---------------------------------------------------------------------











    # ---------------------------------------------------------------------
    # FIXED: Guaranteed Unique Primary Key Generation - NO LENGTH CONSTRAINTS
    # ---------------------------------------------------------------------


    # --- In _generate_unique_primary_key(...), add special_rule handling up-front ---




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
        # A NOT NULL column never receives nulls: the hard `nullable: false`
        # constraint overrides any NULL_PCT / NULL_RATE special rule that might
        # contradict it (the stronger constraint always wins).
        if getattr(column, "nullable", True) is False:
            return 0.0
        explicit = getattr(column, "null_rate", None)
        if explicit is not None and not pd.isna(explicit):
            try:
                return max(0.0, min(1.0, float(explicit)))
            except (TypeError, ValueError):
                # A null_rate that will not parse is a config error, not a
                # normal condition — say so rather than silently falling
                # through to the special-rule path.
                self.logger.warning(
                    f"Column {getattr(column, 'column_name', '?')}: null_rate "
                    f"{explicit!r} is not a number — ignoring it"
                )
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






    # ---------------------------------------------------------------------
    # SDV Training & Data Generation
    # ---------------------------------------------------------------------
    def resolve_engine_name(self) -> str:
        """Which engine this run will use.

        Precedence: constructor argument → ``synthesizer_engine`` config
        setting → registry default (``sdv``, the historical behaviour).
        """
        from sdp.synthesizers import DEFAULT_ENGINE

        if self._engine_override:
            return str(self._engine_override)
        configured = self.config_parser.get_setting('synthesizer_engine', None)
        return str(configured) if configured else DEFAULT_ENGINE

    @property
    def engine_stats(self) -> Optional[Dict[str, Any]]:
        """Fit/sample cost for the engine used, or None if none was fitted."""
        engine = self._engine
        return engine.stats.to_dict() if engine is not None else None

    @property
    def privacy_report(self) -> Optional[Dict[str, Any]]:
        """Privacy accounting, when the engine provides a formal guarantee.

        None for every engine except ``dp-marginal`` — absence here means
        "no formal guarantee was made", which is the honest answer for the
        engines that make none.
        """
        engine = self._engine
        reporter = getattr(engine, "privacy_report", None)
        return reporter() if callable(reporter) else None

    def _engine_options(self) -> Dict[str, Any]:
        """Engine constructor options from the config's ``engine_options``.

        Accepts a dict, or a ``k=v;k=v`` string for Excel configs where a
        cell cannot hold structured data.
        """
        if self._engine_options_override is not None:
            return dict(self._engine_options_override)

        raw = self.config_parser.get_setting('engine_options', None)
        if not raw:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        options: Dict[str, Any] = {}
        for pair in str(raw).split(';'):
            if '=' not in pair:
                continue
            key, _, value = pair.partition('=')
            value = value.strip()
            # Numeric-looking values are far more common than string ones
            # here (epochs, batch_size), so coerce when unambiguous.
            if value.isdigit():
                options[key.strip()] = int(value)
            else:
                options[key.strip()] = value
        return options

    def build_column_domains(self) -> Dict[str, Dict[str, Any]]:
        """Publicly declared value spaces, straight from the config.

        Differential privacy needs a domain that does not come from the
        data — deriving bin edges from the observed min/max, or the category
        list from observed values, is itself a non-private query and makes
        the published epsilon a fiction. The config already declares both
        (``business_values``, ``min_value`` / ``max_value``), so the DP
        engine can measure against it without ever inspecting the data.

        Columns with no declared domain are simply absent here; the engine
        generates them from config rules and spends no budget on them.

        Primary and foreign keys are excluded deliberately: they are
        identifiers, not distributions. PKs must stay unique and FKs are
        overwritten by FK resolution, so modelling either would be pointless
        as well as privacy-relevant.
        """
        from sdp.synthesizers.dp_marginal import ColumnDomain

        # Base types as the config parser reports them: N38 → 'N', DC(18,2) →
        # 'DC'. Use the parser rather than string-slicing the declared type,
        # so this stays correct as the type grammar evolves.
        integer_types = {"N"}
        numeric_types = integer_types | {"DC"}

        domains: Dict[str, Dict[str, Any]] = {}
        for table_name, table_config in self.tables_config.items():
            table_domains: Dict[str, Any] = {}
            for column in table_config.columns:
                if column.is_pk or column.is_fk:
                    continue

                values = self.helpers.parse_business_values(column.business_values) \
                    if hasattr(self.helpers, "parse_business_values") else None
                if not values and column.business_values:
                    values = [v.strip() for v in str(column.business_values).split(";") if v.strip()]

                if values:
                    table_domains[column.column_name] = ColumnDomain(
                        kind="categorical", values=list(values),
                    )
                    continue

                base_type, *_ = self.config_parser.parse_data_type_details(
                    column.data_type or ""
                )
                if base_type in numeric_types and column.min_value is not None \
                        and column.max_value is not None:
                    try:
                        low = float(column.min_value)
                        high = float(column.max_value)
                    except (TypeError, ValueError):
                        continue
                    if high > low:
                        table_domains[column.column_name] = ColumnDomain(
                            kind="numeric", low=low, high=high,
                            integer=base_type in integer_types,
                        )

            if table_domains:
                domains[table_name] = table_domains
        return domains

    def _dp_engine_options(self) -> Dict[str, Any]:
        """Domains and a config-only frame factory for the DP engine.

        ``frame_factory`` is what keeps unprivatised columns honest: they
        are generated purely from the config, so they never see the training
        data and cost no privacy budget.
        """
        return {
            "domains": self.build_column_domains(),
            "frame_factory": lambda table, n: self._generate_table_data(
                self.tables_config[table], n, for_training=False,
            ),
        }

    def train_synthesizer(self, sample_size: int = 200) -> bool:
        """Fit the configured engine. Returns False to mean "generate from
        config rules instead" — a normal outcome, not necessarily an error."""
        from sdp.synthesizers import create as create_engine

        try:
            if self.metadata is None:
                self.logger.error("❌ Metadata not created. Call create_sdv_metadata() first.")
                return False

            configured_sample_size = self.config_parser.get_setting('synthesizer_sample_size', sample_size)
            try:
                sample_size = int(configured_sample_size)
            except (TypeError, ValueError):
                sample_size = sample_size

            engine_name = self.resolve_engine_name()

            # Reuse a cached synthesizer when one matches this exact config and
            # SDV version — skips training entirely. Any mismatch falls through
            # to a normal fit, so a stale model is never used. Cached artifacts
            # are SDV-specific.
            if engine_name == "sdv" and self._try_load_cached_synthesizer():
                return True

            self._apply_seed(self.seed)
            engine_options = dict(self._engine_options())
            if engine_name == "dp-marginal":
                engine_options = {**self._dp_engine_options(), **engine_options}
            engine = create_engine(engine_name, seed=self.seed, **engine_options)
            self._engine = engine

            if not engine.__class__.handles_relationships and engine_name != "rule-based":
                self.logger.info(
                    f"ℹ️ Engine {engine_name!r} models tables independently — "
                    "cross-table integrity comes from FK resolution after sampling."
                )

            self.logger.info(f"🧑‍🤖 Initializing synthesizer engine: {engine_name}")

            # Generate high-quality sample data
            sample_sizes = {t: min(sample_size, 100) for t in self.tables_config.keys()}
            sample_data = self._sanitize_sample_data_for_sdv(
                self._generate_high_quality_sample_data(sample_sizes)
            )

            self.logger.info("🛠️ Fitting synthesizer with enhanced sample data...")
            fitted_sample_data = sample_data

            if not engine.fit(sample_data, self.metadata):
                # The rule-based engine has nothing to fit — that is the point
                # of selecting it, so don't waste a retry on it.
                if engine_name == "rule-based":
                    self.synthesizer = None
                    self.is_fitted = False
                    return False

                self.logger.warning("⚠️ Initial synthesizer fit attempt failed")
                retry_sample_sizes = {t: max(25, min(sample_size, 75)) for t in self.tables_config.keys()}
                retry_sample_data = self._sanitize_sample_data_for_sdv(
                    self._generate_high_quality_sample_data(retry_sample_sizes),
                    aggressive=True,
                )
                # A fresh engine — a half-fitted model is not a safe retry base.
                engine = create_engine(engine_name, seed=self.seed, **engine_options)
                self._engine = engine
                self.logger.info("🔁 Retrying synthesizer fit with aggressively sanitized sample data...")
                if not engine.fit(retry_sample_data, self.metadata):
                    raise RuntimeError(
                        "; ".join(engine.stats.notes) or "synthesizer fit failed"
                    )
                engine.stats.retries += 1
                fitted_sample_data = retry_sample_data

            self.synthesizer = engine.model
            self._fitted_sample_sizes = {
                table_name: len(df)
                for table_name, df in fitted_sample_data.items()
            }
            self.is_fitted = True

            self.logger.info(
                f"✅ Synthesizer trained and fitted successfully "
                f"(engine={engine_name}, {engine.stats.fit_seconds:.1f}s)"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Error training synthesizer: {e}")
            self.logger.info("🔄 Continuing with enhanced fallback generation...")
            self.is_fitted = False
            return False

    # ---------------------------------------------------------------------
    # Model artifact caching — fingerprint-keyed, SDV-version-validated
    # ---------------------------------------------------------------------




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





    def _sample_from_synthesizer(self, records_per_table: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Draw rows from the fitted engine.

        When an engine owns the current model we delegate to it, so
        engine-specific sampling (per-table for single-table engines) and
        cost accounting both apply. When a synthesizer was assigned
        directly — the artifact cache, or a test injecting a double — we
        fall back to the shared multi-table helper, which is the same logic
        the SDV engine uses.
        """
        from sdp.synthesizers import sample_multi_table

        engine = self._engine
        if engine is not None and engine.is_fitted and engine.model is self.synthesizer:
            return engine.sample(records_per_table)

        return sample_multi_table(
            self.synthesizer, records_per_table, self._fitted_sample_sizes,
        )

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

    # ---------------------------------------------------------------------
    # Export Methods
    # ---------------------------------------------------------------------

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
