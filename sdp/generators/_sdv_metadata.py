"""SDV metadata construction and training-sample sanitisation.

Extracted from ``data_generator.py`` as a mixin. ``DataGenerator`` inherits
it, so ``self`` resolves exactly as before — this is a pure move, not a
behaviour change. The split exists so each concern can be read and tested
without loading a 2,400-line class.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from sdv.metadata import Metadata

from sdp.models.config_models import TableConfig

logger = logging.getLogger(__name__)


class SDVMetadataMixin:
    """SDV metadata construction and training-sample sanitisation."""

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

    @staticmethod
    def _resolve_generator_max_length(base_type: str, length: Optional[int]) -> Optional[int]:
        if base_type in {"A", "AN", "NS", "VA"}:
            return int(length) if length is not None else None
        return None

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
