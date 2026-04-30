import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from models.config_models import ColumnConfig, TableConfig, RelationshipConfig
from utils.helpers import DataHelpers


class ConfigParser:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_df: Optional[pd.DataFrame] = None
        self.tables_df: Optional[pd.DataFrame] = None
        self.relationships_df: Optional[pd.DataFrame] = None
        self.run_settings_df: Optional[pd.DataFrame] = None
        self.tables: Dict[str, TableConfig] = {}
        self.relationships: List[RelationshipConfig] = []
        self.helpers = DataHelpers()
        self.run_settings: Dict[str, Any] = {}
        self.available_sheets: List[str] = []
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _split_multi_value(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, float) and np.isnan(value):
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            return [part.strip() for part in cleaned.replace(',', ';').split(';') if part.strip()]
        return [str(value).strip()]

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, float):
            if np.isnan(value):
                return default
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'true', 't', 'yes', 'y', '1'}:
                return True
            if normalized in {'false', 'f', 'no', 'n', '0'}:
                return False
        return default

    @staticmethod
    def _coerce_setting_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            lowered = stripped.lower()
            if lowered in {'true', 'false'}:
                return lowered == 'true'
            if re.fullmatch(r'-?\d+', stripped):
                try:
                    return int(stripped)
                except ValueError:
                    return stripped
            if re.fullmatch(r'-?\d+\.\d+', stripped):
                try:
                    return float(stripped)
                except ValueError:
                    return stripped
            return stripped
        return value

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        return text or None

    def _get_optional_sheet(self, excel_file: pd.ExcelFile, *sheet_names: str) -> Optional[pd.DataFrame]:
        for sheet_name in sheet_names:
            if sheet_name in self.available_sheets:
                return pd.read_excel(excel_file, sheet_name=sheet_name)
        return None

    def get_setting(self, setting_name: str, default: Any = None) -> Any:
        return self.run_settings.get(setting_name, default)

    def _normalize_loaded_frames(self) -> bool:
        if self.config_df is None:
            self.logger.error("❌ Columns configuration could not be loaded")
            return False

        # --- Parse data types into normalized columns ---
        type_details = self.config_df['data_type'].apply(self.parse_data_type_details)
        self.config_df['base_data_type'] = type_details.apply(lambda x: x[0])
        self.config_df['length'] = type_details.apply(lambda x: x[1])
        self.config_df['precision'] = type_details.apply(lambda x: x[2])
        self.config_df['scale'] = type_details.apply(lambda x: x[3])

        # --- Normalize NaN -> None for STRING columns ---
        string_columns = ['ref_table', 'ref_column', 'business_values', 'special_rules']
        for col in string_columns:
            if col in self.config_df.columns:
                self.config_df[col] = self.config_df[col].replace({np.nan: None})

        # --- Normalize NaN -> None for NUMERIC columns ---
        numeric_columns = ['min_value', 'max_value', 'length', 'precision', 'scale']
        for col in numeric_columns:
            if col in self.config_df.columns:
                self.config_df[col] = self.config_df[col].where(~self.config_df[col].isna(), None)

        # --- Booleans: ensure is_pk / is_fk are True/False (not NaN) ---
        for col in ['is_pk', 'is_fk']:
            if col in self.config_df.columns:
                self.config_df[col] = self.config_df[col].fillna(False).astype(bool)

        # --- Standard cleaning for names ---
        if 'table_name' in self.config_df.columns:
            self.config_df['table_name'] = self.config_df['table_name'].astype(str).str.lower().str.strip()

        if 'column_name' in self.config_df.columns:
            self.config_df['column_name'] = self.config_df['column_name'].astype(str).str.strip()

        if 'ref_table' in self.config_df.columns:
            self.config_df['ref_table'] = self.config_df['ref_table'].apply(
                lambda v: v.lower().strip() if isinstance(v, str) else v
            )

        if self.tables_df is not None and not self.tables_df.empty and 'table_name' in self.tables_df.columns:
            self.tables_df['table_name'] = self.tables_df['table_name'].astype(str).str.lower().str.strip()

        if self.relationships_df is not None and not self.relationships_df.empty:
            for name_col in ['source_table', 'target_table']:
                if name_col in self.relationships_df.columns:
                    self.relationships_df[name_col] = self.relationships_df[name_col].astype(str).str.lower().str.strip()

        self.run_settings = {}
        if self.run_settings_df is not None and not self.run_settings_df.empty:
            setting_col = 'setting_name' if 'setting_name' in self.run_settings_df.columns else None
            value_col = 'value' if 'value' in self.run_settings_df.columns else None
            if setting_col and value_col:
                for _, row in self.run_settings_df.iterrows():
                    key = str(row[setting_col]).strip()
                    if key:
                        self.run_settings[key] = self._coerce_setting_value(row[value_col])

        self.logger.info(f"✅ Loaded and cleaned configuration with {len(self.config_df)} columns")
        self.logger.info(f"📊 Detected base data types: {self.config_df['base_data_type'].unique()}")
        return True

    def _load_excel_config(self) -> bool:
        excel_file = pd.ExcelFile(self.config_file)
        self.available_sheets = excel_file.sheet_names

        self.config_df = pd.read_excel(excel_file, sheet_name='Columns')
        self.tables_df = self._get_optional_sheet(excel_file, 'Tables')
        self.relationships_df = self._get_optional_sheet(excel_file, 'Relationships')
        self.run_settings_df = self._get_optional_sheet(excel_file, 'Run_Settings', 'Generation_Defaults')
        return self._normalize_loaded_frames()

    def _yaml_settings_to_frame(self, settings: Any) -> Optional[pd.DataFrame]:
        if settings is None:
            return None

        rows: List[Dict[str, Any]] = []
        if isinstance(settings, dict):
            rows = [
                {'setting_name': key, 'value': value, 'description': ''}
                for key, value in settings.items()
            ]
        elif isinstance(settings, list):
            for item in settings:
                if not isinstance(item, dict):
                    continue
                key = item.get('setting_name', item.get('name'))
                if not key:
                    continue
                rows.append({
                    'setting_name': key,
                    'value': item.get('value'),
                    'description': item.get('description', ''),
                })

        return pd.DataFrame(rows) if rows else None

    def _load_yaml_config(self) -> bool:
        with open(self.config_file, 'r', encoding='utf-8') as handle:
            raw_config = yaml.safe_load(handle) or {}

        if not isinstance(raw_config, dict):
            self.logger.error('❌ YAML configuration must contain a top-level mapping')
            return False

        columns_rows: List[Dict[str, Any]] = []
        tables_rows: List[Dict[str, Any]] = []

        for raw_table in raw_config.get('tables', []) or []:
            if not isinstance(raw_table, dict):
                continue

            table_name = self._optional_string(raw_table.get('name') or raw_table.get('table_name'))
            if not table_name:
                continue

            primary_key_columns = self._split_multi_value(raw_table.get('primary_key_columns'))
            business_key_columns = self._split_multi_value(raw_table.get('business_key_columns'))
            scd2_tracked_columns = self._split_multi_value(raw_table.get('scd2_tracked_columns'))
            partition_columns = self._split_multi_value(raw_table.get('partition_columns'))
            event_time_column = self._optional_string(raw_table.get('event_time_column'))

            tables_rows.append({
                'table_name': table_name,
                'table_kind': raw_table.get('table_kind'),
                'description': raw_table.get('description'),
                'row_count': raw_table.get('row_count', raw_table.get('num_rows')),
                'generation_mode': raw_table.get('generation_mode'),
                'business_key_columns': business_key_columns,
                'primary_key_columns': primary_key_columns,
                'partition_enabled': raw_table.get('partition_enabled'),
                'partition_columns': partition_columns,
                'event_time_column': event_time_column,
                'scd2_enabled': raw_table.get('scd2_enabled'),
                'scd2_tracked_columns': scd2_tracked_columns,
                'delta_eligible': raw_table.get('delta_eligible'),
                'active': raw_table.get('active', True),
                'notes': raw_table.get('notes'),
            })

            pk_set = set(primary_key_columns)
            business_key_set = set(business_key_columns)
            scd2_set = set(scd2_tracked_columns)
            partition_set = set(partition_columns)

            for raw_column in raw_table.get('columns', []) or []:
                if not isinstance(raw_column, dict):
                    continue

                column_name = self._optional_string(raw_column.get('name') or raw_column.get('column_name'))
                if not column_name:
                    continue

                partition_role = raw_column.get('partition_role')
                if partition_role is None and column_name in partition_set:
                    partition_role = 'partition_key'
                if partition_role is None and event_time_column and column_name == event_time_column:
                    partition_role = 'event_time'

                columns_rows.append({
                    'table_name': table_name,
                    'column_name': column_name,
                    'data_type': raw_column.get('data_type'),
                    'is_pk': raw_column.get('is_pk', column_name in pk_set),
                    'is_fk': raw_column.get('is_fk', False),
                    'ref_table': raw_column.get('ref_table'),
                    'ref_column': raw_column.get('ref_column'),
                    'business_values': raw_column.get('business_values'),
                    'special_rules': raw_column.get('special_rules'),
                    'min_value': raw_column.get('min_value'),
                    'max_value': raw_column.get('max_value'),
                    'nullable': raw_column.get('nullable'),
                    'is_business_key_component': raw_column.get('is_business_key_component', column_name in business_key_set),
                    'event_time': raw_column.get('event_time', bool(event_time_column and column_name == event_time_column)),
                    'partition_role': partition_role,
                    'scd2_tracked': raw_column.get('scd2_tracked', column_name in scd2_set),
                })

        relationship_rows: List[Dict[str, Any]] = []
        for raw_relationship in raw_config.get('relationships', []) or []:
            if not isinstance(raw_relationship, dict):
                continue

            source_table = self._optional_string(raw_relationship.get('source_table'))
            target_table = self._optional_string(raw_relationship.get('target_table'))
            if not source_table or not target_table:
                continue

            relationship_rows.append({
                'relationship_name': raw_relationship.get('name', raw_relationship.get('relationship_name')),
                'source_table': source_table,
                'source_columns': raw_relationship.get('source_columns', raw_relationship.get('source_column')),
                'target_table': target_table,
                'target_columns': raw_relationship.get('target_columns', raw_relationship.get('target_column')),
                'cardinality': raw_relationship.get('cardinality', raw_relationship.get('relationship_type')),
                'preserve_on_delta': raw_relationship.get('preserve_on_delta', False),
                'active': raw_relationship.get('active', True),
                'notes': raw_relationship.get('notes'),
            })

        self.config_df = pd.DataFrame(columns_rows)
        self.tables_df = pd.DataFrame(tables_rows) if tables_rows else None
        self.relationships_df = pd.DataFrame(relationship_rows) if relationship_rows else None
        self.run_settings_df = self._yaml_settings_to_frame(raw_config.get('run_settings'))
        self.available_sheets = ['Columns']
        if self.tables_df is not None and not self.tables_df.empty:
            self.available_sheets.append('Tables')
        if self.relationships_df is not None and not self.relationships_df.empty:
            self.available_sheets.append('Relationships')
        if self.run_settings_df is not None and not self.run_settings_df.empty:
            self.available_sheets.append('Run_Settings')

        return self._normalize_loaded_frames()

    # ---------------------------------------------------------------------
    # Data type parsing
    # ---------------------------------------------------------------------
    def parse_data_type_details(self, data_type: str) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
        """
        Universal data type parser.

        Returns:
            (base_type, length, precision, scale)
        Examples:
            DC(18,2) -> ('DC', None, 18, 2)
            NS(15)   -> ('NS', 15, None, None)
            A34      -> ('A', 34, None, None)
            N19      -> ('N', 19, None, None)
            DC       -> ('DC', None, None, None)   # will use defaults later if needed
            D        -> ('D', None, None, None)
            TS       -> ('TS', None, None, None)
        """
        # Handle non-string inputs gracefully
        if not isinstance(data_type, str):
            return str(data_type), None, None, None

        s = data_type.strip().upper()

        # 1) Enhanced format with parentheses, e.g., DC(18,2), NS(15), AN(20), N(10), A(140), VA(256)
        #    Groups: base, param1, param2 (optional)
        enhanced_pattern = r'^([A-Z]+)\((\d+)(?:,(\d+))?\)$'
        m = re.match(enhanced_pattern, s)
        if m:
            base = m.group(1)
            p1 = int(m.group(2))
            p2 = int(m.group(3)) if m.group(3) is not None else None
            # DC(precision, scale)
            if base == 'DC':
                precision = p1
                scale = p2 if p2 is not None else 2
                self.logger.debug(f"Enhanced DC detected: {s} -> precision={precision}, scale={scale}")
                return base, None, precision, scale
            # Other enhanced types use 'length' (A/N/VA/AN/NS)
            self.logger.debug(f"Enhanced type detected: {s} -> base={base}, length={p1}")
            return base, p1, None, None

        # 2) Legacy compact format without parentheses, e.g., A34, N19, VA256
        legacy_pattern = r'^([A-Z]+)(\d+)$'
        m = re.match(legacy_pattern, s)
        if m:
            base = m.group(1)
            length = int(m.group(2))
            self.logger.debug(f"Legacy type detected: {s} -> base={base}, length={length}")
            return base, length, None, None

        # 3) Simple types without parameters
        simple_types = ['DC', 'D', 'DT', 'TS', 'NS', 'AN', 'N', 'A', 'VA']
        if s in simple_types:
            # For NS/AN without params, set a sensible default 'length'
            length = None
            if s == 'NS':
                length = 15
            elif s == 'AN':
                length = 20
            self.logger.debug(f"Simple type detected: {s} -> length={length}")
            return s, length, None, None

        # 4) Fallback: return as-is (treated as string type elsewhere)
        self.logger.debug(f"Unknown data type format, treating as string: {s}")
        return s, None, None, None

    # ---------------------------------------------------------------------
    # Load & normalize
    # ---------------------------------------------------------------------
    def load_config(self) -> bool:
        """
        Load and clean configuration from Excel with enhanced parsing and NaN→None normalization.
        """
        try:
            suffix = Path(self.config_file).suffix.lower()
            if suffix in {'.yaml', '.yml'}:
                return self._load_yaml_config()
            return self._load_excel_config()

        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return False

    # ---------------------------------------------------------------------
    # Build table & relationship models
    # ---------------------------------------------------------------------
    def parse_tables(self) -> Dict[str, TableConfig]:
        """
        Parse tables and columns from the DataFrame with enhanced data type handling.
        """
        self.tables = {}

        if self.config_df is None:
            self.logger.error("❌ Config not loaded; call load_config() first.")
            return self.tables

        tables_meta: Dict[str, Dict[str, Any]] = {}
        if self.tables_df is not None and not self.tables_df.empty:
            for _, row in self.tables_df.iterrows():
                table_name = str(row.get('table_name', '')).lower().strip()
                if not table_name:
                    continue
                is_active = self._to_bool(row.get('active'), True)
                if not is_active:
                    continue
                tables_meta[table_name] = row.to_dict()

        for table_name in self.config_df['table_name'].dropna().unique():
            table_data = self.config_df[self.config_df['table_name'] == table_name]
            columns: List[ColumnConfig] = []
            table_meta = tables_meta.get(table_name, {})

            for _, row in table_data.iterrows():
                column_config = ColumnConfig(
                    table_name=table_name,
                    column_name=row['column_name'],
                    data_type=row['data_type'],     # keep original string for reference
                    is_pk=bool(row.get('is_pk', False)),
                    is_fk=bool(row.get('is_fk', False)),
                    ref_table=row.get('ref_table'),
                    ref_column=row.get('ref_column'),
                    business_values=row.get('business_values'),
                    special_rules=row.get('special_rules'),
                    min_value=row.get('min_value'),
                    max_value=row.get('max_value'),
                    length=row.get('length'),
                    precision=row.get('precision'),
                    scale=row.get('scale'),
                    nullable=self._to_bool(row.get('nullable'), not bool(row.get('is_pk', False))),
                    is_business_key_component=self._to_bool(row.get('is_business_key_component'), False),
                    event_time=self._to_bool(row.get('event_time'), False),
                    partition_role=(
                        None
                        if row.get('partition_role') is None or (isinstance(row.get('partition_role'), float) and np.isnan(row.get('partition_role')))
                        else row.get('partition_role')
                    ),
                    scd2_tracked=self._to_bool(row.get('scd2_tracked'), False),
                )
                columns.append(column_config)

            derived_pk_columns = [col.column_name for col in columns if col.is_pk]
            derived_business_keys = [col.column_name for col in columns if col.is_business_key_component]
            derived_event_time_column = next((col.column_name for col in columns if col.event_time), None)
            derived_scd2_columns = [col.column_name for col in columns if col.scd2_tracked]
            generation_mode = str(table_meta.get('generation_mode') or 'snapshot').strip().lower()
            row_count = table_meta.get('row_count', table_meta.get('initial_row_count', None))
            if row_count is None:
                row_count = self.get_setting('default_records_per_table', 1000)

            self.tables[table_name] = TableConfig(
                name=table_name,
                columns=columns,
                num_rows=int(row_count),
                table_kind=str(table_meta.get('table_kind') or 'transactional').strip().lower(),
                description=self._optional_string(table_meta.get('description')),
                generation_mode=generation_mode,
                business_key_columns=self._split_multi_value(table_meta.get('business_key_columns')) or derived_business_keys,
                primary_key_columns=self._split_multi_value(table_meta.get('primary_key_columns')) or derived_pk_columns,
                partition_enabled=self._to_bool(table_meta.get('partition_enabled'), False),
                partition_columns=self._split_multi_value(table_meta.get('partition_columns') or table_meta.get('default_partition_columns')),
                event_time_column=self._optional_string(table_meta.get('event_time_column')) or derived_event_time_column,
                scd2_enabled=self._to_bool(table_meta.get('scd2_enabled'), generation_mode in {'scd2', 'scd2_ready'}),
                scd2_tracked_columns=self._split_multi_value(table_meta.get('scd2_tracked_columns')) or derived_scd2_columns,
                delta_eligible=self._to_bool(
                    table_meta.get('delta_eligible', table_meta.get('delta_enabled')),
                    generation_mode in {'delta', 'delta_ready'},
                ),
                active=True,
                notes=self._optional_string(table_meta.get('notes')),
            )

        return self.tables

    def parse_relationships(self) -> List[RelationshipConfig]:
        """
        Automatically extract relationships from FK definitions.
        """
        self.relationships = []

        if self.config_df is None:
            self.logger.error("❌ Config not loaded; call load_config() first.")
            return self.relationships

        if self.relationships_df is not None and not self.relationships_df.empty:
            for _, row in self.relationships_df.iterrows():
                if not self._to_bool(row.get('active'), True):
                    continue

                source_columns = self._split_multi_value(row.get('source_columns'))
                target_columns = self._split_multi_value(row.get('target_columns'))
                if len(source_columns) != len(target_columns):
                    self.logger.warning(
                        f"⚠️ Skipping relationship {row.get('relationship_name')} because source/target column counts do not match"
                    )
                    continue

                for idx, (source_column, target_column) in enumerate(zip(source_columns, target_columns), start=1):
                    rel_name = row.get('relationship_name') or f"{row.get('source_table')}_{source_column}_to_{row.get('target_table')}_{target_column}"
                    if len(source_columns) > 1:
                        rel_name = f"{rel_name}#{idx}"
                    rel = RelationshipConfig(
                        name=rel_name,
                        source_table=row['source_table'],
                        source_column=source_column,
                        target_table=row['target_table'],
                        target_column=target_column,
                        relationship_type=row.get('cardinality') or row.get('relationship_type') or 'many_to_one',
                        preserve_on_delta=self._to_bool(row.get('preserve_on_delta'), False),
                        active=True,
                        notes=row.get('notes'),
                    )
                    self.relationships.append(rel)

            self.logger.info(f"✅ Loaded {len(self.relationships)} relationships from Relationships sheet")
            return self.relationships

        # must be explicit booleans and have ref_table/ref_column
        fk_mask = (
            (self.config_df.get('is_fk', False) == True) &
            (self.config_df.get('ref_table').notna()) &
            (self.config_df.get('ref_column').notna())
        )

        fk_data = self.config_df[fk_mask] if 'is_fk' in self.config_df.columns else pd.DataFrame(columns=self.config_df.columns)

        for _, row in fk_data.iterrows():
            rel = RelationshipConfig(
                source_table=row['table_name'],
                source_column=row['column_name'],
                target_table=row['ref_table'],
                target_column=row['ref_column']
            )
            self.relationships.append(rel)

        self.logger.info(f"✅ Automatically identified {len(self.relationships)} relationships from FK definitions")
        return self.relationships

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def validate_config(self) -> bool:
        """
        Validate configuration integrity with enhanced checks.
        """
        if self.config_df is None:
            self.logger.error("❌ Config not loaded; call load_config() first.")
            return False

        # Required columns exist?
        required_columns = ['table_name', 'column_name', 'data_type']
        missing = [c for c in required_columns if c not in self.config_df.columns]
        if missing:
            self.logger.error(f"❌ Missing required columns: {missing}")
            return False

        # Validate base_data_type values
        valid_types_prefixes = ['N', 'DC', 'A', 'VA', 'D', 'DT', 'TS', 'NS', 'AN']
        for _, row in self.config_df.iterrows():
            base_type = row.get('base_data_type', '')
            if base_type and not any(base_type.startswith(v) for v in valid_types_prefixes):
                self.logger.warning(f"⚠️  Warning: Unknown data type format: {row['data_type']}")

            special_rules = row.get('special_rules')
            if special_rules:
                try:
                    raw_length = row.get('length')
                    max_length: Optional[int] = None
                    if raw_length is not None and not (isinstance(raw_length, float) and np.isnan(raw_length)):
                        max_length = int(raw_length)
                    self.helpers.validate_special_rule(
                        special_rules,
                        max_length=max_length,
                        is_pk=bool(row.get('is_pk', False)),
                    )
                except ValueError as exc:
                    print(
                        f"❌ Invalid special rule for {row['table_name']}.{row['column_name']}: {exc}"
                    )
                    return False

        # Validate relationships reference existing tables
        for rel in self.relationships:
            if rel.target_table not in self.tables:
                self.logger.error(f"❌ Reference table not found: {rel.target_table}")
                return False
            if rel.source_table not in self.tables:
                self.logger.error(f"❌ Source table not found: {rel.source_table}")
                return False

        self.logger.info("✅ Configuration validated successfully")
        return True