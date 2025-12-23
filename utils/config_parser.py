import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from models.config_models import ColumnConfig, TableConfig, RelationshipConfig


class ConfigParser:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config_df: Optional[pd.DataFrame] = None
        self.tables: Dict[str, TableConfig] = {}
        self.relationships: List[RelationshipConfig] = []

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
                print(f"✅ Enhanced DC detected: {s} -> precision={precision}, scale={scale}")
                return base, None, precision, scale
            # Other enhanced types use 'length' (A/N/VA/AN/NS)
            print(f"✅ Enhanced type detected: {s} -> base={base}, length={p1}")
            return base, p1, None, None

        # 2) Legacy compact format without parentheses, e.g., A34, N19, VA256
        legacy_pattern = r'^([A-Z]+)(\d+)$'
        m = re.match(legacy_pattern, s)
        if m:
            base = m.group(1)
            length = int(m.group(2))
            print(f"✅ Legacy type detected: {s} -> base={base}, length={length}")
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
            print(f"✅ Simple type detected: {s} -> length={length}")
            return s, length, None, None

        # 4) Fallback: return as-is (treated as string type elsewhere)
        print(f"⚠️  Unknown data type format, treating as string: {s}")
        return s, None, None, None

    # ---------------------------------------------------------------------
    # Load & normalize
    # ---------------------------------------------------------------------
    def load_config(self) -> bool:
        """
        Load and clean configuration from Excel with enhanced parsing and NaN→None normalization.
        """
        try:
            # Read the 'Columns' sheet
            self.config_df = pd.read_excel(self.config_file, sheet_name='Columns')

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
                    # leave numeric values unchanged; convert NaN to None for downstream logic
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
                # keep None as None, lowercase valid strings
                self.config_df['ref_table'] = self.config_df['ref_table'].apply(
                    lambda v: v.lower().strip() if isinstance(v, str) else v
                )

            print(f"✅ Loaded and cleaned configuration with {len(self.config_df)} columns")
            print(f"📊 Detected base data types: {self.config_df['base_data_type'].unique()}")
            return True

        except Exception as e:
            print(f"❌ Error loading config: {e}")
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
            print("❌ Config not loaded; call load_config() first.")
            return self.tables

        for table_name in self.config_df['table_name'].dropna().unique():
            table_data = self.config_df[self.config_df['table_name'] == table_name]
            columns: List[ColumnConfig] = []

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
                    scale=row.get('scale')
                )
                columns.append(column_config)

            self.tables[table_name] = TableConfig(name=table_name, columns=columns)

        return self.tables

    def parse_relationships(self) -> List[RelationshipConfig]:
        """
        Automatically extract relationships from FK definitions.
        """
        self.relationships = []

        if self.config_df is None:
            print("❌ Config not loaded; call load_config() first.")
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

        print(f"✅ Automatically identified {len(self.relationships)} relationships from FK definitions")
        return self.relationships

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def validate_config(self) -> bool:
        """
        Validate configuration integrity with enhanced checks.
        """
        if self.config_df is None:
            print("❌ Config not loaded; call load_config() first.")
            return False

        # Required columns exist?
        required_columns = ['table_name', 'column_name', 'data_type']
        missing = [c for c in required_columns if c not in self.config_df.columns]
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return False

        # Validate base_data_type values
        valid_types_prefixes = ['N', 'DC', 'A', 'VA', 'D', 'DT', 'TS', 'NS', 'AN']
        for _, row in self.config_df.iterrows():
            base_type = row.get('base_data_type', '')
            if base_type and not any(base_type.startswith(v) for v in valid_types_prefixes):
                print(f"⚠️  Warning: Unknown data type format: {row['data_type']}")

        # Validate relationships reference existing tables
        for rel in self.relationships:
            if rel.target_table not in self.tables:
                print(f"❌ Reference table not found: {rel.target_table}")
                return False

        print("✅ Configuration validated successfully")
        return True