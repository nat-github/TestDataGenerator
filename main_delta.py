
#!/usr/bin/env python3
"""
SDV-Based Data Generator with Hive-Style Partitioned Output
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from generators.data_generator import DataGenerator
from utils.data_validator import DataValidator
import pandas as pd


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SDV-Based Test Data Generator with Partitioned Output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate partitioned data
  python main_delta.py --config config/data_config.xlsx --output output --add-partition region=EU
  # Multiple partitions
  python main_delta.py --config config/data_config.xlsx --output output --add-partition region=EU --add-partition edl_partition_date=20250101
'''
    )
    parser.add_argument('--config', required=True, help='Path to Excel configuration file')
    parser.add_argument('--output', default='output', help='Root output directory')
    parser.add_argument('--default-records', type=int, default=1000, help='Default records per table')
    parser.add_argument('--records', nargs='+', help='Table-specific records: table_name:count')
    parser.add_argument('--add-partition', action='append', default=None,
                        help='Add constant partition column(s) as name=value')
    parser.add_argument('--validate', action='store_true', help='Validate relationships after generation')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed logging')
    return parser.parse_args()


def _build_hive_subdir(assignments: List[str] | None) -> List[Tuple[str, str]]:
    """Convert partition assignments into (name, value) pairs"""
    pairs: List[Tuple[str, str]] = []
    if not assignments:
        return pairs
    for it in assignments:
        if '=' not in it:
            raise ValueError(f'Invalid assignment: {it}')
        name, val = it.split('=', 1)
        pairs.append((name.strip(), val.strip()))
    return pairs


def _effective_parquet_output(root: Path, add_partition: List[str] | None) -> Path:
    """Build nested partition directory path"""
    nested = root
    for name, val in _build_hive_subdir(add_partition):
        nested = nested / f"{name}={val}"
    return nested


def get_record_counts(table_names: List[str], args) -> Dict[str, int]:
    """Get record counts from arguments with validation"""
    records_config = {table: args.default_records for table in table_names}
    if args.records:
        for record_arg in args.records:
            if ':' in record_arg:
                try:
                    table_name, count = record_arg.split(':', 1)
                    table_name = table_name.strip().lower()
                    count = int(count)
                    if count <= 0:
                        print(f"⚠ Warning: Record count must be positive for {table_name}. Using default.")
                        count = args.default_records
                    if table_name in records_config:
                        records_config[table_name] = count
                        print(f"📊 {table_name}: {count} records (from command line)")
                    else:
                        print(f"⚠ Warning: Table '{table_name}' not found in configuration")
                except ValueError:
                    print(f"⚠ Warning: Invalid record format: {record_arg}")
    for table_name, count in records_config.items():
        if count <= 0:
            print(f"⚠ Warning: Invalid record count for {table_name}. Using minimum of 1.")
            records_config[table_name] = 1
    return records_config


def main():
    try:
        args = parse_arguments()
        print("🚀 SDV Test Data Generator (Partitioned Output)")
        print("=" * 50)

        # Validate configuration file
        config_file = Path(args.config)
        if not config_file.exists():
            print(f"❌ Configuration file not found: {config_file}")
            sys.exit(1)

        # Create root output directory
        output_root = Path(args.output)
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {output_root.absolute()}")

        # Initialize SDV generator
        print("🔄 Initializing SDV data generator...")
        generator = DataGenerator(args.config)

        # Load configuration
        if not generator.load_configuration():
            print("❌ Failed to load configuration")
            sys.exit(1)

        # Create SDV metadata
        print("📂 Creating SDV metadata...")
        generator.create_sdv_metadata()

        # Get table names
        table_names = list(generator.tables_config.keys())
        if not table_names:
            print("❌ No tables found in configuration")
            sys.exit(1)

        print(f"📂 Tables detected: {len(table_names)}")
        for table_name in table_names:
            print(f" 📊 {table_name}")

        # Get record counts
        records_config = get_record_counts(table_names, args)
        print(f"\n🎯 Generation settings:")
        print(f" Config file: {args.config}")
        print(f" Output root: {args.output}")
        print(f" Default records per table: {args.default_records}")
        print(f" Total tables: {len(table_names)}")

        # Train SDV synthesizer
        print("\n🤖 Training SDV synthesizer...")
        if generator.train_synthesizer():
            print("✅ SDV synthesizer trained successfully")
        else:
            print("⚠ SDV synthesizer training failed - using fallback generation")

        # Generate data
        print("\n🚀 Starting data generation...")
        data = generator.generate_data(records_config)
        if not data:
            print("❌ No data generated")
            sys.exit(1)

        # Verify datetime columns
        print("\n🔍 Verifying datetime columns...")
        for table_name, table_data in data.items():
            datetime_cols = [col for col in table_data.columns if pd.api.types.is_datetime64_any_dtype(table_data[col])]
            if datetime_cols:
                print(f" ✅ {table_name}: {len(datetime_cols)} datetime columns")
            else:
                print(f" ⚠ {table_name}: No datetime columns found")

        # Validate generated data
        print("\n🔍 Validating generated data...")
        empty_tables = sum(1 for df in data.values() if df.empty)
        if empty_tables > 0:
            print(f"⚠ Warning: {empty_tables} tables are empty")
        else:
            print(f"✅ Data validation passed: {sum(len(df) for df in data.values())} total records")

        # Build partitioned output path
        effective_out = _effective_parquet_output(output_root, args.add_partition)
        effective_out.mkdir(parents=True, exist_ok=True)

        # Export to Parquet in partitioned folder
        print(f"\n💾 Exporting to {effective_out}...")
        generator.export_to_parquet(str(effective_out))

        # Verify export
        parquet_files = list(effective_out.glob("*.parquet"))
        if parquet_files:
            print(f"✅ Export successful: {len(parquet_files)} Parquet files created")
        else:
            print("❌ No Parquet files were created")

        # Optional validation
        if args.validate:
            print("\n🔍 Running relationship validation...")
            validator = DataValidator()
            is_valid = validator.validate_relationships(data, generator.relationships)
            if is_valid:
                print("✅ All relationships validated successfully!")
            else:
                print("⚠ Some relationship issues found")

        # Final report
        report = generator.get_generation_report()
        print("\n📊 Generation Report:")
        print(f" Total records: {report['total_records']:,}")
        print(f" Relationships configured: {report['relationships_configured']}")
        print(f" Synthesizer fitted: {report['synthesizer_fitted']}")
        print(f" Status: {report.get('status', 'UNKNOWN')}")
        print(f"\n✅ All files saved to: {effective_out.absolute()}")

        print("\n🎉 Data generation completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
