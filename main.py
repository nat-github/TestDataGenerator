#!/usr/bin/env python3
"""
Main SDV-based Data Generator
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
from generators.data_generator import DataGenerator
from utils.data_validator import DataValidator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SDV-Based Test Data Generator with Relationship Preservation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python main.py --config config/data_config_SEPADD.xlsx

  # Specific output and records
  python main.py --config config/data_config_SEPADD.xlsx --output output/test_run --records payment_instruction:5000

  # With validation
  python main.py --config config/data_config_SEPADD.xlsx --validate

  # Small test run
  python main.py --config config/data_config_SEPADD.xlsx --output output/test_small --default-records 100
        '''
    )

    parser.add_argument('--config', required=True, help='Path to Excel configuration file')
    parser.add_argument('--output', default='output', help='Output directory for Parquet files')
    parser.add_argument('--default-records', type=int, default=1000, help='Default records per table')
    parser.add_argument('--records', nargs='+', help='Table-specific records: table_name:count')
    parser.add_argument('--validate', action='store_true', help='Validate relationships after generation')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed logging')

    return parser.parse_args()


def create_output_directory(output_dir: str) -> bool:
    """Create output directory if it doesn't exist"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {output_path.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Error creating output directory: {e}")
        return False


def validate_config_file(config_path: str) -> bool:
    """Validate that config file exists"""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        return False

    if not config_file.is_file():
        print(f"❌ Configuration path is not a file: {config_file}")
        return False

    if config_file.suffix.lower() not in ['.xlsx', '.xls']:
        print(f"⚠ Warning: Configuration file may not be Excel format: {config_file}")

    print(f"✅ Configuration file: {config_file.absolute()}")
    return True


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

                    # Validate count is positive
                    if count <= 0:
                        print(
                            f"⚠ Warning: Record count must be positive for {table_name}. Using default: {args.default_records}")
                        count = args.default_records

                    if table_name in records_config:
                        records_config[table_name] = count
                        print(f"   📊 {table_name}: {count} records (from command line)")
                    else:
                        print(f"⚠ Warning: Table '{table_name}' not found in configuration")
                except ValueError:
                    print(f"⚠ Warning: Invalid record format: {record_arg}")

    # Final validation - ensure all counts are positive
    for table_name, count in records_config.items():
        if count <= 0:
            print(f"⚠ Warning: Invalid record count for {table_name}: {count}. Using minimum of 1.")
            records_config[table_name] = 1

    return records_config


def main():
    """Main function"""
    try:
        # Parse command line arguments
        args = parse_arguments()

        print("🚀 SDV Test Data Generator")
        print("=" * 50)

        # Validate configuration file
        if not validate_config_file(args.config):
            sys.exit(1)

        # Create output directory
        if not create_output_directory(args.output):
            sys.exit(1)

        # Initialize SDV generator
        print("🔄 Initializing SDV data generator...")
        generator = DataGenerator(args.config)

        # Load configuration
        if not generator.load_configuration():
            print("❌ Failed to load configuration")
            sys.exit(1)

        # Create SDV metadata
        print("📋 Creating SDV metadata...")
        generator.create_sdv_metadata()

        # Get table names
        table_names = list(generator.tables_config.keys())

        if not table_names:
            print("❌ No tables found in configuration")
            sys.exit(1)

        print(f"📋 Tables detected: {len(table_names)}")
        for table_name in table_names:
            print(f"   📊 {table_name}")

        # Get record counts
        records_config = get_record_counts(table_names, args)

        print(f"\n🎯 Generation settings:")
        print(f"   Config file: {args.config}")
        print(f"   Output directory: {args.output}")
        print(f"   Default records per table: {args.default_records}")
        print(f"   Total tables to generate: {len(table_names)}")

        # Try to train SDV synthesizer (but don't fail if it doesn't work)
        print("\n🤖 Training SDV synthesizer...")
        if generator.train_synthesizer():
            print("✅ SDV synthesizer trained successfully")
        else:
            print("⚠ SDV synthesizer training failed - using fallback generation")

        # Generate data
        print("\n🚀 Starting data generation...")
        data = generator.generate_data(records_config)

        # EMERGENCY: Double-check datetime columns
        print("\n🔍 Verifying datetime columns...")
        for table_name, table_data in data.items():
            datetime_cols = [col for col in table_data.columns
                             if pd.api.types.is_datetime64_any_dtype(table_data[col])]
            if datetime_cols:
                print(f"   ✅ {table_name}: {len(datetime_cols)} datetime columns")
                for col in datetime_cols:
                    print(f"      {col}: {table_data[col].dtype}")
            else:
                print(f"   ⚠ {table_name}: No datetime columns found - this may be a problem")

        # Validate generated data
        print("\n🔍 Validating generated data...")
        if not data:
            print("❌ No data generated - dictionaries are empty")
            sys.exit(1)

        # Check each table
        empty_tables = 0
        for table_name, table_data in data.items():
            if table_data.empty:
                print(f"❌ Table {table_name} is empty")
                empty_tables += 1
            else:
                print(f"✅ Table {table_name}: {len(table_data)} records")

        total_records = sum(len(df) for df in data.values())
        if total_records == 0:
            print("❌ CRITICAL: All tables are empty - no data generated")
            sys.exit(1)

        if empty_tables > 0:
            print(f"⚠ Warning: {empty_tables} tables are empty, but {total_records} total records generated")
        else:
            print(f"📊 Data validation passed: {total_records} total records")

        # Export to Parquet
        print(f"\n💾 Exporting to {args.output}...")
        generator.export_to_parquet(args.output)

        # Verify export worked
        output_path = Path(args.output)
        parquet_files = list(output_path.glob("*.parquet"))
        if parquet_files:
            print(f"✅ Export successful: {len(parquet_files)} Parquet files created")
            total_file_records = 0
            for file in parquet_files:
                # Read the file to verify it contains data
                try:
                    file_data = pd.read_parquet(file)
                    file_records = len(file_data)
                    total_file_records += file_records
                    file_size = file.stat().st_size
                    print(f"   📁 {file.name}: {file_records} records ({file_size} bytes)")
                except Exception as e:
                    print(f"   ❌ Error reading {file}: {e}")

            print(f"📦 Total records in files: {total_file_records}")
        else:
            print("❌ CRITICAL: No Parquet files were created")
            # Check what files exist in output directory
            all_files = list(output_path.glob("*"))
            if all_files:
                print(f"   Other files in directory: {[f.name for f in all_files]}")
            else:
                print("   Directory is completely empty")
            sys.exit(1)

        print("\n🎉 Data generation completed successfully!")
        print("=" * 50)
        print(f"📦 Generated data for {len(data)} tables")

        # Show summary from generator report
        report = generator.get_generation_report()
        print(f"📈 Total records generated: {report['total_records']:,}")

        for table_name, count in report['table_record_counts'].items():
            print(f"   {table_name}: {count:,} records")

        # Run validation if requested
        if args.validate:
            print("\n" + "=" * 50)
            print("🔍 Running relationship validation...")

            validator = DataValidator()
            is_valid = validator.validate_relationships(data, generator.relationships)

            if is_valid:
                print("✅ All relationships validated successfully!")
            else:
                print("⚠ Some relationship issues found - check validation report")

            # Show validation summary
            val_report = validator.get_validation_report()
            print(f"   Valid relationships: {val_report.get('valid_count', 0)}")
            print(f"   Invalid relationships: {val_report.get('invalid_count', 0)}")

        # Generate final report
        print(f"\n📊 Generation Report:")
        print(f"   Total records: {report['total_records']:,}")
        print(f"   Relationships configured: {report['relationships_configured']}")
        print(f"   Synthesizer fitted: {report['synthesizer_fitted']}")
        print(f"   Status: {report.get('status', 'UNKNOWN')}")

        print(f"\n✅ All files saved to: {Path(args.output).absolute()}")

        # Final verification
        if report['total_records'] == 0:
            print("\n❌ WARNING: Report shows 0 records despite successful export!")
            print("   This indicates a reporting issue, but files should contain data.")
        else:
            print(f"\n🎊 SUCCESS: Generated and exported {report['total_records']:,} records!")

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Import pandas for file verification
import pandas as pd

if __name__ == "__main__":
    main()
