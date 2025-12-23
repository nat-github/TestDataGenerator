import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, List, Optional


class ParquetDuckDBRunner:
    def __init__(self, parquet_path: str):
        """
        SQL runner for Parquet files using ONLY DuckDB (no SQLite)

        Args:
            parquet_path: Path to directory containing Parquet files or specific file
        """
        self.parquet_path = Path(parquet_path)
        self.conn = duckdb.connect()  # In-memory DuckDB connection
        self.loaded_tables = {}

    def load_parquet_files(self) -> Dict[str, str]:
        """
        Register Parquet files as tables in DuckDB
        Returns dictionary with table names
        """
        if self.parquet_path.is_file():
            files = [self.parquet_path]
        else:
            files = list(self.parquet_path.glob("*.parquet"))

        print(f"📁 Found {len(files)} Parquet file(s)")

        for file_path in files:
            try:
                table_name = file_path.stem.lower()

                # Register Parquet file directly with DuckDB (no pandas needed)
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT * FROM read_parquet('{file_path}')
                """)

                # Get record count
                count_result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

                # Get column info
                col_result = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
                columns = list(col_result['column_name'])

                self.loaded_tables[table_name] = {
                    'records': count_result,
                    'columns': columns,
                    'file_path': str(file_path)
                }

                print(f"✅ Registered {table_name}: {count_result} records")
                print(f" 📋 Columns: {columns}")

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        return self.loaded_tables

    def run_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query using DuckDB

        Args:
            sql_query: SQL query string

        Returns:
            DataFrame with query results
        """
        if not self.loaded_tables:
            self.load_parquet_files()

        print(f"\n🚀 Executing SQL Query:")
        print(f"```sql\n{sql_query}\n```")

        try:
            # Execute query directly in DuckDB
            result = self.conn.execute(sql_query).fetchdf()

            print(f"✅ Query executed successfully")
            print(f"📊 Result: {len(result)} records")

            return result

        except Exception as e:
            print(f"❌ SQL Error: {e}")
            raise

    def get_table_info(self) -> Dict[str, Dict]:
        """Get information about all loaded tables"""
        return self.loaded_tables

    def show_tables(self):
        """Show all available tables"""
        if not self.loaded_tables:
            self.load_parquet_files()

        print("\n📊 AVAILABLE TABLES:")
        for table_name, info in self.loaded_tables.items():
            print(f" 🏷️ {table_name}")
            print(f" 📁 File: {info['file_path']}")
            print(f" 📊 Records: {info['records']}")
            print(f" 📋 Columns: {', '.join(info['columns'])}")
            print()

    def close(self):
        """Close DuckDB connection"""
        self.conn.close()


# One-liner function for quick usage
def query_parquet(parquet_path: str, sql_query: str) -> pd.DataFrame:
    """
    One-liner function to run SQL on Parquet files using DuckDB

    Args:
        parquet_path: Path to Parquet file or directory
        sql_query: SQL query to execute

    Returns:
        DataFrame with results
    """
    runner = ParquetDuckDBRunner(parquet_path)
    result = runner.run_sql(sql_query)
    runner.close()
    return result


# Example usage
if __name__ == "__main__":
    # Install duckdb if needed
    try:
        import duckdb
    except ImportError:
        import subprocess

        subprocess.check_call(["pip", "install", "duckdb"])

    # Initialize runner
    runner = ParquetDuckDBRunner("output/test_orgdeltapart")  # Change to your path

    # Load tables
    runner.load_parquet_files()

    # Show available tables
    runner.show_tables()

    # EXAMPLE QUERIES - All supported by DuckDB

    queries = [
        # Basic SELECT with WHERE
        #      """
         #select INCOMINGINSTRUCTIONKEY, COUNT(*) as count_per_code
         #from payment_transaction
         #GROUP BY INCOMINGINSTRUCTIONKEY
         #HAVING COUNT(*) > 1
          #"""
        """
        select distinct edl_act_dts from cdf_ggm_org_hist
        """
    ]

    # Run all example queries
    for i, sql_query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"🧪 EXAMPLE QUERY {i}")
        print(f"{'=' * 60}")

        try:
            result = runner.run_sql(sql_query)
            if len(result) > 0:
                print(f"📄 Results:")
                print(result.to_string(index=False, max_rows=10))
        except Exception as e:
            print(f"⚠ Query might fail if tables don't exist: {e}")

    # Close connection
    runner.close()