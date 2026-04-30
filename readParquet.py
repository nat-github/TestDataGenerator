from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

from deltalake import DeltaTable
from pyspark.sql import DataFrame, SparkSession


DEFAULT_TABLE_ROOT = "output/account_booking_delta/df_cac_acg_entr"


@dataclass
class TableReadResult:
	table_root: Path
	table_name: str
	df: DataFrame
	storage_format: str
	partition_columns: List[str]
	delta_version: int | None = None


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Read a delta/parquet table root or an entire delta output root and print row counts locally. "
			"If a real Delta Lake table is present, the active files are resolved from the Delta transaction log."
		)
	)
	parser.add_argument(
		"path",
		nargs="?",
		default=DEFAULT_TABLE_ROOT,
		help=(
			"Path to a table root such as output/account_booking_delta/df_cac_acg_entr, "
			"or a delta output root such as output/account_booking_delta"
		),
	)
	parser.add_argument("--show", type=int, default=20, help="Number of rows to display per discovered table")
	parser.add_argument(
		"--partition-column",
		help="Optional partition column to use for partition counts instead of auto-detection",
	)
	return parser


def create_spark() -> SparkSession:
	return (
		SparkSession.builder.appName("ReadDeltaRootCount")
		.master("local[*]")
		.config("spark.sql.session.timeZone", "UTC")
		.getOrCreate()
	)


def is_table_root(path: Path) -> bool:
	if not path.exists() or not path.is_dir():
		return False
	if (path / "_delta_log").is_dir():
		return True
	return any(child.is_dir() and "=" in child.name for child in path.iterdir())


def discover_table_roots(path: Path) -> list[Path]:
	if not path.exists():
		raise FileNotFoundError(f"Path does not exist: {path}")
	if not path.is_dir():
		raise NotADirectoryError(f"Expected a directory, got: {path}")
	if is_table_root(path):
		return [path]

	children = sorted(child for child in path.iterdir() if child.is_dir() and is_table_root(child))
	if not children:
		raise FileNotFoundError(f"No delta/parquet table roots were found under: {path}")
	return children


def read_real_delta_table(spark: SparkSession, table_root: Path) -> TableReadResult:
	delta_table = DeltaTable(str(table_root))
	active_files = delta_table.file_uris()
	if not active_files:
		raise FileNotFoundError(f"No active Delta files were found for: {table_root}")

	metadata = delta_table.metadata()
	partition_columns = list(getattr(metadata, "partition_columns", []) or [])
	df = spark.read.option("basePath", str(table_root)).parquet(*active_files)
	return TableReadResult(
		table_root=table_root,
		table_name=table_root.name,
		df=df,
		storage_format="delta",
		partition_columns=partition_columns,
		delta_version=delta_table.version(),
	)


def read_partitioned_parquet_table(spark: SparkSession, table_root: Path) -> TableReadResult:
	parquet_dirs = sorted(child for child in table_root.iterdir() if child.is_dir() and "=" in child.name)
	if not parquet_dirs:
		raise FileNotFoundError(f"No partition directories were found under: {table_root}")

	partition_columns = sorted({child.name.split("=", 1)[0] for child in parquet_dirs})
	df = spark.read.parquet(*[str(path) for path in parquet_dirs])
	return TableReadResult(
		table_root=table_root,
		table_name=table_root.name,
		df=df,
		storage_format="parquet",
		partition_columns=partition_columns,
	)


def read_table_root(spark: SparkSession, table_root: Path) -> TableReadResult:
	try:
		return read_real_delta_table(spark, table_root)
	except Exception:
		return read_partitioned_parquet_table(spark, table_root)


def resolve_partition_column(result: TableReadResult, requested_partition_column: str | None) -> str | None:
	if requested_partition_column:
		return requested_partition_column if requested_partition_column in result.df.columns else None

	for column in result.partition_columns:
		if column in result.df.columns:
			return column

	for column in result.df.columns:
		if column.endswith("_date") or column.endswith("_dt"):
			return column

	return None


def print_table_summary(result: TableReadResult, show_rows: int, requested_partition_column: str | None) -> int:
	total_count = result.df.count()
	format_suffix = f", version={result.delta_version}" if result.delta_version is not None else ""
	print(f"Table: {result.table_name} ({result.storage_format}{format_suffix})")
	print(f"  Root: {result.table_root}")
	print(f"  Total rows: {total_count}")

	partition_column = resolve_partition_column(result, requested_partition_column)
	if requested_partition_column and partition_column is None:
		print(f"  Requested partition column '{requested_partition_column}' was not found in this table")
	elif partition_column:
		print(f"  Rows by partition ({partition_column}):")
		result.df.groupBy(partition_column).count().orderBy(partition_column).show(truncate=False)
	else:
		print("  No partition column available for grouping")

	if show_rows > 0:
		print(f"  Showing first {show_rows} rows:")
		result.df.show(show_rows, truncate=False)

	return total_count


def main() -> int:
	args = build_parser().parse_args()
	requested_path = Path(args.path).expanduser().resolve()
	table_roots = discover_table_roots(requested_path)

	spark = create_spark()
	try:
		grand_total = 0
		for table_root in table_roots:
			result = read_table_root(spark, table_root)
			grand_total += print_table_summary(result, args.show, args.partition_column)
			print()

		if len(table_roots) > 1:
			print(f"Delta root: {requested_path}")
			print(f"Tables processed: {len(table_roots)}")
			print(f"Grand total rows: {grand_total}")
		return 0
	finally:
		spark.stop()


if __name__ == "__main__":
	raise SystemExit(main())

