
#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import write_deltalake


def _concat_parquet_files(src_dir: Path) -> pa.Table:
    files = sorted(src_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found in: {src_dir}")

    tables = []
    for f in files:
        tbl = pq.read_table(str(f))
        match = re.search(r"edl_partition_date=(\d+)", str(f.parent))
        if not match:
            raise SystemExit(f"❌ Missing partition value in path: {f.parent}")
        partition_value = match.group(1)

        arrays = [*tbl.columns]
        names = [*tbl.schema.names]

        # Always add or overwrite partition column
        if "edl_partition_date" in names:
            idx = names.index("edl_partition_date")
            arrays[idx] = pa.array([partition_value] * tbl.num_rows)
        else:
            arrays.append(pa.array([partition_value] * tbl.num_rows))
            names.append("edl_partition_date")

        tbl = pa.Table.from_arrays(arrays, names)
        tables.append(tbl)

    return pa.concat_tables(tables)

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet to Delta")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    parser.add_argument("--partition-cols", nargs="*", default=["edl_partition_date"])
    args = parser.parse_args()

    src_dir = Path(args.source)
    target_root = Path(args.target)
    target_root.mkdir(parents=True, exist_ok=True)

    big_table = _concat_parquet_files(src_dir).combine_chunks()

    ONE_TERABYTE = 1024 * 1024 * 1024 * 1024

    write_deltalake(
        table_or_uri=str(target_root),
        data=big_table,
        mode=args.mode,
        partition_by=args.partition_cols,
        target_file_size=ONE_TERABYTE
    )

    print(f"✅ Delta table written successfully at: {target_root}")

if __name__ == "__main__":
    main()