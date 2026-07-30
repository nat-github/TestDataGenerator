"""CDC commands: delta and SCD2 post-processing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List


from sdp.services.common import (
    create_output_directory,
    load_config_context,
    validate_config_file,
)
from sdp.services.generation import (
    _derive_changed_snapshot,
    _generate_snapshot,
    _strip_effective_date_columns,
)
from sdp.utils.parquet_post_processor import ParquetPostProcessor

logger = logging.getLogger(__name__)


def run_delta(args) -> int:
    if not validate_config_file(args.config):
        return 1
    if not _validate_snapshot_dirs(previous=args.previous, current=args.current):
        return 1
    if not create_output_directory(args.output):
        return 1

    parser = load_config_context(args.config)
    run_settings = dict(parser.run_settings)
    if args.partition_columns:
        run_settings["delta_partition_columns"] = ";".join(args.partition_columns)
    elif args.partition_column:
        run_settings["delta_partition_column"] = args.partition_column
    if args.partition_start_date:
        run_settings["delta_partition_start_date"] = args.partition_start_date

    processor = ParquetPostProcessor(parser.tables, run_settings)
    summary = processor.generate_delta(args.previous, args.current, args.output, args.tables)
    if not summary:
        logger.warning("No delta output was generated")
        return 0

    logger.info("\nDelta generation summary:")
    for table_name, metrics in summary.items():
        logger.info(f"  {table_name}: {metrics}")
    return 0

def _validate_snapshot_dirs(*, previous: str, current: str) -> bool:
    """Both snapshot directories must exist and hold parquet.

    Without this a typo'd --previous produced "no output was generated" and
    exit 0 — a CDC job that silently did nothing looks like a clean run,
    which is the worst possible outcome for a scheduled pipeline. A run that
    finds both snapshots and detects no changes still exits 0; that is a
    genuine no-op, not a mistake.
    """
    ok = True
    for label, raw in (("--previous", previous), ("--current", current)):
        path = Path(raw)
        if not path.is_dir():
            logger.error(f"{label} snapshot directory does not exist: {path}")
            ok = False
        elif not any(path.glob("*.parquet")):
            logger.error(f"{label} snapshot directory contains no parquet files: {path}")
            ok = False
    return ok


def run_scd2(args) -> int:
    if not validate_config_file(args.config):
        return 1
    if not getattr(args, "simulate", False) and not _validate_snapshot_dirs(
        previous=args.previous, current=args.current
    ):
        return 1
    if not create_output_directory(args.output):
        return 1

    parser = load_config_context(args.config)
    processor = ParquetPostProcessor(parser.tables, parser.run_settings)

    previous_dir = args.previous
    current_dir = args.current
    temp_dirs: List[Path] = []

    if getattr(args, "simulate", False):
        out = Path(args.output)
        previous_dir = str(out.parent / f"{out.name}_sim_v1")
        current_dir = str(out.parent / f"{out.name}_sim_v2")
        temp_dirs = [Path(previous_dir), Path(current_dir)]
        logger.info("Simulate mode: generating baseline snapshot (v1)...")
        _generate_snapshot(args.config, previous_dir, args.default_records, args.seed)
        logger.info("Simulate mode: deriving changed snapshot (v2)...")
        _derive_changed_snapshot(processor, previous_dir, current_dir,
                                 args.change_fraction, args.seed, args.tables, args.change_columns)
    else:
        if not previous_dir or not current_dir:
            logger.error("scd2 needs --previous and --current snapshot directories "
                         "(or use --simulate to generate them from --config).")
            return 1

    try:
        summary = processor.generate_scd2(
            previous_dir=previous_dir,
            current_dir=current_dir,
            output_dir=args.output,
            selected_tables=args.tables,
            effective_timestamp=args.effective_ts,
            previous_effective_timestamp=args.previous_effective_ts,
        )
    finally:
        if getattr(args, "simulate", False) and not getattr(args, "keep_snapshots", False):
            import shutil
            for d in temp_dirs:
                shutil.rmtree(d, ignore_errors=True)

    if not summary:
        logger.warning("No SCD2 output was generated")
        return 0

    logger.info("\nSCD2 generation summary:")
    for table_name, metrics in summary.items():
        logger.info(f"  {table_name}: {metrics}")
    if getattr(args, "simulate", False) and getattr(args, "keep_snapshots", False):
        logger.info(f"  (kept intermediate snapshots: {previous_dir}, {current_dir})")
    if getattr(args, "no_effective_dates", False):
        logger.info("\nRemoving effective_from_ts/effective_to_ts (version dates kept in *_crt_dts columns)...")
        _strip_effective_date_columns(args.output, args.tables)
    return 0
