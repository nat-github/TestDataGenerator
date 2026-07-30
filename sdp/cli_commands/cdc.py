"""CDC commands: delta and SCD2 post-processing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from sdp.generators.data_generator import DataGenerator
from sdp.services.common import (
    configure_logging,
    create_output_directory,
    load_config_context,
    validate_config_file,
    verify_export,
)
from sdp.services.generation import (
    _derive_changed_snapshot,
    _generate_snapshot,
    _strip_effective_date_columns,
)
from sdp.utils.config_parser import ConfigParser
from sdp.utils.data_validator import DataValidator
from sdp.utils.parquet_post_processor import ParquetPostProcessor

logger = logging.getLogger(__name__)


def run_delta(args) -> int:
    if not validate_config_file(args.config):
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

def run_scd2(args) -> int:
    if not validate_config_file(args.config):
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
