"""Shared CLI/service helpers.

Small utilities used by more than one command surface: config validation,
output-directory creation, export verification, logging setup. They live
here rather than in ``cli.py`` so the service layer can use them without
importing the CLI — which is what made the SDK a CLI wrapper.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sdp.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


def load_config_context(config_path: str) -> ConfigParser:
    """Load, parse and validate a config, raising on any failure.

    Lives here rather than in ``cli.py`` because the command modules need
    it and ``cli.py`` imports *them* — the other direction would be a
    circular import.
    """
    parser = ConfigParser(config_path)
    if not parser.load_config():
        raise ValueError("Failed to load configuration")
    parser.parse_tables()
    parser.parse_relationships()
    if not parser.validate_config():
        raise ValueError("Configuration validation failed")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")


def create_output_directory(output_dir: str) -> bool:
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path.absolute()}")
        return True
    except Exception as exc:
        logger.error(f"Error creating output directory: {exc}")
        return False


def validate_config_file(config_path: str) -> bool:
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return False
    if not config_file.is_file():
        logger.error(f"Configuration path is not a file: {config_file}")
        return False
    if config_file.suffix.lower() not in [".xlsx", ".xls", ".yaml", ".yml"]:
        logger.warning(f"Configuration file may not be Excel or YAML format: {config_file}")
    logger.info(f"Configuration file: {config_file.absolute()}")
    return True


def verify_export(output_dir: str) -> int:
    output_path = Path(output_dir)
    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files were found in {output_path}")

    total_file_records = 0
    logger.info(f"Export successful: {len(parquet_files)} parquet files created")
    for file in parquet_files:
        file_data = pd.read_parquet(file)
        file_records = len(file_data)
        total_file_records += file_records
        logger.info(f"  {file.name}: {file_records} records ({file.stat().st_size} bytes)")
    return total_file_records


def _looks_like_date_column(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith("_dts") or "dts" in n or "date" in n
