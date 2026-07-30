"""Generation orchestration — the shared implementation.

``sdp.cli`` and ``sdp.sdk`` both call :func:`generate_dataset`; neither owns
the sequence. Previously this logic lived inside ``cli.run_generate`` and the
SDK reached it by building an argv list and invoking the CLI in-process,
which meant the SDK could never return more than an exit code.

:class:`GenerationRequest` deliberately mirrors the ``generate`` subcommand's
argparse destination names. Helpers here read their inputs with plain
attribute access, so an argparse ``Namespace`` and a ``GenerationRequest``
are interchangeable — that is what let this move happen without rewriting
every helper.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from sdp.generators.data_generator import DataGenerator
from sdp.services.common import (
    create_output_directory,
    validate_config_file,
    verify_export,
    _looks_like_date_column,
)
from sdp.utils.data_validator import DataValidator
from sdp.utils.parquet_post_processor import ParquetPostProcessor

logger = logging.getLogger(__name__)


def get_record_counts(generator: DataGenerator, args) -> Dict[str, int]:
    workbook_default = generator.config_parser.get_setting("default_records_per_table", 1000)
    default_records = args.default_records if args.default_records is not None else int(workbook_default)
    records_config = {
        table_name: max(
            1,
            int(default_records if args.default_records is not None else (table_config.num_rows or default_records)),
        )
        for table_name, table_config in generator.tables_config.items()
        if table_config.active
    }

    if args.records:
        for record_arg in args.records:
            if ":" not in record_arg:
                logger.warning(f"Invalid record format: {record_arg}")
                continue
            try:
                table_name, count = record_arg.split(":", 1)
                table_name = table_name.strip().lower()
                count = max(1, int(count))
                if table_name in records_config:
                    records_config[table_name] = count
                    logger.info(f"  {table_name}: {count} records (from command line)")
                else:
                    logger.warning(f"Table '{table_name}' not found in configuration")
            except ValueError:
                logger.warning(f"Invalid record format: {record_arg}")

    return records_config


def _run_relationship_inference(generator: DataGenerator, confidence: float) -> None:
    try:
        from sdp.llm.relationship_inferrer import RelationshipInferrer
        inferrer = RelationshipInferrer()
        new_rels = inferrer.infer(generator.tables_config, generator.relationships, min_confidence=confidence)
        if new_rels:
            generator.relationships.extend(new_rels)
            logger.info(f"LLM inferred {len(new_rels)} additional relationship(s)")
        else:
            logger.info("LLM found no additional relationships to add")
    except Exception as exc:
        logger.warning(f"LLM relationship inference skipped: {exc}")


def _apply_change(df: pd.DataFrame, col: str, idx) -> None:
    """Write a clearly-different value into df.loc[idx, col], matching dtype."""
    n = len(idx)
    if pd.api.types.is_datetime64_any_dtype(df[col]) or _looks_like_date_column(col):
        s = pd.to_datetime(df[col], errors="coerce")
        tz = getattr(getattr(s, "dt", None), "tz", None)
        fill = pd.Timestamp("2026-01-01", tz=tz) if tz is not None else pd.Timestamp("2026-01-01")
        base = s.loc[idx].fillna(fill)
        s.loc[idx] = base + pd.to_timedelta(range(1, n + 1), unit="D")
        df[col] = s
    elif pd.api.types.is_numeric_dtype(df[col]):
        df.loc[idx, col] = list(range(1, n + 1))
    else:
        df.loc[idx, col] = [f"SCD2_CHANGED_{i}" for i in range(n)]


def _read_versions_per_key(config_path: str) -> Dict[str, int]:
    """Read per-table `versions_per_key` from YAML/JSON. config_parser ignores it."""
    p = Path(config_path)
    suffix = p.suffix.lower()
    specs: Dict[str, int] = {}
    try:
        if suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            return {}
        for t in raw.get("tables", []) or []:
            name = t.get("name") or t.get("table_name")
            vpk = t.get("versions_per_key")
            if name and vpk:
                try:
                    specs[name] = int(vpk)
                except (TypeError, ValueError):
                    pass
    except (OSError, TypeError, AttributeError, ValueError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read versions_per_key from {config_path}: {exc}")
    return specs


def _expand_versions_from_config(config_path: str, output_dir: str, generator, seed) -> None:
    """Config-driven, schema-preserving versioning: repeat each business key
    1..N times (N = versions_per_key) and vary every column listed in
    scd2_tracked_columns. No columns are added or removed; FK integrity holds."""
    specs = _read_versions_per_key(config_path)
    if not specs:
        return
    import numpy as np
    rng = np.random.default_rng(seed if seed is not None else 7)
    out = Path(output_dir)
    logger.info("\nApplying config-driven versions_per_key (schema unchanged)...")
    for table, vmax in specs.items():
        if not vmax or vmax < 2:
            continue
        pq = out / f"{table}.parquet"
        if not pq.exists():
            continue
        tc = generator.tables_config.get(table)
        if tc is None:
            continue
        tracked = list(getattr(tc, "scd2_tracked_columns", []) or [])
        df = pd.read_parquet(pq)
        if df.empty or not tracked:
            logger.info(f"  versions: {table} skipped (no scd2_tracked_columns)")
            continue
        date_cols = [c for c in tracked
                     if c in df.columns and (pd.api.types.is_datetime64_any_dtype(df[c])
                                             or _looks_like_date_column(c))]
        attr_cols = [c for c in tracked if c in df.columns and c not in date_cols]
        if not date_cols and not attr_cols:
            logger.info(f"  versions: {table} skipped (tracked columns not in data)")
            continue
        counts = rng.integers(1, vmax + 1, size=len(df))
        rep_index = np.repeat(np.arange(len(df)), counts)
        expanded = df.iloc[rep_index].reset_index(drop=True)
        version_no = np.concatenate([np.arange(c) for c in counts])
        # Date columns: vectorised shift.
        if date_cols:
            jitter = rng.integers(0, 30, size=len(expanded))
            offset_days = np.where(version_no == 0, 0, version_no * 90 + jitter)
            offset = pd.to_timedelta(offset_days, unit="D")
            for dc in date_cols:
                base = pd.to_datetime(expanded[dc], errors="coerce")
                tz = getattr(getattr(base, "dt", None), "tz", None)
                fill = pd.Timestamp("2026-01-01", tz=tz) if tz is not None else pd.Timestamp("2026-01-01")
                expanded[dc] = base.fillna(fill) + offset
        # Attribute columns: per-key cycling.
        if attr_cols:
            col_cfg = {c.column_name: c for c in tc.columns}
            for ac in attr_cols:
                cc = col_cfg.get(ac)
                if cc is None:
                    continue
                try:
                    bv_list = generator.helpers.parse_business_values(getattr(cc, "business_values", None)) or []
                except (AttributeError, TypeError, ValueError) as exc:
                    logger.warning(
                        f"Could not parse business_values for {ac!r} ({exc}) — "
                        f"SCD2 version expansion will fall back to special rules"
                    )
                    bv_list = []
                special = getattr(cc, "special_rules", None)
                data_type = getattr(cc, "data_type", None)
                col_pos = expanded.columns.get_loc(ac)
                seen_for_key: set = set()
                shortfalls = 0
                for i in range(len(expanded)):
                    if version_no[i] == 0:
                        seen_for_key = {expanded.iat[i, col_pos]}
                        continue
                    new_val = None
                    if bv_list:
                        new_val = next((v for v in bv_list if v not in seen_for_key), None)
                        if new_val is None:
                            new_val = bv_list[(int(version_no[i]) - 1) % len(bv_list)]
                            shortfalls += 1
                    elif special:
                        for _ in range(10):
                            try:
                                cand = generator.helpers.generate_special_value(special, data_type, column_name=ac)
                            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                                logger.debug(
                                    "special rule %r failed for %s: %s", special, ac, exc,
                                )
                                cand = None
                                break
                            if cand is not None and cand not in seen_for_key:
                                new_val = cand
                                break
                        if new_val is None:
                            new_val = cand
                    if new_val is None:
                        new_val = int(version_no[i]) if pd.api.types.is_numeric_dtype(expanded[ac]) \
                                  else f"V{int(version_no[i])}"
                    seen_for_key.add(new_val)
                    expanded.iat[i, col_pos] = new_val
                if shortfalls:
                    logger.warning(f"  versions: {table}.{ac} has fewer business_values "
                                   f"({len(bv_list)}) than versions_per_key={vmax}; "
                                   f"{shortfalls} version(s) had to repeat a value")
        expanded.to_parquet(pq, index=False)
        repeats = int((counts > 1).sum())
        parts = []
        if date_cols:
            parts.append(f"dates vary in {date_cols}")
        if attr_cols:
            parts.append(f"attrs vary in {attr_cols}")
        logger.info(f"  versions: {table} {len(df)} -> {len(expanded)} rows "
                    f"({repeats} keys repeated, up to {vmax} each; " + "; ".join(parts) + ")")


def _read_delta_table_selection(config_path: str) -> Optional[List[str]]:
    """Read per-table `write_delta: true` flags from YAML/JSON. Returns list or
    None (None => caller falls back to converting all tables)."""
    p = Path(config_path)
    suffix = p.suffix.lower()
    selected: List[str] = []
    try:
        if suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            return None
        for t in raw.get("tables", []) or []:
            name = t.get("name") or t.get("table_name")
            if name and t.get("write_delta"):
                selected.append(name)
    except (OSError, TypeError, AttributeError, ValueError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read write_delta flags from {config_path}: {exc}")
        return None
    return selected if selected else None


def _read_delta_partition_overrides(config_path: str) -> Dict[str, str]:
    """Read per-table `delta_partition_col` overrides from YAML/JSON.

    Bypasses config_parser / config_models (same pattern as `write_delta` /
    `versions_per_key`). Returns `{table_name: partition_col}` for tables that
    specify it; tables without it fall back to the CLI default
    `--delta-partition-col`. Used when different source tables need different
    partition columns in the same run (e.g. BOOKING_TM for one, LOAD_DT for
    another).
    """
    p = Path(config_path)
    suffix = p.suffix.lower()
    overrides: Dict[str, str] = {}
    try:
        if suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            import json
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            return overrides
        for t in raw.get("tables", []) or []:
            name = t.get("name") or t.get("table_name")
            col = t.get("delta_partition_col")
            if name and col:
                overrides[name] = str(col)
    except (OSError, TypeError, AttributeError, ValueError, yaml.YAMLError) as exc:
        logger.warning(f"Could not read delta_partition_col overrides from {config_path}: {exc}")
        return {}
    return overrides


def _write_delta_outputs(output_dir: str, partition_col: str, partition_value,
                         selected_tables: Optional[List[str]] = None,
                         partition_overrides: Optional[Dict[str, str]] = None) -> None:
    """Convert selected <output>/<table>.parquet files into Delta tables at
    <output>/<table>/ partitioned by partition_col=partition_value (append mode).

    `partition_overrides` (optional): per-table `{name: col}` mapping for tables
    that need a different partition column than the global default. Tables
    absent from the mapping use `partition_col`.
    """
    try:
        from deltalake import write_deltalake
    except ImportError:
        raise ImportError("deltalake is required for --write-delta. Install: pip install deltalake")
    from datetime import date

    out = Path(output_dir)
    if not partition_value:
        partition_value = date.today().strftime("%Y%m%d")
    flat_parquets = sorted(p for p in out.glob("*.parquet") if p.is_file())
    if not flat_parquets:
        logger.warning("No parquet files to convert to Delta")
        return
    overrides = partition_overrides or {}
    selection_msg = f" for {len(selected_tables)} selected table(s)" if selected_tables else " (all tables)"
    logger.info(f"\nWriting Delta Lake tables (default partition {partition_col}={partition_value}"
                f"{', overrides for ' + str(len(overrides)) + ' table(s)' if overrides else ''})"
                f"{selection_msg} -> {out}/")
    converted = 0
    for pq in flat_parquets:
        table = pq.stem
        if selected_tables is not None and table not in selected_tables:
            logger.info(f"  Skip:  {table:30s} (no write_delta flag -> kept as flat parquet)")
            continue
        df = pd.read_parquet(pq)
        col = overrides.get(table, partition_col)
        if col in df.columns:
            logger.warning(f"  {table}: existing column '{col}' will be overwritten with the run's partition value")
        df[col] = str(partition_value)
        delta_path = out / table

        # Write first, delete the flat parquet only once the Delta write has
        # succeeded. Unlinking first meant a failed write left neither the
        # source nor the Delta table — the flat parquet was already gone.
        try:
            write_deltalake(str(delta_path), df, mode="append", partition_by=[col])
        except Exception as exc:
            logger.error(
                f"  {table}: Delta write failed ({type(exc).__name__}: {exc}) — "
                f"the flat parquet at {pq} was left in place"
            )
            raise

        pq.unlink()
        converted += 1
        logger.info(f"  Delta: {table:30s} {len(df):6d} rows -> "
                    f"{delta_path}/{col}={partition_value}/  (+commit in _delta_log/)")
    if selected_tables and converted == 0:
        logger.warning(f"  --write-delta requested but none of {selected_tables} matched any output parquet")


def _generate_snapshot(config_path: str, output_dir: str, default_records, seed) -> None:
    """Generate one full snapshot (all active tables) into output_dir. Used by scd2 --simulate."""
    generator = DataGenerator(config_path, seed=seed)
    if not generator.load_configuration():
        raise ValueError("Failed to load configuration for snapshot generation")
    generator.create_sdv_metadata()
    workbook_default = generator.config_parser.get_setting("default_records_per_table", 1000)
    base = default_records if default_records is not None else int(workbook_default)
    records_config = {
        name: max(1, int(base if default_records is not None else (cfg.num_rows or base)))
        for name, cfg in generator.tables_config.items() if cfg.active
    }
    if not generator.train_synthesizer():
        logger.warning("SDV training failed for snapshot - using fallback generation")
    data = generator.generate_data(records_config)
    if not data:
        raise ValueError("Snapshot generation produced no data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.export_to_parquet(output_dir)


def _derive_changed_snapshot(processor: ParquetPostProcessor, previous_dir: str, current_dir: str,
                             fraction: float, seed, selected_tables, change_columns) -> None:
    """Copy previous_dir -> current_dir, changing tracked columns on a fraction of rows."""
    import random as _random
    rng = _random.Random(seed if seed is not None else 7)
    prev = Path(previous_dir)
    cur = Path(current_dir)
    cur.mkdir(parents=True, exist_ok=True)
    for pq in sorted(prev.glob("*.parquet")):
        table = pq.stem
        df = pd.read_parquet(pq)
        tc = processor.tables_config.get(table)
        do_table = (not selected_tables) or (table in selected_tables)
        changed_cols = None
        if tc is not None and do_table and not df.empty:
            keys = set(processor._resolve_business_keys(tc))
            if change_columns:
                wanted = list(change_columns)
            elif getattr(tc, "scd2_tracked_columns", None):
                wanted = list(tc.scd2_tracked_columns)
            else:
                wanted = processor._resolve_scd2_tracked_columns(tc)[:1]
            targets = [c for c in wanted if c in df.columns and c not in keys]
            if targets:
                n = max(1, int(len(df) * fraction))
                idx = rng.sample(list(df.index), min(n, len(df)))
                for col in targets:
                    _apply_change(df, col, idx)
                changed_cols = ", ".join(targets)
                logger.info(f"  simulate: changed [{changed_cols}] on {len(idx)}/{len(df)} rows of {table}")
        if changed_cols is None:
            logger.info(f"  simulate: {table} copied unchanged (no change column resolved)")
        df.to_parquet(cur / pq.name, index=False)


def _strip_effective_date_columns(output_dir: str, selected_tables) -> None:
    """Remove effective_from_ts / effective_to_ts from the SCD2 output parquets."""
    drop = ["effective_from_ts", "effective_to_ts"]
    out = Path(output_dir)
    for pq in sorted(out.glob("*.parquet")):
        if selected_tables and pq.stem not in selected_tables:
            continue
        df = pd.read_parquet(pq)
        present = [c for c in drop if c in df.columns]
        if present:
            df.drop(columns=present).to_parquet(pq, index=False)
            logger.info(f"  dropped {present} from {pq.name}")


def _run_gx_validation(generator, *, output_dir: str, tolerance: float, verbose: bool) -> int:
    """Run GX validation against the generated Parquet output and print a summary."""
    try:
        from sdp.validators.gx_validator import (
            HAS_GX, validate_tables, format_report,
        )
    except ImportError as exc:
        logger.error(f"Could not import sdp.validators.gx_validator: {exc}")
        return 1
    if not HAS_GX:
        logger.error(
            "Great Expectations is not installed. Run: poetry install --extras gx"
        )
        return 1
    logger.info("\nRunning Great Expectations validation...")
    report = validate_tables(
        generator.tables_config,
        output_dir=output_dir,
        row_count_tolerance=tolerance,
    )
    # logger, not print: this runs inside the service layer now, so printing
    # would push the report onto stdout of every SDK, REST and MCP caller.
    logger.info(format_report(report, verbose=verbose))
    return 0 if report.success else 2


def _run_er_diagram(generator, args) -> None:
    try:
        from sdp.utils.er_diagram import ERDiagramGenerator
        er_output = getattr(args, "er_output", None) or args.output
        er_formats = getattr(args, "er_format", ["mermaid"])
        gen = ERDiagramGenerator(generator.tables_config, generator.relationships)
        written = gen.save(er_output, formats=er_formats)
        if written:
            logger.info("\nER diagram(s) saved:")
            for p in written:
                logger.info(f"  {p}")
            if any(str(p).endswith(".mmd") for p in written):
                logger.info("  Tip: open .mmd in VS Code (Mermaid extension) or paste into https://mermaid.live")
    except Exception as exc:
        logger.warning(f"ER diagram generation failed (non-fatal): {exc}")


def _run_upload(output_dir: str, uri: str) -> None:
    try:
        from sdp.utils.cloud_uploader import upload_output
        logger.info(f"\nUploading output to {uri} ...")
        paths = upload_output(output_dir, uri)
        logger.info(f"Upload complete: {len(paths)} file(s)")
    except ImportError as exc:
        logger.warning(f"Cloud upload skipped — missing dependency: {exc}")
    except Exception as exc:
        logger.error(f"Cloud upload failed: {exc}")


# ---------------------------------------------------------------------------
# Request / outcome
# ---------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    """Everything a generation run needs, with no argparse dependency.

    Field names match the ``generate`` subcommand's argparse destinations on
    purpose: the helpers above read them with plain attribute access, so an
    argparse ``Namespace`` and a ``GenerationRequest`` are interchangeable.
    That is what let the orchestration move here without rewriting every
    helper it calls.
    """
    config: str
    output: str = "output"

    default_records: Optional[int] = None
    records: Optional[Sequence[str]] = None
    seed: Optional[int] = None
    validate: bool = False

    stream: bool = False
    chunk_size: int = 100_000

    infer_relationships: bool = False
    method: str = "ml"
    llm_confidence: float = 0.7
    ml_confidence: float = 0.55
    feedback_store: Optional[str] = None

    er_diagram: bool = False
    er_format: Sequence[str] = field(default_factory=lambda: ["mermaid"])
    er_output: Optional[str] = None

    upload_to: Optional[str] = None

    validate_with_gx: bool = False
    gx_tolerance: float = 0.5
    gx_fail_on_error: bool = False

    write_delta: bool = False
    delta_partition_col: Optional[str] = None
    delta_partition_value: Optional[str] = None
    delta_tables: Optional[Sequence[str]] = None

    # Engine selection — see Synthesizer_Engines.md
    engine: Optional[str] = None
    engine_options: Optional[Dict[str, Any]] = None
    privacy_report_json: Optional[str] = None

    verbose: bool = False


@dataclass
class GenerationOutcome:
    """The result of a run.

    ``exit_code`` preserves the CLI contract; every other field is what the
    SDK previously could not reach, because its only channel back was a
    process return code.
    """
    exit_code: int
    output_dir: Path
    row_counts: Dict[str, int] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    engine_stats: Optional[Dict[str, Any]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    empty_tables: int = 0
    error: Optional[str] = None

    # Live frames, for callers that want them without a Parquet round-trip.
    frames: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def total_records(self) -> int:
        return sum(self.row_counts.values())


def _failure(message: str, output: str) -> GenerationOutcome:
    logger.error(message)
    return GenerationOutcome(exit_code=1, output_dir=Path(output), error=message)


# ---------------------------------------------------------------------------
# The orchestration
# ---------------------------------------------------------------------------


def generate_dataset(request: GenerationRequest) -> GenerationOutcome:
    """Run a generation: configure, train, generate, export, verify.

    Returns an outcome rather than raising, so the CLI maps it to an exit
    code and the SDK inspects it. Behaviour is identical to the previous
    ``cli.run_generate`` — this is that code, moved.
    """
    if not request.config:
        return _failure("--config is required (omit it only with --list-engines)",
                        request.output)
    if not validate_config_file(request.config):
        return _failure(f"Invalid config file: {request.config}", request.output)
    if not create_output_directory(request.output):
        return _failure(f"Could not create output directory: {request.output}",
                        request.output)

    seed = request.seed
    logger.info("Initializing SDV data generator...")

    generator = DataGenerator(
        request.config, seed=seed,
        engine=request.engine,
        engine_options=request.engine_options,
    )
    if not generator.load_configuration():
        return _failure("Failed to load configuration", request.output)

    if request.infer_relationships:
        _run_relationship_inference(generator, request.llm_confidence)

    logger.info("Creating SDV metadata...")
    generator.create_sdv_metadata()

    table_names = [name for name, cfg in generator.tables_config.items() if cfg.active]
    if not table_names:
        return _failure("No active tables found in configuration", request.output)

    logger.info(f"Tables detected: {len(table_names)}")
    for table_name in table_names:
        logger.info(f"  {table_name}")

    anchored = sorted(getattr(generator, "anchor_data", {}).keys())
    if anchored:
        logger.info(f"Anchored tables (loaded from real source data): {', '.join(anchored)}")

    records_config = get_record_counts(generator, request)
    logger.info("\nGeneration settings:")
    logger.info(f"  Config file: {request.config}")
    logger.info(f"  Output directory: {request.output}")
    logger.info(f"  Total tables to generate: {len(records_config)}")
    if seed is not None:
        logger.info(f"  Seed: {seed}")

    engine_name = generator.resolve_engine_name()
    logger.info(f"\nTraining synthesizer (engine: {engine_name})...")
    if generator.train_synthesizer():
        logger.info(f"Synthesizer trained successfully (engine: {engine_name})")
    elif engine_name == "rule-based":
        # Not a failure — this engine has no model by design.
        logger.info("Generating directly from config rules (no model fitted)")
    else:
        logger.warning(
            f"Synthesizer training failed (engine: {engine_name}) - using fallback generation"
        )

    logger.info("\nStarting data generation...")
    if request.stream and hasattr(generator, "generate_and_export_stream"):
        logger.info("Using streaming generation + incremental export")
        generator.generate_and_export_stream(
            records_config, output_dir=request.output, chunk_size=request.chunk_size,
        )
        total_file_records = verify_export(request.output)
        logger.info(
            f"Streaming export created parquet files with {total_file_records} total records"
        )
        return GenerationOutcome(
            exit_code=0, output_dir=Path(request.output),
            row_counts=dict(records_config),
        )
    if request.stream:
        logger.warning(
            "Streaming mode requested, but this generator build does not expose a "
            "dedicated streaming export method; falling back to standard generation"
        )

    data = generator.generate_data(records_config)
    if not data:
        return _failure("No data generated", request.output)

    stats = generator.engine_stats
    if stats:
        logger.info(
            f"\nEngine cost: {stats['engine']} — fit {stats['fit_seconds']}s, "
            f"sample {stats['sample_seconds']}s "
            f"({stats['fit_rows']} training rows, {stats['sampled_rows']} sampled)"
        )

    privacy = generator.privacy_report
    if privacy:
        _report_privacy(privacy, request.privacy_report_json)

    logger.info("\nValidating generated data...")
    empty_tables = 0
    row_counts: Dict[str, int] = {}
    for table_name, table_data in data.items():
        row_counts[table_name] = len(table_data)
        if table_data.empty:
            logger.warning(f"Table {table_name} is empty")
            empty_tables += 1
        else:
            logger.info(f"Table {table_name}: {len(table_data)} records")

    logger.info(f"\nExporting to {request.output}...")
    generator.export_to_parquet(request.output)
    generator.save_model_artifacts(
        generator.config_parser.get_setting("model_artifact_path", None)
    )
    _expand_versions_from_config(request.config, request.output, generator, seed)
    total_file_records = verify_export(request.output)

    report = generator.get_generation_report()
    logger.info("\nGeneration Report:")
    logger.info(f"  Total records: {report['total_records']:,}")
    logger.info(f"  File records verified: {total_file_records:,}")
    logger.info(f"  Relationships configured: {report['relationships_configured']}")
    logger.info(f"  Synthesizer fitted: {report['synthesizer_fitted']}")
    logger.info(f"  Empty tables: {empty_tables}")
    if report.get("seed") is not None:
        logger.info(f"  Seed: {report['seed']}")
    if report.get("generation_path_summary"):
        logger.info(f"  Generation paths: {report['generation_path_summary']}")

    validation: Optional[Dict[str, Any]] = None
    if request.validate:
        logger.info("\nRunning relationship validation...")
        validator = DataValidator()
        is_valid = validator.validate_relationships(data, generator.relationships)
        validation = dict(validator.get_validation_report())
        validation["is_valid"] = bool(is_valid)
        logger.info(f"  Valid relationships: {validation.get('valid_count', 0)}")
        logger.info(f"  Invalid relationships: {validation.get('invalid_count', 0)}")
        if not is_valid:
            logger.warning("Some relationship issues were found")

    exit_code = 0

    if request.er_diagram:
        _run_er_diagram(generator, request)

    if request.upload_to:
        _run_upload(request.output, request.upload_to)

    if request.validate_with_gx:
        gx_rc = _run_gx_validation(
            generator,
            output_dir=request.output,
            tolerance=request.gx_tolerance,
            verbose=request.verbose,
        )
        if gx_rc != 0 and request.gx_fail_on_error:
            exit_code = gx_rc

    # Kept LAST so GX and upload see flat parquet.
    if request.write_delta and exit_code == 0:
        selected = request.delta_tables or _read_delta_table_selection(request.config)
        partition_overrides = _read_delta_partition_overrides(request.config)
        _write_delta_outputs(
            request.output,
            partition_col=request.delta_partition_col,
            partition_value=request.delta_partition_value,
            selected_tables=selected,
            partition_overrides=partition_overrides,
        )

    logger.info(f"\nAll files saved to: {Path(request.output).absolute()}")
    return GenerationOutcome(
        exit_code=exit_code,
        output_dir=Path(request.output),
        row_counts=row_counts,
        report=dict(report),
        engine_stats=stats,
        privacy_report=privacy,
        validation=validation,
        empty_tables=empty_tables,
        frames=data,
    )


def _report_privacy(privacy: Dict[str, Any], report_path: Optional[str]) -> None:
    """Log the DP accounting and optionally persist it."""
    measured = privacy["measured_columns"]
    logger.info(
        f"\nPrivacy: ε={privacy['epsilon_requested']} per table — "
        f"{len(measured)} column(s) measured under DP, "
        f"{len(privacy['unmeasured_columns'])} generated from config only"
    )
    logger.info(f"  Guarantee: {privacy['guarantee']}")
    for warning in privacy["warnings"]:
        logger.warning(f"  ! {warning}")
    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(privacy, indent=2, default=str), encoding="utf-8")
        logger.info(f"  Wrote privacy report to {out}")
