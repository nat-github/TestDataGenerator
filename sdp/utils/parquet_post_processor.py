from __future__ import annotations

import json
import importlib
import logging
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import pyarrow as pa

from sdp.models.config_models import TableConfig


def _require_deltalake_writer():
    try:
        module = importlib.import_module("deltalake")
        return module.write_deltalake
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'deltalake' package is required for the delta command. "
            "Install project dependencies (for example: poetry install) and try again."
        ) from exc


class ParquetPostProcessor:
    def __init__(self, tables_config: Dict[str, TableConfig], run_settings: Optional[Dict[str, object]] = None):
        self.tables_config = tables_config
        self.run_settings = run_settings or {}
        self.logger = logging.getLogger(__name__)

    def generate_delta(
        self,
        previous_dir: str,
        current_dir: str,
        output_dir: str,
        selected_tables: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        previous_path = Path(previous_dir)
        current_path = Path(current_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tables = self._resolve_selected_tables(selected_tables, mode="delta")
        operation_column = str(self.run_settings.get("operation_column", "operation_type"))
        summary: Dict[str, Dict[str, int]] = {}

        for table_name in tables:
            previous_df = self._read_parquet(previous_path / f"{table_name}.parquet")
            current_df = self._read_parquet(current_path / f"{table_name}.parquet")

            if previous_df.empty and current_df.empty:
                self.logger.info(f"Skipping delta for {table_name}: no parquet found in either snapshot")
                continue

            delta_df = self._build_delta_frame(table_name, previous_df, current_df, operation_column)
            if delta_df.empty:
                self.logger.info(f"No delta detected for {table_name}")
                continue

            export_location = self._write_delta_table(table_name, delta_df, output_path)
            summary[table_name] = {
                "rows": len(delta_df),
                "inserts": int((delta_df[operation_column] == "I").sum()),
                "updates": int((delta_df[operation_column] == "U").sum()),
                "deletes": int((delta_df[operation_column] == "D").sum()),
            }
            self.logger.info(f"Generated delta parquet for {table_name} at {export_location}: {summary[table_name]}")

        return summary

    def generate_scd2(
        self,
        previous_dir: str,
        current_dir: str,
        output_dir: str,
        selected_tables: Optional[Sequence[str]] = None,
        effective_timestamp: Optional[str] = None,
        previous_effective_timestamp: Optional[str] = None,
    ) -> Dict[str, Dict[str, int]]:
        previous_path = Path(previous_dir)
        current_path = Path(current_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        current_effective_ts = self._parse_timestamp(effective_timestamp, default=datetime.now(timezone.utc))
        bootstrap_effective_ts = self._parse_timestamp(
            previous_effective_timestamp,
            default=current_effective_ts - timedelta(seconds=1),
        )
        open_end = pd.Timestamp("9999-12-31 00:00:00")

        tables = self._resolve_selected_tables(selected_tables, mode="scd2")
        summary: Dict[str, Dict[str, int]] = {}

        for table_name in tables:
            table_config = self.tables_config[table_name]
            current_snapshot = self._read_parquet(current_path / f"{table_name}.parquet")
            if current_snapshot.empty:
                self.logger.warning(f"Skipping SCD2 for {table_name}: current snapshot parquet missing or empty")
                continue

            previous_file = previous_path / f"{table_name}.parquet"
            previous_df = self._read_parquet(previous_file)
            history_df = self._bootstrap_history(previous_df, bootstrap_effective_ts, open_end)

            keys = self._resolve_business_keys(table_config)
            tracked_columns = self._resolve_scd2_tracked_columns(table_config)
            current_snapshot = self._deduplicate_snapshot(current_snapshot, keys, table_name, "current")
            current_flag_col = "is_current"
            effective_from_col = "effective_from_ts"
            effective_to_col = "effective_to_ts"
            version_col = "version_num"

            if current_flag_col not in history_df.columns:
                history_df[current_flag_col] = True
            if effective_from_col not in history_df.columns:
                history_df[effective_from_col] = bootstrap_effective_ts
            if effective_to_col not in history_df.columns:
                history_df[effective_to_col] = open_end
            if version_col not in history_df.columns:
                history_df[version_col] = 1

            active_history = history_df[history_df[current_flag_col].fillna(False)].copy()
            active_history = self._deduplicate_snapshot(active_history, keys, table_name, "history")
            active_history["_tracked_hash"] = self._row_hash(active_history, tracked_columns)
            current_snapshot["_tracked_hash"] = self._row_hash(current_snapshot, tracked_columns)

            left = active_history[keys + ["_tracked_hash", version_col]].rename(columns={"_tracked_hash": "_tracked_hash_prev"})
            right = current_snapshot[keys + ["_tracked_hash"]].rename(columns={"_tracked_hash": "_tracked_hash_curr"})
            merged = left.merge(right, on=keys, how="outer", indicator=True)

            updates = merged[(merged["_merge"] == "both") & (merged["_tracked_hash_prev"] != merged["_tracked_hash_curr"])][keys]
            deletes = merged[merged["_merge"] == "left_only"][keys]
            inserts = merged[merged["_merge"] == "right_only"][keys]

            expire_keys = pd.concat([updates, deletes], ignore_index=True).drop_duplicates() if not updates.empty or not deletes.empty else pd.DataFrame(columns=keys)
            if not expire_keys.empty:
                expire_mask = history_df[current_flag_col].fillna(False)
                expire_mask &= history_df[keys].astype("string").agg("|".join, axis=1).isin(
                    expire_keys.astype("string").agg("|".join, axis=1)
                )
                history_df.loc[expire_mask, current_flag_col] = False
                history_df.loc[expire_mask, effective_to_col] = current_effective_ts

            new_rows: List[pd.DataFrame] = []

            if not inserts.empty:
                inserted_rows = current_snapshot.merge(inserts, on=keys, how="inner")
                inserted_rows = inserted_rows.drop(columns=["_tracked_hash"], errors="ignore")
                inserted_rows[effective_from_col] = current_effective_ts
                inserted_rows[effective_to_col] = open_end
                inserted_rows[current_flag_col] = True
                inserted_rows[version_col] = 1
                new_rows.append(inserted_rows)

            if not updates.empty:
                updated_rows = current_snapshot.merge(updates, on=keys, how="inner")
                previous_versions = active_history.merge(updates, on=keys, how="inner")[keys + [version_col]]
                updated_rows = updated_rows.merge(previous_versions, on=keys, how="left", suffixes=("", "_previous"))
                updated_rows = updated_rows.drop(columns=["_tracked_hash"], errors="ignore")
                updated_rows[effective_from_col] = current_effective_ts
                updated_rows[effective_to_col] = open_end
                updated_rows[current_flag_col] = True
                updated_rows[version_col] = updated_rows[version_col].fillna(0).astype(int) + 1
                new_rows.append(updated_rows)

            if new_rows:
                history_df = pd.concat([history_df, *new_rows], ignore_index=True, sort=False)

            history_df = history_df.drop(columns=["_tracked_hash"], errors="ignore")
            export_file = output_path / f"{table_name}.parquet"
            history_df.to_parquet(export_file, index=False)

            summary[table_name] = {
                "rows": len(history_df),
                "new_versions": int(sum(len(frame) for frame in new_rows)),
                "expired_rows": int(len(expire_keys)),
            }
            self.logger.info(f"Generated SCD2 parquet for {table_name}: {summary[table_name]}")

        return summary

    def _resolve_selected_tables(self, selected_tables: Optional[Sequence[str]], mode: str) -> List[str]:
        selected = {table.strip().lower() for table in selected_tables or [] if table.strip()}
        tables: List[str] = []
        for table_name, table_config in self.tables_config.items():
            if selected and table_name not in selected:
                continue
            if not table_config.active:
                continue
            if mode == "delta" and not table_config.delta_eligible:
                continue
            if mode == "scd2" and not table_config.scd2_enabled:
                continue
            tables.append(table_name)
        return tables

    def _resolve_business_keys(self, table_config: TableConfig) -> List[str]:
        keys = list(table_config.business_key_columns or [])
        if not keys:
            keys = [column.column_name for column in table_config.columns if column.is_business_key_component]
        if not keys:
            keys = list(table_config.primary_key_columns or [])
        if not keys:
            keys = [column.column_name for column in table_config.columns if column.is_pk]
        if not keys:
            raise ValueError(f"Table {table_config.name} needs business keys or primary keys for delta/SCD2 processing")
        return keys

    def _resolve_scd2_tracked_columns(self, table_config: TableConfig) -> List[str]:
        tracked_columns = list(table_config.scd2_tracked_columns or [])
        if not tracked_columns:
            tracked_columns = [column.column_name for column in table_config.columns if column.scd2_tracked]
        if not tracked_columns:
            tracked_columns = [
                column.column_name
                for column in table_config.columns
                if column.column_name not in self._resolve_business_keys(table_config)
            ]
        return tracked_columns

    def _build_delta_frame(
        self,
        table_name: str,
        previous_df: pd.DataFrame,
        current_df: pd.DataFrame,
        operation_column: str,
    ) -> pd.DataFrame:
        table_config = self.tables_config[table_name]
        key_columns = self._resolve_business_keys(table_config)

        if previous_df.empty:
            delta_df = current_df.copy()
            delta_df[operation_column] = "I"
            return delta_df

        if current_df.empty:
            delta_df = previous_df.copy()
            delta_df[operation_column] = "D"
            return delta_df

        previous_df = self._deduplicate_snapshot(previous_df, key_columns, table_name, "previous")
        current_df = self._deduplicate_snapshot(current_df, key_columns, table_name, "current")

        previous_hash = previous_df[key_columns].copy()
        previous_hash["_row_hash_prev"] = self._row_hash(previous_df, [col for col in previous_df.columns if col not in key_columns])
        current_hash = current_df[key_columns].copy()
        current_hash["_row_hash_curr"] = self._row_hash(current_df, [col for col in current_df.columns if col not in key_columns])

        merged = previous_hash.merge(current_hash, on=key_columns, how="outer", indicator=True)
        inserts = merged[merged["_merge"] == "right_only"][key_columns]
        deletes = merged[merged["_merge"] == "left_only"][key_columns]
        updates = merged[
            (merged["_merge"] == "both") & (merged["_row_hash_prev"] != merged["_row_hash_curr"])
        ][key_columns]

        delta_frames: List[pd.DataFrame] = []
        if not inserts.empty:
            inserted_rows = current_df.merge(inserts, on=key_columns, how="inner")
            inserted_rows[operation_column] = "I"
            delta_frames.append(inserted_rows)
        if not updates.empty:
            updated_rows = current_df.merge(updates, on=key_columns, how="inner")
            updated_rows[operation_column] = "U"
            delta_frames.append(updated_rows)
        if not deletes.empty:
            deleted_rows = previous_df.merge(deletes, on=key_columns, how="inner")
            deleted_rows[operation_column] = "D"
            delta_frames.append(deleted_rows)

        if not delta_frames:
            return pd.DataFrame()

        return pd.concat(delta_frames, ignore_index=True, sort=False)

    def _write_delta_table(self, table_name: str, delta_df: pd.DataFrame, output_root: Path) -> Path:
        table_dir = output_root / table_name
        self._prepare_delta_target_directory(table_dir)

        export_df, partition_columns = self._prepare_delta_export_frame(table_name, delta_df, table_dir)
        table_dir.mkdir(parents=True, exist_ok=True)

        log_dir = table_dir / "_delta_log"
        backup_dir: Optional[Path] = None
        if log_dir.exists():
            backup_dir = table_dir / f"_delta_log_bak_{uuid.uuid4().hex[:8]}"
            shutil.copytree(str(log_dir), str(backup_dir))

        write_deltalake = _require_deltalake_writer()
        try:
            write_deltalake(
                str(table_dir),
                pa.Table.from_pandas(export_df, preserve_index=False),
                mode="overwrite",
                partition_by=partition_columns or None,
                schema_mode="overwrite",
            )
        except Exception:
            if backup_dir and backup_dir.exists():
                if log_dir.exists():
                    shutil.rmtree(str(log_dir))
                shutil.copytree(str(backup_dir), str(log_dir))
                self.logger.warning(f"Delta write failed for {table_name}; restored previous _delta_log from backup")
            raise
        finally:
            if backup_dir and backup_dir.exists():
                shutil.rmtree(str(backup_dir))

        return table_dir

    def _prepare_delta_export_frame(
        self,
        table_name: str,
        delta_df: pd.DataFrame,
        table_dir: Path,
    ) -> tuple[pd.DataFrame, List[str]]:
        export_df = delta_df.copy()
        partition_columns = self._resolve_delta_partition_columns(self.tables_config[table_name])
        missing_columns = [column for column in partition_columns if column not in export_df.columns]

        if missing_columns:
            if len(partition_columns) > 1:
                raise ValueError(
                    f"Table {table_name} is configured with partition columns {partition_columns}, but delta data is missing {missing_columns}"
                )
            partition_column = partition_columns[0]
            export_df[partition_column] = self._next_delta_partition_value(table_dir, partition_column)

        return export_df, partition_columns

    def _resolve_delta_partition_columns(self, table_config: TableConfig) -> List[str]:
        if table_config.partition_columns:
            return list(table_config.partition_columns)

        configured_columns = self.run_settings.get("delta_partition_columns")
        if configured_columns not in (None, ""):
            if isinstance(configured_columns, str):
                return [part.strip() for part in configured_columns.replace(",", ";").split(";") if part.strip()]
            if isinstance(configured_columns, Sequence):
                return [str(part).strip() for part in configured_columns if str(part).strip()]

        partition_column = str(self.run_settings.get("delta_partition_column", "edl_partition_date")).strip()
        return [partition_column or "edl_partition_date"]

    def _prepare_delta_target_directory(self, table_dir: Path) -> None:
        log_dir = table_dir / "_delta_log"
        if not log_dir.exists():
            return
        if self._is_real_delta_log_directory(log_dir):
            return

        backup_dir = table_dir / f"_delta_log_legacy_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        shutil.move(str(log_dir), str(backup_dir))
        self.logger.info(f"Backed up legacy custom delta log for {table_dir.name} to {backup_dir.name}")

    @staticmethod
    def _is_real_delta_log_directory(log_dir: Path) -> bool:
        first_log = next(iter(sorted(log_dir.glob("*.json"))), None)
        if first_log is None:
            return False

        try:
            with first_log.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    return any(key in payload for key in {"protocol", "metaData", "add", "remove", "commitInfo"})
        except Exception:
            return False

        return False

    def _next_delta_partition_value(self, table_dir: Path, partition_column: str) -> str:
        prefix = f"{partition_column}="
        existing_dates: List[datetime] = []

        if table_dir.exists():
            for child in table_dir.iterdir():
                if not child.is_dir() or not child.name.startswith(prefix):
                    continue
                raw_value = child.name[len(prefix):]
                try:
                    existing_dates.append(datetime.strptime(raw_value, "%Y%m%d"))
                except ValueError:
                    continue

        if existing_dates:
            return (max(existing_dates) + timedelta(days=1)).strftime("%Y%m%d")

        configured_start = self.run_settings.get("delta_partition_start_date")
        if configured_start not in (None, ""):
            timestamp = pd.Timestamp(configured_start)
            return timestamp.strftime("%Y%m%d")

        return datetime.now(timezone.utc).strftime("%Y%m%d")

    @staticmethod
    def _next_delta_commit_version(log_dir: Path) -> int:
        existing_versions: List[int] = []
        if log_dir.exists():
            for child in log_dir.glob("*.json"):
                try:
                    existing_versions.append(int(child.stem))
                except ValueError:
                    continue

        return (max(existing_versions) + 1) if existing_versions else 0

    @staticmethod
    def _row_hash(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
        columns = list(columns)
        if not columns:
            return pd.Series(["" for _ in range(len(df))], index=df.index)

        normalized = pd.DataFrame(index=df.index)
        for column in columns:
            series = df[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                normalized[column] = pd.to_datetime(series, errors="coerce").astype("string")
            else:
                normalized[column] = series.astype("string")
        normalized = normalized.fillna("<NULL>")
        return pd.util.hash_pandas_object(normalized, index=False).astype(str)

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    @staticmethod
    def _parse_timestamp(value: Optional[str], default: datetime) -> pd.Timestamp:
        timestamp = pd.Timestamp(value) if value else pd.Timestamp(default)
        if timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp

    def _bootstrap_history(
        self,
        previous_df: pd.DataFrame,
        bootstrap_effective_ts: pd.Timestamp,
        open_end: pd.Timestamp,
    ) -> pd.DataFrame:
        if previous_df.empty:
            return previous_df.copy()

        if {"effective_from_ts", "effective_to_ts", "is_current", "version_num"}.issubset(previous_df.columns):
            return previous_df.copy()

        history_df = previous_df.copy()
        history_df["effective_from_ts"] = bootstrap_effective_ts
        history_df["effective_to_ts"] = open_end
        history_df["is_current"] = True
        history_df["version_num"] = 1
        return history_df

    def _deduplicate_snapshot(
        self,
        df: pd.DataFrame,
        key_columns: Sequence[str],
        table_name: str,
        snapshot_label: str,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        duplicate_count = int(df.duplicated(subset=list(key_columns), keep="last").sum())
        if duplicate_count:
            self.logger.warning(
                f"Detected {duplicate_count} duplicate business-key rows in {snapshot_label} snapshot for {table_name}; keeping the last row per key"
            )
            return df.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)

        return df

