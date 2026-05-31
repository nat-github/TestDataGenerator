"""FastAPI application for the Synthetic Data Platform.

Endpoints are intentionally stateless: a config file is uploaded with each
request, processed in an isolated temp directory, and the result streamed back.
Nothing is persisted server-side between requests.
"""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - guidance for the install
    raise ImportError(
        "The REST API requires the 'api' extra. Install it with:\n"
        "    pip install 'synthetic-data-platform[api]'\n"
        "    # or, from source:  poetry install --extras api"
    ) from exc

from sdp import __version__
from sdp.sdk import SDPError, SyntheticDataPlatform

# Config file extensions ConfigParser knows how to dispatch on.
_ALLOWED_CONFIG_SUFFIXES = {".xlsx", ".xls", ".yaml", ".yml", ".json"}
_PREVIEW_ROWS = 20


def _save_upload(upload: "UploadFile", contents: bytes, workdir: Path) -> Path:
    """Persist an uploaded config, preserving its suffix for format dispatch."""
    suffix = Path(upload.filename or "config").suffix.lower()
    if suffix not in _ALLOWED_CONFIG_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported config type {suffix!r}. "
            f"Allowed: {sorted(_ALLOWED_CONFIG_SUFFIXES)}",
        )
    cfg_path = workdir / f"config{suffix}"
    cfg_path.write_bytes(contents)
    return cfg_path


def _unzip_upload(upload: "UploadFile", contents: bytes, target_dir: Path) -> Path:
    """Extract an uploaded ZIP (typically a snapshot from /generate) into target_dir.

    Returns the directory the caller should pass to delta/scd2 — if the ZIP
    contains a single top-level folder, the extracted folder itself is returned;
    otherwise target_dir is returned (parquets sit directly inside it).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{upload.filename or 'upload'} is not a valid ZIP: {exc}",
        )
    children = [p for p in target_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return target_dir


def _zip_directory(directory: Path) -> io.BytesIO:
    """Zip every file under ``directory`` into an in-memory buffer."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(directory)))
    buffer.seek(0)
    return buffer


def _parse_records(records: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse a ``"table:count,table:count"`` form field into a dict."""
    if not records:
        return None
    parsed: Dict[str, int] = {}
    for chunk in records.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid records entry {chunk!r}; expected 'table:count'",
            )
        table, _, count = chunk.partition(":")
        try:
            parsed[table.strip()] = int(count)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid record count in {chunk!r}; expected an integer",
            )
    return parsed or None


def _extract_data_zip(upload: "UploadFile", contents: bytes, workdir: Path) -> Path:
    """Extract an uploaded ZIP of Parquet files; return the dir that holds them."""
    if not (upload.filename or "").lower().endswith(".zip"):
        raise HTTPException(
            status_code=415, detail="data must be a .zip of <table>.parquet files"
        )
    zip_path = workdir / "data.zip"
    zip_path.write_bytes(contents)
    extract_dir = workdir / "data"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=422, detail="data zip is corrupt or not a ZIP file")
    if list(extract_dir.glob("*.parquet")):
        return extract_dir
    for nested in sorted(extract_dir.rglob("*.parquet")):
        return nested.parent          # parquet files nested one level down
    raise HTTPException(status_code=422, detail="no .parquet files found in the data zip")


def create_app() -> "FastAPI":
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Synthetic Data Platform API",
        version=__version__,
        description=(
            "HTTP interface for relationship-aware synthetic data generation. "
            "Upload an Excel/YAML/JSON config and receive generated Parquet data."
        ),
    )
    platform = SyntheticDataPlatform(log_level="WARNING")

    @app.get("/", include_in_schema=False)
    def root() -> Dict[str, str]:
        return {
            "name": "Synthetic Data Platform API",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/healthz", tags=["meta"], summary="Liveness/version probe")
    def healthz() -> Dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/generate", tags=["data"], summary="Generate synthetic data")
    async def generate(
        config: UploadFile = File(..., description="Excel/YAML/JSON config file"),
        seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
        default_records: Optional[int] = Form(None, description="Default rows per table"),
        records: Optional[str] = Form(
            None, description="Per-table overrides, e.g. 'orders:500,customers:50'"
        ),
        validate_relationships: bool = Form(
            False,
            alias="validate",
            description="Validate FK relationships after generation",
        ),
        infer_relationships: bool = Form(False, description="Infer missing FK relationships"),
        method: str = Form("ml", description="Inference engine: ml | llm | both"),
        response_format: str = Form(
            "zip", description="'zip' (Parquet download) or 'json' (row preview)"
        ),
        # --- Delta Lake direct write (additive; default = off) ---
        write_delta: bool = Form(False, description="Write each table as a Delta Lake table at <output>/<table>/"),
        delta_partition_col: Optional[str] = Form(None, description="Partition column name (default: BOOKING_TM)"),
        delta_partition_value: Optional[str] = Form(None, description="Partition value for THIS run (default: today YYYYMMDD)"),
        delta_tables: Optional[str] = Form(None, description="Comma-separated list of tables to write as Delta (overrides per-table `write_delta: true` flags in the config)"),
    ):
        """Generate synthetic Parquet data from an uploaded config.

        Returns a ZIP of the generated Parquet files (``response_format=zip``)
        or a JSON preview of the first rows of each table (``json``).
        """
        contents = await config.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_gen_"))
        try:
            cfg_path = _save_upload(config, contents, workdir)
            out_dir = workdir / "output"

            try:
                result = await run_in_threadpool(
                    platform.generate,
                    config=cfg_path,
                    output=out_dir,
                    seed=seed,
                    default_records=default_records,
                    records=_parse_records(records),
                    validate=validate_relationships,
                    infer_relationships=infer_relationships,
                    method=method,
                    write_delta=write_delta,
                    delta_partition_col=delta_partition_col,
                    delta_partition_value=delta_partition_value,
                    delta_tables=[t.strip() for t in delta_tables.split(",") if t.strip()] if delta_tables else None,
                )
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed (exit code {result.exit_code}). "
                    "Check the config — try POST /lint.",
                )

            if response_format == "json":
                preview = {
                    name: {
                        "records": int(len(frame)),
                        "columns": list(frame.columns),
                        # Round-trip through pandas' JSON writer so NaN -> null,
                        # timestamps -> ISO strings, etc. — std json can't encode those.
                        "sample": json.loads(
                            frame.head(_PREVIEW_ROWS).to_json(
                                orient="records", date_format="iso"
                            )
                        ),
                    }
                    for name, frame in result.frames.items()
                }
                return JSONResponse(
                    {
                        "tables": result.tables,
                        "total_records": result.total_records,
                        "preview": preview,
                    }
                )

            if response_format != "zip":
                raise HTTPException(
                    status_code=422,
                    detail="response_format must be 'zip' or 'json'",
                )

            buffer = _zip_directory(out_dir)
            return StreamingResponse(
                buffer,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="generated.zip"'},
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post("/scd2", tags=["data"], summary="Build SCD2 history (with --simulate, no prior snapshots needed)")
    async def scd2(
        config: UploadFile = File(..., description="Excel/YAML/JSON config file"),
        simulate: bool = Form(False, description="Generate baseline + changed snapshot internally and diff (no previous/current needed)"),
        default_records: Optional[int] = Form(None, description="With simulate: rows per table for the baseline"),
        seed: Optional[int] = Form(None, description="With simulate: random seed"),
        change_fraction: Optional[float] = Form(None, description="With simulate: fraction of rows that change between v1 and v2 (default 0.3)"),
        change_columns: Optional[str] = Form(None, description="With simulate: comma-separated tracked columns to change (default: scd2_tracked_columns from config)"),
        no_effective_dates: bool = Form(False, description="Drop effective_from_ts/effective_to_ts from output (version dates stay in *_crt_dts columns)"),
        effective_ts: Optional[str] = Form(None, description="Effective timestamp for current snapshot rows"),
        previous_effective_ts: Optional[str] = Form(None, description="Bootstrap effective timestamp for previous snapshot rows"),
        tables: Optional[str] = Form(None, description="Comma-separated list of tables to process"),
    ):
        """Build SCD Type 2 history from a config — supports `simulate` mode where
        the two snapshots are generated internally so the caller only needs to
        upload the config."""
        contents = await config.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_scd2_"))
        try:
            cfg_path = _save_upload(config, contents, workdir)
            out_dir = workdir / "output"
            try:
                result = await run_in_threadpool(
                    platform.scd2,
                    config=cfg_path,
                    output=out_dir,
                    simulate=simulate,
                    default_records=default_records,
                    seed=seed,
                    change_fraction=change_fraction,
                    change_columns=[c.strip() for c in change_columns.split(",") if c.strip()] if change_columns else None,
                    no_effective_dates=no_effective_dates,
                    effective_ts=effective_ts,
                    previous_effective_ts=previous_effective_ts,
                    tables=[t.strip() for t in tables.split(",") if t.strip()] if tables else None,
                )
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"SCD2 generation failed (exit code {result.exit_code}).",
                )
            buffer = _zip_directory(out_dir)
            return StreamingResponse(
                buffer,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="scd2.zip"'},
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post("/delta", tags=["data"], summary="Compute a CDC delta between two snapshots")
    async def delta(
        config: UploadFile = File(..., description="Excel/YAML/JSON config file"),
        previous: UploadFile = File(..., description="ZIP of the previous snapshot's parquet files"),
        current: UploadFile = File(..., description="ZIP of the current snapshot's parquet files"),
        tables: Optional[str] = Form(None, description="Comma-separated list of tables to process"),
        partition_column: Optional[str] = Form(None, description="Override delta partition column for all tables"),
        partition_columns: Optional[str] = Form(None, description="Comma-separated override of delta partition columns"),
        partition_start_date: Optional[str] = Form(None, description="Override synthetic delta partition start date (YYYYMMDD or timestamp)"),
    ):
        """Compute a CDC delta between two uploaded snapshot ZIPs and return the
        Delta Lake output as a ZIP. Snapshot ZIPs typically come from /generate
        (its zip response is a drop-in input here)."""
        cfg_contents = await config.read()
        prev_contents = await previous.read()
        cur_contents = await current.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_delta_"))
        try:
            cfg_path = _save_upload(config, cfg_contents, workdir)
            prev_dir = _unzip_upload(previous, prev_contents, workdir / "previous")
            cur_dir = _unzip_upload(current, cur_contents, workdir / "current")
            out_dir = workdir / "output"
            try:
                result = await run_in_threadpool(
                    platform.delta,
                    config=cfg_path,
                    previous=prev_dir,
                    current=cur_dir,
                    output=out_dir,
                    tables=[t.strip() for t in tables.split(",") if t.strip()] if tables else None,
                    partition_column=partition_column,
                    partition_columns=[c.strip() for c in partition_columns.split(",") if c.strip()] if partition_columns else None,
                    partition_start_date=partition_start_date,
                )
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Delta generation failed (exit code {result.exit_code}).",
                )
            buffer = _zip_directory(out_dir)
            return StreamingResponse(
                buffer,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="delta.zip"'},
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post("/lint", tags=["data"], summary="Validate a config")
    async def lint(
        config: UploadFile = File(..., description="Excel/YAML/JSON config file"),
    ):
        """Validate a config and return structured issues — no data generated."""
        contents = await config.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_lint_"))
        try:
            cfg_path = _save_upload(config, contents, workdir)
            try:
                result = await run_in_threadpool(platform.lint, cfg_path)
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            return {
                "ok": result.ok,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "issues": result.issues,
                "report": result.report,
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post(
        "/infer-relationships",
        tags=["data"],
        summary="Infer foreign-key relationships",
    )
    async def infer_relationships(
        config: UploadFile = File(..., description="Excel/YAML/JSON config file"),
        method: str = Form("ml", description="Inference engine: ml | llm | both"),
        ml_mode: str = Form("standard", description="ML mode: standard | knowledge-graph"),
    ):
        """Infer missing FK relationships; returns the reviewable YAML config."""
        contents = await config.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_infer_"))
        try:
            cfg_path = _save_upload(config, contents, workdir)
            out_yaml = workdir / "inferred.yaml"
            try:
                result = await run_in_threadpool(
                    platform.infer_relationships,
                    config=cfg_path,
                    config_output=out_yaml,
                    method=method,
                    ml_mode=ml_mode,
                )
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

            if not result.success or not out_yaml.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Inference failed (exit code {result.exit_code}).",
                )
            return {"config_yaml": out_yaml.read_text(encoding="utf-8")}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post("/contract-test", tags=["contracts"], summary="Verify data against a contract")
    async def contract_test(
        contract: UploadFile = File(..., description="Contract config (Excel/YAML/JSON)"),
        data: UploadFile = File(..., description="ZIP of <table>.parquet files to verify"),
        tolerance: float = Form(0.5, description="Permitted row-count deviation (0.5 = ±50%)"),
    ):
        """Verify a dataset against a data contract — severity-tagged checks + verdict."""
        contract_bytes = await contract.read()
        data_bytes = await data.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_contract_"))
        try:
            cfg_path = _save_upload(contract, contract_bytes, workdir)
            data_dir = _extract_data_zip(data, data_bytes, workdir)
            try:
                report = await run_in_threadpool(
                    platform.contract_test,
                    contract=cfg_path,
                    data=data_dir,
                    tolerance=tolerance,
                    contract_name=contract.filename or "data-contract",
                )
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            return report.to_dict()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @app.post("/contract-diff", tags=["contracts"], summary="Detect breaking contract changes")
    async def contract_diff(
        old: UploadFile = File(..., description="Previous contract config"),
        new: UploadFile = File(..., description="New contract config"),
    ):
        """Compare two contract versions — returns breaking / additive / review changes."""
        old_bytes = await old.read()
        new_bytes = await new.read()
        workdir = Path(tempfile.mkdtemp(prefix="sdp_api_cdiff_"))
        try:
            old_dir = workdir / "old"
            new_dir = workdir / "new"
            old_dir.mkdir()
            new_dir.mkdir()
            old_path = _save_upload(old, old_bytes, old_dir)
            new_path = _save_upload(new, new_bytes, new_dir)
            try:
                diff = await run_in_threadpool(platform.contract_diff, old_path, new_path)
            except SDPError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            return diff.to_dict()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return app


# Module-level app so `uvicorn sdp.api:app` works.
app = create_app()
