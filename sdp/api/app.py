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

    return app


# Module-level app so `uvicorn sdp.api:app` works.
app = create_app()
