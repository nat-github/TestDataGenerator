"""Synthetic Data Platform — Streamlit UI v1.

A simple, intuitive interface for the data-generation track. Sized for
≤ 10,000 rows total — for larger runs, point users at the CLI.

Run from the repo root:
    poetry install --extras ui
    poetry run streamlit run sdp/ui/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable when running `streamlit run sdp/ui/...`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "configs"
MAX_ROWS_PER_TABLE_UI = 10_000
PREVIEW_ROWS = 100

CLI_HINT = (
    "For runs above 10,000 rows per table, please use the CLI:\n"
    "    `python main.py generate --config <path> --output <dir> --default-records N`"
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Synthetic Data Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Synthetic Data Platform")
st.caption(
    "Generate realistic, relationship-aware test data from Excel / YAML / "
    "JSON configs. UI cap: 10,000 rows per table. "
    "→ Open **API Mocks** in the sidebar for the OpenAPI / Postman / HAR → "
    "WireMock / Pact / Postman flow."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def discover_bundled_examples() -> List[Path]:
    """Find every YAML / JSON / XLSX example shipped with the platform."""
    out: List[Path] = []
    for ext in ("yaml", "json", "xlsx"):
        out.extend(sorted((EXAMPLES_DIR / ext).glob(f"*.{ext}")))
    return out


def write_uploaded_to_tempdir(uploaded, suffix: str) -> Path:
    """Write a Streamlit UploadedFile to disk so the existing CLI code can read it."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="sdp_upload_"))
    target = tmp_dir / f"config{suffix}"
    target.write_bytes(uploaded.getvalue())
    return target


def lint_config(path: Path) -> Dict[str, Any]:
    """Validate a config without generating data — return a structured report."""
    from sdp.utils.config_parser import ConfigParser
    parser = ConfigParser(str(path))
    report: Dict[str, Any] = {"ok": False, "errors": [], "tables": [], "relationships": 0}
    try:
        if not parser.load_config():
            report["errors"].append("Failed to load config")
            return report
        tables = parser.parse_tables()
        rels = parser.parse_relationships()
        report["ok"] = True
        report["tables"] = [
            {"name": name, "columns": len(cfg.columns), "rows": cfg.num_rows or 0}
            for name, cfg in tables.items()
        ]
        report["relationships"] = len(rels) if rels else 0
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def run_generation(config_path: Path, output_dir: Path, *, default_records: int, seed: int):
    """Run the generator in-process and return (per_table_dataframes, elapsed_seconds)."""
    from sdp.generators.data_generator import DataGenerator
    import pandas as pd
    import pyarrow.parquet as pq

    gen = DataGenerator(str(config_path), seed=seed)
    if not gen.load_configuration():
        raise ValueError(f"Failed to load configuration from {config_path}")

    # Honour the platform's existing API. records_per_table is per-table count.
    records_config = {
        name: min(default_records, MAX_ROWS_PER_TABLE_UI)
        for name in gen.tables_config
        if gen.tables_config[name].active
    }

    started = time.perf_counter()
    gen.create_sdv_metadata()
    gen.train_synthesizer(sample_size=min(default_records, 200))
    gen.generate_data(records_per_table=records_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    gen.export_to_parquet(str(output_dir))
    elapsed = time.perf_counter() - started

    # Read the generated Parquet files back so we can preview
    dataframes: Dict[str, "pd.DataFrame"] = {}
    for parquet_file in sorted(output_dir.glob("*.parquet")):
        dataframes[parquet_file.stem] = pq.read_table(parquet_file).to_pandas()
    return dataframes, elapsed


def zip_output_dir(output_dir: Path) -> bytes:
    """Bundle every parquet file in output_dir into an in-memory zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                zf.write(f, arcname=f.name)
    buf.seek(0)
    return buf.getvalue()


def _render_quality_table(t):
    """Render one TableQualityMetrics inline."""
    import pandas as pd

    metrics_md = [
        f"**Synthetic rows:** {t.row_count_synthetic:,}",
    ]
    if t.row_count_source is not None:
        metrics_md.append(f"**Source rows:** {t.row_count_source:,}")
    if t.fidelity_score is not None:
        metrics_md.append(f"**Fidelity score:** {t.fidelity_score:.3f}")
    if t.correlation_distance is not None:
        metrics_md.append(f"**Correlation distance:** {t.correlation_distance:.3f}")
    if t.privacy_nn_too_close_rate is not None:
        rate = t.privacy_nn_too_close_rate
        flag = " ⚠️" if rate > 0.05 else ""
        metrics_md.append(f"**Privacy NN too-close rate:** {rate:.1%}{flag}")
    st.markdown("&nbsp;&nbsp;|&nbsp;&nbsp;".join(metrics_md))

    if t.notes:
        for note in t.notes:
            st.caption(f"_Note: {note}_")

    # Per-column metrics table
    rows = []
    for c in t.columns:
        rows.append({
            "column": c.column,
            "dtype": c.dtype,
            "null_rate": c.null_rate,
            "unique": c.unique_count,
            "mean": c.mean,
            "std": c.std,
            "min": c.min,
            "max": c.max,
            "ks_stat": c.ks_statistic,
            "tv_dist": c.tv_distance,
            "score": c.distribution_score,
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Top-values bar chart for the first categorical column with values
    cat_with_values = [c for c in t.columns if not c.is_numeric and c.top_values]
    if cat_with_values:
        first_cat = cat_with_values[0]
        st.caption(f"Top values for `{first_cat.column}`")
        chart_df = pd.DataFrame(first_cat.top_values, columns=["value", "count"]).set_index("value")
        st.bar_chart(chart_df)


# ---------------------------------------------------------------------------
# Sidebar — provider info + links
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Synthetic Data Platform** generates realistic Parquet data from "
        "Excel / YAML / JSON configs.\n\n"
        "This UI runs the same generator the CLI does, in-process. "
        "For larger runs, batches, CDC/SCD2, mocks, and LLM features, "
        "use the CLI."
    )
    st.divider()
    st.markdown("**Docs**")
    st.markdown(
        "- [Examples_Walkthrough.md](Examples_Walkthrough.md) — demo playbook\n"
        "- [Yaml_Config_Schema.md](Yaml_Config_Schema.md)\n"
        "- [ML_Relationship_Inference.md](ML_Relationship_Inference.md)\n"
        "- [LLM_Ecosystem.md](LLM_Ecosystem.md)\n"
        "- [Bruno_Workflow.md](Bruno_Workflow.md) — API mocks\n"
    )


# ---------------------------------------------------------------------------
# Step 1 — Choose a config
# ---------------------------------------------------------------------------

st.subheader("1. Pick a config")

config_source = st.radio(
    "Source",
    ["Use bundled example", "Upload my own"],
    horizontal=True,
    label_visibility="collapsed",
)

config_path: Optional[Path] = None

if config_source == "Use bundled example":
    bundled = discover_bundled_examples()
    if not bundled:
        st.warning("No bundled examples found under `examples/configs/`.")
    else:
        labels = [f"{p.parent.name}/{p.name}" for p in bundled]
        choice = st.selectbox(
            "Bundled examples",
            labels,
            index=0,
            help="11 examples ship with the platform. Each has a YAML, JSON, and XLSX variant where applicable.",
        )
        config_path = bundled[labels.index(choice)]
else:
    uploaded = st.file_uploader(
        "Upload an Excel, YAML, or JSON config",
        type=["xlsx", "xls", "yaml", "yml", "json"],
    )
    if uploaded is not None:
        config_path = write_uploaded_to_tempdir(uploaded, suffix=Path(uploaded.name).suffix)
        st.success(f"Loaded `{uploaded.name}` ({uploaded.size:,} bytes)")


# ---------------------------------------------------------------------------
# Step 2 — Settings
# ---------------------------------------------------------------------------

st.subheader("2. Settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)
with settings_col1:
    default_records = st.number_input(
        "Default records per table",
        min_value=1,
        max_value=MAX_ROWS_PER_TABLE_UI,
        value=200,
        step=100,
        help=CLI_HINT,
    )
with settings_col2:
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2**31 - 1,
        value=42,
        help="Same seed + same config → byte-identical output.",
    )
with settings_col3:
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    st.caption(f"Hard cap: **{MAX_ROWS_PER_TABLE_UI:,} rows per table**")


# ---------------------------------------------------------------------------
# Step 3 — Actions
# ---------------------------------------------------------------------------

st.subheader("3. Actions")

action_col1, action_col2, _ = st.columns([1, 1, 3])
with action_col1:
    generate_clicked = st.button("Generate", type="primary", use_container_width=True, disabled=config_path is None)
with action_col2:
    lint_clicked = st.button("Lint config", use_container_width=True, disabled=config_path is None)

# ---------------------------------------------------------------------------
# Lint output
# ---------------------------------------------------------------------------

if lint_clicked and config_path is not None:
    with st.spinner("Validating…"):
        report = lint_config(config_path)
    if report["ok"]:
        st.success("Config is valid.")
        st.markdown(
            "| Table | Columns | Configured rows |\n"
            "|---|---:|---:|\n"
            + "\n".join(
                f"| {t['name']} | {t['columns']} | {t['rows']} |"
                for t in report["tables"]
            )
        )
        st.caption(f"Relationships declared: {report['relationships']}")
    else:
        st.error("Config has errors:")
        for err in report["errors"]:
            st.code(err, language="text")


# ---------------------------------------------------------------------------
# Generate output
# ---------------------------------------------------------------------------

if generate_clicked and config_path is not None:
    with st.spinner(f"Generating up to {default_records:,} rows per table…"):
        run_dir = Path(tempfile.mkdtemp(prefix="sdp_run_"))
        try:
            dataframes, elapsed = run_generation(
                config_path,
                run_dir,
                default_records=int(default_records),
                seed=int(seed),
            )
            st.session_state["last_dataframes"] = dataframes
            st.session_state["last_run_dir"] = str(run_dir)
            st.session_state["last_elapsed"] = elapsed
        except Exception as exc:
            st.session_state["last_error"] = (
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )
            st.session_state["last_dataframes"] = None


error_text = st.session_state.get("last_error")
if error_text and not generate_clicked:
    # only show stale errors if user hasn't kicked off another run
    error_text = None

if error_text:
    st.error("Generation failed.")
    with st.expander("Stack trace"):
        st.code(error_text, language="text")

dataframes = st.session_state.get("last_dataframes")
elapsed = st.session_state.get("last_elapsed")
run_dir_str = st.session_state.get("last_run_dir")

if dataframes:
    total_rows = sum(len(df) for df in dataframes.values())
    st.success(
        f"Generated **{len(dataframes)} table(s)**, **{total_rows:,} row(s)** total in **{elapsed:.2f}s**."
    )

    # --- preview ---
    st.subheader("4. Preview")
    if len(dataframes) == 1:
        only_name, only_df = next(iter(dataframes.items()))
        st.caption(f"Showing first {min(PREVIEW_ROWS, len(only_df))} of {len(only_df):,} rows in `{only_name}`")
        st.dataframe(only_df.head(PREVIEW_ROWS), use_container_width=True)
    else:
        tabs = st.tabs(list(dataframes.keys()))
        for tab, (name, df) in zip(tabs, dataframes.items()):
            with tab:
                st.caption(f"Showing first {min(PREVIEW_ROWS, len(df))} of {len(df):,} rows")
                st.dataframe(df.head(PREVIEW_ROWS), use_container_width=True)

    # --- quality report ---
    st.subheader("5. Quality report")
    st.caption(
        "Statistical fidelity, distribution checks, and a privacy proxy. "
        "Optionally upload **source** Parquet/CSV files to compare against; without source, "
        "you'll get univariate stats + a correlation matrix."
    )

    qcol1, qcol2 = st.columns([2, 3])
    with qcol1:
        source_uploads = st.file_uploader(
            "Source data (optional, for fidelity comparison)",
            type=["parquet", "csv"],
            accept_multiple_files=True,
            help=(
                "Upload one file per table. Filename without extension "
                "must match the synthetic table name (e.g. `users.parquet`)."
            ),
            key="source_uploads",
        )
    with qcol2:
        st.markdown("&nbsp;")
        run_quality_clicked = st.button(
            "Generate quality report",
            type="primary",
            use_container_width=True,
        )

    if run_quality_clicked:
        with st.spinner("Computing quality metrics…"):
            try:
                from sdp.validators.quality_report import quality_report as _qr

                # Build optional source dict from uploads
                source_dfs = None
                if source_uploads:
                    import pandas as pd
                    source_dfs = {}
                    for upload in source_uploads:
                        stem = Path(upload.name).stem
                        suffix = Path(upload.name).suffix.lower()
                        if suffix == ".parquet":
                            source_dfs[stem] = pd.read_parquet(io.BytesIO(upload.getvalue()))
                        elif suffix == ".csv":
                            source_dfs[stem] = pd.read_csv(io.BytesIO(upload.getvalue()))

                report = _qr(dataframes, source=source_dfs or None)
                st.session_state["last_quality_report"] = report
                st.session_state["last_quality_error"] = None
            except Exception as exc:
                st.session_state["last_quality_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state["last_quality_report"] = None

    quality_err = st.session_state.get("last_quality_error")
    if quality_err:
        st.error(f"Quality report failed: {quality_err}")

    quality_report = st.session_state.get("last_quality_report")
    if quality_report is not None:
        # Top-line summary
        if quality_report.has_source:
            fidelity = quality_report.overall_fidelity
            if fidelity is not None:
                if fidelity >= 0.85:
                    st.success(f"Overall fidelity score: **{fidelity:.3f}** (1.0 = identical to source)")
                elif fidelity >= 0.6:
                    st.warning(f"Overall fidelity score: **{fidelity:.3f}** — partial match to source")
                else:
                    st.error(f"Overall fidelity score: **{fidelity:.3f}** — large divergence from source")
            else:
                st.info("Source provided but fidelity could not be computed (insufficient overlap).")
        else:
            st.info("Univariate-only report — no source data provided.")

        # Per-table tabs
        if len(quality_report.tables) == 1:
            only_name = next(iter(quality_report.tables))
            _render_quality_table(quality_report.tables[only_name])
        else:
            qtabs = st.tabs(list(quality_report.tables.keys()))
            for tab, name in zip(qtabs, quality_report.tables.keys()):
                with tab:
                    _render_quality_table(quality_report.tables[name])

        # Download button for the structured JSON
        import json as _json
        st.download_button(
            label="Download report JSON",
            data=_json.dumps(quality_report.to_dict(), indent=2, default=str),
            file_name=f"quality_report_{int(time.time())}.json",
            mime="application/json",
        )

    # --- download ---
    st.subheader("6. Download")
    if run_dir_str:
        zip_bytes = zip_output_dir(Path(run_dir_str))
        st.download_button(
            label="Download all as ZIP",
            data=zip_bytes,
            file_name=f"sdp_run_{int(time.time())}.zip",
            mime="application/zip",
            use_container_width=False,
        )
        st.caption(f"Output staged at `{run_dir_str}` (deleted when the OS cleans up its temp dir)")

    # ---------------------------------------------------------------------
    # Step 7 — Cloud upload (Azure / S3)
    # ---------------------------------------------------------------------
    st.subheader("7. Upload to cloud")
    st.caption(
        "Push the generated Parquet (or any directory of Parquet — delta / SCD2 outputs work too) "
        "to Azure Blob Storage or AWS S3. Credentials are read from environment variables — "
        "**the UI never stores or transmits them**. Set the env vars *before* launching Streamlit "
        "(or pass them through to your Docker container)."
    )

    with st.expander("📋 Where do credentials go? (click for setup)", expanded=False):
        st.markdown(
            "Credentials are read from the **process environment** of whatever is running the "
            "Streamlit server. This means the env vars must be set *before* `streamlit run` is "
            "invoked. The UI cannot ask you for them at runtime — that would mean storing them "
            "somewhere, which is a security trap."
        )
        st.markdown("**Required env vars by provider:**")
        st.markdown(
            "| Provider | Variable(s) — set at least one of these groups |\n"
            "|---|---|\n"
            "| **Azure Blob Storage** | `AZURE_STORAGE_CONNECTION_STRING` *(preferred)* "
            "&nbsp;&nbsp;**OR**&nbsp;&nbsp; `AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_KEY` |\n"
            "| **AWS S3** | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` "
            "&nbsp;&nbsp;(plus optional `AWS_DEFAULT_REGION`, `AWS_SESSION_TOKEN` for STS creds) |"
        )
        st.markdown("---")
        tab_psh, tab_bash, tab_docker, tab_compose = st.tabs([
            "PowerShell (local)", "bash / zsh (local)", "docker run", "docker-compose",
        ])

        with tab_psh:
            st.markdown("Set in the same PowerShell session **before** launching Streamlit:")
            st.code(
                '# Azure (connection string — preferred)\n'
                '$env:AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"\n'
                '\n'
                '# OR Azure account + key\n'
                '$env:AZURE_STORAGE_ACCOUNT = "myaccount"\n'
                '$env:AZURE_STORAGE_KEY     = "your-key"\n'
                '\n'
                '# AWS S3\n'
                '$env:AWS_ACCESS_KEY_ID     = "AKIA..."\n'
                '$env:AWS_SECRET_ACCESS_KEY = "your-secret"\n'
                '$env:AWS_DEFAULT_REGION    = "eu-west-1"\n'
                '\n'
                '# Then launch:\n'
                'poetry run streamlit run ui/streamlit_app.py',
                language="powershell",
            )
            st.caption(
                "💡 To make these persist across PowerShell sessions, use "
                "`[System.Environment]::SetEnvironmentVariable('NAME', 'VALUE', 'User')` and "
                "open a new shell."
            )

        with tab_bash:
            st.markdown("Set in the same shell session **before** launching Streamlit:")
            st.code(
                '# Azure (connection string — preferred)\n'
                'export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"\n'
                '\n'
                '# OR Azure account + key\n'
                'export AZURE_STORAGE_ACCOUNT="myaccount"\n'
                'export AZURE_STORAGE_KEY="your-key"\n'
                '\n'
                '# AWS S3\n'
                'export AWS_ACCESS_KEY_ID="AKIA..."\n'
                'export AWS_SECRET_ACCESS_KEY="your-secret"\n'
                'export AWS_DEFAULT_REGION="eu-west-1"\n'
                '\n'
                '# Then launch:\n'
                'poetry run streamlit run ui/streamlit_app.py',
                language="bash",
            )
            st.caption(
                "💡 To persist, add the `export` lines to `~/.bashrc` / `~/.zshrc` / "
                "`~/.profile` (or use direnv with a project-local `.envrc`)."
            )

        with tab_docker:
            st.markdown("Pass through to the container with `-e` or `--env-file`:")
            st.code(
                '# Inline (one-off) — Azure\n'
                'docker run --rm -p 8501:8501 -v "$PWD:/work" \\\n'
                '  -e AZURE_STORAGE_CONNECTION_STRING="DefaultEndpoints..." \\\n'
                '  sdp:latest streamlit\n'
                '\n'
                '# Inline — S3\n'
                'docker run --rm -p 8501:8501 -v "$PWD:/work" \\\n'
                '  -e AWS_ACCESS_KEY_ID="AKIA..." \\\n'
                '  -e AWS_SECRET_ACCESS_KEY="your-secret" \\\n'
                '  -e AWS_DEFAULT_REGION="eu-west-1" \\\n'
                '  sdp:latest streamlit\n'
                '\n'
                '# Or load from a file (cleaner for multiple vars)\n'
                'docker run --rm -p 8501:8501 -v "$PWD:/work" \\\n'
                '  --env-file ./cloud.env \\\n'
                '  sdp:latest streamlit',
                language="bash",
            )
            st.caption(
                "⚠️ **Don't commit `cloud.env` to git.** Add it to `.gitignore`. "
                "On Windows PowerShell, replace `\\` line continuations with backticks (`` ` ``)."
            )

        with tab_compose:
            st.markdown(
                "Easiest path: drop a `.env` file next to `docker-compose.yml`. "
                "docker-compose auto-loads it into every service's environment."
            )
            st.code(
                '# .env  (next to docker-compose.yml — git-ignored!)\n'
                'AZURE_STORAGE_CONNECTION_STRING=DefaultEndpoints...\n'
                'AWS_ACCESS_KEY_ID=AKIA...\n'
                'AWS_SECRET_ACCESS_KEY=your-secret\n'
                'AWS_DEFAULT_REGION=eu-west-1\n',
                language="bash",
            )
            st.markdown(
                "If you want to be explicit, reference the vars in `docker-compose.yml`:"
            )
            st.code(
                'services:\n'
                '  streamlit:\n'
                '    image: sdp:latest\n'
                '    environment:\n'
                '      - AZURE_STORAGE_CONNECTION_STRING\n'
                '      - AWS_ACCESS_KEY_ID\n'
                '      - AWS_SECRET_ACCESS_KEY\n'
                '      - AWS_DEFAULT_REGION\n'
                '    ports:\n'
                '      - "8501:8501"\n'
                '    volumes:\n'
                '      - ./:/work\n'
                '    command: ["streamlit"]\n',
                language="yaml",
            )
            st.caption(
                "Then `docker compose up streamlit` — the env vars flow through automatically. "
                "Add `.env` to `.gitignore` (the project's `.dockerignore` already excludes it from images)."
            )

        st.markdown("---")
        st.markdown(
            "**After setting the variables, restart Streamlit** so the new process picks up "
            "the env. The credential check below will turn green when the right vars are visible."
        )

    upload_dir_choice = st.radio(
        "What to upload",
        [
            "The generated output above",
            "A different local directory (delta / SCD2 / saved run)",
        ],
        horizontal=False,
        label_visibility="collapsed",
        key="upload_dir_choice",
    )

    upload_dir: Optional[str] = None
    if upload_dir_choice == "The generated output above":
        upload_dir = run_dir_str
    else:
        upload_dir = st.text_input(
            "Local directory to upload (absolute path, must contain Parquet files)",
            placeholder="/path/to/output/snap_v1  or  C:/runs/scd2_history",
            help=(
                "Pick any directory of Parquet files — typically a snapshot, "
                "a `delta` output, or an `scd2` history directory. The uploader "
                "walks recursively."
            ),
        )

    ucol1, ucol2 = st.columns([1, 2])
    with ucol1:
        provider = st.selectbox(
            "Provider",
            ["Azure Blob Storage", "AWS S3"],
            help=(
                "Azure: requires AZURE_STORAGE_CONNECTION_STRING (preferred) or "
                "AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY. "
                "S3: requires AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (and "
                "optionally AWS_DEFAULT_REGION) in the env."
            ),
        )
    with ucol2:
        if provider == "Azure Blob Storage":
            container = st.text_input("Container name", value="synthetic-data")
            prefix = st.text_input("Path / prefix in the container", value=f"runs/{int(time.time())}")
            destination_uri = f"azure://{container}/{prefix}".rstrip("/")
        else:
            bucket = st.text_input("Bucket name", value="my-synthetic-data")
            prefix = st.text_input("Path / prefix in the bucket", value=f"runs/{int(time.time())}")
            destination_uri = f"s3://{bucket}/{prefix}".rstrip("/")

    st.code(f"Destination: {destination_uri}", language="text")

    # Show whether the credentials we need are actually present in the env
    import os as _os
    creds_ok = False
    creds_msg = ""
    if provider == "Azure Blob Storage":
        if _os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            creds_ok = True
            creds_msg = "AZURE_STORAGE_CONNECTION_STRING is set."
        elif _os.environ.get("AZURE_STORAGE_ACCOUNT") and _os.environ.get("AZURE_STORAGE_KEY"):
            creds_ok = True
            creds_msg = "AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY are set."
        else:
            creds_msg = (
                "❌ No Azure credentials found in this Streamlit process's environment. Set "
                "`AZURE_STORAGE_CONNECTION_STRING` (preferred) or "
                "`AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_KEY` **before launching Streamlit**, "
                "then restart it. See the *Where do credentials go?* expander above for "
                "copy-paste setup commands."
            )
    else:
        if _os.environ.get("AWS_ACCESS_KEY_ID") and _os.environ.get("AWS_SECRET_ACCESS_KEY"):
            creds_ok = True
            creds_msg = "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY are set."
        else:
            creds_msg = (
                "❌ No AWS credentials found in this Streamlit process's environment. Set "
                "`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (and optionally "
                "`AWS_DEFAULT_REGION`) **before launching Streamlit**, then restart it. "
                "See the *Where do credentials go?* expander above for copy-paste setup commands."
            )
    if creds_ok:
        st.success(creds_msg)
    else:
        st.warning(creds_msg)

    upload_disabled = (not upload_dir) or (not creds_ok)
    upload_clicked = st.button(
        f"Upload to {provider}",
        type="primary",
        disabled=upload_disabled,
        help="Disabled until a directory is chosen and the matching env credentials are present.",
        key="upload_button",
    )

    if upload_clicked and upload_dir:
        with st.spinner(f"Uploading {upload_dir} → {destination_uri}…"):
            try:
                from sdp.utils.cloud_uploader import upload_output

                if not Path(upload_dir).exists():
                    raise FileNotFoundError(f"Directory not found: {upload_dir}")

                paths = upload_output(upload_dir, destination_uri)
                st.success(
                    f"Uploaded **{len(paths)} file(s)** to `{destination_uri}`."
                )
                with st.expander("Uploaded files", expanded=False):
                    for p in paths[:200]:
                        st.code(p, language="text")
                    if len(paths) > 200:
                        st.caption(f"... and {len(paths) - 200} more")
            except ModuleNotFoundError as exc:
                st.error(
                    f"Cloud SDK missing: {exc}. Install it: "
                    "`pip install azure-storage-blob` (Azure) or `pip install boto3` (S3)."
                )
            except Exception as exc:
                st.error(f"Upload failed: {type(exc).__name__}: {exc}")
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc(), language="text")

