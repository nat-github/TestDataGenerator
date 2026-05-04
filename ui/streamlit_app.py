"""Synthetic Data Platform — Streamlit UI v1.

A simple, intuitive interface for the data-generation track. Sized for
≤ 10,000 rows total — for larger runs, point users at the CLI.

Run from the repo root:
    poetry install --extras ui
    poetry run streamlit run ui/streamlit_app.py
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

# Make the parent project importable when running `streamlit run ui/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    "JSON configs. UI cap: 10,000 rows per table."
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
    from utils.config_parser import ConfigParser
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
    from generators.data_generator import DataGenerator
    from utils.config_parser import ConfigParser
    import pandas as pd
    import pyarrow.parquet as pq

    parser = ConfigParser(str(config_path))
    parser.load_config()
    parser.parse_tables()
    parser.parse_relationships()

    gen = DataGenerator(parser)

    # Honour the platform's existing API. records_per_table is per-table count.
    records_config = {
        name: min(default_records, MAX_ROWS_PER_TABLE_UI)
        for name in gen.tables_config
        if gen.tables_config[name].active
    }

    started = time.perf_counter()
    gen.create_sdv_metadata()
    gen.train_synthesizer(sample_size=min(default_records, 200), seed=seed)
    gen.generate_data(records_config=records_config, seed=seed)
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

    # --- download ---
    st.subheader("5. Download")
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
