"""Streamlit page — API mocks generation.

Upload an OpenAPI spec, Postman collection, or HAR capture; auto-detect
the format; convert to a sdp-mock-v1 config; render to one or more output
formats (WireMock / JSON fixtures / Pact / Postman / OpenAPI examples);
preview a sample mapping; download the bundle as a ZIP.

Same engine as `python main.py mock-init` and `mock-render` — runs
in-process. For LLM-assisted authoring (`mock-enrich`) and stateful
scenarios that need editing, drop back to the CLI.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Same path-shim as the entry script — makes top-level project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from main import _detect_mock_source_type  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OPENAPI_EXAMPLES_DIR = REPO_ROOT / "examples" / "openapi"
MOCKS_EXAMPLES_DIR = REPO_ROOT / "examples" / "mocks"

ALL_FORMATS = ["wiremock", "json", "pact", "postman", "openapi-examples"]
DEFAULT_FORMATS = ["wiremock", "json"]

st.set_page_config(page_title="API Mocks", page_icon="🧪", layout="wide")

st.title("API Mocks")
st.caption(
    "OpenAPI / Postman / HAR → sdp-mock-v1 config → "
    "WireMock + JSON + Pact + Postman + OpenAPI-examples. "
    "Same engine as the CLI, runs in-process."
)


# ---------------------------------------------------------------------------
# Step 1 — Pick the source artefact
# ---------------------------------------------------------------------------

st.subheader("1. Pick the source artefact")

source_mode = st.radio(
    "Source",
    ["Use bundled example", "Upload my own"],
    horizontal=True,
    label_visibility="collapsed",
    key="mocks_source_mode",
)

source_path: Optional[Path] = None
source_label: str = ""

if source_mode == "Use bundled example":
    bundled_specs: List[Path] = []
    if OPENAPI_EXAMPLES_DIR.exists():
        bundled_specs.extend(sorted(OPENAPI_EXAMPLES_DIR.glob("*.yaml")))
        bundled_specs.extend(sorted(OPENAPI_EXAMPLES_DIR.glob("*.yml")))
    if MOCKS_EXAMPLES_DIR.exists():
        bundled_specs.extend(sorted(p for p in MOCKS_EXAMPLES_DIR.glob("*")
                                    if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml", ".har"}))
    if not bundled_specs:
        st.warning("No bundled specs found under `examples/openapi/` or `examples/mocks/`.")
    else:
        labels = [f"{p.parent.name}/{p.name}" for p in bundled_specs]
        choice = st.selectbox(
            "Bundled specs",
            labels,
            index=0,
            help=(
                "OpenAPI specs in `examples/openapi/`; sample Postman + HAR "
                "fixtures in `examples/mocks/`."
            ),
        )
        source_path = bundled_specs[labels.index(choice)]
        source_label = choice
else:
    uploaded = st.file_uploader(
        "Upload an OpenAPI spec, Postman collection, or HAR file",
        type=["yaml", "yml", "json", "har"],
        key="mocks_uploader",
    )
    if uploaded is not None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="sdp_mocks_upload_"))
        target = tmp_dir / uploaded.name
        target.write_bytes(uploaded.getvalue())
        source_path = target
        source_label = uploaded.name
        st.success(f"Loaded `{uploaded.name}` ({uploaded.size:,} bytes)")


# ---------------------------------------------------------------------------
# Auto-detect source format
# ---------------------------------------------------------------------------

detected_type = "auto"
if source_path is not None:
    detected_type = _detect_mock_source_type(str(source_path), "auto")
    if detected_type == "auto":
        st.warning("Could not auto-detect format. Use the override below.")
    else:
        st.info(f"Detected source format: **{detected_type}**")

with st.expander("Override detected format (rare)", expanded=False):
    override = st.selectbox(
        "Source type",
        ["auto", "openapi", "postman", "har"],
        index=["auto", "openapi", "postman", "har"].index(
            detected_type if detected_type in {"openapi", "postman", "har"} else "auto"
        ),
        help="Override only when auto-detection guesses wrong.",
    )

source_type = override if override != "auto" else detected_type


# ---------------------------------------------------------------------------
# Step 2 — Render settings
# ---------------------------------------------------------------------------

st.subheader("2. Render settings")

setcol1, setcol2, setcol3 = st.columns(3)
with setcol1:
    formats = st.multiselect(
        "Output formats",
        ALL_FORMATS,
        default=DEFAULT_FORMATS,
        help=(
            "wiremock = stub mappings  • json = raw fixtures  "
            "• pact = consumer-driven contracts  • postman = collection  "
            "• openapi-examples = inject example: blocks back into the source spec"
        ),
    )
with setcol2:
    examples = st.number_input(
        "Examples per endpoint",
        min_value=1, max_value=50, value=3, step=1,
        help="How many sample stubs to render per endpoint.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0, max_value=2**31 - 1, value=42,
        help="Same seed → byte-identical output across runs.",
    )
with setcol3:
    match_mode = st.radio(
        "Path-matching mode",
        ["concrete", "any"],
        index=0,
        help=(
            "concrete: literal urlPath per example (snapshot tests). "
            "any: regex urlPathPattern (one stub per status — exploratory dev)."
        ),
        horizontal=True,
    )
    pact_consumer = st.text_input("Pact consumer", value="consumer",
                                  help="Only used when 'pact' is selected.")
    pact_provider = st.text_input("Pact provider", value="provider",
                                  help="Only used when 'pact' is selected.")


# ---------------------------------------------------------------------------
# Step 3 — Actions
# ---------------------------------------------------------------------------

st.subheader("3. Actions")

acol1, acol2, _ = st.columns([1, 1, 3])
with acol1:
    init_clicked = st.button(
        "Convert + lint",
        type="secondary",
        use_container_width=True,
        disabled=source_path is None,
        help="Convert the source artefact to sdp-mock-v1 and validate it.",
    )
with acol2:
    render_clicked = st.button(
        "Convert + render",
        type="primary",
        use_container_width=True,
        disabled=source_path is None or not formats,
        help="Convert the source AND render the chosen output formats.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert(source_path: Path, source_type: str):
    if source_type == "openapi":
        from mocks.openapi_importer import import_openapi
        return import_openapi(str(source_path))
    if source_type == "postman":
        from mocks.postman_importer import import_postman
        return import_postman(str(source_path))
    if source_type == "har":
        from mocks.har_importer import import_har
        return import_har(str(source_path))
    raise ValueError(f"Unrecognised source_type: {source_type!r}")


def _render_summary(cfg) -> None:
    """Show the MockConfig summary inline."""
    st.markdown(f"**Endpoints:** {len(cfg.endpoints)}  •  **Schemas:** {len(cfg.schemas)}")
    if cfg.servers:
        st.caption(f"Servers: " + ", ".join(s.url for s in cfg.servers))

    if cfg.endpoints:
        rows = []
        for ep in cfg.endpoints:
            statuses = ",".join(str(r.status) for r in ep.responses)
            rows.append({
                "name": ep.name,
                "method": ep.method,
                "path": ep.path,
                "responses": statuses,
                "tags": ",".join(ep.tags) if ep.tags else "",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _zip_dir(d: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in d.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(d))
    buf.seek(0)
    return buf.getvalue()


def _render_mocks(cfg, formats: List[str], *, examples: int, seed: int,
                  match_mode: str, pact_consumer: str, pact_provider: str,
                  source_path: Path) -> tuple[Path, Dict[str, int]]:
    out_dir = Path(tempfile.mkdtemp(prefix="sdp_mocks_render_"))
    written: Dict[str, int] = {}
    for fmt in formats:
        target = out_dir / fmt
        if fmt == "wiremock":
            from mocks.renderers.wiremock import render_wiremock
            files = render_wiremock(cfg, target, seed=seed,
                                    examples_per_endpoint=examples,
                                    match_mode=match_mode)
        elif fmt == "json":
            from mocks.renderers.json_fixture import render_json_fixtures
            files = render_json_fixtures(cfg, target, seed=seed)
        elif fmt == "pact":
            from mocks.renderers.pact import render_pact
            files = render_pact(cfg, target, consumer=pact_consumer,
                                provider=pact_provider, seed=seed,
                                examples_per_endpoint=examples)
        elif fmt == "postman":
            from mocks.renderers.postman import render_postman
            files = render_postman(cfg, target, seed=seed,
                                   examples_per_endpoint=examples)
        elif fmt == "openapi-examples":
            # Round-trip is only meaningful for OpenAPI sources
            from mocks.renderers.openapi_examples import render_openapi_examples
            files = render_openapi_examples(cfg, target, source_spec=str(source_path),
                                            seed=seed)
        else:
            continue
        written[fmt] = len(files)
    return out_dir, written


# ---------------------------------------------------------------------------
# Convert + lint
# ---------------------------------------------------------------------------

if init_clicked and source_path is not None and source_type != "auto":
    with st.spinner(f"Converting {source_label} ({source_type}) → sdp-mock-v1…"):
        try:
            cfg = _convert(source_path, source_type)
            st.session_state["mocks_cfg"] = cfg
            st.session_state["mocks_source_label"] = source_label
            st.session_state["mocks_source_path"] = str(source_path)
            st.session_state["mocks_render_dir"] = None
            st.session_state["mocks_render_written"] = None
            st.session_state["mocks_error"] = None
        except Exception as exc:
            st.session_state["mocks_cfg"] = None
            st.session_state["mocks_error"] = (
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

# ---------------------------------------------------------------------------
# Convert + render
# ---------------------------------------------------------------------------

if render_clicked and source_path is not None and source_type != "auto":
    with st.spinner(f"Converting + rendering {len(formats)} format(s)…"):
        try:
            cfg = _convert(source_path, source_type)
            out_dir, written = _render_mocks(
                cfg,
                formats=formats,
                examples=int(examples),
                seed=int(seed),
                match_mode=match_mode,
                pact_consumer=pact_consumer,
                pact_provider=pact_provider,
                source_path=source_path,
            )
            st.session_state["mocks_cfg"] = cfg
            st.session_state["mocks_source_label"] = source_label
            st.session_state["mocks_source_path"] = str(source_path)
            st.session_state["mocks_render_dir"] = str(out_dir)
            st.session_state["mocks_render_written"] = written
            st.session_state["mocks_error"] = None
        except Exception as exc:
            st.session_state["mocks_error"] = (
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

mocks_error = st.session_state.get("mocks_error")
if mocks_error:
    st.error("Failed to process the source.")
    with st.expander("Stack trace"):
        st.code(mocks_error, language="text")

cfg = st.session_state.get("mocks_cfg")
render_dir_str = st.session_state.get("mocks_render_dir")
written = st.session_state.get("mocks_render_written")

if cfg is not None:
    st.subheader("4. MockConfig summary")
    _render_summary(cfg)

    # Show the YAML so the user knows what'll be rendered
    with st.expander("View / download sdp-mock-v1 YAML", expanded=False):
        try:
            from mocks.config_parser import dump_mock_config
            tmp = Path(tempfile.mkstemp(suffix=".yaml", prefix="sdp_mock_yaml_")[1])
            dump_mock_config(cfg, tmp)
            yaml_text = tmp.read_text(encoding="utf-8")
            st.code(yaml_text[:6000] + ("\n... (truncated)" if len(yaml_text) > 6000 else ""),
                    language="yaml")
            st.download_button(
                label="Download mock-config YAML",
                data=yaml_text,
                file_name="mocks.yaml",
                mime="text/yaml",
            )
        except Exception as exc:
            st.warning(f"Could not dump YAML: {exc}")

if cfg is not None and render_dir_str:
    st.subheader("5. Rendered output")
    summary_lines = ", ".join(f"**{n}** {fmt}" for fmt, n in (written or {}).items())
    st.success(f"Wrote {summary_lines} file(s).")

    # Preview one sample wiremock mapping if present
    render_dir = Path(render_dir_str)
    if (render_dir / "wiremock" / "mappings").exists():
        sample_files = sorted((render_dir / "wiremock" / "mappings").iterdir())
        if sample_files:
            with st.expander(f"Sample WireMock mapping — `{sample_files[0].name}`", expanded=False):
                try:
                    sample = json.loads(sample_files[0].read_text(encoding="utf-8"))
                    st.json(sample)
                except Exception:
                    st.code(sample_files[0].read_text(encoding="utf-8"), language="json")

    # ZIP everything for download
    zip_bytes = _zip_dir(render_dir)
    st.download_button(
        label=f"Download all rendered mocks ({sum((written or {}).values())} files) as ZIP",
        data=zip_bytes,
        file_name=f"sdp_mocks_{int(time.time())}.zip",
        mime="application/zip",
        use_container_width=False,
    )
    st.caption(
        f"Render staged at `{render_dir_str}`. "
        "Point WireMock standalone at `wiremock/` to serve the stubs."
    )
