#!/usr/bin/env python3
"""Consolidate every platform document into one HTML handbook, then PDF.

Why HTML-then-headless-Chrome rather than a pure-Python PDF library: these
docs are dense with wide markdown tables, and xhtml2pdf/reportlab render
those badly. A browser lays them out properly and honours print CSS, so the
output is genuinely readable at A4.

    python docs/build_handbook.py            # HTML + PDF
    python docs/build_handbook.py --html     # HTML only

Ordering is deliberate — a reader should be able to start at page one and
work forward, so it runs Start here -> Configure -> Generate -> Validate ->
Advanced -> Operate -> Project. Alphabetical order would scatter related
material.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_HTML = REPO / "docs" / "SDP_Handbook.html"
OUT_PDF = REPO / "docs" / "SDP_Handbook.pdf"

# Not about this platform — an Anthropic certification study guide that
# happens to live in the repo root.
EXCLUDE = {"Claude_Architect_Certification.md"}

#: (Section title, [documents in reading order]). Any .md not listed here is
#: appended to "Other documents" so nothing is silently dropped.
SECTIONS: list[tuple[str, list[str]]] = [
    ("Start here", [
        "README.md",
        "Getting_Started.md",
        "Examples_Walkthrough.md",
        "MASTER_COMMANDS.md",
        "One_Stop_Commands.md",
    ]),
    ("Configuring the platform", [
        "Yaml_Config_Schema.md",
        "Json_Config_Schema.md",
        "Regex_Rules.md",
        "Rules_and_Workflows.md",
    ]),
    ("Generating data", [
        "Usage.md",
        "Synthesizer_Engines.md",
        "Differential_Privacy.md",
    ]),
    ("Validating and measuring", [
        "Data_Validation.md",
        "Quality_Reports.md",
        "Data_Contract_Testing.md",
    ]),
    ("API mocks and stubs", [
        "Stubs_Mocks_Plan.md",
        "API_Mocks_Execution_Guide.md",
        "Bruno_Workflow.md",
    ]),
    ("Machine learning and LLM features", [
        "ML_Relationship_Inference.md",
        "ML_Deep_Dive.md",
        "LLM_Ecosystem.md",
        "Data_CDC_AI_LLM_Execution_Guide.md",
    ]),
    ("Interfaces", [
        "SDK_and_API.md",
        "UI_Quickstart.md",
        "MCP_Integration.md",
    ]),
    ("Deployment and packaging", [
        "Packaging.md",
        "Docker_Quickstart.md",
        "Docker_Data_Mocks_Execution_Guide.md",
    ]),
    ("Project and direction", [
        "CLAUDE.md",
        "Platform_Direction.md",
        "PRD_Roadmap.md",
        "V2_Backlog.md",
        "Pending_Items.md",
        "Platform_Presentation.md",
    ]),
]

CSS = """
@page { size: A4; margin: 18mm 15mm 20mm 15mm; }
:root { --ink:#1a1a1a; --muted:#5b6570; --rule:#d7dce1; --accent:#0b5cad;
        --code-bg:#f5f7f9; }
* { box-sizing: border-box; }
body { font: 10.5pt/1.55 "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
       color: var(--ink); margin: 0; }
.page-break { page-break-before: always; }

/* Cover */
.cover { text-align:center; padding-top: 22vh; page-break-after: always; }
.cover h1 { font-size: 30pt; margin: 0 0 .2em; letter-spacing:-.5px; }
.cover .sub { font-size: 13pt; color: var(--muted); margin-bottom: 2.5em; }
.cover .meta { font-size: 10pt; color: var(--muted); line-height: 1.9; }
.cover .rule { width: 90px; border-top: 3px solid var(--accent); margin: 1.5em auto; }

/* Table of contents */
.toc { page-break-after: always; }
.toc h2 { border:0; font-size: 18pt; }
.toc .sec { margin: 1.1em 0 .3em; font-weight:600; color: var(--accent);
            font-size: 11pt; text-transform: uppercase; letter-spacing:.4px; }
.toc ol { list-style:none; margin:0; padding:0 0 0 .2em; }
.toc li { padding: 2px 0; font-size: 10pt; }
.toc a { color: var(--ink); text-decoration:none; }

/* Document chrome */
.doc { page-break-before: always; }
.doc-source { font-size: 8.5pt; color: var(--muted); font-family: ui-monospace, Consolas, monospace;
              border-bottom: 1px solid var(--rule); padding-bottom: 4px; margin-bottom: 14px; }

h1,h2,h3,h4 { line-height:1.25; page-break-after: avoid; margin: 1.4em 0 .5em; }
h1 { font-size: 20pt; border-bottom: 2px solid var(--accent); padding-bottom:.25em; }
h2 { font-size: 15pt; border-bottom: 1px solid var(--rule); padding-bottom:.2em; }
h3 { font-size: 12.5pt; } h4 { font-size: 11pt; }
p, li { orphans:3; widows:3; }
a { color: var(--accent); }

/* Code — wrap rather than clip, so nothing is lost in print */
code { background: var(--code-bg); padding: 1px 4px; border-radius:3px;
       font-family: ui-monospace, Consolas, "Courier New", monospace; font-size: 9pt; }
pre { background: var(--code-bg); border:1px solid var(--rule); border-left:3px solid var(--accent);
      padding: 9px 11px; border-radius:4px; overflow:visible; page-break-inside: avoid;
      white-space: pre-wrap; word-wrap: break-word; font-size: 8.5pt; line-height:1.45; }
pre code { background:none; padding:0; font-size: inherit; }

/* Tables — these docs are table-heavy; keep them on one page where possible */
table { border-collapse: collapse; width:100%; margin: 1em 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border:1px solid var(--rule); padding: 5px 7px; text-align:left;
         vertical-align: top; word-break: break-word; }
th { background:#eef2f6; font-weight:600; }
tr:nth-child(even) td { background:#fafbfc; }

blockquote { border-left:3px solid var(--accent); background:#f7fafd; margin:1em 0;
             padding: .6em .9em; color:#33404d; page-break-inside: avoid; }
hr { border:0; border-top:1px solid var(--rule); margin: 1.6em 0; }
img { max-width: 100%; }

@media print { a { text-decoration: none; } }
"""


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def collect() -> list[tuple[str, list[Path]]]:
    """Resolve the section plan, appending anything unlisted so no doc is lost."""
    listed: set[str] = set()
    sections: list[tuple[str, list[Path]]] = []

    for title, names in SECTIONS:
        docs = []
        for name in names:
            path = REPO / name
            if name in EXCLUDE:
                continue
            if path.is_file():
                docs.append(path)
                listed.add(name)
            else:
                print(f"  ! listed but missing: {name}", file=sys.stderr)
        if docs:
            sections.append((title, docs))

    leftovers = sorted(
        p for p in REPO.glob("*.md")
        if p.name not in listed and p.name not in EXCLUDE
    )
    if leftovers:
        sections.append(("Other documents", leftovers))
        for p in leftovers:
            print(f"  + auto-included: {p.name}", file=sys.stderr)

    return sections


def render_markdown(text: str) -> str:
    import markdown

    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )


def strip_frontmatter(text: str) -> str:
    """Remove YAML front matter (Platform_Presentation.md is a Marp deck)."""
    if text.lstrip().startswith("---"):
        parts = text.lstrip().split("---", 2)
        if len(parts) >= 3:
            return parts[2]
    return text


def build_html() -> Path:
    sections = collect()
    total_docs = sum(len(d) for _, d in sections)

    toc: list[str] = ['<div class="toc"><h2>Contents</h2>']
    body: list[str] = []

    for title, docs in sections:
        toc.append(f'<div class="sec">{html.escape(title)}</div><ol>')
        for path in docs:
            anchor = slug(path.stem)
            toc.append(
                f'<li><a href="#{anchor}">{html.escape(path.stem.replace("_", " "))}</a></li>'
            )
            text = strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
            body.append(
                f'<div class="doc" id="{anchor}">'
                f'<div class="doc-source">{html.escape(path.name)}</div>'
                f"{render_markdown(text)}</div>"
            )
        toc.append("</ol>")
    toc.append("</div>")

    cover = f"""
    <div class="cover">
      <h1>Synthetic Data Platform</h1>
      <div class="sub">Complete Handbook</div>
      <div class="rule"></div>
      <div class="meta">
        Consolidated from {total_docs} documents<br>
        Generated {date.today().isoformat()}<br>
        Covers data generation, CDC/SCD2, engines, differential privacy,<br>
        workflows, quality reporting, contracts, API mocks, SDK/API/MCP
      </div>
    </div>"""

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Synthetic Data Platform — Complete Handbook</title>"
        f"<style>{CSS}</style></head><body>"
        f"{cover}{''.join(toc)}{''.join(body)}"
        "</body></html>"
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"HTML: {OUT_HTML}  ({OUT_HTML.stat().st_size // 1024} KB, {total_docs} documents)")
    return OUT_HTML


def find_browser() -> str | None:
    for candidate in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return shutil.which("msedge") or shutil.which("chrome")


def build_pdf(html_path: Path) -> bool:
    browser = find_browser()
    if not browser:
        print("No Chrome/Edge found — open the HTML and use Print > Save as PDF.")
        return False

    print(f"Rendering PDF via {Path(browser).name} ...")
    if OUT_PDF.exists():
        OUT_PDF.unlink()

    # `--headless=new` is required on current Edge/Chrome: the legacy
    # `--headless` exits 0 and writes nothing for --print-to-pdf. The old
    # flag is kept as a fallback for older builds.
    # A dedicated profile directory is not optional: if any other Edge/Chrome
    # instance is running, a new one attaches to that profile and exits
    # immediately — exit code 0, no PDF, no error message.
    import tempfile

    profile = tempfile.mkdtemp(prefix="sdp_pdf_profile_")

    for headless in ("--headless=new", "--headless"):
        result = subprocess.run(
            [browser, headless, "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
             f"--print-to-pdf={OUT_PDF}", html_path.as_uri()],
            capture_output=True, timeout=900,
        )
        if OUT_PDF.is_file() and OUT_PDF.stat().st_size > 0:
            print(f"PDF:  {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KB)  [{headless}]")
            return True

    print("PDF render failed:", result.stderr.decode("utf-8", "replace")[-400:])
    print("Fallback: open the HTML in a browser and use Print > Save as PDF.")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", action="store_true", help="HTML only, skip the PDF step")
    args = ap.parse_args()

    path = build_html()
    if not args.html:
        build_pdf(path)
