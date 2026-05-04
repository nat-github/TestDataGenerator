"""Convert every YAML example under examples/configs/yaml/ into JSON and XLSX.

Run this whenever the YAML examples change so the JSON and XLSX siblings stay
in lockstep. Single source of truth = YAML; this script regenerates the rest.

Usage (from the repo root):
    poetry run python examples/configs/generate_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import openpyxl
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


HERE = Path(__file__).resolve().parent
YAML_DIR = HERE / "yaml"
JSON_DIR = HERE / "json"
XLSX_DIR = HERE / "xlsx"

# Examples whose features (rules, derived columns, complex CDC) don't map
# cleanly to the four-sheet Excel format. We skip Excel generation for these
# but document the limitation in Examples_Walkthrough.md.
EXCEL_SKIP = {
    "06_rules_and_derived",          # rules + derived cols are YAML-only
    "07_cdc_delta_workflow",         # cdc: block has nested fields Excel can't represent
    "08_scd2_history",               # same — cdc: block
    "11_bare_for_relationship_inference",  # multi-PK on order_items can be done but more complex
}


# ---------------------------------------------------------------------------
# JSON variant
# ---------------------------------------------------------------------------


def yaml_to_json_variant(yaml_doc: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON variant uses sdp-json-v1 as its config_format; otherwise identical."""
    out = dict(yaml_doc)
    out["config_format"] = "sdp-json-v1"
    return out


# ---------------------------------------------------------------------------
# XLSX variant
# ---------------------------------------------------------------------------


def yaml_to_xlsx(doc: Dict[str, Any], path: Path) -> None:
    """Write an Excel workbook with Run_Settings + Tables + Columns + Relationships sheets."""
    if not HAS_OPENPYXL:
        raise RuntimeError("openpyxl is not installed — `pip install openpyxl`")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # drop the default sheet

    # ---- Run_Settings ----
    ws = wb.create_sheet("Run_Settings")
    ws.append(["setting_name", "value", "description"])
    rs = doc.get("run_settings") or {}
    for key, value in rs.items():
        ws.append([key, value, ""])
    if not rs:
        ws.append(["default_records_per_table", 100, "Fallback row count"])

    # ---- Tables ----
    ws = wb.create_sheet("Tables")
    ws.append([
        "table_name", "description", "row_count",
        "primary_key_columns", "active",
    ])
    for table in doc.get("tables", []):
        ws.append([
            table.get("name"),
            table.get("description") or "",
            table.get("rows"),
            ";".join(table.get("primary_key_columns") or []),
            table.get("active", True),
        ])

    # ---- Columns ----
    ws = wb.create_sheet("Columns")
    ws.append([
        "table_name", "column_name", "data_type",
        "is_pk", "is_fk", "ref_table", "ref_column",
        "business_values", "special_rules",
        "min_value", "max_value", "nullable",
    ])
    for table in doc.get("tables", []):
        tname = table.get("name")
        for col in table.get("columns", []):
            ws.append([
                tname,
                col.get("name"),
                col.get("data_type"),
                col.get("is_pk", False),
                col.get("is_fk", False),
                col.get("ref_table"),
                col.get("ref_column"),
                _coerce_values(col.get("business_values")),
                col.get("special_rules"),
                col.get("min_value"),
                col.get("max_value"),
                col.get("nullable", True),
            ])

    # ---- Relationships ----
    ws = wb.create_sheet("Relationships")
    ws.append([
        "relationship_name", "source_table", "source_columns",
        "target_table", "target_columns", "cardinality", "active",
    ])
    for rel in doc.get("relationships", []):
        ws.append([
            rel.get("name"),
            rel.get("source_table"),
            ";".join(rel.get("source_columns") or []),
            rel.get("target_table"),
            ";".join(rel.get("target_columns") or []),
            rel.get("relationship_type") or "many_to_one",
            rel.get("active", True),
        ])

    # Auto-size columns roughly so the file is comfortable to open
    for sheet in wb.worksheets:
        for col_idx, _col in enumerate(sheet.columns, start=1):
            sheet.column_dimensions[get_column_letter(col_idx)].width = 22

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _coerce_values(values: Any) -> Optional[str]:
    """`business_values` can be a list (YAML) or a semicolon-separated string."""
    if values is None:
        return None
    if isinstance(values, list):
        return ";".join(str(v) for v in values)
    return str(values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not YAML_DIR.exists():
        print(f"YAML source dir not found: {YAML_DIR}", file=sys.stderr)
        return 1

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    XLSX_DIR.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(YAML_DIR.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found under {YAML_DIR}", file=sys.stderr)
        return 1

    json_count = 0
    xlsx_count = 0
    for yp in yaml_files:
        stem = yp.stem
        with yp.open(encoding="utf-8") as fp:
            doc = yaml.safe_load(fp)

        # JSON
        json_doc = yaml_to_json_variant(doc)
        json_path = JSON_DIR / f"{stem}.json"
        json_path.write_text(
            json.dumps(json_doc, indent=2, default=str),
            encoding="utf-8",
        )
        json_count += 1

        # XLSX (skip examples that don't map cleanly)
        if stem in EXCEL_SKIP:
            print(f"  - {stem}: JSON [OK]  XLSX skipped (advanced features)")
            continue
        if not HAS_OPENPYXL:
            print(f"  - {stem}: JSON [OK]  XLSX skipped (openpyxl not installed)")
            continue
        xlsx_path = XLSX_DIR / f"{stem}.xlsx"
        yaml_to_xlsx(doc, xlsx_path)
        xlsx_count += 1
        print(f"  - {stem}: JSON [OK]  XLSX [OK]")

    print(f"\nGenerated {json_count} JSON file(s) and {xlsx_count} XLSX file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
