"""ER diagram generation from TableConfig / RelationshipConfig objects.

Primary output: Mermaid ERD (.mmd) — renders in GitHub, VS Code (Mermaid
extension), Notion, and mermaid.live — zero runtime dependencies.

Optional PNG: generated when matplotlib is importable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-type → Mermaid / ERD display type
# ---------------------------------------------------------------------------
_TYPE_MAP: Dict[str, str] = {
    "N": "int", "N6": "int", "N19": "bigint", "N38": "decimal(38,0)",
    "DC": "decimal", "D": "date", "DT": "datetime", "TS": "timestamp",
    "VA": "varchar", "VA1": "varchar(1)", "VA3": "varchar(3)",
    "VA18": "varchar(18)", "VA50": "varchar(50)", "VA256": "varchar(256)",
    "A": "char", "A1": "char(1)", "A3": "char(3)", "A6": "char(6)",
    "A18": "char(18)", "AN": "varchar", "NS": "varchar", "T": "text",
}


def _mermaid_type(data_type: str) -> str:
    upper = (data_type or "").upper()
    if upper in _TYPE_MAP:
        return _TYPE_MAP[upper]
    # Strip length suffix for lookup: VA50 → VA
    base = "".join(c for c in upper if c.isalpha())
    return _TYPE_MAP.get(base, upper.lower() or "varchar")


def _mermaid_key_suffix(col) -> str:
    tags = []
    if col.is_pk:
        tags.append("PK")
    if col.is_fk:
        tags.append("FK")
    return ",".join(tags)


# ---------------------------------------------------------------------------
# Cardinality → Mermaid ERD relationship syntax
# ---------------------------------------------------------------------------
_CARDINALITY_MAP = {
    "one_to_one":   "||--||",
    "one_to_many":  "||--o{",
    "many_to_one":  "}o--||",
    "many_to_many": "}o--o{",
    "1:1":          "||--||",
    "1:n":          "||--o{",
    "n:1":          "}o--||",
    "n:m":          "}o--o{",
}


def _mermaid_cardinality(cardinality: Optional[str]) -> str:
    return _CARDINALITY_MAP.get((cardinality or "one_to_many").lower(), "||--o{")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
class ERDiagramGenerator:
    def __init__(self, tables_config, relationships=None):
        self.tables = tables_config        # Dict[str, TableConfig]
        self.relationships = relationships or []  # List[RelationshipConfig]

    # ------------------------------------------------------------------
    # Mermaid ERD
    # ------------------------------------------------------------------
    def generate_mermaid(self) -> str:
        lines: List[str] = ["erDiagram"]

        for table_name, table_cfg in self.tables.items():
            lines.append(f"    {table_name} {{")
            for col in table_cfg.columns:
                mtype = _mermaid_type(col.data_type)
                col_name = col.column_name
                key_sfx = _mermaid_key_suffix(col)
                comment = f'"{col.data_type}"' if col.data_type else ""
                if key_sfx:
                    lines.append(f"        {mtype} {col_name} {key_sfx} {comment}".rstrip())
                else:
                    lines.append(f"        {mtype} {col_name} {comment}".rstrip())
            lines.append("    }")

        lines.append("")
        seen: set = set()
        for rel in self.relationships:
            parent = rel.target_table
            child = rel.source_table
            cardinality = _mermaid_cardinality(
                getattr(rel, "cardinality", None) or getattr(rel, "relationship_type", "one_to_many")
            )
            col_label = getattr(rel, "source_column", None) or getattr(rel, "source_columns", [""])[0] if hasattr(rel, "source_columns") else ""
            label = f'"{col_label}"'
            key = (parent, child, label)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"    {parent} {cardinality} {child} : {label}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Graphviz DOT
    # ------------------------------------------------------------------
    def generate_dot(self) -> str:
        lines: List[str] = [
            'digraph ER {',
            '    graph [rankdir=LR fontname="Helvetica" fontsize=12];',
            '    node  [shape=record fontname="Helvetica" fontsize=11];',
            '    edge  [fontname="Helvetica" fontsize=10];',
            '',
        ]

        for table_name, table_cfg in self.tables.items():
            cols_dot = "|".join(
                f"{{{'PK ' if c.is_pk else ''}{'FK ' if c.is_fk else ''}{c.column_name} : {c.data_type}}}"
                for c in table_cfg.columns
            )
            label = f"{{{table_name}|{cols_dot}}}"
            lines.append(f'    {table_name} [label="{label}"];')

        lines.append("")
        for rel in self.relationships:
            col_lbl = getattr(rel, "source_column", None) or (getattr(rel, "source_columns", None) or [""])[0]
            lines.append(f'    {rel.target_table} -> {rel.source_table} [label="{col_lbl}"];')

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PNG — proper ERD table boxes via matplotlib (no networkx needed)
    # ------------------------------------------------------------------
    def generate_png(self, output_path: Path) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not available — PNG ER diagram skipped. "
                           "Install matplotlib to enable: pip install matplotlib")
            return False

        # ── layout constants ──────────────────────────────────────────
        COL_W      = 3.2    # box width (data units)
        HDR_H      = 0.45   # header row height
        ROW_H      = 0.32   # column row height
        H_GAP      = 1.8    # horizontal gap between boxes
        V_GAP      = 1.2    # vertical gap between rows of boxes
        COLS_PER_ROW = 3    # boxes per grid row

        table_names = list(self.tables.keys())
        n = len(table_names)

        # ── compute per-table heights and grid positions ───────────────
        heights: Dict[str, float] = {}
        positions: Dict[str, tuple] = {}   # (x_left, y_top) in data coords
        for i, tname in enumerate(table_names):
            ncols = len(self.tables[tname].columns)
            heights[tname] = HDR_H + ncols * ROW_H
            col_idx = i % COLS_PER_ROW
            row_idx = i // COLS_PER_ROW
            x = col_idx * (COL_W + H_GAP)
            # y_top: stack rows downward; use max height in previous grid rows
            prev_rows = row_idx
            y = -prev_rows * (max(heights.values()) + V_GAP) if heights else 0
            positions[tname] = (x, y)

        # Recalculate y after all heights are known
        row_max_h: Dict[int, float] = {}
        for i, tname in enumerate(table_names):
            row_max_h[i // COLS_PER_ROW] = max(
                row_max_h.get(i // COLS_PER_ROW, 0), heights[tname])
        y_tops: Dict[int, float] = {0: 0.0}
        for r in range(1, (n - 1) // COLS_PER_ROW + 1):
            y_tops[r] = y_tops[r - 1] - (row_max_h.get(r - 1, 0) + V_GAP)
        for i, tname in enumerate(table_names):
            col_idx = i % COLS_PER_ROW
            row_idx = i // COLS_PER_ROW
            positions[tname] = (col_idx * (COL_W + H_GAP), y_tops[row_idx])

        # ── canvas size ───────────────────────────────────────────────
        n_cols_used = min(n, COLS_PER_ROW)
        n_rows_used = (n + COLS_PER_ROW - 1) // COLS_PER_ROW
        fig_w = max(10, n_cols_used * (COL_W + H_GAP) + 1)
        total_h = sum(row_max_h.get(r, 0) + V_GAP for r in range(n_rows_used))
        fig_h = max(6, total_h + 1)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.5, n_cols_used * (COL_W + H_GAP) - H_GAP + 0.5)
        ax.set_ylim(-total_h - 0.5, 1.0)

        # ── draw each table box ───────────────────────────────────────
        HEADER_CLR = "#4C8BF5"
        ALT_CLR    = "#F0F4FF"
        WHITE      = "#FFFFFF"
        BORDER_CLR = "#2A5DB0"

        box_centers: Dict[str, tuple] = {}  # (cx, cy) centre of box
        for tname, table_cfg in self.tables.items():
            x0, y0 = positions[tname]
            h = heights[tname]
            cx = x0 + COL_W / 2
            box_centers[tname] = (cx, y0 - h / 2)

            # header
            hdr = mpatches.FancyBboxPatch(
                (x0, y0 - HDR_H), COL_W, HDR_H,
                boxstyle="square,pad=0", linewidth=1.2,
                edgecolor=BORDER_CLR, facecolor=HEADER_CLR, zorder=2)
            ax.add_patch(hdr)
            ax.text(cx, y0 - HDR_H / 2, tname,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white", zorder=3, clip_on=True)

            # column rows
            for j, col in enumerate(table_cfg.columns):
                ry = y0 - HDR_H - (j + 1) * ROW_H
                bg = ALT_CLR if j % 2 == 0 else WHITE
                row_patch = mpatches.FancyBboxPatch(
                    (x0, ry), COL_W, ROW_H,
                    boxstyle="square,pad=0", linewidth=0.6,
                    edgecolor=BORDER_CLR, facecolor=bg, zorder=2)
                ax.add_patch(row_patch)
                tags = ("PK " if col.is_pk else "") + ("FK " if col.is_fk else "")
                mtype = _mermaid_type(col.data_type)
                label = f"{tags}{col.column_name}  {mtype}"
                ax.text(x0 + 0.08, ry + ROW_H / 2, label,
                        ha="left", va="center", fontsize=6.5, color="#1a1a1a",
                        zorder=3, clip_on=True)

        # ── draw relationship arrows ──────────────────────────────────
        seen_rels: set = set()
        for rel in self.relationships:
            parent = rel.target_table
            child  = rel.source_table
            if parent not in box_centers or child not in box_centers:
                continue
            key = (parent, child)
            if key in seen_rels:
                continue
            seen_rels.add(key)
            px, py = box_centers[parent]
            cx2, cy2 = box_centers[child]
            col_lbl = getattr(rel, "source_column", None) or \
                      (getattr(rel, "source_columns", None) or [""])[0]
            ax.annotate(
                "", xy=(cx2, cy2), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.2, connectionstyle="arc3,rad=0.08"),
                zorder=1)
            mx, my = (px + cx2) / 2, (py + cy2) / 2
            ax.text(mx, my, col_lbl, fontsize=6, color="#333",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1),
                    zorder=4)

        ax.set_title("Entity Relationship Diagram", fontsize=12, pad=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"PNG ER diagram saved: {output_path}")
        return True

    # ------------------------------------------------------------------
    # Save all formats
    # ------------------------------------------------------------------
    def save(self, output_dir: str, formats: Optional[List[str]] = None) -> List[Path]:
        """
        Save ER diagram(s) to output_dir.

        formats: list of 'mermaid', 'dot', 'png'  (default: ['mermaid'])
        Returns list of paths written.
        """
        if formats is None:
            formats = ["mermaid"]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        if "mermaid" in formats:
            p = out / "er_diagram.mmd"
            p.write_text(self.generate_mermaid(), encoding="utf-8")
            logger.info(f"Mermaid ER diagram saved: {p}")
            written.append(p)

        if "dot" in formats:
            p = out / "er_diagram.dot"
            p.write_text(self.generate_dot(), encoding="utf-8")
            logger.info(f"DOT ER diagram saved: {p}")
            written.append(p)

        if "png" in formats:
            p = out / "er_diagram.png"
            if self.generate_png(p):
                written.append(p)

        return written
