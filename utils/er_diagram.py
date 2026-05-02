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
    # PNG via matplotlib (optional — skipped gracefully if unavailable)
    # ------------------------------------------------------------------
    def generate_png(self, output_path: Path) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import networkx as nx
        except ImportError:
            logger.warning("matplotlib / networkx not available — PNG ER diagram skipped. "
                           "Install matplotlib to enable: pip install matplotlib")
            return False

        G = nx.DiGraph()
        for table_name in self.tables:
            G.add_node(table_name)
        for rel in self.relationships:
            col_lbl = getattr(rel, "source_column", None) or (getattr(rel, "source_columns", None) or [""])[0]
            G.add_edge(rel.target_table, rel.source_table, label=col_lbl)

        fig, ax = plt.subplots(figsize=(max(12, len(self.tables) * 2), 8))
        pos = nx.spring_layout(G, seed=42, k=2.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000, node_color="#4C8BF5", alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=9, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555", arrows=True,
                               arrowsize=20, connectionstyle="arc3,rad=0.1")
        edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        ax.set_title("Entity Relationship Diagram", fontsize=14, pad=16)
        ax.axis("off")
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
