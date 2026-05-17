import pandas as pd
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys


class ERDiagramGenerator:
    def __init__(self, config_file: str):
        """
        Generate ER diagrams from Excel configuration

        Args:
            config_file: Path to Excel configuration file
        """
        self.config_file = Path(config_file)
        self.config_df = None
        self.tables = {}
        self.relationships = []

    def load_configuration(self) -> bool:
        """Load configuration from Excel file"""
        try:
            if not self.config_file.exists():
                print(f"❌ Configuration file not found: {self.config_file}")
                return False

            self.config_df = pd.read_excel(self.config_file, sheet_name='Columns')
            print(f"✅ Loaded configuration from: {self.config_file}")
            print(f"   📊 Found {len(self.config_df)} columns")
            return True

        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False

    def parse_er_elements(self) -> bool:
        """Parse tables, columns, and relationships from configuration"""
        if self.config_df is None:
            if not self.load_configuration():
                return False

        try:
            # Clean the data
            self.config_df['table_name'] = self.config_df['table_name'].str.strip()
            self.config_df['column_name'] = self.config_df['column_name'].str.strip()

            # Parse tables and their columns
            self.tables = {}
            table_names = self.config_df['table_name'].unique()

            print(f"🔍 Parsing {len(table_names)} tables...")

            for table_name in table_names:
                table_data = self.config_df[self.config_df['table_name'] == table_name]
                columns = []

                for _, row in table_data.iterrows():
                    column_info = {
                        'name': row['column_name'],
                        'data_type': row['data_type'],
                        'is_pk': bool(row.get('is_pk', False)),
                        'is_fk': bool(row.get('is_fk', False)),
                        'ref_table': row.get('ref_table'),
                        'ref_column': row.get('ref_column')
                    }
                    columns.append(column_info)

                self.tables[table_name] = columns

            # Parse relationships from FK definitions
            self.relationships = []
            fk_mask = (self.config_df['is_fk'] == True) & \
                      (self.config_df['ref_table'].notna()) & \
                      (self.config_df['ref_column'].notna())

            fk_data = self.config_df[fk_mask]

            for _, row in fk_data.iterrows():
                relationship = {
                    'source_table': row['table_name'],
                    'source_column': row['column_name'],
                    'target_table': row['ref_table'],
                    'target_column': row['ref_column']
                }
                self.relationships.append(relationship)

            print(f"✅ Parsed {len(self.tables)} tables and {len(self.relationships)} relationships")
            return True

        except Exception as e:
            print(f"❌ Error parsing ER elements: {e}")
            return False

    def generate_graphviz_er_diagram(self, output_file: str = "er_diagram",
                                     format: str = "png", style: str = "detailed") -> bool:
        """
        Generate ER diagram using Graphviz

        Args:
            output_file: Output file name without extension
            format: Output format (png, pdf, svg)
            style: Diagram style ('basic', 'detailed')
        """
        if not self.parse_er_elements():
            return False

        try:
            # Create a directed graph
            dot = graphviz.Digraph(comment='ER Diagram', format=format)

            # Graph attributes
            dot.attr(rankdir='TB', splines='ortho')
            dot.attr('graph', fontname='Arial', fontsize='12')
            dot.attr('node', fontname='Arial', fontsize='10')
            dot.attr('edge', fontname='Arial', fontsize='9')

            print("🎨 Creating Graphviz diagram...")

            for table_name, columns in self.tables.items():
                if style == 'detailed':
                    label = self._create_detailed_table_label(table_name, columns)
                else:
                    label = self._create_basic_table_label(table_name, columns)

                dot.node(table_name, label)

            # Create edges (relationships)
            for rel in self.relationships:
                dot.edge(
                    rel['source_table'],
                    rel['target_table'],
                    label=f"{rel['source_column']} → {rel['target_column']}",
                    style="dashed",
                    color="#E76F51"
                )

            # Render the diagram
            output_path = Path(output_file)
            dot.render(output_path.stem, cleanup=True, format=format)

            print(f"✅ Graphviz ER diagram generated: {output_path.stem}.{format}")
            return True

        except Exception as e:
            print(f"❌ Error generating Graphviz diagram: {e}")
            return False

    def _create_detailed_table_label(self, table_name: str, columns: List[Dict]) -> str:
        """Create detailed HTML table label for Graphviz"""
        pk_count = sum(1 for col in columns if col['is_pk'])
        fk_count = sum(1 for col in columns if col['is_fk'])
        regular_count = len(columns) - pk_count - fk_count

        label = f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4"> <TR><TD BGCOLOR="#2E86AB" COLSPAN="2"><FONT COLOR="white"><B>{table_name}</B></FONT></TD></TR><TR><TD BGCOLOR="#F8F9FA" COLSPAN="2"><I>Columns: {len(columns)} (PK: {pk_count}, FK: {fk_count})</I></TD></TR>'
        # Primary Keys
        for col in columns:
            if col['is_pk']:
                label += f'<TR><TD BGCOLOR="#FFF9C4" ALIGN="CENTER">🔑</TD><TD BGCOLOR="#FFF9C4">{col["name"]} <I>{col["data_type"]}</I></TD></TR>'

        # Foreign Keys
        for col in columns:
            if col['is_fk']:
                ref_info = f" → {col['ref_table']}.{col['ref_column']}" if col['ref_table'] else ""
                label += f'<TR><TD BGCOLOR="#C8E6C9" ALIGN="CENTER">🔗</TD><TD BGCOLOR="#C8E6C9">{col["name"]} <I>{col["data_type"]}</I>{ref_info}</TD></TR>'

        # Regular columns
        for col in columns:
            if not col['is_pk'] and not col['is_fk']:
                label += f'<TR><TD BGCOLOR="white" ALIGN="CENTER">•</TD><TD BGCOLOR="white">{col["name"]} <I>{col["data_type"]}</I></TD></TR>'

        label += '</TABLE>>'
        return label

    def _create_basic_table_label(self, table_name: str, columns: List[Dict]) -> str:
        """Create basic HTML table label for Graphviz"""
        label = f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3"><TR><TD BGCOLOR="lightblue" COLSPAN="1"><B>{table_name}</B></TD></TR>'


        for col in columns:
            if col['is_pk']:
                prefix = "🔑 "
            elif col['is_fk']:
                prefix = "🔗 "
            else:
                prefix = "• "

            label += f'<TR><TD BGCOLOR="white">{prefix}{col["name"]} <I>{col["data_type"]}</I></TD></TR>'

        label += '</TABLE>>'
        return label

    def generate_networkx_er_diagram(self, output_file: str = "er_diagram_networkx.png") -> bool:
        """
        Generate ER diagram using NetworkX and Matplotlib
        """
        if not self.parse_er_elements():
            return False

        try:
            # Create a directed graph
            G = nx.DiGraph()

            # Add nodes (tables)
            for table_name in self.tables.keys():
                G.add_node(table_name)

            # Add edges (relationships)
            for rel in self.relationships:
                G.add_edge(
                    rel['source_table'],
                    rel['target_table'],
                    label=f"{rel['source_column']} → {rel['ref_column']}"
                )

            # Create plot
            plt.figure(figsize=(16, 12))

            # Use spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_color='lightblue',
                node_size=4000,
                alpha=0.9,
                node_shape='s',
                edgecolors='black',
                linewidths=2
            )

            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color='gray',
                arrows=True,
                arrowsize=25,
                arrowstyle='->',
                connectionstyle="arc3,rad=0.1",
                width=2
            )

            # Draw node labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold',
                font_family='Arial'
            )

            # Draw edge labels
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8,
                font_family='Arial'
            )

            plt.title("Entity-Relationship Diagram (NetworkX)", size=16, pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ NetworkX ER diagram generated: {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error generating NetworkX diagram: {e}")
            return False

    def generate_er_analysis_report(self, output_file: str = "er_analysis_report.txt") -> bool:
        """
        Generate a comprehensive ER analysis report
        """
        if not self.parse_er_elements():
            return False

        try:
            report_lines = []

            # Header
            report_lines.append("ENTITY-RELATIONSHIP ANALYSIS REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Source: {self.config_file}")
            report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # Summary
            report_lines.append("SUMMARY")
            report_lines.append("-" * 30)
            report_lines.append(f"Total Tables: {len(self.tables)}")
            report_lines.append(f"Total Columns: {sum(len(cols) for cols in self.tables.values())}")
            report_lines.append(f"Total Relationships: {len(self.relationships)}")
            report_lines.append("")

            # Table Details
            report_lines.append("TABLE DETAILS")
            report_lines.append("-" * 30)
            for table_name, columns in self.tables.items():
                pk_cols = [col['name'] for col in columns if col['is_pk']]
                fk_cols = [col['name'] for col in columns if col['is_fk']]

                report_lines.append(f"📋 {table_name}")
                report_lines.append(f"   Total Columns: {len(columns)}")
                report_lines.append(f"   Primary Keys: {', '.join(pk_cols) if pk_cols else 'None'}")
                report_lines.append(f"   Foreign Keys: {', '.join(fk_cols) if fk_cols else 'None'}")
                report_lines.append("")

            # Relationship Details
            report_lines.append("RELATIONSHIP DETAILS")
            report_lines.append("-" * 30)
            for i, rel in enumerate(self.relationships, 1):
                report_lines.append(f"🔗 Relationship {i}:")
                report_lines.append(f"   Source: {rel['source_table']}.{rel['source_column']}")
                report_lines.append(f"   Target: {rel['target_table']}.{rel['target_column']}")
                report_lines.append("")

            # Data Quality Checks
            report_lines.append("DATA QUALITY CHECKS")
            report_lines.append("-" * 30)

            # Check 1: Tables without primary keys
            tables_without_pk = [name for name, cols in self.tables.items()
                                 if not any(col['is_pk'] for col in cols)]
            if tables_without_pk:
                report_lines.append("⚠️  TABLES WITHOUT PRIMARY KEYS:")
                for table in tables_without_pk:
                    report_lines.append(f"   - {table}")
                report_lines.append("")
            else:
                report_lines.append("✅ All tables have primary keys")
                report_lines.append("")

            # Check 2: Orphaned foreign keys
            orphaned_fks = []
            for rel in self.relationships:
                if rel['target_table'] not in self.tables:
                    orphaned_fks.append(
                        f"{rel['source_table']}.{rel['source_column']} → {rel['target_table']}.{rel['target_column']}")

            if orphaned_fks:
                report_lines.append("⚠️  ORPHANED FOREIGN KEYS:")
                for fk in orphaned_fks:
                    report_lines.append(f"   - {fk}")
                report_lines.append("")
            else:
                report_lines.append("✅ No orphaned foreign keys")
                report_lines.append("")

            # Check 3: Self-referencing tables
            self_refs = [rel for rel in self.relationships
                         if rel['source_table'] == rel['target_table']]
            if self_refs:
                report_lines.append("🔄 SELF-REFERENCING TABLES:")
                for rel in self_refs:
                    report_lines.append(f"   - {rel['source_table']}.{rel['source_column']} → {rel['target_column']}")
                report_lines.append("")

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))

            print(f"✅ ER analysis report generated: {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error generating analysis report: {e}")
            return False

    def generate_all_diagrams(self, output_dir: str = "er_diagrams") -> bool:
        """
        Generate all types of ER diagrams and reports
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            print("🚀 Generating comprehensive ER documentation...")
            print(f"📁 Output directory: {output_path.absolute()}")

            success_count = 0

            # Generate different diagram types
            if self.generate_graphviz_er_diagram(output_path / "er_diagram_basic", style="basic"):
                success_count += 1

            if self.generate_graphviz_er_diagram(output_path / "er_diagram_detailed", style="detailed"):
                success_count += 1

            if self.generate_networkx_er_diagram(output_path / "er_diagram_networkx.png"):
                success_count += 1

            if self.generate_er_analysis_report(output_path / "er_analysis_report.txt"):
                success_count += 1

            print(f"\n📊 Generation Summary: {success_count}/4 tasks completed successfully")

            if success_count > 0:
                print(f"✅ ER documentation generated in: {output_path}/")
                return True
            else:
                print("❌ No ER documentation could be generated")
                return False

        except Exception as e:
            print(f"❌ Error generating all diagrams: {e}")
            return False


def main():
    """Main function for standalone ER diagram generator"""
    parser = argparse.ArgumentParser(
        description='Generate ER diagrams from Excel configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate all diagrams and reports
  python er_diagram_generator.py --config config.xlsx

  # Generate specific output types
  python er_diagram_generator.py --config config.xlsx --graphviz --networkx --report

  # Custom output directory
  python er_diagram_generator.py --config config.xlsx --output my_er_diagrams

  # Specific diagram style
  python er_diagram_generator.py --config config.xlsx --graphviz --style detailed
        '''
    )

    parser.add_argument('--config', '-c', required=True,
                        help='Path to Excel configuration file')
    parser.add_argument('--output', '-o', default='er_diagrams',
                        help='Output directory for diagrams and reports')
    parser.add_argument('--graphviz', action='store_true',
                        help='Generate Graphviz ER diagram')
    parser.add_argument('--networkx', action='store_true',
                        help='Generate NetworkX ER diagram')
    parser.add_argument('--report', action='store_true',
                        help='Generate ER analysis report')
    parser.add_argument('--style', choices=['basic', 'detailed'], default='detailed',
                        help='Graphviz diagram style')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='Output format for Graphviz diagrams')

    args = parser.parse_args()

    print("🔷 ER Diagram Generator")
    print("=" * 50)

    # Initialize generator
    generator = ERDiagramGenerator(args.config)

    # Determine what to generate
    generate_all = not (args.graphviz or args.networkx or args.report)

    try:
        if generate_all:
            # Generate everything
            success = generator.generate_all_diagrams(args.output)
        else:
            # Generate specific items
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)

            success_count = 0

            if args.graphviz:
                output_file = output_path / f"er_diagram_{args.style}"
                if generator.generate_graphviz_er_diagram(output_file, args.format, args.style):
                    success_count += 1

            if args.networkx:
                output_file = output_path / "er_diagram_networkx.png"
                if generator.generate_networkx_er_diagram(output_file):
                    success_count += 1

            if args.report:
                output_file = output_path / "er_analysis_report.txt"
                if generator.generate_er_analysis_report(output_file):
                    success_count += 1

            success = success_count > 0
            print(f"📊 Generation Summary: {success_count} task(s) completed successfully")

        if success:
            print("\n🎉 ER diagram generation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ ER diagram generation failed!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

