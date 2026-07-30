"""Foreign-key resolution and referential integrity.

Extracted from ``data_generator.py`` as a mixin. ``DataGenerator`` inherits
it, so ``self`` resolves exactly as before — this is a pure move, not a
behaviour change. The split exists so each concern can be read and tested
without loading a 2,400-line class.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from sdp.models.config_models import RelationshipConfig

logger = logging.getLogger(__name__)


class RelationshipMixin:
    """Foreign-key resolution and referential integrity."""

    def _iter_relationship_groups(self) -> List[List[RelationshipConfig]]:
        grouped: Dict[Tuple[str, str, str], List[RelationshipConfig]] = {}

        for relationship in self.relationships:
            base_name = relationship.name or (
                f"{relationship.source_table}.{relationship.source_column}->{relationship.target_table}.{relationship.target_column}"
            )
            if "#" in base_name:
                base_name = base_name.rsplit("#", 1)[0]

            key = (relationship.source_table, relationship.target_table, base_name)
            grouped.setdefault(key, []).append(relationship)

        return list(grouped.values())

    def _is_one_to_one_fk(
        self,
        exemplar: RelationshipConfig,
        child_table: str,
        source_columns: List[str],
    ) -> bool:
        """True when the FK must hold *unique* values — a one-to-one relationship.

        That happens when the child FK column(s) are exactly the child table's
        primary key (so the FK value cannot repeat), or when the relationship
        explicitly declares ``relationship_type`` one_to_one.
        """
        declared = str(getattr(exemplar, "relationship_type", "") or "").strip().lower()
        if declared in ("one_to_one", "one-to-one", "1:1"):
            return True
        cfg = self.tables_config.get(child_table)
        if cfg is None:
            return False
        pk_cols = {c.column_name for c in cfg.columns if getattr(c, "is_pk", False)}
        # One-to-one only when the FK columns *are* the whole primary key.
        return bool(pk_cols) and pk_cols == set(source_columns)

    def _sample_fk_values(
        self,
        parent_values: List[Any],
        n: int,
        one_to_one: bool,
        label: str,
    ) -> List[Any]:
        """Pick ``n`` FK values from the parent key pool.

        ``one_to_one`` → sample *without* replacement, so each child row gets a
        distinct parent key and the child primary key stays unique. Otherwise
        sample *with* replacement (one parent row, many child rows).
        """
        if not one_to_one:
            return random.choices(parent_values, k=n)
        distinct = list(dict.fromkeys(parent_values))   # unique, order-preserving
        if len(distinct) >= n:
            return random.sample(distinct, k=n)
        # Not enough distinct parent keys for a true 1:1 — a config-size mismatch.
        # Use every distinct key, top up with repeats, and warn loudly.
        self.logger.warning(
            f"⚠️ One-to-one FK {label}: parent has only {len(distinct)} distinct "
            f"key(s) for {n} child rows — {n - len(distinct)} row(s) cannot be unique."
        )
        pool = distinct + random.choices(distinct, k=n - len(distinct))
        random.shuffle(pool)
        return pool

    def _apply_relationship_group(
        self,
        data: Dict[str, pd.DataFrame],
        relationship_group: List[RelationshipConfig],
    ) -> None:
        exemplar = relationship_group[0]
        parent_table = exemplar.target_table
        child_table = exemplar.source_table

        if parent_table not in data or child_table not in data:
            return

        # Anchor tables hold real data — never rewrite their foreign keys.
        if self._is_anchor_table(child_table):
            return

        parent_df = data[parent_table]
        child_df = data[child_table]
        source_columns = [relationship.source_column for relationship in relationship_group]
        target_columns = [relationship.target_column for relationship in relationship_group]

        if any(column not in parent_df.columns for column in target_columns):
            return
        if any(column not in child_df.columns for column in source_columns):
            return

        one_to_one = self._is_one_to_one_fk(exemplar, child_table, source_columns)

        if len(relationship_group) == 1:
            valid_parent_values = parent_df[target_columns[0]].dropna().tolist()
            if valid_parent_values:
                child_df[source_columns[0]] = self._sample_fk_values(
                    valid_parent_values, len(child_df), one_to_one,
                    f"{child_table}.{source_columns[0]}",
                )
                data[child_table] = child_df
            return

        parent_pairs = parent_df[target_columns].dropna().drop_duplicates()
        if parent_pairs.empty:
            return

        # One-to-one composite FK: each child row needs a distinct parent
        # key-combination — sample without replacement when there are enough.
        replace = True
        if one_to_one and len(parent_pairs) >= len(child_df):
            replace = False
        elif one_to_one:
            self.logger.warning(
                f"⚠️ One-to-one FK {child_table}.{tuple(source_columns)}: parent has "
                f"only {len(parent_pairs)} distinct key combinations for "
                f"{len(child_df)} child rows — uniqueness cannot be fully honoured."
            )
        sampled_parent_rows = parent_pairs.sample(n=len(child_df), replace=replace).reset_index(drop=True)
        child_df = child_df.copy()
        for source_column, target_column in zip(source_columns, target_columns):
            child_df[source_column] = sampled_parent_rows[target_column].tolist()
        data[child_table] = child_df

    def _enforce_relationships_in_sample(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enforce relationships in sample data for better SDV learning"""
        for relationship_group in self.sdv_relationship_groups:
            self._apply_relationship_group(sample_data, relationship_group)

        return sample_data

    def _enforce_all_relationships(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enforce ALL relationships in data"""
        for _ in range(3):
            for relationship_group in self._iter_relationship_groups():
                self._apply_relationship_group(data, relationship_group)

        return data

    def _resolve_foreign_keys(self) -> None:
        """Resolve foreign key relationships in generated_data"""
        if not self.relationships:
            return

        self.logger.info("🔗 Resolving foreign key relationships...")
        resolved_count = 0

        for relationship_group in self._iter_relationship_groups():
            exemplar = relationship_group[0]
            src_t = exemplar.source_table
            tgt_t = exemplar.target_table

            if src_t not in self.generated_data or tgt_t not in self.generated_data:
                continue

            src_df = self.generated_data[src_t]
            original_dtypes = {
                relationship.source_column: src_df[relationship.source_column].dtype
                for relationship in relationship_group
                if relationship.source_column in src_df.columns
            }
            before_frame = src_df[[column for column in original_dtypes]].copy() if original_dtypes else pd.DataFrame()

            self._apply_relationship_group(self.generated_data, relationship_group)
            src_df = self.generated_data[src_t]

            for source_column, original_dtype in original_dtypes.items():
                try:
                    src_df[source_column] = src_df[source_column].astype(original_dtype)
                except (ValueError, TypeError):
                    pass

            if not before_frame.empty and not before_frame.equals(src_df[list(original_dtypes)]):
                resolved_count += 1
                source_columns = ", ".join(rel.source_column for rel in relationship_group)
                target_columns = ", ".join(rel.target_column for rel in relationship_group)
                self.logger.info(f" ✅ Resolved FK: {src_t}.{source_columns} → {tgt_t}.{target_columns}")

        self.logger.info(f"✅ Resolved {resolved_count} foreign key relationships")
