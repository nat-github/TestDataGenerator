"""Tests for the opt-in knowledge-graph relationship inferrer."""
from __future__ import annotations

from pathlib import Path

from ml.relationship_feedback_store import FeedbackStore
from ml.relationship_knowledge_graph import KnowledgeGraphRelationshipInferrer
from models.config_models import ColumnConfig, TableConfig
from utils.config_parser import ConfigParser


REPO_ROOT = Path(__file__).resolve().parents[1]


def _col(name: str, dtype: str = "N10", *, is_pk: bool = False, is_fk: bool = False, table: str = "t") -> ColumnConfig:
    return ColumnConfig(
        table_name=table,
        column_name=name,
        data_type=dtype,
        is_pk=is_pk,
        is_fk=is_fk,
        nullable=not is_pk,
    )


def _table(name: str, columns: list[ColumnConfig], num_rows: int = 100) -> TableConfig:
    return TableConfig(name=name, columns=columns, num_rows=num_rows)


def test_knowledge_graph_prefers_semantic_lookup_targets(tmp_path: Path):
    acct_type = _table("acct_type", [
        _col("CODE", "A4", is_pk=True, table="acct_type"),
        _col("DESCRIPTION", "A64", table="acct_type"),
    ])
    input_mode = _table("input_mode", [
        _col("CODE", "A2", is_pk=True, table="input_mode"),
        _col("DESCRIPTION", "A64", table="input_mode"),
    ])
    terminal_type = _table("terminal_type", [
        _col("CODE", "A3", is_pk=True, table="terminal_type"),
        _col("DESCRIPTION", "A64", table="terminal_type"),
    ])
    fact = _table("card_transactions", [
        _col("TXN_ID", "N10", is_pk=True, table="card_transactions"),
        _col("CARD_AC_TP_CODE", "A4", is_fk=True, table="card_transactions"),
        _col("CARD_INPT_MODE_CODE", "A2", is_fk=True, table="card_transactions"),
        _col("CARD_TXN_TRMNL_TP_CODE", "A3", is_fk=True, table="card_transactions"),
    ], num_rows=1000)

    inferrer = KnowledgeGraphRelationshipInferrer(
        confidence_threshold=0.55,
        feedback_store=FeedbackStore(tmp_path / "fb.jsonl"),
    )
    result = inferrer.infer({
        "acct_type": acct_type,
        "input_mode": input_mode,
        "terminal_type": terminal_type,
        "card_transactions": fact,
    })

    pairs = {
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in result.relationships
    }
    assert ("card_transactions", "CARD_AC_TP_CODE", "acct_type", "CODE") in pairs
    assert ("card_transactions", "CARD_INPT_MODE_CODE", "input_mode", "CODE") in pairs
    assert ("card_transactions", "CARD_TXN_TRMNL_TP_CODE", "terminal_type", "CODE") in pairs
    # The KG mode should suppress CODE->CODE links between lookup tables.
    assert all(pair[0] == "card_transactions" for pair in pairs)


def test_creditcard_config_knowledge_graph_filters_lookup_noise(tmp_path: Path):
    parser = ConfigParser(str(REPO_ROOT / "config" / "Creditcard_no_rel.xlsx"))
    assert parser.load_config() is True
    tables = parser.parse_tables()
    parser.parse_relationships()

    inferrer = KnowledgeGraphRelationshipInferrer(
        confidence_threshold=0.55,
        feedback_store=FeedbackStore(tmp_path / "creditcard_fb.jsonl"),
    )
    result = inferrer.infer(tables)

    pairs = {
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in result.relationships
    }

    expected_subset = {
        ("dc_trx_clrg_issng_dl", "CARD_AC_TP_CODE", "ebx_clearing_acct_type", "CODE"),
        ("dc_trx_clrg_issng_dl", "CARD_AC_PD_CODE", "ebx_clearing_acct_product", "CODE"),
        ("dc_trx_clrg_issng_dl", "CARD_INPT_MODE_CODE", "ebx_clearing_card_input_mode", "CODE"),
        ("dc_trx_clrg_issng_dl", "CARD_TXN_TRMNL_TP_CODE", "ebx_clearing_terminal_type", "CODE"),
        ("dc_trx_clrg_issng_dl", "CARD_TXN_MRCH_CAT_CODE", "ebx_cards_merchant_category", "CODE"),
        ("dc_trx_clrg_issng_dl", "CARD_TXN_TP_CODE", "ebx_clearing_card_trns_type", "CODE"),
    }
    assert expected_subset.issubset(pairs)

    # No relationship should originate from the lookup tables themselves.
    assert all(source == "dc_trx_clrg_issng_dl" for source, _, _, _ in pairs)

    # The new signals should be present for auditability.
    rel = next(r for r in result.relationships if r.source_column == "CARD_AC_TP_CODE")
    assert rel.inference_signals is not None
    assert "kg_table_semantic_similarity" in rel.inference_signals
    assert "kg_base_confidence" in rel.inference_signals

