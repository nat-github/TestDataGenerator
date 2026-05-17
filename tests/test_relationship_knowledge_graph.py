"""Tests for the opt-in knowledge-graph relationship inferrer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sdp.ml.relationship_feedback_store import FeedbackStore
from sdp.ml.relationship_knowledge_graph import KnowledgeGraphRelationshipInferrer
from sdp.ml.semantic_profile import (
    BANKING_PROFILE,
    GENERIC_PROFILE,
    SemanticProfile,
    default_profile,
)
from sdp.models.config_models import ColumnConfig, TableConfig
from sdp.utils.config_parser import ConfigParser


REPO_ROOT = Path(__file__).resolve().parents[1]
_CREDITCARD_CONFIG = REPO_ROOT / "config" / "Creditcard_no_rel.xlsx"


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


@pytest.mark.skipif(
    not _CREDITCARD_CONFIG.exists(),
    reason="config/Creditcard_no_rel.xlsx fixture is not present in the repo",
)
def test_creditcard_config_knowledge_graph_filters_lookup_noise(tmp_path: Path):
    parser = ConfigParser(str(_CREDITCARD_CONFIG))
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


# ---------------------------------------------------------------------------
# Inclusion dependency — the data-led reliability gate
# ---------------------------------------------------------------------------

def test_inclusion_dependency_breaks_a_name_tie(tmp_path: Path):
    """When two parents are an equally good name match, real data decides.

    `account_type` and `account_status` both score 1.0 on name similarity for
    a child column called `account_code`. Only `account_type` actually contains
    the child's values, so the knowledge graph must pick it and drop the other.
    """
    account_type = _table("account_type", [
        _col("CODE", "A4", is_pk=True, table="account_type"),
        _col("DESCRIPTION", "A64", table="account_type"),
    ])
    account_status = _table("account_status", [
        _col("CODE", "A4", is_pk=True, table="account_status"),
        _col("DESCRIPTION", "A64", table="account_status"),
    ])
    events = _table("events", [
        _col("EVENT_ID", "N10", is_pk=True, table="events"),
        _col("ACCOUNT_CODE", "A4", is_fk=True, table="events"),
    ], num_rows=4)

    sample_data = {
        "account_type": pd.DataFrame({"CODE": ["AA", "BB", "CC"]}),
        "account_status": pd.DataFrame({"CODE": ["X1", "X2"]}),
        "events": pd.DataFrame({
            "EVENT_ID": [1, 2, 3, 4],
            "ACCOUNT_CODE": ["AA", "BB", "AA", "CC"],
        }),
    }

    inferrer = KnowledgeGraphRelationshipInferrer(
        confidence_threshold=0.55,
        feedback_store=FeedbackStore(tmp_path / "fb.jsonl"),
    )
    result = inferrer.infer(
        {"account_type": account_type, "account_status": account_status, "events": events},
        sample_data=sample_data,
    )

    pairs = {
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in result.relationships
    }
    assert ("events", "ACCOUNT_CODE", "account_type", "CODE") in pairs
    # account_status shares the values of nobody — the inclusion gate drops it.
    assert ("events", "ACCOUNT_CODE", "account_status", "CODE") not in pairs

    rel = next(r for r in result.relationships if r.source_column == "ACCOUNT_CODE")
    assert rel.inference_signals["kg_inclusion_evidence"] == 1.0
    assert rel.inference_signals["kg_value_inclusion"] >= 0.99


# ---------------------------------------------------------------------------
# FK-cycle resolution
# ---------------------------------------------------------------------------

def test_knowledge_graph_breaks_fk_cycles(tmp_path: Path):
    """A mutual FK cycle between two tables must not survive intact."""
    account = _table("account", [
        _col("ACCOUNT_ID", "N10", is_pk=True, table="account"),
        _col("BRANCH_REF", "N10", is_fk=True, table="account"),
    ])
    branch = _table("branch", [
        _col("BRANCH_ID", "N10", is_pk=True, table="branch"),
        _col("ACCOUNT_REF", "N10", is_fk=True, table="branch"),
    ])

    inferrer = KnowledgeGraphRelationshipInferrer(
        confidence_threshold=0.55,
        feedback_store=FeedbackStore(tmp_path / "fb.jsonl"),
    )
    result = inferrer.infer({"account": account, "branch": branch})

    table_pairs = {(r.source_table, r.target_table) for r in result.relationships}
    # Both directions cannot coexist — that would be an account<->branch cycle.
    assert not ({("account", "branch"), ("branch", "account")} <= table_pairs)


# ---------------------------------------------------------------------------
# Genericity — the graph machinery does not depend on banking vocabulary
# ---------------------------------------------------------------------------

def test_default_profile_merges_generic_and_banking():
    merged = default_profile()
    # Every generic + banking alias survives the merge.
    for key in GENERIC_PROFILE.aliases:
        assert key in merged.aliases
    for key in BANKING_PROFILE.aliases:
        assert key in merged.aliases
    assert "ebx" in merged.stopwords          # banking stopword
    assert "id" in merged.generic_key_names    # generic key name


def test_inferrer_works_with_generic_only_profile(tmp_path: Path):
    """A non-banking schema infers correctly with zero domain vocabulary."""
    customer = _table("customer", [
        _col("CUSTOMER_ID", "N10", is_pk=True, table="customer"),
        _col("CUSTOMER_NAME", "A64", table="customer"),
    ])
    order = _table("order", [
        _col("ORDER_ID", "N10", is_pk=True, table="order"),
        _col("CUSTOMER_ID", "N10", is_fk=True, table="order"),
    ], num_rows=500)

    inferrer = KnowledgeGraphRelationshipInferrer(
        confidence_threshold=0.55,
        feedback_store=FeedbackStore(tmp_path / "fb.jsonl"),
        profile=GENERIC_PROFILE,  # no banking vocabulary at all
    )
    result = inferrer.infer({"customer": customer, "order": order})

    pairs = {
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in result.relationships
    }
    assert ("order", "CUSTOMER_ID", "customer", "CUSTOMER_ID") in pairs

