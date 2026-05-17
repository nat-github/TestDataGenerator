"""Tests for the `infer-relationships` and `record-feedback` CLI subcommands.

These exercise the wiring between argparse → run_* dispatchers → ML modules,
without invoking a subprocess. We call ``main([...])`` directly and assert on
the files it writes and on the feedback store side-effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from main import main, parse_arguments
from sdp.ml.relationship_feedback_store import FeedbackStore


def _two_table_yaml(path: Path) -> Path:
    """Write a minimal sdp-yaml-v1 config with an obvious orders -> customers FK
    that the inferrer should discover."""
    cfg: dict[str, Any] = {
        "config_format": "sdp-yaml-v1",
        "run_settings": {"default_records_per_table": 50},
        "tables": [
            {
                "name": "customers",
                "rows": 50,
                "primary_key_columns": ["customer_id"],
                "columns": [
                    {"name": "customer_id", "data_type": "N10", "is_pk": True, "nullable": False},
                    {"name": "name", "data_type": "VA64"},
                ],
            },
            {
                "name": "orders",
                "rows": 100,
                "primary_key_columns": ["order_id"],
                "columns": [
                    {"name": "order_id", "data_type": "N10", "is_pk": True, "nullable": False},
                    {"name": "customer_id", "data_type": "N10"},
                    {"name": "amount", "data_type": "DC"},
                ],
            },
        ],
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Argparse — flag plumbing
# ---------------------------------------------------------------------------


def test_argparse_infer_relationships_defaults_to_ml():
    args = parse_arguments([
        "infer-relationships",
        "--config", "x.yaml",
        "--config-output", "y.yaml",
    ])
    assert args.command == "infer-relationships"
    assert args.method == "ml"
    assert args.ml_mode == "standard"
    assert args.simple_yaml is None
    assert args.ml_confidence == pytest.approx(0.55)
    assert args.llm_confidence == pytest.approx(0.7)


def test_argparse_infer_relationships_accepts_knowledge_graph_mode():
    args = parse_arguments([
        "infer-relationships",
        "--config", "x.yaml",
        "--config-output", "y.yaml",
        "--ml-mode", "knowledge-graph",
    ])
    assert args.command == "infer-relationships"
    assert args.ml_mode == "knowledge-graph"


def test_argparse_infer_relationships_accepts_simple_yaml_flag():
    args = parse_arguments([
        "infer-relationships",
        "--config", "x.yaml",
        "--config-output", "y.yaml",
        "--simple-yaml",
    ])
    assert args.command == "infer-relationships"
    assert args.simple_yaml is True


def test_argparse_infer_relationships_accepts_review_yaml_flag():
    args = parse_arguments([
        "infer-relationships",
        "--config", "x.yaml",
        "--config-output", "y.yaml",
        "--review-yaml",
    ])
    assert args.command == "infer-relationships"
    assert args.simple_yaml is False


def test_argparse_record_feedback_requires_both_files():
    args = parse_arguments([
        "record-feedback",
        "--inferred", "a.yaml",
        "--reviewed", "b.yaml",
        "--feedback-store", "fb.jsonl",
    ])
    assert args.command == "record-feedback"
    assert args.inferred == "a.yaml"
    assert args.reviewed == "b.yaml"
    assert args.feedback_store == "fb.jsonl"


# ---------------------------------------------------------------------------
# infer-relationships end-to-end
# ---------------------------------------------------------------------------


def test_infer_relationships_writes_reviewable_yaml(tmp_path: Path, capsys):
    config_path = _two_table_yaml(tmp_path / "in.yaml")
    out_path = tmp_path / "out.yaml"
    fb_path = tmp_path / "fb.jsonl"

    rc = main([
        "infer-relationships",
        "--config", str(config_path),
        "--config-output", str(out_path),
        "--feedback-store", str(fb_path),
        "--method", "ml",
    ])
    assert rc == 0
    assert out_path.exists()

    doc = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert doc.get("config_format") == "sdp-yaml-v1"
    rels = doc.get("relationships") or []
    # Should contain at least one inferred FK with the ML annotation
    inferred = [r for r in rels if r.get("inferred_by_ml")]
    assert inferred, f"expected at least one inferred_by_ml entry; got {rels}"
    target = next(
        r for r in inferred
        if r["source_table"] == "orders"
        and "customer_id" in (r.get("source_columns") or [])
    )
    assert target["target_table"] == "customers"
    assert "customer_id" in target["target_columns"]
    assert 0 < target["ml_confidence"] <= 1
    # SME instructions present so reviewers know what to do
    assert "_review_metadata" in doc
    assert doc["_review_metadata"]["summary"]["average_confidence"] is not None
    assert doc["_review_metadata"]["summary"]["recommendation"]
    assert target["confidence_band"] in {"high", "medium", "low"}
    assert target["review_recommendation"]
    assert "_review_debug" in doc
    out = capsys.readouterr().out
    assert "Inference summary" in out
    assert "ML mode:                standard" in out
    assert "Average confidence:" in out
    assert "Recommendation:" in out


def test_infer_relationships_emits_er_diagram(tmp_path: Path):
    config_path = _two_table_yaml(tmp_path / "in.yaml")
    out_path = tmp_path / "out.yaml"
    er_path = tmp_path / "diagram.mmd"
    fb_path = tmp_path / "fb.jsonl"

    rc = main([
        "infer-relationships",
        "--config", str(config_path),
        "--config-output", str(out_path),
        "--er-output", str(er_path),
        "--feedback-store", str(fb_path),
        "--method", "ml",
    ])
    assert rc == 0
    assert er_path.exists()
    text = er_path.read_text(encoding="utf-8")
    # Mermaid ER diagrams open with `erDiagram` — sanity check it's not empty
    assert text.strip(), "ER diagram file is empty"


def test_infer_relationships_can_write_simple_yaml(tmp_path: Path):
    config_path = _two_table_yaml(tmp_path / "in.yaml")
    out_path = tmp_path / "simple.yaml"
    fb_path = tmp_path / "fb.jsonl"

    rc = main([
        "infer-relationships",
        "--config", str(config_path),
        "--config-output", str(out_path),
        "--feedback-store", str(fb_path),
        "--method", "ml",
        "--simple-yaml",
    ])
    assert rc == 0

    doc = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert doc.get("config_format") == "sdp-yaml-v1"
    assert "tables" in doc
    assert "relationships" in doc
    assert "_review_metadata" not in doc
    assert "_review_debug" not in doc

    customers = next(t for t in doc["tables"] if t["name"] == "customers")
    assert customers["primary_key_columns"] == ["customer_id"]
    assert any(col["name"] == "customer_id" for col in customers["columns"])

    rel = next(r for r in doc["relationships"] if r["source_table"] == "orders")
    assert rel["target_table"] == "customers"
    assert "ml_confidence" not in rel
    assert "review_status" not in rel
    assert "review_recommendation" not in rel


def test_knowledge_graph_mode_defaults_to_simple_yaml(tmp_path: Path):
    config_path = _two_table_yaml(tmp_path / "in.yaml")
    out_path = tmp_path / "kg-simple.yaml"
    fb_path = tmp_path / "fb.jsonl"

    rc = main([
        "infer-relationships",
        "--config", str(config_path),
        "--config-output", str(out_path),
        "--feedback-store", str(fb_path),
        "--method", "ml",
        "--ml-mode", "knowledge-graph",
    ])
    assert rc == 0

    doc = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert doc.get("config_format") == "sdp-yaml-v1"
    assert "tables" in doc
    assert "relationships" in doc
    assert "_review_metadata" not in doc
    assert "_review_debug" not in doc


def test_knowledge_graph_mode_can_still_force_review_yaml(tmp_path: Path):
    config_path = _two_table_yaml(tmp_path / "in.yaml")
    out_path = tmp_path / "kg-review.yaml"
    fb_path = tmp_path / "fb.jsonl"

    rc = main([
        "infer-relationships",
        "--config", str(config_path),
        "--config-output", str(out_path),
        "--feedback-store", str(fb_path),
        "--method", "ml",
        "--ml-mode", "knowledge-graph",
        "--review-yaml",
    ])
    assert rc == 0

    doc = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert doc.get("config_format") == "sdp-yaml-v1"
    assert "_review_metadata" in doc
    assert "relationships" in doc


# ---------------------------------------------------------------------------
# record-feedback end-to-end
# ---------------------------------------------------------------------------


def test_record_feedback_persists_accept_reject_and_added(tmp_path: Path, capsys):
    """Round-trip: produce an inferred YAML, hand-edit it (drop one, add one),
    feed both to record-feedback, and verify the store reflects the SME's edits."""
    fb_path = tmp_path / "fb.jsonl"

    inferred_doc = {
        "config_format": "sdp-yaml-v1",
        "relationships": [
            {
                # SME will keep this one → counted as accept
                "name": "orders_customer_id_to_customers_customer_id",
                "source_table": "orders", "source_columns": ["customer_id"],
                "target_table": "customers", "target_columns": ["customer_id"],
                "relationship_type": "many_to_one", "active": True,
                "inferred_by_ml": True, "ml_confidence": 0.82,
                "inference_signals": {"name_similarity": 1.0, "type_compatibility": 1.0,
                                       "value_subset": 1.0, "pk_likeness": 1.0},
            },
            {
                # SME will delete this one → counted as reject
                "name": "orders_amount_to_customers_customer_id",
                "source_table": "orders", "source_columns": ["amount"],
                "target_table": "customers", "target_columns": ["customer_id"],
                "relationship_type": "many_to_one", "active": True,
                "inferred_by_ml": True, "ml_confidence": 0.56,
                "inference_signals": {"name_similarity": 0.2, "type_compatibility": 1.0,
                                       "value_subset": 0.3, "pk_likeness": 1.0},
            },
        ],
    }
    reviewed_doc = {
        "config_format": "sdp-yaml-v1",
        "relationships": [
            inferred_doc["relationships"][0],  # kept
            {
                # Brand-new entry the SME added — counted as SME-added
                "name": "orders_billing_id_to_customers_customer_id",
                "source_table": "orders", "source_columns": ["billing_id"],
                "target_table": "customers", "target_columns": ["customer_id"],
                "relationship_type": "many_to_one", "active": True,
            },
        ],
    }
    inferred_path = tmp_path / "inferred.yaml"
    reviewed_path = tmp_path / "reviewed.yaml"
    inferred_path.write_text(yaml.safe_dump(inferred_doc, sort_keys=False), encoding="utf-8")
    reviewed_path.write_text(yaml.safe_dump(reviewed_doc, sort_keys=False), encoding="utf-8")

    rc = main([
        "record-feedback",
        "--inferred", str(inferred_path),
        "--reviewed", str(reviewed_path),
        "--feedback-store", str(fb_path),
    ])
    assert rc == 0
    assert fb_path.exists()

    entries = FeedbackStore(fb_path).load()
    # 1 accept + 1 reject + 1 SME-added = 3 records
    assert len(entries) == 3

    accepted = [e for e in entries if e.accepted]
    rejected = [e for e in entries if not e.accepted]
    assert len(accepted) == 2
    assert len(rejected) == 1

    # The rejected one is the orders.amount → customers.customer_id pair
    assert rejected[0].source_column == "amount"

    # The SME-added entry has empty signals (only contributes to pattern memory)
    sme_added = [e for e in accepted if not e.signals]
    assert len(sme_added) == 1
    assert sme_added[0].source_column == "billing_id"

    out = capsys.readouterr().out
    assert "Accepted (kept):  1" in out
    assert "Rejected (gone):  1" in out
    assert "SME-added:        1" in out


def test_record_feedback_round_trip_with_inferrer_output(tmp_path: Path):
    """End-to-end: run infer-relationships → SME removes everything → record-feedback
    should record those as rejections, no exceptions."""
    config_path = _two_table_yaml(tmp_path / "in.yaml")
    inferred_path = tmp_path / "inferred.yaml"
    fb_path = tmp_path / "fb.jsonl"

    rc = main([
        "infer-relationships",
        "--config", str(config_path),
        "--config-output", str(inferred_path),
        "--feedback-store", str(fb_path),
        "--method", "ml",
    ])
    assert rc == 0

    # SME rejects everything: hand them an empty relationships list
    reviewed_doc = {"config_format": "sdp-yaml-v1", "relationships": []}
    reviewed_path = tmp_path / "reviewed.yaml"
    reviewed_path.write_text(yaml.safe_dump(reviewed_doc, sort_keys=False), encoding="utf-8")

    rc = main([
        "record-feedback",
        "--inferred", str(inferred_path),
        "--reviewed", str(reviewed_path),
        "--feedback-store", str(fb_path),
    ])
    assert rc == 0

    entries = FeedbackStore(fb_path).load()
    # All entries should be rejections
    assert entries
    assert all(not e.accepted for e in entries)
