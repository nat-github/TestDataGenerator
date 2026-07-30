"""`infer-relationships` and `record-feedback` commands.

Relationship inference has the largest helper surface of any command
group — inference summaries, reviewable YAML, ER output, feedback
round-tripping.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from sdp.services.common import (
    configure_logging,
    validate_config_file,
)

logger = logging.getLogger(__name__)


def run_infer_relationships(args) -> int:
    """Infer FK relationships from a config and emit ER diagram + reviewable YAML."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.utils.config_parser import ConfigParser
        from sdp.ml.relationship_inferrer import MLRelationshipInferrer
        from sdp.ml.relationship_knowledge_graph import KnowledgeGraphRelationshipInferrer
        from sdp.ml.relationship_feedback_store import FeedbackStore

        if not validate_config_file(args.config):
            return 1

        parser = ConfigParser(args.config)
        if not parser.load_config():
            logger.error("Failed to load config")
            return 1
        tables = parser.parse_tables()
        existing = parser.parse_relationships()

        inferred: list = []
        method = args.method
        sample_data = _load_relationship_sample_data(getattr(args, "sample_data", None), tables)

        if method in ("ml", "both"):
            store = FeedbackStore(args.feedback_store) if args.feedback_store else FeedbackStore()
            inferrer_cls = KnowledgeGraphRelationshipInferrer if getattr(args, "ml_mode", "standard") == "knowledge-graph" else MLRelationshipInferrer
            ml_inferrer = inferrer_cls(
                confidence_threshold=args.ml_confidence,
                feedback_store=store,
            )
            ml_result = ml_inferrer.infer(
                tables,
                existing_relationships=existing,
                sample_data=sample_data,
            )
            inferred.extend(ml_result.relationships)
            logger.info(
                f"ML inferred {len(ml_result.relationships)} relationship(s) "
                f"using mode={getattr(args, 'ml_mode', 'standard')}; "
                f"classifier_fitted={ml_result.classifier_fitted} "
                f"(n={ml_result.classifier_examples} feedback examples)"
            )

        if method in ("llm", "both"):
            try:
                from sdp.llm.relationship_inferrer import RelationshipInferrer as LLMInferrer
                llm = LLMInferrer(confidence_threshold=args.llm_confidence)
                # When method=both, only ask LLM about relationships ML missed
                seen_pairs = {(r.source_table, r.source_column, r.target_table, r.target_column)
                              for r in inferred} if method == "both" else set()
                llm_result = llm.infer(tables, existing_relationships=existing)
                for rel in llm_result.relationships:
                    pair = (rel.source_table, rel.source_column, rel.target_table, rel.target_column)
                    if pair not in seen_pairs:
                        inferred.append(rel)
                logger.info(f"LLM contributed {len(llm_result.relationships)} relationship(s)")
            except Exception as exc:
                logger.warning(f"LLM inference unavailable, continuing with ML only: {exc}")

        inference_summary = _build_inference_summary(
            inferred,
            ml_mode=getattr(args, "ml_mode", "standard") if method in ("ml", "both") else None,
        )

        # Write YAML output
        output_path = Path(args.config_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simple_yaml = _should_write_simple_yaml(args)
        if simple_yaml:
            _write_simple_relationship_yaml(output_path, tables, existing, inferred)
        else:
            _write_reviewable_yaml(output_path, existing, inferred, inference_summary)
        yaml_mode = "simple" if simple_yaml else "review"
        logger.info(f"✅ Wrote {yaml_mode} YAML: {output_path}")

        # Optional ER diagram
        if args.er_output:
            _write_er_diagram(Path(args.er_output), tables, existing + inferred)

        # Print summary
        print("\n=== Inference summary ===")
        print(f"Method:                 {method}")
        if method in ("ml", "both"):
            print(f"ML mode:                {getattr(args, 'ml_mode', 'standard')}")
        print(f"Existing relationships: {len(existing)}")
        print(f"Inferred relationships: {len(inferred)}")
        if inference_summary.get("average_confidence") is not None:
            print(f"Average confidence:     {inference_summary['average_confidence']:.2f}")
        print(f"Recommendation:         {inference_summary['recommendation']}")
        for rel in inferred:
            conf = rel.ml_confidence if rel.inferred_by_ml else rel.llm_confidence
            tag = "ML" if rel.inferred_by_ml else "LLM"
            print(f"  [{tag} {conf:.2f}] {rel.source_table}.{rel.source_column} -> {rel.target_table}.{rel.target_column}")
        return 0
    except Exception as exc:
        logger.error(f"infer-relationships failed: {exc}")
        logger.debug('Full traceback:', exc_info=True)
        return 1

def _relationship_confidence(rel) -> Optional[float]:
    return rel.ml_confidence if rel.inferred_by_ml else rel.llm_confidence

def _should_write_simple_yaml(args) -> bool:
    """Resolve the effective YAML mode for infer-relationships.

    Default behaviour remains unchanged for the standard ML path, but the
    opt-in knowledge-graph mode now defaults to a cleaner, simple YAML unless
    callers explicitly request the richer review document via --review-yaml.
    """
    explicit = getattr(args, "simple_yaml", None)
    if explicit is not None:
        return bool(explicit)
    return getattr(args, "ml_mode", "standard") == "knowledge-graph"

def _confidence_band(confidence: Optional[float]) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"

def _relationship_recommendation(confidence: Optional[float]) -> str:
    band = _confidence_band(confidence)
    if band == "high":
        return "Spot-check, then likely keep"
    if band == "medium":
        return "Review before accepting"
    if band == "low":
        return "Review carefully or reject"
    return "Manual review required"

def _build_inference_summary(inferred_rels, ml_mode: Optional[str] = None) -> Dict[str, object]:
    confidences = [c for c in (_relationship_confidence(r) for r in inferred_rels) if c is not None]
    average_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    bands = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for rel in inferred_rels:
        bands[_confidence_band(_relationship_confidence(rel))] += 1

    recommendation = "No inferred relationships. Review schema hints or provide sample data."
    if inferred_rels:
        if bands["low"] == 0 and (average_confidence or 0.0) >= 0.80:
            recommendation = "High-confidence set. Spot-check key relationships, then keep the rest if they look right."
        elif bands["low"] <= max(1, len(inferred_rels) // 4):
            recommendation = "Mostly medium/high confidence. Review the medium-confidence relationships before accepting."
        else:
            recommendation = "Several low-confidence relationships exist. Review every inferred relationship carefully."
        if ml_mode == "standard" and bands["low"] > 0:
            recommendation += " If ambiguity remains, try --ml-mode knowledge-graph."

    return {
        "average_confidence": average_confidence,
        "confidence_bands": bands,
        "recommendation": recommendation,
    }

def _relationship_sort_key(rel) -> tuple:
    source_cols = rel.get("source_columns") or [rel.get("source_column") or ""]
    target_cols = rel.get("target_columns") or [rel.get("target_column") or ""]
    return (
        str(rel.get("source_table", "")),
        str(source_cols[0]),
        str(rel.get("target_table", "")),
        str(target_cols[0]),
        str(rel.get("name", "")),
    )

def _build_review_relationship_entry(rel, *, inferred: bool) -> Dict[str, object]:
    entry: Dict[str, object] = {
        "name": rel.name,
        "source_table": rel.source_table,
        "source_columns": [rel.source_column],
        "target_table": rel.target_table,
        "target_columns": [rel.target_column],
        "relationship_type": rel.relationship_type,
        "active": rel.active,
    }
    if not inferred:
        entry["review_status"] = "existing"
        return entry

    confidence = _relationship_confidence(rel)
    entry["review_status"] = "pending_review"
    entry["confidence_band"] = _confidence_band(confidence)
    entry["review_recommendation"] = _relationship_recommendation(confidence)
    if rel.inferred_by_ml:
        entry["inferred_by_ml"] = True
        entry["ml_confidence"] = rel.ml_confidence
    if rel.inferred_by_llm:
        entry["inferred_by_llm"] = True
        entry["llm_confidence"] = rel.llm_confidence
    entry["notes"] = "REVIEW: keep to accept, delete to reject, or edit the columns/table names to correct it."
    return entry

def _build_simple_table_entry(table_cfg: object) -> Dict[str, object]:
    return {
        "name": table_cfg.name,
        "rows": table_cfg.num_rows,
        "primary_key_columns": list(table_cfg.primary_key_columns or []),
        "columns": [
            {
                "name": col.column_name,
                "data_type": col.data_type,
                "is_pk": bool(col.is_pk),
                "is_fk": bool(col.is_fk),
                "nullable": bool(col.nullable),
            }
            for col in table_cfg.columns
        ],
    }

def _build_simple_relationship_entry(rel) -> Dict[str, object]:
    return {
        "name": rel.name,
        "source_table": rel.source_table,
        "source_columns": [rel.source_column],
        "target_table": rel.target_table,
        "target_columns": [rel.target_column],
        "relationship_type": rel.relationship_type,
        "active": rel.active,
    }

def _write_simple_relationship_yaml(path: Path, tables, existing_rels, inferred_rels) -> None:
    """Write a minimal YAML containing only table schemas and relationships."""
    import yaml as _yaml

    payload = {
        "config_format": "sdp-yaml-v1",
        "tables": [
            _build_simple_table_entry(table_cfg)
            for _, table_cfg in sorted(tables.items(), key=lambda item: item[0])
        ],
        "relationships": sorted(
            [_build_simple_relationship_entry(r) for r in [*existing_rels, *inferred_rels]],
            key=_relationship_sort_key,
        ),
    }
    path.write_text(_yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def _write_reviewable_yaml(path: Path, existing_rels, inferred_rels, inference_summary: Optional[Dict[str, object]] = None) -> None:
    """Emit a YAML containing both the original relationships (kept) and the
    inferred ones (annotated with confidence + signals) so the SME can edit
    in place — delete what's wrong, keep what's right.
    """
    import yaml as _yaml
    rel_dicts = []
    debug_signals = {}
    for r in existing_rels:
        rel_dicts.append(_build_review_relationship_entry(r, inferred=False))
    for r in inferred_rels:
        rel_dicts.append(_build_review_relationship_entry(r, inferred=True))
        if r.name and r.inference_signals:
            debug_signals[r.name] = dict(r.inference_signals)

    rel_dicts = sorted(rel_dicts, key=_relationship_sort_key)

    payload = {
        "config_format": "sdp-yaml-v1",
        "_review_metadata": {
            "instructions": "Review each entry under 'relationships'. Keep accurate ones, delete inaccurate ones, "
                            "fix any column-name mistakes, then run `python main.py record-feedback "
                            "--inferred <this-file> --reviewed <your-edited-file>` to teach the system.",
            "existing_count": len(existing_rels),
            "inferred_count": len(inferred_rels),
            "summary": inference_summary or _build_inference_summary(inferred_rels),
        },
        "relationships": rel_dicts,
    }
    if debug_signals:
        payload["_review_debug"] = {
            "inference_signals_by_relationship": debug_signals,
        }
    path.write_text(_yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def _load_relationship_sample_data(sample_data_arg: Optional[str], tables: Dict[str, object]) -> Optional[Dict[str, pd.DataFrame]]:
    """Load optional sample data for the value-subset signal.

    The CLI advertises `--sample-data`; keep it opt-in so existing behaviour is
    unchanged when callers do not provide it.
    """
    if not sample_data_arg:
        return None

    root = Path(sample_data_arg)
    if not root.exists() or not root.is_dir():
        logger.warning(f"sample-data path is not a readable directory: {root}")
        return None

    loaded: Dict[str, pd.DataFrame] = {}
    wanted = {str(name).lower() for name in tables.keys()}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        if stem not in wanted:
            continue
        try:
            if path.suffix.lower() == ".csv":
                loaded[stem] = pd.read_csv(path, nrows=5000)
            elif path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
                loaded[stem] = frame.head(5000) if len(frame) > 5000 else frame
        except Exception as exc:
            logger.warning(f"sample-data: skipped {path.name}: {exc}")

    if loaded:
        logger.info(f"Loaded sample data for {len(loaded)} table(s) from {root}")
        return loaded

    logger.warning(f"No matching sample CSV/Parquet files found in {root}")
    return None

def _write_er_diagram(out_path: Path, tables, relationships) -> None:
    """Emit an ER diagram in the format implied by the file extension."""
    try:
        from sdp.utils.er_diagram import ERDiagramGenerator
    except Exception as exc:
        logger.warning(f"ER diagram generator unavailable: {exc}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower().lstrip(".")
    fmt = {"mmd": "mermaid", "dot": "dot", "png": "png"}.get(suffix, "mermaid")
    gen = ERDiagramGenerator(tables, relationships)
    if fmt == "png":
        if gen.generate_png(out_path):
            logger.info(f"✅ Wrote ER diagram: {out_path}")
        return
    text = gen.generate_dot() if fmt == "dot" else gen.generate_mermaid()
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"✅ Wrote ER diagram: {out_path}")

def run_record_feedback(args) -> int:
    """Diff inferred-vs-reviewed YAML and persist accept/reject deltas."""
    configure_logging(getattr(args, "verbose", False))
    try:
        import yaml as _yaml
        from sdp.ml.relationship_feedback_store import FeedbackStore, FeedbackEntry

        inferred_doc = _yaml.safe_load(Path(args.inferred).read_text(encoding="utf-8")) or {}
        reviewed_doc = _yaml.safe_load(Path(args.reviewed).read_text(encoding="utf-8")) or {}
        debug_signals = ((inferred_doc.get("_review_debug") or {}).get("inference_signals_by_relationship") or {})

        def _index(doc) -> dict:
            out = {}
            for r in (doc.get("relationships") or []):
                src_cols = r.get("source_columns") or [r.get("source_column")]
                tgt_cols = r.get("target_columns") or [r.get("target_column")]
                for s_col, t_col in zip(src_cols or [], tgt_cols or []):
                    if not s_col or not t_col:
                        continue
                    key = (str(r.get("source_table", "")).lower(), s_col,
                           str(r.get("target_table", "")).lower(), t_col)
                    out[key] = r
            return out

        inferred = _index(inferred_doc)
        reviewed = _index(reviewed_doc)

        # Only score entries the inferrer originally proposed (others are user-authored).
        inferred_only = {k: v for k, v in inferred.items()
                         if v.get("inferred_by_ml") or v.get("inferred_by_llm")}

        store = FeedbackStore(args.feedback_store) if args.feedback_store else FeedbackStore()
        accept_count = reject_count = 0
        for key, original in inferred_only.items():
            kept = key in reviewed
            original_name = original.get("name")
            entry = FeedbackEntry(
                source_table=key[0],
                source_column=key[1],
                target_table=key[2],
                target_column=key[3],
                accepted=kept,
                signals=dict(original.get("inference_signals") or debug_signals.get(original_name) or {}),
                predicted_confidence=original.get("ml_confidence") or original.get("llm_confidence"),
                note="recorded via record-feedback",
            )
            store.append(entry)
            if kept:
                accept_count += 1
            else:
                reject_count += 1

        # Detect SME-added relationships (in reviewed but not in inferred) — these
        # are positive examples the inferrer missed. Recorded with empty signals
        # so they only contribute to pattern-memory, not classifier training.
        added = 0
        for key, r in reviewed.items():
            if key in inferred:
                continue
            entry = FeedbackEntry(
                source_table=key[0], source_column=key[1],
                target_table=key[2], target_column=key[3],
                accepted=True, signals={},
                note="SME-added (inferrer missed)",
            )
            store.append(entry)
            added += 1

        print("=== Feedback recorded ===")
        print(f"  Accepted (kept):  {accept_count}")
        print(f"  Rejected (gone):  {reject_count}")
        print(f"  SME-added:        {added}")
        print(f"  Store path:       {store.path}")
        stats = store.stats()
        print(f"  Total in store:   {stats['total']} ({stats['accepted']} accepted, {stats['rejected']} rejected)")
        return 0
    except Exception as exc:
        logger.error(f"record-feedback failed: {exc}")
        logger.debug('Full traceback:', exc_info=True)
        return 1
