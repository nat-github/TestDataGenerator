"""`mock-*` commands — init, render, enrich, lint.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from sdp.generators.data_generator import DataGenerator
from sdp.services.common import (
    configure_logging,
    create_output_directory,
    load_config_context,
    validate_config_file,
    verify_export,
)
from sdp.utils.config_parser import ConfigParser
from sdp.utils.data_validator import DataValidator
from sdp.utils.parquet_post_processor import ParquetPostProcessor

logger = logging.getLogger(__name__)


def run_mock_init(args) -> int:
    """Convert an OpenAPI / Postman / HAR artefact into a sdp-mock-v1 YAML config."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import dump_mock_config

        source_type = _detect_mock_source_type(args.source, getattr(args, "source_type", "auto"))

        if source_type == "openapi":
            from sdp.mocks.openapi_importer import import_openapi
            cfg = import_openapi(args.source)
        elif source_type == "postman":
            from sdp.mocks.postman_importer import import_postman
            cfg = import_postman(args.source)
        elif source_type == "har":
            from sdp.mocks.har_importer import import_har
            cfg = import_har(args.source)
        else:
            logger.error(f"Could not detect source format for {args.source} — pass --source-type explicitly")
            return 1

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_mock_config(cfg, out_path)

        print("=== mock-init ===")
        print(f"  Source:      {args.source}")
        print(f"  Source type: {source_type}")
        print(f"  Output:      {out_path}")
        print(f"  Endpoints:   {len(cfg.endpoints)}")
        print(f"  Schemas:     {len(cfg.schemas)}")
        for ep in cfg.endpoints[:10]:
            statuses = ",".join(str(r.status) for r in ep.responses)
            print(f"    {ep.method:6s} {ep.path}  -> [{statuses}]  ({ep.name})")
        if len(cfg.endpoints) > 10:
            print(f"    ... and {len(cfg.endpoints) - 10} more")
        return 0
    except Exception as exc:
        logger.error(f"mock-init failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

def _detect_mock_source_type(source_path: str, override: str) -> str:
    """Heuristic detection between openapi / postman / har sources."""
    if override and override != "auto":
        return override
    p = Path(source_path)
    if not p.exists():
        return "auto"
    suffix = p.suffix.lower()
    raw = ""
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError as exc:
        # Unreadable file — fall through to extension-based detection, which
        # is all the caller can do anyway.
        logger.debug("Could not sniff %s for type detection: %s", p, exc)
    # HAR files always contain a top-level "log": with "version" + "entries"
    if '"log"' in raw and '"entries"' in raw:
        return "har"
    # Postman collections start with an `info` block referencing the schema
    if "schema.getpostman.com" in raw or '"_postman_id"' in raw:
        return "postman"
    # OpenAPI specs declare `openapi:` (3.x) or `swagger:` (2.x)
    if "openapi:" in raw or "swagger:" in raw or '"openapi"' in raw or '"swagger"' in raw:
        return "openapi"
    # Fallback by extension
    if suffix in (".yaml", ".yml"):
        return "openapi"
    return "auto"

def run_mock_render(args) -> int:
    """Render mocks (WireMock stubs / JSON fixtures / …) from a sdp-mock-v1 config."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import load_mock_config

        cfg = load_mock_config(args.config)
        formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_files = 0
        for fmt in formats:
            target = out_dir / fmt
            if fmt == "wiremock":
                from sdp.mocks.renderers.wiremock import render_wiremock
                files = render_wiremock(
                    cfg, target,
                    seed=args.seed,
                    examples_per_endpoint=args.examples,
                    match_mode=args.match_mode,
                )
                print(f"  wiremock:        {len(files)} mapping(s) -> {target}")
                total_files += len(files)
            elif fmt in ("json", "json-fixture", "fixture"):
                from sdp.mocks.renderers.json_fixture import render_json_fixtures
                files = render_json_fixtures(cfg, target, seed=args.seed)
                print(f"  json:            {len(files)} fixture(s) -> {target}")
                total_files += len(files)
            elif fmt == "pact":
                from sdp.mocks.renderers.pact import render_pact
                files = render_pact(
                    cfg, target,
                    consumer=getattr(args, "pact_consumer", "consumer"),
                    provider=getattr(args, "pact_provider", "provider"),
                    seed=args.seed,
                    examples_per_endpoint=args.examples,
                )
                print(f"  pact:            {len(files)} contract(s) -> {target}")
                total_files += len(files)
            elif fmt == "postman":
                from sdp.mocks.renderers.postman import render_postman
                files = render_postman(
                    cfg, target,
                    seed=args.seed,
                    examples_per_endpoint=args.examples,
                )
                print(f"  postman:         {len(files)} collection(s) -> {target}")
                total_files += len(files)
            elif fmt in ("openapi-examples", "openapi"):
                source = getattr(args, "openapi_source", None)
                if not source:
                    logger.error("openapi-examples format requires --openapi-source <spec>")
                    continue
                from sdp.mocks.renderers.openapi_examples import render_openapi_examples
                files = render_openapi_examples(
                    cfg, target,
                    source_spec=source,
                    seed=args.seed,
                    overwrite_existing=getattr(args, "openapi_overwrite", False),
                )
                print(f"  openapi-examples: {len(files)} file(s) -> {target}")
                total_files += len(files)
            else:
                logger.warning(f"Unknown format: {fmt} — skipping")

        print("\n=== mock-render ===")
        print(f"  Config:    {args.config}")
        print(f"  Output:    {out_dir}")
        print(f"  Formats:   {formats}")
        print(f"  Examples:  {args.examples or 'config-default'}")
        print(f"  Seed:      {args.seed if args.seed is not None else 'random'}")
        print(f"  Total:     {total_files} file(s)")
        return 0 if total_files > 0 else 1
    except Exception as exc:
        logger.error(f"mock-render failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

def run_mock_enrich(args) -> int:
    """Use an LLM to fill missing examples and draft missing 4xx/5xx responses."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import load_mock_config, dump_mock_config
        from sdp.mocks.llm_enricher import enrich

        cfg = load_mock_config(args.config)
        enriched, result = enrich(
            cfg,
            fill_missing_examples=not getattr(args, "no_fill_examples", False),
            draft_error_responses=not getattr(args, "no_draft_errors", False),
            provider=getattr(args, "llm_provider", None),
            model=getattr(args, "llm_model", None),
            base_url=getattr(args, "llm_base_url", None),
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_mock_config(enriched, out_path)

        print("=== mock-enrich ===")
        print(f"  Source:           {args.config}")
        print(f"  Output:           {out_path}")
        print(f"  Examples added:   {result.examples_added}")
        print(f"  Errors drafted:   {result.error_responses_added}")
        if result.schemas_touched:
            print(f"  Schemas touched:  {', '.join(sorted(set(result.schemas_touched)))}")
        if result.endpoints_touched:
            print(f"  Endpoints touched:{', '.join(sorted(set(result.endpoints_touched)))}")
        for warn in result.warnings:
            print(f"  WARN: {warn}")
        return 0
    except Exception as exc:
        logger.error(f"mock-enrich failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

def run_mock_lint(args) -> int:
    """Validate a sdp-mock-v1 config and pretty-print the report."""
    configure_logging(getattr(args, "verbose", False))
    try:
        from sdp.mocks.config_parser import lint_mock_config

        report = lint_mock_config(args.config)
        print(f"=== mock-lint: {report['path']} ===")
        if report["ok"]:
            print(f"  OK  ({report.get('endpoints', 0)} endpoints, {report.get('schemas', 0)} schemas)")
        else:
            print("  FAIL")
        for err in report.get("errors", []):
            print(f"  ERROR:   {err}")
        for warn in report.get("warnings", []):
            print(f"  WARN:    {warn}")
        return 0 if report["ok"] else 1
    except Exception as exc:
        logger.error(f"mock-lint failed: {exc}")
        return 1
