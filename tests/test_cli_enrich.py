"""Tests for the `enrich` CLI subcommand wiring."""
from __future__ import annotations

from pathlib import Path

import yaml

from main import main, parse_arguments


def _minimal_enrichable_yaml(path: Path) -> Path:
    cfg = {
        "config_format": "sdp-yaml-v1",
        "tables": [
            {
                "name": "users",
                "rows": 2,
                "columns": [
                    {"name": "id", "data_type": "N10", "is_pk": True, "nullable": False},
                    {"name": "email", "data_type": "VA64", "nullable": True},
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_argparse_enrich_defaults():
    args = parse_arguments([
        "enrich",
        "--config", "in.yaml",
        "--output", "out.yaml",
    ])
    assert args.command == "enrich"
    assert args.confidence == 0.7


def test_enrich_cli_passes_confidence_threshold(monkeypatch, tmp_path: Path):
    config_path = _minimal_enrichable_yaml(tmp_path / "in.yaml")
    output_path = tmp_path / "enriched.yaml"

    captured = {}

    class DummyEnricher:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def enrich(self, tables, output_yaml_path=None):
            captured["tables_count"] = len(tables)
            if output_yaml_path:
                Path(output_yaml_path).write_text("config_format: sdp-yaml-v1\n", encoding="utf-8")
            return "config_format: sdp-yaml-v1\n"

        def get_suggestions(self, tables):
            return []

    monkeypatch.setattr("sdp.llm.schema_enricher.SchemaEnricher", DummyEnricher)

    rc = main([
        "enrich",
        "--config", str(config_path),
        "--output", str(output_path),
        "--confidence", "0.65",
    ])

    assert rc == 0
    assert captured["init_kwargs"]["confidence_threshold"] == 0.65
    assert captured["tables_count"] == 1

