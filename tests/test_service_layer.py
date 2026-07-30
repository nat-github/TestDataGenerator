"""Unit tests for the extracted service layer and split modules.

Before the refactor these paths were only reachable through a full CLI
invocation, so a signature change inside them showed up as a mysterious
end-to-end failure — or, in the MCP case, not at all. These tests address
each extracted unit directly.
"""
from __future__ import annotations

import ast
import inspect
import logging
import pathlib

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def simple_config(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        """
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 10
tables:
  - name: users
    active: true
    num_rows: 10
    columns:
      - name: user_id
        data_type: N38
        is_primary_key: true
      - name: tier
        data_type: VA3
        business_values: "gold;silver"
""".strip(),
        encoding="utf-8",
    )
    return str(config)


# ---------------------------------------------------------------------------
# services/common.py
# ---------------------------------------------------------------------------


def test_validate_config_file_accepts_a_real_config(simple_config):
    from sdp.services.common import validate_config_file

    assert validate_config_file(simple_config) is True


def test_validate_config_file_rejects_a_missing_file(tmp_path):
    from sdp.services.common import validate_config_file

    assert validate_config_file(str(tmp_path / "nope.yaml")) is False


def test_create_output_directory_makes_nested_paths(tmp_path):
    from sdp.services.common import create_output_directory

    target = tmp_path / "a" / "b" / "c"
    assert create_output_directory(str(target)) is True
    assert target.is_dir()


def test_verify_export_counts_rows(tmp_path):
    from sdp.services.common import verify_export

    pd.DataFrame({"x": range(7)}).to_parquet(tmp_path / "t1.parquet", index=False)
    pd.DataFrame({"x": range(3)}).to_parquet(tmp_path / "t2.parquet", index=False)

    assert verify_export(str(tmp_path)) == 10


def test_verify_export_raises_when_nothing_was_written(tmp_path):
    """An export that produced no parquet is a failed run, not a zero-row
    one — verify_export says so rather than returning 0."""
    from sdp.services.common import verify_export

    with pytest.raises(FileNotFoundError, match="No parquet files"):
        verify_export(str(tmp_path))


def test_load_config_context_raises_on_bad_config(tmp_path):
    from sdp.services.common import load_config_context

    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a valid config\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config_context(str(bad))


def test_load_config_context_returns_a_parsed_config(simple_config):
    from sdp.services.common import load_config_context

    parser = load_config_context(simple_config)
    assert "users" in parser.tables


# ---------------------------------------------------------------------------
# services/generation.py — request / outcome
# ---------------------------------------------------------------------------


def test_request_defaults_are_safe():
    from sdp.services.generation import GenerationRequest

    request = GenerationRequest(config="c.yaml")
    assert request.output == "output"
    assert request.seed is None
    assert request.write_delta is False
    assert request.er_format == ["mermaid"]


def test_request_er_format_default_is_not_shared():
    """A mutable default shared between instances is a classic bug."""
    from sdp.services.generation import GenerationRequest

    a = GenerationRequest(config="a.yaml")
    b = GenerationRequest(config="b.yaml")
    a.er_format.append("dot")
    assert b.er_format == ["mermaid"]


def test_request_matches_argparse_destinations():
    """Helpers read the request with plain attribute access, so its fields
    must stay aligned with the generate subcommand's argparse dests. This is
    the invariant that let the orchestration move out of cli.py."""
    from sdp.cli_parser import build_parser
    from sdp.services.generation import GenerationRequest

    args = build_parser().parse_args(["generate", "--config", "c.yaml"])
    request_fields = set(GenerationRequest.__dataclass_fields__)

    # Every argparse dest the service reads must exist on the request.
    for dest in ("config", "output", "default_records", "records", "seed",
                 "validate", "stream", "chunk_size", "er_diagram", "upload_to",
                 "write_delta", "engine"):
        assert hasattr(args, dest), f"argparse lost {dest}"
        assert dest in request_fields, f"GenerationRequest lost {dest}"


def test_outcome_ok_and_totals():
    from sdp.services.generation import GenerationOutcome

    outcome = GenerationOutcome(
        exit_code=0, output_dir=pathlib.Path("."), row_counts={"a": 3, "b": 4},
    )
    assert outcome.ok is True
    assert outcome.total_records == 7


def test_outcome_failure_is_not_ok():
    from sdp.services.generation import GenerationOutcome

    assert GenerationOutcome(exit_code=1, output_dir=pathlib.Path(".")).ok is False


# ---------------------------------------------------------------------------
# services/generation.py — generate_dataset
# ---------------------------------------------------------------------------


def test_generate_dataset_end_to_end(simple_config, tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(
        config=simple_config, output=str(tmp_path / "out"),
        default_records=10, seed=42, engine="rule-based",
    ))

    assert outcome.ok, outcome.error
    assert outcome.row_counts == {"users": 10}
    assert outcome.total_records == 10
    assert outcome.report["total_records"] == 10
    assert (tmp_path / "out" / "users.parquet").exists()
    assert set(outcome.frames) == {"users"}


def test_generate_dataset_missing_config_returns_failure(tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(
        config=str(tmp_path / "absent.yaml"), output=str(tmp_path / "out"),
    ))
    assert outcome.ok is False
    assert outcome.exit_code == 1
    assert outcome.error


def test_generate_dataset_blank_config_is_rejected(tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(config="", output=str(tmp_path)))
    assert outcome.ok is False
    assert "--config" in (outcome.error or "")


def test_generate_dataset_is_seed_reproducible(simple_config, tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    def run(name):
        return generate_dataset(GenerationRequest(
            config=simple_config, output=str(tmp_path / name),
            default_records=10, seed=7, engine="rule-based",
        )).frames["users"]

    pd.testing.assert_frame_equal(run("a"), run("b"))


def test_generate_dataset_reports_engine_stats(simple_config, tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(
        config=simple_config, output=str(tmp_path / "out"),
        default_records=5, engine="rule-based",
    ))
    assert outcome.engine_stats is not None
    assert outcome.engine_stats["engine"] == "rule-based"


def test_generate_dataset_privacy_report_only_for_dp(simple_config, tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    plain = generate_dataset(GenerationRequest(
        config=simple_config, output=str(tmp_path / "plain"),
        default_records=5, engine="rule-based",
    ))
    assert plain.privacy_report is None

    dp = generate_dataset(GenerationRequest(
        config=simple_config, output=str(tmp_path / "dp"),
        default_records=5, engine="dp-marginal", seed=1,
    ))
    assert dp.privacy_report is not None
    assert dp.privacy_report["epsilon_requested"] == 1.0


def test_generate_dataset_validation_populates_outcome(simple_config, tmp_path):
    from sdp.services.generation import GenerationRequest, generate_dataset

    outcome = generate_dataset(GenerationRequest(
        config=simple_config, output=str(tmp_path / "out"),
        default_records=5, engine="rule-based", validate=True,
    ))
    assert outcome.validation is not None
    assert "is_valid" in outcome.validation


def test_service_never_imports_the_cli():
    """The whole point of the extraction: the CLI depends on the service,
    never the reverse."""
    for module in ("common.py", "generation.py"):
        source = (REPO_ROOT / "sdp" / "services" / module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("sdp.cli"), \
                    f"{module} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("sdp.cli"), \
                        f"{module} imports {alias.name}"


# ---------------------------------------------------------------------------
# cli.py stays a thin adapter
# ---------------------------------------------------------------------------


def test_cli_run_generate_delegates_to_the_service(monkeypatch, simple_config, tmp_path):
    import sdp.cli as cli
    from sdp.services.generation import GenerationOutcome

    captured = {}

    def fake(request):
        captured["request"] = request
        return GenerationOutcome(exit_code=0, output_dir=tmp_path)

    monkeypatch.setattr(cli, "generate_dataset", fake)
    args = cli.build_parser().parse_args([
        "generate", "--config", simple_config, "--output", str(tmp_path),
        "--seed", "5", "--engine", "rule-based",
    ])
    assert cli.run_generate(args) == 0
    assert captured["request"].seed == 5
    assert captured["request"].engine == "rule-based"


def test_cli_module_is_small():
    """A ratchet: cli.py went 2,299 -> ~360 lines. If it starts growing
    again, orchestration is leaking back in."""
    lines = (REPO_ROOT / "sdp" / "cli.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) < 600, f"cli.py is {len(lines)} lines — is logic leaking back?"


def test_data_generator_module_is_smaller():
    """Same ratchet for the generator: 2,396 -> ~1,160 after the mixin split."""
    path = REPO_ROOT / "sdp" / "generators" / "data_generator.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 1500, f"data_generator.py is {len(lines)} lines"


def test_engine_option_parsing_is_cli_only():
    """`--engine-option KEY=VALUE` is CLI-shaped input; the service takes a
    dict. The translation belongs in exactly one place."""
    from sdp.cli import _parse_engine_options, _request_from_args, build_parser

    assert _parse_engine_options(["epochs=7"]) == {"epochs": 7}

    args = build_parser().parse_args([
        "generate", "--config", "c.yaml", "--engine-option", "epochs=7",
        "--epsilon", "0.25",
    ])
    request = _request_from_args(args)
    assert request.engine_options == {"epochs": 7, "epsilon": 0.25}


# ---------------------------------------------------------------------------
# DataGenerator mixin composition
# ---------------------------------------------------------------------------


MIXINS = {
    "AnchorMixin": ["_resolve_source_path", "_load_anchor_tables",
                    "_is_anchor_table", "_inject_anchor_tables"],
    "PrimaryKeyMixin": ["_initialize_pk_tracking", "_generate_unique_primary_key",
                        "_validate_and_fix_pk_uniqueness"],
    "SDVMetadataMixin": ["create_sdv_metadata", "_sanitize_sample_data_for_sdv"],
    "RelationshipMixin": ["_resolve_foreign_keys", "_enforce_all_relationships"],
    "ModelCacheMixin": ["_config_fingerprint", "save_model_artifacts"],
    "ArrowExportMixin": ["export_to_parquet", "_to_arrow_dc"],
}


@pytest.mark.parametrize("mixin_name,methods", list(MIXINS.items()))
def test_mixin_provides_its_methods(mixin_name, methods):
    import sdp.generators._anchors as anchors
    import sdp.generators._arrow_export as arrow
    import sdp.generators._model_cache as cache
    import sdp.generators._primary_keys as pks
    import sdp.generators._relationships as rels
    import sdp.generators._sdv_metadata as meta

    lookup = {
        "AnchorMixin": anchors.AnchorMixin,
        "ArrowExportMixin": arrow.ArrowExportMixin,
        "ModelCacheMixin": cache.ModelCacheMixin,
        "PrimaryKeyMixin": pks.PrimaryKeyMixin,
        "RelationshipMixin": rels.RelationshipMixin,
        "SDVMetadataMixin": meta.SDVMetadataMixin,
    }
    cls = lookup[mixin_name]
    for method in methods:
        assert hasattr(cls, method), f"{mixin_name} lost {method}"


def test_data_generator_inherits_every_mixin():
    from sdp.generators.data_generator import DataGenerator

    names = {c.__name__ for c in DataGenerator.__mro__}
    assert MIXINS.keys() <= names


def test_no_method_is_defined_twice_across_mixins():
    """Two mixins defining the same name would make behaviour depend on MRO
    order — silently, and differently from the pre-split class."""
    import sdp.generators._anchors as anchors
    import sdp.generators._arrow_export as arrow
    import sdp.generators._model_cache as cache
    import sdp.generators._primary_keys as pks
    import sdp.generators._relationships as rels
    import sdp.generators._sdv_metadata as meta

    seen, clashes = {}, []
    for module in (anchors, arrow, cache, pks, rels, meta):
        cls = next(v for k, v in vars(module).items() if k.endswith("Mixin"))
        for name, value in vars(cls).items():
            if name.startswith("__") or not callable(getattr(value, "__func__", value)):
                continue
            if name in seen:
                clashes.append(f"{name}: {seen[name]} and {cls.__name__}")
            seen[name] = cls.__name__
    assert not clashes, clashes


def test_arrow_decimal_roundtrip():
    """Direct unit test of a mixin method that previously needed a full run."""
    from sdp.generators._arrow_export import ArrowExportMixin

    array = ArrowExportMixin._to_arrow_dc(
        pd.Series(["1.005", "2.5", "-3.14159"]), precision=10, scale=2,
    )
    values = [str(v) for v in array.to_pylist()]
    assert values == ["1.01", "2.50", "-3.14"]


def test_arrow_bigint_rejects_oversized_values():
    from sdp.generators._arrow_export import ArrowExportMixin

    array = ArrowExportMixin._to_arrow_bigint(pd.Series(["12", "not-a-number", None]))
    values = array.to_pylist()
    assert values[0] == 12
    assert values[1] is None and values[2] is None


# ---------------------------------------------------------------------------
# cli_commands package
# ---------------------------------------------------------------------------


COMMAND_MODULES = {
    "cdc": ["run_delta", "run_scd2"],
    "config_tools": ["run_lint", "run_enrich", "run_collibra_import",
                     "run_infer_config", "run_pii_scan"],
    "mocks": ["run_mock_init", "run_mock_render", "run_mock_enrich", "run_mock_lint"],
    "quality": ["run_validate_data", "run_quality_report",
                "run_contract_test_cmd", "run_contract_diff_cmd"],
    "relationships": ["run_infer_relationships", "run_record_feedback"],
}


@pytest.mark.parametrize("module_name,handlers", list(COMMAND_MODULES.items()))
def test_command_module_exposes_its_handlers(module_name, handlers):
    import importlib

    module = importlib.import_module(f"sdp.cli_commands.{module_name}")
    for handler in handlers:
        fn = getattr(module, handler, None)
        assert fn is not None, f"{module_name} lost {handler}"
        assert callable(fn)
        # Every handler takes argparse args and returns an exit code.
        assert list(inspect.signature(fn).parameters) == ["args"]


def test_every_subcommand_has_a_dispatch_entry():
    """A registered subcommand with no handler fails only at runtime."""
    import sdp.cli as cli

    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    )
    for command in subparsers.choices:
        assert command in cli.KNOWN_COMMANDS, f"{command} missing from KNOWN_COMMANDS"


def test_cli_reexports_names_that_main_shim_depends_on():
    """`main.py` does `from sdp.cli import *`, and callers import helpers
    from sdp.cli. The split must not break either."""
    import sdp.cli as cli

    for name in ("build_parser", "main", "parse_arguments", "run_generate",
                 "run_delta", "run_scd2", "run_lint", "run_quality_report",
                 "validate_config_file", "load_config_context", "configure_logging"):
        assert hasattr(cli, name), f"sdp.cli no longer exports {name}"


def test_main_shim_still_resolves():
    import main

    assert callable(main.main)
    assert callable(main.build_parser)
