#!/usr/bin/env python3
"""The ``sdp`` command line — argument dispatch only.

Everything this module used to do now lives beside the concern it serves:

============================== ==========================================
Module                         Owns
============================== ==========================================
``sdp/cli_parser.py``          the argparse tree
``sdp/services/generation.py`` the generation run itself
``sdp/services/common.py``     shared config/output helpers
``sdp/cli_commands/*.py``      one module per command group
============================== ==========================================

What remains here is argv normalisation, the ``generate`` adapter that
turns flags into a :class:`GenerationRequest`, and ``main``'s dispatch
table. Names are re-exported below because ``main.py`` does
``from sdp.cli import *`` and callers import helpers from here.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional, Sequence

from sdp.cli_commands.cdc import run_delta, run_scd2  # noqa: F401
from sdp.cli_commands.config_tools import (  # noqa: F401
    run_collibra_import,
    run_enrich,
    run_infer_config,
    run_lint,
    run_pii_scan,
)
from sdp.cli_commands.mocks import (  # noqa: F401
    _detect_mock_source_type,
    run_mock_enrich,
    run_mock_init,
    run_mock_lint,
    run_mock_render,
)
from sdp.cli_commands.quality import (  # noqa: F401
    _parse_targets,
    _serialise_report,
    run_contract_diff_cmd,
    run_contract_test_cmd,
    run_quality_report,
    run_validate_data,
)
from sdp.cli_commands.relationships import (  # noqa: F401
    _write_reviewable_yaml,
    run_infer_relationships,
    run_record_feedback,
)
from sdp.cli_parser import build_parser  # noqa: F401
from sdp.services.common import (  # noqa: F401
    configure_logging,
    create_output_directory,
    load_config_context,
    validate_config_file,
    verify_export,
)
from sdp.services.generation import (  # noqa: F401
    GenerationOutcome,
    GenerationRequest,
    generate_dataset,
    get_record_counts,
)
from sdp.utils.config_parser import ConfigParser


logger = logging.getLogger(__name__)
KNOWN_COMMANDS = {"generate", "delta", "scd2", "lint", "enrich", "collibra-import",
                  "infer-config", "pii-scan", "infer-relationships", "record-feedback",
                  "mock-init", "mock-render", "mock-lint", "mock-enrich",
                  "validate-data", "quality-report", "contract-test", "contract-diff"}


def _normalize_argv(argv: Sequence[str]) -> List[str]:
    if not argv:
        return ["generate"]
    if argv[0] in KNOWN_COMMANDS:
        return list(argv)
    return ["generate", *argv]




def parse_arguments(argv: Sequence[str] | None = None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    return parser.parse_args(_normalize_argv(argv))
















# ---------------------------------------------------------------------------
# SCD2 / Delta / versions_per_key — ported from the patched old generator.
# All of these are opt-in and have no effect on existing configs/commands.
# ---------------------------------------------------------------------------
























def _request_from_args(args) -> GenerationRequest:
    """Map an argparse Namespace onto the service's request type.

    The only place CLI-shaped input is translated. ``--engine-option`` is
    parsed here, and ``--epsilon`` is folded in as an engine option, so the
    service never sees a raw ``KEY=VALUE`` string.
    """
    engine_options = _parse_engine_options(getattr(args, "engine_option", None))
    if getattr(args, "epsilon", None) is not None:
        # --epsilon is a first-class flag because its value is a promise to a
        # regulator, not a tuning knob.
        engine_options = {**(engine_options or {}), "epsilon": args.epsilon}

    return GenerationRequest(
        config=args.config,
        output=args.output,
        default_records=getattr(args, "default_records", None),
        records=getattr(args, "records", None),
        seed=getattr(args, "seed", None),
        validate=getattr(args, "validate", False),
        stream=getattr(args, "stream", False),
        chunk_size=getattr(args, "chunk_size", 100_000),
        infer_relationships=getattr(args, "infer_relationships", False),
        method=getattr(args, "method", "ml"),
        llm_confidence=getattr(args, "llm_confidence", 0.7),
        ml_confidence=getattr(args, "ml_confidence", 0.55),
        feedback_store=getattr(args, "feedback_store", None),
        er_diagram=getattr(args, "er_diagram", False),
        er_format=getattr(args, "er_format", ["mermaid"]),
        er_output=getattr(args, "er_output", None),
        upload_to=getattr(args, "upload_to", None),
        validate_with_gx=getattr(args, "validate_with_gx", False),
        gx_tolerance=getattr(args, "gx_tolerance", 0.5),
        gx_fail_on_error=getattr(args, "gx_fail_on_error", False),
        write_delta=getattr(args, "write_delta", False),
        delta_partition_col=getattr(args, "delta_partition_col", None),
        delta_partition_value=getattr(args, "delta_partition_value", None),
        delta_tables=getattr(args, "delta_tables", None),
        engine=getattr(args, "engine", None),
        engine_options=engine_options,
        privacy_report_json=getattr(args, "privacy_report_json", None),
        verbose=getattr(args, "verbose", False),
    )


def run_generate(args) -> int:
    """CLI adapter over :func:`sdp.services.generation.generate_dataset`.

    All this does is translate flags and return an exit code — the run
    itself belongs to the service, so the SDK gets identical behaviour
    without going through argparse.
    """
    logger.info("SDV Test Data Generator")
    logger.info("=" * 50)

    if getattr(args, "list_engines", False):
        _print_engines()
        return 0

    return generate_dataset(_request_from_args(args)).exit_code


















































# ===========================================================================
# Stubs / Mocks track — `mock-init`, `mock-render`, `mock-lint`
# ===========================================================================














def _parse_engine_options(raw: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """``["epochs=300", "batch_size=500"]`` → ``{"epochs": 300, ...}``.

    Digit-only values become ints — engine options are overwhelmingly
    numeric (epochs, batch_size), and passing "300" where an int is
    expected fails deep inside the engine with a poor message.
    """
    if not raw:
        return None
    options: Dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--engine-option expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        value = value.strip()
        options[key.strip()] = int(value) if value.isdigit() else value
    return options


def _print_engines() -> None:
    """Print the engine registry — name, availability, description."""
    from sdp.synthesizers import DEFAULT_ENGINE, describe

    print("Available generation engines:\n")
    for name, description, available in describe():
        mark = "  " if available else "! "
        default = "  (default)" if name == DEFAULT_ENGINE else ""
        print(f"{mark}{name:<16}{description}{default}")
    print("\n  ! = registered but dependencies unavailable")
    print("  Select with --engine NAME, or `synthesizer_engine` in Run_Settings.")












def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    configure_logging(getattr(args, "verbose", False))

    try:
        if args.command == "generate":
            return run_generate(args)
        if args.command == "delta":
            return run_delta(args)
        if args.command == "scd2":
            return run_scd2(args)
        if args.command == "lint":
            return run_lint(args)
        if args.command == "enrich":
            return run_enrich(args)
        if args.command == "collibra-import":
            return run_collibra_import(args)
        if args.command == "infer-config":
            return run_infer_config(args)
        if args.command == "pii-scan":
            return run_pii_scan(args)
        if args.command == "infer-relationships":
            return run_infer_relationships(args)
        if args.command == "record-feedback":
            return run_record_feedback(args)
        if args.command == "mock-init":
            return run_mock_init(args)
        if args.command == "mock-render":
            return run_mock_render(args)
        if args.command == "mock-lint":
            return run_mock_lint(args)
        if args.command == "mock-enrich":
            return run_mock_enrich(args)
        if args.command == "validate-data":
            return run_validate_data(args)
        if args.command == "quality-report":
            return run_quality_report(args)
        if args.command == "contract-test":
            return run_contract_test_cmd(args)
        if args.command == "contract-diff":
            return run_contract_diff_cmd(args)
        logger.error(f"Unknown command: {args.command}")
        return 1
    except FileNotFoundError as exc:
        logger.error(f"File error: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
