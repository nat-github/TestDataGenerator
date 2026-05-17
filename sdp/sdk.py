"""Programmatic SDK facade for the Synthetic Data Platform.

This module is the supported in-process entry point. It wraps the same code
paths the ``sdp`` CLI uses, so behaviour is identical — the SDK simply
translates keyword arguments into the argument vector the CLI parser expects
and returns structured result objects instead of process exit codes.

Example
-------
    from sdp import SyntheticDataPlatform

    sdp = SyntheticDataPlatform()
    result = sdp.generate(config="config/Acct_bkng.xlsx", output="output/run_01", seed=42)
    if result.success:
        for name, frame in result.frames.items():
            print(name, len(frame))

The REST API (:mod:`sdp.api`) is a thin HTTP layer over this same facade.
"""
from __future__ import annotations

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class SDPError(RuntimeError):
    """Raised when an SDK operation fails before/around CLI dispatch."""


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CommandResult:
    """Outcome of a generic CLI-backed command."""

    command: str
    exit_code: int
    argv: List[str]

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@dataclasses.dataclass
class GenerationResult:
    """Outcome of :meth:`SyntheticDataPlatform.generate`.

    Parquet is always written to ``output_dir``; :attr:`frames` lazily reads it
    back into pandas DataFrames for in-process use.
    """

    output_dir: Path
    exit_code: int
    argv: List[str]
    _frames: Optional[Dict[str, "pd.DataFrame"]] = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def tables(self) -> List[str]:
        """Names of the generated tables (parquet file stems)."""
        return sorted(p.stem for p in self.output_dir.glob("*.parquet"))

    @property
    def frames(self) -> Dict[str, "pd.DataFrame"]:
        """The generated tables as in-memory DataFrames (read once, cached)."""
        if self._frames is None:
            import pandas as pd

            self._frames = {
                p.stem: pd.read_parquet(p)
                for p in sorted(self.output_dir.glob("*.parquet"))
            }
        return self._frames

    @property
    def total_records(self) -> int:
        return sum(len(f) for f in self.frames.values())


@dataclasses.dataclass
class LintResult:
    """Outcome of :meth:`SyntheticDataPlatform.lint`."""

    issues: List[Dict[str, Any]]
    report: str

    @property
    def errors(self) -> List[Dict[str, Any]]:
        return [i for i in self.issues if i.get("level") == "error"]

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        return [i for i in self.issues if i.get("level") == "warning"]

    @property
    def ok(self) -> bool:
        """True when the config has no error-level issues."""
        return not self.errors


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class SyntheticDataPlatform:
    """In-process facade over the Synthetic Data Platform.

    Parameters
    ----------
    log_level:
        Root log level applied on construction (e.g. ``"WARNING"``, ``"INFO"``).
        The underlying CLI handlers log progress; raise this to quieten them.
    """

    def __init__(self, *, log_level: str = "WARNING") -> None:
        try:
            logging.getLogger().setLevel(log_level)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            logger.warning("Ignoring invalid log_level %r", log_level)

    # -- generic dispatch ---------------------------------------------------

    def run(self, *argv: str) -> CommandResult:
        """Run any CLI command in-process. Escape hatch for unwrapped commands.

        ``sdp.run("generate", "--config", "x.xlsx", "--output", "out")``
        """
        from sdp.cli import main as cli_main

        args = [str(a) for a in argv]
        exit_code = cli_main(args)
        command = args[0] if args else ""
        return CommandResult(command=command, exit_code=int(exit_code or 0), argv=args)

    # -- generate -----------------------------------------------------------

    def generate(
        self,
        config: PathLike,
        output: Optional[PathLike] = None,
        *,
        default_records: Optional[int] = None,
        records: Optional[Union[Dict[str, int], Sequence[str]]] = None,
        seed: Optional[int] = None,
        validate: bool = False,
        infer_relationships: bool = False,
        method: str = "ml",
        ml_confidence: Optional[float] = None,
        llm_confidence: Optional[float] = None,
        stream: bool = False,
        chunk_size: Optional[int] = None,
        er_diagram: bool = False,
        er_format: Optional[Sequence[str]] = None,
        upload_to: Optional[str] = None,
        feedback_store: Optional[PathLike] = None,
        validate_with_gx: bool = False,
        gx_tolerance: Optional[float] = None,
        verbose: bool = False,
        extra_args: Optional[Sequence[str]] = None,
    ) -> GenerationResult:
        """Generate synthetic Parquet data from a config.

        When ``output`` is omitted a temporary directory is created and kept
        (so :attr:`GenerationResult.frames` can read it back); the caller owns
        cleanup of that directory.
        """
        out_dir = Path(output) if output is not None else Path(
            tempfile.mkdtemp(prefix="sdp_generate_")
        )
        argv: List[str] = ["generate", "--config", str(config), "--output", str(out_dir)]

        if default_records is not None:
            argv += ["--default-records", str(default_records)]
        if records:
            argv.append("--records")
            if isinstance(records, dict):
                argv += [f"{k}:{v}" for k, v in records.items()]
            else:
                argv += [str(r) for r in records]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if validate:
            argv.append("--validate")
        if infer_relationships:
            argv += ["--infer-relationships", "--method", method]
        if ml_confidence is not None:
            argv += ["--ml-confidence", str(ml_confidence)]
        if llm_confidence is not None:
            argv += ["--llm-confidence", str(llm_confidence)]
        if stream:
            argv.append("--stream")
        if chunk_size is not None:
            argv += ["--chunk-size", str(chunk_size)]
        if er_diagram:
            argv.append("--er-diagram")
        if er_format:
            argv += ["--er-format", *[str(f) for f in er_format]]
        if upload_to:
            argv += ["--upload-to", str(upload_to)]
        if feedback_store is not None:
            argv += ["--feedback-store", str(feedback_store)]
        if validate_with_gx:
            argv.append("--validate-with-gx")
        if gx_tolerance is not None:
            argv += ["--gx-tolerance", str(gx_tolerance)]
        if verbose:
            argv.append("--verbose")
        if extra_args:
            argv += [str(a) for a in extra_args]

        result = self.run(*argv)
        return GenerationResult(
            output_dir=out_dir, exit_code=result.exit_code, argv=result.argv
        )

    # -- lint ---------------------------------------------------------------

    def lint(self, config: PathLike) -> LintResult:
        """Validate a config and return structured issues (no data generated)."""
        from sdp.utils.config_parser import ConfigParser

        parser = ConfigParser(str(config))
        if not parser.load_config():
            raise SDPError(f"Failed to load configuration: {config}")
        parser.parse_tables()
        parser.parse_relationships()

        issues = parser.lint_config()
        report = parser.format_lint_report(issues)
        return LintResult(
            issues=[dataclasses.asdict(i) for i in issues],
            report=report,
        )

    # -- delta / scd2 -------------------------------------------------------

    def delta(
        self,
        config: PathLike,
        previous: PathLike,
        current: PathLike,
        output: PathLike,
        *,
        tables: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> CommandResult:
        """Compute a CDC delta between two snapshot directories."""
        argv = ["delta", "--config", str(config), "--previous", str(previous),
                "--current", str(current), "--output", str(output)]
        if tables:
            argv += ["--tables", *[str(t) for t in tables]]
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)

    def scd2(
        self,
        config: PathLike,
        previous: PathLike,
        current: PathLike,
        output: PathLike,
        *,
        tables: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> CommandResult:
        """Build SCD2 history from two snapshot directories."""
        argv = ["scd2", "--config", str(config), "--previous", str(previous),
                "--current", str(current), "--output", str(output)]
        if tables:
            argv += ["--tables", *[str(t) for t in tables]]
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)

    # -- relationship inference --------------------------------------------

    def infer_relationships(
        self,
        config: PathLike,
        config_output: PathLike,
        *,
        method: str = "ml",
        ml_mode: str = "standard",
        ml_confidence: Optional[float] = None,
        llm_confidence: Optional[float] = None,
        er_output: Optional[PathLike] = None,
        sample_data: Optional[PathLike] = None,
        feedback_store: Optional[PathLike] = None,
        verbose: bool = False,
    ) -> CommandResult:
        """Infer missing FK relationships and write a reviewable config."""
        argv = ["infer-relationships", "--config", str(config),
                "--config-output", str(config_output),
                "--method", method, "--ml-mode", ml_mode]
        if ml_confidence is not None:
            argv += ["--ml-confidence", str(ml_confidence)]
        if llm_confidence is not None:
            argv += ["--llm-confidence", str(llm_confidence)]
        if er_output is not None:
            argv += ["--er-output", str(er_output)]
        if sample_data is not None:
            argv += ["--sample-data", str(sample_data)]
        if feedback_store is not None:
            argv += ["--feedback-store", str(feedback_store)]
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)

    # -- validation / quality ----------------------------------------------

    def validate_data(
        self,
        config: PathLike,
        input_dir: PathLike,
        *,
        tolerance: Optional[float] = None,
        report_json: Optional[PathLike] = None,
        fail_on_error: bool = False,
        verbose: bool = False,
    ) -> CommandResult:
        """Validate generated Parquet against a Great Expectations suite."""
        argv = ["validate-data", "--config", str(config), "--input", str(input_dir)]
        if tolerance is not None:
            argv += ["--tolerance", str(tolerance)]
        if report_json is not None:
            argv += ["--report-json", str(report_json)]
        if fail_on_error:
            argv.append("--fail-on-error")
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)

    def quality_report(
        self,
        generated: PathLike,
        *,
        source: Optional[PathLike] = None,
        output_html: Optional[PathLike] = None,
        output_json: Optional[PathLike] = None,
        privacy_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> CommandResult:
        """Produce a statistical fidelity / privacy report."""
        argv = ["quality-report", "--generated", str(generated)]
        if source is not None:
            argv += ["--source", str(source)]
        if output_html is not None:
            argv += ["--output-html", str(output_html)]
        if output_json is not None:
            argv += ["--output-json", str(output_json)]
        if privacy_threshold is not None:
            argv += ["--privacy-threshold", str(privacy_threshold)]
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)

    # -- mocks --------------------------------------------------------------

    def mock_init(
        self,
        source: PathLike,
        output: PathLike,
        *,
        source_type: Optional[str] = None,
        verbose: bool = False,
    ) -> CommandResult:
        """Convert an OpenAPI / Postman / HAR artefact into an sdp-mock-v1 config."""
        argv = ["mock-init", "--from", str(source), "--output", str(output)]
        if source_type:
            argv += ["--source-type", source_type]
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)

    def mock_render(
        self,
        config: PathLike,
        output: PathLike,
        *,
        formats: Union[str, Sequence[str]] = "wiremock",
        examples: Optional[int] = None,
        match_mode: Optional[str] = None,
        seed: Optional[int] = None,
        openapi_source: Optional[PathLike] = None,
        verbose: bool = False,
    ) -> CommandResult:
        """Render a mock config to wiremock / json / pact / postman / openapi-examples."""
        fmt = formats if isinstance(formats, str) else ",".join(formats)
        argv = ["mock-render", "--config", str(config), "--output", str(output),
                "--format", fmt]
        if examples is not None:
            argv += ["--examples", str(examples)]
        if match_mode:
            argv += ["--match-mode", match_mode]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if openapi_source is not None:
            argv += ["--openapi-source", str(openapi_source)]
        if verbose:
            argv.append("--verbose")
        return self.run(*argv)
