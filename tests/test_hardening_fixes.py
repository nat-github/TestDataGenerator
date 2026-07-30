"""Regression guards for defects found in the platform assessment.

Each test here maps to a specific defect that shipped:

- `collibra_importer` used `Path` without importing it (NameError on every
  write)
- delta writes hardcoded a destructive overwrite with no way to opt out
- the SDK could not reach the engine flags at all
- packaging metadata used a deprecated layout and shipped pytest as a
  runtime dependency
"""
from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# collibra_importer NameError
# ---------------------------------------------------------------------------


def test_collibra_importer_has_path_imported():
    """`Path` was used at module level without being imported, so writing a
    config raised NameError. Importing the name is the whole fix."""
    from sdp.utils import collibra_importer

    assert hasattr(collibra_importer, "Path")


def test_collibra_importer_writes_a_config(tmp_path, monkeypatch):
    """Exercise the exact line that raised NameError — the config write.

    Everything above it is network I/O, so the session and the two fetch
    helpers are stubbed; the write path itself runs for real.
    """
    from sdp.utils.collibra_importer import CollibraImporter

    importer = CollibraImporter.__new__(CollibraImporter)   # no network/auth

    class _FakeSession:
        def find_assets(self, *a, **k):
            return [{"id": "asset-1", "displayName": "Anything"}]

    importer._session = _FakeSession()
    monkeypatch.setattr(CollibraImporter, "fetch_dataset",
                        lambda self, asset_id: {"id": asset_id}, raising=False)
    monkeypatch.setattr(
        CollibraImporter, "to_config",
        lambda self, fetched, table_name=None: {
            "tables": [{"name": "t", "columns": [{"name": "c"}]}]
        },
        raising=False,
    )

    out = tmp_path / "nested" / "cfg.yaml"
    written = importer.import_dataset("Anything", str(out))

    assert Path(written).exists()
    assert "tables:" in Path(written).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Delta write mode
# ---------------------------------------------------------------------------


def _processor(settings=None):
    from sdp.utils.parquet_post_processor import ParquetPostProcessor

    return ParquetPostProcessor({}, run_settings=settings or {})


def test_delta_write_mode_defaults_to_overwrite():
    """Unchanged default — flipping it would silently change what every
    existing delta output means."""
    assert _processor()._delta_write_mode() == "overwrite"


def test_delta_write_mode_is_configurable():
    assert _processor({"delta_write_mode": "append"})._delta_write_mode() == "append"
    assert _processor({"delta_write_mode": "error"})._delta_write_mode() == "error"


def test_delta_write_mode_is_case_insensitive():
    assert _processor({"delta_write_mode": " APPEND "})._delta_write_mode() == "append"


def test_unknown_delta_write_mode_falls_back_safely(caplog):
    proc = _processor({"delta_write_mode": "nonsense"})
    with caplog.at_level("WARNING"):
        assert proc._delta_write_mode() == "overwrite"
    assert "Unknown delta_write_mode" in caplog.text


# ---------------------------------------------------------------------------
# SDK reaches the engine flags
# ---------------------------------------------------------------------------


def test_sdk_generate_accepts_engine_arguments():
    """The engine layer was unreachable from the SDK — a gap introduced by
    adding engines to the CLI only."""
    import inspect

    from sdp.sdk import SyntheticDataPlatform

    params = inspect.signature(SyntheticDataPlatform.generate).parameters
    for name in ("engine", "engine_options", "epsilon"):
        assert name in params, f"SDK cannot reach --{name}"


def test_sdk_passes_engine_settings_to_the_service(monkeypatch, tmp_path):
    """The SDK calls the service directly, so the assertion is on the typed
    request rather than on a constructed argv."""
    import sdp.sdk as sdk_module
    from sdp.services.generation import GenerationOutcome
    from sdp.sdk import SyntheticDataPlatform

    captured = {}

    def fake_generate(request):
        captured["request"] = request
        return GenerationOutcome(exit_code=0, output_dir=tmp_path)

    monkeypatch.setattr(sdk_module, "generate_dataset", fake_generate)
    SyntheticDataPlatform().generate(
        config="cfg.yaml", output=str(tmp_path),
        engine="dp-marginal", epsilon=0.5, engine_options={"numeric_bins": 30},
    )

    request = captured["request"]
    assert request.engine == "dp-marginal"
    assert request.engine_options == {"numeric_bins": 30}


def test_sdk_does_not_route_generation_through_argparse(monkeypatch, tmp_path):
    """Regression: `generate` used to build an argv list and call cli.main,
    which capped what it could return at an exit code."""
    from sdp.services.generation import GenerationOutcome
    from sdp.sdk import SyntheticDataPlatform

    def explode(self, *argv):
        raise AssertionError(f"SDK fell back to the CLI: {argv}")

    monkeypatch.setattr(SyntheticDataPlatform, "run", explode)
    monkeypatch.setattr(
        "sdp.sdk.generate_dataset",
        lambda request: GenerationOutcome(
            exit_code=0, output_dir=tmp_path, row_counts={"users": 5},
        ),
    )

    result = SyntheticDataPlatform().generate(config="cfg.yaml", output=str(tmp_path))
    assert result.success
    # Richer than an exit code — the point of the refactor.
    assert result.row_counts == {"users": 5}


def test_sdk_extra_args_still_routes_through_the_cli(monkeypatch, tmp_path):
    """Raw CLI flags have no typed equivalent, so that escape hatch stays."""
    from sdp.sdk import CommandResult, SyntheticDataPlatform

    captured = {}

    def fake_run(self, *argv):
        captured["argv"] = list(argv)
        return CommandResult(command="generate", exit_code=0, argv=list(argv))

    monkeypatch.setattr(SyntheticDataPlatform, "run", fake_run)
    SyntheticDataPlatform().generate(
        config="cfg.yaml", output=str(tmp_path), extra_args=["--some-new-flag"],
    )
    assert "--some-new-flag" in captured["argv"]


# ---------------------------------------------------------------------------
# Packaging metadata
# ---------------------------------------------------------------------------


def _pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_uses_the_standard_project_table():
    data = _pyproject()
    assert "project" in data, "packaging metadata should use PEP 621 [project]"
    assert data["project"]["name"] == "synthetic-data-platform"
    assert data["project"]["requires-python"]


def test_pyproject_has_no_placeholder_author():
    authors = _pyproject()["project"].get("authors", [])
    rendered = str(authors).lower()
    assert "your name" not in rendered
    assert "you@example.com" not in rendered


def test_test_tooling_is_not_a_runtime_dependency():
    """pytest shipping in the wheel's install_requires was the defect."""
    data = _pyproject()
    runtime = data["tool"]["poetry"].get("dependencies", {})
    for package in ("pytest", "pytest-bdd", "pytest-cov", "ruff"):
        assert package not in runtime, f"{package} must live in the dev group"

    dev = data["tool"]["poetry"]["group"]["dev"]["dependencies"]
    assert "pytest" in dev and "ruff" in dev


def test_console_scripts_still_declared():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["sdp"] == "sdp.cli:main"
    assert scripts["sdp-api"] == "sdp.api.__main__:main"


def test_lint_config_present_and_scoped():
    data = _pyproject()
    lint = data["tool"]["ruff"]["lint"]
    assert "F" in lint["select"], "undefined names / unused imports must be caught"
    # FastAPI's Depends()/File() idiom is not the bug B008 targets.
    assert "B008" in data["tool"]["ruff"]["lint"]["per-file-ignores"]["sdp/api/app.py"]


# ---------------------------------------------------------------------------
# Dead code and packaging layout
# ---------------------------------------------------------------------------


def test_unreachable_er_generator_is_gone():
    """506 lines defining a second, unreferenced ERDiagramGenerator."""
    assert not (REPO_ROOT / "sdp" / "generators" / "ERGenerator.py").exists()


def test_slim_docker_variant_exists():
    slim = REPO_ROOT / "Dockerfile.slim"
    assert slim.exists()

    # Check the install commands themselves, not prose that mentions --extras.
    installs = [line.strip() for line in slim.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("RUN poetry install")]
    assert installs, "slim image must install the project"
    assert all("--extras" not in line for line in installs), installs
    assert all("--without dev" in line for line in installs), installs

    # The full image is the one that carries every extra.
    full = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--extras" in full


# ---------------------------------------------------------------------------
# Library output hygiene
# ---------------------------------------------------------------------------


#: The CLI presentation layer — stdout is its output channel, so printing
#: here is correct. Everything else is library code whose output would leak
#: into SDK, REST API and MCP consumers.
CLI_LAYER = ("cli.py", "cli_parser.py")
CLI_PACKAGES = ("cli_commands",)


def test_library_modules_do_not_print():
    """Service and library modules must log, not print.

    This is why `_run_gx_validation` had to change when it moved into the
    service layer: printing its report was fine while it lived in `cli.py`
    and wrong the moment SDK/API/MCP callers ran the same code.

    Parsed with `ast` rather than grepped: a substring search counts
    docstring examples and `_config_fingerprint(` as hits, which is how a
    'mixed print/logging' finding can be reported against a codebase that
    has no such problem.
    """
    import ast

    offenders = []
    for path in (REPO_ROOT / "sdp").rglob("*.py"):
        if path.name in CLI_LAYER or path.parent.name in CLI_PACKAGES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, f"library modules must log, not print: {offenders}"
