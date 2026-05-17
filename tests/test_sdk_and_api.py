"""Tests for the programmatic SDK facade (sdp.sdk) and the REST API (sdp.api).

The SDK tests run on the core install. The API tests are skipped automatically
unless the optional ``api`` extra (fastapi) is installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sdp import SyntheticDataPlatform
from sdp.sdk import CommandResult, GenerationResult, LintResult

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_YAML = REPO_ROOT / "examples" / "configs" / "yaml" / "01_simple_users.yaml"
BARE_YAML = REPO_ROOT / "examples" / "configs" / "yaml" / "11_bare_for_relationship_inference.yaml"


# ---------------------------------------------------------------------------
# SDK — lint
# ---------------------------------------------------------------------------

def test_sdk_lint_returns_structured_result():
    sdp = SyntheticDataPlatform()
    result = sdp.lint(SIMPLE_YAML)

    assert isinstance(result, LintResult)
    assert isinstance(result.issues, list)
    assert isinstance(result.report, str) and result.report
    # A shipped example config must be free of error-level issues.
    assert result.ok is True
    assert result.errors == []


def test_sdk_lint_missing_config_raises():
    sdp = SyntheticDataPlatform()
    with pytest.raises(Exception):
        sdp.lint(REPO_ROOT / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# SDK — generate
# ---------------------------------------------------------------------------

def test_sdk_generate_produces_frames(tmp_path: Path):
    sdp = SyntheticDataPlatform()
    result = sdp.generate(
        config=SIMPLE_YAML,
        output=tmp_path / "out",
        default_records=25,
        seed=42,
    )

    assert isinstance(result, GenerationResult)
    assert result.success, f"generation failed: exit={result.exit_code}"
    assert result.tables, "expected at least one generated table"
    assert result.total_records > 0
    for name, frame in result.frames.items():
        assert not frame.empty, f"table {name} is empty"


def test_sdk_generate_defaults_to_temp_output():
    sdp = SyntheticDataPlatform()
    result = sdp.generate(config=SIMPLE_YAML, default_records=10, seed=1)
    try:
        assert result.success
        assert result.output_dir.exists()
        assert result.total_records > 0
    finally:
        import shutil

        shutil.rmtree(result.output_dir, ignore_errors=True)


def test_sdk_generate_is_seed_reproducible(tmp_path: Path):
    sdp = SyntheticDataPlatform()
    a = sdp.generate(config=SIMPLE_YAML, output=tmp_path / "a", default_records=20, seed=7)
    b = sdp.generate(config=SIMPLE_YAML, output=tmp_path / "b", default_records=20, seed=7)
    assert a.success and b.success
    assert a.tables == b.tables
    for name in a.tables:
        assert a.frames[name].equals(b.frames[name])


# ---------------------------------------------------------------------------
# SDK — generic dispatch
# ---------------------------------------------------------------------------

def test_sdk_run_dispatches_to_cli(tmp_path: Path):
    sdp = SyntheticDataPlatform()
    result = sdp.run(
        "generate", "--config", str(SIMPLE_YAML),
        "--output", str(tmp_path / "run_out"), "--default-records", "10", "--seed", "3",
    )
    assert isinstance(result, CommandResult)
    assert result.command == "generate"
    assert result.success


# ---------------------------------------------------------------------------
# REST API — skipped unless the `api` extra is installed
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient

    from sdp.api import app as _api_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

requires_api = pytest.mark.skipif(
    not HAS_FASTAPI, reason="REST API requires the 'api' extra (fastapi)"
)


@pytest.fixture(scope="module")
def client():
    return TestClient(_api_app)


@requires_api
def test_api_healthz(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


@requires_api
def test_api_lint(client: TestClient):
    with SIMPLE_YAML.open("rb") as fh:
        resp = client.post("/lint", files={"config": ("01_simple_users.yaml", fh, "application/x-yaml")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["error_count"] == 0


@requires_api
def test_api_lint_rejects_unknown_extension(client: TestClient):
    resp = client.post("/lint", files={"config": ("bad.txt", b"nonsense", "text/plain")})
    assert resp.status_code == 415


@requires_api
def test_api_generate_zip(client: TestClient):
    with SIMPLE_YAML.open("rb") as fh:
        resp = client.post(
            "/generate",
            files={"config": ("01_simple_users.yaml", fh, "application/x-yaml")},
            data={"default_records": "15", "seed": "42", "response_format": "zip"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert len(resp.content) > 0


@requires_api
def test_api_generate_json_preview(client: TestClient):
    with SIMPLE_YAML.open("rb") as fh:
        resp = client.post(
            "/generate",
            files={"config": ("01_simple_users.yaml", fh, "application/x-yaml")},
            data={"default_records": "12", "seed": "42", "response_format": "json"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_records"] > 0
    assert body["tables"]
    assert body["preview"]
