"""Loader and validator for `sdp-mock-v1` documents.

Accepts YAML or JSON; returns a fully-validated `MockConfig`. Errors are
surfaced with as much locator context as we can give (path through the
document) so authors can find typos quickly.

Sibling to `utils/config_parser.py` — same role, different model family.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import ValidationError

from sdp.models.mock_models import MockConfig

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = (".yaml", ".yml", ".json")


class MockConfigError(ValueError):
    """Raised when a `sdp-mock-v1` document fails to load or validate."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mock_config(path: Union[str, Path]) -> MockConfig:
    """Load and validate a `sdp-mock-v1` YAML or JSON file.

    Raises
    ------
    MockConfigError
        On any read, parse, or validation failure. The message includes the
        file path and (for ValidationError cases) the field-path with an
        explanation of what went wrong.
    """
    p = Path(path)
    if not p.exists():
        raise MockConfigError(f"Mock config file not found: {p}")
    if not p.is_file():
        raise MockConfigError(f"Mock config path is not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "Mock config has unexpected extension %r — proceeding anyway", p.suffix
        )

    raw = p.read_text(encoding="utf-8")
    return load_mock_config_str(raw, source=str(p))


def load_mock_config_str(raw: str, *, source: str = "<string>") -> MockConfig:
    """Parse a `sdp-mock-v1` YAML or JSON string into a MockConfig.

    Tries JSON first (cheap parse), falls back to YAML (which also accepts
    JSON). The `source` label is only used to make error messages useful.
    """
    raw = raw.strip()
    if not raw:
        raise MockConfigError(f"{source}: file is empty")

    parsed = _parse_yaml_or_json(raw, source)
    if not isinstance(parsed, dict):
        raise MockConfigError(
            f"{source}: top-level document must be a mapping, got {type(parsed).__name__}"
        )

    fmt = parsed.get("config_format")
    if fmt and fmt != "sdp-mock-v1":
        raise MockConfigError(
            f"{source}: unsupported config_format {fmt!r}; expected 'sdp-mock-v1'"
        )

    try:
        return MockConfig.model_validate(parsed)
    except ValidationError as exc:
        raise MockConfigError(_format_validation_error(exc, source)) from exc


def lint_mock_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Validate without raising — return a structured report instead.

    Useful for CLI `mock-lint` so the user sees every problem at once
    instead of fixing them one validate-fail at a time.
    """
    p = Path(path)
    report: Dict[str, Any] = {"path": str(p), "ok": False, "errors": [], "warnings": []}
    try:
        cfg = load_mock_config(p)
    except MockConfigError as exc:
        report["errors"].append(str(exc))
        return report

    # Soft warnings
    if not cfg.endpoints:
        report["warnings"].append("MockConfig has no endpoints — nothing to render")
    if not cfg.servers:
        report["warnings"].append(
            "MockConfig has no servers — renderers will use ./localhost defaults"
        )
    for ep in cfg.endpoints:
        if not ep.responses:
            # Caught earlier by validation but defensive.
            report["warnings"].append(f"endpoint {ep.name!r} has no responses")
        success = [r for r in ep.responses if 200 <= r.status < 300]
        if not success:
            report["warnings"].append(
                f"endpoint {ep.name!r} declares no 2xx response"
            )

    report["ok"] = not report["errors"]
    report["endpoints"] = len(cfg.endpoints)
    report["schemas"] = len(cfg.schemas)
    return report


def dump_mock_config(cfg: MockConfig, path: Union[str, Path]) -> None:
    """Serialise a MockConfig back to YAML — used by mock-init / round-trip tests."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(mode="json", exclude_none=True, by_alias=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_yaml_or_json(raw: str, source: str) -> Any:
    """Try JSON first, then YAML. Both raise MockConfigError on failure."""
    if raw.lstrip().startswith(("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MockConfigError(f"{source}: invalid JSON: {exc}") from exc
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise MockConfigError(f"{source}: invalid YAML: {exc}") from exc


def _format_validation_error(exc: ValidationError, source: str) -> str:
    """Turn a pydantic ValidationError into a multi-line author-friendly message."""
    lines = [f"{source}: {exc.error_count()} validation error(s):"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
