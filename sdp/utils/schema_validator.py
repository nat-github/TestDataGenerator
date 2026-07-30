"""JSON Schema validation for YAML/JSON configs.

``schemas/sdp_config.schema.json`` existed for IDE autocompletion but
nothing enforced it, so a config could be schema-invalid and still load —
the parser is deliberately tolerant, skipping what it does not understand.
That tolerance is right for generation (one bad rule should not stop a run)
and wrong for authoring, where a typo silently dropping a `rules:` block is
exactly what you want to be told about.

This module closes that gap. ``lint`` reports schema violations alongside
its semantic checks; generation is unaffected.

Why not validate on every load
------------------------------
The schema is intentionally permissive (``additionalProperties: true`` on
tables and columns) because configs carry site-specific keys. Hard-failing
a run on a schema mismatch would break working configs for no safety gain.
Schema errors are therefore reported by ``lint`` and treated as warnings by
default; ``--strict-schema`` promotes them to errors for CI.

Excel configs are skipped: the schema describes the YAML/JSON document
shape, and an Excel workbook has no such document to validate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "sdp_config.schema.json"

#: Candidate locations, most specific first: bundled inside the installed
#: package, then the repo checkout. Resolved lazily so a missing file is a
#: clear error at call time rather than an import failure.
_SCHEMA_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "schemas" / _SCHEMA_NAME,   # sdp/schemas/
    Path(__file__).resolve().parents[2] / "schemas" / _SCHEMA_NAME,   # repo root
)


def _default_schema_path() -> Path:
    for candidate in _SCHEMA_CANDIDATES:
        if candidate.is_file():
            return candidate
    return _SCHEMA_CANDIDATES[-1]                     # for the error message


#: Resolved at import for callers that want to show it; may not exist.
SCHEMA_PATH = _default_schema_path()

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}


class SchemaUnavailable(RuntimeError):
    """The schema file or the jsonschema package could not be loaded."""


@dataclass
class SchemaViolation:
    """One schema error, located in the document."""
    path: str            # dotted/indexed location, e.g. "workflows[0].transitions[1]"
    message: str
    validator: str = ""

    def __str__(self) -> str:
        where = self.path or "<root>"
        return f"{where}: {self.message}"


def schema_supported(config_path: str) -> bool:
    """Whether this config is a document the schema can describe.

    Excel workbooks are not — the schema covers the YAML/JSON shape.
    """
    return Path(config_path).suffix.lower() in (_YAML_SUFFIXES | _JSON_SUFFIXES)


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the config schema. Raises ``SchemaUnavailable`` if it is missing."""
    path = Path(schema_path) if schema_path else _default_schema_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaUnavailable(f"Schema not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaUnavailable(f"Schema at {path} is not valid JSON: {exc}") from exc


def load_document(config_path: str) -> Any:
    """Parse a YAML or JSON config into a plain Python object."""
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in _JSON_SUFFIXES:
        return json.loads(text)

    import yaml

    return yaml.safe_load(text)


def _format_path(absolute_path) -> str:
    """jsonschema's deque of keys/indices → ``workflows[0].transitions[1]``."""
    parts: List[str] = []
    for item in absolute_path:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
        else:
            parts.append(str(item))
    return ".".join(parts)


def validate_document(
    document: Any,
    schema: Optional[Dict[str, Any]] = None,
) -> List[SchemaViolation]:
    """Validate an already-parsed config document.

    Returns every violation, sorted by location, rather than stopping at the
    first — a config with three typos should report three.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:                       # pragma: no cover
        raise SchemaUnavailable(
            "jsonschema is not installed — cannot validate the config schema"
        ) from exc

    validator = Draft202012Validator(schema if schema is not None else load_schema())
    violations = [
        SchemaViolation(
            path=_format_path(error.absolute_path),
            message=error.message,
            validator=str(error.validator),
        )
        for error in validator.iter_errors(document)
    ]
    return sorted(violations, key=lambda v: (v.path, v.message))


def validate_config_file(
    config_path: str,
    schema: Optional[Dict[str, Any]] = None,
) -> List[SchemaViolation]:
    """Validate a YAML/JSON config file against the schema.

    Returns ``[]`` for Excel configs, which the schema does not describe.
    A file that will not parse is reported as a single root-level
    violation rather than raising — ``lint`` should show it like any other
    problem.
    """
    if not schema_supported(config_path):
        logger.debug("Schema validation skipped for non-document config: %s", config_path)
        return []

    try:
        document = load_document(config_path)
    except Exception as exc:
        return [SchemaViolation(path="", message=f"could not parse: {exc}",
                                validator="parse")]

    if document is None:
        return [SchemaViolation(path="", message="config file is empty",
                                validator="parse")]

    return validate_document(document, schema)
