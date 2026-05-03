"""JSON fixture renderer — bare response/request bodies, no HTTP wrapper.

For each endpoint and response variant, write a JSON file containing just
the body. Useful when consumers want raw fixtures (unit-test golden files,
contract examples, manual API exploration).

Output structure:

    output/
      <endpoint_name>__<status>.json    (one file per endpoint+status)
      schemas/
        <SchemaName>.json               (one file per reusable schema)
      requests/
        <endpoint_name>.json            (request bodies, where defined)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from mocks.template_engine import TemplateEngine
from models.mock_models import MockConfig

logger = logging.getLogger(__name__)


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def render_json_fixtures(
    config: MockConfig,
    output_dir: Path,
    *,
    seed: Optional[int] = None,
    pretty: bool = True,
) -> List[Path]:
    """Render JSON response (and request) bodies into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_dir = output_dir / "requests"
    schemas_dir = output_dir / "schemas"

    engine = TemplateEngine(config, seed=seed)
    written: List[Path] = []

    indent = 2 if pretty else None

    # 1. Endpoint response bodies
    for endpoint in config.endpoints:
        for response in endpoint.responses:
            if response.body_schema is None and response.body_template is None:
                continue
            body = (
                engine._render_template_string(response.body_template)
                if response.body_template is not None
                else engine.render_field(response.body_schema)
            )
            fname = _safe(f"{endpoint.name}__{response.status}.json")
            out = output_dir / fname
            out.write_text(_dump(body, indent), encoding="utf-8")
            written.append(out)

        # 2. Endpoint request body (if any)
        if endpoint.request.body_schema is not None:
            requests_dir.mkdir(parents=True, exist_ok=True)
            body = engine.render_field(endpoint.request.body_schema)
            out = requests_dir / _safe(f"{endpoint.name}.json")
            out.write_text(_dump(body, indent), encoding="utf-8")
            written.append(out)

    # 3. Reusable schemas — one canonical example each
    if config.schemas:
        schemas_dir.mkdir(parents=True, exist_ok=True)
        for name, schema in config.schemas.items():
            body = engine.render_schema(schema)
            out = schemas_dir / _safe(f"{name}.json")
            out.write_text(_dump(body, indent), encoding="utf-8")
            written.append(out)

    logger.info("json_fixture: wrote %d file(s) to %s", len(written), output_dir)
    return written


def _dump(value: Any, indent: Optional[int]) -> str:
    if isinstance(value, str):
        # Raw template strings — already a literal payload, write verbatim
        return value
    return json.dumps(value, indent=indent, default=str)


def _safe(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", name).strip("_")
