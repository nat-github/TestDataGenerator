"""OpenAPI examples enricher — round-trip a spec and inject `example:` blocks.

Takes an existing OpenAPI 3.x document and returns an enriched copy where:

  - Every operation response gets a representative ``example:`` populated
    from the rendered MockConfig values.
  - Every named ``components.schemas`` entry gets a canonical ``example:``.
  - Authored ``example:`` / ``examples:`` blocks are preserved (we never
    overwrite — only fill in missing ones).

The result is a self-contained spec that documentation tools (Redoc,
Stoplight, Swagger UI) can render with realistic sample values, so
consumers don't see empty schemas in their preview.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from sdp.mocks.template_engine import TemplateEngine
from sdp.models.mock_models import MockConfig

logger = logging.getLogger(__name__)


def render_openapi_examples(
    config: MockConfig,
    output_dir: Path,
    *,
    source_spec: Union[str, Path, Dict[str, Any]],
    seed: Optional[int] = None,
    overwrite_existing: bool = False,
) -> List[Path]:
    """Write an enriched OpenAPI YAML alongside the original.

    Parameters
    ----------
    source_spec:
        Path to the original OpenAPI YAML/JSON, or an already-loaded dict.
    overwrite_existing:
        When True, replace existing ``example:`` blocks. Defaults to False
        (preserve hand-authored examples — only fill in missing ones).

    Returns
    -------
    A list with the single output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(source_spec)
    enriched = enrich_in_place(
        spec=deepcopy(spec),
        config=config,
        seed=seed,
        overwrite_existing=overwrite_existing,
    )

    out_path = output_dir / "openapi-enriched.yaml"
    out_path.write_text(yaml.safe_dump(enriched, sort_keys=False), encoding="utf-8")
    logger.info("openapi_examples: wrote enriched spec to %s", out_path)
    return [out_path]


# ---------------------------------------------------------------------------
# Pure-function entry point (no I/O) — tests use this directly
# ---------------------------------------------------------------------------


def enrich_in_place(
    *,
    spec: Dict[str, Any],
    config: MockConfig,
    seed: Optional[int] = None,
    overwrite_existing: bool = False,
) -> Dict[str, Any]:
    """Mutate ``spec`` to add example blocks where missing.

    The MockConfig is the source of truth for *what* values to inject; the
    OpenAPI spec is the source of truth for *where* to inject them.
    """
    engine = TemplateEngine(
        config, seed=seed, ignore_authored_examples=overwrite_existing,
    )

    # 1. components.schemas — one canonical example per named schema
    components = spec.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    for schema_name, schema_node in list(schemas.items()):
        if not isinstance(schema_node, dict):
            continue
        if not overwrite_existing and "example" in schema_node:
            continue
        mock_schema = config.get_schema(schema_name)
        if mock_schema is None:
            continue
        schema_node["example"] = engine.render_schema(mock_schema)

    # 2. paths.<path>.<method>.responses.<status>.content.<media>.example
    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method in ("get", "post", "put", "patch", "delete", "head", "options"):
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue
            endpoint = _find_endpoint(config, path=path, method=method.upper(), op=op)
            if endpoint is None:
                continue
            _inject_response_examples(
                op=op,
                endpoint=endpoint,
                engine=engine,
                overwrite_existing=overwrite_existing,
            )
            _inject_request_body_example(
                op=op,
                endpoint=endpoint,
                engine=engine,
                overwrite_existing=overwrite_existing,
            )

    return spec


# ---------------------------------------------------------------------------
# Walking helpers
# ---------------------------------------------------------------------------


def _find_endpoint(
    config: MockConfig,
    *,
    path: str,
    method: str,
    op: Dict[str, Any],
):
    """Match a (path, method) OpenAPI operation back to its `EndpointConfig`."""
    op_id = op.get("operationId")
    if op_id:
        # The importer snake-cases operationId — try that first.
        from sdp.mocks.openapi_importer import _Importer
        cand = _Importer.__dict__["_snake_case"].__func__(op_id)
        ep = config.get_endpoint(cand)
        if ep is not None:
            return ep
    # Fallback: linear search by (method, path)
    for ep in config.endpoints:
        if ep.method == method and ep.path == path:
            return ep
    return None


def _inject_response_examples(
    *,
    op: Dict[str, Any],
    endpoint,
    engine: TemplateEngine,
    overwrite_existing: bool,
) -> None:
    responses = op.get("responses") or {}
    for status_str, response_node in responses.items():
        if not isinstance(response_node, dict):
            continue
        try:
            status = int(status_str)
        except ValueError:
            continue
        # Find matching ResponseTemplate
        rt = next((r for r in endpoint.responses if r.status == status), None)
        if rt is None or rt.body_schema is None:
            continue
        content = response_node.setdefault("content", {})
        media_type = rt.body_content_type
        media_node = content.setdefault(media_type, {})
        if not overwrite_existing and "example" in media_node:
            continue
        media_node["example"] = engine.render_field(rt.body_schema)


def _inject_request_body_example(
    *,
    op: Dict[str, Any],
    endpoint,
    engine: TemplateEngine,
    overwrite_existing: bool,
) -> None:
    if endpoint.request.body_schema is None:
        return
    request_body = op.get("requestBody")
    if not isinstance(request_body, dict):
        return
    content = request_body.setdefault("content", {})
    media_type = endpoint.request.content_type or "application/json"
    media_node = content.setdefault(media_type, {})
    if not overwrite_existing and "example" in media_node:
        return
    media_node["example"] = engine.render_field(endpoint.request.body_schema)


# ---------------------------------------------------------------------------
# Source spec loading
# ---------------------------------------------------------------------------


def _load_spec(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(source, dict):
        return source
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"OpenAPI source spec not found: {p}")
    raw = p.read_text(encoding="utf-8")
    return yaml.safe_load(raw)
