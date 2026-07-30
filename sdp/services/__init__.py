"""Service layer — orchestration shared by every entry surface.

The CLI, the Python SDK, the REST API and the MCP server are presentation
layers over these functions. Nothing here imports ``sdp.cli``.
"""
from sdp.services.generation import (
    GenerationOutcome,
    GenerationRequest,
    generate_dataset,
)

__all__ = ["GenerationOutcome", "GenerationRequest", "generate_dataset"]
