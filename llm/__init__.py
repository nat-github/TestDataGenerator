"""
llm — GenAI / LLM integration layer for the Synthetic Data Platform.

Modules
-------
client          Shared Anthropic client factory with prompt caching.
relationship_inferrer
                Infers FK/PK relationships between tables using Claude.
schema_enricher
                Enriches bare column configs with suggested rules, types,
                business values, and null rates using Claude.
"""

from llm.client import get_client, DEFAULT_MODEL  # noqa: F401
