"""
Shared Anthropic client factory.

Usage
-----
    from llm.client import get_client, DEFAULT_MODEL

    client = get_client()          # reads ANTHROPIC_API_KEY from env
    client = get_client(api_key="sk-ant-...")  # explicit key

The client is module-level cached after the first call so all LLM modules
share a single connection pool.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

# Prompt-cache-friendly system prompt shared across all LLM calls.
# Keeping it stable (same text every call) maximises cache hit rate.
_SYSTEM_PROMPT = """\
You are an expert data engineer assistant embedded in a synthetic test-data \
generation platform called the FDL Synthetic Data Platform. \
You help infer relationships between database tables, enrich column \
configurations with realistic generation rules, and explain data schemas \
to business users.

Always respond with valid JSON unless the task instructions say otherwise.
Be concise and precise. Never hallucinate column names or table names that \
were not provided to you."""


@lru_cache(maxsize=1)
def get_client(api_key: Optional[str] = None):
    """Return a cached Anthropic client.  Reads ANTHROPIC_API_KEY from env if api_key is None."""
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'anthropic' package is required for LLM features. "
            "Run: poetry add anthropic"
        ) from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before using GenAI features:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    client = anthropic.Anthropic(api_key=key)
    logger.debug("Anthropic client initialised (model=%s)", DEFAULT_MODEL)
    return client


def system_prompt() -> str:
    return _SYSTEM_PROMPT
