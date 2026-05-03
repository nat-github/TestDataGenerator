"""Unified LLM provider abstraction.

Lets the rest of the codebase send chat-completion requests to any of:

  * **anthropic**    — hosted Claude via the official SDK (default)
  * **openai**       — hosted OpenAI (GPT-4o, GPT-4o-mini, o1, …)
  * **lm-studio**    — local LLM via LM Studio's OpenAI-compatible server
                       (http://localhost:1234/v1)
  * **ollama**       — local LLM via Ollama's OpenAI-compatible endpoint
                       (http://localhost:11434/v1)
  * **azure-openai** — Azure OpenAI Service deployments
  * **groq**         — Groq Cloud (fast Llama / Mixtral inference)
  * **together**     — Together AI (open-weight model hosting)
  * **openrouter**   — OpenRouter (gateway to 100+ models, single key)

Almost everything except `anthropic` uses the OpenAI chat-completions wire format,
so they share a single HTTP path. We use ``requests`` (already a project
dependency) instead of pulling in the ``openai`` SDK as a hard dependency —
that keeps the install lean and works equally well for any OpenAI-compatible
backend (LM Studio, Ollama, vLLM, llama.cpp's server, etc.).

Configuration is resolved in this order, most-specific wins:

  1. Explicit kwargs to ``chat(...)``
  2. Environment variables (``SDP_LLM_*``, then provider-specific keys)
  3. Built-in per-provider defaults

Quick local-LLM example
-----------------------
::

    # Point the platform at LM Studio (running on default port)
    export SDP_LLM_PROVIDER=lm-studio
    export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"

    python main.py infer-relationships --config bare.yaml \
        --config-output suggested.yaml --method llm

Or call from Python directly::

    from llm.multi_provider import chat

    text = chat(
        messages=[{"role": "user", "content": "Reply with the JSON {\"ok\": true}"}],
        provider="lm-studio",
        model="meta-llama-3.1-8b-instruct",
        system="You are an expert assistant. Reply with valid JSON only.",
    )
    print(text)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderProfile:
    """Static defaults for a provider."""
    name: str
    kind: str                   # "anthropic" | "openai-compat"
    default_base_url: Optional[str]
    default_model: str
    api_key_env: Optional[str]  # primary env var name (None means no key required)
    needs_api_key: bool


PROVIDERS: Dict[str, ProviderProfile] = {
    "anthropic": ProviderProfile(
        name="anthropic", kind="anthropic",
        default_base_url=None,
        default_model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY", needs_api_key=True,
    ),
    "openai": ProviderProfile(
        name="openai", kind="openai-compat",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY", needs_api_key=True,
    ),
    "lm-studio": ProviderProfile(
        name="lm-studio", kind="openai-compat",
        default_base_url="http://localhost:1234/v1",
        default_model="local-model",
        api_key_env=None, needs_api_key=False,
    ),
    "ollama": ProviderProfile(
        name="ollama", kind="openai-compat",
        default_base_url="http://localhost:11434/v1",
        default_model="llama3.2",
        api_key_env=None, needs_api_key=False,
    ),
    "azure-openai": ProviderProfile(
        name="azure-openai", kind="openai-compat",
        default_base_url=None,  # must be supplied via env: AZURE_OPENAI_ENDPOINT
        default_model="gpt-4o",
        api_key_env="AZURE_OPENAI_API_KEY", needs_api_key=True,
    ),
    "groq": ProviderProfile(
        name="groq", kind="openai-compat",
        default_base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY", needs_api_key=True,
    ),
    "together": ProviderProfile(
        name="together", kind="openai-compat",
        default_base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY", needs_api_key=True,
    ),
    "openrouter": ProviderProfile(
        name="openrouter", kind="openai-compat",
        default_base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4",
        api_key_env="OPENROUTER_API_KEY", needs_api_key=True,
    ),
}


# Generic env var overrides
ENV_PROVIDER = "SDP_LLM_PROVIDER"
ENV_MODEL = "SDP_LLM_MODEL"
ENV_BASE_URL = "SDP_LLM_BASE_URL"
ENV_API_KEY = "SDP_LLM_API_KEY"


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------


@dataclass
class ResolvedConfig:
    provider: ProviderProfile
    model: str
    base_url: Optional[str]
    api_key: Optional[str]


def resolve_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedConfig:
    """Resolve provider configuration from explicit args + env vars + defaults."""
    name = (provider or os.environ.get(ENV_PROVIDER) or "anthropic").lower().strip()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {name!r}. "
            f"Supported: {sorted(PROVIDERS.keys())}"
        )
    profile = PROVIDERS[name]

    resolved_model = model or os.environ.get(ENV_MODEL) or profile.default_model
    resolved_base_url = base_url or os.environ.get(ENV_BASE_URL) or profile.default_base_url

    # Azure needs an endpoint — read from its dedicated env if not already set
    if name == "azure-openai" and not resolved_base_url:
        resolved_base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")

    # Resolve API key: explicit > generic env > provider-specific env
    resolved_key = api_key or os.environ.get(ENV_API_KEY)
    if not resolved_key and profile.api_key_env:
        resolved_key = os.environ.get(profile.api_key_env)

    if profile.needs_api_key and not resolved_key:
        raise EnvironmentError(
            f"Provider {name!r} requires an API key. Set {profile.api_key_env} "
            f"(or {ENV_API_KEY}) in your environment, or pass api_key=... explicitly."
        )
    if profile.kind == "openai-compat" and not resolved_base_url:
        raise EnvironmentError(
            f"Provider {name!r} requires a base_url. "
            f"Set {ENV_BASE_URL} in your environment or pass base_url=... explicitly."
        )

    return ResolvedConfig(
        provider=profile,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_key,
    )


# ---------------------------------------------------------------------------
# Public API: chat()
# ---------------------------------------------------------------------------


def chat(
    messages: List[Dict[str, str]],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: int = 120,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Send a chat completion request and return the assistant's text content.

    Parameters
    ----------
    messages:
        List of ``{"role": "user"|"assistant", "content": str}`` dicts.
    provider:
        One of ``PROVIDERS`` (e.g. ``"anthropic"``, ``"lm-studio"``). Defaults
        to ``$SDP_LLM_PROVIDER`` or ``"anthropic"``.
    model:
        Provider-specific model identifier. Defaults to ``$SDP_LLM_MODEL`` or
        the provider's default.
    base_url:
        Override for OpenAI-compatible providers (e.g. LM Studio on a custom
        port). Ignored by ``anthropic``.
    api_key:
        Explicit API key. Falls back to env vars per provider.
    system:
        Optional system prompt. For Anthropic this becomes a cache-controlled
        system block; for OpenAI-compat backends it's prepended as a system
        message.
    max_tokens, temperature, timeout:
        Standard generation knobs.
    extra_body:
        Additional fields merged into the OpenAI-compat request body — useful
        for ``response_format``, ``top_p``, vendor-specific options, etc.
        Ignored by ``anthropic`` (use the SDK directly for Claude-only knobs).
    """
    cfg = resolve_config(provider=provider, model=model, base_url=base_url, api_key=api_key)
    logger.debug("LLM chat → provider=%s model=%s", cfg.provider.name, cfg.model)

    if cfg.provider.kind == "anthropic":
        return _chat_anthropic(messages, system, cfg, max_tokens, temperature)
    return _chat_openai_compat(messages, system, cfg, max_tokens, temperature, timeout, extra_body)


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


def _chat_anthropic(
    messages: List[Dict[str, str]],
    system: Optional[str],
    cfg: ResolvedConfig,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'anthropic' package is required to use provider='anthropic'. "
            "Run: poetry add anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=cfg.api_key)
    system_param: Any = None
    if system:
        # cache_control keeps repeated system prompts on the prompt cache,
        # which is by far the cheapest knob you can turn for token cost.
        system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system_param is not None:
        kwargs["system"] = system_param

    response = client.messages.create(**kwargs)
    if not response.content:
        return ""
    # Anthropic returns a list of content blocks; we only ever ask for text.
    return getattr(response.content[0], "text", "") or ""


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (used by openai, lm-studio, ollama, groq, …)
# ---------------------------------------------------------------------------


def _chat_openai_compat(
    messages: List[Dict[str, str]],
    system: Optional[str],
    cfg: ResolvedConfig,
    max_tokens: int,
    temperature: float,
    timeout: int,
    extra_body: Optional[Dict[str, Any]],
) -> str:
    import requests  # already a project dep

    full_messages: List[Dict[str, str]] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    body: Dict[str, Any] = {
        "model": cfg.model,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra_body:
        body.update(extra_body)

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        # Azure uses ``api-key``, everyone else uses ``Authorization: Bearer``.
        if cfg.provider.name == "azure-openai":
            headers["api-key"] = cfg.api_key
        else:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    response = requests.post(url, json=body, headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"{cfg.provider.name} returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response shape from {cfg.provider.name}: {json.dumps(data)[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def list_providers() -> List[str]:
    """Sorted provider names — useful for --help text and docs."""
    return sorted(PROVIDERS.keys())


def describe_provider(name: str) -> ProviderProfile:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name!r}")
    return PROVIDERS[name]
