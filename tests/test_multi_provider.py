"""Tests for the unified LLM provider abstraction (`llm.multi_provider`).

These exercise:
  - config resolution (explicit args > env vars > built-in defaults)
  - OpenAI-compat request shape (LM Studio, Ollama, Groq, …)
  - Anthropic-specific dispatch path (system prompt becomes a cache_control block)
  - HTTP errors surface as RuntimeError with provider context
  - Bearer auth vs Azure's `api-key` header
  - System prompt prepending for OpenAI-compat / system param for Anthropic

No real network calls — `requests.post` is monkeypatched and the Anthropic SDK
is stubbed in-place.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from llm import multi_provider
from llm.multi_provider import (
    PROVIDERS,
    ResolvedConfig,
    chat,
    describe_provider,
    list_providers,
    resolve_config,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


def _ok_payload(text: str = "hello world") -> Dict[str, Any]:
    """OpenAI-compatible chat.completions response shape."""
    return {
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "model": "fake",
    }


@pytest.fixture
def _capture_requests(monkeypatch: pytest.MonkeyPatch):
    """Intercept requests.post and capture (url, json_body, headers, timeout)."""
    captured: Dict[str, Any] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(_ok_payload("captured"))

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    return captured


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch):
    """Strip any LLM-related env vars so tests stay deterministic."""
    for key in [
        "SDP_LLM_PROVIDER", "SDP_LLM_MODEL", "SDP_LLM_BASE_URL", "SDP_LLM_API_KEY",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
        "TOGETHER_API_KEY", "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
    ]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Provider registry sanity
# ---------------------------------------------------------------------------


def test_provider_registry_lists_all_supported_backends():
    names = list_providers()
    for required in ["anthropic", "openai", "lm-studio", "ollama",
                     "azure-openai", "groq", "together", "openrouter"]:
        assert required in names


def test_describe_provider_returns_profile():
    p = describe_provider("lm-studio")
    assert p.kind == "openai-compat"
    assert p.default_base_url.startswith("http://localhost:1234")
    assert p.needs_api_key is False


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_resolve_uses_built_in_defaults_for_lm_studio():
    cfg = resolve_config(provider="lm-studio")
    assert cfg.provider.name == "lm-studio"
    assert cfg.base_url == "http://localhost:1234/v1"
    # No key required; resolves to None
    assert cfg.api_key is None


def test_explicit_args_beat_env_and_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SDP_LLM_PROVIDER", "lm-studio")
    monkeypatch.setenv("SDP_LLM_MODEL", "env-model")
    cfg = resolve_config(provider="ollama", model="explicit-model")
    assert cfg.provider.name == "ollama"
    assert cfg.model == "explicit-model"


def test_env_vars_override_provider_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SDP_LLM_PROVIDER", "lm-studio")
    monkeypatch.setenv("SDP_LLM_BASE_URL", "http://host.docker.internal:1234/v1")
    monkeypatch.setenv("SDP_LLM_MODEL", "qwen2.5-coder-7b-instruct")
    cfg = resolve_config()
    assert cfg.provider.name == "lm-studio"
    assert cfg.base_url == "http://host.docker.internal:1234/v1"
    assert cfg.model == "qwen2.5-coder-7b-instruct"


def test_resolve_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_config(provider="not-a-thing")


def test_anthropic_requires_api_key():
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        resolve_config(provider="anthropic")


def test_anthropic_picks_up_provider_specific_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = resolve_config(provider="anthropic")
    assert cfg.api_key == "sk-ant-test"


def test_generic_sdp_api_key_works_for_any_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SDP_LLM_API_KEY", "generic-key")
    cfg = resolve_config(provider="openai")
    assert cfg.api_key == "generic-key"


def test_azure_openai_demands_endpoint():
    """Azure has no built-in default base URL — fail loudly when missing."""
    with pytest.raises(EnvironmentError):
        resolve_config(provider="azure-openai", api_key="k")


def test_azure_openai_uses_endpoint_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-aoai.openai.azure.com/openai/deployments/foo")
    cfg = resolve_config(provider="azure-openai")
    assert cfg.base_url.startswith("https://my-aoai.openai.azure.com")


# ---------------------------------------------------------------------------
# OpenAI-compat HTTP path — request shape
# ---------------------------------------------------------------------------


def test_lm_studio_request_shape(_capture_requests: Dict[str, Any]):
    out = chat(
        messages=[{"role": "user", "content": "hi"}],
        provider="lm-studio",
        model="local-model",
    )
    assert out == "captured"
    assert _capture_requests["url"] == "http://localhost:1234/v1/chat/completions"
    body = _capture_requests["json"]
    assert body["model"] == "local-model"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    # LM Studio doesn't need a key — no Authorization header
    assert "Authorization" not in (_capture_requests["headers"] or {})


def test_system_prompt_is_prepended_for_openai_compat(_capture_requests: Dict[str, Any]):
    chat(
        messages=[{"role": "user", "content": "hi"}],
        provider="lm-studio",
        system="You are a strict JSON-only assistant.",
    )
    msgs = _capture_requests["json"]["messages"]
    assert msgs[0]["role"] == "system"
    assert "JSON-only" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_openai_uses_bearer_auth(monkeypatch: pytest.MonkeyPatch, _capture_requests: Dict[str, Any]):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chat(messages=[{"role": "user", "content": "hi"}], provider="openai")
    assert _capture_requests["headers"]["Authorization"] == "Bearer sk-test"
    assert _capture_requests["url"].startswith("https://api.openai.com/v1")


def test_azure_uses_api_key_header(monkeypatch: pytest.MonkeyPatch, _capture_requests: Dict[str, Any]):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://acme.openai.azure.com/openai/deployments/x")
    chat(messages=[{"role": "user", "content": "hi"}], provider="azure-openai")
    headers = _capture_requests["headers"]
    assert headers.get("api-key") == "azure-key"
    # Azure must NOT also set Authorization Bearer
    assert "Authorization" not in headers


def test_extra_body_is_merged_into_request(_capture_requests: Dict[str, Any]):
    chat(
        messages=[{"role": "user", "content": "hi"}],
        provider="lm-studio",
        extra_body={"response_format": {"type": "json_object"}, "top_p": 0.9},
    )
    body = _capture_requests["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["top_p"] == 0.9


def test_http_error_is_surfaced_with_provider_context(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResponse({}, status_code=500, text="internal server error")

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(RuntimeError, match="lm-studio returned HTTP 500"):
        chat(messages=[{"role": "user", "content": "hi"}], provider="lm-studio")


def test_unexpected_response_shape_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResponse({"unexpected": "shape"})

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(RuntimeError, match="Unexpected response shape from lm-studio"):
        chat(messages=[{"role": "user", "content": "hi"}], provider="lm-studio")


# ---------------------------------------------------------------------------
# Anthropic dispatch path
# ---------------------------------------------------------------------------


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, captured: Dict[str, Any]):
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeAnthropicResponse("anthropic-ok")


class _FakeAnthropicClient:
    def __init__(self, captured: Dict[str, Any], **_kwargs):
        self.messages = _FakeMessages(captured)


def test_anthropic_dispatch_uses_sdk(monkeypatch: pytest.MonkeyPatch):
    """The anthropic backend should hit the SDK, not requests.post."""
    captured: Dict[str, Any] = {}

    fake_anthropic_module = type("M", (), {})()
    fake_anthropic_module.Anthropic = lambda **kw: _FakeAnthropicClient(captured, **kw)

    import sys
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")

    out = chat(
        messages=[{"role": "user", "content": "hi"}],
        provider="anthropic",
        model="claude-sonnet-4-6",
        system="be terse",
        max_tokens=100,
    )
    assert out == "anthropic-ok"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["max_tokens"] == 100
    # System should be wrapped with cache_control for prompt caching
    sys_param = captured["system"]
    assert isinstance(sys_param, list)
    assert sys_param[0]["text"] == "be terse"
    assert sys_param[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_omits_system_param_when_not_provided(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}
    fake_anthropic_module = type("M", (), {})()
    fake_anthropic_module.Anthropic = lambda **kw: _FakeAnthropicClient(captured, **kw)

    import sys
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")

    chat(messages=[{"role": "user", "content": "hi"}], provider="anthropic")
    assert "system" not in captured


# ---------------------------------------------------------------------------
# Default provider is anthropic when SDP_LLM_PROVIDER unset
# ---------------------------------------------------------------------------


def test_default_provider_is_anthropic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    cfg = resolve_config()
    assert cfg.provider.name == "anthropic"
