"""Quickstart for the unified LLM provider abstraction.

Sends a simple "list 3 country codes as JSON" prompt to whichever provider you
configure — Anthropic Claude, hosted OpenAI, LM Studio, Ollama, Groq, etc.
The same `chat()` call works against all of them; only the env vars change.

Run any of these from the project root:

    # 1. Local LLM via LM Studio (no API key needed)
    #    Start LM Studio, load a model, click "Start Server" (defaults to :1234)
    SDP_LLM_PROVIDER=lm-studio \\
    SDP_LLM_MODEL="meta-llama-3.1-8b-instruct" \\
        poetry run python examples/llm_quickstart.py

    # 2. Local LLM via Ollama (no API key needed)
    #    `ollama pull llama3.2` first
    SDP_LLM_PROVIDER=ollama \\
    SDP_LLM_MODEL="llama3.2" \\
        poetry run python examples/llm_quickstart.py

    # 3. Hosted OpenAI
    SDP_LLM_PROVIDER=openai \\
    OPENAI_API_KEY=sk-... \\
        poetry run python examples/llm_quickstart.py

    # 4. Anthropic Claude
    SDP_LLM_PROVIDER=anthropic \\
    ANTHROPIC_API_KEY=sk-ant-... \\
        poetry run python examples/llm_quickstart.py

    # 5. Groq (hosted Llama-3.3, very fast)
    SDP_LLM_PROVIDER=groq \\
    GROQ_API_KEY=gsk_... \\
        poetry run python examples/llm_quickstart.py

    # 6. Custom OpenAI-compatible host (vLLM, llama.cpp server, anywhere else)
    SDP_LLM_PROVIDER=lm-studio \\
    SDP_LLM_BASE_URL=http://my-gpu-box.local:8000/v1 \\
    SDP_LLM_MODEL=qwen2.5-coder-32b-instruct \\
        poetry run python examples/llm_quickstart.py
"""
from __future__ import annotations

import json
import os
import sys

# Make this script runnable from the project root without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdp.llm.multi_provider import chat, list_providers, resolve_config  # noqa: E402


SYSTEM_PROMPT = (
    "You are a strict-format assistant. Respond ONLY with a JSON array of "
    "exactly three uppercase ISO-3166 alpha-2 country codes. No prose."
)

USER_PROMPT = "Give me three country codes from continents you've never been to."


def main() -> int:
    cfg = resolve_config()
    print(f"Provider:  {cfg.provider.name}")
    print(f"Model:     {cfg.model}")
    print(f"Base URL:  {cfg.base_url or '(SDK)'}")
    print(f"API key:   {'set' if cfg.api_key else 'not set'}")
    print(f"Available: {', '.join(list_providers())}")
    print()

    print(">>> Sending prompt...")
    print(f"system: {SYSTEM_PROMPT}")
    print(f"user:   {USER_PROMPT}")
    print()

    text = chat(
        messages=[{"role": "user", "content": USER_PROMPT}],
        system=SYSTEM_PROMPT,
        max_tokens=200,
        temperature=0.0,
    )

    print(">>> Response:")
    print(text)
    print()

    # Try to parse it — local models can be sloppy with formatting, so we
    # tolerate code-fence wrappers and extra text.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        )
    try:
        parsed = json.loads(cleaned)
        print(f">>> Parsed JSON: {parsed!r}")
    except json.JSONDecodeError as exc:
        print(f">>> Could not parse as JSON: {exc}")
        print("    (Hosted models almost always nail the format. Smaller local "
              "models sometimes need a stricter system prompt or a JSON-mode "
              "extra_body={'response_format': {'type': 'json_object'}}.)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
