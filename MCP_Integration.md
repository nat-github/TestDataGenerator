# MCP Integration

> **TL;DR** — this platform ships an MCP server. Wire it into Claude Desktop,
> Claude Code, **LM Studio**, Cursor, or any other MCP-aware client and the
> assistant can drive data generation, mocks, validation, and relationship
> inference on the user's behalf — using **whichever LLM you've configured**,
> not just Claude.

---

## Table of contents

1. [What is MCP?](#1-what-is-mcp)
2. [Why MCP matters](#2-why-mcp-matters)
3. [How this platform benefits from MCP](#3-how-this-platform-benefits-from-mcp)
4. [Architecture in this solution](#4-architecture-in-this-solution)
5. [The 9 tools + 1 resource exposed](#5-the-9-tools--1-resource-exposed)
6. [Install + run the server](#6-install--run-the-server)
7. [Wire into LM Studio](#7-wire-into-lm-studio)
8. [Wire into Claude Desktop](#8-wire-into-claude-desktop)
9. [Wire into Claude Code](#9-wire-into-claude-code)
10. [Wire into Cursor / Zed / OpenAI Agents SDK](#10-wire-into-cursor--zed--openai-agents-sdk)
11. [Multi-provider LLM tools](#11-multi-provider-llm-tools)
12. [Demo prompts](#12-demo-prompts)
13. [Safety notes](#13-safety-notes)
14. [Troubleshooting](#14-troubleshooting)
15. [FAQ](#15-faq)
16. [Implementation walkthrough](#16-implementation-walkthrough)

---

## 1. What is MCP?

**MCP — Model Context Protocol** — is an open, JSON-RPC-over-stdio (or HTTP)
protocol that lets AI assistants talk to external systems through one
uniform interface.

The mental model:

```
┌────────────────────┐     JSON-RPC      ┌─────────────────────────┐
│   AI client         │   (stdio / HTTP)  │   MCP server             │
│  (LM Studio, Claude│ <───────────────► │  (this project's         │
│   Desktop, Cursor, │  tool calls,      │   mcp_server module)     │
│   Zed, ChatGPT...)  │  resource reads,  │                          │
│                     │  prompts          │                          │
└────────────────────┘                    └────────────┬────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │  Synthetic Data         │
                                          │  Platform internals:    │
                                          │  generators/, mocks/,   │
                                          │  ml/, llm/multi_provider│
                                          │  validators/            │
                                          └─────────────────────────┘
```

The protocol defines three primitives:

| Primitive | Purpose | This platform exposes |
|---|---|:---:|
| **Tools** | Functions the assistant can call (with JSON-Schema-typed args) | **9** |
| **Resources** | Read-only data the assistant can fetch by URI | **1** |
| **Prompts** | Reusable prompt templates | not used |

### Why use a *protocol* at all?

Before MCP, each assistant + integration combination needed bespoke glue.
Want Claude Desktop to talk to your database? Custom plugin. Want Cursor to
do the same? Different plugin. Want ChatGPT? Yet another integration.

MCP makes this an N+M problem instead of N×M:

- **Servers** (your integrations) speak MCP
- **Clients** (the assistants) speak MCP
- Both ends compose without custom code

It's the USB-C of agent-tool integration. Open standard, widely adopted,
client-agnostic.

---

## 2. Why MCP matters

Without MCP, getting the platform's capabilities into an AI assistant looks
like one of these:

- **Hard-code the assistant.** Bake every CLI command into a custom prompt.
  Brittle, opaque, one-shot.
- **Build a wrapper API.** Stand up a REST/gRPC service in front of the
  platform. Adds infra, auth, deploy story.
- **Custom plugin per assistant.** Different code for Claude vs ChatGPT vs
  Cursor. N×M maintenance.

MCP collapses all of this. Implement once as a server, every MCP-aware
client gets the capability for free.

The shape of value:

| Without MCP | With MCP |
|---|---|
| User runs `python main.py mock-init --from spec.yaml --output mocks.yaml && python main.py mock-render ...` (memorising flags) | User says "make WireMock stubs from spec.yaml". Assistant chains the tool calls. |
| Each CLI flag is a separate context the user has to remember | Assistant introspects the tool schemas and asks for what it needs |
| Errors are stack traces the user has to interpret | Errors come back as data the assistant can react to ("the file doesn't exist — should I look in `examples/`?") |
| LLM-using features hard-locked to a specific provider | Provider chosen per-call via parameters — local LM Studio, Ollama, hosted Anthropic / OpenAI / Groq, all interchangeable |

---

## 3. How this platform benefits from MCP

### Three audiences, one engine

| Audience | Surface | What they want |
|---|---|---|
| Hands-on developer | CLI | Speed, reproducibility, scriptability |
| SME / non-technical reviewer | **Streamlit UI** | Click-through flow, no commands to memorise |
| AI-mediated user | **MCP server** | "Just do the thing" — natural language, multi-step workflows |

All three surfaces call the same underlying functions in `generators/`,
`mocks/`, `ml/`, `llm/`, and `validators/`. Adding a feature once exposes
it everywhere — no duplication.

### Concrete benefits with MCP wired up

1. **Conversational orchestration.** "Generate 500 rows from the e-commerce
   example, then validate them with Great Expectations" becomes one
   sentence. The assistant chains `generate_data` → `validate_data`,
   formats the result, surfaces failures.

2. **Discoverability without docs.** `list_examples` lets the assistant
   browse the bundled config library and pick the relevant one without the
   user having to know the file layout. The `sdp://example/{name}`
   resource lets the assistant *read* a config before deciding how to use
   it.

3. **Cross-platform reach.** The same MCP server works with Claude
   Desktop, Claude Code, Cursor, Zed, **LM Studio**, and the OpenAI Agents
   SDK. Pick the assistant your team already uses.

4. **Provider-agnostic LLM features.** Tools that need an LLM
   (`infer_relationships --method=llm`, `mock_enrich`) accept
   `llm_provider` / `llm_model` / `llm_base_url` parameters that route
   through `llm/multi_provider.py`. **An LM Studio user can ask LM
   Studio's model to call `infer_relationships` and have those LLM calls
   *also* go to LM Studio** — no Anthropic key required.

5. **Safety without policy code.** The `generate_data` tool clamps row
   counts to 10,000 (mirroring the UI cap) so an agent can't accidentally
   request a billion-row run that bills real money.

6. **One implementation, many invocations.** Whether a user types `python
   main.py validate-data ...` in a terminal, clicks Validate in
   Streamlit, or asks LM Studio's chat to validate output, the *same
   function* runs.

---

## 4. Architecture in this solution

```
                           ┌───────────────────────────────┐
                           │   AI client                    │
                           │   (LM Studio / Claude Desktop /│
                           │    Cursor / Zed / OpenAI       │
                           │    Agents SDK / etc.)           │
                           └────────────┬──────────────────┘
                                        │ JSON-RPC
                                        │ (stdio by default)
                                        ▼
       ┌─────────────────────────────────────────────────────┐
       │   mcp_server/server.py  (FastMCP)                   │
       │                                                       │
       │   @mcp.tool() generate_data(...)                     │
       │   @mcp.tool() validate_data(...)                     │
       │   @mcp.tool() infer_relationships(..., llm_provider=)│
       │   @mcp.tool() mock_init / mock_render / mock_enrich  │
       │   @mcp.tool() lint_config / list_examples            │
       │   @mcp.tool() llm_diagnose                            │
       │   @mcp.resource("sdp://example/{name}")              │
       └────────────┬────────────────────────────────────────┘
                    │
        ┌───────────┼─────────────┬───────────────┐
        ▼           ▼             ▼               ▼
 ┌────────────┐ ┌────────┐  ┌──────────┐  ┌──────────────────┐
 │generators/ │ │mocks/  │  │ml/        │  │llm/multi_provider│
 │data_gen    │ │render  │  │relationship│  │ (8 backends —    │
 │utils/parser│ │import  │  │_inferrer   │  │  Anthropic,      │
 │parquet     │ │scenarios│ │           │  │  OpenAI, LM      │
 │post-proc   │ │llm_enrich│ │          │  │  Studio, Ollama, │
 └────────────┘ └────────┘  └──────────┘  │  Groq, Together, │
                                           │  Azure, OpenRouter│
                                           └──────────────────┘
                                                    │
       validators/gx_validator.py ◄─────────────────┤
       (Great Expectations 1.x)                     │
                                                    ▼
                                           Hosted or local LLM
                                           (per llm_provider arg)
```

The MCP server itself **never makes LLM calls directly**. When a tool needs
an LLM (e.g. `mock_enrich`), it calls into `llm/multi_provider.chat()`,
which honours the `llm_provider` / `llm_model` / `llm_base_url` arguments
or, if those are absent, the `SDP_LLM_PROVIDER` / `SDP_LLM_MODEL` /
`SDP_LLM_BASE_URL` environment variables.

This separation is deliberate: the MCP server is a thin orchestrator. The
provider abstraction lives below it, so swapping LLM backends never
touches the MCP layer.

---

## 5. The 9 tools + 1 resource exposed

| Tool | What it does | Uses LLM? |
|---|---|:---:|
| `generate_data` | Run the synthesizer; emit Parquet (capped at 10k rows / table) | no |
| `lint_config` | Validate an Excel/YAML/JSON config without generating | no |
| `validate_data` | Run Great Expectations against generated Parquet (requires `--extras gx`) | no |
| `infer_relationships` | ML or LLM relationship inference; `method` ∈ `{ml, llm, both}` | optional |
| `mock_init` | OpenAPI / Postman / HAR → `sdp-mock-v1` YAML | no |
| `mock_render` | `sdp-mock-v1` → WireMock / JSON / Pact / Postman / OpenAPI examples | no |
| `mock_enrich` | LLM fills missing schema examples + drafts 4xx/5xx responses | yes |
| `llm_diagnose` | Pings the configured LLM, returns the resolved provider + reply | yes |
| `list_examples` | Discover bundled example configs | no |

| Resource template | Returns |
|---|---|
| `sdp://example/{name}` | Read a bundled example config by filename |

Tools that take an LLM also accept `llm_provider`, `llm_model`,
`llm_base_url` — see §11.

---

## 6. Install + run the server

```bash
# One-time: install the optional `mcp` extra
poetry install --extras mcp

# (Optional, but recommended) add Great Expectations for validate_data
poetry install --extras "mcp gx"

# Run the server (stdio transport)
poetry run python -m mcp_server.server
```

Stdio is the default. The server runs as a long-lived subprocess; clients
launch it themselves through their config (you don't typically run it by
hand — see §7–§10).

To pre-set a default LLM provider for the whole server, export env vars
before launch (or set them in the client's config block):

```bash
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
export SDP_LLM_BASE_URL=http://localhost:1234/v1
```

---

## 7. Wire into LM Studio

LM Studio added MCP client support in the **0.3.x** series. Once enabled,
the local model in LM Studio can call any MCP server's tools and read its
resources — same protocol Claude Desktop uses.

### Step 1 — confirm LM Studio's MCP support

In LM Studio: **Settings → Developer → Model Context Protocol (MCP)**.
If it's not on, enable it. (If your LM Studio version doesn't show an MCP
section, update to a newer build.)

### Step 2 — add the platform's MCP server

LM Studio reads MCP server configs from a JSON file named **`mcp.json`**
in its application data folder. The exact path varies by OS — LM Studio
exposes a "Reveal mcp.json" button in the MCP settings panel.

Add this entry:

```json
{
  "mcpServers": {
    "synthetic-data-platform": {
      "command": "C:/Users/natar/TestDataGeneration/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/natar/TestDataGeneration",
      "env": {
        "SDP_LLM_PROVIDER": "lm-studio",
        "SDP_LLM_MODEL": "meta-llama-3.1-8b-instruct",
        "SDP_LLM_BASE_URL": "http://localhost:1234/v1"
      }
    }
  }
}
```

(macOS / Linux: replace the `.venv\Scripts\python.exe` with
`.venv/bin/python`.)

The `env` block is the killer detail: **the LM Studio model is now
configured to call MCP tools that themselves use LM Studio for their LLM
calls.** When the model invokes `mock_enrich` or `infer_relationships
--method llm`, the platform's `multi_provider.chat()` routes back to LM
Studio's local server. Closed loop, no API key, fully offline.

### Step 3 — restart LM Studio

After saving `mcp.json`, restart LM Studio. The MCP settings panel should
show "synthetic-data-platform" as a connected server with 9 tools listed.

### Step 4 — try it

In a new chat with any local model loaded:

> "List the synthetic-data examples this platform ships with, then
> generate 500 rows from the e-commerce one. Put them in
> `D:/runs/from_lm_studio`."

LM Studio's model calls `list_examples`, then `generate_data`, then reports
back. No subprocess, no CLI, no Anthropic key.

---

## 8. Wire into Claude Desktop

Claude Desktop reads MCP configs from `claude_desktop_config.json`:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%AppData%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

Add the same `mcpServers` block:

```json
{
  "mcpServers": {
    "synthetic-data-platform": {
      "command": "C:/Users/natar/TestDataGeneration/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/natar/TestDataGeneration"
    }
  }
}
```

Restart Claude Desktop. The platform's tools appear under the tools
(hammer) icon in any new conversation.

---

## 9. Wire into Claude Code

Claude Code reads `~/.claude.json` (or the project-local `.claude/mcp.json`):

```json
{
  "mcpServers": {
    "synthetic-data-platform": {
      "command": "C:/Users/natar/TestDataGeneration/.venv/Scripts/python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/natar/TestDataGeneration"
    }
  }
}
```

Type `/mcp` in the Claude Code TUI to verify the server is connected and
its tools are listed.

---

## 10. Wire into Cursor / Zed / OpenAI Agents SDK

Same shape, different config file:

| Client | Config file |
|---|---|
| Cursor | `~/.cursor/mcp.json` |
| Zed | `~/.config/zed/settings.json` (under `"context_servers"`) |
| OpenAI Agents SDK | passed to the SDK constructor at runtime |

The `command` / `args` / `cwd` / `env` keys carry over verbatim. **The
server itself is client-agnostic** — it speaks MCP, that's it.

---

## 11. Multi-provider LLM tools

This is the "make it configurable for any model" piece.

The two MCP tools that use an LLM (`infer_relationships` with
`method="llm"` and `mock_enrich`) accept three optional parameters:

| Parameter | Purpose |
|---|---|
| `llm_provider` | One of `anthropic`, `openai`, `lm-studio`, `ollama`, `azure-openai`, `groq`, `together`, `openrouter` |
| `llm_model` | Provider-specific model identifier |
| `llm_base_url` | Override base URL (e.g. for a custom LM Studio port or a self-hosted vLLM) |

When omitted, these read from the env vars `SDP_LLM_PROVIDER`,
`SDP_LLM_MODEL`, `SDP_LLM_BASE_URL` in that order, and fall back to the
hosted Anthropic default (which then needs `ANTHROPIC_API_KEY`).

### The full provider matrix

| Provider | Default base URL | API key env | Notes |
|---|---|---|---|
| `anthropic` | (SDK default) | `ANTHROPIC_API_KEY` | Hosted Claude |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Hosted GPT |
| `lm-studio` | `http://localhost:1234/v1` | (none) | **Local — no key** |
| `ollama` | `http://localhost:11434/v1` | (none) | **Local — no key** |
| `azure-openai` | `$AZURE_OPENAI_ENDPOINT` | `AZURE_OPENAI_API_KEY` | Enterprise |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | Fast hosted Llama |
| `together` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | Open-weight hosting |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | Gateway to ~100 models |

### Sanity-check the connection with `llm_diagnose`

Before kicking off a long job, ask the assistant to call `llm_diagnose`:

> "Diagnose the LLM connection."

```json
{
  "ok": true,
  "provider": "lm-studio",
  "model": "meta-llama-3.1-8b-instruct",
  "base_url": "http://localhost:1234/v1",
  "api_key_set": false,
  "reply": "ready",
  "elapsed_ms": 642
}
```

If the model isn't reachable, you get `{"ok": false, "error": "..."}` —
the assistant can surface that without crashing the conversation.

---

## 12. Demo prompts

Once the server is wired up, try these in any MCP-aware client:

> **Discovery**
> "List the synthetic-data examples this platform ships with."
>
> Calls `list_examples`. Returns YAML / JSON / XLSX bundles + OpenAPI specs + mock fixtures.

> **Validation only**
> "Lint `examples/configs/yaml/02_ecommerce_relationships.yaml`."
>
> Calls `lint_config`. Reports table names, column counts, relationships.

> **Generation + validation**
> "Generate 1,000 rows from `examples/configs/yaml/04_distributions_and_business_values.yaml`
> with seed 99 into `output/from_mcp`, then validate it with Great
> Expectations."
>
> Chains `generate_data` → `validate_data`. Surfaces any failed expectations
> with samples.

> **OpenAPI → mocks**
> "Convert `examples/openapi/medium_tasks.yaml` into a sdp-mock-v1 config
> and render WireMock + Postman stubs from it."
>
> Chains `mock_init` → `mock_render`. Tells you where artefacts landed.

> **LLM-driven enrichment, fully local**
> "Read `examples/configs/yaml/10_bare_for_llm_enrichment.yaml`. Use a
> local LM Studio model to suggest enrichments for it."
>
> Reads via `sdp://example/...`, then calls `mock_enrich` with
> `llm_provider="lm-studio"`. The whole loop runs against your local
> model — no Anthropic key.

> **Pre-flight**
> "Diagnose the LLM connection before we run anything heavy."
>
> Calls `llm_diagnose`. Confirms the model is reachable in <1s.

---

## 13. Safety notes

- **Hard 10k row cap on `generate_data`.** Mirrors the Streamlit UI cap.
  An agent can't request a billion-row generation by mistake.
- **No network calls inside the server.** LLM calls happen only when the
  user explicitly invokes a tool that needs one (`infer_relationships
  --method=llm`, `mock_enrich`, `llm_diagnose`). They route through
  `llm/multi_provider.py` and respect the same env vars the rest of the
  platform does.
- **Read-only resource handler.** `sdp://example/{name}` only serves
  bundled examples; it can't be tricked into reading arbitrary paths.
- **Errors are returned as data.** Every tool catches its own exceptions
  and returns `{"ok": false, "error": "..."}` so the agent sees failures
  as a value to react to, not a transport-level break.
- **No secrets in tool args.** API keys come from env vars; the tool
  parameters never carry them.

---

## 14. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Client says "MCP server failed to start" | Path or `cwd` wrong in the config. Try the venv's Python directly: `C:/Users/natar/TestDataGeneration/.venv/Scripts/python.exe`. |
| Client connects but lists 0 tools | The MCP package isn't installed in that venv. `poetry install --extras mcp`. |
| `validate_data` fails with "Great Expectations not installed" | Run `poetry install --extras gx`. |
| `infer_relationships --method=llm` returns "API key required" | The default provider is `anthropic` and `ANTHROPIC_API_KEY` is unset. Either set it, or pass `llm_provider="lm-studio"`/`"ollama"` (no key), or set `SDP_LLM_PROVIDER` in the client's `env` block. |
| LM Studio's model can't reach the server's MCP tools | Check that LM Studio's "Developer → Model Context Protocol" toggle is on, the `mcp.json` is the file LM Studio actually reads (use the "Reveal" button), and you've restarted LM Studio after editing. |
| Tool calls work but never reach the right local model | The platform's `multi_provider` defaults to **port 1234** for LM Studio. If LM Studio is on a different port, set `SDP_LLM_BASE_URL=http://localhost:<port>/v1` in the `env` block of `mcp.json`. |
| `llm_diagnose` says `ok: false` with `EnvironmentError` | Provider needs an API key but it's not in the env. Set it in the `mcp.json` `env` block so the launched server inherits it. |
| Tests pass but a real client gets stale tool descriptions | Restart the client. Tool schemas are fetched at MCP handshake; new tools don't appear until the next handshake. |

---

## 15. FAQ

**Can I expose the MCP server over HTTP instead of stdio?**
The current server uses stdio (the default). To switch, set
`SDP_MCP_TRANSPORT=sse` (or whichever transport your MCP SDK supports)
before launch. Most clients prefer stdio for local servers, so this is
rarely needed.

**Can I add a new tool?**
Yes. Open `mcp_server/server.py`, decorate a function with `@mcp.tool()`,
write a docstring aimed at the *agent* (when to call it, what it
returns), and add a smoke test under `tests/test_mcp_server.py`. The
docstring becomes the tool's description that clients show to the model.

**Why aren't `delta` / `scd2` / `pii-scan` exposed?**
Those are file-pair workflows that benefit less from agent invocation
(they need two specific snapshots in specific paths). They can be added
if a use case emerges — the pattern is the same as the existing tools.

**Can the server run on one machine and the client on another?**
Yes, with caveats. Stdio transport requires the client to launch the
server as a subprocess, so they share a host. For remote servers, use
the SSE/HTTP transport (and authenticate accordingly). Most teams keep
the server local.

**Do the tools share state across calls?**
No. Each tool call is independent. State that needs to persist
(feedback store for relationship inference, generated data, etc.) lives
on disk and is referenced by path arguments.

**What if my LLM doesn't support tool calling?**
The MCP server doesn't need the *user's* LLM to support tool calling —
the *MCP client* handles that. As long as the client (LM Studio,
Claude Desktop, etc.) speaks MCP, the model behind it can be any model
the client supports. Smaller models will be less reliable at choosing
the right tool for the job, but the protocol works regardless.

**Why a 10k row cap on `generate_data`?**
Defense against accidental large runs. Agents that generate code can
easily hallucinate "1000000" instead of "1000" — the cap turns that into
a 10k run instead of a billion. Power users go through the CLI for
larger jobs; the cap doesn't apply there.

---

## 16. Implementation walkthrough

For developers who want to understand or extend the server.

### `mcp_server/server.py` — the FastMCP setup

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("synthetic-data-platform")

@mcp.tool()
def generate_data(config_path: str, output_dir: str, ...) -> Dict[str, Any]:
    """[docstring read by clients as the tool description]"""
    # Look up the config, run the generator, return a serialisable summary
    ...
```

FastMCP introspects the function signature and docstring to build the
JSON-Schema tool definition the client receives. **Type hints matter** —
they become the tool's parameter schema. Optional params get
`required: false` automatically.

### Errors as data

Every tool wraps its body in `try / except` and returns
`{"ok": False, "error": "..."}` on failure. This means:

- The agent sees a structured failure it can reason about
- Transport never breaks — the JSON-RPC reply is always well-formed
- The user gets a graceful explanation, not a stack trace

### Provider passthrough — the "configurable for any model" piece

```python
@mcp.tool()
def infer_relationships(
    config_path: str, config_output: str,
    method: str = "ml",
    ml_confidence: float = 0.55,
    llm_provider: Optional[str] = None,    # ← new
    llm_model: Optional[str] = None,       # ← new
    llm_base_url: Optional[str] = None,    # ← new
) -> Dict[str, Any]:
    ...
    if method in ("llm", "both"):
        llm = RelationshipInferrer(
            confidence_threshold=0.7,
            provider=llm_provider,         # ← threaded through
            model=llm_model,               # ← threaded through
            base_url=llm_base_url,         # ← threaded through
        )
```

`RelationshipInferrer.__init__` already accepts those kwargs (we wired
them up when we built the multi-provider abstraction). They flow into
`llm/multi_provider.chat()`, which dispatches to the right backend.

### `llm_diagnose` — the introspection tool

```python
@mcp.tool()
def llm_diagnose(provider=None, model=None, base_url=None) -> Dict[str, Any]:
    cfg = resolve_config(provider=provider, model=model, base_url=base_url)
    reply = chat(messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                 provider=provider, model=model, base_url=base_url, max_tokens=20)
    return {
        "ok": True, "provider": cfg.provider.name, "model": cfg.model,
        "base_url": cfg.base_url, "api_key_set": bool(cfg.api_key),
        "reply": reply.strip()[:200],
    }
```

This tool exists *only* to validate the LLM wiring. If `llm_diagnose`
works, every other LLM-using tool will too. If it returns
`{"ok": false}`, fix the env / config before calling anything heavier.

### Adding a new tool — the recipe

1. Add a function in `mcp_server/server.py`, decorate with `@mcp.tool()`.
2. Write the docstring **for the agent**: when to call it, what to
   expect back, any side effects.
3. Wrap the body in `try/except`; return `{"ok": False, "error": ...}` on
   failure.
4. Add a test in `tests/test_mcp_server.py` (use `_call(tool_name, ...)`
   to invoke the underlying function directly).
5. Update `test_expected_tools_registered` to include the new tool name.
6. Run the existing test suite to confirm you haven't broken anything.

That's it. No client config to change — clients re-fetch tool schemas on
the next handshake.

---

## 17. Where MCP fits in the platform's roadmap

MCP is the agent-driven counterpart to the CLI and Streamlit UI:

| Audience | Surface |
|---|---|
| Hands-on developer | CLI (`python main.py ...`) |
| SME / non-technical reviewer | Streamlit UI (`streamlit run ui/streamlit_app.py`) |
| AI-mediated user | **MCP server** (any MCP-aware client) |

All three surfaces share the same engine. Adding a feature once exposes
it everywhere.

That's the architectural payoff for keeping `generators/`, `mocks/`,
`ml/`, `llm/`, and `validators/` as pure Python with clean function
boundaries: the same code drives a CLI command, a Streamlit page, and an
MCP tool — no duplication.
