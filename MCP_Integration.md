# MCP Integration

How to drive the Synthetic Data Platform from Claude (Desktop, Code, or any
other MCP-aware client) using the Model Context Protocol.

---

## What is MCP?

**MCP (Model Context Protocol)** is an open protocol — first published by
Anthropic in late 2024, now adopted by OpenAI, Google, and most major IDEs —
that lets AI assistants talk to external systems through a single, uniform
JSON-RPC interface.

The mental model:

```
┌──────────────────┐   stdio / HTTP / SSE    ┌──────────────────────┐
│   AI client      │  <───────────────────►  │    MCP server        │
│  (Claude Desktop, │                         │  (this project's     │
│  Claude Code,     │   tool calls,           │   mcp_server module) │
│  Cursor, etc.)    │   resource reads        │                      │
└──────────────────┘                          └──────────┬───────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │  The platform's      │
                                              │  generators, parsers, │
                                              │  inferrers, etc.      │
                                              └──────────────────────┘
```

The protocol defines three primitives:

| Primitive | Purpose |
|---|---|
| **Tools** | Functions the assistant can call (name + JSON-Schema typed args). |
| **Resources** | Read-only handles to data the assistant can fetch (URIs). |
| **Prompts** | Reusable prompt templates — *not* used by this server. |

This project implements **6 tools** and **1 resource template**.

---

## Why this matters for the platform

Today, using the platform means typing CLI commands. With MCP, an AI
assistant can drive the platform on the user's behalf:

> User: "Generate 500 rows of user data with realistic emails and IBANs."
>
> Claude (via MCP):
>   1. Calls `list_examples` — finds `01_simple_users.yaml`.
>   2. Calls `lint_config` to confirm it's valid.
>   3. Calls `generate_data` with that config, default_records=500.
>   4. Reports the output path back to the user.

The user gets the same result as the CLI flow, but conversationally and
without remembering subcommand names. Same output, lower friction.

This pattern composes well with the platform's other AI features:

- **Schema enrichment** — Claude can call `infer_relationships` after
  generation, then call `generate_data` again with the enriched config.
- **Mocks track** — "Make WireMock stubs from this OpenAPI spec" becomes a
  single conversational request.
- **Discoverability** — `list_examples` + `read_example` resource together
  let Claude browse the bundled examples without the user having to know
  the directory layout.

---

## Tools exposed by this server

Names match the MCP spec; descriptions are written for the agent.

| Tool | Purpose |
|---|---|
| `generate_data` | Run the synthesizer. Hard-capped at 10,000 rows per table — for larger runs the user is steered toward the CLI. |
| `lint_config` | Validate an Excel/YAML/JSON config without generating. |
| `infer_relationships` | Run the ML or LLM relationship inferrer over a bare config. |
| `mock_init` | Convert an OpenAPI / Postman / HAR artefact into a `sdp-mock-v1` YAML. |
| `mock_render` | Render mocks (WireMock / JSON / Pact / Postman / OpenAPI examples) from a `sdp-mock-v1` config. |
| `list_examples` | Enumerate the bundled example configs. |

## Resources exposed

| Resource template | Returns |
|---|---|
| `sdp://example/{name}` | Read a bundled example config by filename (e.g. `sdp://example/01_simple_users.yaml`). |

The resource template lets a client fetch the contents of any bundled
example without going through a tool call — handy when the assistant wants
to *understand* a config before deciding how to use it.

---

## Install + run the server

```bash
# One-time: install the optional `mcp` extra
poetry install --extras mcp

# Run the server (stdio transport, default)
poetry run python -m mcp_server.server
```

The server speaks MCP over stdio by default. To use a different transport
(if the SDK version supports it), set `SDP_MCP_TRANSPORT`.

---

## Wire it into Claude Desktop

Claude Desktop loads MCP servers from `claude_desktop_config.json`. On macOS
that's `~/Library/Application Support/Claude/claude_desktop_config.json`;
on Windows it's `%AppData%\Claude\claude_desktop_config.json`.

Add an `mcpServers` entry:

```json
{
  "mcpServers": {
    "synthetic-data-platform": {
      "command": "poetry",
      "args": [
        "--directory", "C:/Users/natar/TestDataGeneration",
        "run", "python", "-m", "mcp_server.server"
      ]
    }
  }
}
```

Replace the path with your own clone's location. Restart Claude Desktop, and
the platform's tools will appear under the "tools" hammer-icon menu in any
new conversation.

> **Tip:** if Poetry isn't on Claude Desktop's PATH, point `command` at the
> Poetry-managed Python interpreter directly:
> `C:/Users/natar/TestDataGeneration/.venv/Scripts/python.exe` and drop the
> `--directory ... run` arguments.

---

## Wire it into Claude Code

Claude Code reads `~/.claude.json` (or the project-local `.claude/mcp.json`).
Add the same shape:

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

In the Claude Code TUI, type `/mcp` to verify the server is connected and
its tools are available.

---

## Wire it into Cursor / other clients

Most MCP-aware IDEs follow the same shape:

- A single command to launch the server (`python -m mcp_server.server`)
- stdio as the default transport
- A JSON config block under `mcpServers`

Refer to your client's docs for the exact file path. The server itself is
client-agnostic.

---

## Demo prompts

Once the server is wired up, try these prompts in any MCP-aware client:

> "List the synthetic-data examples this platform ships with."
>
> Claude calls `list_examples`, returns a tidy summary.

> "Lint `examples/configs/yaml/02_ecommerce_relationships.yaml`."
>
> Claude calls `lint_config`, reports table names + column counts.

> "Generate 1,000 rows from `examples/configs/yaml/04_distributions_and_business_values.yaml` with seed 99 and put them in `output/from_mcp`."
>
> Claude calls `generate_data` with the right arguments, confirms the output.

> "Convert `examples/openapi/medium_tasks.yaml` into a sdp-mock-v1 config and render WireMock + Postman stubs from it."
>
> Claude chains `mock_init` → `mock_render` and tells you where the artefacts landed.

> "Read the bare-for-LLM-enrichment example, then explain what columns the LLM would likely enrich."
>
> Claude reads `sdp://example/10_bare_for_llm_enrichment.yaml` and analyses
> it — no tool call needed, the resource is enough.

---

## Safety notes

- **Hard cap on row count.** The `generate_data` tool clamps `default_records`
  to 10,000 per table. An agent can't accidentally request a billion-row
  generation.
- **No network calls.** The server runs everything in-process. The only LLM
  calls happen if the user explicitly asks for `infer_relationships --method
  llm`, which routes through `llm/multi_provider.py` and respects the same
  env vars (`SDP_LLM_PROVIDER`, etc.) the rest of the platform uses.
- **Read-only resource handler.** The `sdp://example/...` resource template
  only serves bundled example files; it can't be tricked into reading
  arbitrary paths on disk.
- **Errors are returned, not raised.** Every tool catches its own
  exceptions and returns `{"ok": false, "error": "..."}`. The agent sees
  failures as data, not as MCP transport errors.

---

## How the server maps onto the platform's other features

| Platform capability | MCP exposure |
|---|---|
| `python main.py generate ...` | `generate_data` tool |
| `python main.py lint ...` | `lint_config` tool |
| `python main.py infer-relationships ...` | `infer_relationships` tool |
| `python main.py mock-init ...` | `mock_init` tool |
| `python main.py mock-render ...` | `mock_render` tool |
| Example browsing (`examples/configs/`) | `list_examples` tool + `sdp://example/{name}` resource |
| `python main.py delta` / `scd2` | *Not yet exposed* — they're file-pair workflows that benefit less from agent invocation. Add a tool if you find a use case. |
| `python main.py pii-scan` | *Not yet exposed* — similar reasoning. |
| `python main.py enrich` (LLM schema enrichment) | *Not yet exposed* — would require the agent to handle an LLM-within-an-LLM dance. |
| `python main.py mock-enrich` | *Not yet exposed* — same reasoning. |
| `python main.py record-feedback` | *Not yet exposed* — feedback is fundamentally a human-in-the-loop step. |

Adding more tools is straightforward: copy the pattern in
`mcp_server/server.py`, decorate the function with `@mcp.tool()`, write a
docstring aimed at the agent, and add a smoke test under
`tests/test_mcp_server.py`.

---

## Tests

```bash
poetry run pytest tests/test_mcp_server.py -v
```

The smoke tests confirm:

- The server has the right name and registers all 6 tools.
- The resource template (`sdp://example/{name}`) is registered.
- Each tool returns the expected shape on a known-good input.
- The 10k row cap is enforced even when an agent passes a much larger value.
- Unknown formats / missing files surface as `{"ok": false}` rather than
  raising.

End-to-end MCP transport testing (round-tripping JSON-RPC over stdio with a
real client) is left to manual verification — wire the server into Claude
Desktop, ask it to call a tool, confirm the result.

---

## Where MCP fits in the platform's roadmap

MCP is the agent-driven counterpart to the Streamlit UI:

| Audience | Surface |
|---|---|
| Hands-on user | CLI |
| SME / non-technical user | Streamlit UI |
| AI-mediated user (via Claude / Cursor / etc.) | **MCP server** |

All three surfaces call the same underlying functions in `generators/`,
`mocks/`, and `ml/`. Adding a feature once makes it available everywhere.

That's the architectural payoff for keeping the data-track and mocks-track
logic as pure Python with clean function boundaries: the same code drives a
CLI command, a Streamlit page, and an MCP tool — no duplication.
