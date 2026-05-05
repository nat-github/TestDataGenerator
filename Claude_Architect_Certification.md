# Claude Certified Architect — Self-Contained Study Guide

A 5-day study companion. Designed so you can sit with this single document
and walk into the exam ready.

> **One thing to verify outside this guide:** Anthropic's official
> certification page itself, to confirm the *current* syllabus, format,
> exam-day rules, and prerequisites. Spend 30 minutes there before you
> start studying. Everything else you need is below.

---

## 0. How to use this guide

- **Day 1–5:** follow the daily plan. Each day is ~3 hours of focused work
  plus exercises.
- **Day 6 (exam morning):** run the cheat sheet (§9) then the architecture
  problems in §7 out loud.
- **Practice with your hands** — every section has API examples you can
  paste into a Python REPL. Reading is not enough. Build.
- **Skip nothing.** The "boring" sections (caching, evaluation, safety) are
  what separate prompt engineers from architects. They're disproportionately
  represented on architect exams.

---

## 1. What "architect-level" means

You're being tested on:

| Competency | What that looks like |
|---|---|
| **Solution design** | Choosing the right Claude features for a given problem |
| **Cost / latency / quality trade-offs** | Model selection, caching strategy, batching |
| **Production patterns** | Deployment, evaluation, safety, observability |
| **Integration architecture** | Agents, RAG, tools, MCP, multi-modal |
| **Failure mode analysis** | How things break, how to detect, how to mitigate |

Pure prompt-engineering tricks are *foundational* — assumed knowledge, not
the bar. The bar is "given this messy real-world problem, defend your
choices."

---

## 2. Day-by-day plan

Calibrated to ~3 focused hours/day. Today is the start of your study week.

### Day 1 — Models, API surface, prompt fundamentals

- §3 (Model family) + §4 (API surface) + §5 (Prompt engineering)
- **Exercises:** §3-Ex, §5-Ex
- Build a Python script that classifies 5 emails into intent + urgency using
  Sonnet, with at least one few-shot example

### Day 2 — Caching, batching, cost engineering

- §6 (Cost engineering)
- **Exercises:** §6-Ex
- Compute the cost of 1M tickets through Sonnet under three caching
  scenarios — show your working

### Day 3 — Tools, agents, MCP, computer use

- §7 (Tool use) + §8 (Agents) + §9 (MCP) + §10 (Computer use)
- **Exercises:** §7-Ex, §8-Ex
- Wire up a basic MCP server that exposes a calculator tool. Test it
  against Claude Desktop or Claude Code

### Day 4 — Architecture patterns: RAG, multi-agent, structured extraction

- §11 (RAG) + §12 (Multi-agent) + §13 (Structured extraction) + §14 (Vision)
- **Exercises:** §11-Ex, §12-Ex
- Walk through the §17 architecture problems out loud

### Day 5 — Safety, evaluation, observability + mock exam

- §15 (Safety) + §16 (Evaluation) + §18 (Mock exam)
- **Exercises:** §16-Ex, §18 (full mock)
- Final pass over the §19 cheat sheet

### Day 6 — Exam morning

- 30 min: cheat sheet recall
- 30 min: architecture problems out loud
- 60 min: light review of weakest area
- Then go take the exam

---

## 3. The Claude model family

Memorise this table cold. Architect questions often start with "given this
constraint, which model would you choose."

| Model | Model ID | Strengths | Pick when |
|---|---|---|---|
| **Claude Opus 4.7** | `claude-opus-4-7` | Best reasoning, hardest problems, multi-step agents | Architecture decisions, complex code generation, long agent loops, ambiguous tasks |
| **Claude Sonnet 4.6** | `claude-sonnet-4-6` | Balanced cost/quality | Default for most production. Tool use, RAG, structured extraction |
| **Claude Haiku 4.5** | `claude-haiku-4-5-20251001` | Fast, cheap, narrow tasks | Classification, routing, summarisation, high-throughput pipelines |

**Architect rule:** route by task complexity. Opus for the hard turn,
Sonnet for everything else, Haiku for parallelisable narrow work.
*Never default to the biggest model.*

### Cascading model pattern (memorise)

```
                    user query
                        │
                        ▼
                  Haiku classifier
                        │
              ┌─────────┼─────────┐
       simple │   medium│         │ complex
              ▼         ▼         ▼
            Haiku     Sonnet     Opus
```

Saves 60-90% cost vs always using Opus, with ≤5% quality drop on most
workloads.

### §3-Ex Exercise

Without looking, list:
1. The three model IDs above.
2. One scenario for each where it's the *correct* choice.
3. One scenario for each where it's the *wrong* choice.

---

## 4. The API surface — must know cold

```python
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are an expert assistant.",          # system prompt
    messages=[                                       # conversation
        {"role": "user", "content": "Hello"},
    ],
    tools=[...],                                     # tool definitions
    tool_choice={"type": "auto"},                    # tool routing
    stop_sequences=["</done>"],                      # early termination
    temperature=0.0,                                 # determinism knob
    stream=True,                                     # SSE streaming
)
```

### Things that show up in scenario questions

- **Roles**: `user`, `assistant`. **No `system` role inside messages** —
  system goes in its own param.
- **Content types**: `text`, `image`, `tool_use`, `tool_result`,
  `document`.
- **Stop reasons**: `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`.
  An architect must know these — they drive control flow in agents.
- **Token limits**: 200K context window. Output max varies per model.
  Extended thinking expands reasoning budget separately.
- **Streaming**: SSE; partial JSON works for tool use too. Use `stream=True`
  + `with client.messages.stream(...)` context manager.
- **Async**: `AsyncAnthropic` for asyncio code paths.

### Authentication

- Env var: `ANTHROPIC_API_KEY`
- Bedrock: `AnthropicBedrock` client; uses AWS credentials
- Vertex AI: `AnthropicVertex` client; uses GCP credentials
- All three SDKs share the same `messages.create()` shape

### Cost basics

- Pricing is **per million input tokens** + **per million output tokens**
- Cache hits cost ~10% of fresh input tokens
- Cache writes cost ~1.25× fresh input tokens
- Batch API gives **50% off both** input and output

---

## 5. Prompt engineering fundamentals

### The five tactics that matter most

1. **Be specific.** "Summarise" → bad. "Write a 3-sentence summary covering
   decision, owners, and deadline" → good.
2. **Use XML tags** to delimit input vs instructions:
   `<context>...</context><question>...</question>`. Claude is heavily
   trained to respect them.
3. **Few-shot examples** in `<examples>` blocks dramatically improve
   consistency on niche formats.
4. **Chain of thought** — for hard tasks, ask Claude to reason
   step-by-step *inside* `<thinking>` tags before giving the final answer.
5. **Prefill the assistant turn** to force a specific output start: pass
   an opening `{"role": "assistant", "content": "{"}` to coerce JSON.

### Example: structured extraction with prefill

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    system="Extract entities and return JSON only. No prose.",
    messages=[
        {"role": "user", "content": "Ada Lovelace, ada@ex.com, +44 7700"},
        {"role": "assistant", "content": "{"},   # prefill forces JSON start
    ],
)
# response.content[0].text → '"name": "Ada Lovelace", "email": "...", ...}'
# Prepend "{" before parsing.
```

### Structured outputs — three approaches

| Approach | When | Trade-off |
|---|---|---|
| **Tool use with one tool** | Strict JSON schema required | Most reliable; Claude validates |
| **System-prompt JSON instruction + prefill** | Lightweight, no tool overhead | Less reliable on edge cases |
| **Extended thinking + JSON** | Hard reasoning *and* JSON | Slower; reasoning in thinking blocks |

### Extended thinking

```python
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=16384,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[...],
)
```

- Use it when reasoning quality matters more than latency
- Budget capped per call; thinking tokens are billed
- The reasoning happens in `thinking` content blocks before the final
  `text` block

### System prompts that earn their keep

A good system prompt is:

- ≥1024 tokens — eligible for prompt caching on Sonnet/Opus
  (Haiku has lower thresholds; check current docs)
- Stable across calls — changes invalidate the cache
- Contains: role, capabilities, format rules, examples, safety guardrails

### §5-Ex Exercise

1. Take a free-text customer email and design a Claude call that returns
   `{intent, urgency, entities[]}` reliably. Use tool use.
2. Write a system prompt for a code-review assistant that's at least 1024
   tokens. Test it with `cache_control`.
3. Demonstrate the difference in output between asking for JSON vs
   prefilling `{` as the assistant turn.

---

## 6. Caching, batching, and cost engineering

This is where architects earn their salary. Every architect-level scenario
can be improved with one of the levers in this section.

### Prompt caching — the single biggest cost lever

```python
system=[
    {
        "type": "text",
        "text": LONG_STABLE_SYSTEM_PROMPT,   # ≥1024 tokens for Sonnet/Opus
        "cache_control": {"type": "ephemeral"},
    }
]
```

**Numbers to memorise:**

- **TTL**: ~5 minutes (ephemeral). Refresh by re-sending the cached block.
- **Hit cost**: ~10% of fresh input tokens
- **Write cost**: ~1.25× fresh input tokens
- **Break-even**: 2 hits per write (cache loses on a single-shot call)
- **Breakpoints**: up to 4 per request — strategy: cache system prompt,
  tools, large documents, conversation history *separately*
- **Per-key**: cache is scoped to your API key, not shared across orgs

**Architect rule:** if your system prompt is stable and ≥1024 tokens,
*always* cache it. Don't think about it.

### Caching strategy for a multi-turn agent

A typical agent has four caching opportunities at different cadences:

```
breakpoint 1 → system prompt    (changes rarely — cache always)
breakpoint 2 → tool definitions (changes on tool updates — cache always)
breakpoint 3 → static context   (RAG corpus, knowledge base — cache when stable)
breakpoint 4 → conversation     (grows over turns — cache after ≥2 turns)
```

### Batch API

Async, **50% discount** off both input and output tokens.

- Submit up to 100K requests per batch
- Results within 24 hours (usually faster)
- Polling endpoint for status
- **When to use:** offline pipelines, evals, large-scale data labelling,
  document processing
- **When NOT to use:** anything user-facing — async by definition

### Cost optimisation playbook (memorise this list)

1. **Right-size the model** — Haiku/Sonnet/Opus per task
2. **Cache the stable bits** — system prompt, tools, RAG context
3. **Batch the offline work** — anything that doesn't need a live response
4. **Prompt-compress the variable bits** — drop redundant context,
   summarise long history
5. **Stream when possible** — perceived latency matters for UX
6. **Use `stop_sequences`** to terminate early when you have what you need

### §6-Ex Exercise

A SaaS product runs **1M support tickets/month** through Sonnet. Each
ticket: 3K-token system prompt + 500-token user query + 300-token
response. Calculate the cost under:

- (a) **No caching, no batch.** Live serving.
- (b) **System prompt cached.** Live serving, ~5K cache hits per breakpoint.
- (c) **Cached + batch API.** Offline pipeline.

(Use approximate Sonnet pricing: $3/M input, $15/M output. Cache hit ~$0.30/M
on input. Verify against current pricing on exam day — these change.)

**Ballpark answer:**
- (a) ~$13.5K/month
- (b) ~$5.5K/month (system prompt is ~85% of input tokens; cache hits
  reduce that to 10%)
- (c) ~$2.7K/month (further 50% off batch)

The point isn't the exact number — it's the multiplicative gain when
levers stack.

---

## 7. Tool use

### The API shape

```python
tools=[{
    "name": "get_weather",
    "description": "Returns current weather for a city. Use this when the user asks about weather conditions, temperature, or forecasts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}]
```

### Tool descriptions matter more than you think

The description is how Claude decides whether to call the tool. Bad
descriptions cause silent under-utilisation. Architect-level practice:

- Lead with **what the tool does** in one sentence
- Add **when to use it** explicitly
- Document **what it returns** (helps Claude reason about results)
- Warn about **side effects** ("This sends an email — only call after
  user confirmation")

### `tool_choice` modes

| Mode | Behaviour | Use case |
|---|---|---|
| `{"type": "auto"}` | Claude decides whether to call any tool | Default — most agents |
| `{"type": "any"}` | Claude must call *some* tool | "Always do something" workflows |
| `{"type": "tool", "name": "X"}` | Must call this specific tool | Forced extraction, structured outputs |
| `{"type": "none"}` | No tool calls | Pure-text response |

### Parallel tool use

Claude can request multiple tools in one turn. Execute them concurrently
and return all results in one user message:

```python
# response.content might contain:
# [tool_use(id="1", name="get_weather", input={"city": "Paris"}),
#  tool_use(id="2", name="get_weather", input={"city": "Tokyo"})]

# Reply with both tool_results in a single user message:
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "1", "content": "..."},
    {"type": "tool_result", "tool_use_id": "2", "content": "..."},
]}
```

### §7-Ex Exercise

Write a JSON Schema for a tool `book_flight(from, to, depart_date,
return_date?, passengers, class)`. Make `class` an enum. Add proper
constraints to dates. Write the description such that Claude knows when
*not* to call it.

---

## 8. Agents

### The agent loop

```
1. Send user message + system prompt + tools to Claude
2. While stop_reason == "tool_use":
     execute every tool_use block (in parallel where safe)
     append a user message with corresponding tool_result blocks
     re-send the conversation
3. Final assistant message is the answer
```

**Architect concerns** (every one is a likely exam question):

- **State**: where does conversation history live? In-memory, Redis,
  a DB? Survives restarts?
- **Cost bound**: max iterations per session, max tokens per iteration
- **Time bound**: per-tool timeouts, per-session timeouts
- **Observability**: log every tool call, every token-usage record, every
  latency
- **Recovery**: tool failures → return error to Claude as `tool_result`,
  let it retry or escalate. Model errors → retry with backoff.
- **Safety**: bound the loop so it can't run away. Require human
  confirmation for irreversible tools (writes, sends, payments).

### Multi-agent patterns

| Pattern | Architecture |
|---|---|
| **Single agent** | One Claude instance + tools. Default. |
| **Planner + executor** | Strong model plans, weaker model executes each step. Cost-efficient. |
| **Specialist team** | Domain agents (security, code, ops) coordinated by an orchestrator |
| **Reflection loop** | Agent's output is graded by a second agent; reflect → retry on low score |

### When to NOT use a multi-agent system

- The task is single-step
- The team-coordination cost exceeds the quality gain
- You can't bound the total cost or runtime
- You don't have evals to prove it's better than a single agent

### §8-Ex Exercise

Design an agent for "given a GitHub PR URL, summarise the changes."

- List the tools it needs
- Set the loop bound and tokens budget
- Write the system prompt outline
- List two failure modes and how you detect each

---

## 9. MCP — Model Context Protocol

### What it is

An open standard from Anthropic (late 2024) for connecting AI assistants
to external systems through a uniform JSON-RPC interface. Now adopted by
OpenAI, Google, Cursor, Zed, Claude Desktop, Claude Code, and many others.

### Three primitives

| Primitive | Purpose |
|---|---|
| **Tools** | Callable functions (name + JSON-Schema typed args) |
| **Resources** | Read-only data handles (URIs) |
| **Prompts** | Reusable prompt templates |

### Transports

| Transport | When |
|---|---|
| **stdio** | Default. Local servers launched as subprocesses. |
| **HTTP/SSE** | Remote servers, multi-tenant scenarios. |

### Server vs. client roles

- **Server** = exposes capabilities (your platform integration)
- **Client** = consumes capabilities (Claude Desktop, Cursor, etc.)

A single client can talk to many MCP servers simultaneously.

### Architect-level decisions about MCP

- **Reuse vs. build**: prefer existing servers (filesystem, github,
  postgres, slack) before writing your own.
- **Tools or resources?** Tools for actions / state changes. Resources
  for read-only data. Don't smuggle reads into tools.
- **Configuration**: server config in client config files
  (`claude_desktop_config.json`, `~/.claude.json`).
- **Security**: stdio servers have full access to the filesystem they're
  launched on. Sandbox accordingly.

### Example: the FastMCP shape

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-platform")

@mcp.tool()
def search_docs(query: str) -> dict:
    """Search the docs corpus for `query`. Use when the user asks about
    project documentation."""
    ...

@mcp.resource("docs://{path}")
def read_doc(path: str) -> str:
    """Read a doc by path."""
    ...

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 10. Computer use

### What it is

Lets Claude take screenshots and emit mouse/keyboard actions to drive
arbitrary GUI applications.

### When to use it

- Legacy app automation (no API exists)
- GUI testing
- Cross-app workflows that span tools without bridges

### When NOT to use it

- An API exists — always prefer the API
- Latency-sensitive flows
- High-volume workloads — slow and expensive

### Architectural caveats

- **Sandboxed deployment is essential** — never give it root on a real
  machine. Run inside a VM or container.
- **Brittle** — UI changes break flows. Add visual checks.
- **Observable** — log every screenshot + action.
- **Bounded** — strict iteration cap.

---

## 11. RAG with Claude

### The canonical pattern

```
query → embed → vector search → top-k chunks → cached system prompt
                                                      +
                                              <retrieved>...</retrieved>
                                                      +
                                              user question
                                                      ↓
                                                   Claude
                                                      ↓
                                                cited answer
```

### Key knobs

| Knob | Default | When to change |
|---|---|---|
| Chunk size | 500-1000 tokens | Smaller for fact-dense docs, larger for narrative |
| Overlap | 50-100 tokens | Larger if context spans chunks |
| Top-k | 5-10 | More for noisy corpus, fewer for tight one |
| Reranker | Cohere Rerank, BGE, Voyage | When top-k recall is fine but precision is poor |
| Citations | Inline `[doc_id:chunk]` markers | When provenance matters |

### Files API + Citations

For document-grounded Q&A, prefer the Files API + Citations:

```python
# Upload once
file = client.files.create(file=open("manual.pdf", "rb"))

# Reference in a message — Claude returns cited spans automatically
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{
        "role": "user",
        "content": [
            {"type": "document", "source": {"type": "file", "file_id": file.id},
             "citations": {"enabled": True}},
            {"type": "text", "text": "What's the warranty period?"},
        ],
    }],
)
# Response includes citation spans pointing back at the document.
```

### Architect failure modes for RAG

| Symptom | Most likely cause |
|---|---|
| Hallucinated answers | Top-k missed the relevant chunk; check retrieval first |
| Confidently wrong | Chunks too small, lost context; or no rerank |
| Irrelevant citations | Embedding model mismatch with corpus type |
| High cost | Forgot to cache the static corpus context |
| Slow at p99 | Vector search is the bottleneck, not Claude |

### §11-Ex Exercise

Design RAG for "answer questions about our 200-page employee handbook."
Specify: embeddings model, vector store, chunk size, top-k, caching
strategy, evaluation approach.

---

## 12. Multi-agent architecture

### When multi-agent wins

- Task is genuinely decomposable (planner / executor / critic, etc.)
- Each role benefits from a different model size
- You can parallelise without coordination overhead
- You have evals to prove the multi-agent setup beats a single agent

### When multi-agent loses (common trap)

- Coordination overhead exceeds quality gain
- Total cost balloons (each agent pays its own caching, system prompt)
- Failure modes multiply — debugging gets harder
- You don't have evals → "more agents = better" is just faith

### Reference patterns

#### Plan-and-execute

```
user → Opus (plan) → list of steps
                        │
                ┌───────┼───────┐
                ▼       ▼       ▼
              Sonnet  Sonnet  Sonnet  (parallel execution)
                │       │       │
                └───────┼───────┘
                        ▼
                  Opus (synthesise)
                        ▼
                     answer
```

#### Reflexion / self-critique

```
user → Sonnet → draft answer
                    │
                    ▼
             Opus (critique)
                    │
                    ▼
        score ≥ 0.8 ? return : retry with feedback
```

### §12-Ex Exercise

Design a multi-agent system for "code review." Roles: linter, reasoner,
security. Specify: which model for each, how they compose, cost ceiling,
how you'd prove it's better than single-agent.

---

## 13. Structured extraction patterns

### Use case: turn unstructured text into typed records

Three approaches in order of robustness:

1. **Tool use with one tool** — most reliable. The tool's input schema
   is the contract.
2. **JSON-mode + prefill** — lighter-weight; less reliable on edge cases.
3. **Extended thinking + JSON** — for hard extraction where reasoning
   matters.

### Architect-level extraction concerns

- **Schema design**: tight types (enums, regex, ranges) catch hallucinations
- **Validation**: parse + schema-check after every call; never trust the
  model
- **Recovery**: on parse failure, retry with stricter prefill; escalate to
  Opus on second fail
- **Throughput**: extraction is embarrassingly parallel — use the Batch
  API for bulk loads

---

## 14. Vision (multi-modal inputs)

### Key facts

- Image input via `{"type": "image", "source": ...}` content blocks
- Source can be base64 or a URL
- Common use cases: OCR, screenshot analysis, chart interpretation,
  document understanding
- Token cost: roughly proportional to image dimensions; resize before
  sending if you don't need the detail

### When NOT to use vision

- The data is already structured upstream — text wins on cost & accuracy
- Volume is huge and per-image precision doesn't matter — train a
  cheaper specialised model

### Architect rule

For document Q&A, prefer **Files API + Citations** over pasting images of
each page. Files API is built for this and produces cleaner citations.

---

## 15. Safety architecture

Architect exams test safety heavily. These patterns are not optional.

### System-prompt safety

- Stable refusal language ("If asked about X, respond with: ...")
- Scope guardrails ("You only answer questions about our product")
- Explicit "do not" lists (forbidden topics, forbidden actions)
- Cache the safety prompt — it's stable and benefits hugely

### Input validation

- Filter user input *before* it reaches the model — basic prompt
  injection defence
- Strip / escape control sequences in user-supplied content
- Length-limit free-text inputs
- For tool-using agents, validate tool inputs against your schema

### Output validation

- Parse + schema-check structured outputs
- Never `eval` model output. Never. Even from Opus.
- For SQL, parse + check operation type before execution
- For commands, allow-list the binaries that can be invoked

### Rate limiting + abuse detection

- Per-user budgets (token / dollar / request count)
- Anomaly detection (sudden spikes, unusual patterns)
- Alerting on cost overruns

### Content moderation pattern

```
user input → Haiku classifier (refuse/allow) → Sonnet/Opus (handle)
```

Cheap upfront filter saves expensive downstream calls.

### Logging & audit

- Every prompt, every response, with PII scrubbing
- Retention policy aligned with compliance (GDPR, etc.)
- Searchable for incident response

### Agentic safety

- **Bounded loops**: hard cap on iterations
- **Sandboxed tools**: never give a model root on a production system
- **Human approval gates** for irreversible actions: writes, sends,
  payments, deletions
- **Principle of least privilege**: smallest tool set the agent needs

### Memorise this principle

> **Treat model output like user input.** Validate, sanitise, never
> blindly execute.

---

## 16. Evaluation

The architect's blind spot. The exam will test you on this disproportionately.

### Eval-driven development

Build the eval set *before* building the system. Without evals you can't
prove a change is an improvement.

### Eval components

| Layer | What it does |
|---|---|
| **Golden dataset** | 50-200 hand-labelled examples covering happy path + edge cases |
| **Automated grader** | Stronger model (Opus) grades a weaker model's output, returning a score + rationale |
| **Regression suite** | Every prompt change → re-run the eval before deploying |
| **Online eval** | Sample production interactions, route through the grader, alert on drift |
| **Human review** | Periodic spot-check by domain experts |

### Grader design

Two-shot graders work well:

```python
grader_prompt = """You are evaluating a customer-support response.
Score the response 1-5 on each of:
- accuracy (is the information correct?)
- relevance (does it address the question?)
- tone (is it appropriately empathetic?)

Respond with JSON: {accuracy, relevance, tone, rationale}.

<question>{question}</question>
<response>{response}</response>
<reference_answer>{golden_answer}</reference_answer>
"""
```

Use Opus for grading even if your production model is Sonnet — the grader
should be smarter than the system being graded.

### Eval tools to know by name

- **Anthropic's eval cookbook** patterns
- **Promptfoo** — open-source, YAML-based eval runner
- **LangSmith** — LangChain's tracing + eval tooling
- **Braintrust** — production-grade eval platform
- **Inspect** — UK AISI's Python eval framework

### §16-Ex Exercise

For your customer-support agent, design:
1. The structure of the golden dataset (what fields, ~10 example rows).
2. The grader prompt.
3. The CI integration: when does it run, what's the pass threshold, what
   happens on regression?

---

## 17. Architecture problems — practice walkthroughs

For each, time yourself: **15 minutes max** to talk through your answer
out loud. The model answer follows so you can self-grade.

### Problem 1: 100k-query support assistant

> "Design a customer-support assistant that handles 100k queries/day across
> 6 languages, with an SLA of 3 seconds. Walk me through your model
> choices, caching, fallbacks, and evaluation."

**Architect answer outline:**

- **Routing layer**: Haiku classifier identifies language + intent +
  escalation flag. ~50ms per call. Cached system prompt on the classifier.
- **Main answering layer**: Sonnet for ~85% of cases. RAG over the
  product knowledge base, cached corpus context, top-k=5.
- **Escalation layer**: Opus for `escalation_flag=true` (refunds, complex
  multi-product, regulatory questions). ~5% of volume.
- **Caching strategy**: classifier system prompt cached (~3K tokens),
  Sonnet system prompt cached (~5K tokens), tool definitions cached.
  Per-language RAG corpus cached separately.
- **SLA strategy**: stream responses; first-token latency target 800ms.
  Run classifier + RAG retrieval in parallel.
- **Fallbacks**: API failure → fall back to FAQ; rate limit → queue with
  retry; tool failure → return graceful error to Claude, let it ask the
  user to clarify.
- **Evaluation**: weekly grader run against 200-example golden set
  (per-language, per-intent matrix); production sampling 1% of traffic
  routed through the grader.
- **Cost ceiling**: per-conversation budget (e.g. $0.05); per-day
  org-wide budget alerts.

### Problem 2: 10M PDF entity extraction

> "A bank wants to extract entities from 10M scanned PDFs over the next
> quarter. Architect this."

**Architect answer outline:**

- **Pipeline shape**: offline batch — *not* live serving. This is the
  Batch API's home turf.
- **Model**: Sonnet for the bulk; Opus for low-confidence retries (~5%).
- **Vision approach**: Files API + Citations. Group multi-page docs into
  single Files; let Claude reason across pages within a file.
- **Schema**: tool-use with one extraction tool per document type;
  validate output against a strict JSON Schema.
- **Throughput**: batches of 10K requests; fits the quarter at ~110K/day.
- **Cost engineering**: batch (50% off) + cache the extraction tool
  schema + cache the per-doc-type system prompt.
- **Quality**: 1% sample reviewed by humans; auto-grader run on 5% to
  catch drift.
- **Failure handling**: low-confidence extractions retried with Opus
  + extended thinking; persistent failures land in a human-review queue.
- **Storage**: results in a structured DB, with Files-API IDs as
  references for audit.

### Problem 3: Multi-agent code review

> "You're building a multi-agent system for code review. Three roles:
> linter, reasoner, security. How do you compose them, bound their cost,
> and observe failures?"

**Architect answer outline:**

- **Composition**: parallel — all three see the same diff, produce
  independent reports. Synthesiser (Opus) merges into a unified review.
- **Models**:
  - Linter: Haiku (cheap, narrow rules)
  - Reasoner: Sonnet (architectural & logic feedback)
  - Security: Sonnet (CWE-aware prompt; possibly Opus on flagged code)
  - Synthesiser: Opus (writes the final review)
- **Bounds**: per-PR token cap (e.g. 50K input, 10K output combined).
  Per-role timeout 30s. Synthesiser only runs if at least 2 of 3 roles
  succeed.
- **Caching**: each role's system prompt + style guide cached separately.
- **Observability**: log per-role outputs, latencies, token counts. Log
  the synthesiser's view of each role's contribution.
- **Failure modes**: any role times out → synthesiser proceeds without
  it, notes the gap; all roles fail → fall back to "automated review
  unavailable" status.
- **Evaluation**: golden set of 50 PRs with known issues + a hand-graded
  ideal review. Weekly grader run; track precision/recall on detected
  issues.

### Problem 4: Migrating Sonnet 4.5 → 4.6 safely

> "How would you migrate a production app from Sonnet 4.5 to Sonnet 4.6
> safely?"

**Architect answer outline:**

1. **Run the existing eval suite** against 4.6 first. If your evals don't
   exist, build them now — migrations without evals are reckless.
2. **Compare scores** on the golden set. Look for regressions, especially
   on tool use and structured outputs (most common cause of subtle
   migration breakage).
3. **Shadow-deploy**: route 1% of production traffic to 4.6 in parallel
   with 4.5. Compare outputs offline.
4. **Canary**: 10% live traffic on 4.6 with monitoring on quality
   metrics, latency, cost.
5. **Watch token counts**: model upgrades sometimes change tokenisation
   behaviour. Cost can shift even at the same per-token price.
6. **Re-tune temperature & prompts** if needed — newer models may need
   different prompting.
7. **Roll forward** with a quick rollback path. Keep 4.5 wired up as
   fallback for 30 days.
8. **Communicate** to downstream consumers if outputs subtly change.

### Problem 5: Tool-using agent vs MCP

> "What's the difference between a tool-using agent and an MCP-based
> system, and when do you choose which?"

**Architect answer outline:**

- **Tool-using agent**: tools defined inline in your application code,
  inside `client.messages.create(tools=[...])`. Tools are private to your
  app.
- **MCP**: tools exposed by a separate process (the MCP server) over
  JSON-RPC. Tools are shareable across clients (Claude Desktop, Cursor,
  your custom app).

| Pick tool-use when | Pick MCP when |
|---|---|
| Tools are app-specific | Tools should be reusable across clients |
| You control both ends of the call | You want third-party agents (Cursor, Claude Desktop) to use your tools |
| Latency matters most | You want a clean process boundary |
| Single deployment | Multi-product surface |

You can do both: implement tools in MCP and *also* embed the same MCP
server's tools into your bespoke agent.

---

## 18. Mock exam questions

Time yourself: **75 seconds per multiple choice, 10 minutes per scenario.**
Answers and rationale at the end.

### Multiple choice

**Q1.** A user-facing chatbot's p99 latency is 4.5s. The system uses
Sonnet 4.6 with a 6K-token system prompt and no caching. Which single
change would have the largest latency impact?

A. Switch to Opus 4.7 with extended thinking
B. Add prompt caching to the system prompt
C. Switch to Haiku 4.5
D. Use the Batch API

**Q2.** You have 100K documents to classify into 12 categories overnight.
Which approach minimises cost?

A. Sonnet, live API, with caching
B. Opus, Batch API
C. Haiku, Batch API, cached system prompt + few-shot examples
D. Multi-agent: Haiku filter → Sonnet classifier

**Q3.** Your tool-using agent runs in a loop until `stop_reason ==
"end_turn"`. In production you observe runaway loops on rare inputs. The
*minimal* fix is:

A. Add a hard iteration cap and a token budget
B. Switch to MCP for the tools
C. Move to Opus with extended thinking
D. Add a reflexion agent to grade outputs

**Q4.** A RAG system grounded in a 1M-document corpus is hallucinating.
Top-k is 5; chunks are 1000 tokens; embeddings are
`text-embedding-3-large`. The first thing to check is:

A. Whether to switch from Sonnet to Opus
B. Whether the relevant chunks are actually in top-k for the failing
   queries
C. Whether to add extended thinking
D. Whether to increase chunk size

**Q5.** You're building an agent that can send emails. Which control
matters most architecturally?

A. Use Opus for the email-writing tool
B. Cache the email tool's schema
C. Require human approval before the send action executes
D. Run the agent in extended-thinking mode

**Q6.** Prompt caching has TTL ~5 minutes. Your traffic is bursty: 100
calls in 30 seconds, then idle for 20 minutes. Caching is:

A. A clear win — burst hits amortise the write cost
B. A clear loss — idle time wastes the cache
C. Break-even — depends on burst size
D. Wrong tool — use the Batch API instead

**Q7.** You're choosing between:
(a) Tool use with `tool_choice: any`
(b) Tool use with `tool_choice: tool` naming a specific tool

For coercing structured output of one specific shape, the better choice
is:

A. (a) — it allows flexibility
B. (b) — it forces the specific tool
C. Neither — use prefill
D. Both equivalent

**Q8.** A multi-agent system has a planner (Opus), three executors
(Sonnet, parallel), and a synthesiser (Opus). One executor times out.
The architect's *correct* default is:

A. Retry the failing executor twice, escalate to user on third fail
B. Synthesiser proceeds with the surviving two; result tagged "partial"
C. Cancel the whole pipeline and return an error
D. Have the synthesiser re-run the failing executor itself

### Scenario questions

**Q9.** A bank's compliance team wants to scan inbound emails for
suspicious activity (money laundering signals). Volume: 5M emails/day.
Latency: not user-facing, must complete within 24 hours. Quality:
false negatives are unacceptable; false positives are reviewed by
analysts.

Design the system. Cover: model choice(s), caching, batch vs live,
schema for the output, evaluation strategy, escalation path,
observability.

**Q10.** A startup wants Claude to autonomously triage GitHub issues:
label them, suggest assignees, and post a first response. They have
50 repos, 200 issues/day total. They want this running in production
in 2 weeks.

Architect this. Cover: tool design, agent loop bounds, MCP vs
in-process, evaluation, safety (write actions on a public repo!),
roll-out plan.

### Answers

**Q1: C.** Haiku is 3-5× faster. Caching helps cost but the latency win is
in input-token throughput, dwarfed by the model's own forward-pass time.
The biggest p99 lever is model size. (B is the second-best answer if
quality must stay identical.)

**Q2: C.** Haiku at 50% Batch discount with cached system prompt is the
minimum cost. Multi-agent (D) adds an extra Haiku filter call, increasing
total cost.

**Q3: A.** Iteration cap and token budget are the minimal fix for runaway
loops. Other options are improvements but don't address the immediate
safety issue.

**Q4: B.** Always check retrieval first before tweaking the model.
Hallucination in RAG is most often "the right chunk wasn't in top-k."

**Q5: C.** Human approval gates for irreversible actions are
architectural safety bedrock. The other options are quality/cost knobs;
they don't fix the safety hole.

**Q6: A.** Burst traffic is exactly when caching wins — many hits per
write within the TTL. The 20-minute idle period only loses you the *next*
window's first call.

**Q7: B.** `tool_choice: tool` named-and-required is the most reliable
way to coerce a specific schema. (Prefill works too but is less robust on
edge cases.)

**Q8: B.** Resilient degradation. Synthesiser proceeds with what it has,
tags the result as partial. Retries (A) increase total latency unbounded;
hard-fail (C) is too brittle for production; (D) creates ill-defined
responsibility boundaries.

**Q9 model answer:**

- **Pipeline**: offline batch (24h SLA → Batch API).
- **Model cascade**: Haiku first-pass classifier (low-cost negative
  filter, ~95% of mail). Sonnet on the ~5% flagged. Opus on the ~0.1%
  high-risk.
- **Schema**: tool-use with strict JSON: `{flagged, severity,
  signals[], confidence, recommended_action}`.
- **Caching**: each tier's system prompt cached. Compliance taxonomy
  (~5K tokens) cached.
- **Evaluation**: 500-email hand-graded golden set, monthly refresh.
  Recall is the primary metric (false negatives unacceptable).
  Auto-grader on 5% production sample.
- **Escalation path**: severity ≥ "high" → analyst queue with
  full context + Claude's rationale.
- **Observability**: per-tier latency, cost, hit-rate of each escalation
  level. Drift alerts on signal-distribution shifts.

**Q10 model answer:**

- **Tool design**: 4 tools — `label_issue(labels[])`,
  `suggest_assignees(usernames[])`, `post_comment(body)`,
  `query_repo_context(question)`.
- **Loop bounds**: max 5 iterations, max 20K tokens per issue, 60s
  timeout.
- **MCP vs in-process**: MCP. The startup wants this reusable across
  triggers (webhooks, manual invocation, future Slack bot). MCP server
  exposes the same tools to all surfaces.
- **Evaluation**: 50 historical issues with hand-labelled "what should
  happen." Run weekly + on every prompt change.
- **Safety (critical for write actions on a public repo)**:
  - `post_comment` requires confirmation step until trust is built —
    initial deploy posts as draft visible only to the bot account; human
    approves transition to public.
  - `suggest_assignees` posts a comment with @mentions but does not
    self-assign.
  - All actions logged with full context for audit.
  - Per-repo daily action budget; alert on spikes.
- **Roll-out**:
  - Week 1: shadow mode (compute decisions, log them, take no action).
  - Week 2 day 1-3: 10% of issues, comments only, no labels/assignees.
  - Week 2 day 4-7: 100% with all actions, daily review of outputs.

---

## 19. The cheat sheet (memorise this)

### Model selection
> **Haiku** for narrow & fast. **Sonnet** for default production.
> **Opus** for hardest reasoning. **Cascade** if you can.

### Cost controls
> **Cache** stable. **Batch** async. **Compress** variable.
> **Stream** live. **Stop** early.

### Caching numbers
> **≥1024 tokens.** **5-min TTL.** **~10% on hit.** **~1.25× on write.**
> **4 breakpoints.** **Break even at 2 hits.**

### Agent design
> **Bound** iterations. **Sandbox** tools. **Log** everything.
> **Approval** for irreversible. **Smallest** privilege set.

### RAG knobs
> Chunk **500-1000**. Overlap **50-100**. Top-k **5-10**.
> Cache the corpus. Cite spans.

### Safety
> **Filter** input. **Validate** output. **Rate-limit**.
> **Audit-log** with PII scrubbed. **Bound** agents.

### Evals
> Build set **first**. Hand-label **50-200**. Auto-grade with
> **stronger** model. Run on **every change**. **Sample** production.

### MCP
> **Tools** (functions) + **Resources** (read-only) + **Prompts**
> (templates). **stdio** default. **Reuse** before building.

### Stop reasons
> `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`.

### Tool-choice modes
> `auto`, `any`, `tool`, `none`.

### Architectural triage order for any scenario
> 1. **Right model**? 2. **Caching set up**? 3. **Batch eligible**?
> 4. **Bounded loops**? 5. **Evaluation suite**? 6. **Safety gates**?

---

## 20. Day-of exam checklist (60 minutes before)

- [ ] Recall the model lineup (Haiku/Sonnet/Opus) and their model IDs
- [ ] Recall the four `stop_reason` values
- [ ] Recall the four `tool_choice` modes
- [ ] Recall caching numbers (TTL, threshold, hit/write %)
- [ ] Recall the cost optimisation playbook (six items)
- [ ] Recall MCP's three primitives
- [ ] Recall the agent loop in plain English
- [ ] Recall the architectural triage order (six items)
- [ ] Eat. Hydrate. Don't cram. Trust the prep.

---

## 21. Way forward — after the exam

Whether you pass or not, the work doesn't stop:

- **Anthropic publishes new model cards & cookbooks frequently.** Subscribe
  to their RSS / dev newsletter.
- **MCP servers are a fast-moving ecosystem.** Watch the official server
  registry; new ones land weekly.
- **The eval cookbook is the most actionable Anthropic resource.** Read
  every example.
- **Ship something.** A pass certificate without a project is meaningless.
  Build one of the architecture problems from §17 end-to-end.

---

## 22. Final words

Architect-level certifications are about **judgment under constraints**, not
trivia recall. The questions you can't memorise — "what would you do given
this messy real-world problem" — are the ones the exam optimises for.

If you've worked through this guide, done the exercises with your hands
(not just read), and rehearsed the architecture problems out loud — you're
ready.

Don't second-guess. Trust the prep. Walk in calm.

Good luck.
