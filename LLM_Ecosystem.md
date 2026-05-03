# The LLM Ecosystem — A Practitioner's Map

A curated reference for picking, hosting, and orchestrating large language
models. Aimed at engineers integrating LLMs into a product or back-office
pipeline (which is what this project does), not at researchers training
foundation models from scratch.

> Vintage marker: these tables reflect the public landscape as of early-to-mid
> 2026. Model release cadences are aggressive — always confirm the latest
> version on the vendor's site before depending on a specific model name.

---

## 1. The four layers

```
┌───────────────────────────────────────────────────────────────────────┐
│ 4. Application orchestration (LangChain, LlamaIndex, Haystack, …)     │
├───────────────────────────────────────────────────────────────────────┤
│ 3. Provider abstraction / routing (LiteLLM, this project's            │
│    multi_provider.py, OpenAI-Compat layer in vLLM/Ollama/LM Studio)   │
├───────────────────────────────────────────────────────────────────────┤
│ 2. Inference runtimes (Ollama, LM Studio, vLLM, llama.cpp, TGI,       │
│    SGLang, Triton)                                                    │
├───────────────────────────────────────────────────────────────────────┤
│ 1. Model weights (Claude, GPT, Gemini, Llama, Qwen, Mistral, …)       │
└───────────────────────────────────────────────────────────────────────┘
```

The trick is to pick one item per layer that suits your latency, cost,
privacy, and capability requirements. They are mostly orthogonal.

---

## 2. Hosted providers (closed-weight)

| Provider | Flagship models (early 2026) | Strengths | Trade-offs |
|---|---|---|---|
| **Anthropic** | Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 | Long context (200K+), strong reasoning, prompt caching, agentic / tool use | Closed weights, US-only data residency by default |
| **OpenAI** | GPT-4.1, GPT-4o, o3 / o4-mini (reasoning), GPT-Image | Mature API, multi-modal, function calling, structured outputs, batch API | Pricing complexity, occasional rate-limit cliffs |
| **Google DeepMind** | Gemini 2.5 Pro / Flash / Nano | 1M+ token context, native multi-modal, cheapest per-token at the high end | Region-gated features, more variable JSON adherence |
| **AWS Bedrock** | Multi-vendor (Anthropic, Meta, Mistral, Cohere, Amazon Titan) | Single bill, IAM, data residency, private-link | Adds an AWS abstraction tax; not all features expose through Bedrock immediately |
| **Azure OpenAI** | OpenAI models in Azure tenants | Compliance, VPC isolation, private deployments | Per-deployment quota, slower model rollouts than OpenAI proper |
| **xAI** | Grok-3, Grok-Vision | Real-time X data, fast | Smaller ecosystem, fewer enterprise controls |
| **Cohere** | Command R+, Embed v4, Rerank v3 | Excellent retrieval/RAG models, strong enterprise support | Smaller capability ceiling at the frontier |
| **Mistral** | Mistral Large 3, Codestral, Devstral, Nemo | EU data residency, strong open weights too | Less raw reasoning headroom than the top three |

### Hosted gateways (one key, many models)

| Gateway | What it is |
|---|---|
| **OpenRouter** | Single API key, single bill, ~100+ models routed (OpenAI, Anthropic, Mistral, Llama, Qwen, …). Useful for evaluation. |
| **Together AI** | Hosted open-weight models (Llama, Qwen, DeepSeek, Mixtral, Flux). Pay-per-token. |
| **Groq Cloud** | Custom LPU silicon — sub-100ms first-token latency on Llama-3.3-70B and similar. |
| **Fireworks AI** | Same model menu as Together; competitive pricing on big context. |
| **Perplexity API** | Llama / Mistral with web-search baked in (`sonar` family). |

---

## 3. Open-weight model families

You can run any of these locally, or call them through Together / Fireworks /
Groq / Bedrock. Sizes span ~1B (laptop-friendly) to ~700B (multi-GPU rigs).

| Family | Author | Notable releases | Notes |
|---|---|---|---|
| **Llama** | Meta | Llama 3.3 70B, Llama 3.1 405B, Llama 3.2 (1B/3B/11B-vision/90B-vision) | The de-facto base model for most fine-tuners and the open-weight ecosystem |
| **Qwen** | Alibaba | Qwen 3 / Qwen 2.5 (0.5B → 72B), Qwen2.5-Coder (1.5B → 32B), Qwen-VL | Strong on multilingual + code; Apache-2.0 |
| **DeepSeek** | DeepSeek AI | DeepSeek-R1, DeepSeek-V3, DeepSeek-Coder | R1 is a frontier-class reasoning model with open weights |
| **Mistral** | Mistral AI | Mistral 7B / 8x7B / 8x22B (Mixtral), Mistral-Nemo, Codestral | Apache-2.0; mixture-of-experts variants are inference-efficient |
| **Gemma** | Google | Gemma 2 (2B/9B/27B), Gemma 3 | Strong small-model performance; permissive licence |
| **Phi** | Microsoft | Phi-4 (14B), Phi-3.5-mini / vision / MoE | Excellent quality at small parameter counts |
| **Yi** | 01.AI | Yi-1.5 (6B/9B/34B), Yi-Coder | Bilingual EN/ZH, friendly licence |
| **Command-R** | Cohere | Command R+, R7B | Open weights for non-commercial use; RAG-tuned |
| **Falcon** | TII | Falcon 180B, Falcon-Mamba | Apache-2.0 historically |
| **OLMo** | AI2 | OLMo-2 (7B/13B) | Fully open: weights, training data, training code — the gold standard for reproducible research |
| **SmolLM** | Hugging Face | SmolLM-2 (135M/360M/1.7B) | Tiny — fits in mobile / browser via WebLLM |

### Choosing a size

| Hardware | Reasonable model |
|---|---|
| Laptop CPU only | Phi-3.5-mini, Qwen 2.5 1.5B/3B, Llama 3.2 1B/3B (all 4-bit GGUF) |
| 8GB GPU | 7B–8B class (Llama 3.1 8B, Qwen 2.5 7B, Gemma 2 9B) at q4 |
| 16–24GB GPU | 13B–14B class, or 32B at q4 |
| 48GB GPU (A6000) | 70B class at q4, 32B class unquantised |
| 2× 80GB GPU (A100/H100) | 70B class at fp16, 405B at heavy quant |

### Quantisation in one paragraph

GGUF (CPU / Apple Silicon / mixed) and GPTQ / AWQ / EXL2 (GPU) compress
weights from fp16 down to 8-, 6-, 5-, 4-, 3-bit. q4_K_M is the most-used
sweet spot on llama.cpp / Ollama — typically ≥97% of full-precision quality
at 25% of the VRAM. Below q3, quality drops fast.

---

## 4. Local inference runtimes

If you want to run a model on your own hardware — for privacy, cost, or
offline use — pick **one** of these. Each exposes an HTTP server; most speak
the OpenAI chat-completions wire format, which is why this project's
`multi_provider` defaults work against all of them.

| Runtime | Sweet spot | API surface | Notes |
|---|---|---|---|
| **Ollama** | Easy default. CLI + REST. macOS / Linux / Windows. | OpenAI-compat at `http://localhost:11434/v1` and a native API at `:11434/api`. | Model library is curated (`ollama pull llama3.2`). Great for laptops. |
| **LM Studio** | GUI-driven. Browse models on Hugging Face, click to download, hit "Start Server". | OpenAI-compat at `http://localhost:1234/v1`. | Best UX for non-CLI users. Supports MLX on Apple Silicon for fast Apple-native inference. |
| **llama.cpp** | The C/C++ engine under everything else. | `llama-server` ships an OpenAI-compat endpoint. | Embeddable; runs anywhere C compiles, including phones. |
| **vLLM** | Production GPU serving. PagedAttention, continuous batching. | OpenAI-compat. | Best throughput per GPU. The default for self-hosting at scale. |
| **TGI** (Text Generation Inference) | Hugging Face's GPU server, similar niche to vLLM. | OpenAI-compat (recent versions). | Tightly integrated with Hugging Face Inference Endpoints. |
| **SGLang** | High-throughput, structured-generation focus. | OpenAI-compat. | Strong for constrained decoding (regex / JSON / grammar). |
| **MLX-LM** | Apple Silicon native. | Python API; OpenAI-compat via wrappers. | Best perf on M-series Macs. |
| **Triton Inference Server** | NVIDIA's enterprise serving stack. | gRPC / HTTP. | More general than just LLMs; common in big shops. |
| **WebLLM / transformers.js** | In-browser via WebGPU / WASM. | JS only. | Tiny models only (1–3B), but no server cost at all. |
| **NIM** | NVIDIA Inference Microservices. | OpenAI-compat. | Pre-packaged containers for Llama, Mistral, etc., for k8s deployment. |
| **Mistral.rs** / **candle** | Rust-native LLM serving. | OpenAI-compat. | Lean alternative to llama.cpp for Rust shops. |

### LM Studio specifically (relevant to this project)

LM Studio runs a local OpenAI-compatible server. Once a model is loaded and
the server is started, configure this project to use it via env vars:

```bash
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"
# Default base URL is http://localhost:1234/v1 — no need to set unless changed
```

Then any of the LLM-using commands work:

```bash
python main.py infer-relationships --config bare.yaml \
    --config-output suggested.yaml --method llm

python main.py enrich --config bare.xlsx --output enriched.yaml

python examples/llm_quickstart.py
```

The same is true for Ollama with `SDP_LLM_PROVIDER=ollama` and
`SDP_LLM_MODEL=llama3.2`.

---

## 5. Provider abstractions / routers

| Tool | What it does | When to use |
|---|---|---|
| **LiteLLM** | Drop-in OpenAI-Python replacement; routes to 100+ providers via one call. Self-hosted proxy mode for org-wide routing + spend tracking. | When you need vendor portability without writing the abstraction yourself |
| **OpenRouter** (also a hosted provider) | Cloud router. One API key, models on demand. | Eval and prototyping when you don't want to manage credentials |
| **Portkey** | Gateway with caching, fallbacks, observability. | Mid-to-large orgs running LLM features in production |
| **PromptLayer** | Logging + prompt versioning sidecar. | Teams iterating heavily on prompts |
| **`llm` CLI by Simon Willison** | Local CLI tool that abstracts providers, with plugins for almost every model. | Power-user shell workflows |
| **This project's `llm/multi_provider.py`** | Tiny in-tree abstraction (Anthropic + OpenAI-compat HTTP). | When you want zero extra deps and full control |

---

## 6. Application orchestration frameworks

These sit *on top of* the provider abstraction and add memory, RAG, tool
loops, agents, and graph-of-thought patterns.

| Framework | Language | Strengths | Trade-offs |
|---|---|---|---|
| **LangChain** | Py / JS | Largest ecosystem, biggest tutorial corpus, plugins for every vector DB and tool | API churn; can feel heavy for simple flows |
| **LangGraph** | Py / JS | LangChain's stateful-graph DSL — explicit nodes/edges for agents, durable checkpoints | Newer; concepts take some ramp-up |
| **LlamaIndex** | Py / JS | RAG-first design, ingestion pipelines, knowledge-graph-aware retrieval | Naming overlaps with the unrelated Llama model family |
| **Haystack** | Py | German-engineered RAG framework, strong production focus | Smaller community than LangChain |
| **DSPy** | Py | Declarative + automatic prompt optimisation; treats prompts like trainable code | Learning curve; rewards thinking in pipelines |
| **Semantic Kernel** | C# / Py / Java | Microsoft's orchestration; first-class on .NET | Most polished outside Python territory |
| **PydanticAI** | Py | Typed agents on Pydantic models; structured outputs by construction | Newer, smaller |
| **AutoGen** | Py | Multi-agent conversation patterns | Best when the problem really is multi-agent |
| **CrewAI** | Py | "Crews" of specialised agents; declarative roles | Opinionated structure |
| **Vercel AI SDK** | TS / JS | Streaming UI primitives for React/Next/Vue | Frontend-focused |
| **Mastra** | TS | TS-native agents + workflows + evals | Newest of the bunch |

For this project we deliberately stay below the orchestration layer: the
relationship inferrer and schema enricher are single-shot prompts with JSON
output, so a framework would be more weight than benefit. If the project ever
grows agent-style behaviour (e.g. iterative schema refinement), LangGraph or
PydanticAI would slot in cleanly above `multi_provider`.

---

## 7. RAG building blocks (when you need retrieval)

| Layer | Common picks |
|---|---|
| Vector DB | Chroma, Qdrant, Weaviate, Milvus, pgvector, LanceDB, Vespa, Pinecone (hosted) |
| Embeddings | OpenAI `text-embedding-3-large`, Cohere Embed v4, Voyage 3, Nomic, BGE, GTE, E5 |
| Rerankers | Cohere Rerank v3, BGE-reranker, Voyage-rerank |
| Document parsing | Unstructured.io, Docling, MinerU, LlamaParse, marker |
| Chunking | LangChain text splitters, LlamaIndex Node parsers, Chonkie |

This project doesn't currently do RAG — schema-inference prompts are
self-contained and fit comfortably in context. Stubs/mocks generation is the
likely first place this changes.

---

## 8. Evaluation, safety, ops

| Need | Tooling |
|---|---|
| Eval / regression | Promptfoo, OpenAI Evals, Inspect (UK AISI), Ragas (RAG-specific), DeepEval, Braintrust, LangSmith |
| Tracing / observability | Langfuse, LangSmith, Arize Phoenix, OpenLLMetry, Helicone |
| Guardrails | Guardrails AI, NeMo Guardrails, LLM Guard, Pydantic validators on structured output |
| Red-teaming | Garak, PyRIT, Promptfoo's adversarial mode |
| Cost tracking | LiteLLM proxy spend logs, Helicone, Portkey |

---

## 9. How this project plugs in

| Layer | This project's choice |
|---|---|
| Model | Default: Anthropic Claude Sonnet 4.6 (via `llm/client.py:DEFAULT_MODEL`). Override per call. |
| Runtime / endpoint | Hosted Anthropic by default; LM Studio / Ollama / vLLM / Azure OpenAI / Groq / Together / OpenRouter via `SDP_LLM_PROVIDER` env var |
| Provider abstraction | In-tree `llm/multi_provider.py` — single `chat()` function; uses `requests` for OpenAI-compat HTTP, Anthropic SDK for Claude |
| Application logic | Plain Python in `llm/relationship_inferrer.py`, `llm/schema_enricher.py`, and `ml/relationship_inferrer.py` (the heuristic alternative) |

### Switching providers in one line

```bash
# Free local LLM via LM Studio
export SDP_LLM_PROVIDER=lm-studio
export SDP_LLM_MODEL="meta-llama-3.1-8b-instruct"

# Free local LLM via Ollama
export SDP_LLM_PROVIDER=ollama
export SDP_LLM_MODEL="llama3.2"

# Hosted OpenAI
export SDP_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Groq (fastest hosted Llama)
export SDP_LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_...

# Custom OpenAI-compatible server (vLLM, llama.cpp server, anywhere else)
export SDP_LLM_PROVIDER=lm-studio    # any openai-compat profile works
export SDP_LLM_BASE_URL=http://my-gpu-box.local:8000/v1
export SDP_LLM_MODEL=qwen2.5-coder-32b-instruct
```

See `examples/llm_quickstart.py` for a runnable demo that prints whatever
provider you've configured.

---

## 10. Pointers for further reading

- **Hugging Face Hub** — the model registry the open-weight world revolves
  around. <https://huggingface.co/models>
- **Artificial Analysis** — independent benchmark + price-per-token tracker.
  Useful for picking models without vendor bias.
- **lmsys Chatbot Arena** — head-to-head human voting; the closest thing to a
  consensus quality leaderboard.
- **Simon Willison's blog (`simonwillison.net`)** — running commentary on the
  practical end of LLMs; especially good for the local + CLI tooling layer.
- **Latent Space, Dwarkesh Podcast, AI Engineer Summit talks** — for the
  "what's about to change" view.
