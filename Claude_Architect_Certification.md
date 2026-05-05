# Claude Certified Architect — Foundations (CCA-F)
## Self-Contained Study Guide

> **Source of truth:** this guide is built from Anthropic's official
> *Claude Certified Architect – Foundations Certification Exam Guide*
> (v0.1, 10 Feb 2025) and the *CCA-F FAQs*, both shipped to candidates.
> Every domain, task statement, scenario, and sample question below is
> drawn directly from those documents. Use this as your single reference.

---

## Table of contents

1. [Exam logistics](#1-exam-logistics)
2. [Target candidate & prerequisites](#2-target-candidate--prerequisites)
3. [Domains, weightings, scenarios](#3-domains-weightings-scenarios)
4. [Domain 1 — Agentic Architecture & Orchestration (27%)](#4-domain-1--agentic-architecture--orchestration-27)
5. [Domain 2 — Tool Design & MCP Integration (18%)](#5-domain-2--tool-design--mcp-integration-18)
6. [Domain 3 — Claude Code Configuration & Workflows (20%)](#6-domain-3--claude-code-configuration--workflows-20)
7. [Domain 4 — Prompt Engineering & Structured Output (20%)](#7-domain-4--prompt-engineering--structured-output-20)
8. [Domain 5 — Context Management & Reliability (15%)](#8-domain-5--context-management--reliability-15)
9. [Claude commands & configuration reference](#9-claude-commands--configuration-reference)
10. [12 sample exam questions with explanations](#10-12-sample-exam-questions-with-explanations)
11. [Preparation exercises](#11-preparation-exercises)
12. [In-scope and out-of-scope topics](#12-in-scope-and-out-of-scope-topics)
13. [Anti-patterns checklist](#13-anti-patterns-checklist-distractor-patterns)
14. [Cheat sheet](#14-cheat-sheet)
15. [5-day study plan](#15-5-day-study-plan)
16. [Day-of checklist](#16-day-of-checklist)
17. [Final words](#17-final-words)

---

## 1. Exam logistics

| | |
|---|---|
| **Level** | ~301 (intermediate practitioner) |
| **Format** | Multiple choice, one correct answer + three distractors per question |
| **Attempts** | **One attempt only** (per registration) |
| **Cost** | **$99** (some partners may have a promo code) |
| **Proctoring** | ProctorFree |
| **Scoring** | Scaled score 100–1,000 |
| **Pass mark** | **720** on the live exam (target **900+** on the practice exam to be confident) |
| **Penalty for guessing** | **None** — never leave a question unanswered |
| **Scenarios** | 6 total in the bank; **4 are presented at random** on your exam |
| **Badge** | LinkedIn-shareable CCA-F badge on pass |
| **Access** | https://anthropic.skilljar.com/claude-certified-architect-foundations-certification |

**Practical implications for exam day:**

- Because there's only one attempt, the practice exam matters: hit ≥ 900/1000 there before booking.
- Because there's no penalty, **answer every question** even if you have to guess.
- Because 4 scenarios are picked from 6, **prepare all six**. The two you don't see still teach the patterns the others test.
- Because it's proctored, treat the exam day like a live presentation: quiet space, working webcam, full ID, no second monitor.

---

## 2. Target candidate & prerequisites

The exam is intended for a **solution architect** who designs and implements production applications with Claude. Anthropic's stated bar:

- **6+ months hands-on experience** with Claude APIs, Agent SDK, Claude Code, and MCP
- Has built agentic applications using the **Claude Agent SDK** — multi-agent orchestration, subagent delegation, tool integration, lifecycle hooks
- Has configured **Claude Code** for team workflows — CLAUDE.md files, Agent Skills, MCP server integrations, plan mode
- Has designed **MCP** tool and resource interfaces for backend system integration
- Has engineered prompts for reliable **structured output** with JSON schemas, few-shot examples, extraction patterns
- Manages **context windows** across long documents, multi-turn conversations, multi-agent handoffs
- Has integrated Claude into **CI/CD pipelines**
- Makes sound **escalation and reliability decisions** — error handling, human-in-the-loop, self-evaluation

**Prerequisites (from the FAQs, verbatim):**

> Completion of at least all 200-level courses in the attached course catalog, working familiarity with Agent SDK (there's not a course for this in our platform yet), and having built solutions with Claude Code, Agent SDK, Anthropic API, and MCP.

If you haven't shipped at least one project across all four (Code, Agent SDK, API, MCP), the FAQs are blunt: **skipping the practice and the guide will likely result in not passing**.

---

## 3. Domains, weightings, scenarios

### Five scored domains

| # | Domain | Weight |
|---|---|---:|
| 1 | **Agentic Architecture & Orchestration** | **27%** |
| 2 | **Tool Design & MCP Integration** | **18%** |
| 3 | **Claude Code Configuration & Workflows** | **20%** |
| 4 | **Prompt Engineering & Structured Output** | **20%** |
| 5 | **Context Management & Reliability** | **15%** |

The largest domain is **Domain 1 (agentic architecture)** at over a quarter of the exam. Don't under-study it.

### Six scenarios (4 are picked at random for your exam)

| # | Scenario | Primary domains tested |
|---|---|---|
| 1 | **Customer Support Resolution Agent** — Agent SDK + custom MCP tools (`get_customer`, `lookup_order`, `process_refund`, `escalate_to_human`); 80%+ first-contact resolution target | 1, 2, 5 |
| 2 | **Code Generation with Claude Code** — code gen, refactoring, debugging, docs; CLAUDE.md, custom slash commands, plan mode | 3, 5 |
| 3 | **Multi-Agent Research System** — coordinator delegating to web-search / document-analysis / synthesis / report-generator subagents | 1, 2, 5 |
| 4 | **Developer Productivity with Claude** — agent that explores codebases using Read/Write/Bash/Grep/Glob plus MCP servers | 2, 3, 1 |
| 5 | **Claude Code for CI/CD** — automated code reviews, test generation, PR feedback; minimise false positives | 3, 4 |
| 6 | **Structured Data Extraction** — extracting from unstructured docs, JSON-schema validation, edge-case handling | 4, 5 |

**Study tactic:** read the scenario text out loud once. Imagine you're the architect on call. You'll see the same vocabulary in the exam questions, often verbatim.

---

## 4. Domain 1 — Agentic Architecture & Orchestration (27%)

The largest domain. Seven task statements covering agentic loops, coordinator-subagent patterns, hooks, decomposition, and session management.

### Task 1.1 — Design and implement agentic loops for autonomous task execution

**Knowledge of:**
- The agentic loop lifecycle: send request → inspect `stop_reason` (`"tool_use"` vs `"end_turn"`) → execute requested tools → return results for the next iteration
- How tool results are appended to conversation history so the model can reason about the next action
- The distinction between **model-driven decision-making** (Claude reasons about which tool to call next based on context) and **pre-configured decision trees / fixed tool sequences**

**Skills in:**
- Implementing agentic loop control flow that **continues when `stop_reason == "tool_use"` and terminates when `stop_reason == "end_turn"`**
- Adding tool results to conversation context between iterations so the model incorporates new information
- **Avoiding anti-patterns:** parsing natural-language signals to determine loop termination, setting arbitrary iteration caps as the *primary* stopping mechanism, or checking for assistant text content as a completion indicator

The loop in plain code:

```python
while True:
    response = client.messages.create(model=..., messages=conv, tools=...)
    if response.stop_reason == "end_turn":
        break                                  # done
    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                conv.append({"role": "user",
                             "content": [{"type": "tool_result",
                                          "tool_use_id": block.id,
                                          "content": result}]})
        conv.append({"role": "assistant", "content": response.content})
```

> **Exam trap:** any answer that proposes "stop the loop when the assistant says 'done'" or "loop a fixed N times" is a distractor. The architect always uses `stop_reason`.

### Task 1.2 — Orchestrate multi-agent systems with coordinator-subagent patterns

**Knowledge of:**
- **Hub-and-spoke architecture** where a coordinator agent manages all inter-subagent communication, error handling, and information routing
- Subagents operate with **isolated context** — they do *not* automatically inherit the coordinator's conversation history
- The role of the coordinator: task decomposition, delegation, result aggregation, deciding which subagents to invoke based on query complexity
- The risk of **overly narrow task decomposition** by the coordinator → incomplete coverage of broad topics

**Skills in:**
- Designing coordinators that **dynamically select** which subagents to invoke rather than always routing through the full pipeline
- **Partitioning research scope** across subagents to minimise duplication (assigning distinct subtopics or source types to each agent)
- Implementing **iterative refinement loops**: coordinator evaluates synthesis output for gaps, re-delegates with targeted queries, re-invokes synthesis until coverage is sufficient
- Routing **all** subagent communication through the coordinator for observability, consistent error handling, and controlled information flow

> **Exam trap (Sample Q7):** if the coordinator decomposed "creative industries" into only `digital art / graphic design / photography`, the root cause is **the coordinator's decomposition**, not the downstream agents. Always look upstream first.

### Task 1.3 — Configure subagent invocation, context passing, and spawning

**Knowledge of:**
- The **`Task` tool** as the mechanism for spawning subagents
- **`allowedTools` must include `"Task"`** for a coordinator to invoke subagents
- Subagent context **must be explicitly provided in the prompt** — subagents do not inherit parent context or share memory
- The **`AgentDefinition`** configuration: descriptions, system prompts, tool restrictions
- **Fork-based session management** for exploring divergent approaches from a shared analysis baseline

**Skills in:**
- Including complete findings from prior agents directly in the subagent's prompt
- Using **structured data formats** to separate content from metadata (source URLs, document names, page numbers) when passing context between agents to preserve attribution
- **Spawning parallel subagents by emitting multiple `Task` tool calls in a single coordinator response** rather than across separate turns
- Designing coordinator prompts that specify research **goals and quality criteria** rather than step-by-step procedural instructions, to enable subagent adaptability

### Task 1.4 — Implement multi-step workflows with enforcement and handoff patterns

**Knowledge of:**
- **Programmatic enforcement (hooks, prerequisite gates) vs prompt-based guidance** for workflow ordering
- When deterministic compliance is required (identity verification before financial operations), prompt instructions alone have a non-zero failure rate
- **Structured handoff protocols** for mid-process escalation: customer details, root cause analysis, recommended actions

**Skills in:**
- Implementing **programmatic prerequisites** that block downstream tool calls until prerequisite steps complete (block `process_refund` until `get_customer` returns a verified customer ID)
- Decomposing multi-concern customer requests into distinct items, then investigating each in parallel using shared context before synthesizing a unified resolution
- Compiling structured handoff summaries (customer ID, root cause, refund amount, recommended action) when escalating to humans who lack the conversation transcript

> **Sample Q1 takeaway:** if the requirement is "the agent must always do X before Y" and errors are financial, **the answer is hooks/programmatic enforcement, not prompt instructions or few-shot examples**.

### Task 1.5 — Apply Agent SDK hooks for tool call interception and data normalization

**Knowledge of:**
- Hook patterns (e.g. **`PostToolUse`**) that intercept tool results for transformation before the model processes them
- Hook patterns that intercept outgoing tool calls to enforce compliance rules (e.g. blocking refunds above a threshold)
- The distinction between **hooks for deterministic guarantees** vs **prompt instructions for probabilistic compliance**

**Skills in:**
- Implementing `PostToolUse` hooks to normalise heterogeneous data formats (Unix timestamps, ISO 8601, numeric status codes) from different MCP tools before the agent processes them
- Implementing tool-call interception hooks that block policy-violating actions (refunds > $500) and redirect to alternative workflows (human escalation)
- **Choosing hooks over prompt-based enforcement when business rules require guaranteed compliance**

### Task 1.6 — Design task decomposition strategies for complex workflows

**Knowledge of:**
- **Fixed sequential pipelines (prompt chaining)** vs **dynamic adaptive decomposition** based on intermediate findings
- Prompt chaining patterns that break reviews into sequential steps (analyse each file individually, then run a cross-file integration pass)
- The value of **adaptive investigation plans** that generate subtasks based on what is discovered at each step

**Skills in:**
- Selecting the right pattern: **prompt chaining for predictable multi-aspect reviews**, **dynamic decomposition for open-ended investigation tasks**
- Splitting large code reviews into per-file local analysis + a separate cross-file integration pass to avoid attention dilution
- Decomposing open-ended tasks ("add comprehensive tests to a legacy codebase") by mapping structure → identifying high-impact areas → creating a prioritised plan that adapts as dependencies are discovered

### Task 1.7 — Manage session state, resumption, and forking

**Knowledge of:**
- **Named session resumption** via `--resume <session-name>`
- **`fork_session`** for creating independent branches from a shared analysis baseline
- The importance of informing the agent about changes to previously analysed files when resuming after code modifications
- Why **starting a new session with a structured summary** is more reliable than resuming with stale tool results

**Skills in:**
- Using `--resume` with session names to continue named investigation sessions across work sessions
- Using `fork_session` to create parallel exploration branches (e.g. comparing two testing strategies from a shared codebase analysis)
- **Choosing between session resumption (when prior context is mostly valid) and starting fresh with injected summaries (when prior tool results are stale)**
- Informing a resumed session about specific file changes for targeted re-analysis rather than full re-exploration

---

## 5. Domain 2 — Tool Design & MCP Integration (18%)

Five task statements on writing tool descriptions, error responses, distributing tools, configuring MCP servers, and using built-in tools.

### Task 2.1 — Design effective tool interfaces with clear descriptions and boundaries

**Knowledge of:**
- **Tool descriptions are the primary mechanism LLMs use for tool selection** — minimal descriptions cause unreliable selection among similar tools
- Good descriptions include **input formats, example queries, edge cases, and boundary explanations**
- How **ambiguous or overlapping descriptions** cause misrouting (e.g. `analyze_content` vs `analyze_document` with near-identical text)
- The impact of **system-prompt wording** on tool selection: keyword-sensitive instructions can create unintended tool associations

**Skills in:**
- Writing tool descriptions that clearly differentiate each tool's purpose, expected inputs, outputs, and **when to use it vs similar alternatives**
- Renaming tools and updating descriptions to eliminate functional overlap (rename `analyze_content` → `extract_web_results` with a web-specific description)
- **Splitting generic tools into purpose-specific tools** (`analyze_document` → `extract_data_points` + `summarize_content` + `verify_claim_against_source`)
- Reviewing system prompts for keyword-sensitive instructions that might override well-written tool descriptions

> **Sample Q2 takeaway:** when two similar tools have minimal descriptions and the model picks the wrong one, **the first step is to expand the descriptions**. Few-shot examples (extra tokens), routing layers (over-engineered), and tool consolidation (bigger change) are all distractors.

### Task 2.2 — Implement structured error responses for MCP tools

**Knowledge of:**
- The MCP **`isError`** flag pattern
- Error categories: **transient** (timeouts, service unavailability), **validation** (invalid input), **business** (policy violations), **permission**
- Why uniform error responses ("Operation failed") prevent the agent from making appropriate recovery decisions
- The difference between **retryable and non-retryable** errors

**Skills in:**
- Returning structured error metadata: **`errorCategory`** (`transient` / `validation` / `permission`), **`isRetryable`** boolean, human-readable descriptions
- Including `retriable: false` flags + customer-friendly explanations for business-rule violations
- **Local error recovery within subagents** for transient failures; only propagate to the coordinator when irrecoverable, including what was attempted and partial results
- Distinguishing **access failures** (need retry decisions) from **valid empty results** (successful queries with no matches)

### Task 2.3 — Distribute tools appropriately across agents and configure tool choice

**Knowledge of:**
- Giving an agent **too many tools (e.g. 18 vs 4–5)** degrades selection reliability
- Agents with tools outside their specialisation tend to misuse them (synthesis agent attempting web searches)
- **Scoped tool access:** give agents only the tools needed for their role, with limited cross-role tools for high-frequency needs
- **`tool_choice` options:** `"auto"`, `"any"`, and **forced selection** `{"type": "tool", "name": "..."}`

**Skills in:**
- Restricting each subagent's tool set to its role; preventing cross-specialisation misuse
- Replacing generic tools with constrained alternatives (`fetch_url` → `load_document` that validates document URLs)
- Providing scoped cross-role tools for high-frequency needs (a `verify_fact` tool for the synthesis agent) while routing complex cases through the coordinator
- Using **forced** `tool_choice` to ensure a specific tool is called first (force `extract_metadata` before enrichment)
- Setting **`tool_choice: "any"`** to guarantee the model calls a tool rather than returning conversational text

> **Sample Q9 takeaway:** the principle of **least privilege** wins. Give the synthesis agent a *scoped* `verify_fact` tool for the 85% common case, route complex cases through the coordinator. Do **not** give it the full web-search arsenal.

### Task 2.4 — Integrate MCP servers into Claude Code and agent workflows

**Knowledge of:**
- **MCP server scoping:** project-level (`.mcp.json`) for shared team tooling vs user-level (`~/.claude.json`) for personal/experimental servers
- **Environment variable expansion** in `.mcp.json` (e.g. `${GITHUB_TOKEN}`) for credential management
- Tools from all configured MCP servers are discovered at connection time and available simultaneously
- **MCP resources** as a mechanism for exposing content catalogs (issue summaries, doc hierarchies, DB schemas) to reduce exploratory tool calls

**Skills in:**
- Configuring shared MCP servers in `.mcp.json` with env-var expansion for auth tokens
- Configuring personal/experimental MCP servers in user-scoped `~/.claude.json`
- Enhancing MCP tool descriptions to explain capabilities and outputs in detail, **preventing the agent from preferring built-in tools (Grep) over more capable MCP tools**
- **Choosing existing community MCP servers over custom implementations** for standard integrations (Jira); reserve custom servers for team-specific workflows
- Exposing content catalogs as MCP resources to give agents visibility without exploratory calls

### Task 2.5 — Select and apply built-in tools (Read, Write, Edit, Bash, Grep, Glob) effectively

**Knowledge of:**
- **Grep** for content search (function names, error messages, import statements)
- **Glob** for file path pattern matching (find files by name/extension)
- **Read/Write** for full file ops; **Edit** for targeted modifications via unique text matching
- When `Edit` fails due to non-unique text matches, **use Read + Write as fallback**

**Skills in:**
- Selecting Grep for code-content searches (find all callers of a function)
- Selecting Glob for naming patterns (`**/*.test.tsx`)
- Using Read → Write when Edit can't find a unique anchor
- **Building codebase understanding incrementally:** Grep to find entry points, then Read to follow imports and trace flows — rather than reading all files upfront
- Tracing function usage across wrapper modules: identify all exported names → search for each name across the codebase

---

## 6. Domain 3 — Claude Code Configuration & Workflows (20%)

Six task statements on CLAUDE.md, slash commands, skills, path-specific rules, plan mode, and CI/CD integration.

### Task 3.1 — Configure CLAUDE.md files with appropriate hierarchy, scoping, and modular organization

**Knowledge of:**
- **Configuration hierarchy:** user-level (`~/.claude/CLAUDE.md`) → project-level (`.claude/CLAUDE.md` or root `CLAUDE.md`) → directory-level (subdirectory `CLAUDE.md` files)
- **User-level settings apply only to that user** — instructions in `~/.claude/CLAUDE.md` are not shared with teammates via version control
- **`@import`** syntax for referencing external files to keep CLAUDE.md modular
- **`.claude/rules/`** directory as an alternative to a monolithic CLAUDE.md

**Skills in:**
- Diagnosing hierarchy issues (a new team member missing instructions because they live in user-level rather than project-level)
- Using `@import` to selectively include relevant standards files in each package's CLAUDE.md
- Splitting large CLAUDE.md into focused topic files in `.claude/rules/` (`testing.md`, `api-conventions.md`, `deployment.md`)
- Using **`/memory`** to verify which memory files are loaded and diagnose inconsistent behaviour

### Task 3.2 — Create and configure custom slash commands and skills

**Knowledge of:**
- **Project-scoped commands** in `.claude/commands/` (shared via VCS) vs **user-scoped** in `~/.claude/commands/` (personal)
- **Skills** in `.claude/skills/` with `SKILL.md` files supporting frontmatter:
  - **`context: fork`** — runs the skill in an isolated sub-agent context, preventing skill outputs from polluting the main conversation
  - **`allowed-tools`** — restrict tool access during skill execution
  - **`argument-hint`** — prompt for required parameters when invoked without arguments
- **Personal skill customisation:** create personal variants in `~/.claude/skills/` with different names to avoid affecting teammates

**Skills in:**
- Creating project-scoped slash commands in `.claude/commands/` for team-wide availability via VCS
- Using `context: fork` to isolate skills with verbose output (codebase analysis) or exploratory context (brainstorming alternatives)
- Configuring `allowed-tools` to restrict tool access during skill execution (limit to file writes to prevent destructive actions)
- Using `argument-hint` to prompt developers for required parameters
- **Choosing between skills (on-demand, task-specific) and CLAUDE.md (always-loaded universal standards)**

> **Sample Q4 takeaway:** for a team-wide custom slash command, the location is `.claude/commands/` in the project repo. `~/.claude/commands/` is personal, CLAUDE.md is for instructions, and `.claude/config.json` doesn't exist.

### Task 3.3 — Apply path-specific rules for conditional convention loading

**Knowledge of:**
- **`.claude/rules/`** files with YAML frontmatter **`paths`** field containing glob patterns
- Path-scoped rules **load only when editing matching files**, reducing irrelevant context and token usage
- **Glob-pattern rules beat directory-level CLAUDE.md** for conventions spanning multiple directories (test files spread throughout a codebase)

**Skills in:**
- Creating `.claude/rules/` files with frontmatter path scoping:
  ```yaml
  ---
  paths: ["terraform/**/*"]
  ---
  ```
- Using glob patterns to apply conventions by file *type* regardless of directory location (`**/*.test.tsx` for all test files)
- Choosing path-specific rules over subdirectory CLAUDE.md when conventions must apply to files spread across the codebase

> **Sample Q6 takeaway:** for conventions that apply to files by *type* spread across many directories, **`.claude/rules/` with glob patterns** wins. CLAUDE.md infers, skills require manual invocation, and per-directory CLAUDE.md can't span directories.

### Task 3.4 — Determine when to use plan mode vs direct execution

**Knowledge of:**
- **Plan mode** is for complex tasks: large-scale changes, multiple valid approaches, architectural decisions, multi-file modifications
- **Direct execution** is for simple, well-scoped changes (single validation check, one-file bug fix)
- Plan mode enables **safe codebase exploration and design before committing**, preventing costly rework
- **The Explore subagent** for isolating verbose discovery output and returning summaries

**Skills in:**
- Selecting plan mode for architectural tasks: microservice restructuring, library migrations affecting 45+ files, choosing between integration approaches
- Selecting direct execution for well-understood changes: single-file bug fix with a clear stack trace, adding a date-validation conditional
- Using the Explore subagent for verbose discovery phases to prevent context-window exhaustion
- Combining plan mode for investigation with direct execution for implementation

> **Sample Q5 takeaway:** monolith→microservices restructuring across dozens of files **always picks plan mode**. "Start direct, switch later" is a distractor — the complexity is already known.

### Task 3.5 — Apply iterative refinement techniques for progressive improvement

**Knowledge of:**
- **Concrete input/output examples** are the most effective way to communicate expected transformations when prose is interpreted inconsistently
- **Test-driven iteration:** write test suites first, iterate by sharing test failures
- **The interview pattern:** have Claude ask questions to surface considerations the developer didn't anticipate before implementing
- **When to provide all issues in one message** (interacting problems) vs sequentially (independent problems)

**Skills in:**
- Providing 2–3 concrete input/output examples to clarify transformation requirements
- Writing test suites covering expected behaviour, edge cases, performance — then iterating by sharing failures
- Using the interview pattern to surface design considerations (cache invalidation strategies, failure modes) in unfamiliar domains
- Providing specific test cases with example input + expected output to fix edge-case handling
- **Addressing multiple interacting issues in a single detailed message; sequential iteration for independent issues**

### Task 3.6 — Integrate Claude Code into CI/CD pipelines

**Knowledge of:**
- **`-p` (or `--print`)** flag for non-interactive mode in pipelines
- **`--output-format json`** and **`--json-schema`** for enforced structured output in CI
- **CLAUDE.md** as the mechanism for providing project context (testing standards, fixtures, review criteria) to CI-invoked Claude Code
- **Session context isolation:** the same Claude session that *generated* code is less effective at *reviewing* its own changes than an independent review instance

**Skills in:**
- Running Claude Code in CI with `-p` to prevent interactive input hangs
- Using `--output-format json` with `--json-schema` for machine-parseable findings (auto-posted as inline PR comments)
- Including prior review findings when re-running on new commits, instructing Claude to report only **new or still-unaddressed** issues to avoid duplicate comments
- Providing existing test files in context so test generation avoids duplicate scenarios
- Documenting testing standards, valuable test criteria, and available fixtures in CLAUDE.md

> **Sample Q10 takeaway:** if the pipeline hangs waiting for input, the answer is **`-p`**. `CLAUDE_HEADLESS=true`, `--batch`, and `< /dev/null` are distractors (the first two don't exist; the third doesn't address the syntax).

---

## 7. Domain 4 — Prompt Engineering & Structured Output (20%)

Six task statements on explicit criteria, few-shot, JSON schemas, validation loops, batch processing, and multi-pass review.

### Task 4.1 — Design prompts with explicit criteria to improve precision and reduce false positives

**Knowledge of:**
- **Explicit criteria > vague instructions** ("flag comments only when claimed behaviour contradicts actual code behaviour" vs "check that comments are accurate")
- Generic instructions like "be conservative" or "only report high-confidence findings" **fail** to improve precision compared to specific categorical criteria
- The impact of **false positive rates on developer trust:** high-FP categories undermine confidence in accurate categories

**Skills in:**
- Writing specific review criteria that define which issues to **report** (bugs, security) vs **skip** (minor style, local patterns)
- **Temporarily disabling high-FP categories** to restore developer trust while improving prompts
- Defining explicit severity criteria with concrete code examples for each severity level

### Task 4.2 — Apply few-shot prompting to improve output consistency and quality

**Knowledge of:**
- Few-shot is **the most effective technique** for consistently formatted, actionable output when detailed instructions alone produce inconsistent results
- The role of few-shot in demonstrating **ambiguous-case handling** (tool selection for ambiguous requests)
- Few-shot enables **generalisation to novel patterns** rather than matching only pre-specified cases
- Effectiveness for **reducing hallucination** in extraction (informal measurements, varied document structures)

**Skills in:**
- **Creating 2–4 targeted few-shot examples** for ambiguous scenarios that show reasoning for why one action was chosen over plausible alternatives
- Including examples that demonstrate desired output format (location, issue, severity, suggested fix)
- Providing examples distinguishing **acceptable patterns from genuine issues** to reduce false positives while enabling generalisation
- Using few-shot for varied document structures (inline citations vs bibliographies, methodology sections vs embedded details)
- Adding examples showing correct extraction from documents with varied formats to address empty/null extraction of required fields

### Task 4.3 — Enforce structured output using tool use and JSON schemas

**Knowledge of:**
- **Tool use with JSON schemas** is the most reliable approach for guaranteed schema-compliant output, eliminating JSON syntax errors
- **`tool_choice` distinctions:**
  - `"auto"` — model may return text instead of calling a tool
  - `"any"` — model must call a tool but can choose which
  - `{"type": "tool", "name": "..."}` — model must call this specific named tool
- **Strict schemas eliminate syntax errors but NOT semantic errors** (line items that don't sum to total, values in wrong fields)
- Schema design: required vs optional, **enum + `"other"` + detail string patterns** for extensible categories

**Skills in:**
- Defining extraction tools with JSON schemas as input parameters; extracting structured data from the `tool_use` response
- Setting `tool_choice: "any"` to guarantee structured output when multiple extraction schemas exist and the document type is unknown
- Forcing a specific tool with `{"type": "tool", "name": "extract_metadata"}` to ensure a particular extraction runs first
- **Designing schema fields as optional/nullable when source documents may not contain the information** — prevents the model from fabricating values
- Adding enum values like `"unclear"` for ambiguous cases and `"other"` + detail fields
- Including format-normalization rules in prompts alongside strict output schemas

### Task 4.4 — Implement validation, retry, and feedback loops for extraction quality

**Knowledge of:**
- **Retry-with-error-feedback:** appending specific validation errors to the prompt on retry to guide correction
- **Limits of retry:** retries are ineffective when the required information is **simply absent** from the source document (vs format/structural errors)
- Feedback-loop design: tracking which code constructs trigger findings (`detected_pattern` field) for systematic dismissal-pattern analysis
- **Semantic validation errors** (don't sum, wrong field) vs **schema syntax errors** (eliminated by tool use)

**Skills in:**
- Implementing follow-up requests including the original document, the failed extraction, and specific validation errors
- Identifying when retries will be **ineffective** (information exists only externally) vs **succeed** (format mismatches, structural output errors)
- Adding `detected_pattern` fields to enable analysis of false-positive patterns when developers dismiss findings
- Designing self-correction validation flows: extract `calculated_total` alongside `stated_total` to flag discrepancies; add `conflict_detected` booleans for inconsistent source data

### Task 4.5 — Design efficient batch processing strategies

**Knowledge of:**
- **Message Batches API:** **50% cost savings**, up to **24-hour processing window**, **no guaranteed latency SLA**
- Batch is appropriate for non-blocking, latency-tolerant workloads (overnight reports, weekly audits, nightly test generation)
- Batch is **inappropriate for blocking workflows** (pre-merge checks)
- **The batch API does NOT support multi-turn tool calling within a single request**
- **`custom_id`** fields for correlating batch request/response pairs

**Skills in:**
- **Matching API approach to latency requirements:** synchronous for blocking pre-merge checks, batch for overnight/weekly analysis
- Calculating batch submission frequency based on SLA constraints (4-hour windows to guarantee 30-hour SLA with 24-hour batch processing)
- Handling failures: resubmit only failed documents (identified by `custom_id`) with appropriate modifications (chunking docs that exceeded context limits)
- Using prompt refinement on a sample set before batch-processing large volumes

> **Sample Q11 takeaway:** if you have one blocking workflow + one overnight workflow, **only the overnight one moves to batch**. Switching the blocking one is the trap.

### Task 4.6 — Design multi-instance and multi-pass review architectures

**Knowledge of:**
- **Self-review limitations:** a model retains reasoning context from generation, making it less likely to question its own decisions in the same session
- **Independent review instances** (without prior reasoning context) are more effective than self-review or extended thinking
- **Multi-pass review:** split large reviews into per-file local passes + cross-file integration passes to avoid attention dilution and contradictory findings

**Skills in:**
- Using a **second independent Claude instance** to review generated code without the generator's reasoning context
- Splitting large multi-file reviews into focused per-file passes for local issues + integration passes for cross-file data flow
- Running verification passes where the model self-reports confidence alongside each finding for calibrated review routing

> **Sample Q12 takeaway:** when a single-pass review of 14 files produces inconsistent depth and contradictory findings, **split into per-file + integration passes**. Don't ask developers to chunk PRs, don't switch models hoping bigger context fixes attention quality, and don't use majority-voting (which suppresses legitimate findings).

---

## 8. Domain 5 — Context Management & Reliability (15%)

Six task statements on context preservation, escalation, error propagation, large-codebase exploration, human review, and information provenance.

### Task 5.1 — Manage conversation context to preserve critical information across long interactions

**Knowledge of:**
- **Progressive summarisation risks:** condensing numerical values, percentages, dates, customer-stated expectations into vague summaries
- **The "lost in the middle" effect:** models reliably process information at the **beginning and end** of long inputs but may omit findings from middle sections
- Tool results accumulate in context disproportionately to relevance (40+ fields per order lookup when only 5 are relevant)
- The importance of **passing complete conversation history** in subsequent API requests for coherence

**Skills in:**
- Extracting transactional facts (amounts, dates, order numbers, statuses) into a **persistent "case facts" block** included in each prompt, *outside* summarized history
- Extracting and persisting structured issue data into a separate context layer for multi-issue sessions
- **Trimming verbose tool outputs** to only relevant fields before they accumulate
- Placing **key findings summaries at the beginning** of aggregated inputs and organising detailed results with explicit section headers
- Requiring subagents to include metadata (dates, source locations, methodological context) in structured outputs
- Modifying upstream agents to return **structured data (key facts, citations, relevance scores) instead of verbose content** when downstream agents have limited context budgets

### Task 5.2 — Design effective escalation and ambiguity resolution patterns

**Knowledge of:**
- **Appropriate escalation triggers:** customer requests for a human, policy exceptions/gaps (not just complex cases), inability to make meaningful progress
- The distinction between **escalating immediately** when a customer explicitly demands it vs **offering to resolve** when the issue is straightforward
- **Sentiment-based escalation and self-reported confidence scores are unreliable proxies** for actual case complexity
- Multiple customer matches require **clarification** (request additional identifiers), not heuristic selection

**Skills in:**
- Adding **explicit escalation criteria with few-shot examples** to the system prompt
- Honouring explicit customer requests for human agents **immediately, without first attempting investigation**
- Acknowledging frustration while offering resolution when within capability; escalating only if the customer reiterates
- Escalating when policy is **ambiguous or silent** on the customer's specific request (competitor price-matching when policy only addresses own-site adjustments)
- Instructing the agent to **ask for additional identifiers** when tool results return multiple matches, rather than selecting heuristically

> **Sample Q3 takeaway:** when the agent over-escalates simple cases and under-escalates complex ones, the answer is **explicit criteria + few-shot examples**, not self-reported confidence scores (poorly calibrated), classifier models (over-engineered), or sentiment analysis (wrong proxy).

### Task 5.3 — Implement error propagation strategies across multi-agent systems

**Knowledge of:**
- **Structured error context** (failure type, attempted query, partial results, alternative approaches) enables intelligent coordinator recovery
- **Access failures** (need retry decisions) vs **valid empty results** (successful queries with no matches)
- Why **generic error statuses** ("search unavailable") hide valuable context
- Why **silently suppressing errors** (returning empty as success) and **terminating entire workflows on single failures** are *both* anti-patterns

**Skills in:**
- Returning structured error context including failure type, what was attempted, partial results, potential alternatives
- Distinguishing access failures from valid empty results so the coordinator can decide
- Subagents implementing **local recovery for transient failures**; only propagating when irrecoverable, with what was attempted + partial results
- Structuring synthesis output with **coverage annotations** indicating well-supported findings vs gaps from unavailable sources

> **Sample Q8 takeaway:** when a subagent times out, the architect's answer is **structured error context with failure type / attempted query / partial results / alternatives**. Generic statuses, silent success, and workflow termination are all wrong.

### Task 5.4 — Manage context effectively in large codebase exploration

**Knowledge of:**
- **Context degradation in extended sessions:** models start giving inconsistent answers and referencing "typical patterns" rather than specific classes discovered earlier
- **Scratchpad files** for persisting key findings across context boundaries
- **Subagent delegation** to isolate verbose exploration output while the main agent coordinates high-level understanding
- **Structured state persistence** for crash recovery: agents export state to a known location, coordinator loads a manifest on resume

**Skills in:**
- Spawning subagents to investigate specific questions ("find all test files," "trace refund flow dependencies") while the main agent preserves coordination
- Maintaining scratchpad files recording key findings, referencing them for subsequent questions
- Summarising findings from one exploration phase before spawning subagents for the next
- Designing crash recovery using structured agent state exports (manifests) the coordinator loads on resume
- Using **`/compact`** to reduce context usage during extended exploration sessions

### Task 5.5 — Design human review workflows and confidence calibration

**Knowledge of:**
- The risk that **aggregate accuracy metrics** (97% overall) may mask poor performance on specific document types or fields
- **Stratified random sampling** for measuring error rates in high-confidence extractions and detecting novel error patterns
- **Field-level confidence scores** calibrated using labelled validation sets for routing review attention
- Validating accuracy by **document type and field segment** before automating high-confidence extractions

**Skills in:**
- Stratified random sampling of high-confidence extractions for ongoing error rate measurement
- Analysing accuracy by document type and field to verify consistent performance across all segments before reducing human review
- Field-level confidence scores → calibrated review thresholds using labelled validation sets
- Routing low-confidence extractions and ambiguous/contradictory documents to human review, prioritising limited reviewer capacity

### Task 5.6 — Preserve information provenance and handle uncertainty in multi-source synthesis

**Knowledge of:**
- **Source attribution is lost during summarisation** when findings are compressed without preserving claim-source mappings
- **Structured claim-source mappings** that the synthesis agent must preserve and merge
- Handling **conflicting statistics from credible sources:** annotate conflicts with attribution rather than arbitrarily selecting
- **Temporal data:** require publication/collection dates in structured outputs to prevent temporal differences from being misread as contradictions

**Skills in:**
- Requiring subagents to output structured **claim-source mappings** (URLs, document names, relevant excerpts) preserved through synthesis
- Structuring reports with **explicit sections distinguishing well-established findings from contested ones**, preserving original characterisations and methodological context
- Document analysis that **includes conflicting values explicitly annotated**, letting the coordinator decide reconciliation
- Including publication/data-collection dates in structured outputs for correct temporal interpretation
- Rendering different content types appropriately — **financial as tables, news as prose, technical findings as structured lists** — rather than uniform formatting

---

## 9. Claude commands & configuration reference

This section is a quick-recall card for every command, file path, flag, and frontmatter option that appears in the syllabus. Memorise this — these are the things distractor answers will get *almost* right.

### 9.1 — Slash commands (built-in)

| Command | What it does | When to use |
|---|---|---|
| **`/memory`** | Show which memory (CLAUDE.md, rules) files are currently loaded | Diagnose inconsistent behaviour across sessions / verify hierarchy |
| **`/compact`** | Reduce context usage in extended sessions by compressing history | When context fills with verbose discovery output |
| `/review`, `/test`, `/refactor`, etc. | **User-defined custom commands** — see §9.4 | Team-wide reusable workflows |

### 9.2 — Claude Code CLI flags

| Flag | Effect |
|---|---|
| **`-p`** or **`--print`** | Non-interactive mode — process prompt, write to stdout, exit. **Required in CI/CD pipelines.** |
| **`--output-format json`** | Emit a structured JSON output instead of free-form text. Combine with `--json-schema`. |
| **`--json-schema <path>`** | Constrain output to the specified JSON schema. Used together with `--output-format json` for machine-parseable CI findings. |
| **`--resume <session-name>`** | Continue a specific named prior conversation. |

### 9.3 — Configuration file locations & precedence

```
USER LEVEL (your machine, not shared)
  ~/.claude/CLAUDE.md             general standards across all projects
  ~/.claude/commands/             personal slash commands
  ~/.claude/skills/               personal skills
  ~/.claude.json                  personal MCP server configs (experimental)

PROJECT LEVEL (in repo, shared via VCS)
  CLAUDE.md   or   .claude/CLAUDE.md     team-wide project context
  .claude/rules/<topic>.md               path-scoped rule files
  .claude/commands/<name>.md             team slash commands
  .claude/skills/<name>/SKILL.md         team skills
  .mcp.json                              shared MCP server configs

DIRECTORY LEVEL
  <subdir>/CLAUDE.md             scoped to that directory's files
```

**Precedence rule of thumb:** more specific overrides more general. Directory-level beats project-level beats user-level for conflicting instructions. **User-level instructions never reach teammates** unless they replicate the file.

### 9.4 — `.claude/commands/<name>.md` — custom slash commands

A markdown file under `.claude/commands/` becomes a `/<name>` command. Project-scoped variants ride along in version control automatically.

```markdown
---
description: Run our team's standard PR review checklist
---
Review the pull request on the current branch against:
- Security: SQL injection, XSS, auth checks
- Reliability: error handling, retry behaviour
- Tests: coverage of new branches

Output as a JSON list of {file, line, severity, suggestion}.
```

### 9.5 — `.claude/skills/<name>/SKILL.md` — custom skills

Skills are richer than commands — they can fork their own context, restrict tool access, and prompt for arguments.

```markdown
---
description: Explore an unfamiliar codebase and produce a structured summary
context: fork                           # isolated sub-agent context
allowed-tools: [Read, Grep, Glob]      # no Write or Bash
argument-hint: directory_path
---
Investigate the directory at {{argument}}.
Map module structure, identify entry points, and trace 3 critical user flows.
Return a structured summary; do NOT modify any files.
```

**Frontmatter options worth memorising:**

| Field | Purpose |
|---|---|
| **`context: fork`** | Run in an isolated sub-agent context. Verbose output / brainstorming alternatives don't pollute the main session. |
| **`allowed-tools`** | Restrict which tools the skill may call — defence in depth against destructive actions. |
| **`argument-hint`** | When the user invokes the skill without arguments, prompt them for what's needed. |

### 9.6 — `.claude/rules/<topic>.md` — path-scoped rules

```markdown
---
paths: ["**/*.test.tsx", "**/*.spec.ts"]
---
All tests must:
- Use Vitest (not Jest)
- Mock network calls via msw, never via fetch-mock
- Avoid snapshot testing for anything user-facing
```

This file's rules **only load when Claude is editing files matching the paths globs.** Use this for cross-cutting conventions (testing, security) that apply by file *type* regardless of directory.

### 9.7 — `.mcp.json` — project MCP servers

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "internal-jira": {
      "command": "uv",
      "args": ["run", "python", "-m", "our_company.mcp.jira_server"],
      "env": { "JIRA_API_KEY": "${JIRA_API_KEY}" }
    }
  }
}
```

**Things to remember:**

- **`${VAR}`** expansion lets you reference env vars without committing secrets.
- All servers' tools are **discovered at connection time** and available simultaneously to the agent.
- For experimental / personal servers, put them in `~/.claude.json` instead — they won't be shared.

### 9.8 — `@import` syntax in CLAUDE.md

```markdown
# CLAUDE.md (root)

@import .claude/rules/security.md
@import .claude/rules/api-conventions.md
@import packages/billing/CLAUDE.md
```

Use to keep CLAUDE.md modular: each package or topic gets its own focused file, the root composes.

### 9.9 — Built-in tools (Domain 2.5)

| Tool | Purpose |
|---|---|
| **`Read`** | Read file contents |
| **`Write`** | Write/overwrite full file |
| **`Edit`** | Targeted modification via unique text matching (falls back to Read+Write when text isn't unique) |
| **`Bash`** | Execute shell commands |
| **`Grep`** | **Content** search across files (function names, error messages, imports) |
| **`Glob`** | **Path** pattern matching (`**/*.test.tsx`) |
| **`Task`** | Spawn a subagent — **must be in `allowedTools` for a coordinator to invoke subagents** |
| **`Explore`** | Subagent that isolates verbose discovery output and returns summaries |

### 9.10 — Agent SDK essentials

| Concept | What it is |
|---|---|
| **Agentic loop** | While `stop_reason == "tool_use"`: execute tools, append results, re-send. Stop on `"end_turn"`. |
| **`allowedTools`** | Per-agent tool whitelist. Must include `"Task"` for coordinators. |
| **`AgentDefinition`** | Configuration of a subagent: description, system prompt, tool restrictions. |
| **`fork_session`** | Create independent branches from a shared baseline for divergent exploration. |
| **`PostToolUse` hook** | Intercept tool *results* for transformation (data normalisation across sources). |
| **Tool-call interception hook** | Intercept *outgoing* tool calls to enforce compliance (block refunds > $500). |

### 9.11 — `tool_choice` modes (Claude API)

| Value | Behaviour |
|---|---|
| `{"type": "auto"}` | Model decides whether to call a tool |
| `{"type": "any"}` | Model **must** call some tool — guarantees structured output |
| `{"type": "tool", "name": "X"}` | Model must call this specific named tool |
| `{"type": "none"}` | No tool calls allowed — pure-text response |

### 9.12 — Message Batches API

| Property | Value |
|---|---|
| Cost discount | **50% off both input and output** |
| Latency | Up to **24 hours** — no SLA guarantee |
| Multi-turn tool use | **NOT supported** within a single batch request |
| Correlation | **`custom_id`** for request/response pairing |
| Use for | Overnight reports, weekly audits, nightly test generation |
| Don't use for | Pre-merge checks, anything user-facing |

---

## 10. 12 sample exam questions with explanations

The next 12 questions are **drawn directly from the official practice test**. Memorise the patterns — they reflect the exam's distractor style.

### Scenario: Customer Support Resolution Agent

**Q1.** Production data shows that in 12% of cases, your agent skips `get_customer` entirely and calls `lookup_order` using only the customer's stated name, occasionally leading to misidentified accounts and incorrect refunds. What change would most effectively address this reliability issue?

A) Add a programmatic prerequisite that blocks `lookup_order` and `process_refund` calls until `get_customer` has returned a verified customer ID.
B) Enhance the system prompt to state that customer verification via `get_customer` is mandatory before any order operations.
C) Add few-shot examples showing the agent always calling `get_customer` first, even when customers volunteer order details.
D) Implement a routing classifier that analyses each request and enables only the subset of tools appropriate for that request type.

**Correct: A.** When a specific tool sequence is required for critical business logic, **programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot**. Options B and C rely on probabilistic LLM compliance — insufficient when errors have financial consequences. D addresses tool *availability*, not tool *ordering*.

---

**Q2.** Production logs show the agent frequently calls `get_customer` when users ask about orders (e.g. "check my order #12345"), instead of `lookup_order`. Both tools have minimal descriptions ("Retrieves customer information" / "Retrieves order details") and accept similar identifier formats. What's the most effective first step to improve tool selection reliability?

A) Add few-shot examples to the system prompt demonstrating correct tool selection patterns.
B) Expand each tool's description to include input formats it handles, example queries, edge cases, and boundaries explaining when to use it versus similar tools.
C) Implement a routing layer that parses user input before each turn and pre-selects the appropriate tool.
D) Consolidate both tools into a single `lookup_entity` tool that accepts any identifier.

**Correct: B.** **Tool descriptions are the primary mechanism LLMs use for tool selection.** Minimal descriptions cause unreliable selection among similar tools. B is the low-effort, high-leverage fix. Few-shot adds tokens without addressing the root cause. A routing layer is over-engineered. Consolidation is a valid architectural choice but isn't a "first step."

---

**Q3.** Your agent achieves 55% first-contact resolution, well below the 80% target. Logs show it escalates straightforward cases (standard damage replacements with photo evidence) while attempting to autonomously handle complex situations requiring policy exceptions. What's the most effective way to improve escalation calibration?

A) Add explicit escalation criteria to your system prompt with few-shot examples demonstrating when to escalate versus resolve autonomously.
B) Have the agent self-report a confidence score (1–10) and route to humans below a threshold.
C) Deploy a separate classifier model trained on historical tickets to predict which requests need escalation.
D) Implement sentiment analysis to detect customer frustration and auto-escalate.

**Correct: A.** Explicit criteria + few-shot directly addresses unclear decision boundaries. **B fails because LLM self-reported confidence is poorly calibrated** — the agent is *already* confidently wrong on hard cases. C is over-engineered. D solves a different problem; sentiment doesn't correlate with case complexity.

---

### Scenario: Code Generation with Claude Code

**Q4.** You want to create a custom `/review` slash command that runs your team's standard code review checklist. This command should be available to every developer when they clone or pull the repository. Where should you create this command file?

A) `.claude/commands/` in the project repository
B) `~/.claude/commands/` in each developer's home directory
C) The `CLAUDE.md` file at the project root
D) A `.claude/config.json` file with a commands array

**Correct: A.** Project-scoped slash commands live in `.claude/commands/`. Version-controlled, automatically available to teammates. B is for personal commands. C is for instructions, not commands. **D describes a mechanism that doesn't exist** in Claude Code — a classic distractor that sounds plausible.

---

**Q5.** You've been assigned to restructure a monolithic application into microservices. Dozens of files, decisions about service boundaries and module dependencies. Which approach should you take?

A) Enter plan mode to explore, understand dependencies, and design before changes.
B) Start with direct execution and let the implementation reveal natural service boundaries.
C) Direct execution with comprehensive upfront instructions.
D) Direct execution; switch to plan mode if unexpected complexity appears.

**Correct: A.** Plan mode is **designed for exactly this**: large-scale changes, multiple valid approaches, architectural decisions. Plan mode enables safe exploration before commitment. B risks costly rework. C assumes you already know the right structure. D ignores that complexity is *already known*, not emergent.

---

**Q6.** Your codebase has distinct areas with different conventions: React functional components, async/await API handlers, repository-pattern DB models. **Test files are spread throughout** alongside the code they test. You want all tests to follow the same conventions regardless of location. Most maintainable approach?

A) Create rule files in `.claude/rules/` with YAML frontmatter glob patterns.
B) Consolidate all conventions in the root CLAUDE.md.
C) Create skills in `.claude/skills/` for each code type.
D) Place a separate CLAUDE.md in each subdirectory.

**Correct: A.** `.claude/rules/` with glob patterns (e.g. `**/*.test.tsx`) **applies conventions automatically based on file paths regardless of directory**. B relies on inference. C requires manual invocation. D **can't span directories** — CLAUDE.md is directory-bound, but tests live everywhere.

---

### Scenario: Multi-Agent Research System

**Q7.** Topic: "impact of AI on creative industries." Each subagent succeeds: web search finds articles, document analysis summarises correctly, synthesis produces coherent output. **But final reports cover only visual arts** — missing music, writing, film. Coordinator logs show it decomposed the topic into "AI in digital art creation," "AI in graphic design," "AI in photography." Root cause?

A) The synthesis agent lacks instructions for identifying coverage gaps.
B) The coordinator's task decomposition is too narrow.
C) The web search agent's queries aren't comprehensive enough.
D) The document analysis agent filters out non-visual sources due to overly restrictive relevance criteria.

**Correct: B.** The coordinator's logs reveal it directly: **decomposed "creative industries" into only visual-arts subtasks**. Subagents executed correctly within scope; the problem is *what they were assigned*. **Always look upstream first when downstream agents work but coverage fails.**

---

**Q8.** The web search subagent times out. How should this failure flow back to the coordinator for intelligent recovery?

A) Return structured error context including the failure type, attempted query, partial results, and potential alternative approaches.
B) Implement automatic retry with exponential backoff in the subagent; return generic "search unavailable" only after exhausting retries.
C) Catch the timeout in the subagent and return an empty result set marked as successful.
D) Propagate the timeout exception to a top-level handler that terminates the entire workflow.

**Correct: A.** **Structured error context enables intelligent coordinator recovery decisions** — retry with a modified query, try alternatives, or proceed with partial results. B's generic status hides context. C silently suppresses errors as success → guaranteed incomplete output. D terminates unnecessarily when recovery could succeed.

---

**Q9.** Synthesis agent frequently needs to verify claims while combining findings — currently round-trips through the coordinator, adding 40% latency. Evaluation: 85% of verifications are simple fact-checks (dates, names, statistics), 15% require deeper investigation. How to reduce overhead?

A) Give the synthesis agent a scoped `verify_fact` tool for simple lookups; complex verifications still delegate to web search via coordinator.
B) Have synthesis batch its verification needs and return them to the coordinator at end-of-pass.
C) Give synthesis access to all web search tools so it can handle any verification directly.
D) Have web search proactively cache extra context around each source during initial research.

**Correct: A.** **Principle of least privilege.** Give synthesis exactly what it needs for the 85% common case while preserving coordination for complex 15%. B creates blocking dependencies. C over-provisions, violating separation of concerns. D relies on speculative caching.

---

### Scenario: Claude Code for CI/CD

**Q10.** Your pipeline runs `claude "Analyze this pull request for security issues"` but hangs. Logs show Claude Code is waiting for interactive input. Correct approach?

A) Add the `-p` flag: `claude -p "Analyze this pull request..."`
B) Set environment variable `CLAUDE_HEADLESS=true`.
C) Redirect stdin from `/dev/null`.
D) Add the `--batch` flag.

**Correct: A.** **`-p` (or `--print`) is the documented non-interactive mode.** Processes prompt, outputs to stdout, exits. B's env var doesn't exist. **D's `--batch` flag doesn't exist either** (don't confuse with the *Message Batches API* — different concept). C is a Unix workaround that doesn't address Claude Code's syntax.

---

**Q11.** Two workflows currently use real-time Claude calls: (1) a **blocking pre-merge check**, (2) an **overnight technical-debt report**. Manager proposes switching both to Message Batches API for 50% savings. How should you evaluate?

A) Use batch only for the technical-debt reports; keep real-time for pre-merge.
B) Switch both with status polling.
C) Keep real-time for both to avoid batch result ordering issues.
D) Switch both with timeout fallback to real-time.

**Correct: A.** Batch API has **24-hour processing window with no latency SLA** — unsuitable for blocking pre-merge checks where developers wait, ideal for overnight jobs. B relies on "often faster" which isn't acceptable for blocking workflows. C reflects a misconception (`custom_id` correlates batch results). D adds unnecessary complexity when the simple answer is matching API to use case.

---

**Q12.** A PR modifies 14 files. Your single-pass review produces inconsistent depth, misses obvious bugs, and gives **contradictory feedback** — flagging a pattern as bad in one file while approving identical code elsewhere. How to restructure?

A) Split into focused passes: per-file local analysis + a separate cross-file integration pass.
B) Require developers to split large PRs into 3–4 file submissions.
C) Switch to a higher-tier model with a larger context window.
D) Run three independent passes and only flag issues appearing in 2+ runs.

**Correct: A.** Splitting into focused passes addresses **attention dilution** directly. File-by-file ensures consistent depth; integration pass catches cross-file issues. B shifts burden to developers. **C misunderstands that larger context doesn't fix attention quality.** D would actually *suppress* legitimate issues that only surface intermittently.

---

## 11. Preparation exercises

Anthropic recommends these four hands-on exercises. Each reinforces one or more domains.

### Exercise 1 — Build a Multi-Tool Agent with Escalation Logic

**Reinforces:** Domain 1, Domain 2, Domain 5

1. Define **3–4 MCP tools** with detailed descriptions clearly differentiating purpose, inputs, boundaries. Include at least two with similar functionality requiring careful description to avoid selection confusion.
2. Implement an **agentic loop** that checks `stop_reason` to decide whether to continue tool execution or present the final response. Handle both `"tool_use"` and `"end_turn"`.
3. Add **structured error responses**: `errorCategory` (transient/validation/permission), `isRetryable` boolean, human-readable descriptions. Verify the agent retries transients and explains business errors to users.
4. Implement a **programmatic hook** that intercepts tool calls to enforce a business rule (block ops above a threshold), redirecting to escalation.
5. Test with **multi-concern messages** and verify the agent decomposes, handles each concern, and synthesises a unified response.

### Exercise 2 — Configure Claude Code for a Team Development Workflow

**Reinforces:** Domain 3, Domain 2

1. Create a **project-level CLAUDE.md** with universal coding/testing standards. Verify project-level instructions apply across team members.
2. Create **`.claude/rules/`** files with frontmatter glob patterns (`paths: ["src/api/**/*"]` for API conventions, `paths: ["**/*.test.*"]` for testing). Test rules load only on matching files.
3. Create a **project-scoped skill** in `.claude/skills/` with `context: fork` and `allowed-tools` restrictions. Verify isolation.
4. Configure an **MCP server in `.mcp.json`** with env-var expansion. Add a personal experimental server in `~/.claude.json`. Verify both available simultaneously.
5. Test **plan mode vs direct execution**: a single-file bug fix, a library migration, a new feature with multiple valid approaches. Observe when plan mode adds value.

### Exercise 3 — Build a Structured Data Extraction Pipeline

**Reinforces:** Domain 4, Domain 5

1. Define an **extraction tool** with a JSON schema: required + optional fields, an enum with `"other"` + detail, nullable fields for absent info. Process documents missing some fields and verify the model returns null rather than fabricating.
2. Implement a **validation-retry loop**: when validation fails, send a follow-up with the document, the failed extraction, and the specific error. Track which errors are retry-resolvable (format mismatches) vs not (info absent).
3. Add **few-shot examples** for varied formats (inline citations vs bibliographies, narrative vs tables). Verify improved structural variety handling.
4. Design a **batch processing strategy**: submit 100 docs via Message Batches API, handle failures by `custom_id`, resubmit with chunking for oversized documents. Calculate total processing time vs SLA.
5. Implement **human review routing**: model outputs field-level confidence scores, route low-confidence to humans, analyse accuracy by document type and field.

### Exercise 4 — Design and Debug a Multi-Agent Research Pipeline

**Reinforces:** Domain 1, Domain 2, Domain 5

1. Build a **coordinator agent** delegating to ≥ 2 subagents (web search, document analysis). Ensure `allowedTools` includes `"Task"`; explicitly pass findings into each subagent's prompt.
2. Implement **parallel subagent execution** by emitting multiple `Task` calls in a single coordinator response. Measure the latency improvement vs sequential.
3. Design **structured output for subagents** separating content from metadata: claim, evidence excerpt, source URL/document name, publication date. Verify synthesis preserves source attribution.
4. **Simulate a subagent timeout** and verify the coordinator receives structured error context. Test that it proceeds with partial results and annotates coverage gaps.
5. Test with **conflicting sources** (two credible sources, different statistics) and verify synthesis preserves both with attribution rather than arbitrarily selecting, and structures the report distinguishing well-established from contested findings.

---

## 12. In-scope and out-of-scope topics

### In scope (per the official guide, verbatim)

- Agentic loop implementation: control flow on `stop_reason`, tool result handling, loop termination
- Multi-agent orchestration: coordinator-subagent, task decomposition, parallel subagent execution, iterative refinement loops
- Subagent context management: explicit context passing, structured state persistence, crash recovery using manifests
- Tool interface design: descriptions, splitting vs consolidating, naming to reduce ambiguity
- MCP tool and resource design: resources for content catalogs, tools for actions, description quality for adoption
- MCP server configuration: project vs user scope, env var expansion, multi-server simultaneous access
- Error handling and propagation: structured responses, transient/business/permission errors, local recovery before escalation
- Escalation decision-making: explicit criteria, honouring customer preferences, policy gap identification
- CLAUDE.md configuration: hierarchy, `@import`, `.claude/rules/` glob patterns
- Custom commands and skills: project vs user scope, `context: fork`, `allowed-tools`, `argument-hint`
- Plan mode vs direct execution: complexity assessment, architectural decisions
- Iterative refinement: input/output examples, test-driven, interview pattern, sequential vs parallel issue resolution
- Structured output via `tool_use`: schema design, `tool_choice`, nullable fields to prevent hallucination
- Few-shot prompting: ambiguous scenarios, format consistency, false-positive reduction
- Batch processing: Message Batches API appropriateness, latency tolerance, failure handling by `custom_id`
- Context window optimisation: trimming verbose tool outputs, structured fact extraction, position-aware ordering
- Human review workflows: confidence calibration, stratified sampling, accuracy segmentation
- Information provenance: claim-source mappings, temporal data, conflict annotation, coverage gap reporting

### NOT on the exam (verbatim)

These will **not** appear. Don't waste study time on them:

- Fine-tuning Claude or training custom models
- Claude API authentication, billing, account management
- Detailed implementation of specific languages/frameworks beyond what tool/schema config requires
- Deploying or hosting MCP servers (infrastructure, networking, container orchestration)
- Claude's internal architecture, training process, model weights
- Constitutional AI, RLHF, safety training
- Embedding models or vector DB implementation
- **Computer use** (browser automation, desktop interaction)
- **Vision / image analysis**
- **Streaming API implementation** or SSE
- Rate limiting, quotas, pricing calculations
- OAuth, API key rotation, auth protocols
- Specific cloud configurations (AWS, GCP, Azure)
- Performance benchmarks or model comparison metrics
- **Prompt caching implementation details** (you only need to know it exists)
- Token counting algorithms or tokenisation specifics

> **Big takeaway:** this is **not** a general "Claude API" exam. It's specifically about **Agent SDK + Claude Code + MCP + structured output**. If a topic isn't in those four buckets, it's almost certainly out of scope.

---

## 13. Anti-patterns checklist (distractor patterns)

Distractors on the exam follow predictable shapes. When you see one of these in an answer choice, treat it as a strong "probably wrong" signal:

| Distractor pattern | Why it's wrong |
|---|---|
| "Add a system prompt instruction that says ..." (when the requirement is *deterministic*) | Prompt instructions are probabilistic. Use hooks/programmatic enforcement instead. |
| "Set arbitrary iteration cap as the loop termination" | Use `stop_reason`. Iteration caps are a safety net, not the primary mechanism. |
| "Parse the assistant's text to determine completion" | Use `stop_reason`, not text parsing. |
| "Self-reported confidence score" / "sentiment analysis" for escalation | Both are unreliable proxies. Use explicit criteria + few-shot. |
| "Switch to a higher-tier model with a larger context window" | Larger context ≠ better attention. Address attention dilution by splitting passes. |
| "Use majority voting across N independent runs" | Suppresses legitimate intermittent findings. |
| "Have the agent batch up requests for end-of-pass" | Creates blocking dependencies between steps. |
| "Implement a routing classifier / separate ML model" | Over-engineered when prompt optimisation hasn't been tried. |
| "Catch the error and return empty success" | Silent suppression — coordinator can't recover. |
| "Terminate the entire workflow on subagent failure" | Lose partial results. |
| "Generic 'service unavailable' error" | Hides context needed for recovery. |
| "Start direct execution; switch to plan mode if it gets complex" | If the complexity is already known, start in plan mode. |
| "Consolidate two ambiguous tools into one generic tool" | First fix the descriptions; consolidation is bigger surgery. |
| "Add few-shot examples" (when the issue is *missing tool descriptions*) | Few-shot adds tokens without addressing the root cause. |
| "Increase batch frequency to make blocking workflows latency-tolerant" | Blocking ≠ batchable, ever. |
| "Give the agent more tools to be more capable" | Too many tools degrade selection. Principle of least privilege. |
| "`CLAUDE_HEADLESS=true`" / "`--batch` CLI flag" / "`.claude/config.json`" | These don't exist. Plausible-sounding but fictitious. |
| "User-level (`~/.claude/...`) for team-shared settings" | User-level isn't shared. Use project-level. |

---

## 14. Cheat sheet

### Domain weightings (memorise)
> **27% / 18% / 20% / 20% / 15%** — Agentic / Tools+MCP / Claude Code / Prompt+Output / Context

### The five domain mantras
- **Domain 1 — Agentic:** loop on `stop_reason`. Hooks for determinism, prompts for guidance.
- **Domain 2 — Tools/MCP:** descriptions are the selection mechanism. Least privilege.
- **Domain 3 — Claude Code:** project for shared, user for personal. Plan mode for architecture.
- **Domain 4 — Prompt/Output:** explicit criteria > vague. tool_use for guaranteed structure.
- **Domain 5 — Context:** facts persist outside summaries. Structured errors enable recovery.

### Stop reasons
> `"tool_use"` → execute tools, append, loop. `"end_turn"` → done. `"max_tokens"` → output cap hit. `"stop_sequence"` → custom stop matched.

### `tool_choice` modes
> `auto`, `any`, `tool`, `none`.

### Architectural triage order
1. **Right model layer**? (single agent / multi-agent / coordinator-subagent)
2. **Right tool descriptions**? (clear, differentiated, scoped)
3. **Deterministic where required**? (hooks, prerequisites)
4. **Structured outputs**? (tool_use + JSON schema)
5. **Structured errors**? (categories, retryable flags)
6. **Context preserved**? (case facts persisted, tool outputs trimmed)
7. **Escalation criteria explicit**? (with few-shot)
8. **Eval / sample / human review**? (stratified, segmented)

### Claude Code config locations
> User: `~/.claude/CLAUDE.md`, `~/.claude/commands/`, `~/.claude.json`. Project: `CLAUDE.md`, `.claude/rules/`, `.claude/commands/`, `.claude/skills/`, `.mcp.json`.

### CLI flags
> **`-p`** / **`--print`** non-interactive. **`--output-format json`** + **`--json-schema`** structured. **`--resume <name>`** continue session.

### Slash commands you must know
> **`/memory`** verify loaded files. **`/compact`** reduce context.

### Skill frontmatter you must know
> `context: fork` isolated context. `allowed-tools` restrict tool set. `argument-hint` prompt for args.

### Rule frontmatter
> `paths: [...]` glob list — load only when editing matching files.

### Batch API
> 50% off. ≤24h. **No multi-turn tool use.** `custom_id` correlation. **Never** for blocking workflows.

---

## 15. 5-day study plan

A focused plan rebalanced to the actual domain weightings.

### Day 1 — Domain 1 (27%) — Agentic Architecture & Orchestration
- Read §4 cover to cover. Re-read every "Skills in" bullet.
- Run **Exercise 1** end to end (multi-tool agent + escalation).
- Drill: write the agentic loop in pseudocode from memory.
- Re-do Sample Q1, Q7, Q8, Q9 explaining your reasoning out loud.

### Day 2 — Domain 3 (20%) + Domain 2 (18%) = 38% — Claude Code + Tools/MCP
- Read §6 (Claude Code config) and §5 (Tools/MCP).
- Run **Exercise 2** end to end (Claude Code team workflow).
- Memorise §9 (Claude commands & config reference) — file paths, flags, frontmatter.
- Re-do Sample Q2, Q4, Q5, Q6, Q10.

### Day 3 — Domain 4 (20%) — Prompt Engineering & Structured Output
- Read §7 cover to cover.
- Run **Exercise 3** end to end (extraction pipeline).
- Drill `tool_choice` modes and JSON-schema design from memory.
- Re-do Sample Q11, Q12.

### Day 4 — Domain 5 (15%) + Multi-agent depth — Context & Reliability
- Read §8 cover to cover.
- Run **Exercise 4** end to end (multi-agent research).
- Re-read the §13 anti-patterns checklist.
- Re-do Sample Q3.

### Day 5 — Mock exam + last-mile review
- Take the official **Practice Exam**. Aim for ≥ 900/1000.
- For every question you missed, write *why* the right answer is right and *why each distractor is wrong*. This is the highest-value study activity in the entire week.
- Re-read the **§14 cheat sheet** and **§13 anti-patterns**.
- One final pass of the six scenarios — say each one's primary domains out loud.

### Day 6 (exam morning)
- 15 min: read §14 cheat sheet.
- 15 min: skim §13 anti-patterns.
- 15 min: rehearse the six scenarios out loud.
- Then go take the exam.

---

## 16. Day-of checklist

- [ ] Recite the five domains and their weightings (27/18/20/20/15)
- [ ] List the four `stop_reason` values
- [ ] List the four `tool_choice` modes
- [ ] Define `context: fork`, `allowed-tools`, `argument-hint`
- [ ] Difference between `.claude/rules/`, `.claude/commands/`, `.claude/skills/`
- [ ] Difference between project-scope (`.mcp.json`) and user-scope (`~/.claude.json`)
- [ ] CLI flags: `-p`, `--output-format json`, `--json-schema`, `--resume`
- [ ] Slash commands: `/memory`, `/compact`
- [ ] Built-in tools: Read / Write / Edit / Bash / Grep / Glob / Task / Explore
- [ ] When to use programmatic enforcement vs prompts (deterministic vs probabilistic)
- [ ] When to use plan mode vs direct execution (architectural vs scoped)
- [ ] When to use Batch API vs synchronous (latency-tolerant vs blocking)
- [ ] The architectural triage order (8 steps)
- [ ] **Eat. Hydrate. Don't cram. Trust the prep.**

---

## 17. Final words

This exam tests **judgment under realistic constraints**. The questions you can't memorise — "what would you do given this messy production scenario" — are exactly what it's optimising for.

Two things separate the people who pass from the people who don't:

1. **Hands-on practice.** The four exercises in §11 aren't optional — they teach the muscle memory the exam grades. Do them.
2. **Pattern recognition on distractors.** §13 lists the wrong-answer shapes. Internalise them so when you see a "set environment variable that doesn't exist" or "use confidence scores to escalate," you spot it instantly.

Trust the prep. Walk in calm. Read each question twice — once to find what's being asked, once to spot the distractor pattern. Skip nothing (no penalty for guessing). Flag and return to anything you're not sure about.

You've got this. Good luck.

---

*Self-contained study guide. Built from the official CCA-F Exam Guide v0.1 (Feb 10 2025) and CCA-F FAQs.*
