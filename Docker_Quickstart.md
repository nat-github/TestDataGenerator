# Docker Quickstart

Run the platform without installing Python, Poetry, or any of the heavy
dependencies (SDV, Great Expectations, Streamlit, MCP) on your host. The
Docker path is **purely additive** — the local Poetry workflow continues
to work unchanged.

> **Use Docker when:** you don't want to install Python locally, you're on a
> shared / CI machine, or you want a reproducible runtime.
> **Use Poetry when:** you want fastest iteration, IDE integration, and the
> minimum disk footprint.

---

## Prerequisites

- **Docker 20.10+** (older 19.x works but won't support `docker compose` —
  use the standalone `docker-compose` binary instead).
- (Optional) **Docker Compose v2** if you want the `docker compose up`
  shortcut.

---

## Build the image

From the repo root:

```bash
docker build -t sdp:latest .
```

First build is slow (~5–10 min) because it installs SDV, Great Expectations,
Streamlit, MCP SDK, scipy, scikit-learn, and friends. Subsequent builds
are fast (cached unless `pyproject.toml` / `poetry.lock` change).

The final image is ~1 GB; the bulk is SDV + scipy + pandas. To trim, drop
extras you don't need from the Dockerfile's `poetry install --extras`
line.

---

## Run modes

The image has a small entrypoint shim so verbs are short.

### 1. CLI — one-shot subcommand

```bash
# Mount the repo (or any directory with your configs) at /work
docker run --rm -v "$PWD:/work" sdp:latest \
  generate --config /work/examples/configs/yaml/01_simple_users.yaml \
           --output /work/output --default-records 200 --seed 42
```

Anything you'd type after `python main.py` works the same way after
`docker run ... sdp:latest`.

### 2. Streamlit UI — long-running

```bash
docker run --rm -p 8501:8501 -v "$PWD:/work" sdp:latest streamlit
# then open http://localhost:8501
```

The `-p 8501:8501` publishes the container's Streamlit port. Generated
data and uploaded configs go into the mounted `/work` volume so they
survive the container exit.

### 3. MCP server — stdio (advanced)

```bash
docker run --rm -i -v "$PWD:/work" sdp:latest mcp
```

In practice you wire this directly into your AI client's MCP config,
not run it by hand. See `MCP_Integration.md` for the wiring.

### 4. Drop into a shell

```bash
docker run --rm -it -v "$PWD:/work" sdp:latest shell
```

Useful for poking around inside the image — debug missing files,
inspect installed deps, etc.

---

## docker-compose

A `docker-compose.yml` is included for convenience. Two services share
the same image:

```bash
# Build once
docker compose build

# CLI as a one-shot run
docker compose run --rm cli \
  generate --config /work/examples/configs/yaml/02_ecommerce_relationships.yaml \
           --output /work/output --default-records 500 --seed 7

# Streamlit UI
docker compose up streamlit
# Ctrl-C to stop. http://localhost:8501 while it's up.
```

> **Older Docker (19.x):** use `docker-compose` (hyphen) instead of
> `docker compose` (space).

---

## Volume mounts — getting data in and out

The image declares `/work` as a volume. Mount your local working
directory there:

```bash
-v "$PWD:/work"             # whole repo (Linux/macOS)
-v "%CD%:/work"             # whole repo (Windows cmd)
-v "${PWD}:/work"           # whole repo (PowerShell)
-v "/some/configs:/work"    # just a configs dir
```

**Inside the container, paths must be `/work/...`**, not your host paths.
Example: a config at `./examples/configs/yaml/01_simple_users.yaml` on
the host is `/work/examples/configs/yaml/01_simple_users.yaml` inside the
container.

---

## Common workflows

### Generate + validate + quality report — one volume mount

```bash
docker run --rm -v "$PWD:/work" sdp:latest \
  generate --config /work/examples/configs/yaml/02_ecommerce_relationships.yaml \
           --output /work/output --default-records 500 --seed 7 \
           --validate --validate-with-gx

docker run --rm -v "$PWD:/work" sdp:latest \
  quality-report --generated /work/output \
                 --output-html /work/quality.html \
                 --output-json /work/quality.json
```

Open `quality.html` in your browser; ingest `quality.json` into a
dashboard.

### Mocks rendering pipeline

```bash
docker run --rm -v "$PWD:/work" sdp:latest \
  mock-init --from /work/examples/openapi/medium_tasks.yaml \
            --output /work/mocks/tasks.yaml

docker run --rm -v "$PWD:/work" sdp:latest \
  mock-render --config /work/mocks/tasks.yaml \
              --output /work/stubs/tasks \
              --format wiremock,json,pact,postman \
              --examples 5 --seed 42 --match-mode any
```

Then `java -jar wiremock.jar --root-dir stubs/tasks/wiremock --port 8081`
on the host (or stand up another container) to actually serve the stubs.

---

## Side-by-side: Docker vs local Poetry

| Task | Docker | Local Poetry |
|---|---|---|
| **Install** | `docker build -t sdp:latest .` (one-time, ~5–10 min) | `poetry install --extras "gx ui mcp mimesis"` (one-time, ~3–5 min) |
| **Generate data** | `docker run --rm -v "$PWD:/work" sdp:latest generate ...` | `python main.py generate ...` |
| **Streamlit UI** | `docker run --rm -p 8501:8501 -v "$PWD:/work" sdp:latest streamlit` | `streamlit run ui/streamlit_app.py` |
| **MCP wiring** | Reference the docker run command in your AI-client's MCP config | Reference your venv's Python in your AI-client's MCP config |
| **Disk footprint** | ~1 GB image + your data | ~1.5 GB venv + your data |
| **Iteration speed** | Slower (rebuild on dep changes) | Fastest |
| **Reproducibility** | Highest (pinned Python version, OS libs) | High (Poetry lock) |

Both are first-class. Pick what fits the box you're on.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `docker: 'compose' is not a docker command` | Old Docker (<20.10). Use `docker-compose` (hyphen) instead, or upgrade. |
| Build dies in `poetry install` | Network issue inside the build, or wheel missing for ARM. Try `--platform linux/amd64`. |
| Streamlit at localhost:8501 returns 404 | Container is bound to `0.0.0.0`. Confirm `-p 8501:8501` was on the run command. |
| `Permission denied` writing to /work | Host directory is owned by a UID/GID the container's `sdp` user can't write to. Either `chmod -R 777` the directory, or run with `--user "$(id -u):$(id -g)"`. |
| Image is too big | Drop extras you don't use from the Dockerfile's `poetry install --extras` line. The mocks/UI/MCP/GX/mimesis combo is ~600 MB of deps. |
| Tests don't run inside the image | The image installs `--without dev`, so pytest isn't available. Build a separate dev image: `RUN poetry install --extras "..."` (drop `--without dev`). |

---

## When to drop back to local Poetry

- You're iterating on platform code and want sub-second feedback
- You need IDE features (debugger, type-checker) hooked into the venv
- You're running long agent loops via MCP — the docker stdio path adds
  ~1s per tool call
- You want to extend the platform itself (new MCP tools, new renderers)

---

## What's not in the image

These are intentionally **not** baked in — install on the host as needed:

- **WireMock standalone** (Java app) — install separately, point `--root-dir`
  at the rendered stubs
- **Bruno / Postman** — desktop apps; the platform produces collections
  they import
- **LM Studio / Ollama** — separate installs; the platform's MCP server
  in the container connects out to them via `host.docker.internal`
  (Mac/Win) or the host gateway (Linux)
- **Java runtime** — only needed for WireMock; not a platform dep
