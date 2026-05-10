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

## Onboarding — never used Docker before?

This section is for people doing this from scratch on a fresh laptop.
**Skip to "[Prerequisites](#prerequisites)" if Docker is already on your
machine.**

### What Docker is, in one paragraph

Docker packages an application together with everything it needs to run
(Python, libraries, OS-level tooling) into a single self-contained
**image**. You then run **containers** from that image. The container is
isolated from your host — no Python install, no dependency conflicts, no
"works on my machine." Anyone with Docker can run our exact build with
`docker run`.

### Step 1 — install Docker Desktop

Pick the right installer for your OS. Docker Desktop is the GUI + CLI
combo most users want.

| OS | Where to get it | Notes |
|---|---|---|
| **Windows 10/11** | Docker Desktop for Windows from the Docker website | Needs WSL 2 (the installer prompts you). Pro / Enterprise / Education editions for free; some companies require a paid licence — check with your employer. |
| **macOS** | Docker Desktop for Mac (separate installers for Intel and Apple Silicon) | Free for personal use. |
| **Linux (Ubuntu / Debian / Fedora / etc.)** | Either Docker Desktop *or* Docker Engine via your package manager | Docker Engine alone is enough; Docker Desktop adds the GUI. |

After install:
- **Windows:** the installer will ask to enable WSL 2 and may need a reboot.
- **Mac:** open Docker Desktop from Applications. Wait for the whale icon in the menu bar to settle to "Running."
- **Linux (Engine only):** add your user to the `docker` group so you don't need `sudo`:
  `sudo usermod -aG docker "$USER"` then log out and back in.

### Step 2 — verify Docker works

Open a new terminal and run:

```bash
docker --version
docker run --rm hello-world
```

Expected:
- `docker --version` prints something like `Docker version 24.0.7, build ...`
- `docker run hello-world` pulls a tiny image and prints "Hello from Docker!"

If either fails, Docker Desktop probably isn't running. On Windows / Mac,
launch Docker Desktop and wait for it to finish starting.

### Step 3 — clone or copy this repo

```bash
git clone <your-repo-url> TestDataGeneration
cd TestDataGeneration
```

(If you already have the repo, just `cd` into it.)

### Step 4 — build the platform image

From the repo root:

```bash
docker build -t sdp:latest .
```

This is a **one-time** step. First run takes 5–10 minutes because it
installs SDV, Great Expectations, Streamlit, the MCP SDK, and friends.
Subsequent builds are seconds (Docker caches layers; only changes to
`pyproject.toml` / `poetry.lock` / source bust the cache).

You'll see a lot of output — that's normal. Successful build ends with:

```
Successfully tagged sdp:latest
```

Verify the image exists:

```bash
docker images | grep sdp
# sdp   latest   <hash>   <size>
```

### Step 5 — first run: launch the Streamlit UI

```bash
# Linux / Mac
docker run --rm -p 8501:8501 -v "$PWD:/work" sdp:latest streamlit

# Windows PowerShell
docker run --rm -p 8501:8501 -v "${PWD}:/work" sdp:latest streamlit

# Windows cmd
docker run --rm -p 8501:8501 -v "%CD%:/work" sdp:latest streamlit
```

Wait for the line `You can now view your Streamlit app in your browser`,
then open **http://localhost:8501**. You should see the Synthetic Data
Platform UI with the **Generate Data** and **API Mocks** pages in the
sidebar. Press `Ctrl-C` in the terminal to stop the server.

### Step 6 — run a CLI command

```bash
# Linux / Mac
docker run --rm -v "$PWD:/work" sdp:latest \
  generate --config /work/examples/configs/yaml/01_simple_users.yaml \
           --output /work/output --default-records 200 --seed 42

# Windows PowerShell
docker run --rm -v "${PWD}:/work" sdp:latest `
  generate --config /work/examples/configs/yaml/01_simple_users.yaml `
           --output /work/output --default-records 200 --seed 42
```

Check the result on your host (the `output/` directory inside the repo
will now contain `users.parquet`).

> **You're up.** Skip ahead to [Common workflows](#common-workflows) for
> end-to-end recipes, or [docker-compose](#docker-compose) if you'd
> rather use the shorthand.

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
