# Synthetic Data Platform — multi-purpose image
#
# Single image runs:
#   - the CLI:        docker run --rm -v "$PWD:/work" sdp generate --config /work/X.yaml --output /work/out
#   - Streamlit UI:   docker run --rm -p 8501:8501 -v "$PWD:/work" sdp streamlit
#   - the MCP server: docker run --rm -i -v "$PWD:/work" sdp mcp
#
# Local Poetry path is unaffected — using Docker is purely additive.
#
# Build:
#   docker build -t sdp:latest .
#
# All optional extras (gx, ui, mcp, mimesis) are installed so a single image
# covers every feature. For a leaner image, drop the unused extras from the
# `poetry install` line below.

# ---------------------------------------------------------------------------
# Builder stage — install deps into a venv we copy out
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tools needed for some wheels (cffi, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy only dependency manifests first so layer caching works
COPY pyproject.toml poetry.lock ./

# Install runtime deps + every optional extra into ./.venv
RUN poetry install \
        --extras "gx ui mcp mimesis" \
        --no-root \
        --without dev

# Copy the source after deps so source edits don't bust the dep cache layer
COPY . .

# Install the project itself (deps already in .venv)
RUN poetry install --extras "gx ui mcp mimesis" --without dev


# ---------------------------------------------------------------------------
# Runtime stage — slim, no build tools
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Minimal runtime libs (libgomp for scipy/sklearn on some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for safety
RUN useradd --create-home --shell /bin/bash sdp

WORKDIR /app

# Copy the populated venv + the source from the builder
COPY --from=builder --chown=sdp:sdp /app /app

# Entrypoint shim — translates verbs (streamlit / mcp / shell) to commands;
# everything else falls through to `python main.py <args>`.
COPY --chown=root:root docker/sdp-entrypoint.sh /usr/local/bin/sdp-entrypoint.sh
RUN chmod +x /usr/local/bin/sdp-entrypoint.sh

USER sdp

# Volume for configs in / data out — mount your local directory here
VOLUME ["/work"]
WORKDIR /work

# Streamlit default port (only used by the streamlit subcommand)
EXPOSE 8501

ENTRYPOINT ["sdp-entrypoint.sh"]
CMD ["help"]
