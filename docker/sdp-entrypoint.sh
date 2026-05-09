#!/bin/bash
# Entrypoint shim for the Synthetic Data Platform Docker image.
# Translates short verbs (streamlit, mcp, shell) to actual commands;
# everything else falls through to `python main.py <args>`.
set -e

case "${1:-help}" in
  streamlit)
    shift
    exec python -m streamlit run /app/ui/streamlit_app.py \
      --server.address=0.0.0.0 --server.port=8501 --server.headless=true "$@"
    ;;
  mcp)
    shift
    exec python -m mcp_server.server "$@"
    ;;
  shell)
    shift
    exec /bin/bash "$@"
    ;;
  help|--help|-h|"")
    cat <<'HELP'
Synthetic Data Platform — Docker entrypoint

Usage:
  docker run --rm -v "$PWD:/work" sdp <subcommand> [args...]

Subcommands:
  generate, delta, scd2, lint, ...   Any python main.py CLI subcommand
  streamlit                          Run the Streamlit UI on :8501
  mcp                                Run the MCP server (stdio)
  shell                              Drop into a bash shell

Examples:
  docker run --rm -v "$PWD:/work" sdp \
    generate --config /work/config.yaml --output /work/output --seed 42

  docker run --rm -p 8501:8501 -v "$PWD:/work" sdp streamlit

  docker run --rm -v "$PWD:/work" sdp \
    quality-report --generated /work/output --output-html /work/report.html

  docker run --rm -v "$PWD:/work" sdp \
    validate-data --config /work/config.yaml --input /work/output --verbose
HELP
    ;;
  *)
    # Anything else → treat as `python main.py <args>`
    exec python /app/main.py "$@"
    ;;
esac
