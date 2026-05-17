"""Console entry point for the REST API — ``sdp-api`` / ``python -m sdp.api``."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the Synthetic Data Platform REST API via uvicorn."""
    parser = argparse.ArgumentParser(
        prog="sdp-api",
        description="Run the Synthetic Data Platform REST API.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev)")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (ignored with --reload)")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "uvicorn is not installed. Install the API extra:\n"
            "    pip install 'synthetic-data-platform[api]'\n"
        )
        return 1

    uvicorn.run(
        "sdp.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
