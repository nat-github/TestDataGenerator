#!/usr/bin/env python3
"""Backward-compatible CLI entry point.

The CLI implementation now lives in :mod:`sdp.cli`. This shim keeps
``python main.py ...`` and ``from main import <name>`` working unchanged after
the package restructure. The installed console script is ``sdp`` (see
pyproject.toml ``[tool.poetry.scripts]``).
"""
from __future__ import annotations

import sys

# Re-export the full public CLI surface so existing `from main import X` callers
# (and tests) keep resolving against the moved module.
from sdp.cli import *  # noqa: F401,F403
from sdp.cli import build_parser, main, parse_arguments  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
