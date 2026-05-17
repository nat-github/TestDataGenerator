"""REST API for the Synthetic Data Platform.

A thin HTTP layer over :class:`sdp.sdk.SyntheticDataPlatform`. Requires the
``api`` extra::

    pip install "synthetic-data-platform[api]"

Run it::

    sdp-api --host 0.0.0.0 --port 8000
    # or
    uvicorn sdp.api:app --reload

Interactive docs are served at ``/docs``.
"""
from __future__ import annotations

from sdp.api.app import app, create_app

__all__ = ["app", "create_app"]
