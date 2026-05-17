"""Synthetic Data Platform — relationship-aware synthetic data generation.

The ``sdp`` package bundles the full platform: data generation, config parsing,
delta/SCD2 processing, relationship inference, API mocks, the CLI, a programmatic
SDK facade (:mod:`sdp.sdk`) and a REST API (:mod:`sdp.api`).

Quick start (SDK):

    from sdp import SyntheticDataPlatform

    sdp = SyntheticDataPlatform()
    result = sdp.generate(config="config/Acct_bkng.xlsx", output="output/run_01", seed=42)
"""
from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__", "SyntheticDataPlatform", "GenerationResult"]


def __getattr__(name: str):
    # Lazy re-export so `import sdp` stays cheap and free of heavy imports.
    if name in ("SyntheticDataPlatform", "GenerationResult"):
        from sdp.sdk import GenerationResult, SyntheticDataPlatform

        return {"SyntheticDataPlatform": SyntheticDataPlatform,
                "GenerationResult": GenerationResult}[name]
    raise AttributeError(f"module 'sdp' has no attribute {name!r}")
