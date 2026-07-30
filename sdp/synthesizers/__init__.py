"""Pluggable generation engines.

    from sdp.synthesizers import create, describe

    engine = create("ctgan", seed=42, epochs=200)
    engine.fit(sample_data, metadata)
    data = engine.sample({"customers": 1000})
    print(engine.stats.to_dict())

See ``base.py`` for the contract and ``registry.py`` for registration.
"""
from sdp.synthesizers.base import EngineStats, Synthesizer, sample_multi_table
from sdp.synthesizers.registry import (
    DEFAULT_ENGINE,
    create,
    describe,
    get,
    names,
    register,
    unregister,
)

__all__ = [
    "DEFAULT_ENGINE",
    "EngineStats",
    "Synthesizer",
    "create",
    "describe",
    "get",
    "names",
    "register",
    "sample_multi_table",
    "unregister",
]
