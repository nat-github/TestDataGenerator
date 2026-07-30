"""Engine registry — name → synthesizer class.

Entries are registered lazily by import path so that adding an engine never
costs import time for runs that do not use it: resolving ``"ctgan"`` must
not drag torch into a rule-based run.

Third-party engines register themselves the same way::

    from sdp.synthesizers.registry import register
    register("my-engine", "my_package.engines:MyEngine")
"""
from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Tuple, Type

from sdp.synthesizers.base import Synthesizer

logger = logging.getLogger(__name__)

# name → "module.path:ClassName", resolved on first use.
_REGISTRY: Dict[str, str] = {}

# Resolved classes, cached after the first import.
_RESOLVED: Dict[str, Type[Synthesizer]] = {}

#: Engine used when the config names none. Preserves historical behaviour.
DEFAULT_ENGINE = "sdv"


def register(name: str, target: str) -> None:
    """Register ``name`` against a ``"module:Class"`` path."""
    _REGISTRY[name] = target
    _RESOLVED.pop(name, None)


def unregister(name: str) -> None:
    """Remove an engine. Mainly for tests."""
    _REGISTRY.pop(name, None)
    _RESOLVED.pop(name, None)


def names() -> List[str]:
    """Every registered engine name, sorted."""
    return sorted(_REGISTRY)


def get(name: str) -> Type[Synthesizer]:
    """Resolve an engine class by name.

    Raises ``KeyError`` with the list of valid names on an unknown engine,
    and ``ImportError`` when the target cannot be imported — both are
    configuration errors worth failing loudly on.
    """
    if name in _RESOLVED:
        return _RESOLVED[name]

    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown synthesizer engine {name!r}. Available: {', '.join(names())}"
        )

    target = _REGISTRY[name]
    module_path, _, class_name = target.partition(":")
    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    _RESOLVED[name] = engine_cls
    return engine_cls


def create(
    name: Optional[str] = None,
    *,
    seed: Optional[int] = None,
    **options,
) -> Synthesizer:
    """Instantiate an engine by name (``DEFAULT_ENGINE`` when omitted)."""
    engine_cls = get(name or DEFAULT_ENGINE)
    if not engine_cls.is_available():
        raise RuntimeError(
            f"Synthesizer engine {engine_cls.name!r} is registered but its "
            f"dependencies are not installed."
        )
    return engine_cls(seed=seed, **options)


def describe() -> List[Tuple[str, str, bool]]:
    """``(name, description, available)`` for every engine.

    Availability is probed defensively: a broken third-party engine should
    show as unavailable, not break the listing.
    """
    out: List[Tuple[str, str, bool]] = []
    for name in names():
        try:
            engine_cls = get(name)
            out.append((name, engine_cls.description, engine_cls.is_available()))
        except Exception as exc:
            logger.debug("engine %s failed to resolve: %s", name, exc)
            out.append((name, f"(failed to load: {exc})", False))
    return out


# --- built-in engines --------------------------------------------------
# Registered by path, so none of these modules is imported until used.

register("sdv", "sdp.synthesizers.sdv_hma:HMAEngine")
register("gaussian-copula", "sdp.synthesizers.single_table:GaussianCopulaEngine")
register("ctgan", "sdp.synthesizers.single_table:CTGANEngine")
register("tvae", "sdp.synthesizers.single_table:TVAEEngine")
register("rule-based", "sdp.synthesizers.rule_based:RuleBasedEngine")
