"""Pluggable semantic vocabulary for knowledge-graph relationship inference.

WHY THIS MODULE EXISTS
----------------------
The knowledge-graph inferrer has two kinds of logic:

* **Domain-agnostic graph reasoning** — inclusion-dependency edges, hub/degree
  priors, global one-parent assignment and FK-cycle resolution. None of this
  needs to know which industry the schema came from.
* **Name-based semantic matching** — the *only* part that benefits from knowing
  that ``acct`` means ``account`` or ``ccy`` means ``currency``.

Previously that domain vocabulary was hard-coded inside the inferrer, which made
the whole feature look banking-specific. It now lives here as a swappable
``SemanticProfile`` so a new deployment can extend or replace it
(``SDP_SEMANTIC_PROFILE=/path/to/profile.yaml``) without touching inference
code, and the graph machinery stays generic.

Load order (later layers win on key collisions):

  1. built-in ``GENERIC_PROFILE`` — universal abbreviations
  2. built-in ``BANKING_PROFILE`` — card/clearing vocabulary (the platform's
     bundled example configs are card-clearing schemas)
  3. optional user YAML via the ``SDP_SEMANTIC_PROFILE`` env var

The merged default is byte-for-byte equivalent to the vocabulary the inferrer
used before this module existed — so behaviour is unchanged out of the box.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

ENV_PROFILE = "SDP_SEMANTIC_PROFILE"


@dataclass
class SemanticProfile:
    """A swappable bundle of abbreviation / stopword knowledge for name matching.

    Attributes
    ----------
    aliases:
        Maps an abbreviation token to its canonical form (``acct`` -> ``account``).
    stopwords:
        Tokens that carry no entity meaning and are dropped before matching
        (``id``, ``code``, ``tbl``, ...).
    generic_key_names:
        Single-token names that, on their own, identify nothing in particular
        (``id``, ``code``, ``key``). A parent PK called exactly one of these is
        treated as ambiguous unless semantics disambiguate it.
    """

    aliases: Dict[str, str] = field(default_factory=dict)
    stopwords: Set[str] = field(default_factory=set)
    generic_key_names: Set[str] = field(default_factory=set)

    def merged_with(self, other: "SemanticProfile") -> "SemanticProfile":
        """Return a new profile with ``other`` layered on top of ``self``."""
        merged = SemanticProfile(
            aliases=dict(self.aliases),
            stopwords=set(self.stopwords),
            generic_key_names=set(self.generic_key_names),
        )
        merged.aliases.update(other.aliases)
        merged.stopwords |= other.stopwords
        merged.generic_key_names |= other.generic_key_names
        return merged

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticProfile":
        """Build a profile from a parsed YAML/JSON mapping."""
        return cls(
            aliases={
                str(k).strip().lower(): str(v).strip().lower()
                for k, v in (data.get("aliases") or {}).items()
            },
            stopwords={str(w).strip().lower() for w in (data.get("stopwords") or [])},
            generic_key_names={
                str(w).strip().lower() for w in (data.get("generic_key_names") or [])
            },
        )


# Universal abbreviations — true across virtually every business domain.
GENERIC_PROFILE = SemanticProfile(
    aliases={
        "acct": "account",
        "ac": "account",
        "pd": "product",
        "tp": "type",
        "trns": "transaction",
        "txn": "transaction",
        "cat": "category",
        "src": "source",
        "svc": "service",
    },
    stopwords={
        "id", "key", "code", "codes", "number", "num", "no", "ref",
        "fk", "pk", "tbl", "table", "row", "data", "record", "seq",
    },
    generic_key_names={
        "id", "key", "code", "number", "num", "no", "identifier", "identification",
    },
)

# Banking / cards domain vocabulary. Bundled because the platform's primary
# example configs are card-clearing schemas; harmless for other domains and
# fully overridable via SDP_SEMANTIC_PROFILE.
BANKING_PROFILE = SemanticProfile(
    aliases={
        "trmnl": "terminal",
        "mrch": "merchant",
        "ccy": "currency",
        "curr": "currency",
        "issur": "issuer",
        "issuer": "issuer",
        "bnk": "bank",
        "identn": "identification",
        "identif": "identification",
        "ident": "identification",
        "inpt": "input",
        "orig": "original",
        "ntw": "network",
        "atm": "atm",
        "pos": "pos",
        "iso": "iso",
        "isocurrencies": "currency",
        "currencies": "currency",
        "n3": "numericcode",
        "numericcode": "numericcode",
        "isocurrencycode": "isocurrencycode",
        "bindescription": "bankidentificationdescription",
        "bin": "bankidentificationnumber",
        "merchantcategory": "merchantcategory",
        "merchantgroup": "merchantgroup",
    },
    stopwords={
        "ebx", "dc", "dl", "hdr", "header", "clrg", "clearing",
        "card", "cards", "trx", "txnseq", "sqn",
    },
    generic_key_names=set(),
)


def _load_user_profile() -> Optional[SemanticProfile]:
    """Load the optional user profile pointed to by ``SDP_SEMANTIC_PROFILE``."""
    path = os.environ.get(ENV_PROFILE)
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("semantic_profile: %s=%s not found; ignoring", ENV_PROFILE, path)
        return None
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        logger.info("semantic_profile: loaded user vocabulary from %s", path)
        return SemanticProfile.from_dict(data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("semantic_profile: failed to load %s: %s", path, exc)
        return None


def default_profile() -> SemanticProfile:
    """Return the merged default vocabulary (generic + banking + optional user).

    The generic layer alone is sufficient for most schemas; the banking layer is
    bundled so the platform's card-clearing example configs keep working out of
    the box. A user YAML referenced by ``SDP_SEMANTIC_PROFILE`` is merged last
    and wins on key collisions.
    """
    profile = GENERIC_PROFILE.merged_with(BANKING_PROFILE)
    user = _load_user_profile()
    if user is not None:
        profile = profile.merged_with(user)
    return profile
