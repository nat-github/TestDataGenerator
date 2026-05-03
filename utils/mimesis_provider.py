"""Mimesis adapter — handles `MIMESIS_*` special rules.

Mimesis (https://github.com/lk-geimfari/mimesis) is a lightweight, fast
synthetic-data library that complements Faker. It's added as an *optional*
dependency: configs that don't use `MIMESIS_*` rules don't pay any cost,
and users who haven't installed Mimesis get a clear actionable error only
when they try to use one of the new rules.

Authoring format:
    MIMESIS_NAME              -> Person.full_name(), default locale
    MIMESIS_FIRST_NAME:de_DE  -> Person.first_name() with German locale
    MIMESIS_EMAIL:fr_FR       -> Person.email() with French locale

Existing rules (NAME, FIRST_NAME, EMAIL, ...) are unaffected — they keep
routing to Faker. The two stacks live side by side.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded module references. Filled on first call to _ensure_mimesis().
_mimesis_module = None
_locale_enum = None
_generic_cls = None
_import_error: Optional[Exception] = None


# ---------------------------------------------------------------------------
# Locale mapping — translate Faker-style "de_DE" to Mimesis's Locale enum.
# Mimesis uses two-letter codes (with a few exceptions like pt-br), so we
# reduce the Faker locale to its language part and look it up.
# ---------------------------------------------------------------------------

# Faker locale -> Mimesis locale string (matches Mimesis Locale enum values)
_LOCALE_MAP: Dict[str, str] = {
    "en": "en", "en_US": "en", "en_GB": "en-gb",
    "de": "de", "de_DE": "de", "de_AT": "de-at", "de_CH": "de-ch",
    "fr": "fr", "fr_FR": "fr",
    "es": "es", "es_ES": "es", "es_MX": "es-mx",
    "it": "it", "it_IT": "it",
    "pt": "pt", "pt_PT": "pt", "pt_BR": "pt-br",
    "nl": "nl", "nl_NL": "nl", "nl_BE": "nl-be",
    "ru": "ru", "ru_RU": "ru",
    "ja": "ja", "ja_JP": "ja",
    "ko": "ko", "ko_KR": "ko",
    "zh": "zh", "zh_CN": "zh", "zh_TW": "zh",
    "pl": "pl", "pl_PL": "pl",
    "sv": "sv", "sv_SE": "sv",
    "tr": "tr", "tr_TR": "tr",
    "uk": "uk", "uk_UA": "uk",
    "cs": "cs", "cs_CZ": "cs",
    "da": "da", "da_DK": "da",
    "et": "et", "et_EE": "et",
    "fa": "fa", "fa_IR": "fa",
    "is": "is", "is_IS": "is",
    "fi": "fi", "fi_FI": "fi",
    "no": "no", "no_NO": "no",
    "hu": "hu", "hu_HU": "hu",
}


# ---------------------------------------------------------------------------
# Lazy import — Mimesis is optional, so we only import when needed.
# ---------------------------------------------------------------------------


def _ensure_mimesis() -> bool:
    """Import Mimesis on first use. Returns True on success.

    Caches both the success path and the failure path so we don't re-attempt
    the import on every call (cheap, but cleaner logs).
    """
    global _mimesis_module, _locale_enum, _generic_cls, _import_error
    if _mimesis_module is not None:
        return True
    if _import_error is not None:
        return False
    try:
        import mimesis as _mimesis  # type: ignore
        from mimesis import Generic  # type: ignore
        from mimesis.locales import Locale  # type: ignore
        _mimesis_module = _mimesis
        _generic_cls = Generic
        _locale_enum = Locale
        return True
    except ImportError as exc:
        _import_error = exc
        return False


def is_available() -> bool:
    """Cheap check used by tests. Triggers the lazy import."""
    return _ensure_mimesis()


# ---------------------------------------------------------------------------
# Locale helpers
# ---------------------------------------------------------------------------


def _resolve_locale(locale_str: Optional[str]) -> Any:
    """Translate a Faker-style locale into a Mimesis Locale enum member.

    Falls back to Locale.EN when the requested locale isn't in the map. Returns
    None if Mimesis isn't importable (caller must check is_available first).
    """
    if not _ensure_mimesis():
        return None
    if not locale_str:
        return _locale_enum.EN

    key = str(locale_str).strip()
    mapped = _LOCALE_MAP.get(key) or _LOCALE_MAP.get(key.lower()) or _LOCALE_MAP.get(key.split("_")[0])
    if not mapped:
        return _locale_enum.EN

    # Mimesis stores locale codes as the enum value (e.g. "de", "pt-br").
    # Walk the enum to find the matching member.
    for member in _locale_enum:
        if str(member.value).lower() == mapped.lower():
            return member
    return _locale_enum.EN


_generic_cache: Dict[str, Any] = {}


def _get_generic(locale_str: Optional[str]) -> Any:
    """Cached Generic provider per locale — Mimesis providers are cheap to
    instantiate but caching keeps the hot path tight."""
    if not _ensure_mimesis():
        return None
    key = locale_str or "default"
    if key not in _generic_cache:
        loc = _resolve_locale(locale_str)
        _generic_cache[key] = _generic_cls(locale=loc)
    return _generic_cache[key]


# ---------------------------------------------------------------------------
# Rule registry — maps the suffix after `MIMESIS_` to a callable.
# Each callable takes the Generic provider and returns a value.
# Adding a new rule is one line below — no other code changes.
# ---------------------------------------------------------------------------

_RULES: Dict[str, Callable[[Any], Any]] = {
    # People
    "NAME":         lambda g: g.person.full_name(),
    "FULL_NAME":    lambda g: g.person.full_name(),
    "FIRST_NAME":   lambda g: g.person.first_name(),
    "LAST_NAME":    lambda g: g.person.last_name(),
    "USERNAME":     lambda g: g.person.username(),
    "EMAIL":        lambda g: g.person.email(),
    "PHONE":        lambda g: g.person.telephone(),
    "GENDER":       lambda g: g.person.gender(),
    "TITLE":        lambda g: g.person.title(),
    "OCCUPATION":   lambda g: g.person.occupation(),
    "NATIONALITY":  lambda g: g.person.nationality(),

    # Address
    "ADDRESS":      lambda g: g.address.address(),
    "CITY":         lambda g: g.address.city(),
    "STATE":        lambda g: g.address.state(),
    "COUNTRY":      lambda g: g.address.country(),
    "COUNTRY_CODE": lambda g: g.address.country_code(),
    "POSTCODE":     lambda g: g.address.postal_code(),
    "ZIP":          lambda g: g.address.zip_code() if hasattr(g.address, "zip_code") else g.address.postal_code(),
    "STREET":       lambda g: g.address.street_name(),
    "STREET_NUMBER": lambda g: g.address.street_number(),
    "LATITUDE":     lambda g: g.address.latitude(),
    "LONGITUDE":    lambda g: g.address.longitude(),

    # Finance / payment — note: IBAN/BIC/SWIFT are NOT in Mimesis; use the
    # existing Faker-backed `IBAN`, `BIC`, `SWIFT` rules for those.
    "COMPANY":      lambda g: g.finance.company(),
    "CURRENCY":     lambda g: g.finance.currency_iso_code(),
    "PRICE":        lambda g: g.finance.price(),
    "STOCK_TICKER": lambda g: g.finance.stock_ticker(),
    "CREDIT_CARD":  lambda g: g.payment.credit_card_number(),
    "CC_EXP":       lambda g: g.payment.credit_card_expiration_date(),
    "CVV":          lambda g: g.payment.cvv(),

    # Internet / tech
    "URL":          lambda g: g.internet.url(),
    "IPV4":         lambda g: g.internet.ip_v4(),
    "IPV6":         lambda g: g.internet.ip_v6(),
    "MAC":          lambda g: g.internet.mac_address(),
    "USER_AGENT":   lambda g: g.internet.user_agent(),

    # Cryptographic / identifiers
    "UUID":         lambda g: g.cryptographic.uuid(),
    "TOKEN":        lambda g: g.cryptographic.token_hex(),
    "HASH":         lambda g: g.cryptographic.hash(),

    # Text
    "WORD":         lambda g: g.text.word(),
    "SENTENCE":     lambda g: g.text.sentence(),
    "TEXT":         lambda g: g.text.text(),
    "QUOTE":        lambda g: g.text.quote(),
    "COLOR":        lambda g: g.text.color(),

    # Datetime
    "DATE":         lambda g: g.datetime.date().isoformat(),
    "TIME":         lambda g: g.datetime.time().isoformat(),
    "DATETIME":     lambda g: g.datetime.datetime().isoformat(),
    "TIMEZONE":     lambda g: g.datetime.timezone(),
}


# ---------------------------------------------------------------------------
# Public entry point — called from helpers.generate_special_value
# ---------------------------------------------------------------------------


class MimesisNotInstalledError(ImportError):
    """Raised when a MIMESIS_* rule is requested but the library isn't installed."""


def generate(rule_suffix: str, locale: Optional[str] = None) -> Any:
    """Resolve a MIMESIS_<rule_suffix>[:<locale>] special rule into a value.

    Args:
        rule_suffix: The part after `MIMESIS_`, e.g. `NAME`, `FIRST_NAME`,
            `EMAIL`. Case-insensitive.
        locale: Optional Faker-style locale string (`de_DE`, `fr_FR`, `ja_JP`).

    Returns:
        A generated value, or None if the rule isn't recognised.

    Raises:
        MimesisNotInstalledError: when the library isn't installed.
    """
    if not _ensure_mimesis():
        raise MimesisNotInstalledError(
            "Mimesis is not installed. Install it with: "
            "poetry install --extras mimesis  (or: pip install mimesis>=17)"
        )

    if not rule_suffix:
        return None

    key = str(rule_suffix).strip().upper()
    fn = _RULES.get(key)
    if fn is None:
        logger.warning("mimesis_provider: unknown rule MIMESIS_%s", key)
        return None

    try:
        return fn(_get_generic(locale))
    except Exception as exc:
        logger.warning("mimesis_provider: failed to generate MIMESIS_%s: %s", key, exc)
        return None


def list_supported_rules() -> list[str]:
    """Return the sorted list of `MIMESIS_<rule>` suffixes supported."""
    return sorted(_RULES.keys())
