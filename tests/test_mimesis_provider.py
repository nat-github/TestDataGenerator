"""Tests for the Mimesis adapter and the MIMESIS_ prefix dispatch.

Skipped automatically if Mimesis isn't installed (it's an optional extra).
"""
from __future__ import annotations

import pytest

from utils import mimesis_provider as mp
from utils.helpers import DataHelpers

mimesis_available = pytest.mark.skipif(
    not mp.is_available(),
    reason="mimesis not installed (poetry install --extras mimesis)",
)


@mimesis_available
def test_basic_rules_return_strings() -> None:
    for rule in ["NAME", "FIRST_NAME", "LAST_NAME", "EMAIL", "USERNAME", "CITY", "COUNTRY"]:
        value = mp.generate(rule)
        assert isinstance(value, str), f"{rule} should return a string, got {type(value)}"
        assert value, f"{rule} should not be empty"


@mimesis_available
def test_locale_suffix_changes_locale() -> None:
    # Generate many German first names and confirm at least one is non-ASCII or
    # otherwise plausibly German vs the default English pool.
    names_de = {mp.generate("FIRST_NAME", "de_DE") for _ in range(20)}
    names_en = {mp.generate("FIRST_NAME", "en_US") for _ in range(20)}
    # Different locales should not yield identical name pools.
    assert names_de != names_en


@mimesis_available
def test_unknown_rule_returns_none_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="utils.mimesis_provider"):
        result = mp.generate("DEFINITELY_NOT_A_RULE")
    assert result is None
    assert any("unknown rule" in rec.message.lower() for rec in caplog.records)


@mimesis_available
def test_helpers_dispatch_routes_mimesis_prefix_to_provider() -> None:
    """The single dispatch branch in helpers.generate_special_value should
    route MIMESIS_<rule> to the adapter, leaving plain `NAME` going to Faker.
    """
    helpers = DataHelpers()

    # Mimesis path
    mimesis_name = helpers.generate_special_value("MIMESIS_NAME", "VA64")
    assert isinstance(mimesis_name, str) and mimesis_name

    # Faker path — unaffected by Mimesis presence
    faker_name = helpers.generate_special_value("NAME", "VA64")
    assert isinstance(faker_name, str) and faker_name


@mimesis_available
def test_helpers_dispatch_with_locale_suffix() -> None:
    helpers = DataHelpers()
    val = helpers.generate_special_value("MIMESIS_FIRST_NAME:fr_FR", "VA32")
    assert isinstance(val, str) and val


@mimesis_available
def test_invalid_locale_falls_back_to_english_not_crash() -> None:
    # An unknown locale should not raise — it should silently fall back.
    val = mp.generate("NAME", "xx_XX")
    assert isinstance(val, str) and val


def test_is_available_matches_can_import_mimesis() -> None:
    """Confirm the availability gate. This test runs even if Mimesis is missing."""
    available = mp.is_available()
    if available:
        import mimesis  # noqa: F401  — succeeds when adapter says yes
    else:
        with pytest.raises(ImportError):
            import mimesis  # noqa: F401


def test_unknown_rule_when_mimesis_missing_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the not-installed code path even when Mimesis is present, by
    poking the adapter's internal cache. Confirms the user-facing error
    contains an actionable install hint."""
    # Reset cached state and force a "not installed" result.
    monkeypatch.setattr(mp, "_mimesis_module", None)
    monkeypatch.setattr(mp, "_locale_enum", None)
    monkeypatch.setattr(mp, "_generic_cls", None)
    monkeypatch.setattr(mp, "_import_error", ImportError("simulated missing mimesis"))

    with pytest.raises(mp.MimesisNotInstalledError) as excinfo:
        mp.generate("NAME")
    assert "install" in str(excinfo.value).lower()
    assert "mimesis" in str(excinfo.value).lower()


@mimesis_available
def test_list_supported_rules_returns_sorted() -> None:
    rules = mp.list_supported_rules()
    assert rules == sorted(rules)
    assert "NAME" in rules
    assert "EMAIL" in rules
