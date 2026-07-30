"""Isolated unit tests for `DataHelpers`.

`helpers.py` is the largest module in the package and is exercised almost
entirely through full generation runs. These tests call it directly, so a
regression in value generation is attributable rather than showing up as
"the parquet looks wrong".
"""
from __future__ import annotations

import re

import pytest

from sdp.utils.helpers import DataHelpers


@pytest.fixture
def helpers():
    return DataHelpers()


# ---------------------------------------------------------------------------
# Business values
# ---------------------------------------------------------------------------


def test_parse_business_values_splits_on_semicolons(helpers):
    assert helpers.parse_business_values("A;B;C") == ["A", "B", "C"]


def test_parse_business_values_trims_and_drops_empties(helpers):
    assert helpers.parse_business_values(" A ; ; B ") == ["A", "B"]


def test_parse_business_values_empty_inputs(helpers):
    assert helpers.parse_business_values(None) in (None, [])
    assert helpers.parse_business_values("") in (None, [])


def test_parse_business_values_single_value(helpers):
    assert helpers.parse_business_values("ONLY") == ["ONLY"]


# ---------------------------------------------------------------------------
# Regex generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", [
    r"\d{4}",
    r"[A-Z]{3}-\d{2}",
    r"CUST-\d{6}",
    r"[0-9]{2}[A-Z]{2}",
])
def test_generate_from_regex_matches_its_pattern(helpers, pattern):
    for _ in range(20):
        value = helpers.generate_from_regex_rule(f"REGEX:{pattern}")
        assert re.fullmatch(pattern, value), f"{value!r} does not match {pattern}"


def test_generate_from_regex_varies(helpers):
    values = {helpers.generate_from_regex_rule(r"REGEX:\d{6}") for _ in range(30)}
    assert len(values) > 1, "regex generation returned a constant"


def test_generate_from_regex_alternation(helpers):
    values = {helpers.generate_from_regex_rule(r"REGEX:(cat|dog)") for _ in range(30)}
    assert values <= {"cat", "dog"}


# ---------------------------------------------------------------------------
# Special rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule,check", [
    ("EMAIL", lambda v: "@" in v),
    ("UUID", lambda v: len(v) == 36 and v.count("-") == 4),
    ("IPV4", lambda v: len(v.split(".")) == 4),
    ("NAME", lambda v: isinstance(v, str) and v.strip()),
])
def test_special_rules_produce_plausible_values(helpers, rule, check):
    value = helpers.generate_special_value(rule, "VA256", column_name="c")
    assert value is not None
    assert check(str(value)), f"{rule} produced {value!r}"


def test_ipv4_octets_are_in_range(helpers):
    for _ in range(20):
        octets = [int(p) for p in str(
            helpers.generate_special_value("IPV4", "VA256", column_name="ip")
        ).split(".")]
        assert len(octets) == 4
        assert all(0 <= o <= 255 for o in octets)


def test_locale_suffix_is_accepted(helpers):
    value = helpers.generate_special_value("NAME:de_DE", "VA256", column_name="n")
    assert isinstance(value, str) and value.strip()


def test_unknown_special_rule_does_not_crash(helpers):
    """An unrecognised rule must degrade, not stop generation."""
    value = helpers.generate_special_value("NOT_A_REAL_RULE", "VA256", column_name="c")
    assert value is None or isinstance(value, str)


# ---------------------------------------------------------------------------
# Column batches
# ---------------------------------------------------------------------------


def test_batch_uses_business_values_when_present(helpers):
    values = helpers.generate_column_batch(
        {"data_type": "VA16", "business_values": "RED;GREEN;BLUE"}, n=50,
    )
    assert len(values) == 50
    assert set(values) <= {"RED", "GREEN", "BLUE"}


def test_batch_respects_numeric_bounds(helpers):
    values = helpers.generate_column_batch(
        {"data_type": "N38", "min_value": 10, "max_value": 20}, n=100,
    )
    assert len(values) == 100
    assert all(10 <= int(v) <= 20 for v in values)


def test_batch_decimal_type_returns_floats(helpers):
    values = helpers.generate_column_batch(
        {"data_type": "DC", "min_value": 0, "max_value": 5}, n=20,
    )
    assert all(isinstance(float(v), float) for v in values)
    assert all(0 <= float(v) <= 5 for v in values)


def test_batch_honours_special_rules(helpers):
    values = helpers.generate_column_batch(
        {"data_type": "VA256", "special_rules": "EMAIL"}, n=15,
    )
    assert len(values) == 15
    assert all("@" in str(v) for v in values)


def test_batch_of_zero_is_empty(helpers):
    assert helpers.generate_column_batch({"data_type": "VA16"}, n=0) == []


def test_batch_falls_back_for_a_bare_column(helpers):
    values = helpers.generate_column_batch({"data_type": "VA32"}, n=10)
    assert len(values) == 10


# ---------------------------------------------------------------------------
# Currency helpers
# ---------------------------------------------------------------------------


def test_currency_code_is_three_letters(helpers):
    for _ in range(20):
        code = helpers.generate_currency_code()
        assert len(code) == 3 and code.isupper()


def test_currency_code_honours_a_preference(helpers):
    codes = {helpers.generate_currency_code(prefer=["EUR", "USD"]) for _ in range(30)}
    assert codes <= {"EUR", "USD"}


def test_currency_code_ignores_invalid_preferences(helpers):
    """An unknown preferred code must not produce an invalid currency."""
    code = helpers.generate_currency_code(prefer=["NOT_A_CODE"])
    assert helpers.is_valid_currency_code(code)


def test_currency_label_pairs_code_and_name(helpers):
    label = helpers.generate_currency_label()
    assert any(sep in label for sep in ("–", "-", " "))
    assert helpers.is_valid_currency_code(label[:3])


def test_is_valid_currency_code(helpers):
    assert helpers.is_valid_currency_code("EUR") is True
    assert helpers.is_valid_currency_code("ZZZ") is False


# ---------------------------------------------------------------------------
# Locale handling
# ---------------------------------------------------------------------------


def test_known_locale_returns_a_faker(helpers):
    assert helpers._get_locale_faker("nl_NL") is not None


def test_locale_fakers_are_cached(helpers):
    """Constructing a Faker per value would dominate generation cost."""
    assert helpers._get_locale_faker("fr_FR") is helpers._get_locale_faker("fr_FR")


def test_unknown_locale_still_returns_a_faker(helpers):
    assert helpers._get_locale_faker("zz_ZZ") is not None


# ---------------------------------------------------------------------------
# NL postcode formatting
# ---------------------------------------------------------------------------


def test_wellformed_postcode_is_normalised(helpers):
    assert helpers._normalize_nl_postcode("1234ab") == "1234 AB"


def test_postcode_with_space_is_preserved(helpers):
    assert helpers._normalize_nl_postcode("1234 AB") == "1234 AB"


def test_unrecognised_postcode_is_returned_unchanged_or_regenerated(helpers):
    out = helpers._normalize_nl_postcode("nonsense")
    assert isinstance(out, str) and out
