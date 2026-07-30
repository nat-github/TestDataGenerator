
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import random
import re
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)


SAFE_DATETIME_MIN = pd.Timestamp("1900-01-01 00:00:00")
SAFE_DATETIME_MAX = pd.Timestamp("2262-04-11 23:47:16")


REGEX_DEFAULT_MAX_REPEAT = 5
REGEX_GENERATION_RETRIES = 200
REGEX_WORD_CHARSET = string.ascii_letters + string.digits + "_"
REGEX_PUNCTUATION_CHARSET = "-_.:/@#"
REGEX_DOT_CHARSET = string.ascii_letters + string.digits
SPECIAL_RULE_SEPARATOR = ";;"
NULL_RATE_RULE_PATTERN = re.compile(r"^NULL_RATE\s*=\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))$", re.IGNORECASE)
NULL_PCT_RULE_PATTERN = re.compile(r"^NULL_PCT\s*=\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))$", re.IGNORECASE)

# ----------------------------------------------------------------------
# ISO 3166-1 alpha-2 mapping (curated subset)
# Keep keys uppercase; values are official English short names.
# ----------------------------------------------------------------------
ISO_ALPHA2_TO_NAME: Dict[str, str] = {
    # Core EU + neighbors + common trading partners
    "NL": "Netherlands",
    "BE": "Belgium",
    "DE": "Germany",
    "FR": "France",
    "ES": "Spain",
    "IT": "Italy",
    "PT": "Portugal",
    "IE": "Ireland",
    "GB": "United Kingdom",
    "LU": "Luxembourg",
    "LI": "Liechtenstein",
    "AT": "Austria",
    "CH": "Switzerland",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "IS": "Iceland",
    "PL": "Poland",
    "CZ": "Czechia",
    "SK": "Slovakia",
    "HU": "Hungary",
    "RO": "Romania",
    "BG": "Bulgaria",
    "GR": "Greece",
    "SI": "Slovenia",
    "HR": "Croatia",
    "EE": "Estonia",
    "LV": "Latvia",
    "LT": "Lithuania",
    # Americas
    "US": "United States",
    "CA": "Canada",
    "MX": "Mexico",
    # APAC (sample)
    "AU": "Australia",
    "NZ": "New Zealand",
    "JP": "Japan",
    "CN": "China",
    "IN": "India",
    "SG": "Singapore",
}
NAME_TO_ISO_ALPHA2: Dict[str, str] = {v.lower(): k for k, v in ISO_ALPHA2_TO_NAME.items()}

# ----------------------------------------------------------------------
# ISO 4217 currency mapping (practical subset)
# ----------------------------------------------------------------------
CURRENCY_CODE_TO_NAME: Dict[str, str] = {
    # Europe
    "EUR": "Euro",
    "GBP": "Pound Sterling",
    "CHF": "Swiss Franc",
    "NOK": "Norwegian Krone",
    "SEK": "Swedish Krona",
    "DKK": "Danish Krone",
    "PLN": "Polish Złoty",
    "CZK": "Czech Koruna",
    "HUF": "Hungarian Forint",
    "RON": "Romanian Leu",
    "BGN": "Bulgarian Lev",
    "HRK": "Croatian Kuna",
    # Americas
    "USD": "US Dollar",
    "CAD": "Canadian Dollar",
    "MXN": "Mexican Peso",
    # APAC
    "JPY": "Japanese Yen",
    "CNY": "Chinese Yuan",
    "INR": "Indian Rupee",
    "AUD": "Australian Dollar",
    "NZD": "New Zealand Dollar",
    "SGD": "Singapore Dollar",
}
CURRENCY_NAME_TO_CODE: Dict[str, str] = {v.lower(): k for k, v in CURRENCY_CODE_TO_NAME.items()}


class DataHelpers:
    # Compiled regex + AST cache — shared across all instances (pattern string → (compiled, ast))
    _regex_ast_cache: Dict[str, tuple] = {}

    # Locales used for GLOBAL_* rules — well-supported by Faker
    _GLOBAL_LOCALES: List[str] = [
        'en_US', 'en_GB', 'en_AU', 'en_CA',
        'de_DE', 'fr_FR', 'es_ES', 'it_IT', 'nl_NL',
        'pt_BR', 'ja_JP', 'zh_CN', 'ko_KR',
        'ru_RU', 'pl_PL', 'tr_TR', 'sv_SE', 'da_DK',
    ]
    # Locales whose Faker generates valid IBANs for their country
    _IBAN_LOCALES: List[str] = [
        'nl_NL', 'de_DE', 'fr_FR', 'es_ES', 'it_IT', 'en_GB',
        'de_AT', 'de_CH', 'pl_PL', 'sv_SE', 'da_DK', 'fi_FI',
        'pt_PT', 'hu_HU',
    ]
    # Mapping ISO 3166-1 alpha-2 → Faker locale for IBAN generation
    _CC_TO_IBAN_LOCALE: Dict[str, str] = {
        'NL': 'nl_NL', 'DE': 'de_DE', 'FR': 'fr_FR', 'BE': 'nl_BE',
        'ES': 'es_ES', 'IT': 'it_IT', 'GB': 'en_GB', 'AT': 'de_AT',
        'CH': 'de_CH', 'SE': 'sv_SE', 'DK': 'da_DK', 'FI': 'fi_FI',
        'NO': 'no_NO', 'PL': 'pl_PL', 'PT': 'pt_PT', 'HU': 'hu_HU',
        'RO': 'ro_RO', 'IE': 'en_IE', 'LU': 'lb_LU', 'SK': 'sk_SK',
    }

    def __init__(self):
        # Dutch locale for realistic data (kept as default for backward compat)
        self.faker = Faker('nl_NL')
        # Per-locale Faker cache — populated lazily by _get_locale_faker()
        self._locale_fakers: Dict[str, Any] = {'nl_NL': self.faker}

        # Common Dutch banks (BIC/IBAN bank codes; 4-letter preferred)
        # 'REVOLT' retained for legacy/back-compat; we normalize to 4 letters.
        self.dutch_banks: Dict[str, str] = {
            'ABNA': 'ABN AMRO',
            'INGB': 'ING Bank',
            'RABO': 'Rabobank',
            'SNSB': 'SNS Bank',
            'ASNB': 'ASN Bank',
            'FRBK': 'Friesland Bank',
            'TRIO': 'Triodos Bank',
            'KNAB': 'Knab Bank',
            'BUNQ': 'bunq',
            'REVOLT': 'Revolut',  # legacy key
        }

        # Valid Dutch location codes for BIC (illustrative)
        self.dutch_location_codes: List[str] = ['2A', '2B', '2S']
        # Valid example branch codes
        self.branch_codes: List[str] = ['XXX', 'AMA', 'ROT', 'UTS', 'AMS', 'EUR', 'NLD']

        # Curated list of common NL cities for extra realism/consistency
        self._nl_cities: List[str] = [
            "Amsterdam", "Rotterdam", "Den Haag", "Utrecht", "Eindhoven",
            "Tilburg", "Groningen", "Almere", "Breda", "Nijmegen",
            "Enschede", "Haarlem", "Arnhem", "Zaanstad", "Zwolle",
            "Leeuwarden", "Leiden", "Maastricht", "Dordrecht", "Amersfoort",
        ]
        self._generic_payment_codes: Dict[str, List[str]] = {
            "reason": ["AC01", "AM04", "FF01", "FR01", "MS02", "RC01"],
            "purpose": ["SALA", "SUPP", "PENS", "GDSV", "OTHR", "TAXS"],
            "transaction_type": ["PMNT", "TRSF", "CARD", "CASH", "FEE", "SEPA"],
            "delivery_channel": ["MOB", "WEB", "ATM", "API", "BRN"],
            "delivery_system": ["SEPA", "SWIFT", "TARGET", "CORE", "RT1"],
            "local_instrument": ["CORE", "B2B", "INST", "SDVA"],
            "indicator": ["Y", "N"],
            "boolean_text": ["True", "False"],
        }

    # ------------------------------------------------------------------
    # ISO 3166-1 helpers (country)
    # ------------------------------------------------------------------
    def get_country_name(self, alpha2: str) -> Optional[str]:
        if not alpha2:
            return None
        return ISO_ALPHA2_TO_NAME.get(str(alpha2).upper())

    def get_country_code(self, country_name: str) -> Optional[str]:
        if not country_name:
            return None
        return NAME_TO_ISO_ALPHA2.get(str(country_name).strip().lower())

    def is_valid_country_code(self, alpha2: str) -> bool:
        return self.get_country_name(alpha2) is not None

    def generate_country_code(self, prefer: Optional[List[str]] = None) -> str:
        codes = list(ISO_ALPHA2_TO_NAME.keys())
        if prefer:
            prefer = [c.upper() for c in prefer if c and c.upper() in ISO_ALPHA2_TO_NAME]
            if prefer:
                return random.choice(prefer)
        return random.choice(codes)

    def generate_country_label(self, prefer: Optional[List[str]] = None, with_dash: bool = True) -> str:
        code = self.generate_country_code(prefer)
        name = ISO_ALPHA2_TO_NAME[code]
        sep = " – " if with_dash else " "
        return f"{code}{sep}{name}"

    # ------------------------------------------------------------------
    # ISO 4217 helpers (currency)
    # ------------------------------------------------------------------
    def get_currency_name(self, code: str) -> Optional[str]:
        if not code:
            return None
        return CURRENCY_CODE_TO_NAME.get(str(code).upper())

    def get_currency_code(self, name: str) -> Optional[str]:
        if not name:
            return None
        return CURRENCY_NAME_TO_CODE.get(str(name).strip().lower())

    def is_valid_currency_code(self, code: str) -> bool:
        return self.get_currency_name(code) is not None

    def generate_currency_code(self, prefer: Optional[List[str]] = None) -> str:
        if prefer:
            prefer = [c.upper() for c in prefer if c and self.is_valid_currency_code(c)]
            if prefer:
                return random.choice(prefer)
        # NL-centric bias
        weighted = ["EUR"] * 5 + ["USD"] * 3 + ["GBP"] * 2 + [
            "CHF", "SEK", "DKK", "NOK", "PLN", "CZK", "HUF", "RON", "BGN", "JPY", "CNY", "INR", "AUD", "NZD", "SGD"
        ]
        return random.choice(weighted)

    def generate_currency_label(self, prefer: Optional[List[str]] = None, with_dash: bool = True) -> str:
        code = self.generate_currency_code(prefer)
        name = CURRENCY_CODE_TO_NAME[code]
        sep = " – " if with_dash else " "
        return f"{code}{sep}{name}"

    # ------------------------------------------------------------------
    # NL Address helpers
    # ------------------------------------------------------------------
    def _normalize_nl_postcode(self, postcode: str) -> str:
        """Normalize to Dutch postcode format '1234 AB'."""
        if not postcode:
            return postcode
        s = str(postcode).upper().replace(" ", "")
        if len(s) >= 6 and s[:4].isdigit() and s[4:6].isalpha():
            return f"{s[:4]} {s[4:6]}"
        try:
            pc = self.faker.postcode().upper().replace(" ", "")
            if pc[:4].isdigit() and pc[4:6].isalpha():
                return f"{pc[:4]} {pc[4:6]}"
        except (AttributeError, IndexError, TypeError) as exc:
            # The active locale may not provide postcode(), or may return a
            # shape this NL-specific formatter cannot use. Returning the
            # input unchanged is the intended fallback.
            logger.debug("postcode formatting fell back to the raw value: %s", exc)
        return postcode

    def generate_nl_address_components(
        self,
        include_house_letter: bool = True,
        include_unit: bool = False,
    ) -> Dict[str, str]:
        """Return NL address components."""
        street = self.faker.street_name()
        raw_nr = self.faker.building_number()
        digits = "".join(ch for ch in raw_nr if ch.isdigit()) or str(random.randint(1, 199))
        letter = (
            "".join(ch for ch in raw_nr if ch.isalpha())[:1].upper()
            if include_house_letter and random.random() < 0.4
            else ""
        )
        unit = f"unit {random.randint(1, 20)}" if include_unit and random.random() < 0.3 else ""
        city = self.generate_nl_city_nm()
        postcode = self.generate_nl_pst_code()
        country = "Netherlands"
        return {
            "street": street,
            "house_number": digits,
            "house_letter": letter,
            "unit": unit,
            "postcode": postcode,
            "city": city,
            "country": country,
        }

    def format_nl_address(
        self,
        components: Optional[Dict[str, str]] = None,
        extra_line: Optional[str] = None,
    ) -> str:
        """Format Dutch address like standard print format."""
        if components is None:
            components = self.generate_nl_address_components()
        street = components.get("street", "")
        nr = components.get("house_number", "")
        letter = components.get("house_letter", "")
        unit = components.get("unit", "")
        postcode = self._normalize_nl_postcode(components.get("postcode", ""))
        city = components.get("city", "")
        country = components.get("country", "Netherlands")
        line1_parts = [f"{street} {nr}{letter}"]
        if unit:
            line1_parts.append(unit)
        line1 = " ".join(p for p in line1_parts if p)
        line2 = f"{postcode} {city}".strip()
        parts = [line1, line2, country]
        if extra_line:
            parts.insert(0, extra_line.strip())
        return "\n".join(p for p in parts if p)

    # NL-specific helpers
    def generate_nl_city_nm(self) -> str:
        """Return a realistic Dutch city name."""
        return random.choice(self._nl_cities) if random.random() < 0.7 else self.faker.city()

    def generate_nl_cty_code(self, prefer: Optional[List[str]] = None) -> str:
        """Return a country code with NL-centric bias."""
        if prefer:
            prefer = [c.upper() for c in prefer if c and self.is_valid_country_code(c)]
        bias = prefer or ["NL", "BE", "DE", "FR", "GB"]
        return self.generate_country_code(prefer=bias)

    def generate_nl_pst_code(self) -> str:
        """Return a normalized Dutch postcode in '1234 AB' format."""
        return self._normalize_nl_postcode(self.faker.postcode())

    # ------------------------------------------------------------------
    # Core generators (numeric/alphanumeric/decimal)
    # ------------------------------------------------------------------
    def _generate_numeric_string(self, length: int) -> str:
        if length <= 0:
            return ""
        if length == 1:
            return str(random.randint(0, 9))
        first = str(random.randint(1, 9))
        rest = ''.join(str(random.randint(0, 9)) for _ in range(length - 1))
        return first + rest

    def _generate_alphanumeric_string(self, length: int) -> str:
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(random.choices(chars, k=max(0, length)))

    def _generate_decimal_value(
        self,
        precision: int,
        scale: int,
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> float:
        if min_val is None or pd.isna(min_val):
            min_val = 0.0
        if max_val is None or pd.isna(max_val):
            max_val = (10 ** (precision - scale)) - (10 ** (-scale))
        integer_digits = max(0, precision - scale)
        max_integer = (10 ** integer_digits) - 1
        integer_part = random.randint(0, max_integer)
        fractional_part = random.randint(0, (10 ** scale) - 1)
        value = integer_part + (fractional_part / (10 ** scale))
        value = min(max(value, float(min_val)), float(max_val))
        return round(value, scale)

    def _generate_numeric_value(
        self,
        length: int,
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> int:
        if min_val is None or pd.isna(min_val):
            min_val = 10 ** (max(1, length) - 1)
        if max_val is None or pd.isna(max_val):
            max_val = (10 ** max(1, length)) - 1
        return random.randint(int(min_val), int(max_val))

    # ------------------------------------------------------------------
    # Business values / special rules / generic samplers
    # ------------------------------------------------------------------
    def parse_business_values(self, values_str: str) -> Optional[List[str]]:
        """Parse semicolon-separated business values, stripping quotes and blanks."""
        if pd.isna(values_str) or not str(values_str).strip():
            return None
        cleaned = str(values_str).replace("'", "").replace('"', "")
        values = [v.strip() for v in cleaned.split(';') if v.strip()]
        return values if values else None

    @staticmethod
    def _split_special_rule_tokens(rule: Optional[str]) -> List[str]:
        if rule is None or pd.isna(rule):
            return []
        return [token.strip() for token in str(rule).split(SPECIAL_RULE_SEPARATOR) if token and token.strip()]

    @staticmethod
    def _parse_null_modifier(token: str) -> Optional[tuple[str, float]]:
        token = token.strip()
        match = NULL_RATE_RULE_PATTERN.fullmatch(token)
        if match:
            value = float(match.group(1))
            if not 0.0 <= value <= 1.0:
                raise ValueError('NULL_RATE must be between 0 and 1')
            return 'null_probability', value

        match = NULL_PCT_RULE_PATTERN.fullmatch(token)
        if match:
            percent = float(match.group(1))
            if not 0.0 <= percent <= 100.0:
                raise ValueError('NULL_PCT must be between 0 and 100')
            return 'null_probability', percent / 100.0

        return None

    def parse_special_rule_config(self, rule: Optional[str]) -> Dict[str, Any]:
        tokens = self._split_special_rule_tokens(rule)
        parsed: Dict[str, Any] = {
            'value_rule': None,
            'modifiers': {},
            'tokens': tokens,
        }

        for token in tokens:
            null_modifier = self._parse_null_modifier(token)
            if null_modifier is not None:
                key, value = null_modifier
                if key in parsed['modifiers']:
                    raise ValueError('Duplicate NULL_RATE/NULL_PCT modifier is not allowed')
                parsed['modifiers'][key] = value
                continue

            if parsed['value_rule'] is not None:
                raise ValueError(
                    'Only one value-generating special rule is allowed per cell; '
                    f'use {SPECIAL_RULE_SEPARATOR} only for modifiers such as NULL_RATE'
                )
            parsed['value_rule'] = token

        return parsed

    def get_special_rule_value_directive(self, rule: Optional[str]) -> Optional[str]:
        return self.parse_special_rule_config(rule)['value_rule']

    def get_special_rule_null_probability(self, rule: Optional[str]) -> Optional[float]:
        return self.parse_special_rule_config(rule)['modifiers'].get('null_probability')

    @staticmethod
    def is_regex_rule(rule: Optional[str]) -> bool:
        if rule is None or pd.isna(rule):
            return False
        tokens = DataHelpers._split_special_rule_tokens(rule)
        value_tokens: List[str] = []
        for token in tokens:
            try:
                if DataHelpers._parse_null_modifier(token) is None:
                    value_tokens.append(token)
            except ValueError:
                return False
        if len(value_tokens) != 1:
            return False
        text = value_tokens[0]
        if not text:
            return False
        return text.upper().startswith('REGEX:') or text.upper().startswith('REGEX=')

    @staticmethod
    def extract_regex_pattern(rule: Optional[str]) -> Optional[str]:
        if rule is None or pd.isna(rule):
            return None
        tokens = DataHelpers._split_special_rule_tokens(rule)
        value_tokens: List[str] = []
        for token in tokens:
            try:
                if DataHelpers._parse_null_modifier(token) is None:
                    value_tokens.append(token)
            except ValueError:
                return None
        if len(value_tokens) != 1:
            return None
        text = value_tokens[0]
        if not text:
            return None
        if text.upper().startswith('REGEX:') or text.upper().startswith('REGEX='):
            pattern = text[6:].strip()
            return pattern or None
        return None

    def validate_special_rule(
        self,
        rule: Optional[str],
        max_length: Optional[int] = None,
        is_pk: bool = False,
    ) -> None:
        parsed = self.parse_special_rule_config(rule)
        if is_pk and 'null_probability' in parsed['modifiers']:
            raise ValueError('NULL_RATE/NULL_PCT is not allowed on primary-key columns')

        value_rule = parsed['value_rule']
        if value_rule and self.is_regex_rule(value_rule):
            self.validate_regex_rule(value_rule, max_length=max_length)

    def validate_regex_rule(self, rule: str, max_length: Optional[int] = None) -> None:
        pattern = self.extract_regex_pattern(rule)
        if not pattern:
            raise ValueError('Regex rule must start with REGEX: or REGEX= followed by a pattern')

        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f'Invalid regex pattern: {exc}') from exc

        ast = self._parse_regex_pattern(pattern)
        min_length = self._regex_min_length(ast)
        if max_length is not None and min_length > max_length:
            raise ValueError(
                f'Regex minimum length {min_length} exceeds configured column length {max_length}'
            )

    def generate_from_regex_rule(self, rule: str, max_length: Optional[int] = None) -> str:
        pattern = self.extract_regex_pattern(rule)
        if not pattern:
            raise ValueError('Regex rule must start with REGEX: or REGEX= followed by a pattern')

        cache = self.__class__._regex_ast_cache
        if pattern not in cache:
            cache[pattern] = (re.compile(pattern), self._parse_regex_pattern(pattern))
        compiled, ast = cache[pattern]
        min_length = self._regex_min_length(ast)
        if max_length is not None and min_length > max_length:
            raise ValueError(
                f'Regex minimum length {min_length} exceeds configured column length {max_length}'
            )

        for _ in range(REGEX_GENERATION_RETRIES):
            candidate = self._generate_from_regex_ast(ast)
            if max_length is not None and len(candidate) > max_length:
                continue
            if compiled.fullmatch(candidate):
                return candidate

        raise ValueError(
            f'Could not generate a value matching regex within {REGEX_GENERATION_RETRIES} attempts: {pattern}'
        )

    def _parse_regex_pattern(self, pattern: str) -> Dict[str, Any]:
        pos = 0

        def current() -> Optional[str]:
            return pattern[pos] if pos < len(pattern) else None

        def parse_expression() -> Dict[str, Any]:
            nonlocal pos
            options = [parse_sequence()]
            while current() == '|':
                pos += 1
                options.append(parse_sequence())
            return options[0] if len(options) == 1 else {'type': 'choice', 'options': options}

        def parse_sequence() -> Dict[str, Any]:
            nonlocal pos
            parts: List[Dict[str, Any]] = []
            while pos < len(pattern) and current() not in {')', '|'}:
                parts.append(parse_quantified())
            if not parts:
                return {'type': 'literal', 'value': ''}
            return parts[0] if len(parts) == 1 else {'type': 'sequence', 'parts': parts}

        def parse_quantified() -> Dict[str, Any]:
            nonlocal pos
            node = parse_atom()
            char = current()
            if char is None:
                return node
            if char == '?':
                pos += 1
                return {'type': 'repeat', 'node': node, 'min': 0, 'max': 1}
            if char == '*':
                pos += 1
                return {'type': 'repeat', 'node': node, 'min': 0, 'max': REGEX_DEFAULT_MAX_REPEAT}
            if char == '+':
                pos += 1
                return {'type': 'repeat', 'node': node, 'min': 1, 'max': REGEX_DEFAULT_MAX_REPEAT}
            if char == '{':
                pos += 1
                match = re.match(r'(\d+)(?:,(\d*)?)?}', pattern[pos:])
                if not match:
                    raise ValueError(f'Unsupported or invalid regex quantifier near: {pattern[pos - 1:]}')
                min_count = int(match.group(1))
                if match.group(2) is None:
                    max_count = min_count
                elif match.group(2) == '':
                    max_count = min_count + REGEX_DEFAULT_MAX_REPEAT
                else:
                    max_count = int(match.group(2))
                if max_count < min_count:
                    raise ValueError('Regex quantifier upper bound cannot be less than lower bound')
                pos += len(match.group(0))
                return {'type': 'repeat', 'node': node, 'min': min_count, 'max': max_count}
            return node

        def parse_atom() -> Dict[str, Any]:
            nonlocal pos
            char = current()
            if char is None:
                return {'type': 'literal', 'value': ''}
            if char in {'^', '$'}:
                pos += 1
                return {'type': 'literal', 'value': ''}
            if char == '(':
                if pattern[pos:pos + 2] == '(?':
                    raise ValueError('Unsupported regex feature: lookarounds and special groups are not supported')
                pos += 1
                node = parse_expression()
                if current() != ')':
                    raise ValueError('Unclosed group in regex pattern')
                pos += 1
                return node
            if char == '[':
                return parse_char_class()
            if char == '\\':
                pos += 1
                return parse_escape(in_class=False)
            if char == '.':
                pos += 1
                return {'type': 'class', 'chars': list(REGEX_DOT_CHARSET)}
            pos += 1
            return {'type': 'literal', 'value': char}

        def parse_char_class() -> Dict[str, Any]:
            nonlocal pos
            pos += 1
            negate = False
            if current() == '^':
                negate = True
                pos += 1

            chars: List[str] = []
            while pos < len(pattern) and current() != ']':
                if current() == '\\':
                    pos += 1
                    chars.extend(parse_escape(in_class=True)['chars'])
                    continue
                start = pattern[pos]
                if pos + 2 < len(pattern) and pattern[pos + 1] == '-' and pattern[pos + 2] != ']':
                    end = pattern[pos + 2]
                    chars.extend(chr(code) for code in range(ord(start), ord(end) + 1))
                    pos += 3
                    continue
                chars.append(start)
                pos += 1

            if current() != ']':
                raise ValueError('Unclosed character class in regex pattern')
            pos += 1

            if negate:
                chars = [
                    ch for ch in (REGEX_WORD_CHARSET + REGEX_PUNCTUATION_CHARSET)
                    if ch not in set(chars)
                ]

            if not chars:
                raise ValueError('Regex character class cannot be empty')
            return {'type': 'class', 'chars': list(dict.fromkeys(chars))}

        def parse_escape(in_class: bool) -> Dict[str, Any]:
            nonlocal pos
            if pos >= len(pattern):
                raise ValueError('Dangling escape in regex pattern')
            token = pattern[pos]
            pos += 1
            mapping = {
                'd': list(string.digits),
                'D': list(string.ascii_letters + REGEX_PUNCTUATION_CHARSET),
                'w': list(REGEX_WORD_CHARSET),
                'W': list(REGEX_PUNCTUATION_CHARSET),
                's': [' '],
                'S': list(string.ascii_letters + string.digits),
                't': ['\t'],
                'n': ['\n'],
                '\\': ['\\'],
            }
            if token in mapping:
                return {'type': 'class', 'chars': mapping[token]}
            if token.isdigit():
                raise ValueError('Unsupported regex feature: backreferences are not supported')
            return {'type': 'literal', 'value': token} if not in_class else {'type': 'class', 'chars': [token]}

        ast = parse_expression()
        if pos != len(pattern):
            raise ValueError(f'Unsupported trailing regex pattern content: {pattern[pos:]}')
        return ast

    def _regex_min_length(self, node: Dict[str, Any]) -> int:
        node_type = node['type']
        if node_type == 'literal':
            return len(node['value'])
        if node_type == 'class':
            return 1
        if node_type == 'sequence':
            return sum(self._regex_min_length(part) for part in node['parts'])
        if node_type == 'choice':
            return min(self._regex_min_length(option) for option in node['options'])
        if node_type == 'repeat':
            return node['min'] * self._regex_min_length(node['node'])
        raise ValueError(f'Unsupported regex AST node type: {node_type}')

    def _generate_from_regex_ast(self, node: Dict[str, Any]) -> str:
        node_type = node['type']
        if node_type == 'literal':
            return node['value']
        if node_type == 'class':
            return random.choice(node['chars'])
        if node_type == 'sequence':
            return ''.join(self._generate_from_regex_ast(part) for part in node['parts'])
        if node_type == 'choice':
            return self._generate_from_regex_ast(random.choice(node['options']))
        if node_type == 'repeat':
            count = random.randint(node['min'], node['max'])
            return ''.join(self._generate_from_regex_ast(node['node']) for _ in range(count))
        raise ValueError(f'Unsupported regex AST node type: {node_type}')

    # ---------- Banking formats & helpers ----------

    def _generate_valid_bic(self) -> str:
        """
        Generate a valid-looking BIC: AAAA BB CC DDD (institution, country, location, branch).
        """
        bank_key = random.choice(list(self.dutch_banks.keys()))
        bank_code = "".join(ch for ch in bank_key if ch.isalpha()).upper()[:4]
        if len(bank_code) < 4:
            bank_code = (bank_code + "X" * 4)[:4]
        country_code = 'NL'
        location_code = random.choice(self.dutch_location_codes)
        branch_code = random.choice(self.branch_codes)
        return f"{bank_code}{country_code}{location_code}{branch_code}"

    # --- IBAN check digits: ISO 13616 (Mod-97) ---
    # References: IBAN mod-97 algorithm and standard.  # noqa
    #   - ibantest.com: FAQ & algorithm summary (Mod-97).  # noqa
    #   - ISO 13616 background (Wikipedia summary).        # noqa
    def _iban_check_digits(self, country_code: str, bban: str) -> str:
        """
        Compute IBAN check digits (two digits) using ISO 13616 (mod-97).
        """
        rearranged = f"{bban}{country_code}00"
        converted = []
        for ch in rearranged:
            if ch.isalpha():
                converted.append(str(ord(ch.upper()) - 55))  # A=10 ... Z=35
            else:
                converted.append(ch)
        num_str = "".join(converted)
        remainder = 0
        for c in num_str:
            remainder = (remainder * 10 + int(c)) % 97
        return f"{98 - remainder:02d}"

    # --- NL IBAN (18 chars): NL + 2 + 4-letter bank + 10-digit acct ---
    # References: NL IBAN format & length.  # noqa
    #   - ibantest.com (Regex & blocks)      # noqa
    #   - bank.codes / wise.com details      # noqa
    def _generate_dutch_iban(self) -> str:
        raw_bank = random.choice(list(self.dutch_banks.keys()))
        bank_code_4 = "".join(ch for ch in raw_bank if ch.isalpha()).upper()[:4]
        if len(bank_code_4) < 4:
            bank_code_4 = (bank_code_4 + "X" * 4)[:4]
        account_num = ''.join(str(random.randint(0, 9)) for _ in range(10))
        bban = f"{bank_code_4}{account_num}"
        check = self._iban_check_digits("NL", bban)
        return f"NL{check}{bban}"

    # --- Country-aware BBAN (domestic formats) ---
    # Structures from SWIFT IBAN registry summaries:
    #   NL: 4!a10!n
    #   DE: 8!n10!n
    #   BE: 3!n7!n2!n (last 2 digits: national mod-97)
    #   FR: 5!n5!n11!c2!n (national key simplified here)
    #   ES: 4!n4!n2!n10!n (two national digits simplified here)
    #   IT: 1!a5!n5!n12!c (CIN + ABI + CAB + account)
    def _generate_bban(self, country_code: Optional[str] = None) -> str:
        supported = ['NL', 'DE', 'BE', 'FR', 'ES', 'IT']
        cc = (country_code or '').upper()
        if cc not in supported:
            cc = random.choices(supported, weights=[5, 3, 2, 2, 2, 2], k=1)[0]

        if cc == 'NL':
            raw_bank = random.choice(list(self.dutch_banks.keys()))
            bank_code_4 = "".join(ch for ch in raw_bank if ch.isalpha()).upper()[:4]
            if len(bank_code_4) < 4:
                bank_code_4 = (bank_code_4 + "X" * 4)[:4]
            account_num = ''.join(str(random.randint(0, 9)) for _ in range(10))
            return f"{bank_code_4}{account_num}"  # NL BBAN 4!a10!n  [1](https://www.ibantest.com/en/iban-structure/netherlands)

        if cc == 'DE':
            blz = ''.join(str(random.randint(0, 9)) for _ in range(8))
            acct = ''.join(str(random.randint(0, 9)) for _ in range(10))
            return f"{blz}{acct}"  # DE BBAN 8!n10!n          [4](https://www.ibantest.com/en/iban-structure/germany)

        if cc == 'BE':
            bank3 = ''.join(str(random.randint(0, 9)) for _ in range(3))
            acct7 = ''.join(str(random.randint(0, 9)) for _ in range(7))
            base = int(f"{bank3}{acct7}")
            r = base % 97
            check2 = 97 if r == 0 else r
            return f"{bank3}{acct7}{check2:02d}"  # BE BBAN 3!n7!n2!n [5](https://www.ibantest.com/en/iban-structure/belgium)

        if cc == 'FR':
            bank5 = ''.join(str(random.randint(0, 9)) for _ in range(5))
            branch5 = ''.join(str(random.randint(0, 9)) for _ in range(5))
            acct11 = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(11))
            key2 = f"{random.randint(0, 99):02d}"  # simplified RIB key
            return f"{bank5}{branch5}{acct11}{key2}"  # FR BBAN 5!n5!n11!c2!n [6](https://www.ibantest.com/en/iban-structure/france)

        if cc == 'ES':
            bank4 = ''.join(str(random.randint(0, 9)) for _ in range(4))
            branch4 = ''.join(str(random.randint(0, 9)) for _ in range(4))
            chk2 = ''.join(str(random.randint(0, 9)) for _ in range(2))  # simplified national digits
            acct10 = ''.join(str(random.randint(0, 9)) for _ in range(10))
            return f"{bank4}{branch4}{chk2}{acct10}"  # ES BBAN 4!n4!n2!n10!n [7](https://www.ibantest.com/en/iban-structure/spain)

        if cc == 'IT':
            cin1 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            abi5 = ''.join(str(random.randint(0, 9)) for _ in range(5))
            cab5 = ''.join(str(random.randint(0, 9)) for _ in range(5))
            acct12 = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(12))
            return f"{cin1}{abi5}{cab5}{acct12}"  # IT BBAN 1!a5!n5!n12!c [8](https://www.ibantest.com/en/iban-structure/italy)

        # Fallback (should not occur)
        return ''.join(str(random.randint(0, 9)) for _ in range(12))

    # --- European IBAN builder (supported CCs) ---
    def _generate_european_iban(self, prefer: Optional[str] = None) -> str:
        supported = ['NL', 'DE', 'BE', 'FR', 'ES', 'IT']
        cc = (prefer or '').upper()
        country_code = cc if cc in supported else random.choice(supported)
        bban = self._generate_bban(country_code)
        check = self._iban_check_digits(country_code, bban)
        return f"{country_code}{check}{bban}"

    def _generate_dutch_phone(self) -> str:
        # mobile
        if random.random() > 0.5:
            return f"+31 6 {random.randint(10000000, 99999999)}"
        # landline
        area_codes = [
            '10','20','23','24','26','30','33','35','36','38','40','43','45','46','50',
            '53','55','58','70','71','72','73','74','75','76','77','78','79'
        ]
        area = random.choice(area_codes)
        return f"+31 {area} {random.randint(1000000, 9999999)}"

    def _generate_dutch_bank_account(self) -> str:
        return ''.join(str(random.randint(0, 9)) for _ in range(9))

    @staticmethod
    def _infer_base_type(data_type: str) -> str:
        dt = str(data_type or '').upper()
        for prefix in ('VA', 'AN', 'NS', 'DC', 'DT', 'TS'):
            if dt.startswith(prefix):
                return prefix
        for prefix in ('N', 'A', 'D', 'T'):
            if dt.startswith(prefix):
                return prefix
        return dt

    @staticmethod
    def _truncate_text(value: Any, max_length: Optional[int]) -> Any:
        if value is None or max_length is None:
            return value
        text = str(value).strip()
        if len(text) <= max_length:
            return text
        clipped = text[:max_length].rstrip(' ,;-')
        return clipped or text[:max_length]

    def _target_text_length(self, max_length: int, floor: int = 6) -> int:
        if max_length <= floor:
            return max_length
        upper = min(max_length, max(floor + 4, 120))
        lower = min(floor, upper)
        return random.randint(lower, upper)

    def _generate_sentence_like_text(self, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if max_length <= 12:
            return self._truncate_text(self.faker.word(), max_length)
        if max_length <= 40:
            text = self.faker.sentence(nb_words=random.randint(2, 5))
            return self._truncate_text(text, max_length)

        target = self._target_text_length(max_length, floor=18)
        chunks: List[str] = []
        while len(" ".join(chunks)) < target:
            chunks.append(self.faker.sentence(nb_words=random.randint(4, 10)).strip())
            if len(chunks) > 4:
                break
        return self._truncate_text(" ".join(chunks), max_length)

    def _generate_structured_reference(self, column_name: str, max_length: Optional[int]) -> str:
        tokens = [token.upper()[:4] for token in re.split(r'[^A-Za-z0-9]+', column_name or '') if token]
        prefix = ''.join(tokens[:2]) or 'REF'
        suffix_length = max(4, min(10, (max_length or 16) - len(prefix) - 1))
        suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=suffix_length))
        return self._truncate_text(f"{prefix}-{suffix}", max_length)

    def _select_code_value(self, column_name: str, max_length: Optional[int]) -> str:
        col = (column_name or '').lower()
        if any(token in col for token in ['rsn', 'reason']):
            candidates = self._generic_payment_codes['reason']
        elif any(token in col for token in ['purp', 'ctgy', 'category']):
            candidates = self._generic_payment_codes['purpose']
        elif any(token in col for token in ['tx_tp', 'txn_tp', 'transaction_type', 'dtld_tx_t', 'trn_type']):
            candidates = self._generic_payment_codes['transaction_type']
        elif any(token in col for token in ['dlvry_chan', 'channel']):
            candidates = self._generic_payment_codes['delivery_channel']
        elif any(token in col for token in ['dlvry_sys', 'system']):
            candidates = self._generic_payment_codes['delivery_system']
        elif any(token in col for token in ['lcl_instrm', 'instrument']):
            candidates = self._generic_payment_codes['local_instrument']
        elif any(token in col for token in ['cdt_dbt_ind']):
            candidates = ['CRDT', 'DBIT']
        elif any(token in col for token in ['flag', 'flg', 'bool']):
            candidates = self._generic_payment_codes['indicator'] if (max_length or 1) <= 1 else self._generic_payment_codes['boolean_text']
        elif any(token in col for token in ['ind', 'indicator']):
            candidates = ['Y', 'N'] if (max_length or 1) <= 1 else ['Yes', 'No']
        else:
            return self._generate_structured_reference(column_name, max_length)

        valid = [candidate for candidate in candidates if max_length is None or len(candidate) <= max_length]
        return random.choice(valid or candidates)

    def _looks_like_name_field(self, col: str) -> bool:
        return (
            'name' in col
            or 'naam' in col
            or col.endswith('_nm')
            or '_nm_' in col
            or col.endswith('_name')
        )

    def _looks_like_company_field(self, col: str) -> bool:
        return any(token in col for token in ['org', 'company', 'bank', 'scheme', 'merchant', 'institution', 'corp'])

    def _single_line_address(self, components: Dict[str, str], line_no: int, max_length: Optional[int]) -> str:
        if line_no == 1:
            text = f"{components['street']} {components['house_number']}{components['house_letter']}".strip()
        else:
            text = f"{components['postcode']} {components['city']}".strip()
        return self._truncate_text(text, max_length)

    # ------------------------------------------------------------------
    # Locale helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_locale(s: str) -> str:
        s = s.strip()
        if '_' in s:
            lang, country = s.split('_', 1)
            return f"{lang.lower()}_{country.upper()}"
        _single = {
            'en': 'en_US', 'de': 'de_DE', 'fr': 'fr_FR', 'es': 'es_ES',
            'it': 'it_IT', 'nl': 'nl_NL', 'pt': 'pt_BR', 'ja': 'ja_JP',
            'zh': 'zh_CN', 'ko': 'ko_KR', 'ru': 'ru_RU', 'pl': 'pl_PL',
            'ar': 'ar_AA', 'tr': 'tr_TR', 'sv': 'sv_SE', 'da': 'da_DK',
            'fi': 'fi_FI', 'nb': 'no_NO', 'hi': 'hi_IN',
        }
        return _single.get(s.lower(), f"{s.lower()}_{s.upper()}")

    def _get_locale_faker(self, locale: str) -> Faker:
        if locale not in self._locale_fakers:
            try:
                self._locale_fakers[locale] = Faker(locale)
            except (AttributeError, TypeError, ValueError) as exc:
                # Faker raises AttributeError for locales it does not ship.
                # Falling back is right, but silently returning US data for a
                # config that asked for de_DE is worth a warning.
                logger.warning(
                    f"Faker has no locale {locale!r} ({type(exc).__name__}) — "
                    f"falling back to en_US for this column"
                )
                self._locale_fakers[locale] = Faker('en_US')
        return self._locale_fakers[locale]

    def _random_global_faker(self) -> Faker:
        return self._get_locale_faker(random.choice(self._GLOBAL_LOCALES))

    # ------------------------------------------------------------------
    # International banking generators
    # ------------------------------------------------------------------
    def _generate_iban(self, country_code: Optional[str] = None) -> str:
        if country_code:
            locale = self._CC_TO_IBAN_LOCALE.get(country_code.upper(),
                                                   random.choice(self._IBAN_LOCALES))
        else:
            locale = random.choice(self._IBAN_LOCALES)
        try:
            return self._get_locale_faker(locale).iban()
        except Exception:
            return self._generate_dutch_iban()

    def _generate_us_routing(self) -> str:
        # ABA routing: 9-digit, first two digits are routing symbol prefix (01-12, 21-32, etc.)
        prefix = random.randint(11, 32)
        return f"{prefix:02d}{random.randint(1000000, 9999999)}"

    def _generate_us_account(self) -> str:
        return ''.join(str(random.randint(0, 9)) for _ in range(random.randint(8, 12)))

    def _generate_uk_sortcode(self) -> str:
        return f"{random.randint(10, 99):02d}-{random.randint(0, 99):02d}-{random.randint(0, 99):02d}"

    def _generate_uk_account(self) -> str:
        return f"{random.randint(10000000, 99999999)}"

    def _generate_au_bsb(self) -> str:
        # BSB: state-prefix (0-8) + 2 digits dash 3 digits
        state = random.randint(0, 8)
        return f"{state}{random.randint(10, 99)}-{random.randint(100, 999)}"

    def _generate_in_ifsc(self) -> str:
        # IFSC: 4-letter bank code + 0 + 6 alphanumeric branch
        banks = ['SBIN', 'HDFC', 'ICIC', 'AXIS', 'KOTK', 'IOBA', 'CNRB', 'PUNB', 'BKID', 'UBIN']
        branch = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
        return f"{random.choice(banks)}0{branch}"

    def _generate_clabe(self) -> str:
        # Mexican CLABE: 18 digits (bank 3 + city 3 + account 11 + check 1)
        return ''.join(str(random.randint(0, 9)) for _ in range(18))

    def _generate_ca_transit(self) -> str:
        # Canadian: 5-digit transit + dash + 3-digit institution
        return f"{random.randint(10000, 99999)}-{random.randint(100, 999)}"

    def _generate_sg_nric(self) -> str:
        prefix = random.choice(['S', 'T', 'F', 'G'])
        digits = ''.join(str(random.randint(0, 9)) for _ in range(7))
        suffix = random.choice('ABCDEFGHIZJ')
        return f"{prefix}{digits}{suffix}"

    # ------------------------------------------------------------------
    # National ID / tax number generators
    # ------------------------------------------------------------------
    def _generate_ssn(self) -> str:
        # US SSN: AAA-GG-SSSS
        area = random.randint(100, 899)
        group = random.randint(10, 99)
        serial = random.randint(1000, 9999)
        return f"{area:03d}-{group:02d}-{serial:04d}"

    def _generate_us_ein(self) -> str:
        prefix = random.randint(10, 99)
        return f"{prefix}-{random.randint(1000000, 9999999)}"

    def _generate_uk_ni(self) -> str:
        # NI: XX 99 99 99 A  (valid prefix pairs, no D,F,I,Q,U,V as first; no D,F,I,O,Q,U,V as second)
        valid_prefixes = ['AB', 'AE', 'AH', 'AK', 'AL', 'AM', 'AP', 'AR', 'AS', 'AT',
                          'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BE', 'BH', 'BK', 'BL',
                          'BM', 'BT', 'CA', 'CB', 'CE', 'CG', 'CH', 'CJ', 'CK', 'CL',
                          'CR', 'EA', 'EB', 'EC', 'EE', 'EG', 'EH', 'EJ', 'EK', 'EL']
        prefix = random.choice(valid_prefixes)
        nums = ''.join(str(random.randint(0, 9)) for _ in range(6))
        return f"{prefix} {nums[:2]} {nums[2:4]} {nums[4:6]} {random.choice('ABCD')}"

    def _generate_in_pan(self) -> str:
        # PAN: AAAAA9999A
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
        digits = ''.join(str(random.randint(0, 9)) for _ in range(4))
        return f"{letters}{digits}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"

    def _generate_in_aadhar(self) -> str:
        # Aadhaar: 12 digits, first digit 2-9
        first = random.randint(2, 9)
        rest = ''.join(str(random.randint(0, 9)) for _ in range(11))
        d = str(first) + rest
        return f"{d[:4]} {d[4:8]} {d[8:12]}"

    def _generate_au_tfn(self) -> str:
        return ''.join(str(random.randint(0, 9)) for _ in range(random.choice([8, 9])))

    def _generate_au_abn(self) -> str:
        d = ''.join(str(random.randint(0, 9)) for _ in range(11))
        return f"{d[:2]} {d[2:5]} {d[5:8]} {d[8:11]}"

    def _generate_br_cpf(self) -> str:
        d = ''.join(str(random.randint(0, 9)) for _ in range(11))
        return f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:11]}"

    def _generate_br_cnpj(self) -> str:
        d = ''.join(str(random.randint(0, 9)) for _ in range(14))
        return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:14]}"

    def _generate_fr_siren(self) -> str:
        return ''.join(str(random.randint(0, 9)) for _ in range(9))

    def _generate_fr_siret(self) -> str:
        siren = self._generate_fr_siren()
        nic = ''.join(str(random.randint(0, 9)) for _ in range(5))
        return f"{siren}{nic}"

    def _generate_de_tax(self) -> str:
        # German Steuernummer: 10-11 digits (state-dependent, simplified)
        return ''.join(str(random.randint(0, 9)) for _ in range(11))

    def _generate_za_id(self) -> str:
        yy = random.randint(0, 99)
        mm = random.randint(1, 12)
        dd = random.randint(1, 28)
        seq = random.randint(0, 9999)
        return f"{yy:02d}{mm:02d}{dd:02d}{seq:04d}{random.randint(1000, 9999)}"[:13]

    # ------------------------------------------------------------------
    # Passport / travel document generators
    # ------------------------------------------------------------------
    def _generate_passport(self, country_code: Optional[str] = None) -> str:
        cc = (country_code or random.choice(['US', 'GB', 'DE', 'FR', 'NL', 'AU', 'CA', 'IN', 'JP', 'CN'])).upper()
        if cc == 'US':
            return random.choice(string.ascii_uppercase) + ''.join(str(random.randint(0, 9)) for _ in range(8))
        if cc == 'GB':
            return ''.join(random.choices(string.ascii_uppercase, k=2)) + ''.join(str(random.randint(0, 9)) for _ in range(7))
        if cc in ('DE', 'NL', 'FR', 'AT', 'CH'):
            return ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
        if cc == 'AU':
            return random.choice(string.ascii_uppercase) + ''.join(str(random.randint(1000000, 9999999)))
        # Generic: letter(s) + 7 digits
        return random.choice(string.ascii_uppercase) + ''.join(str(random.randint(0, 9)) for _ in range(7))

    # ------------------------------------------------------------------
    # Driver's licence generators
    # ------------------------------------------------------------------
    def _generate_drivers_licence(self, country_code: Optional[str] = None) -> str:
        cc = (country_code or random.choice(['US', 'GB', 'DE', 'AU', 'CA', 'IN'])).upper()
        if cc == 'US':
            # Simplified generic US format (varies by state)
            return random.choice(string.ascii_uppercase) + ''.join(str(random.randint(0, 9)) for _ in range(7))
        if cc == 'GB':
            # DVLA: SURNAME(5) + YYMM + DD + 2 letters + 1 digit + 2 letters
            letters = ''.join(random.choices(string.ascii_uppercase, k=5))
            yy = random.randint(50, 99)
            mm = random.randint(1, 12)
            dd = random.randint(1, 28)
            return f"{letters}{yy:02d}{mm:02d}{dd:02d}{''.join(random.choices(string.ascii_uppercase, k=2))}{random.randint(1,9)}{''.join(random.choices(string.ascii_uppercase, k=2))}"
        if cc == 'DE':
            return ''.join(random.choices(string.ascii_uppercase, k=1)) + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        if cc == 'AU':
            return ''.join(str(random.randint(0, 9)) for _ in range(random.choice([8, 9])))
        if cc == 'CA':
            return random.choice(string.ascii_uppercase) + ''.join(str(random.randint(0, 9)) for _ in range(12))
        if cc == 'IN':
            state = random.choice(['DL', 'MH', 'KA', 'TN', 'UP', 'RJ', 'GJ'])
            return f"{state}{random.randint(10, 99)}{random.randint(10, 99)}{random.randint(1000000000, 9999999999)}"[:16]
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))

    # ------------------------------------------------------------------
    # VAT / tax registration generators
    # ------------------------------------------------------------------
    def _generate_eu_vat(self, country_code: Optional[str] = None) -> str:
        cc = (country_code or random.choice(['DE', 'FR', 'NL', 'GB', 'ES', 'IT', 'BE', 'AT', 'PL'])).upper()
        if cc == 'DE':
            return f"DE{random.randint(100000000, 999999999)}"
        if cc == 'FR':
            key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
            return f"FR{key}{random.randint(100000000, 999999999)}"
        if cc == 'NL':
            return f"NL{random.randint(100000000, 999999999)}B{random.randint(10, 99)}"
        if cc == 'GB':
            return f"GB{random.randint(100000000, 999999999)}"
        if cc == 'ES':
            return f"ES{''.join(random.choices(string.ascii_uppercase + string.digits, k=9))}"
        if cc == 'IT':
            return f"IT{random.randint(10000000000, 99999999999)}"
        if cc == 'BE':
            return f"BE{random.randint(1000000000, 9999999999)}"
        if cc == 'AT':
            return f"ATU{random.randint(10000000, 99999999)}"
        if cc == 'PL':
            return f"PL{random.randint(1000000000, 9999999999)}"
        return f"{cc}{random.randint(100000000, 999999999)}"

    def _generate_gst_in(self) -> str:
        # Indian GST: 15 chars = 2-digit state + PAN (10) + 1 + Z + checksum
        state = f"{random.randint(1, 37):02d}"
        pan = self._generate_in_pan()
        return f"{state}{pan}{random.randint(1, 9)}Z{random.choice(string.digits + string.ascii_uppercase)}"

    # ------------------------------------------------------------------
    # Network / digital identifiers
    # ------------------------------------------------------------------
    def _generate_ipv4(self) -> str:
        return '.'.join(str(random.randint(1, 254)) for _ in range(4))

    def _generate_ipv6(self) -> str:
        return ':'.join(f"{random.randint(0, 65535):04x}" for _ in range(8))

    def _generate_mac(self) -> str:
        return ':'.join(f"{random.randint(0, 255):02x}" for _ in range(6))

    def _generate_uuid(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _generate_url(self) -> str:
        try:
            return self.faker.url()
        except Exception:
            tlds = ['com', 'org', 'net', 'io', 'co']
            return f"https://www.{self.faker.word()}.{random.choice(tlds)}"

    # ------------------------------------------------------------------
    # Healthcare identifiers
    # ------------------------------------------------------------------
    def _generate_nhs_number(self) -> str:
        # NHS: 10 digits (simplified — actual check digit omitted for synthetic data)
        return ''.join(str(random.randint(0, 9)) for _ in range(10))

    def _generate_au_medicare(self) -> str:
        # Medicare card: 10 digits + 1 reference (IRN)
        d = ''.join(str(random.randint(0, 9)) for _ in range(10))
        return f"{d[:4]} {d[4:9]} {d[9]}-{random.randint(1, 9)}"

    def _generate_de_krankenversicherung(self) -> str:
        # GKV: 1 letter + 9 digits
        return random.choice(string.ascii_uppercase) + ''.join(str(random.randint(0, 9)) for _ in range(9))

    def _generate_us_npi(self) -> str:
        # NPI (National Provider Identifier): 10 digits
        return ''.join(str(random.randint(0, 9)) for _ in range(10))

    # ------------------------------------------------------------------
    # Cryptocurrency addresses
    # ------------------------------------------------------------------
    _BASE58_CHARS = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

    def _generate_btc_address(self) -> str:
        # P2PKH: starts with 1, 26-34 base58 chars
        length = random.randint(26, 34)
        return '1' + ''.join(random.choices(self._BASE58_CHARS, k=length - 1))

    def _generate_eth_address(self) -> str:
        return '0x' + ''.join(random.choices('0123456789abcdef', k=40))

    def _generate_sol_address(self) -> str:
        # Solana: 32-44 base58 chars
        return ''.join(random.choices(self._BASE58_CHARS, k=44))

    # ------------------------------------------------------------------
    # Product / barcode identifiers
    # ------------------------------------------------------------------
    def _generate_ean13(self) -> str:
        d = [random.randint(0, 9) for _ in range(12)]
        check = (10 - sum(v * (1 if i % 2 == 0 else 3) for i, v in enumerate(d)) % 10) % 10
        return ''.join(map(str, d)) + str(check)

    def _generate_ean8(self) -> str:
        d = [random.randint(0, 9) for _ in range(7)]
        check = (10 - sum(v * (3 if i % 2 == 0 else 1) for i, v in enumerate(d)) % 10) % 10
        return ''.join(map(str, d)) + str(check)

    def _generate_upc_a(self) -> str:
        d = [random.randint(0, 9) for _ in range(11)]
        check = (10 - sum(v * (3 if i % 2 == 0 else 1) for i, v in enumerate(d)) % 10) % 10
        return ''.join(map(str, d)) + str(check)

    def _generate_isbn13(self) -> str:
        prefix = random.choice(['978', '979'])
        group = str(random.randint(0, 9))
        publisher = ''.join(str(random.randint(0, 9)) for _ in range(4))
        title = ''.join(str(random.randint(0, 9)) for _ in range(4))
        raw = prefix + group + publisher + title
        d = [int(c) for c in raw]
        check = (10 - sum(v * (1 if i % 2 == 0 else 3) for i, v in enumerate(d)) % 10) % 10
        return f"{raw}-{check}"

    # ------------------------------------------------------------------
    # Chinese Unified Social Credit Code (USCC)
    # ------------------------------------------------------------------
    _USCC_CHARSET = '0123456789ABCDEFGHJKLMNPQRTUWXY'

    def _generate_cn_uscc(self) -> str:
        # 18 chars: 1-digit reg type + 6-digit admin code + 9-char org code + 1 check char
        reg_type = random.choice('1259')
        admin_code = ''.join(str(random.randint(0, 9)) for _ in range(6))
        org_code = ''.join(random.choices(self._USCC_CHARSET, k=9))
        check = random.choice(self._USCC_CHARSET)
        return f"{reg_type}{admin_code}{org_code}{check}"

    # ------------------------------------------------------------------
    # Special rules dispatcher
    # ------------------------------------------------------------------
    def generate_special_value(
        self,
        special_rule: str,
        data_type: str,
        column_name: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Any:
        """
        Generate values based on special rules.

        Dutch/European (original):
          NL_IBAN, EU_IBAN, EU_IBAN:CC, BBAN, BBAN:CC, BIC/SWIFT, NL_PHONE
          NAME, FIRST_NAME, LAST_NAME, COMPANY, ADDRESS, EMAIL, PHONE
          CITY_NM, CTY_CODE, PST_CODE, COUNTRY_CODE, CURRENCY_CODE/ISO4217

        Locale-parameterised (append :<locale> to any personal/address rule):
          NAME:de_DE, FIRST_NAME:ja_JP, LAST_NAME:zh_CN, COMPANY:fr_FR
          ADDRESS:en_GB, PHONE:en_US, EMAIL:es_ES, POSTCODE:de_DE, CITY:pt_BR

        Global random (random locale each call):
          GLOBAL_NAME, GLOBAL_FIRST_NAME, GLOBAL_LAST_NAME, GLOBAL_FULL_NAME
          GLOBAL_COMPANY, GLOBAL_ADDRESS, GLOBAL_PHONE, GLOBAL_EMAIL
          GLOBAL_POSTCODE, GLOBAL_CITY, GLOBAL_COUNTRY_CODE

        International banking:
          IBAN          – random IBAN-country IBAN
          IBAN:CC       – IBAN for specific country (e.g. IBAN:DE)
          US_ROUTING    – ABA routing number (9 digits)
          US_ACCOUNT    – US bank account (8-12 digits)
          UK_SORTCODE   – UK sort code (XX-XX-XX)
          UK_ACCOUNT    – UK bank account (8 digits)
          AU_BSB        – Australian BSB (XXX-XXX)
          AU_ACCOUNT    – Australian account number
          IN_IFSC       – Indian IFSC code
          IN_ACCOUNT    – Indian bank account (9-18 digits)
          CLABE         – Mexican CLABE (18 digits)
          CA_TRANSIT    – Canadian transit+institution
          SG_NRIC       – Singapore NRIC

        National ID / tax numbers:
          SSN / US_SSN  – US Social Security Number
          EIN / US_EIN  – US Employer Identification Number
          NI_NUMBER / UK_NI – UK National Insurance
          PAN / IN_PAN  – Indian Permanent Account Number
          AADHAR        – Indian Aadhaar (12 digits)
          TFN / AU_TFN  – Australian Tax File Number
          ABN / AU_ABN  – Australian Business Number
          CPF / BR_CPF  – Brazilian CPF
          CNPJ / BR_CNPJ – Brazilian CNPJ
          SIREN / FR_SIREN – French company SIREN (9 digits)
          SIRET / FR_SIRET – French company SIRET (14 digits)
          DE_STEUER     – German Steuernummer
          ZA_ID         – South African ID number
        """
        if not special_rule or pd.isna(special_rule):
            return None
        rule = str(special_rule).strip().upper()
        col = (column_name or '').lower()
        _ = data_type, column_name
        parsed = self.parse_special_rule_config(special_rule)
        raw_rule = parsed['value_rule']

        if not raw_rule:
            return None

        if self.is_regex_rule(raw_rule):
            return self.generate_from_regex_rule(raw_rule, max_length=max_length)

        rule = raw_rule.upper()

        # Parse optional locale suffix from raw_rule (preserves case: de_DE, ja_JP)
        # Locale format:  xx_XX (e.g. de_DE)  — distinguished from country codes (e.g. DE)
        # by presence of underscore or being lowercase
        _has_colon = ':' in raw_rule
        if _has_colon:
            _raw_base, _raw_suffix = raw_rule.split(':', 1)
            _raw_suffix = _raw_suffix.strip()
            _is_locale_sfx = bool(re.match(r'^[a-zA-Z]{2,3}_[a-zA-Z]{2,4}$', _raw_suffix))
            _rule_base = _raw_base.upper()
            _rule_locale = self._normalize_locale(_raw_suffix) if _is_locale_sfx else None
        else:
            _rule_base = rule
            _rule_locale = None
        _lf = self._get_locale_faker(_rule_locale) if _rule_locale else self.faker

        # Mimesis-namespaced rules — opt-in via the MIMESIS_ prefix.
        # Existing rules (NAME, EMAIL, ...) keep routing to Faker; only rules
        # explicitly starting with MIMESIS_ delegate here. The library is an
        # optional dependency, so users who don't author MIMESIS_* rules pay
        # nothing — the import only happens on first dispatch.
        if _rule_base.startswith('MIMESIS_'):
            from sdp.utils import mimesis_provider
            return mimesis_provider.generate(_rule_base[len('MIMESIS_'):], _rule_locale)

        # IBANs
        if rule == 'NL_IBAN':
            return self._generate_dutch_iban()
        if rule.startswith('EU_IBAN'):
            parts = rule.split(':', 1)
            prefer = parts[1].strip().upper() if len(parts) > 1 else None
            return self._generate_european_iban(prefer)

        # BBANs (country-aware)
        if rule.startswith('BBAN'):
            parts = rule.split(':', 1)
            cc = parts[1].strip().upper() if len(parts) > 1 else None
            return self._generate_bban(cc)

        # BIC
        if rule == 'BIC':
            return self._generate_valid_bic()

        # Contact / personal  (support optional :locale suffix on all of these)
        if rule == 'EMAIL' or _rule_base == 'EMAIL':
            return self._truncate_text(_lf.email(), max_length)
        if rule == 'PHONE' or _rule_base == 'PHONE':
            if _rule_locale:
                return self._truncate_text(_lf.phone_number(), max_length)
            return self._truncate_text(self._generate_dutch_phone(), max_length)
        if rule in {'NL_PHONE', 'DUTCH_PHONE'}:
            return self._truncate_text(self._generate_dutch_phone(), max_length)
        if rule in {'COMPANY', 'COMPANY_NM', 'ORGANISATION', 'ORG_NM', 'COUNTERPARTY'} \
                or _rule_base in {'COMPANY', 'COMPANY_NM', 'ORGANISATION', 'ORG_NM', 'COUNTERPARTY'}:
            return self._truncate_text(_lf.company(), max_length)
        if rule in {'FIRST_NAME', 'GIVEN_NAME', 'VOORNAAM'} \
                or _rule_base in {'FIRST_NAME', 'GIVEN_NAME'}:
            return self._truncate_text(_lf.first_name(), max_length)
        if rule in {'LAST_NAME', 'SURNAME', 'FAMILY_NAME', 'ACHTERNAAM'} \
                or _rule_base in {'LAST_NAME', 'SURNAME', 'FAMILY_NAME'}:
            return self._truncate_text(_lf.last_name(), max_length)
        if rule in {'NAME', 'PERSON_NAME', 'FULL_NAME'} \
                or _rule_base in {'NAME', 'PERSON_NAME', 'FULL_NAME'}:
            if self._looks_like_company_field(col):
                return self._truncate_text(_lf.company(), max_length)
            return self._truncate_text(_lf.name(), max_length)
        if rule == 'ADDRESS' or _rule_base == 'ADDRESS':
            if _rule_locale:
                return self._truncate_text(_lf.address(), max_length)
            components = self.generate_nl_address_components()
            if 'adr_line1' in col or 'address_line1' in col:
                return self._single_line_address(components, 1, max_length)
            if 'adr_line2' in col or 'address_line2' in col:
                return self._single_line_address(components, 2, max_length)
            return self._truncate_text(self.format_nl_address(components), max_length)
        if rule == 'POSTCODE' or _rule_base == 'POSTCODE':
            return self._truncate_text(_lf.postcode(), max_length)
        if rule == 'CITY' or _rule_base == 'CITY':
            return self._truncate_text(_lf.city(), max_length)
        if rule == 'BANK_ACCOUNT':
            return self._truncate_text(self._generate_dutch_bank_account(), max_length)
        if rule == 'BANK_NAME':
            return self._truncate_text(random.choice(list(self.dutch_banks.values())), max_length)
        if rule == 'BANK_CODE':
            raw = random.choice(list(self.dutch_banks.keys()))
            code4 = "".join(ch for ch in raw if ch.isalpha()).upper()[:4]
            return self._truncate_text((code4 + "X" * 4)[:4] if len(code4) < 4 else code4, max_length)

        # Country/city/postcode (Dutch originals kept for backward compat)
        if rule == 'COUNTRY_CODE':
            return self._truncate_text(self.generate_country_code(prefer=['NL', 'BE', 'DE', 'FR', 'US']), max_length)
        if rule == 'CITY_NM':
            return self._truncate_text(self.generate_nl_city_nm(), max_length)
        if rule == 'CTY_CODE':
            return self._truncate_text(self.generate_nl_cty_code(), max_length)
        if rule == 'PST_CODE':
            return self._truncate_text(self.generate_nl_pst_code(), max_length)

        # Currency
        if rule in {'CURRENCY_CODE', 'CURR_CODE', 'ISO_CURRENCY', 'ISO4217', 'CURR'}:
            return self._truncate_text(self.generate_currency_code(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # GLOBAL RANDOM-LOCALE RULES
        # ─────────────────────────────────────────────────────────────────────
        if rule == 'GLOBAL_NAME':
            f = self._random_global_faker()
            return self._truncate_text(f.company() if self._looks_like_company_field(col) else f.name(), max_length)
        if rule == 'GLOBAL_FIRST_NAME':
            return self._truncate_text(self._random_global_faker().first_name(), max_length)
        if rule == 'GLOBAL_LAST_NAME':
            return self._truncate_text(self._random_global_faker().last_name(), max_length)
        if rule in {'GLOBAL_FULL_NAME', 'GLOBAL_PERSON_NAME'}:
            return self._truncate_text(self._random_global_faker().name(), max_length)
        if rule == 'GLOBAL_COMPANY':
            return self._truncate_text(self._random_global_faker().company(), max_length)
        if rule == 'GLOBAL_ADDRESS':
            return self._truncate_text(self._random_global_faker().address(), max_length)
        if rule == 'GLOBAL_PHONE':
            return self._truncate_text(self._random_global_faker().phone_number(), max_length)
        if rule == 'GLOBAL_EMAIL':
            return self._truncate_text(self._random_global_faker().email(), max_length)
        if rule == 'GLOBAL_POSTCODE':
            return self._truncate_text(self._random_global_faker().postcode(), max_length)
        if rule == 'GLOBAL_CITY':
            return self._truncate_text(self._random_global_faker().city(), max_length)
        if rule == 'GLOBAL_COUNTRY_CODE':
            return self._truncate_text(self.generate_country_code(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # INTERNATIONAL BANKING
        # ─────────────────────────────────────────────────────────────────────
        if rule == 'IBAN' or rule.startswith('IBAN:'):
            cc = rule.split(':', 1)[1].strip() if ':' in rule else None
            return self._truncate_text(self._generate_iban(cc), max_length)
        if rule in {'SWIFT', 'BIC_SWIFT', 'SWIFT_CODE'}:
            return self._truncate_text(self._generate_valid_bic(), max_length)
        if rule in {'US_ROUTING', 'ABA_ROUTING', 'US_ABA'}:
            return self._truncate_text(self._generate_us_routing(), max_length)
        if rule in {'US_ACCOUNT', 'US_BANK_ACCOUNT'}:
            return self._truncate_text(self._generate_us_account(), max_length)
        if rule in {'UK_SORTCODE', 'GB_SORTCODE', 'SORT_CODE'}:
            return self._truncate_text(self._generate_uk_sortcode(), max_length)
        if rule in {'UK_ACCOUNT', 'GB_ACCOUNT', 'UK_BANK_ACCOUNT'}:
            return self._truncate_text(self._generate_uk_account(), max_length)
        if rule in {'AU_BSB', 'BSB'}:
            return self._truncate_text(self._generate_au_bsb(), max_length)
        if rule in {'AU_ACCOUNT', 'AU_BANK_ACCOUNT'}:
            return self._truncate_text(self._generate_us_account(), max_length)  # same pattern: 8-12 digits
        if rule in {'IN_IFSC', 'IFSC'}:
            return self._truncate_text(self._generate_in_ifsc(), max_length)
        if rule in {'IN_ACCOUNT', 'IN_BANK_ACCOUNT'}:
            return self._truncate_text(''.join(str(random.randint(0, 9)) for _ in range(random.randint(9, 18))), max_length)
        if rule in {'CLABE', 'MX_CLABE'}:
            return self._truncate_text(self._generate_clabe(), max_length)
        if rule in {'CA_TRANSIT', 'CA_BANK'}:
            return self._truncate_text(self._generate_ca_transit(), max_length)
        if rule in {'SG_NRIC', 'NRIC'}:
            return self._truncate_text(self._generate_sg_nric(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # NATIONAL ID / TAX NUMBERS
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'SSN', 'US_SSN', 'SOCIAL_SECURITY'}:
            return self._truncate_text(self._generate_ssn(), max_length)
        if rule in {'EIN', 'US_EIN', 'EMPLOYER_ID'}:
            return self._truncate_text(self._generate_us_ein(), max_length)
        if rule in {'NI_NUMBER', 'UK_NI', 'GB_NI', 'NATIONAL_INSURANCE'}:
            return self._truncate_text(self._generate_uk_ni(), max_length)
        if rule in {'PAN', 'IN_PAN', 'INDIA_PAN'}:
            return self._truncate_text(self._generate_in_pan(), max_length)
        if rule in {'AADHAR', 'AADHAAR', 'IN_AADHAR'}:
            return self._truncate_text(self._generate_in_aadhar(), max_length)
        if rule in {'TFN', 'AU_TFN', 'TAX_FILE_NUMBER'}:
            return self._truncate_text(self._generate_au_tfn(), max_length)
        if rule in {'ABN', 'AU_ABN', 'AUSTRALIAN_BUSINESS'}:
            return self._truncate_text(self._generate_au_abn(), max_length)
        if rule in {'CPF', 'BR_CPF'}:
            return self._truncate_text(self._generate_br_cpf(), max_length)
        if rule in {'CNPJ', 'BR_CNPJ'}:
            return self._truncate_text(self._generate_br_cnpj(), max_length)
        if rule in {'SIREN', 'FR_SIREN'}:
            return self._truncate_text(self._generate_fr_siren(), max_length)
        if rule in {'SIRET', 'FR_SIRET'}:
            return self._truncate_text(self._generate_fr_siret(), max_length)
        if rule in {'DE_STEUER', 'DE_STEUERNUMMER'}:
            return self._truncate_text(self._generate_de_tax(), max_length)
        if rule in {'ZA_ID', 'SA_ID', 'ZA_ID_NUMBER'}:
            return self._truncate_text(self._generate_za_id(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # PASSPORT / TRAVEL DOCUMENTS
        # ─────────────────────────────────────────────────────────────────────
        if rule == 'PASSPORT' or rule.startswith('PASSPORT:'):
            cc = rule.split(':', 1)[1].strip() if ':' in rule else None
            return self._truncate_text(self._generate_passport(cc), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # DRIVER'S LICENCE
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'DRIVERS_LICENCE', 'DRIVERS_LICENSE', 'DL', 'DRIVING_LICENCE'} \
                or rule.startswith('DL:'):
            cc = rule.split(':', 1)[1].strip() if ':' in rule else None
            return self._truncate_text(self._generate_drivers_licence(cc), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # VAT / GST REGISTRATION NUMBERS
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'EU_VAT', 'VAT'} or rule.startswith('EU_VAT:') or rule.startswith('VAT:'):
            cc = rule.split(':', 1)[1].strip() if ':' in rule else None
            return self._truncate_text(self._generate_eu_vat(cc), max_length)
        if rule in {'GST_IN', 'IN_GST', 'GSTIN'}:
            return self._truncate_text(self._generate_gst_in(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # NETWORK / DIGITAL IDENTIFIERS
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'IPV4', 'IP_ADDRESS', 'IP'}:
            return self._truncate_text(self._generate_ipv4(), max_length)
        if rule in {'IPV6', 'IP6'}:
            return self._truncate_text(self._generate_ipv6(), max_length)
        if rule in {'MAC_ADDRESS', 'MAC'}:
            return self._truncate_text(self._generate_mac(), max_length)
        if rule in {'UUID', 'GUID'}:
            return self._truncate_text(self._generate_uuid(), max_length)
        if rule in {'URL', 'WEBSITE'}:
            return self._truncate_text(self._generate_url(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # HEALTHCARE IDENTIFIERS
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'NHS_NUMBER', 'NHS', 'UK_NHS'}:
            return self._truncate_text(self._generate_nhs_number(), max_length)
        if rule in {'AU_MEDICARE', 'MEDICARE'}:
            return self._truncate_text(self._generate_au_medicare(), max_length)
        if rule in {'DE_GKV', 'DE_KRANKEN', 'GKV'}:
            return self._truncate_text(self._generate_de_krankenversicherung(), max_length)
        if rule in {'NPI', 'US_NPI'}:
            return self._truncate_text(self._generate_us_npi(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # CRYPTOCURRENCY ADDRESSES
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'BTC_ADDRESS', 'BITCOIN', 'BTC'}:
            return self._truncate_text(self._generate_btc_address(), max_length)
        if rule in {'ETH_ADDRESS', 'ETHEREUM', 'ETH'}:
            return self._truncate_text(self._generate_eth_address(), max_length)
        if rule in {'SOL_ADDRESS', 'SOLANA', 'SOL'}:
            return self._truncate_text(self._generate_sol_address(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # PRODUCT / BARCODE IDENTIFIERS
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'EAN13', 'EAN_13', 'BARCODE'}:
            return self._truncate_text(self._generate_ean13(), max_length)
        if rule in {'EAN8', 'EAN_8'}:
            return self._truncate_text(self._generate_ean8(), max_length)
        if rule in {'UPC_A', 'UPCA', 'UPC'}:
            return self._truncate_text(self._generate_upc_a(), max_length)
        if rule in {'ISBN13', 'ISBN_13', 'ISBN'}:
            return self._truncate_text(self._generate_isbn13(), max_length)

        # ─────────────────────────────────────────────────────────────────────
        # CHINESE UNIFIED SOCIAL CREDIT CODE
        # ─────────────────────────────────────────────────────────────────────
        if rule in {'CN_USCC', 'USCC', 'CN_CREDIT_CODE'}:
            return self._truncate_text(self._generate_cn_uscc(), max_length)

        return None

    # ------------------------------------------------------------------
    # Column-aware realistic generator fallback
    # ------------------------------------------------------------------
    def generate_realistic_dutch_data(
        self,
        column_name: str,
        data_type: str,
        max_length: Optional[int] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Any:
        """Heuristic generator using column name patterns (NL-centric and bounded by data-type length)."""
        col = (column_name or '').lower()
        base_type = self._infer_base_type(data_type)

        if base_type in {'D', 'DT', 'TS', 'N', 'DC'}:
            return self.generate_sample_value(data_type, {'min_value': min_val, 'max_value': max_val})

        if 'iban' in col:
            return self._truncate_text(self._generate_dutch_iban(), max_length)
        if 'bban' in col:
            return self._truncate_text(self._generate_bban('NL'), max_length)
        if any(k in col for k in ['account', 'acct']) and any(k in col for k in ['id', 'nr', 'number', 'ref']):
            return self._truncate_text(self._generate_dutch_iban() if (max_length or 0) >= 18 else self._generate_dutch_bank_account(), max_length)
        if any(k in col for k in ['bank_name', 'bank_nm']):
            return self._truncate_text(random.choice(list(self.dutch_banks.values())), max_length)
        if 'bic' in col or 'swift' in col:
            return self._truncate_text(self._generate_valid_bic(), max_length)
        if 'phone' in col or 'telefoon' in col:
            return self._truncate_text(self._generate_dutch_phone(), max_length)
        if 'email' in col or 'mail' in col:
            return self._truncate_text(self.faker.email(), max_length)

        if any(k in col for k in ['adr_line1', 'address_line1', 'adresregel1']):
            return self._single_line_address(self.generate_nl_address_components(), 1, max_length)
        if any(k in col for k in ['adr_line2', 'address_line2', 'adresregel2']):
            return self._single_line_address(self.generate_nl_address_components(), 2, max_length)
        if 'address' in col or 'adres' in col:
            return self._truncate_text(self.format_nl_address(self.generate_nl_address_components()), max_length)

        if self._looks_like_name_field(col):
            if self._looks_like_company_field(col) or any(k in col for k in ['cdtr', 'merchant', 'scheme', 'orgtr']):
                return self._truncate_text(self.faker.company(), max_length)
            return self._truncate_text(self.faker.name(), max_length)

        # NL fields
        if any(k in col for k in ['city_nm', 'city', 'stad', 'plaats']):
            return self._truncate_text(self.generate_nl_city_nm(), max_length)
        if any(k in col for k in ['cty_code', 'country_code', 'ctry', 'land_code', 'land']):
            return self._truncate_text(self.generate_nl_cty_code(), max_length)
        if any(k in col for k in ['pst_code', 'postcode', 'post_code', 'zip']):
            return self._truncate_text(self.generate_nl_pst_code(), max_length)

        # Currency fields
        if any(k in col for k in ['currency', 'curr', 'ccy', 'currency_code', 'curr_code', 'iso_currency']):
            return self._truncate_text(self.generate_currency_code(), max_length)

        # Country generic
        if 'country' in col or 'land' in col or 'ctry' in col:
            return self._truncate_text(self.generate_country_code(prefer=['NL', 'BE', 'DE', 'FR']), max_length)

        if any(k in col for k in ['code', '_cd', 'status', 'type', 'flag', 'ind', 'channel', 'system', 'instrm']):
            return self._truncate_text(self._select_code_value(column_name, max_length), max_length)

        if any(k in col for k in ['ref', 'identifier', 'msg_id', 'mndt_id', 'tx_id']) or col.endswith('_id'):
            return self._generate_structured_reference(column_name, max_length)

        if base_type == 'NS':
            length = max_length or 15
            return self._generate_numeric_string(length)

        if base_type in {'VA', 'A', 'AN', 'T'}:
            return self._generate_sentence_like_text(max_length or 64)

        # Datetime-like types handled elsewhere by callers; fallback:
        return self.generate_sample_value(data_type, {'min_value': min_val, 'max_value': max_val})

    # ------------------------------------------------------------------
    # Generic sampler (legacy paths)
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_safe_timestamp(value: Any, normalize: bool = False) -> Optional[pd.Timestamp]:
        try:
            parsed = pd.to_datetime(value, errors='coerce')
        except Exception:
            return None

        if pd.isna(parsed):
            return None

        ts = pd.Timestamp(parsed)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)

        if ts < SAFE_DATETIME_MIN or ts > SAFE_DATETIME_MAX:
            return None

        return ts.normalize() if normalize else ts

    def generate_column_batch(self, config: Dict[str, Any], n: int) -> List[Any]:
        """Generate *n* values for a non-PK column using the fastest available path.

        Priority:
          1. Business values          → random.choices (O(n) lookup)
          2. Distribution sampling    → numpy vectorised
          3. Numeric/decimal ranges   → numpy vectorised
          4. Special rules            → list comprehension over Faker/generators
          5. Fallback                 → per-value generate_sample_value calls
        """
        data_type    = (config.get("data_type") or "VA256").upper()
        business_val = config.get("business_values")
        special_rule = config.get("special_rules")
        min_val      = config.get("min_value")
        max_val      = config.get("max_value")
        distribution = config.get("distribution")

        # 1. Business values — fastest path
        if business_val:
            bv_list = self.parse_business_values(business_val)
            if bv_list:
                return random.choices(bv_list, k=n)

        # 2. Distribution-based numeric sampling (statistically realistic)
        if distribution and (data_type.startswith("N") or data_type == "DC"):
            try:
                from sdp.ml.distribution_fitter import DistributionFitter
                lo = float(min_val) if min_val is not None else None
                hi = float(max_val) if max_val is not None else None
                arr = DistributionFitter().sample(distribution, n, min_val=lo, max_val=hi)
                if data_type.startswith("N"):
                    return [int(round(v)) for v in arr]
                return [round(float(v), 2) for v in arr]
            except (ImportError, AttributeError, KeyError, TypeError,
                    ValueError, ArithmeticError) as exc:
                # A distribution descriptor that scipy cannot sample is a
                # config-quality problem, so it is worth a line — but it must
                # not stop generation: plain range sampling below is a valid
                # substitute.
                logger.warning(
                    "Distribution sampling failed (%s: %s) — falling back to "
                    "uniform range generation", type(exc).__name__, exc,
                )

        # 3. Numeric / decimal ranges — numpy vectorised
        if data_type.startswith("N"):
            length_str = data_type[1:] if len(data_type) > 1 else ""
            length = int(length_str) if length_str.isdigit() else 10
            max_cap = int(min(float(max_val) if max_val is not None else 10 ** length, 10 ** 8))
            min_bound = int(float(min_val) if min_val is not None else 1)
            if min_bound >= max_cap:
                max_cap = min_bound + 1
            return np.random.randint(min_bound, max_cap + 1, n).tolist()

        if data_type == "DC":
            lo = float(min_val) if min_val is not None else 1.0
            hi = float(max_val) if max_val is not None else 1000.0
            return [round(v, 2) for v in np.random.uniform(lo, hi, n).tolist()]

        # 4. Special rules — list comprehension (avoids per-call dict overhead)
        if special_rule:
            return [
                self.generate_special_value(special_rule, data_type) or ""
                for _ in range(n)
            ]

        # 5. Fallback — per-value (date types, text, edge cases)
        return [self.generate_sample_value(data_type, config) for _ in range(n)]

    def generate_sample_value(self, data_type: str, config: Dict[str, Any]) -> Any:
        business_values = config.get('business_values')
        special_rules = config.get('special_rules')
        min_val = config.get('min_value')
        max_val = config.get('max_value')
        column_name = config.get('column_name')
        max_length = config.get('max_length')

        if business_values:
            selected = random.choice(business_values)
            if data_type in ['D', 'DT', 'TS']:
                parsed = self._coerce_safe_timestamp(selected, normalize=(data_type == 'D'))
                if parsed is not None:
                    return parsed
            else:
                return selected

        if special_rules:
            sv = self.generate_special_value(
                special_rules,
                data_type,
                column_name=column_name,
                max_length=max_length,
            )
            if sv is not None:
                return sv

        # Simple generators by data_type signature
        if isinstance(data_type, str) and data_type.upper().startswith('N'):
            length_str = data_type[1:] if len(data_type) > 1 else ''
            length = int(length_str) if length_str.isdigit() else 10
            max_cap = min(max_val if max_val else 10 ** length, 10 ** 8)
            min_bound = min_val if min_val else 1
            return random.randint(int(min_bound), int(max_cap))

        elif isinstance(data_type, str) and data_type.upper() == 'DC':
            lo = float(min_val) if min_val is not None else 1.0
            hi = float(max_val) if max_val is not None else 1000.0
            return round(random.uniform(lo, hi), 2)

        elif data_type in ['D', 'DT', 'TS']:
            date_only = (data_type == 'D')
            if min_val is not None or max_val is not None:
                lo = self._coerce_safe_timestamp(min_val, normalize=date_only) if min_val is not None else None
                hi = self._coerce_safe_timestamp(max_val, normalize=date_only) if max_val is not None else None
                lo = lo if lo is not None else pd.Timestamp('2000-01-01')
                hi = hi if hi is not None else pd.Timestamp('2030-12-31')
                if lo > hi:
                    lo, hi = hi, lo
                span_seconds = max(int((hi - lo).total_seconds()), 1)
                offset = timedelta(seconds=random.randint(0, span_seconds))
                generated = lo + offset
                return generated.normalize() if date_only else generated
            days_offset = random.randint(-365, 0)
            rt = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            generated = pd.Timestamp(datetime.now() + timedelta(days=days_offset) + rt)
            return generated.normalize() if date_only else generated

        elif isinstance(data_type, str) and data_type.upper().startswith('VA'):
            length_str = data_type[2:] if len(data_type) > 2 else ''
            length = int(length_str) if length_str.isdigit() else 255
            if length <= 50:
                return self.faker.word()[:length]
            return self.faker.text(max_nb_chars=min(length, 200))

        elif isinstance(data_type, str) and data_type.upper().startswith('A'):
            length_str = data_type[1:] if len(data_type) > 1 else ''
            length = int(length_str) if length_str.isdigit() else 1
            return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=length))

        else:
            return self.faker.word()

    # ------------------------------------------------------------------
    # SDV mapping (unchanged semantic)
    # ------------------------------------------------------------------
    def map_to_sdv_type(
        self,
        data_type: str,
        column_name: str,
        business_values: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Map custom data types to SDV sdtype hints."""
        dt = (data_type or '').upper()
        name_lower = (column_name or '').lower()
        mapping: Dict[str, Any] = {'sdtype': 'categorical'}

        if business_values:
            mapping['sdtype'] = 'categorical'
        elif dt.startswith('N') or dt == 'DC':
            mapping['sdtype'] = 'numerical'
        elif dt in ['D', 'DT', 'TS']:
            mapping['sdtype'] = 'datetime'
        elif dt in ['T']:
            mapping['sdtype'] = 'categorical'
        elif name_lower == 'id' or name_lower.endswith('_id') or name_lower.endswith('uuid'):
            mapping['sdtype'] = 'id'
        else:
            mapping['sdtype'] = 'text'

        return mapping