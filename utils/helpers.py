
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from faker import Faker

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
    def __init__(self):
        # Dutch locale for realistic data
        self.faker = Faker('nl_NL')

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
        codes = list(CURRENCY_CODE_TO_NAME.keys())
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
        except Exception:
            pass
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

    # ------------------------------------------------------------------
    # Special rules dispatcher
    # ------------------------------------------------------------------
    def generate_special_value(self, special_rule: str, data_type: str) -> Any:
        """
        Generate values based on special rules (Dutch banking + address + country/currency codes).
        Supported (key ones):
          - 'NL_IBAN'     -> Dutch IBAN (18 chars)  (NL + 2 + 4!a + 10!n)
          - 'EU_IBAN'     -> Random EU IBAN from {NL,DE,BE,FR,ES,IT}
          - 'EU_IBAN:CC'  -> IBAN for specific country code (e.g., 'EU_IBAN:DE')
          - 'BBAN'        -> Country-aware BBAN with NL bias
          - 'BBAN:CC'     -> BBAN for specific country code
          - 'BIC'         -> Valid-looking BIC (8+3)
          - 'CITY_NM', 'CTY_CODE', 'PST_CODE'
          - 'CURRENCY_CODE' | 'CURR_CODE' | 'ISO_CURRENCY' | 'ISO4217' | 'CURR'
        """
        if not special_rule or pd.isna(special_rule):
            return None
        rule = str(special_rule).strip().upper()

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

        # Contact / personal
        if rule == 'EMAIL':
            return self.faker.email()
        if rule == 'PHONE':
            return self._generate_dutch_phone()
        if rule == 'NAME':
            return self.faker.name()
        if rule == 'ADDRESS':
            return self.format_nl_address(self.generate_nl_address_components())
        if rule == 'BANK_ACCOUNT':
            return self._generate_dutch_bank_account()
        if rule == 'BANK_NAME':
            return random.choice(list(self.dutch_banks.values()))
        if rule == 'BANK_CODE':
            # Return a normalized 4-letter bank code
            raw = random.choice(list(self.dutch_banks.keys()))
            code4 = "".join(ch for ch in raw if ch.isalpha()).upper()[:4]
            return (code4 + "X" * 4)[:4] if len(code4) < 4 else code4

        # Country/city/postcode
        if rule == 'COUNTRY_CODE':
            return self.generate_country_code(prefer=['NL', 'BE', 'DE', 'FR', 'US'])
        if rule == 'CITY_NM':
            return self.generate_nl_city_nm()
        if rule == 'CTY_CODE':
            return self.generate_nl_cty_code()
        if rule == 'PST_CODE':
            return self.generate_nl_pst_code()

        # Currency
        if rule in {'CURRENCY_CODE', 'CURR_CODE', 'ISO_CURRENCY', 'ISO4217', 'CURR'}:
            return self.generate_currency_code()

        return None

    # ------------------------------------------------------------------
    # Column-aware realistic generator fallback
    # ------------------------------------------------------------------
    def generate_realistic_dutch_data(self, column_name: str, data_type: str) -> Any:
        """Heuristic generator using column name patterns (NL-centric)."""
        col = (column_name or '').lower()
        if any(k in col for k in ['iban', 'account', 'bank']):
            return self._generate_dutch_iban()
        if 'bic' in col or 'swift' in col:
            return self._generate_valid_bic()
        if 'phone' in col or 'telefoon' in col:
            return self._generate_dutch_phone()
        if 'address' in col or 'adres' in col:
            return self.format_nl_address(self.generate_nl_address_components())
        if 'name' in col or 'naam' in col:
            if 'debit' in col or 'dbtr' in col:
                return f"{self.faker.first_name()} {self.faker.last_name()}"
            if 'credit' in col or 'cdtr' in col:
                return self.faker.company()
            return self.faker.name()

        # NL fields
        if any(k in col for k in ['city_nm', 'city', 'stad', 'plaats']):
            return self.generate_nl_city_nm()
        if any(k in col for k in ['cty_code', 'country_code', 'ctry', 'land_code', 'land']):
            return self.generate_nl_cty_code()
        if any(k in col for k in ['pst_code', 'postcode', 'post_code', 'zip']):
            return self.generate_nl_pst_code()

        # Currency fields
        if any(k in col for k in ['currency', 'curr', 'ccy', 'currency_code', 'curr_code', 'iso_currency']):
            return self.generate_currency_code()

        # Country generic
        if 'country' in col or 'land' in col or 'ctry' in col:
            return self.generate_country_code(prefer=['NL', 'BE', 'DE', 'FR'])

        # Datetime-like types handled elsewhere by callers; fallback:
        return self.generate_sample_value(data_type, {})

    # ------------------------------------------------------------------
    # Generic sampler (legacy paths)
    # ------------------------------------------------------------------
    def generate_sample_value(self, data_type: str, config: Dict[str, Any]) -> Any:
        business_values = config.get('business_values')
        special_rules = config.get('special_rules')
        min_val = config.get('min_value')
        max_val = config.get('max_value')

        if business_values:
            selected = random.choice(business_values)
            if data_type in ['D', 'DT', 'TS']:
                parsed = pd.to_datetime(selected, errors='coerce')
                if not pd.isna(parsed):
                    return parsed
            return selected

        if special_rules:
            sv = self.generate_special_value(special_rules, data_type)
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
            # Produce pandas Timestamp for consistency
            days_offset = random.randint(-365, 0)
            rt = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
            return pd.Timestamp(datetime.now() + timedelta(days=days_offset) + rt)

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

        if dt.startswith('N') or dt == 'DC':
            mapping['sdtype'] = 'numerical'
        elif dt in ['D', 'DT', 'TS']:
            mapping['sdtype'] = 'datetime'
        elif dt in ['T']:
            mapping['sdtype'] = 'categorical'
        elif any(k in name_lower for k in ['key', 'id', 'code']):
            mapping['sdtype'] = 'id'
        elif business_values:
            mapping['sdtype'] = 'categorical'
            mapping['order_by'] = business_values
        else:
            mapping['sdtype'] = 'text'

        return mapping