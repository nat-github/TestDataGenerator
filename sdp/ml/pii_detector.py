"""PII / sensitive column detection.

Rule-based scanner that matches column names and sampled values against
known PII patterns. Zero extra dependencies — uses only stdlib re and pandas.

Output: List[PIIFinding], each carrying column name, PII type, confidence
score (0–1), suggested special_rule keyword, and severity level.

Integrates with:
  - auto_config.AutoConfigInferrer  (suggests special_rules during inference)
  - CLI `pii-scan` command           (standalone scan on data files or configs)
  - lint command                     (optional PII warning pass)
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name-based detection rules
# (regex, pii_type, suggested_rule, severity, confidence)
# ---------------------------------------------------------------------------
_NAME_RULES: List[Tuple[str, str, str, str, float]] = [
    # ── HIGH severity ────────────────────────────────────────────────────────
    (r"ssn|social.?sec|social.?ins(urance)?",           "SSN",              "SSN",                  "HIGH",   0.95),
    (r"passport.?(num|no|id|nr)?",                       "PASSPORT",         "PASSPORT",             "HIGH",   0.95),
    (r"nat(ional)?.?id|nid",                             "NATIONAL_ID",      "UK_NI",                "HIGH",   0.90),
    (r"drivers?.?li[cs]|driving.?li[cs]|\bdl\b.?(num|no)?", "DRIVERS_LICENSE",  "DRIVERS_LICENCE",      "HIGH",   0.90),
    (r"aadhaar|aadhar",                                  "AADHAAR",          "IN_AADHAAR",           "HIGH",   0.98),
    (r"\bpan\b|pan.?(num|no|card)",                      "PAN",              "IN_PAN",               "HIGH",   0.90),
    (r"tfn|tax.?file.?num",                              "TFN",              "AU_TFN",               "HIGH",   0.92),
    (r"medicare.?(num|no|id)?",                          "MEDICARE",         "AU_MEDICARE",          "HIGH",   0.88),
    (r"nhs.?(num|no|id)?",                               "NHS",              "NHS_NUMBER",           "HIGH",   0.90),
    (r"nric|fin.?(num|no)?",                             "NRIC",             "SG_NRIC",              "HIGH",   0.90),
    (r"za.?id|south.?africa.?id",                        "ZA_ID",            "ZA_ID",                "HIGH",   0.90),
    # ── MEDIUM severity ──────────────────────────────────────────────────────
    (r"e.?mail|email.?addr",                             "EMAIL",            "EMAIL",                "MEDIUM", 0.95),
    (r"phone|mobile|cell|tel(ephone)?|contact.?num",     "PHONE",            "PHONE",                "MEDIUM", 0.90),
    (r"(^|_)(first|given).?name|fname|given.?name",      "FIRST_NAME",       "FIRST_NAME",           "MEDIUM", 0.88),
    (r"(^|_)(last|sur|family).?name|lname|surname",      "LAST_NAME",        "LAST_NAME",            "MEDIUM", 0.88),
    (r"full.?name|display.?name|(^|_)name($|_)",         "NAME",             "NAME",                 "MEDIUM", 0.78),
    (r"iban",                                            "IBAN",             "IBAN",                 "MEDIUM", 0.98),
    (r"swift|bic.?(code)?",                              "SWIFT_BIC",        "SWIFT",                "MEDIUM", 0.92),
    (r"sort.?code|sortcode",                             "SORT_CODE",        "UK_SORTCODE",          "MEDIUM", 0.90),
    (r"routing.?(num|no)?|aba.?(num|no)?",               "ROUTING",          "US_ROUTING",           "MEDIUM", 0.85),
    (r"account.?(num|no|number)",                        "ACCOUNT_NUMBER",   "US_ACCOUNT",           "MEDIUM", 0.72),
    (r"ip.?addr|ip.?address",                            "IP_ADDRESS",       "IPV4",                 "MEDIUM", 0.92),
    (r"mac.?addr|mac.?address",                          "MAC_ADDRESS",      "MAC",                  "MEDIUM", 0.90),
    (r"dob|birth.?date|date.?of.?birth",                 "DATE_OF_BIRTH",    None,                   "MEDIUM", 0.92),
    (r"tax.?(id|num|code|ref)|vat.?(num|id|no)?",        "TAX_ID",           "EU_VAT",               "MEDIUM", 0.82),
    (r"\bein\b|employer.?id",                            "EIN",              "US_EIN",               "MEDIUM", 0.85),
    (r"nin|\bni\b.?(num|no)?|ni.?number",                "NI_NUMBER",        "UK_NI",                "MEDIUM", 0.88),
    (r"cpf|cnpj",                                        "BR_TAX_ID",        "BR_CPF",               "MEDIUM", 0.95),
    (r"\bsiren\b|\bsiret\b",                             "FR_COMPANY_ID",    "FR_SIREN",             "MEDIUM", 0.95),
    (r"clabe",                                           "CLABE",            "CLABE",                "MEDIUM", 0.95),
    (r"ifsc",                                            "IFSC",             "IN_IFSC",              "MEDIUM", 0.95),
    (r"bsb.?(num|no|code)?",                             "BSB",              "AU_BSB",               "MEDIUM", 0.92),
    (r"gstin|gst.?(num|id|no)?",                         "GSTIN",            "GSTIN",                "MEDIUM", 0.92),
    (r"npi.?(num|no|id)?",                               "NPI",              "US_NPI",               "MEDIUM", 0.88),
    (r"btc.?addr|bitcoin.?addr",                         "BTC_ADDRESS",      "BTC_ADDRESS",          "MEDIUM", 0.92),
    (r"eth.?addr|ethereum.?addr",                        "ETH_ADDRESS",      "ETH_ADDRESS",          "MEDIUM", 0.92),
    # ── LOW severity ─────────────────────────────────────────────────────────
    (r"addr(ess)?|street|postcode|zip.?code|postal",     "ADDRESS",          "ADDRESS",              "LOW",    0.72),
    (r"company|employer|org(anization|isation)?",        "COMPANY",          "COMPANY",              "LOW",    0.65),
    (r"gender|sex($|_)",                                 "GENDER",           None,                   "LOW",    0.80),
    (r"\bage\b|birth.?year",                             "AGE",              None,                   "LOW",    0.70),
    (r"salary|income|wage|\bpay\b",                      "SALARY",           None,                   "LOW",    0.75),
    (r"uuid|guid",                                       "UUID",             "UUID",                 "LOW",    0.85),
]

_COMPILED_NAME_RULES = [
    (re.compile(pat, re.IGNORECASE), pii_type, rule, severity, conf)
    for pat, pii_type, rule, severity, conf in _NAME_RULES
]

# Value-pattern rules: applied to sampled values for confirmation
_VALUE_RULES: List[Tuple[str, str, str, float]] = [
    (r"^[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}$",             "EMAIL",        "EMAIL",        0.95),
    (r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}$",              "IBAN",         "IBAN",         0.95),
    (r"^\d{3}-\d{2}-\d{4}$",                            "SSN",          "SSN",          0.98),
    (r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",          "IPV4",         "IPV4",         0.95),
    (r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$",          "MAC",          "MAC",          0.98),
    (r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$",   "UUID",         "UUID",         0.98),
    (r"^\+?[\d\s\-(). ]{7,15}$",                        "PHONE",        "PHONE",        0.55),
    (r"^[A-Z][0-9]{7,9}$",                              "PASSPORT",     "PASSPORT",     0.75),
    (r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$",  "SWIFT_BIC",    "SWIFT",        0.85),
]
_COMPILED_VALUE_RULES = [
    (re.compile(pat, re.IGNORECASE), pii_type, rule, conf)
    for pat, pii_type, rule, conf in _VALUE_RULES
]

_SEVERITY_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class PIIFinding:
    column: str
    table: str
    pii_type: str
    confidence: float
    severity: str
    suggested_rule: Optional[str]
    evidence: str


class PIIDetector:
    """
    Scan DataFrames or TableConfig dicts for columns that likely contain PII.

    Two-pass approach:
      1. Column-name pattern matching (fast, always runs)
      2. Value sampling (runs when data is available, adds confidence)
    """

    def __init__(
        self,
        value_sample_size: int = 30,
        confidence_threshold: float = 0.60,
    ):
        self.value_sample_size = value_sample_size
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scan_dataframe(
        self, df: pd.DataFrame, table_name: str = "table"
    ) -> List[PIIFinding]:
        findings = []
        for col in df.columns:
            f = self._scan_column(col, table_name, df[col])
            if f:
                findings.append(f)
        return findings

    def scan_config(self, tables_config: Dict) -> List[PIIFinding]:
        """Scan a dict of {table_name: TableConfig} — no data values needed."""
        findings = []
        for table_name, table_cfg in tables_config.items():
            cols = getattr(table_cfg, "columns", [])
            for col in cols:
                col_name = getattr(col, "column_name", str(col))
                f = self._scan_column(col_name, table_name, None)
                if f:
                    findings.append(f)
        return findings

    def format_report(self, findings: List[PIIFinding]) -> str:
        if not findings:
            return "✅ No PII columns detected above confidence threshold."
        by_sev: Dict[str, List[PIIFinding]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for f in findings:
            by_sev.get(f.severity, by_sev["LOW"]).append(f)
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
        lines = [f"⚠️  PII scan — {len(findings)} finding(s)\n"]
        for sev in ("HIGH", "MEDIUM", "LOW"):
            group = by_sev[sev]
            if not group:
                continue
            lines.append(f"{icon[sev]} {sev} ({len(group)})")
            for f in group:
                hint = f"  → add special_rules: {f.suggested_rule}" if f.suggested_rule else ""
                lines.append(f"  {f.table}.{f.column}  [{f.pii_type}]  conf={f.confidence:.0%}{hint}")
                lines.append(f"    ↳ {f.evidence}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _scan_column(
        self,
        col_name: str,
        table_name: str,
        series: Optional[pd.Series],
    ) -> Optional[PIIFinding]:
        best: Optional[PIIFinding] = None
        best_conf = self.confidence_threshold

        # Pass 1 — name patterns
        for pattern, pii_type, rule, severity, conf in _COMPILED_NAME_RULES:
            if pattern.search(col_name):
                if conf > best_conf:
                    best_conf = conf
                    best = PIIFinding(
                        column=col_name,
                        table=table_name,
                        pii_type=pii_type,
                        confidence=conf,
                        severity=severity,
                        suggested_rule=rule,
                        evidence=f"column name matches /{pattern.pattern}/",
                    )

        # Pass 2 — value sampling (only when data provided and worth checking)
        if series is not None and len(series) > 0:
            if best is None or _SEVERITY_ORDER.get(best.severity, 0) < _SEVERITY_ORDER["HIGH"]:
                sample = series.dropna().astype(str).head(self.value_sample_size)
                if len(sample) > 0:
                    for pattern, pii_type, rule, conf in _COMPILED_VALUE_RULES:
                        hit_rate = sample.apply(lambda v: bool(pattern.match(v))).mean()
                        adjusted = round(conf * hit_rate, 3)
                        if adjusted > best_conf:
                            best_conf = adjusted
                            best = PIIFinding(
                                column=col_name,
                                table=table_name,
                                pii_type=pii_type,
                                confidence=adjusted,
                                severity="MEDIUM",
                                suggested_rule=rule,
                                evidence=f"{hit_rate:.0%} of sampled values match {pii_type} pattern",
                            )

        return best
