"""Persistent feedback store for the ML relationship inferrer.

Two layers:

1. **Pattern memory** — exact column-name pairs the SME has accepted or
   rejected previously. Stored as a normalised key (`source_table.col -> target_table.col`)
   alongside accept/reject counts. Used to boost or penalise confidence on
   future runs without needing a trained model.

2. **Training corpus** — the raw signal vectors plus the SME's verdict, used
   by `relationship_classifier.py` once enough examples accumulate.

Both layers live in the same JSONL file so a single read pulls everything.
The file format is append-only and forward-compatible — older entries
without a field are tolerated.

Default location: `<project_root>/ml_feedback/relationship_feedback.jsonl`.
Override with the `feedback_path` argument or the `SDP_FEEDBACK_PATH` env var.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DIR = "ml_feedback"
DEFAULT_FILENAME = "relationship_feedback.jsonl"
ENV_OVERRIDE = "SDP_FEEDBACK_PATH"


@dataclass
class FeedbackEntry:
    """A single SME decision recorded for future learning."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    accepted: bool
    signals: Dict[str, float] = field(default_factory=dict)
    predicted_confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    note: Optional[str] = None

    def key(self) -> str:
        """Canonical pair key used for pattern memory lookups.

        Table names are case-insensitive; column names preserve case so that
        `Customer_ID` and `customer_id` collapse intentionally — table
        identity matters more than case for matching.
        """
        return f"{self.source_table.lower()}.{self.source_column}->{self.target_table.lower()}.{self.target_column}"

    def column_pattern_key(self) -> str:
        """Lighter key matching only column names, ignoring tables.

        Lets the memory generalise across schemas — if `customer_id ->
        customer.customer_id` was accepted in one project, the same column
        pair in a new project gets a confidence boost.
        """
        return f"{self.source_column.lower()}->{self.target_column.lower()}"

    def to_json(self) -> str:
        return json.dumps({
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "accepted": self.accepted,
            "signals": self.signals,
            "predicted_confidence": self.predicted_confidence,
            "timestamp": self.timestamp,
            "note": self.note,
        })

    @classmethod
    def from_json(cls, raw: str) -> "FeedbackEntry":
        d = json.loads(raw)
        return cls(
            source_table=str(d.get("source_table", "")),
            source_column=str(d.get("source_column", "")),
            target_table=str(d.get("target_table", "")),
            target_column=str(d.get("target_column", "")),
            accepted=bool(d.get("accepted", False)),
            signals=dict(d.get("signals") or {}),
            predicted_confidence=d.get("predicted_confidence"),
            timestamp=float(d.get("timestamp") or 0.0),
            note=d.get("note"),
        )


class FeedbackStore:
    """Append-only JSONL feedback log + in-memory pattern index."""

    def __init__(self, feedback_path: Optional[Path | str] = None) -> None:
        if feedback_path is not None:
            self.path = Path(feedback_path)
        elif os.environ.get(ENV_OVERRIDE):
            self.path = Path(os.environ[ENV_OVERRIDE])
        else:
            self.path = Path.cwd() / DEFAULT_DIR / DEFAULT_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def load(self) -> List[FeedbackEntry]:
        """Read all entries. Tolerates malformed lines (logs a warning, skips)."""
        if not self.path.exists():
            return []
        entries: List[FeedbackEntry] = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(FeedbackEntry.from_json(line))
                    except (ValueError, TypeError) as exc:
                        logger.warning("feedback_store: skipped malformed line %d: %s", lineno, exc)
        except OSError as exc:
            logger.warning("feedback_store: unable to read %s: %s", self.path, exc)
        return entries

    def stats(self) -> Dict[str, int]:
        """Quick counts — useful for deciding whether to activate the classifier."""
        entries = self.load()
        accepted = sum(1 for e in entries if e.accepted)
        rejected = sum(1 for e in entries if not e.accepted)
        return {"total": len(entries), "accepted": accepted, "rejected": rejected}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def append(self, entry: FeedbackEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")

    def append_many(self, entries: Iterable[FeedbackEntry]) -> int:
        count = 0
        with self.path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(entry.to_json() + "\n")
                count += 1
        return count

    # ------------------------------------------------------------------
    # Pattern memory — lookup operations
    # ------------------------------------------------------------------
    def lookup_pattern(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
    ) -> Tuple[int, int]:
        """Return (accept_count, reject_count) for the exact table.col pair.

        Used to apply a strong confidence boost when the same pair has been
        seen in this project before.
        """
        target_key = f"{source_table.lower()}.{source_column}->{target_table.lower()}.{target_column}"
        accept = reject = 0
        for entry in self.load():
            if entry.key() == target_key:
                if entry.accepted:
                    accept += 1
                else:
                    reject += 1
        return accept, reject

    def lookup_column_pattern(
        self,
        source_column: str,
        target_column: str,
    ) -> Tuple[int, int]:
        """Return (accept_count, reject_count) for the column-name pair only.

        This generalises across projects — if `cust_id -> customer.id` was
        accepted in dataset A, the same column-name pair in dataset B gets
        a milder boost even though the tables differ.
        """
        target_key = f"{source_column.lower()}->{target_column.lower()}"
        accept = reject = 0
        for entry in self.load():
            if entry.column_pattern_key() == target_key:
                if entry.accepted:
                    accept += 1
                else:
                    reject += 1
        return accept, reject

    # ------------------------------------------------------------------
    # Training corpus (used by relationship_classifier)
    # ------------------------------------------------------------------
    def training_data(
        self,
        require_signals: bool = True,
    ) -> Tuple[List[Dict[str, float]], List[int]]:
        """Materialise (X, y) for sklearn training.

        X is a list of signal dicts; y is a list of 1/0 labels (accepted/rejected).
        Entries without a signals dict are skipped when require_signals=True
        (default) so the classifier only trains on examples it can score.
        """
        X: List[Dict[str, float]] = []
        y: List[int] = []
        for entry in self.load():
            if require_signals and not entry.signals:
                continue
            X.append(dict(entry.signals or {}))
            y.append(1 if entry.accepted else 0)
        return X, y
