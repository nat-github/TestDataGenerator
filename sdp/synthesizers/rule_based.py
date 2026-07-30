"""Rule-based engine — the always-available path.

This engine trains nothing. It exists so that "no model" is a *choice you
can name* rather than only a failure mode: selecting ``--engine rule-based``
skips model fitting entirely and generates from the config's regex
patterns, business values, Faker rules and ranges.

That path already exists in ``DataGenerator`` as the fallback used when a
model fails to fit. Rather than duplicate ~600 lines of column-level
generation here, ``fit`` reports "not fitted" — which is the honest answer,
since there is no model — and ``DataGenerator`` proceeds down the fallback
branch it would have taken anyway.

Why bother registering it at all:

- it makes the choice explicit and greppable in configs and CI
- it skips the cost of a fit that would be discarded
- ``--list-engines`` shows every generation strategy in one place
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sdp.synthesizers.base import Synthesizer


class RuleBasedEngine(Synthesizer):
    """Config-driven generation — no model, always available."""

    name = "rule-based"
    description = "Config-driven generation (regex, business values, Faker) — no model fitted"
    handles_relationships = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    def fit(
        self,
        sample_data: Dict[str, "object"],
        metadata: Optional[Any] = None,
    ) -> bool:
        """Always returns False — there is nothing to fit.

        Not a failure: the caller reads False as "generate from config
        rules", which is precisely what was asked for.
        """
        self._note("rule-based engine selected — skipping model training")
        self._fitted = False
        return False

    def sample(self, records_per_table: Dict[str, int]) -> Dict[str, "object"]:
        raise NotImplementedError(
            "The rule-based engine has no model to sample from; "
            "DataGenerator generates directly from the config instead."
        )
