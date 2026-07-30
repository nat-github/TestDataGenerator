"""Tests for `validators/utility.py` and `validators/bias.py`.

The fixtures build datasets with a *known* answer — a learnable signal
that synthetic data either preserves or destroys, and a group skew that
is either faithful or amplified — so the assertions check the metric
actually measures what it claims rather than merely returning a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.validators.bias import ColumnBiasMetrics, compute_bias
from sdp.validators.quality_report import quality_report
from sdp.validators.utility import (
    UtilityMetrics,
    compute_utility,
    select_target_column,
)


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _learnable(n: int = 600, seed: int = 7, *, signal: bool = True) -> pd.DataFrame:
    """Frame where `approved` is a deterministic function of `income` and
    `score` — unless `signal=False`, which makes the label pure noise."""
    rng = _rng(seed)
    income = rng.normal(50_000, 12_000, n)
    score = rng.integers(300, 850, n)
    if signal:
        logit = (income - 50_000) / 12_000 + (score - 575) / 150
        approved = (logit > 0).astype(int)
    else:
        approved = rng.integers(0, 2, n)
    return pd.DataFrame({
        "income": income,
        "score": score,
        "region": rng.choice(["north", "south", "east"], n),
        "approved": approved.astype(str),
    })


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def test_target_selection_prefers_low_cardinality_categorical():
    df = _learnable()
    target = select_target_column(df, list(df.columns))
    assert target == "approved"          # 2 classes beats region's 3


def test_target_selection_skips_identifier_columns():
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(100)],   # unique → identifier
        "tier": ["gold", "silver"] * 50,
    })
    assert select_target_column(df, list(df.columns)) == "tier"


def test_target_selection_returns_none_when_all_columns_are_ids():
    df = pd.DataFrame({"id": [f"C{i}" for i in range(50)]})
    assert select_target_column(df, list(df.columns)) is None


def test_explicit_target_overrides_auto_selection():
    df = _learnable(300)
    result = compute_utility(df, df, target="region")
    assert result is not None
    assert result.target_column == "region"


# ---------------------------------------------------------------------------
# TSTR — utility
# ---------------------------------------------------------------------------


def test_faithful_synthetic_data_scores_high_utility():
    """Synthetic data carrying the same signal should rival real data."""
    real = _learnable(600, seed=1)
    synthetic = _learnable(600, seed=2)      # same process, different draw

    result = compute_utility(synthetic, real)
    assert result is not None
    assert result.task == "binary"
    assert result.metric == "roc_auc"
    assert result.utility_ratio is not None
    assert result.utility_ratio > 0.7, result.notes
    assert result.verdict in {"excellent — as useful as real data",
                              "good — minor loss of signal",
                              "degraded — noticeable loss of signal"}


def test_noise_synthetic_data_scores_low_utility():
    """Synthetic data whose label is random should score near zero — this
    is the case fidelity metrics miss, since the marginals still match."""
    real = _learnable(600, seed=1, signal=True)
    noise = _learnable(600, seed=2, signal=False)

    result = compute_utility(noise, real)
    assert result is not None
    assert result.utility_ratio is not None
    assert result.utility_ratio < 0.5, (result.utility_ratio, result.notes)


def test_utility_scores_both_models_on_the_same_test_set():
    real = _learnable(400)
    result = compute_utility(_learnable(400, seed=9), real)
    assert result is not None
    assert result.n_test > 0
    # 30% of 400, allowing for the dropna in the target
    assert result.n_test == pytest.approx(120, abs=5)


def test_single_class_synthetic_target_is_zero_utility():
    """Mode collapse — the generator emitted one label. No model can be
    trained, so utility is zero rather than an error."""
    real = _learnable(400)
    collapsed = _learnable(400, seed=3)
    collapsed["approved"] = "1"

    result = compute_utility(collapsed, real)
    assert result is not None
    assert result.score_synthetic == 0.0
    assert result.utility_ratio == 0.0
    assert any("one class" in n for n in result.notes)


def test_regression_target_uses_r2():
    rng = _rng()
    n = 400
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "x": x,
        "noise": rng.normal(0, 1, n),
        "amount": x * 100 + rng.normal(0, 5, n),   # continuous → regression
    })
    result = compute_utility(df, df, target="amount")
    assert result is not None
    assert result.task == "regression"
    assert result.metric == "r2"
    assert result.score_real is not None and result.score_real > 0.5


def test_multiclass_at_chance_reports_no_ratio_not_a_flattering_one():
    """Regression: macro F1's chance level is 1/k, not 0.

    Four unpredictable classes score ~0.25 for both real and synthetic
    models. A raw ratio calls that "excellent — as useful as real data";
    the honest answer is that nothing was learnable either way.
    """
    rng = _rng(3)
    n = 800
    real = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "grp": rng.choice(["w", "x", "y", "z"], n),      # independent of a/b
    })
    syn = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "grp": rng.choice(["w", "x", "y", "z"], n),
    })

    result = compute_utility(syn, real, target="grp")
    assert result is not None
    assert result.metric == "macro_f1"
    assert result.score_real == pytest.approx(0.25, abs=0.1)   # chance
    assert result.utility_ratio is None, result.utility_ratio
    assert any("chance level" in n for n in result.notes)
    assert result.verdict == "not computable"


def test_chance_level_per_metric():
    from sdp.validators.utility import _chance_level

    assert _chance_level("roc_auc", 2) == 0.5
    assert _chance_level("macro_f1", 4) == 0.25
    assert _chance_level("macro_f1", 10) == pytest.approx(0.1)
    assert _chance_level("r2", 2) == 0.0


def test_binary_at_chance_reports_no_ratio():
    rng = _rng(4)
    n = 600
    frame = lambda seed: pd.DataFrame({
        "a": _rng(seed).normal(0, 1, n),
        "flag": _rng(seed + 50).integers(0, 2, n).astype(str),   # noise label
    })
    result = compute_utility(frame(2), frame(1), target="flag")
    assert result is not None
    assert result.utility_ratio is None
    assert any("chance level" in n for n in result.notes)


def test_too_few_source_rows_is_noted_not_raised():
    tiny = _learnable(8)
    result = compute_utility(tiny, tiny)
    assert result is not None
    assert result.utility_ratio is None
    assert any("too few" in n for n in result.notes)


def test_no_shared_columns_returns_none():
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"y": [1, 2, 3]})
    assert compute_utility(a, b) is None


def test_utility_is_deterministic_across_runs():
    real = _learnable(400, seed=1)
    syn = _learnable(400, seed=2)
    first = compute_utility(syn, real)
    second = compute_utility(syn, real)
    assert first is not None and second is not None
    assert first.utility_ratio == second.utility_ratio


def test_categories_only_in_synthetic_data_do_not_break_encoding():
    real = _learnable(300, seed=1)
    syn = _learnable(300, seed=2)
    syn.loc[syn.index[:20], "region"] = "west"      # unseen in real data
    result = compute_utility(syn, real)
    assert result is not None
    assert not any("failed" in n for n in result.notes), result.notes


# ---------------------------------------------------------------------------
# Bias — representation
# ---------------------------------------------------------------------------


def _grouped(shares: dict, n: int = 1000, seed: int = 5) -> pd.DataFrame:
    """Frame whose `region` column follows the given share mapping."""
    rng = _rng(seed)
    groups = list(shares)
    probs = [shares[g] for g in groups]
    return pd.DataFrame({
        "region": rng.choice(groups, n, p=probs),
        "amount": rng.normal(100, 10, n),
    })


def test_faithful_representation_is_not_flagged():
    src = _grouped({"north": 0.5, "south": 0.3, "east": 0.2}, seed=1)
    syn = _grouped({"north": 0.5, "south": 0.3, "east": 0.2}, seed=2)

    result = {b.column: b for b in compute_bias(syn, src)}
    assert result["region"].verdict == "faithful"
    assert result["region"].flagged_groups == []


def test_representation_drift_is_detected_and_quantified():
    src = _grouped({"north": 0.5, "south": 0.3, "east": 0.2}, seed=1)
    syn = _grouped({"north": 0.8, "south": 0.15, "east": 0.05}, seed=2)

    region = {b.column: b for b in compute_bias(syn, src)}["region"]
    assert region.verdict in {"representation drift", "severe representation drift"}
    assert "north" in region.flagged_groups
    assert region.max_representation_shift > 0.2

    north = next(g for g in region.groups if g.group == "north")
    assert north.delta > 0                      # over-represented
    assert north.ratio is not None and north.ratio > 1.4


def test_group_missing_from_synthetic_is_noted():
    src = _grouped({"north": 0.6, "south": 0.3, "rare": 0.1}, seed=1)
    syn = _grouped({"north": 0.7, "south": 0.3}, seed=2)

    region = {b.column: b for b in compute_bias(syn, src)}["region"]
    assert any("absent from synthetic" in n and "rare" in n for n in region.notes)


def test_group_invented_by_generator_is_noted():
    src = _grouped({"north": 0.6, "south": 0.4}, seed=1)
    syn = _grouped({"north": 0.5, "south": 0.3, "atlantis": 0.2}, seed=2)

    region = {b.column: b for b in compute_bias(syn, src)}["region"]
    assert any("not present in source" in n and "atlantis" in n for n in region.notes)


def test_identifier_columns_are_not_treated_as_groups():
    src = pd.DataFrame({"id": [f"C{i}" for i in range(200)], "tier": ["a", "b"] * 100})
    syn = pd.DataFrame({"id": [f"C{i}" for i in range(200)], "tier": ["a", "b"] * 100})
    columns = [b.column for b in compute_bias(syn, src)]
    assert "id" not in columns
    assert "tier" in columns


# ---------------------------------------------------------------------------
# Bias — outcome disparity
# ---------------------------------------------------------------------------


def _with_outcome(rates: dict, n_per_group: int = 300, seed: int = 3) -> pd.DataFrame:
    """Frame where each group approves at its specified rate."""
    rng = _rng(seed)
    rows = []
    for group, rate in rates.items():
        approved = rng.random(n_per_group) < rate
        rows.append(pd.DataFrame({
            "region": group,
            "approved": np.where(approved, "yes", "no"),
        }))
    return pd.concat(rows, ignore_index=True)


def test_outcome_disparity_preserved_is_not_amplified():
    src = _with_outcome({"north": 0.6, "south": 0.4}, seed=1)
    syn = _with_outcome({"north": 0.6, "south": 0.4}, seed=2)

    region = {b.column: b for b in
              compute_bias(syn, src, outcome_column="approved")}["region"]
    assert region.outcome_column == "approved"
    assert region.source_disparity == pytest.approx(0.2, abs=0.08)
    assert abs(region.disparity_amplification) < 0.08


def test_outcome_disparity_amplification_is_detected():
    """The generator widens a 20-point gap into a 70-point gap."""
    src = _with_outcome({"north": 0.6, "south": 0.4}, seed=1)
    syn = _with_outcome({"north": 0.9, "south": 0.2}, seed=2)

    region = {b.column: b for b in
              compute_bias(syn, src, outcome_column="approved")}["region"]
    assert region.synthetic_disparity > region.source_disparity
    assert region.disparity_amplification > 0.3
    assert region.verdict == "outcome disparity amplified"


def test_outcome_parity_skipped_when_outcome_is_not_two_valued():
    src = _grouped({"north": 0.5, "south": 0.5}, seed=1)
    src["status"] = ["a", "b", "c"] * (len(src) // 3) + ["a"] * (len(src) % 3)
    syn = src.copy()

    region = {b.column: b for b in
              compute_bias(syn, src, outcome_column="status")}["region"]
    assert region.disparity_amplification is None


def test_outcome_column_is_not_reported_as_its_own_group():
    src = _with_outcome({"north": 0.6, "south": 0.4}, seed=1)
    columns = [b.column for b in compute_bias(src, src, outcome_column="approved")]
    assert "approved" not in columns


# ---------------------------------------------------------------------------
# Integration with the quality report
# ---------------------------------------------------------------------------


def test_report_includes_utility_and_bias_when_source_given():
    real = _learnable(400, seed=1)
    syn = _learnable(400, seed=2)

    report = quality_report({"loans": syn}, source={"loans": real})
    table = report.tables["loans"]
    assert table.utility is not None
    assert table.utility.target_column == "approved"
    assert report.overall_utility is not None
    assert table.bias                              # region was measured


def test_report_without_source_has_no_utility_or_bias():
    report = quality_report({"loans": _learnable(200)})
    table = report.tables["loans"]
    assert table.utility is None
    assert table.bias == []
    assert report.overall_utility is None
    assert report.bias_flagged_tables == []


def test_utility_and_bias_can_be_switched_off():
    real = _learnable(300, seed=1)
    syn = _learnable(300, seed=2)
    report = quality_report({"loans": syn}, source={"loans": real},
                            with_utility=False, with_bias=False)
    assert report.tables["loans"].utility is None
    assert report.tables["loans"].bias == []


def test_explicit_target_flows_through_the_report():
    real = _learnable(300, seed=1)
    syn = _learnable(300, seed=2)
    report = quality_report({"loans": syn}, source={"loans": real},
                            targets={"loans": "region"})
    assert report.tables["loans"].utility.target_column == "region"


def test_report_dict_round_trips_through_json():
    import json

    real = _learnable(300, seed=1)
    syn = _learnable(300, seed=2)
    report = quality_report({"loans": syn}, source={"loans": real})
    payload = json.loads(json.dumps(report.to_dict(), default=str))

    utility = payload["tables"]["loans"]["utility"]
    assert utility["target_column"] == "approved"
    assert "verdict" in utility                 # property, added explicitly
    bias = payload["tables"]["loans"]["bias"]
    assert bias and "groups" in bias[0]
    assert "delta" in bias[0]["groups"][0]      # property, added explicitly


def test_markdown_renders_utility_and_bias_sections():
    real = _learnable(400, seed=1)
    syn = _learnable(400, seed=2)
    syn["region"] = "north"                     # force a bias flag

    md = quality_report({"loans": syn}, source={"loans": real}).to_markdown()
    assert "Utility — can a model still learn from this data?" in md
    assert "Utility ratio" in md
    assert "Bias" in md
    assert "region" in md


def test_bias_failure_does_not_lose_the_rest_of_the_report(monkeypatch):
    """A crash in the new metrics must degrade to a note, not a lost report."""
    import sdp.validators.quality_report as qr

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic explosion")

    monkeypatch.setattr(qr, "compute_bias", boom)
    real = _learnable(200, seed=1)
    report = quality_report({"loans": _learnable(200, seed=2)},
                            source={"loans": real})

    table = report.tables["loans"]
    assert table.fidelity_score is not None      # fidelity survived
    assert any("bias failed" in n for n in table.notes)
