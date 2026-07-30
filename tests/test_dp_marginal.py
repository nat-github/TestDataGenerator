"""Tests for the differentially private marginal engine.

DP is easy to claim and easy to get wrong, so these tests target the
*properties that make the guarantee real* rather than just exercising the
code path:

- the domain comes from the config, never from the data
- the budget composes correctly (epsilons add within a table)
- noise is actually applied, and more noise at lower epsilon
- columns without a declared domain never touch the data
- the accounting reported matches what was spent
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdp.synthesizers import create
from sdp.synthesizers.dp_marginal import (
    NULL_TOKEN,
    ColumnDomain,
    DPMarginalEngine,
)


def _frame_factory(columns):
    """Config-only frame producer, as DataGenerator supplies."""
    def factory(table: str, n: int) -> pd.DataFrame:
        return pd.DataFrame({c: [None] * n for c in columns})
    return factory


def _categorical_domain():
    return {"loans": {"status": ColumnDomain(kind="categorical",
                                             values=["approved", "declined"])}}


def _skewed(n=1000, approved=0.9):
    k = int(n * approved)
    return {"loans": pd.DataFrame({"status": ["approved"] * k + ["declined"] * (n - k)})}


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_engine_is_registered_and_available():
    engine = create("dp-marginal")
    assert isinstance(engine, DPMarginalEngine)
    assert DPMarginalEngine.is_available() is True
    assert DPMarginalEngine.handles_relationships is False


def test_non_positive_epsilon_is_rejected():
    """epsilon <= 0 is not 'perfect privacy', it is an invalid budget."""
    with pytest.raises(ValueError, match="epsilon"):
        DPMarginalEngine(epsilon=0)
    with pytest.raises(ValueError, match="epsilon"):
        DPMarginalEngine(epsilon=-1)


def test_too_few_bins_rejected():
    with pytest.raises(ValueError, match="numeric_bins"):
        DPMarginalEngine(numeric_bins=1)


def test_fit_without_domains_refuses():
    """No public domain means nothing can be measured safely."""
    engine = DPMarginalEngine()
    assert engine.fit(_skewed()) is False
    assert engine.is_fitted is False


def test_fit_with_only_undeclared_columns_refuses():
    engine = DPMarginalEngine(domains={"loans": {}})
    assert engine.fit(_skewed()) is False
    assert any("declare" in n.lower() or "domain" in n.lower()
               for n in engine.stats.notes)


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------


def test_marginal_tracks_the_data_at_high_epsilon():
    """With a generous budget the noisy histogram should resemble reality."""
    engine = DPMarginalEngine(epsilon=1000.0, domains=_categorical_domain(),
                              frame_factory=_frame_factory(["status"]))
    assert engine.fit(_skewed(2000, approved=0.9)) is True

    out = engine.sample({"loans": 4000})
    approved_rate = (out["loans"]["status"] == "approved").mean()
    assert approved_rate == pytest.approx(0.9, abs=0.05)


def test_low_epsilon_destroys_the_signal():
    """The privacy/accuracy trade-off must be real, not decorative.

    At a tiny budget the noise dominates the counts, so the output should
    drift well away from the true 90/10 split.
    """
    truth = _skewed(2000, approved=0.9)
    rates = []
    for seed in range(8):
        engine = DPMarginalEngine(epsilon=0.001, seed=seed,
                                  domains=_categorical_domain(),
                                  frame_factory=_frame_factory(["status"]))
        engine.fit(truth)
        out = engine.sample({"loans": 500})
        rates.append((out["loans"]["status"] == "approved").mean())

    # Some run must land far from the true rate; if every run reproduced
    # 0.9 at epsilon=0.001, no meaningful noise was applied.
    assert max(abs(r - 0.9) for r in rates) > 0.15, rates


def test_noise_is_applied_at_all():
    """Identical data, different noise draws → different distributions."""
    truth = _skewed(500, approved=0.5)
    outputs = []
    for seed in (1, 2, 3):
        engine = DPMarginalEngine(epsilon=0.05, seed=seed,
                                  domains=_categorical_domain(),
                                  frame_factory=_frame_factory(["status"]))
        engine.fit(truth)
        outputs.append(engine._marginals["loans"]["status"][1])

    assert outputs[0] != outputs[1] or outputs[1] != outputs[2]


def test_values_outside_the_declared_domain_are_dropped():
    """Dropping is a data-independent rule, so it costs no privacy — but it
    must actually happen, or the domain is not really the domain."""
    data = {"loans": pd.DataFrame({"status": ["approved"] * 50 + ["SURPRISE"] * 50})}
    engine = DPMarginalEngine(epsilon=1000.0, domains=_categorical_domain(),
                              frame_factory=_frame_factory(["status"]))
    engine.fit(data)
    out = engine.sample({"loans": 300})

    assert "SURPRISE" not in set(out["loans"]["status"].dropna())


def test_nulls_are_privatised_rather_than_read_in_the_clear():
    """Null rate is itself sensitive, so it gets its own noisy bucket."""
    data = {"loans": pd.DataFrame({"status": ["approved"] * 50 + [None] * 50})}
    engine = DPMarginalEngine(epsilon=1000.0, domains=_categorical_domain(),
                              frame_factory=_frame_factory(["status"]))
    engine.fit(data)

    labels, probabilities = engine._marginals["loans"]["status"]
    assert labels[-1] == NULL_TOKEN
    assert probabilities[-1] > 0.2                # roughly the true 50%

    out = engine.sample({"loans": 400})
    assert out["loans"]["status"].isna().sum() > 0


def test_numeric_domain_respects_declared_bounds():
    """Bin edges come from config min/max — output must stay inside them."""
    domains = {"loans": {"amount": ColumnDomain(kind="numeric", low=0.0, high=100.0)}}
    data = {"loans": pd.DataFrame({"amount": np.random.default_rng(1).uniform(0, 100, 500)})}

    engine = DPMarginalEngine(epsilon=100.0, domains=domains,
                              frame_factory=_frame_factory(["amount"]))
    assert engine.fit(data) is True
    out = engine.sample({"loans": 1000})["loans"]["amount"].dropna()

    assert out.min() >= 0.0
    assert out.max() <= 100.0


def test_integer_domain_yields_integers():
    domains = {"loans": {"score": ColumnDomain(kind="numeric", low=300, high=850,
                                               integer=True)}}
    data = {"loans": pd.DataFrame({"score": [400, 500, 600, 700] * 50})}
    engine = DPMarginalEngine(epsilon=50.0, domains=domains,
                              frame_factory=_frame_factory(["score"]))
    engine.fit(data)
    values = engine.sample({"loans": 200})["loans"]["score"].dropna()

    assert all(float(v).is_integer() for v in values)


def test_out_of_range_values_are_clipped_not_dropped():
    """Clipping to declared bounds is data-independent and keeps the mass."""
    domains = {"loans": {"amount": ColumnDomain(kind="numeric", low=0.0, high=10.0)}}
    data = {"loans": pd.DataFrame({"amount": [500.0] * 100})}   # all above the max
    engine = DPMarginalEngine(epsilon=100.0, domains=domains,
                              frame_factory=_frame_factory(["amount"]))
    engine.fit(data)

    labels, probabilities = engine._marginals["loans"]["amount"]
    assert probabilities[-2] > 0.5               # piled into the top bin


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------


def test_budget_splits_across_measured_columns():
    """Sequential composition: per-column epsilons must sum to the total."""
    domains = {"loans": {
        "status": ColumnDomain(kind="categorical", values=["a", "b"]),
        "tier": ColumnDomain(kind="categorical", values=["x", "y"]),
        "amount": ColumnDomain(kind="numeric", low=0.0, high=10.0),
    }}
    data = {"loans": pd.DataFrame({
        "status": ["a", "b"] * 50, "tier": ["x", "y"] * 50,
        "amount": [1.0] * 100,
    })}

    engine = DPMarginalEngine(epsilon=3.0, domains=domains,
                              frame_factory=_frame_factory(["status", "tier", "amount"]))
    assert engine.fit(data) is True

    report = engine.privacy_report()
    spent = [c["epsilon"] for c in report["measured_columns"]]
    assert len(spent) == 3
    assert all(e == pytest.approx(1.0) for e in spent)
    assert sum(spent) == pytest.approx(3.0)
    assert report["per_table_epsilon"]["loans"] == pytest.approx(3.0)


def test_undeclared_columns_consume_no_budget_and_are_reported():
    domains = {"loans": {"status": ColumnDomain(kind="categorical", values=["a", "b"])}}
    data = {"loans": pd.DataFrame({"status": ["a", "b"] * 50,
                                   "free_text": ["anything"] * 100})}

    engine = DPMarginalEngine(epsilon=1.0, domains=domains,
                              frame_factory=_frame_factory(["status", "free_text"]))
    engine.fit(data)
    report = engine.privacy_report()

    assert [c["column"] for c in report["measured_columns"]] == ["status"]
    unmeasured = {c["column"]: c["reason"] for c in report["unmeasured_columns"]}
    assert "free_text" in unmeasured
    assert "config" in unmeasured["free_text"]


def test_undeclared_column_values_come_from_config_not_data():
    """The privacy claim depends on this: an unmeasured column must not
    reflect the training data at all."""
    domains = {"loans": {"status": ColumnDomain(kind="categorical", values=["a", "b"])}}
    data = {"loans": pd.DataFrame({"status": ["a"] * 100,
                                   "secret": ["PRIVATE-VALUE"] * 100})}

    def factory(table, n):
        return pd.DataFrame({"status": [None] * n, "secret": ["from-config"] * n})

    engine = DPMarginalEngine(epsilon=100.0, domains=domains, frame_factory=factory)
    engine.fit(data)
    out = engine.sample({"loans": 50})

    assert set(out["loans"]["secret"]) == {"from-config"}
    assert "PRIVATE-VALUE" not in set(out["loans"]["secret"])


def test_epsilon_per_table_not_shared_across_tables():
    """The guarantee is per table; each table gets the full budget."""
    domains = {
        "a": {"c": ColumnDomain(kind="categorical", values=["x", "y"])},
        "b": {"c": ColumnDomain(kind="categorical", values=["x", "y"])},
    }
    data = {
        "a": pd.DataFrame({"c": ["x", "y"] * 25}),
        "b": pd.DataFrame({"c": ["x", "y"] * 25}),
    }
    engine = DPMarginalEngine(epsilon=2.0, domains=domains,
                              frame_factory=_frame_factory(["c"]))
    engine.fit(data)
    report = engine.privacy_report()

    assert report["per_table_epsilon"] == {"a": pytest.approx(2.0), "b": pytest.approx(2.0)}


def test_report_states_the_limits_of_the_guarantee():
    """It must not read as a blanket claim — cross-table entities are not
    covered, and saying so is the point."""
    engine = DPMarginalEngine(epsilon=1.0, domains=_categorical_domain(),
                              frame_factory=_frame_factory(["status"]))
    engine.fit(_skewed(100))
    guarantee = engine.privacy_report()["guarantee"]

    assert "per row" in guarantee
    assert "multiple tables" in guarantee


def test_seeded_run_warns_that_the_guarantee_is_void():
    """A known seed means the noise can be subtracted."""
    engine = DPMarginalEngine(epsilon=1.0, seed=42, domains=_categorical_domain(),
                              frame_factory=_frame_factory(["status"]))
    engine.fit(_skewed(100))
    warnings = engine.privacy_report()["warnings"]

    assert any("seed" in w for w in warnings)


def test_unseeded_run_has_no_seed_warning():
    engine = DPMarginalEngine(epsilon=1.0, domains=_categorical_domain(),
                              frame_factory=_frame_factory(["status"]))
    engine.fit(_skewed(100))
    assert not any("seed" in w for w in engine.privacy_report()["warnings"])


def test_sample_without_frame_factory_raises():
    engine = DPMarginalEngine(epsilon=1.0, domains=_categorical_domain())
    engine.fit(_skewed(100))
    with pytest.raises(ValueError, match="frame_factory"):
        engine.sample({"loans": 10})


# ---------------------------------------------------------------------------
# DataGenerator wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def dp_config(tmp_path):
    config = tmp_path / "dp.yaml"
    config.write_text(
        """
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 50
tables:
  - name: loans
    active: true
    num_rows: 50
    columns:
      - name: loan_id
        data_type: N38
        is_primary_key: true
      - name: status
        data_type: VA3
        business_values: "approved;declined;pending"
      - name: amount
        data_type: N38
        min_value: 100
        max_value: 5000
      - name: notes
        data_type: VA256
""".strip(),
        encoding="utf-8",
    )
    return str(config)


def test_domains_are_built_from_the_config(dp_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(dp_config)
    assert gen.load_configuration() is True
    domains = gen.build_column_domains()["loans"]

    assert domains["status"].kind == "categorical"
    assert set(domains["status"].values) == {"approved", "declined", "pending"}
    assert domains["amount"].kind == "numeric"
    assert (domains["amount"].low, domains["amount"].high) == (100.0, 5000.0)


def test_primary_keys_are_never_dp_modelled(dp_config):
    """Identifiers are not distributions — and PKs must stay unique."""
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(dp_config)
    gen.load_configuration()
    assert "loan_id" not in gen.build_column_domains()["loans"]


def test_columns_without_a_declared_domain_are_absent(dp_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(dp_config)
    gen.load_configuration()
    assert "notes" not in gen.build_column_domains()["loans"]


def test_end_to_end_dp_generation(dp_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(dp_config, engine="dp-marginal", seed=7)
    assert gen.load_configuration() is True
    gen.create_sdv_metadata()
    assert gen.train_synthesizer() is True

    data = gen.generate_data({"loans": 200})
    loans = data["loans"]

    assert len(loans) == 200
    assert loans["loan_id"].is_unique                    # PK integrity kept
    assert set(loans["status"].dropna()) <= {"approved", "declined", "pending"}

    report = gen.privacy_report
    assert report is not None
    assert report["epsilon_requested"] == 1.0
    assert {c["column"] for c in report["measured_columns"]} == {"status", "amount"}


def test_epsilon_option_flows_from_constructor(dp_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(dp_config, engine="dp-marginal",
                        engine_options={"epsilon": 0.25}, seed=3)
    gen.load_configuration()
    gen.create_sdv_metadata()
    assert gen.train_synthesizer() is True
    assert gen.privacy_report["epsilon_requested"] == 0.25


def test_other_engines_report_no_privacy_guarantee(dp_config):
    """Absence of a report means 'no formal guarantee', which is honest."""
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(dp_config, engine="gaussian-copula", seed=1)
    gen.load_configuration()
    gen.create_sdv_metadata()
    gen.train_synthesizer()
    assert gen.privacy_report is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_exposes_epsilon_and_privacy_report_flags():
    from sdp.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--config", "c.yaml", "--engine", "dp-marginal",
        "--epsilon", "0.5", "--privacy-report-json", "p.json",
    ])
    assert args.epsilon == 0.5
    assert args.privacy_report_json == "p.json"


def test_dp_engine_listed_by_list_engines(capsys):
    from sdp.cli import _print_engines

    _print_engines()
    assert "dp-marginal" in capsys.readouterr().out
