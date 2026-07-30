"""Tests for the synthesizer plugin interface.

Covers the registry, the shared multi-table sampling helper, the engine
contract, and the `DataGenerator` wiring — including the compatibility
guarantees the rest of the codebase depends on (`synthesizer`, `is_fitted`,
and a directly injected synthesizer double still working).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pytest

from sdp.synthesizers import (
    DEFAULT_ENGINE,
    EngineStats,
    Synthesizer,
    create,
    describe,
    get,
    names,
    register,
    sample_multi_table,
    unregister,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _RecordingEngine(Synthesizer):
    """Minimal conforming engine — records what it was asked to do."""

    name = "test-recording"
    description = "test double"

    def __init__(self, *, seed=None, **options):
        super().__init__(seed=seed, **options)
        self.fit_calls: list = []
        self.fail_fit = bool(options.get("fail_fit", False))

    def fit(self, sample_data, metadata=None) -> bool:
        self.fit_calls.append((sample_data, metadata))
        if self.fail_fit:
            self._note("asked to fail")
            return False
        self.stats.fit_rows = sum(len(df) for df in sample_data.values())
        self._fitted = True
        return True

    def sample(self, records_per_table):
        out = {t: pd.DataFrame({"x": range(n)}) for t, n in records_per_table.items()}
        self.stats.sampled_rows = sum(len(df) for df in out.values())
        return out


class _MultiTableModel:
    """Stands in for an SDV multi-table synthesizer."""

    def __init__(self, *, rows: int = 100, accepts_num_rows: bool = True):
        self.rows = rows
        self.accepts_num_rows = accepts_num_rows
        self.last_call: Optional[tuple] = None

    def sample(self, num_rows=None, scale=None):
        if num_rows is not None and not self.accepts_num_rows:
            raise TypeError("sample() got an unexpected keyword argument 'num_rows'")
        self.last_call = ("num_rows", num_rows) if num_rows is not None else ("scale", scale)
        if num_rows is not None:
            return {t: pd.DataFrame({"x": range(n)}) for t, n in num_rows.items()}
        return {"orders": pd.DataFrame({"x": range(self.rows)})}


@pytest.fixture(autouse=True)
def _clean_registry():
    """Remove test-registered engines so tests cannot leak into each other."""
    yield
    for name in list(names()):
        if name.startswith("test-"):
            unregister(name)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_engines_are_registered():
    registered = names()
    for expected in ("sdv", "gaussian-copula", "ctgan", "tvae", "rule-based"):
        assert expected in registered


def test_default_engine_is_sdv():
    """The historical behaviour — changing this changes every existing config."""
    assert DEFAULT_ENGINE == "sdv"


def test_unknown_engine_lists_the_valid_names():
    with pytest.raises(KeyError) as exc:
        get("no-such-engine")
    assert "sdv" in str(exc.value)


def test_register_and_resolve_third_party_engine():
    register("test-recording", f"{__name__}:_RecordingEngine")
    assert get("test-recording") is _RecordingEngine
    engine = create("test-recording", seed=42)
    assert isinstance(engine, _RecordingEngine)
    assert engine.seed == 42


def test_create_passes_options_through():
    register("test-recording", f"{__name__}:_RecordingEngine")
    engine = create("test-recording", epochs=7)
    assert engine.options["epochs"] == 7


def test_describe_reports_availability():
    rows = {name: (desc, available) for name, desc, available in describe()}
    assert rows["rule-based"][1] is True          # never has dependencies
    assert rows["sdv"][0]                          # has a description


def test_registry_does_not_import_engine_modules_eagerly():
    """Resolving 'rule-based' must not drag torch in via the ctgan entry."""
    import sys

    for module in list(sys.modules):
        if module.startswith("sdp.synthesizers.single_table"):
            del sys.modules[module]
    get("rule-based")
    assert "sdp.synthesizers.single_table" not in sys.modules


# ---------------------------------------------------------------------------
# Shared multi-table sampling
# ---------------------------------------------------------------------------


def test_sample_multi_table_trims_to_requested_rows():
    model = _MultiTableModel()
    out = sample_multi_table(model, {"orders": 5, "customers": 3})
    assert len(out["orders"]) == 5
    assert len(out["customers"]) == 3
    assert model.last_call[0] == "num_rows"


def test_sample_multi_table_falls_back_to_scale():
    """Older SDV builds take `scale`, not `num_rows`."""
    model = _MultiTableModel(rows=400, accepts_num_rows=False)
    out = sample_multi_table(model, {"orders": 200}, {"orders": 100})
    assert model.last_call == ("scale", 2.0)       # 200 requested / 100 fitted
    assert len(out["orders"]) == 200


def test_sample_multi_table_raises_when_short():
    model = _MultiTableModel(rows=3, accepts_num_rows=False)
    with pytest.raises(ValueError, match="expected at least"):
        sample_multi_table(model, {"orders": 50}, {"orders": 50})


def test_sample_multi_table_rejects_uninitialised_model():
    with pytest.raises(ValueError, match="not initialized"):
        sample_multi_table(None, {"orders": 5})


def test_sample_multi_table_rejects_non_dict_result():
    class Bad:
        def sample(self, **kwargs):
            return pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError, match="non-dictionary"):
        sample_multi_table(Bad(), {"orders": 1})


# ---------------------------------------------------------------------------
# Engine contract
# ---------------------------------------------------------------------------


def test_engine_records_fit_and_sample_cost():
    engine = _RecordingEngine()
    engine.fit({"orders": pd.DataFrame({"x": range(10)})})
    engine.sample({"orders": 4})

    stats = engine.stats.to_dict()
    assert stats["engine"] == "test-recording"
    assert stats["fit_rows"] == 10
    assert stats["sampled_rows"] == 4
    assert stats["fit_seconds"] >= 0.0
    assert stats["total_seconds"] == pytest.approx(
        stats["fit_seconds"] + stats["sample_seconds"]
    )


def test_failed_fit_returns_false_rather_than_raising():
    engine = _RecordingEngine(fail_fit=True)
    assert engine.fit({"orders": pd.DataFrame({"x": [1]})}) is False
    assert engine.is_fitted is False
    assert engine.stats.notes


def test_rule_based_engine_reports_unfitted_by_design():
    engine = create("rule-based")
    assert engine.is_available() is True
    assert engine.fit({"orders": pd.DataFrame({"x": [1]})}) is False
    assert engine.is_fitted is False
    with pytest.raises(NotImplementedError):
        engine.sample({"orders": 1})


def test_hma_engine_reports_relationship_handling():
    from sdp.synthesizers.sdv_hma import HMAEngine
    from sdp.synthesizers.single_table import CTGANEngine

    assert HMAEngine.handles_relationships is True
    assert CTGANEngine.handles_relationships is False


def test_hma_engine_fit_without_metadata_fails_cleanly():
    from sdp.synthesizers.sdv_hma import HMAEngine

    engine = HMAEngine()
    assert engine.fit({"orders": pd.DataFrame({"x": [1]})}, metadata=None) is False
    assert any("metadata" in n for n in engine.stats.notes)


def test_hma_engine_adopts_a_pre_fitted_model():
    """The artifact cache restores a pickled model rather than retraining."""
    from sdp.synthesizers.sdv_hma import HMAEngine

    engine = HMAEngine()
    model = _MultiTableModel()
    engine.adopt(model, {"orders": 100})

    assert engine.is_fitted is True
    assert engine.model is model
    assert engine.fitted_sample_sizes == {"orders": 100}
    assert len(engine.sample({"orders": 5})["orders"]) == 5


def test_single_table_engine_model_identity_is_stable():
    """DataGenerator identity-checks engine.model, so it must not be rebuilt."""
    from sdp.synthesizers.single_table import GaussianCopulaEngine

    engine = GaussianCopulaEngine()
    engine._models = {"orders": object()}
    assert engine.model is engine.model


# ---------------------------------------------------------------------------
# DataGenerator wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config(tmp_path):
    """Minimal single-table YAML config."""
    config = tmp_path / "cfg.yaml"
    config.write_text(
        """
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 20
tables:
  - name: customers
    active: true
    num_rows: 20
    columns:
      - name: customer_id
        data_type: N38
        is_primary_key: true
      - name: tier
        data_type: VA3
        business_values: "gold;silver;bronze"
""".strip(),
        encoding="utf-8",
    )
    return str(config)


def test_default_engine_is_used_when_unspecified(simple_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config)
    assert gen.resolve_engine_name() == "sdv"


def test_constructor_engine_overrides_config(simple_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config, engine="rule-based")
    assert gen.resolve_engine_name() == "rule-based"


def test_config_setting_selects_engine(tmp_path):
    from sdp.generators.data_generator import DataGenerator

    config = tmp_path / "cfg.yaml"
    config.write_text(
        """
config_format: sdp-yaml-v1
run_settings:
  synthesizer_engine: rule-based
tables:
  - name: customers
    active: true
    num_rows: 5
    columns:
      - name: customer_id
        data_type: N38
        is_primary_key: true
""".strip(),
        encoding="utf-8",
    )
    gen = DataGenerator(str(config))
    assert gen.load_configuration() is True        # settings are read at parse time
    assert gen.resolve_engine_name() == "rule-based"


def test_rule_based_engine_generates_without_fitting(simple_config):
    """End to end on the explicit no-model path."""
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config, engine="rule-based", seed=42)
    assert gen.load_configuration() is True
    gen.create_sdv_metadata()

    assert gen.train_synthesizer() is False        # nothing to fit, by design
    assert gen.is_fitted is False

    data = gen.generate_data({"customers": 20})
    assert len(data["customers"]) == 20
    assert data["customers"]["customer_id"].is_unique


def test_engine_options_parsed_from_config_string(tmp_path):
    """Excel cells cannot hold a dict, so `k=v;k=v` is accepted."""
    from sdp.generators.data_generator import DataGenerator

    config = tmp_path / "cfg.yaml"
    config.write_text(
        """
config_format: sdp-yaml-v1
run_settings:
  engine_options: "epochs=250;verbose=no"
tables:
  - name: customers
    active: true
    num_rows: 5
    columns:
      - name: customer_id
        data_type: N38
        is_primary_key: true
""".strip(),
        encoding="utf-8",
    )
    gen = DataGenerator(str(config))
    assert gen.load_configuration() is True
    assert gen._engine_options() == {"epochs": 250, "verbose": "no"}


def test_constructor_engine_options_win_over_config(simple_config):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config, engine_options={"epochs": 5})
    assert gen._engine_options() == {"epochs": 5}


def test_injected_synthesizer_double_still_samples(simple_config):
    """Back-compat: tests assign `generator.synthesizer` directly and expect
    sampling to work without any engine involved."""
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config)
    assert gen.load_configuration() is True
    gen.synthesizer = _MultiTableModel()
    gen.is_fitted = True

    out = gen._sample_from_synthesizer({"customers": 7})
    assert len(out["customers"]) == 7


def test_engine_stats_none_before_training(simple_config):
    from sdp.generators.data_generator import DataGenerator

    assert DataGenerator(simple_config).engine_stats is None


def test_unknown_engine_falls_back_to_rule_based_generation(simple_config):
    """A bad engine name must not lose the run — training fails, the
    fallback still produces data."""
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config, engine="does-not-exist", seed=1)
    assert gen.load_configuration() is True
    gen.create_sdv_metadata()

    assert gen.train_synthesizer() is False
    data = gen.generate_data({"customers": 10})
    assert len(data["customers"]) == 10


def test_artifact_caching_skipped_for_non_sdv_engines(simple_config, tmp_path):
    from sdp.generators.data_generator import DataGenerator

    gen = DataGenerator(simple_config, engine="gaussian-copula")
    assert gen.load_configuration() is True
    gen.is_fitted = True
    gen.synthesizer = object()
    gen.metadata = object()
    monkey = {"save_model_artifact": True}
    gen.config_parser.get_setting = lambda key, default=None: monkey.get(key, default)

    assert gen.save_model_artifacts(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_cli_parses_engine_flags():
    from sdp.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--config", "c.yaml", "--engine", "ctgan",
        "--engine-option", "epochs=300", "--engine-option", "batch_size=100",
    ])
    assert args.engine == "ctgan"
    assert args.engine_option == ["epochs=300", "batch_size=100"]


def test_cli_engine_option_parsing_coerces_ints():
    from sdp.cli import _parse_engine_options

    assert _parse_engine_options(["epochs=300", "name=fast"]) == {
        "epochs": 300, "name": "fast",
    }
    assert _parse_engine_options(None) is None


def test_cli_engine_option_rejects_malformed_pairs():
    from sdp.cli import _parse_engine_options

    with pytest.raises(ValueError, match="KEY=VALUE"):
        _parse_engine_options(["epochs"])


def test_cli_list_engines_prints_registry(capsys):
    from sdp.cli import _print_engines

    _print_engines()
    out = capsys.readouterr().out
    assert "sdv" in out and "(default)" in out
    assert "rule-based" in out
