"""Tests for fingerprint-keyed synthesizer artifact caching.

A fitted SDV synthesizer is saved as ``model_<fingerprint>.pkl`` + a JSON
sidecar. A later run reuses it only when the config fingerprint AND the SDV
version match — otherwise it retrains. The cache is never a source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path

from sdp.generators.data_generator import DataGenerator

_BASE = """\
config_format: sdp-yaml-v1
run_settings:
  default_records_per_table: 50
  save_model_artifact: true
  model_artifact_path: {artifacts}
tables:
  - name: users
    rows: 50
    primary_key_columns: [user_id]
    columns:
      - name: user_id
        data_type: N10
        is_pk: true
        nullable: false
      - name: full_name
        data_type: VA64
        special_rules: NAME
{extra}
"""

_EXTRA_COLUMN = """\
      - name: email
        data_type: VA128
        special_rules: EMAIL
"""


def _write_config(tmp_path: Path, artifacts: Path, extra: str = "") -> Path:
    cfg = tmp_path / f"cfg_{abs(hash(extra)) % 10000}.yaml"
    cfg.write_text(_BASE.format(artifacts=artifacts.as_posix(), extra=extra), encoding="utf-8")
    return cfg


def _loaded(cfg: Path, seed: int = 42) -> DataGenerator:
    gen = DataGenerator(str(cfg), seed=seed)
    assert gen.load_configuration()
    gen.create_sdv_metadata()
    return gen


def test_config_fingerprint_stable_and_sensitive(tmp_path: Path):
    artifacts = tmp_path / "models"
    cfg = _write_config(tmp_path, artifacts)
    cfg_modified = _write_config(tmp_path, artifacts, extra=_EXTRA_COLUMN)

    fp1 = _loaded(cfg)._config_fingerprint()
    fp2 = _loaded(cfg)._config_fingerprint()
    fp_mod = _loaded(cfg_modified)._config_fingerprint()

    assert fp1 == fp2, "same config must produce the same fingerprint"
    assert fp1 != fp_mod, "a structural config change must change the fingerprint"
    assert len(fp1) == 16


def test_model_artifact_saved_and_reused(tmp_path: Path):
    artifacts = tmp_path / "models"
    cfg = _write_config(tmp_path, artifacts)

    # First run — no cache, trains and saves.
    gen1 = _loaded(cfg, seed=42)
    assert gen1.train_synthesizer()
    assert gen1.save_model_artifacts() is not None

    fp = gen1._config_fingerprint()
    assert (artifacts / f"model_{fp}.pkl").is_file()
    assert (artifacts / f"model_{fp}.json").is_file()
    sidecar = json.loads((artifacts / f"model_{fp}.json").read_text(encoding="utf-8"))
    assert sidecar["fingerprint"] == fp
    assert sidecar["sdv_version"]

    # Second run — a fresh generator reuses the cached synthesizer.
    gen2 = _loaded(cfg, seed=99)
    assert gen2._try_load_cached_synthesizer() is True
    assert gen2.is_fitted
    assert gen2.synthesizer is not None


def test_cached_model_ignored_on_sdv_version_mismatch(tmp_path: Path):
    artifacts = tmp_path / "models"
    cfg = _write_config(tmp_path, artifacts)

    gen1 = _loaded(cfg)
    assert gen1.train_synthesizer()
    gen1.save_model_artifacts()
    fp = gen1._config_fingerprint()

    # Corrupt the recorded SDV version — the cache must be refused.
    sidecar_path = artifacts / f"model_{fp}.json"
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    data["sdv_version"] = "0.0.0-not-installed"
    sidecar_path.write_text(json.dumps(data), encoding="utf-8")

    gen2 = _loaded(cfg)
    assert gen2._try_load_cached_synthesizer() is False, \
        "a model saved under a different SDV version must not be reused"


def test_cache_disabled_when_setting_off(tmp_path: Path):
    """With save_model_artifact off, nothing is saved and nothing is loaded."""
    artifacts = tmp_path / "models"
    cfg = tmp_path / "cfg_off.yaml"
    cfg.write_text(_BASE.format(artifacts=artifacts.as_posix(), extra="")
                   .replace("save_model_artifact: true", "save_model_artifact: false"),
                   encoding="utf-8")

    gen = _loaded(cfg)
    assert gen.train_synthesizer()
    assert gen.save_model_artifacts() is None
    assert gen._try_load_cached_synthesizer() is False
