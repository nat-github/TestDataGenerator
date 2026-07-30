"""Fingerprint-keyed synthesizer artifact caching.

Extracted from ``data_generator.py`` as a mixin. ``DataGenerator`` inherits
it, so ``self`` resolves exactly as before — this is a pure move, not a
behaviour change. The split exists so each concern can be read and tested
without loading a 2,400-line class.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


from sdv.multi_table import HMASynthesizer


logger = logging.getLogger(__name__)


class ModelCacheMixin:
    """Fingerprint-keyed synthesizer artifact caching."""

    def _config_fingerprint(self) -> str:
        """A stable 16-hex hash of the parsed config (tables, columns,
        relationships). Any structural or rule change to the config changes
        this hash; an unchanged config keeps it. Used to key cached synthesizer
        artifacts so a model is only ever reused for the config it was fit on.
        """
        payload = {
            "tables": {
                name: tc.model_dump(mode="json")
                for name, tc in sorted(self.tables_config.items())
            },
            "relationships": sorted(
                (r.model_dump(mode="json") for r in self.relationships),
                key=lambda d: json.dumps(d, sort_keys=True, default=str),
            ),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _model_artifact_dir(self, output_dir: Optional[str] = None) -> Path:
        """Resolve the directory where synthesizer artifacts are kept."""
        target = output_dir or self.config_parser.get_setting('model_artifact_path', 'output/models')
        return Path(target)

    def _try_load_cached_synthesizer(self) -> bool:
        """Load a previously-saved synthesizer when one matches this exact config.

        Returns True (and sets ``self.synthesizer`` / ``is_fitted``) only when a
        cached artifact exists whose config fingerprint **and** SDV version
        match. Any mismatch, missing file, or load error → False, so the caller
        retrains. The cache is never a source of truth: a stale model is
        ignored, not used.
        """
        if not bool(self.config_parser.get_setting('save_model_artifact', False)):
            return False
        try:
            import sdv as _sdv
            fingerprint = self._config_fingerprint()
            artifact_dir = self._model_artifact_dir()
            model_file = artifact_dir / f"model_{fingerprint}.pkl"
            sidecar_file = artifact_dir / f"model_{fingerprint}.json"
            if not model_file.is_file() or not sidecar_file.is_file():
                return False

            sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
            cached_version = sidecar.get("sdv_version")
            if cached_version != _sdv.__version__:
                self.logger.info(
                    f"ℹ️ Cached model SDV version {cached_version!r} != installed "
                    f"{_sdv.__version__!r} — retraining instead of loading."
                )
                return False

            # Prefer the newer sdv.utils.load_synthesizer; fall back to the
            # class method on older SDV builds that lack it.
            try:
                from sdv.utils import load_synthesizer as _load_synthesizer
            except ImportError:
                _load_synthesizer = None
            if _load_synthesizer is not None:
                synthesizer = _load_synthesizer(str(model_file))
            else:
                synthesizer = HMASynthesizer.load(filepath=str(model_file))
            self.synthesizer = synthesizer
            self._fitted_sample_sizes = {
                str(k): int(v) for k, v in (sidecar.get("fitted_sample_sizes") or {}).items()
            }
            self.is_fitted = True
            self.logger.info(
                f"♻️ Reusing cached synthesizer {model_file.name} "
                f"(config fingerprint {fingerprint}) — skipping training."
            )
            return True
        except Exception as exc:
            self.logger.warning(f"⚠️ Could not reuse cached synthesizer ({exc}); retraining.")
            self.is_fitted = False
            self.synthesizer = None
            return False

    def save_model_artifacts(self, output_dir: Optional[str] = None) -> Optional[Path]:
        """Save the fitted synthesizer, keyed by config fingerprint, when enabled.

        Writes ``model_<fingerprint>.pkl`` plus a ``model_<fingerprint>.json``
        sidecar (config fingerprint, SDV version, fitted sample sizes) and a
        ``model_<fingerprint>.metadata.json``. The fingerprint key means a later
        run reuses the model only for the exact config it was trained on; an SDV
        upgrade is detected on load and triggers a retrain.
        """
        if not self.is_fitted or self.synthesizer is None or self.metadata is None:
            return None

        save_enabled = self.config_parser.get_setting('save_model_artifact', False)
        if not bool(save_enabled):
            return None

        # The artifact format (single pickled synthesizer + SDV-version
        # sidecar) only describes the SDV engine. Other engines hold a model
        # per table; silently writing an unloadable artifact would be worse
        # than not caching.
        engine_name = self.resolve_engine_name()
        if engine_name != "sdv":
            self.logger.info(
                f"ℹ️ Model artifact caching is only supported for the 'sdv' engine "
                f"(current: {engine_name!r}) — skipping save."
            )
            return None

        artifact_dir = self._model_artifact_dir(output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        fingerprint = self._config_fingerprint()
        model_file = artifact_dir / f"model_{fingerprint}.pkl"
        sidecar_file = artifact_dir / f"model_{fingerprint}.json"
        metadata_file = artifact_dir / f"model_{fingerprint}.metadata.json"

        # Artifact for this exact config is already on disk — nothing to do.
        if model_file.is_file() and sidecar_file.is_file():
            self.logger.info(f"💾 Synthesizer artifact already current: {model_file.name}")
            return artifact_dir

        try:
            self.metadata.save_to_json(filepath=str(metadata_file))
        except Exception as exc:
            self.logger.warning(f"⚠️ Could not save metadata artifact: {exc}")

        try:
            import sdv as _sdv
            self.synthesizer.save(filepath=str(model_file))
            sidecar = {
                "fingerprint": fingerprint,
                "sdv_version": _sdv.__version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tables": sorted(self.tables_config.keys()),
                "fitted_sample_sizes": self._fitted_sample_sizes,
            }
            sidecar_file.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            self.logger.info(
                f"💾 Saved synthesizer artifact {model_file.name} "
                f"(config fingerprint {fingerprint})"
            )
            return artifact_dir
        except Exception as exc:
            self.logger.warning(f"⚠️ Could not save synthesizer artifact: {exc}")
            return artifact_dir if metadata_file.exists() else None
