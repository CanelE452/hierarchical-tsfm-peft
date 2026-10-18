"""Serialized whole physical inference including outer RevIN and its inverse."""
import importlib
from pathlib import Path
import sys

from .revin import B2RevIN, model_factory, revin_policy, scientific_sources


_campaign = Path(__file__).resolve().parents[1] / "informs_broad_validation_20261015_v1"
if str(_campaign) not in sys.path:
    sys.path.insert(0, str(_campaign))
_deployment = importlib.import_module("deployment")
if Path(_deployment.__file__).resolve() != (_campaign / "deployment.py").resolve():
    raise ValueError("Pinned physical deployment module import root changed")


class RevINRoute(_deployment.Route):
    def __init__(self, *args, ctx=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            if self.bundle is not None and self.family in ("B2", "B0"):
                enabled = self.bundle["spec"]["revin"]
                if type(enabled) is not bool or enabled != isinstance(self.model, B2RevIN):
                    raise ValueError("Serialized RevIN policy differs from loaded model")
                if self.bundle["normalization"]["revin"] != revin_policy(enabled):
                    raise ValueError("Serialized RevIN normalization policy changed")
                if ctx is not None and self.bundle["normalization"]["new_scientific_source_hashes"] != scientific_sources(ctx):
                    raise ValueError("Serialized RevIN scientific source binding changed")
        except BaseException:
            self.close()
            raise

    def _predict_normalized(self, x, physical, observed):
        if not isinstance(self.model, B2RevIN):
            return super()._predict_normalized(x, physical, observed)
        window, context = self._step("RevIN_norm", lambda: self.model.revin.normalize(x))
        base = self.model.base
        z = self._step("P_E", lambda: base.encode(window, self.latent_k))
        future = self._foundation(z)
        prediction = self._step("D", lambda: base.decode(window, z, future, self.latent_k))
        return self._step("RevIN_inverse", lambda: self.model.revin.inverse(prediction, context))


def load_route(route_spec, task, cache_root, *, ctx, **kwargs):
    with model_factory(ctx):
        return RevINRoute(route_spec, task, cache_root, ctx=ctx, **kwargs)
