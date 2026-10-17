"""Jena near-capacity control: explicit learned cross-time mixing removed.

The two original whole-window LayerNorm operations still share window statistics.
The control therefore is not fully time independent. Training is caller driven.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "B2_CAP_POINTWISE"
BOUNDS = dict(A_fit=[0, 10482], A_stop=[10482, 12579], M_cal=[12579, 14675],
              V_select=[14675, 16772], T_final=[16772, 20965])
PUBLIC = Path(__file__).resolve().parents[3] / "research/informs_broad_validation_20261015_v1"


def source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _modules():
    sys.path.insert(0, str(PUBLIC))
    import fit_base as fit
    import training
    import data_contract as data
    for module in (fit, training, data):
        if Path(module.__file__).resolve().parent != PUBLIC:
            raise ValueError("Scientific module root changed")
    return fit, training, data


def _new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


class PointwiseMLP(nn.Module):
    def __init__(self, channels=21, hidden=774):
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.input(x)))


class PointwiseCapacityMixer(nn.Module):
    def __init__(self, original_mixer, seed):
        super().__init__()
        self.temporal_norm = original_mixer.temporal_norm
        self.feature_norm = original_mixer.feature_norm
        self.feature = original_mixer.feature
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.temporal = PointwiseMLP(21, 774)

    def forward(self, x):
        y = x + self.temporal(self.temporal_norm(x))
        return y + self.feature(self.feature_norm(y))


def make_cap_from_b2(base, seed):
    """Replace only the mixer in a caller-supplied fresh B2 adapter, in place."""
    if (base.length, base.horizon, base.channels, base.latent_k, base.front.kind) != (512, 48, 21, 6, "B2"):
        raise ValueError("Only original Jena B2 L512/H48/C21/K6 is supported")
    if any(parameter.is_cuda for parameter in base.parameters()):
        raise ValueError("CPU-only capacity control required")
    if isinstance(base.front.mixer, PointwiseCapacityMixer):
        raise ValueError("Capacity control already installed")
    base.front.mixer = PointwiseCapacityMixer(base.front.mixer, seed)
    base.architecture = ARCHITECTURE
    return base


def create_cap(spec, task, fit_end, foundation):
    """Create the original seeded B2 first, then isolate the replacement RNG."""
    fit, _, _ = _modules()
    original = copy.deepcopy(spec)
    original["family"] = "B2"
    if original.get("device", "cpu") != "cpu" or int(original["seed"]) not in (92601, 92602):
        raise ValueError("Only two registered CPU seeds are supported")
    model, normalization = fit.create_model(original, task, fit_end, foundation)
    return make_cap_from_b2(model, original["seed"]), normalization


def make_spec(seed):
    if seed not in (92601, 92602):
        raise ValueError("Only the two registered Jena seeds are supported")
    return dict(run_id=f"mechanism_jena_B2_CAP_POINTWISE_k6_s{seed}_v1", dataset="jena",
                family="B2", architecture=ARCHITECTURE, latent_k=6, seed=seed, fold="primary",
                epochs=[.25, 1., 3.], lr=.001, device="cpu", microbatch=4, effective_batch=16,
                F_row_microbatch=128, checkpoint_inputs=False, lora_rank=8, lora_alpha=16,
                stage_label="NEW_FINITE_MECHANISM_CAPACITY_CONTROL", cap_control_sha256=source_sha())


def _stage(spec, task, fit_end, stop_bounds, epochs, private, public, foundation,
           deadline, contract):
    fit, _, _ = _modules()
    model, normalization = create_cap(spec, task, fit_end, foundation)
    try:
        return fit._stage_body(spec, task, fit_end, stop_bounds, epochs, private, public,
                               foundation, deadline, contract, model, normalization)
    finally:
        if callable(getattr(model, "close", None)):
            model.close()


def fit_cap(seed, task, foundation, new_private_run, new_public_run, deadline):
    """Finite original endpoint search and selected-length fresh A60 refit."""
    fit, training, data = _modules()
    if time.time() >= deadline or not math.isfinite(deadline):
        raise ValueError("Live finite new-campaign deadline required")
    if task["physical"].shape != (16772, 21) or task["split_ranges"] != BOUNDS:
        raise ValueError("Original legal Jena prefix and splits required")
    private, public = Path(new_private_run).resolve(), Path(new_public_run).resolve()
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    spec = make_spec(seed)
    scientific_source, files = fit._source_contract()
    contract = dict(source_sha=scientific_source,
                    config_sha=training.fingerprint(dict(spec=spec, architecture=ARCHITECTURE,
                                                        cap_control_sha256=source_sha())),
                    data_sha=task["data_sha256"], model_sha=fit._model_sha(foundation))
    _new(public / "SOURCE_CONTRACT.json", dict(**contract, files=files, spec=spec,
         cap_control_sha256=source_sha(), architecture=ARCHITECTURE,
         whole_window_LayerNorm_preserved=True, explicit_learned_cross_time_mixing=False,
         complete_time_independence_claimed=False, normalization_fit="stage prefix only",
         pointwise_branch_shape=[21, 774, 21], original_B2_changed=False))
    search = _stage(spec, task, 10482, BOUNDS["A_stop"], [.25, 1., 3.],
                                 private / "search", public / "search", foundation, deadline, contract)
    _new(public / "SEARCH_SUMMARY.json", search["summary"])
    _new(public / "SEARCH_CURVES.json", search["curves"])
    if search["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="PARTIAL", phase="search", search=search, spec=spec, contract=contract)
    epoch = search["summary"]["selected_epoch"]
    if epoch not in (0., .25, 1., 3.) or search["summary"]["completed_epochs"] != [0., .25, 1., 3.]:
        raise ValueError("Original full endpoint search required")
    refit = _stage(spec, task, 12579, None, [epoch] if epoch else [],
                                private / "refit", public / "refit", foundation, deadline, contract)
    _new(public / "REFIT_SUMMARY.json", refit["summary"])
    _new(public / "REFIT_CURVES.json", refit["curves"])
    if refit["summary"]["status"] != "SEGMENT_COMPLETE":
        return dict(status="PARTIAL", phase="refit", search=search, refit=refit, spec=spec, contract=contract)
    if source_sha() != spec["cap_control_sha256"]:
        raise ValueError("Capacity-control source changed during fit")
    bundle = dict(model=refit["model"], normalization=refit["normalization"], spec=spec,
                  contract=refit["summary"]["contract"], selected_epoch=epoch,
                  model_state_sha=training.fingerprint({key: data.array_sha256(value.numpy())
                                                      for key, value in refit["model"].items()}),
                  search_summary=search["summary"], refit_summary=refit["summary"],
                  architecture=ARCHITECTURE, cap_control_sha256=source_sha())
    artifact = private / "selected_bundle.pt"
    with artifact.open("xb") as handle:
        torch.save(bundle, handle)
    return dict(status="COMPLETE_STAGES", model=refit["model"], normalization=refit["normalization"],
                selected_epoch=epoch, search=search, refit=refit, bundle=bundle,
                artifact=artifact, spec=spec, contract=refit["summary"]["contract"])


if __name__ == "__main__":
    print(json.dumps(dict(status="CALLER_CONTROLLED_CAP_CONTROL_API", training_launched=False)))
