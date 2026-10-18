"""Read-only canonical parents and frozen, full-window TS2Vec exports."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random as python_random

import numpy as np
import torch

from research.professor_compressor_direct_20261011.models_gt import GTConvAE
from research.professor_compressor_direct_20261011 import ts2_adapter


ROOT = Path(__file__).resolve().parents[2]
PARENT = Path("research/professor_compressor_direct_20261011")
CONTROL_PINS = {
    "protocol.json": "c49f2064629779f425003ca129034ef77705cfc92a0c87907bd0c7d29414bc53",
    "runs.json": "b3039a56768026775e839130cd49ac89ebf86ef3375358efb38202153f075639",
}
RESULT_PINS = {
    "jena_GT_full_s92601": "1aa6dd1296879962b60bc48d6295218300480f3d3c75d5e6c246daff405bc4f9",
    "jena_GT_full_s92602": "9401d9768d290942491cac90215211e11ecc08ac1377052cb64ec860267f725a",
    "jena_TS2_d320_s92601_retry_batch4_v4": "a5f07966f4abd51ef2efa267037a8335810d3f3f24c443373018f9f384cbe781",
    "jena_TS2_d320_s92602_retry_batch4_v4": "12048472f99ec66aa0f47d247249f7cce114960e80446ff8f67a46efdf202f57",
}
# GT's original receipt predates file digests for checkpoints. Pin the selected
# complete B files explicitly, alongside the already-pinned result receipts.
GT_CHECKPOINT_PINS = {
    "jena_GT_full_s92601": "014aef056a3b29cb6836539f8923651d075bcafdf8d3fa1a511e9df9be3f0d3b",
    "jena_GT_full_s92602": "981d061c723da0b57a4a78ed88bea75fb21a90b56ef072dfe3094c72243a06ef",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(path, expected):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Parent SHA mismatch: {Path(path).name}")
    return actual


def _asset_path(asset_root, value, identifier, filename):
    relative = Path(str(value).replace("\\", "/"))
    expected = Path(".cache/professor_compressor_direct_20261011/runs") / identifier / filename
    if relative != expected:
        raise ValueError(f"Parent artifact must belong to exact canonical run: {identifier}/{filename}")
    return asset_root / expected


def _verify_sources(result):
    checked = {}
    historical = None
    for value, expected in result["source_pins"].items():
        relative = Path(value.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Parent source pin must be repository relative")
        if relative == PARENT / "protocol.json" and result["job"]["family"] == "GT":
            snapshot = Path("source_snapshots/protocol_memory_retry_v3/protocol_v2.json")
            _require_sha(ROOT / PARENT / snapshot, expected)
            historical = {"source": snapshot.as_posix(), "sha256": expected,
                          "status": "verified_original_protocol_snapshot"}
            checked[relative.as_posix()] = expected
        else:
            checked[relative.as_posix()] = _require_sha(ROOT / relative, expected)
    if result["job"]["family"] == "GT" and historical is None:
        raise ValueError("GT receipt must pin its historical protocol")
    return checked, historical


def _prediction_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ("CALIBRATED_F", "targets", "mask", "origins", "mean", "std")}
    for key in ("CALIBRATED_F", "targets", "mask"):
        if arrays[key].shape != (371, 48, 21):
            raise ValueError(f"Historical parent prediction shape changed: {key}")
    if arrays["mask"].dtype != np.bool_ or arrays["origins"].shape != (371,):
        raise ValueError("Historical parent mask/origin schema changed")
    if not np.issubdtype(arrays["origins"].dtype, np.integer) or np.any(np.diff(arrays["origins"]) <= 0):
        raise ValueError("Historical parent origins must be increasing integers")
    for key in ("mean", "std"):
        if arrays[key].shape != (21,) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid historical normalization: {key}")
    if np.any(arrays["std"] <= 0):
        raise ValueError("Historical parent standard deviations must be positive")
    if not np.isfinite(arrays["CALIBRATED_F"]).all() or not np.isfinite(arrays["targets"][arrays["mask"]]).all():
        raise ValueError("Historical observed targets/predictions must be finite")
    return arrays


def build_bindings(asset_root):
    """Validate canonical receipts and return publishable metadata/private assets.

    Only TRAIN/VAL parent artifacts are opened. This function performs no model
    fitting, CUDA initialization, filesystem writes, or original-loader calls.
    """
    asset_root = Path(asset_root).resolve()
    base = ROOT / PARENT
    for filename, expected in CONTROL_PINS.items():
        _require_sha(base / filename, expected)
    protocol = _read_json(base / "protocol.json")
    runs = _read_json(base / "runs.json")
    public, private = {}, {}
    common_arrays, common_contract = None, None
    for seed in (92601, 92602):
        public[str(seed)], private[str(seed)] = {}, {}
        for family in ("GT", "TS2"):
            identifier = (f"jena_GT_full_s{seed}" if family == "GT" else
                          f"jena_TS2_d320_s{seed}_retry_batch4_v4")
            planned = [job for job in protocol["jobs"] if job["id"] == identifier]
            completed = [run for run in runs["runs"] if run["job"]["id"] == identifier]
            if len(planned) != 1 or len(completed) != 1:
                raise ValueError(f"Missing/duplicate canonical parent: {identifier}")
            result_path = base / "runs" / identifier / "result.json"
            _require_sha(result_path, RESULT_PINS[identifier])
            result = _read_json(result_path)
            for job in (planned[0], completed[0]["job"], result["job"]):
                if (job["id"], job["dataset"], job["seed"], job["family"]) != (identifier, "jena", seed, family):
                    raise ValueError(f"Canonical parent mapping differs: {identifier}")
                if family == "GT" and job.get("no_spatial") is not False:
                    raise ValueError("Canonical GT parent must retain the spatial product")
                if family == "TS2" and job.get("dim") != 320:
                    raise ValueError("Canonical TS2 parent must have 320 features")
            if result["status"] != "complete" or completed[0]["status"] != "complete_saved_predictions_verified":
                raise ValueError(f"Canonical parent is incomplete: {identifier}")
            expected_result_path = (PARENT / "runs" / identifier / "result.json").as_posix()
            if completed[0]["result_path"].replace("\\", "/") != expected_result_path:
                raise ValueError("Canonical manifest points to another result")
            sources, historical = _verify_sources(result)
            prediction = _asset_path(asset_root, result["prediction_path"], identifier, "prediction.npz")
            diagnostics = _asset_path(asset_root, result["diagnostic_path"], identifier, "diagnostic.npz")
            _require_sha(prediction, result["prediction_sha256"])
            _require_sha(diagnostics, result["diagnostic_sha256"])
            arrays = _prediction_arrays(prediction)
            if common_arrays is None:
                common_arrays, common_contract = arrays, result["data_contract_hash"]
            elif result["data_contract_hash"] != common_contract or any(
                    not np.array_equal(arrays[key], common_arrays[key], equal_nan=True)
                    for key in ("targets", "mask", "origins", "mean", "std")):
                raise ValueError("Parent physical coordinates, targets, masks or origins differ")
            metadata = {"id": identifier, "seed": seed, "dataset": "jena", "family": family,
                        "result_sha256": RESULT_PINS[identifier],
                        "prediction_sha256": result["prediction_sha256"],
                        "diagnostic_sha256": result["diagnostic_sha256"],
                        "data_contract_hash": result["data_contract_hash"], "source_pins": sources,
                        "canonical_protocol_sha256": CONTROL_PINS["protocol.json"],
                        "canonical_runs_sha256": CONTROL_PINS["runs.json"],
                        "selection_exposure": "existing development VAL; subsequent development diagnosis"}
            local = {"result": result, "result_path": result_path, "prediction": prediction,
                     "diagnostics": diagnostics, "arrays": arrays}
            calibration, training = result["calibration"], result["training"]
            if family == "GT":
                checkpoint = _asset_path(asset_root, calibration["checkpoint_path"], identifier,
                                         "forecast_calibrated_best.pt")
                checkpoint_sha = GT_CHECKPOINT_PINS[identifier]
                metadata.update(no_spatial=False, historical_protocol=historical,
                                checkpoint_filename=checkpoint.name,
                                selection_history={"reconstruction_epoch": training["selected_epoch"],
                                                   "reconstruction_selection": training["selection"],
                                                   "forecast_epoch": calibration["selected_epoch"],
                                                   "forecast_selection": calibration["selection"],
                                                   "restored_state": "complete native E/shared product/D(B), graph from checkpoint buffer"})
            else:
                encoder = training["encoder"]
                config = encoder["configuration"]
                required = {"input_dims": 21, "output_dims": 320, "hidden_dims": 64, "depth": 10,
                            "batch_size": 4, "max_train_length": 3000}
                if any(config[key] != value for key, value in required.items()):
                    raise ValueError("Canonical TS2 encoder configuration differs")
                checkpoint = _asset_path(asset_root, encoder["checkpoint"], identifier, "encoder_swa.pt")
                checkpoint_sha = encoder["checkpoint_sha256"]
                decoder = _asset_path(asset_root, calibration["best_checkpoint"], identifier,
                                      "forecast_calibrated_decoder_best.pt")
                decoder_sha = calibration["best_checkpoint_sha256"]
                decoder_available = decoder.is_file()
                if decoder_available:
                    _require_sha(decoder, decoder_sha)
                metadata.update(checkpoint_filename=checkpoint.name, encoder_configuration=required,
                                decoder_checkpoint_filename=decoder.name, decoder_checkpoint_sha256=decoder_sha,
                                decoder_available=decoder_available,
                                historical_protocol={"source": "protocol.json", "sha256": CONTROL_PINS["protocol.json"],
                                                     "status": "canonical_batch4_v4_protocol; not independently pinned in original TS2 receipt"},
                                selection_history={"encoder": "fixed official contrastive iteration budget; SWA inference",
                                                   "decoder_reconstruction_epoch": training["decoder_reconstruction"]["best_epoch"],
                                                   "decoder_reconstruction_selection": training["decoder_reconstruction"]["checkpoint_selection"],
                                                   "forecast_epoch": calibration["best_epoch"],
                                                   "forecast_selection": calibration["checkpoint_selection"]})
                local.update(decoder_checkpoint=decoder, decoder_checkpoint_sha256=decoder_sha)
            _require_sha(checkpoint, checkpoint_sha)
            metadata["checkpoint_sha256"] = checkpoint_sha
            public[str(seed)][family] = metadata
            private[str(seed)][family] = {**metadata, **local, "checkpoint": checkpoint}
    return public, private


@contextmanager
def _isolated_rng(seed=None):
    python_state = python_random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if seed is not None:
            python_random.seed(seed)
            np.random.seed(seed + 1)
            # torch.manual_seed also queues CUDA seeding before CUDA initialization.
            # Seed only the CPU generator used by the CPU-only constructors.
            torch.random.default_generator.manual_seed(seed + 2)
        yield
    finally:
        python_random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _freeze_ts2(model):
    ts2_adapter.freeze_encoder(model)
    return model


def load_gt(binding, device="cpu"):
    """Restore the complete calibrated native GT E/product/D from its own graph."""
    _require_sha(binding["checkpoint"], binding["checkpoint_sha256"])
    with _isolated_rng():
        saved = torch.load(binding["checkpoint"], map_location="cpu", weights_only=True)
        state = saved["state_dict"]
        model = GTConvAE(state["product.graph"], no_spatial=False)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
    return model


def load_ts2(binding, device="cpu", random=False):
    """Construct on CPU, restore SWA only, then move only inference SWA to device."""
    if not random:
        _require_sha(binding["checkpoint"], binding["checkpoint_sha256"])
    with _isolated_rng(int(binding["seed"])):
        model = ts2_adapter.official_class()(
            input_dims=21, output_dims=320, hidden_dims=64, depth=10,
            device="cpu", lr=.001, batch_size=4, max_train_length=3000, temporal_unit=0,
        )
        if not random:
            state = torch.load(binding["checkpoint"], map_location="cpu", weights_only=True)
            model.net.load_state_dict(state, strict=True)
        _freeze_ts2(model)
        model.net.to(device)
        model.device = str(device)
    return model


def load_ts2_decoder(binding, device="cpu"):
    _require_sha(binding["decoder_checkpoint"], binding["decoder_checkpoint_sha256"])
    with _isolated_rng():
        saved = torch.load(binding["decoder_checkpoint"], map_location="cpu", weights_only=True)
        model = ts2_adapter.ReconstructionDecoder(320, 21)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to(device).eval().requires_grad_(False)
    return model


def export_tensor(encoder, x):
    """Frozen SWA [B,L,21] -> [B,L,320], retaining the entire supplied history."""
    if x.ndim != 3 or x.shape[-1] != 21 or not x.shape[0] or not x.shape[1]:
        raise ValueError("TS2Vec export requires nonempty [B,L,21]")
    _freeze_ts2(encoder)
    device = next(encoder.net.parameters()).device
    with torch.no_grad():
        return encoder.net(x.detach().to(device=device, dtype=torch.float32).clone(), mask="all_true")


def export_official(encoder, x_numpy):
    """Official NumPy parity reference; isolate DataLoader's global RNG draw."""
    _freeze_ts2(encoder)
    with _isolated_rng():
        return ts2_adapter.export_offline(encoder, x_numpy, batch_size=8)
