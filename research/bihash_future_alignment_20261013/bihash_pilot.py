"""Bounded Bi-Hash future-alignment pilot.

This is not an ExoTimer reproduction.  It transfers the Bi-Hash mechanism
from text/time-series alignment to a past/future representation question:
can a learnable hash boundary trained on FIT futures make the past-only
forecast path more predictive?
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / HERE.name
SMALLK = ROOT / "research" / "smallk_predictive_alignment_20261013"
sys.path.insert(0, str(SMALLK))

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

import engine as en
from alignment import future_coordinates, nonoverlap_permutation, summary_numpy, summary_torch

DATASETS = ("jena", "robin")
SEEDS = (92601, 92602)
CHECKPOINTS = (0, 50, 100, 200)
HISTORY, HORIZON = 512, 48
METHODS = ("BASE", "SUMMARY_MSE", "BIHASH", "BIHASH_SHUFFLED")
DEFAULT_UPDATES = 200


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value, replace=False):
    path = Path(path)
    resolved = path.resolve()
    if not (resolved.is_relative_to(HERE.resolve()) or resolved.is_relative_to(CACHE.resolve())):
        raise ValueError("WRITE_OUTSIDE_BIHASH_CAMPAIGN")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if replace:
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)


def save_checkpoint(path, value):
    path = Path(path)
    resolved = path.resolve()
    if not resolved.is_relative_to(CACHE.resolve()):
        raise ValueError("CHECKPOINT_OUTSIDE_BIHASH_CACHE")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        torch.save(value, f)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


class Limit:
    def __init__(self, tag, gpu=True):
        self.tag = tag
        self.gpu = gpu
        self.started = time.perf_counter()
        self.proc = psutil.Process()
        self.record = {
            "tag": tag,
            "pid": os.getpid(),
            "started_unix": time.time(),
            "gpu_process": gpu,
            "wall_seconds": 0.0,
            "peak_sampled_rss_bytes": 0,
            "peak_allocated_vram_bytes": 0,
            "new_bytes": 0,
            "status": "RUNNING",
        }
        self.path = HERE / "resources" / f"{tag}_{os.getpid()}_{time.time_ns()}.json"
        self.last_storage = -60.0
        self.check()

    def previous_wall(self):
        total = 0.0
        for p in (HERE / "resources").glob("*.json"):
            if p == self.path:
                continue
            try:
                r = json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if r.get("gpu_process") == self.gpu:
                total += float(r.get("wall_seconds", 0.0))
        return total

    def check(self):
        self.record["wall_seconds"] = time.perf_counter() - self.started
        self.record["peak_sampled_rss_bytes"] = max(
            self.record["peak_sampled_rss_bytes"], self.proc.memory_info().rss
        )
        if torch.cuda.is_available():
            self.record["peak_allocated_vram_bytes"] = max(
                self.record["peak_allocated_vram_bytes"], torch.cuda.max_memory_allocated()
            )
        if self.record["wall_seconds"] - self.last_storage > 30:
            self.record["new_bytes"] = sum(
                p.stat().st_size for base in (HERE, CACHE) if base.exists() for p in base.rglob("*") if p.is_file()
            )
            self.last_storage = self.record["wall_seconds"]
        self.record["cumulative_wall_seconds"] = self.previous_wall() + self.record["wall_seconds"]
        write_json(self.path, self.record, replace=True)
        if self.record["cumulative_wall_seconds"] > 5400:
            raise RuntimeError("BIHASH_PILOT_WALL_LIMIT_5400S")
        if max(self.record["peak_sampled_rss_bytes"], self.record["peak_allocated_vram_bytes"]) > 8 * 1024**3:
            raise RuntimeError("BIHASH_PILOT_RAM_OR_VRAM_LIMIT_8GIB")
        if self.record["new_bytes"] > 4 * 1024**3:
            raise RuntimeError("BIHASH_PILOT_STORAGE_LIMIT_4GIB")

    def close(self, status, error=None):
        self.record["wall_seconds"] = time.perf_counter() - self.started
        self.record["status"] = status
        self.record["error"] = error
        write_json(self.path, self.record, replace=True)


def metric(data, prediction, target, observed):
    return en.metrics(data, prediction, target, observed)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device="cuda")


def bool_tensor(x):
    return torch.as_tensor(x, dtype=torch.bool, device="cuda")


class BiHashFutureAlignment(nn.Module):
    """FIT-future target alignment with ExoTimer-style learnable hash projections."""

    def __init__(self, target, observed, history, basis, origins, seed, bits=16, temperature=0.7):
        super().__init__()
        k = int(basis.shape[1])
        self.bits = int(bits)
        self.temperature = float(temperature)
        self.q = nn.Linear(k, k)
        with torch.no_grad():
            self.q.weight.copy_(torch.eye(k))
            self.q.bias.zero_()
        coord, valid, projection_diagnostic = future_coordinates(target, observed, history[:, -1], basis, True)
        summ, smask = summary_numpy(coord, valid)
        usable = smask.all(1)
        means = np.array([summ[smask[:, j], j].mean() if smask[:, j].any() else 0.0 for j in range(3 * k)], np.float32)
        stds = np.array([summ[smask[:, j], j].std() if smask[:, j].any() else 1.0 for j in range(3 * k)], np.float32)
        stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
        normalized = ((summ - means) / stds).astype(np.float32)
        latent_std = coord[valid].std(0) if valid.any() else np.ones(k, np.float32)
        latent_std = np.where(latent_std > 1e-6, latent_std, 1.0).astype(np.float32)

        usable_indices = np.flatnonzero(usable)
        permutation = np.arange(len(origins))
        shuffle_error = None
        try:
            if len(usable_indices) < 2:
                raise ValueError("INSUFFICIENT_COMPLETE_ORIGINS_FOR_SHUFFLE")
            permutation[usable_indices] = usable_indices[nonoverlap_permutation(origins[usable], seed, HORIZON)]
        except ValueError as err:
            shuffle_error = str(err)

        d = 3 * k
        torch.manual_seed(seed + 17)
        self.future_hash = nn.Linear(d, bits)
        self.past_hash = nn.Linear(d, bits)
        self.hybrid = nn.Sequential(nn.Linear(2 * d, max(16, 2 * d)), nn.GELU(), nn.Linear(max(16, 2 * d), d))
        for name, array in {
            "target": coord,
            "valid": valid,
            "summary_target": normalized,
            "summary_valid": smask,
            "means": means,
            "stds": stds,
            "latent_std": latent_std,
            "shuffle": permutation,
        }.items():
            self.register_buffer(name, torch.from_numpy(array))
        self.metadata = {
            "basis_sha256": ahash(basis),
            "target_sha256": ahash(coord),
            "valid_sha256": ahash(valid),
            "summary_target_sha256": ahash(normalized),
            "summary_mean": means.tolist(),
            "summary_std": stds.tolist(),
            "complete_summary_origins": int(usable.sum()),
            "valid_horizons": int(valid.sum()),
            "total_horizons": int(valid.size),
            "shuffle_available": shuffle_error is None,
            "shuffle_failure_reason": shuffle_error,
            "shuffle_is_bijection": bool(len(np.unique(permutation)) == len(origins)),
            "shuffle_min_origin_distance_complete_subset": int(np.min(np.abs(origins[usable] - origins[permutation[usable]]))) if usable.any() else None,
            "bits": int(bits),
            "temperature": float(temperature),
            "projection_rule": "FIT future only; observed weighted LS to deployed latent basis",
            **projection_diagnostic,
        }

    def summary(self, delta, valid):
        return (summary_torch(self.q(delta), valid) - self.means) / self.stds

    def soft_hash(self, value):
        logits = torch.stack((value, -value), dim=-1)
        probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        return 2.0 * probs[..., 0] - 1.0

    def deterministic_hash(self, value):
        return torch.sign(value).clamp_min(0) * 2.0 - 1.0

    def continuous_loss(self, delta, indices, shuffled=False):
        valid = self.valid[indices]
        summary = self.summary(delta, valid)
        target_indices = self.shuffle[indices] if shuffled else indices
        mask = self.summary_valid[indices].all(1) & self.summary_valid[target_indices].all(1)
        if not mask.any():
            raise ValueError("NO_COMPLETE_SUMMARY_BATCH")
        return (summary[mask] - self.summary_target[target_indices][mask]).square().mean()

    def bihash_loss(self, delta, indices, shuffled=False):
        valid = self.valid[indices]
        summary = self.summary(delta, valid)
        target_indices = self.shuffle[indices] if shuffled else indices
        usable = self.summary_valid[indices].all(1) & self.summary_valid[target_indices].all(1)
        if not usable.any():
            raise ValueError("NO_COMPLETE_SUMMARY_BATCH")
        h = summary[usable]
        p = self.summary_target[target_indices][usable]
        bh = self.soft_hash(self.past_hash(h))
        bp = self.soft_hash(self.future_hash(p))
        logits = bh @ bp.T / math.sqrt(self.bits)
        labels = torch.arange(len(h), device=h.device)
        ce = F.cross_entropy(logits, labels)
        weights = torch.softmax(logits, dim=1)
        retrieved = weights @ p
        predicted = self.hybrid(torch.cat((h, retrieved), dim=1))
        mse = (predicted - p).square().mean()
        bit_balance = (bh.mean(0).square().mean() + bp.mean(0).square().mean())
        return ce + 0.2 * mse + 0.02 * bit_balance

    def loss(self, method, delta, indices):
        if method == "BASE":
            return delta.sum() * 0.0
        if method == "SUMMARY_MSE":
            return self.continuous_loss(delta, indices, False)
        if method == "BIHASH":
            return self.bihash_loss(delta, indices, False)
        if method == "BIHASH_SHUFFLED":
            if not self.metadata["shuffle_available"]:
                raise ValueError(self.metadata["shuffle_failure_reason"])
            return self.bihash_loss(delta, indices, True)
        raise ValueError(f"UNKNOWN_METHOD:{method}")

    def diagnostics(self, delta, split_name):
        with torch.no_grad():
            valid = self.valid[: len(delta)] if split_name == "fit" else torch.ones(delta.shape[:2], dtype=torch.bool, device=delta.device)
            if split_name != "fit":
                # TUNE has no stored future hash target.  Use distribution health only.
                value = self.q(delta)
                flat = value.reshape(-1, value.shape[-1]).double()
                cov = torch.cov(flat.T)
                eig = torch.linalg.eigvalsh(cov.reshape(value.shape[-1], value.shape[-1]) if value.shape[-1] == 1 else cov).clamp_min(0)
                weights = eig / eig.sum().clamp_min(1e-20)
                return {
                    "q_variance": flat.var(0).cpu().tolist(),
                    "q_effective_rank": float(torch.exp(-(weights * weights.clamp_min(1e-20).log()).sum())),
                }
            summary = self.summary(delta, valid)
            usable = self.summary_valid.all(1)
            h = summary[usable]
            p = self.summary_target[usable]
            bh = self.deterministic_hash(self.past_hash(h))
            bp = self.deterministic_hash(self.future_hash(p))
            logits = bh @ bp.T / math.sqrt(self.bits)
            nearest = logits.argmax(1)
            top1 = float((nearest == torch.arange(len(h), device=h.device)).float().mean()) if len(h) else None
            before = ((delta - self.target) / self.latent_std).square()
            after = ((self.q(delta) - self.target) / self.latent_std).square()
            mask = self.valid[:, :, None]
            flat = self.q(delta).reshape(-1, delta.shape[-1]).double()
            cov = torch.cov(flat.T)
            if delta.shape[-1] == 1:
                cov = cov.reshape(1, 1)
            eig = torch.linalg.eigvalsh(cov).clamp_min(0)
            weights = eig / eig.sum().clamp_min(1e-20)
            return {
                "fit_hash_top1_self_retrieval": top1,
                "fit_hash_diag_logit_mean": float(torch.diag(logits).mean()) if len(h) else None,
                "fit_hash_offdiag_logit_mean": float((logits.sum() - torch.diag(logits).sum()) / max(logits.numel() - len(h), 1)) if len(h) else None,
                "before_q_latent_future_normalized_mse": float((before * mask).sum() / (mask.sum().clamp_min(1) * delta.shape[-1])),
                "after_q_latent_future_normalized_mse": float((after * mask).sum() / (mask.sum().clamp_min(1) * delta.shape[-1])),
                "q_variance": flat.var(0).cpu().tolist(),
                "q_effective_rank": float(torch.exp(-(weights * weights.clamp_min(1e-20).log()).sum())),
                "past_hash_bit_mean": bh.mean(0).cpu().tolist(),
                "future_hash_bit_mean": bp.mean(0).cpu().tolist(),
            }


def predict_with_q(history, codec, bolt, alignment, limit):
    result = []
    with torch.no_grad():
        for i in range(0, len(history), 8):
            x = tensor(history[i : i + 8])
            z = codec.encode(x)
            future = bolt(z)
            delta = future - z[:, -1:]
            q_delta = alignment.q(delta)
            q_future = z[:, -1:] + q_delta
            pred = x[:, -1:] + codec.decode(q_future) - codec.decode(z[:, -1:])
            result.append(pred.cpu().numpy())
            limit.check()
    return np.concatenate(result)


def fit_one(data, history, target, observed, seed, method, weight, updates, bolt, output, limit):
    c = data["x"].shape[1]
    k = max(1, c // 4)
    mean, basis, init = en.ex.initialization(data, k, "PCA")
    codec = en.Codec(mean, basis, seed, "ANCHOR_NONLINEAR_D").to("cuda")
    alignment = None if method == "BASE" else BiHashFutureAlignment(
        target["fit"], observed["fit"], history["fit"], basis, data["origins"]["fit"], seed
    ).to("cuda")
    folder = CACHE / output.name / "models" / f"{data['dataset']}_{method}_w{weight:g}_s{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    params = list(codec.parameters())
    if alignment is not None:
        params += list(alignment.parameters())
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0)
    order = np.random.default_rng(seed).integers(0, len(history["fit"]), size=(updates, 4))
    checkpoints = []
    curve = []
    started = time.perf_counter()
    for update in range(updates + 1):
        if update in CHECKPOINTS or update == updates:
            pred = {s: en.predict(history[s], codec, bolt, limit) for s in ("fit", "tune")}
            artifact = save_checkpoint(
                folder / f"u{update}.pt",
                {
                    "codec": en.ex.previous.clone_state(codec),
                    "alignment": None if alignment is None else en.ex.previous.clone_state(alignment),
                    "method": method,
                    "weight": weight,
                    "seed": seed,
                    "update": update,
                },
            )
            checkpoint = {
                "update": update,
                **artifact,
                "forecast": {s: metric(data, pred[s], target[s], observed[s]) for s in pred},
            }
            if alignment is not None:
                pred_q = {s: predict_with_q(history[s], codec, bolt, alignment, limit) for s in ("fit", "tune")}
                checkpoint["forecast_q"] = {s: metric(data, pred_q[s], target[s], observed[s]) for s in pred_q}
            checkpoints.append(checkpoint)
            print(
                json.dumps(
                    {
                        "dataset": data["dataset"],
                        "seed": seed,
                        "method": method,
                        "weight": weight,
                        "update": update,
                        "tune_mse": checkpoint["forecast"]["tune"]["normalized"]["mse"],
                        "tune_q_mse": checkpoint.get("forecast_q", {}).get("tune", {}).get("normalized", {}).get("mse"),
                        "elapsed": time.perf_counter() - started,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        if update == updates:
            break
        idx = order[update]
        optimizer.zero_grad(set_to_none=True)
        out, delta = codec.forward_path(tensor(history["fit"][idx]), bolt, True)
        main = en.ex.previous.macro_loss(out, tensor(target["fit"][idx]), bool_tensor(observed["fit"][idx]))
        aux = main * 0.0 if alignment is None else alignment.loss(method, delta, torch.as_tensor(idx, device="cuda"))
        loss = main + weight * aux
        if not torch.isfinite(loss):
            raise ValueError("NONFINITE_LOSS")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad):
            raise ValueError("NONFINITE_GRADIENT")
        optimizer.step()
        curve.append(
            {
                "update": update + 1,
                "main_loss": float(main.detach()),
                "aux_loss": float(aux.detach()),
                "loss": float(loss.detach()),
                "gradient_norm": float(grad),
            }
        )
        if update % 10 == 0:
            limit.check()

    selected = min(checkpoints, key=lambda cp: cp["forecast"]["tune"]["normalized"]["mse"])
    selected_q = None
    if alignment is not None:
        selected_q = min(checkpoints, key=lambda cp: cp["forecast_q"]["tune"]["normalized"]["mse"])
    saved = torch.load(ROOT / selected["path"], map_location="cpu", weights_only=True)
    codec.load_state_dict(saved["codec"])
    if alignment is not None:
        alignment.load_state_dict(saved["alignment"])
    diagnostics = {}
    if alignment is not None:
        fit_delta = en.predict(history["fit"], codec, bolt, limit, delta=True)
        tune_delta = en.predict(history["tune"], codec, bolt, limit, delta=True)
        diagnostics["fit"] = alignment.diagnostics(tensor(fit_delta), "fit")
        diagnostics["tune"] = alignment.diagnostics(tensor(tune_delta), "tune")
    return {
        "dataset": data["dataset"],
        "seed": seed,
        "method": method,
        "weight": weight,
        "updates": updates,
        "batch_size": 4,
        "architecture": "ANCHOR_NONLINEAR_D",
        "initialization": init,
        "training_wall_seconds": time.perf_counter() - started,
        "order_sha256": ahash(order),
        "checkpoints": checkpoints,
        "selected_update": selected["update"],
        "selected_checkpoint": selected,
        "selected_q_update": None if selected_q is None else selected_q["update"],
        "selected_q_checkpoint": selected_q,
        "curve": curve,
        "alignment_metadata": None if alignment is None else alignment.metadata,
        "diagnostics": diagnostics,
        "selection_rule": "TUNE normalized observed channel-macro MSE; q path selected separately only for q-path diagnostic",
    }


def fixture(output):
    en.configure()
    protected = en.ex.access_guard()
    data = en.ex.load_data("jena", sizes=(64, 32, None))
    history, target, observed, _ = en.ex.arrays(data, HISTORY)
    c = data["x"].shape[1]
    mean, basis, _ = en.ex.initialization(data, max(1, c // 4), "PCA")
    align = BiHashFutureAlignment(target["fit"], observed["fit"], history["fit"], basis, data["origins"]["fit"], 92601).to("cuda")
    codec = en.Codec(mean, basis, 92601, "ANCHOR_NONLINEAR_D").to("cuda")
    bolt = en.ex.previous.FrozenBolt()
    en.replace_normalizer(bolt)
    x = tensor(history["fit"][:4])
    out, delta = codec.forward_path(x, bolt, True)
    loss = align.loss("BIHASH", delta, torch.arange(4, device="cuda"))
    loss.backward()
    hash_grad = sum(float(p.grad.detach().abs().sum()) for p in align.parameters() if p.grad is not None)
    codec_grad = sum(float(p.grad.detach().abs().sum()) for p in codec.parameters() if p.grad is not None)
    result = {
        "status": "PASS",
        "protected_test_guarded": protected,
        "dataset": "jena",
        "fit_origins": int(len(data["origins"]["fit"])),
        "tune_origins": int(len(data["origins"]["tune"])),
        "output_shape": list(out.shape),
        "loss": float(loss.detach()),
        "hash_parameters_receive_gradient_sum_abs": hash_grad,
        "codec_parameters_receive_gradient_sum_abs": codec_grad,
        "alignment_metadata": align.metadata,
        "source_modules": {
            "engine": str(Path(en.__file__).relative_to(ROOT)),
            "common": str(Path(sys.modules["common"].__file__).relative_to(ROOT)),
            "alignment": str(Path(sys.modules["alignment"].__file__).relative_to(ROOT)),
            "exotimer_pdf": "https://ojs.aaai.org/index.php/AAAI/article/download/39908/43869",
        },
    }
    if hash_grad <= 0 or codec_grad <= 0:
        raise ValueError("BIHASH_FIXTURE_NO_GRADIENT")
    write_json(output / "FIXTURE.json", result)
    return result


def run(tag, datasets, seeds, methods, updates, weight):
    output = HERE / f"run_{tag}"
    output.mkdir(parents=True, exist_ok=False)
    limit = Limit(tag, gpu=True)
    result = {
        "status": "RUNNING",
        "tag": tag,
        "started_unix": time.time(),
        "head": os.popen("git rev-parse HEAD").read().strip(),
        "datasets": list(datasets),
        "seeds": list(seeds),
        "methods": list(methods),
        "updates": updates,
        "weight": weight,
        "claims_scope": "development pilot only; no protected TEST and no fresh independent EVAL",
        "purpose": "Test whether ExoTimer-style learnable Bi-Hash boundaries can train past-only representations useful for future forecasting.",
        "source_modules": {},
        "cases": [],
    }
    try:
        en.configure()
        protected = en.ex.access_guard()
        result["protected_test_guarded"] = protected
        result["source_modules"] = {
            "engine": str(Path(en.__file__).relative_to(ROOT)),
            "common": str(Path(sys.modules["common"].__file__).relative_to(ROOT)),
            "alignment": str(Path(sys.modules["alignment"].__file__).relative_to(ROOT)),
            "this_script": str(Path(__file__).relative_to(ROOT)),
        }
        normalizer_fixture = en.normalizer_fixture()
        result["zero_safe_normalizer_fixture"] = normalizer_fixture
        bolt = en.ex.previous.FrozenBolt()
        en.replace_normalizer(bolt)
        for dataset in datasets:
            data = en.ex.load_data(dataset, sizes=(256, 64, None))
            history, target, observed, _ = en.ex.arrays(data, HISTORY)
            result.setdefault("data_contracts", {})[dataset] = {
                "source": data["source"],
                "source_hash": data["source_hash"],
                "fit_origins": int(len(data["origins"]["fit"])),
                "tune_origins": int(len(data["origins"]["tune"])),
                "val_origins_loaded_not_evaluated": int(len(data["origins"]["val"])),
                "normalization_fit_scope": data["normalization_fit_scope"],
                "channels": int(data["x"].shape[1]),
            }
            for seed in seeds:
                for method in methods:
                    case = fit_one(data, history, target, observed, seed, method, weight if method != "BASE" else 0.0, updates, bolt, output, limit)
                    result["cases"].append(case)
                    write_json(output / "RESULT.partial.json", result, replace=True)
        result["status"] = "COMPLETE"
        result["resources"] = limit.record
        assert en.ex.previous.state_hash(bolt.model) == bolt.initial_hash
        result["bolt_hash_unchanged"] = True
        write_json(output / "RESULT.json", result)
        limit.close("COMPLETE")
        print(json.dumps({"status": "COMPLETE", "path": str((output / "RESULT.json").relative_to(ROOT))}), flush=True)
    except Exception as err:
        result["status"] = "FAILED_PRESERVED"
        result["error"] = repr(err)
        result["traceback"] = traceback.format_exc()
        write_json(output / "RESULT.failed.json", result)
        limit.close("FAILED_PRESERVED", repr(err))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.fixture:
        output = HERE / f"run_{args.tag}"
        output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(fixture(output), ensure_ascii=False), flush=True)
    else:
        run(args.tag, tuple(args.datasets), tuple(args.seeds), tuple(args.methods), int(args.updates), float(args.weight))


if __name__ == "__main__":
    main()
