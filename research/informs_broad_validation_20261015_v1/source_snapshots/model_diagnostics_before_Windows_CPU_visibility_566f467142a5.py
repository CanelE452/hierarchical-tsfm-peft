"""CPU-only encoder dependency and supported-FLOP diagnostics, without real F."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

import data_contract
import evaluation
import fit_base
import networks


def past_perturbation_probe(model, x):
    t = x.shape[1] // 2
    changed = x.clone()
    delta = torch.linspace(.2, 1.2, x.shape[2], dtype=x.dtype)[None, None]
    changed[:, t + 1:] += delta
    with torch.no_grad():
        before, after = model.encode(x), model.encode(changed)
    prefix = (before[:, :t + 1] - after[:, :t + 1]).abs()
    return dict(perturb_rows_start=t + 1, prefix_rows_stop_inclusive=t,
                prefix_max_abs_change=float(prefix.max()), prefix_mean_abs_change=float(prefix.mean()),
                prefix_exactly_unchanged=bool(torch.equal(before[:, :t + 1], after[:, :t + 1])),
                future_target_was_input=False,
                interpretation="Dependence inside supplied complete past window; not future-target leakage")


def synthetic_shift_probe(model, seed=20261015):
    length, horizon, channels = model.length, model.horizon, model.channels
    rng = np.random.default_rng(seed)
    long = rng.normal(size=(1, length + horizon, channels)).astype(np.float32)
    first, second = torch.from_numpy(long[:, :length].copy()), torch.from_numpy(long[:, horizon:].copy())
    left, right = 32, 16
    if length - horizon - left - right <= 0:
        raise ValueError("No internal overlap remains after declared boundary exclusions")
    with torch.no_grad():
        a, b = model.encode(first), model.encode(second)
    aligned_a = a[:, horizon + left:length - right]
    aligned_b = b[:, left:length - horizon - right]
    difference = (aligned_a - aligned_b).abs()
    return dict(seed=seed, synthetic_only=True, real_extra_history_read=False,
                vector_shape=list(long.shape), vector_fingerprint=evaluation.array_fingerprint(long),
                shift=horizon, excluded_left=left, excluded_right=right,
                compared_positions=int(difference.shape[1]), overlap_max_abs_change=float(difference.max()),
                overlap_mean_abs_change=float(difference.mean()), overlap_exactly_unchanged=bool(torch.equal(aligned_a, aligned_b)),
                interpretation="Synthetic overlap alignment, not a real extra-context performance benchmark")


def decoder_prefix_probe(model, x):
    k = max(1, model.latent_k // 2)
    with torch.no_grad():
        z = model.encode(x, k)
        future = z[:, -1:].expand(-1, model.horizon, -1)
        actual = model.decode(x, z, future, k)
        joined = torch.cat((z, future), 1)
        expected = x[:, -1:].expand(-1, model.horizon, -1).clone()
        for index, lag in enumerate(networks.LAGS):
            feature = joined[:, model.length-lag:model.length-lag+model.horizon] - z[:, -1-lag, None]
            expected += feature @ model.D[index, :k]
    return dict(probe_k=k, stored_kmax=model.latent_k, decoder_shape=list(model.D.shape),
                max_abs_difference=float((actual - expected).abs().max()),
                allclose=bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)), atol=1e-5, rtol=1e-4,
                formula="For each lag i, sum features_i @ D[i,:K,:]", real_F_calls=0)


def _profile(operation):
    with torch.no_grad():
        operation()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True, record_shapes=True) as profile:
        with torch.no_grad():
            operation()
    rows = [dict(operator=row.key, calls=int(row.count), supported_flops=int(row.flops or 0)) for row in profile.key_averages()]
    return dict(supported_flops=sum(row["supported_flops"] for row in rows),
                mm_flops=sum(row["supported_flops"] for row in rows if row["operator"] in {"aten::mm", "aten::bmm", "aten::addmm"}),
                zero_or_unaccounted_operators=[row["operator"] for row in rows if not row["supported_flops"]], operators=rows)


def profile_modules(model, x):
    with torch.no_grad():
        z = model.encode(x)
        dummy = z[:, -1:].expand(-1, model.horizon, -1).clone()
    encoder = _profile(lambda:model.encode(x))
    decoder = _profile(lambda:model.decode(x, z, dummy))
    lag_multiplier = 7 if model.encoder_kind != "pointwise" else 1
    b, length, c = x.shape
    k = model.latent_k
    p_params = sum(p.numel() for p in model.front.parameters())
    e_params = model.E.numel() + (model.lag_weights.numel() if hasattr(model, "lag_weights") else 0)
    return dict(profile_device="CPU", batch=b, L=length, H=model.horizon, C=c, K=k, foundation_calls=0,
                encoder_supported_flops=encoder["supported_flops"], decoder_supported_flops=decoder["supported_flops"],
                encoder_mm_flops=encoder["mm_flops"], decoder_mm_flops=decoder["mm_flops"],
                analytical_E_matmul_flops=2 * b * length * c * k * lag_multiplier,
                analytical_D_matmul_flops=2 * b * model.horizon * 7 * k * c,
                P_parameters=p_params, E_parameters=e_params, D_parameters=model.D.numel(),
                total_trainable_parameters=networks.parameter_count(model),
                state_bytes=sum(value.numel() * value.element_size() for value in model.state_dict().values()),
                buffer_bytes=sum(value.numel() * value.element_size() for value in model.buffers()),
                encoder_profile=encoder, decoder_profile=decoder,
                flop_scope="Forward-only supported torch.profiler CPU arithmetic; P/E and D separate; no real F",
                excluded="Unsupported nonlinear/reduction operations, memory movement, backward, F and end-to-end deployment cost",
                dummy_future="Repeated final past latent; shape/compute probe, no forecast accuracy meaning")


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run_diagnostics(snapshot_path, cache_root, asset_root):
    if torch.cuda.is_initialized():
        raise ValueError("Use a fresh CPU process; an existing CUDA context is outside this diagnostic")
    torch.set_num_threads(2)
    data_contract.ROOT, data_contract.CACHE = Path(asset_root).resolve(), Path(cache_root).resolve()
    snapshot_path = Path(snapshot_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    output = snapshot_path.parent
    public = Path(__file__).resolve().parent
    source_names = ("model_diagnostics.py", "fit_base.py", "networks.py", "data_contract.py", "evaluation.py", "training.py")
    sources = {name:data_contract.sha256_file(Path(__file__).with_name(name)) for name in source_names}
    tasks, rows = {}, []
    started = time.time()
    for item in snapshot["rows"]:
        if item["status"] != "INCLUDED_COMPLETE_AT_SNAPSHOT":
            rows.append(dict(run_id=item["run_id"], dataset=item["dataset"], family=item["family"], latent_k=item["latent_k"], status=item["status"]))
            continue
        try:
            receipt_path = public / "runs" / item["run_id"] / "BASE_FIT_RECEIPT.json"
            if data_contract.sha256_file(receipt_path) != item["receipt_sha256"]:
                raise ValueError("Complete-at-start receipt changed")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            root = Path(cache_root).resolve()
            artifact = (root / receipt["artifact_cache_relative_path"]).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise ValueError("Diagnostic artifact absent/outside allowed cache")
            actual_sha = data_contract.sha256_file(artifact)
            if actual_sha != item["artifact_sha256"] or actual_sha != receipt["artifact_sha256"]:
                raise ValueError("Selected checkpoint SHA differs from receipt/start snapshot")
            saved = torch.load(artifact, map_location="cpu", weights_only=False)
            task = tasks.setdefault(item["dataset"], None)
            if task is None:
                task = data_contract.load_task(item["dataset"], cache_dir=cache_root)
                tasks[item["dataset"]] = task
            if saved["contract"]["data_sha"] != task["data_sha256"]:
                raise ValueError("Diagnostic Task data SHA differs from selected model")
            own = saved["normalization"]
            if fit_base.scaler_fingerprint(own["mean"], own["std"]) != saved["contract"]["scaler_sha"]:
                raise ValueError("Diagnostic model normalization SHA differs")
            spec = dict(saved["spec"], device="cpu")
            if spec["family"] not in {"B0", "B2", "LAG_CAUSAL", "LAG_BIDIR"}:
                raise ValueError("Unexpected diagnostic architecture")
            model, rebuilt = fit_base.create_model(spec, task, int(own["fit_end"]), object())
            if not np.array_equal(own["mean"], rebuilt["mean"]) or not np.array_equal(own["std"], rebuilt["std"]):
                raise ValueError("Recreated CPU normalization differs")
            model.load_state_dict(saved["model"], strict=True)
            if any(not torch.equal(value, saved["model"][name]) for name,value in model.state_dict().items()):
                raise ValueError("CPU model did not load selected weights exactly")
            model.eval()
            bounds = data_contract.split_ranges(task, spec.get("fold", "primary"))
            origin = int(data_contract.origins(bounds["V_select"], task["L"], task["H"], task["H"])[0])
            x_np, past_observed = data_contract.history_window(task, origin, own["mean"], own["std"])
            x = torch.from_numpy(x_np[None])
            perturb = past_perturbation_probe(model, x)
            shift = synthetic_shift_probe(model)
            prefix = decoder_prefix_probe(model, x)
            complexity = profile_modules(model, x)
            lag_max = float(model.lag_weights.detach().abs().max()) if hasattr(model, "lag_weights") else None
            expected_prefix_causal = spec["family"] in {"B0", "LAG_CAUSAL"}
            status = "COMPLETE_VALID" if prefix["allclose"] and (not expected_prefix_causal or perturb["prefix_exactly_unchanged"]) else "FAILED_DIAGNOSTIC"
            result = dict(status=status, run_id=item["run_id"], dataset=item["dataset"], family=spec["family"], K=model.latent_k,
                          selected_search_epoch=saved.get("selected_search_epoch"), receipt_sha256=item["receipt_sha256"],
                          checkpoint_sha256=actual_sha, data_sha=task["data_sha256"], scaler_sha=saved["contract"]["scaler_sha"],
                          checkpoint_training_source_sha=saved["contract"]["source_sha"], diagnostic_sources=sources,
                          real_past_origin=origin, real_past_bounds=[origin - task["L"], origin],
                          real_history_fingerprint=evaluation.array_fingerprint(x_np), real_past_mask_fingerprint=evaluation.array_fingerprint(past_observed),
                          real_future_target_read=False, real_F_loaded=False, real_F_calls=0,
                          lag_weights_max_abs=lag_max, prefix_causality_expected=expected_prefix_causal,
                          zero_change_is_not_universal_architecture_proof=True,
                          perturbation=perturb, synthetic_shift=shift, decoder_prefix=prefix, complexity=complexity)
            _write_new(output / f"{item['run_id']}.json", result)
            rows.append(dict(run_id=item["run_id"], dataset=item["dataset"], family=spec["family"], latent_k=model.latent_k, status=status,
                        selected_epoch=saved.get("selected_search_epoch"), lag_weights_max_abs=lag_max,
                        past_prefix_max_abs_change=perturb["prefix_max_abs_change"], synthetic_overlap_max_abs_change=shift["overlap_max_abs_change"],
                        decoder_prefix_allclose=prefix["allclose"], decoder_prefix_max_abs_change=prefix["max_abs_difference"],
                        P_parameters=complexity["P_parameters"], E_parameters=complexity["E_parameters"], D_parameters=complexity["D_parameters"],
                        total_trainable_parameters=complexity["total_trainable_parameters"], encoder_supported_flops=complexity["encoder_supported_flops"],
                        decoder_supported_flops=complexity["decoder_supported_flops"], encoder_mm_flops=complexity["encoder_mm_flops"],
                        decoder_mm_flops=complexity["decoder_mm_flops"], analytical_E_matmul_flops=complexity["analytical_E_matmul_flops"],
                        analytical_D_matmul_flops=complexity["analytical_D_matmul_flops"], checkpoint_sha256=actual_sha, data_sha=task["data_sha256"]))
        except Exception as error:
            row = dict(run_id=item["run_id"], dataset=item["dataset"], family=item["family"], latent_k=item["latent_k"], status="FAILED_DIAGNOSTIC", reason=repr(error))
            rows.append(row)
            _write_new(output / f"FAILURE_{item['run_id']}.json", row)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "ENCODER_DIAGNOSTICS.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    receipt = dict(status="COMPLETE_VALID" if not any(row["status"] == "FAILED_DIAGNOSTIC" for row in rows) else "PARTIAL_DIAGNOSTIC_FAILURE",
                   completed=sum(row["status"] == "COMPLETE_VALID" for row in rows), skipped=sum(row["status"].startswith("SKIPPED") for row in rows),
                   failed=sum(row["status"] == "FAILED_DIAGNOSTIC" for row in rows), sources=sources,
                   input_snapshot_sha256=data_contract.sha256_file(snapshot_path), started_unix=started, finished_unix=time.time(),
                   CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), cuda_initialized=torch.cuda.is_initialized(),
                   torch_CPU_threads=torch.get_num_threads(), real_F_calls=0, training_steps=0, future_targets_read=False,
                   registry_modified=False, scores_modified=False, interpretation="Stage/compute verification, not training rerun or forecast gain")
    _write_new(output / "DIAGNOSTIC_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--asset-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run_diagnostics(args.snapshot, args.cache_root, args.asset_root), ensure_ascii=False, indent=2))
