"""E4 protocol: pretraining family lock and real K-row shared-prefix inference.

No data/F loading, process launching, registry mutation or T_final interface.
Root owns scheduling and actual GPU execution.
"""
import copy
import csv
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

import data_contract
import evaluation
import fit_base
import prediction_cache
import training


def k_grid(task):
    c = len(task["columns"])
    values = [math.ceil(c / 4), math.ceil(c / 2), c]
    if len(set(values)) != 3:
        raise ValueError("E4 requires three distinct quarter/half/full prefixes")
    return values


def _folder(task, public_root, fold):
    return Path(public_root) / "ordered" / f"{fold}_{task['name']}"


def _sources():
    return {name: data_contract.sha256_file(Path(__file__).with_name(name)) for name in
            ("ordered_pilot.py", "prediction_cache.py", "fit_base.py", "training.py", "networks.py", "data_contract.py", "evaluation.py", "gpu_runtime.py")}


def _run_id(cache):
    return str(cache["metadata"]["run_id"])


def _candidate_k(cache):
    if "inference_latent_k" in cache["metadata"]:
        return int(cache["metadata"]["inference_latent_k"])
    match = re.search(r"_k(\d+)_s(\d+)$", _run_id(cache))
    if not match or int(match.group(2)) != 92601:
        raise ValueError("Family selection requires registered screen-seed K IDs")
    return int(match.group(1))


def lock_family(task, v_caches, public_root, fold="primary"):
    """Lowest mean V across q/half/C, tie B0, before any E4 training."""
    grid = k_grid(task)
    lock_path = _folder(task, public_root, fold) / "FAMILY_LOCK.json"
    if not lock_path.exists():
        for label in ("ORDERED_MULTI_K", "MAXK_PCA_TRAINED"):
            if (Path(public_root) / "runs" / f"{fold}_{task['name']}_E4_{label}_k{len(task['columns'])}_s92601").exists():
                raise ValueError("Cannot create a pretraining family lock after an E4 fit started")
    bounds = data_contract.split_ranges(task, fold)
    origins = data_contract.origins(bounds["V_select"], task["L"], task["H"], task["H"])
    mean, std, _ = data_contract.fit_statistics(task, bounds["A_stop"][1])
    scaler_sha = fit_base.scaler_fingerprint(mean, std)
    truths, masks = zip(*(data_contract.target_window(task, int(o), mean, std) for o in origins))
    truths, masks = np.stack(truths), np.stack(masks)
    chosen = {}
    for cache in v_caches:
        meta = cache["metadata"]
        family = meta.get("family")
        if family not in {"B0", "B2"}:
            continue
        if "_E1_" not in _run_id(cache):
            raise ValueError("Pretraining family selection requires independent E1 results")
        k = _candidate_k(cache)
        if k not in grid:
            continue
        if (family, k) in chosen:
            raise ValueError("Duplicate independent family/K candidate")
        if meta.get("split") != "V_select" or meta.get("fold", "primary") != fold:
            raise ValueError("Family candidate time fold/split differs")
        if meta.get("data_sha256") != task["data_sha256"] or meta.get("scaler_sha") != scaler_sha:
            raise ValueError("Family candidate data/scaler differs")
        if not meta.get("source_sha") or not meta.get("model_sha"):
            raise ValueError("Family candidate lacks source/model identity")
        for name, expected in (("origins", origins), ("channel_ids", np.asarray(task["columns"])),
                               ("target_normalized", truths), ("observed", masks)):
            if not np.array_equal(cache[name], expected):
                raise ValueError("Family candidate differs from complete registered Task V cohort: " + name)
        prediction = np.asarray(cache["pred_normalized"], np.float64)
        if prediction.shape != truths.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid independent candidate forecast")
        counts = masks.sum((0, 1))
        active = np.flatnonzero(counts)
        if not len(active):
            raise ValueError("Family V cohort has no observed target")
        mse = math.fsum(float(np.square((prediction[:, :, c] - truths[:, :, c])[masks[:, :, c]]).sum()) / int(counts[c]) for c in active) / len(active)
        if not math.isclose(float(cache["score"]["mse"]), mse, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("Family candidate V score differs from cached predictions")
        chosen[family, k] = dict(run_id=_run_id(cache), family=family, latent_k=k, mse=mse,
                                model_sha=meta["model_sha"], source_sha=meta["source_sha"],
                                prediction_sha=evaluation.array_fingerprint(cache["pred_normalized"]),
                                archive_sha256=cache.get("manifest", {}).get("archive_sha256"))
    if set(chosen) != {(family, k) for family in ("B0", "B2") for k in grid}:
        raise ValueError("Before E4 training require both B0/B2 and every q/half/C independent V result")
    means = {family: math.fsum(chosen[family, k]["mse"] for k in grid) / len(grid) for family in ("B0", "B2")}
    family = "B0" if means["B0"] <= means["B2"] else "B2"
    lock = dict(locked=True, chosen_before_E4_training=True, dataset=task["name"], fold=fold,
                data_sha=task["data_sha256"], scaler_sha=scaler_sha, k_grid=grid, seed=92601,
                chosen_family=family, mean_V_mse=means, rule="minimum equal mean V MSE at q/half/C; tie B0",
                candidates=[chosen[key] for key in sorted(chosen)], source_files=_sources())
    prediction_cache._json_new_or_same(lock_path, lock)
    return lock


def make_specs(task, lock, *, epochs=None, device="cuda"):
    if lock.get("locked") is not True or lock.get("chosen_family") not in {"B0", "B2"} or lock.get("data_sha") != task["data_sha256"] or lock["k_grid"] != k_grid(task):
        raise ValueError("Invalid pretraining family lock")
    fold, c, seed = lock["fold"], len(task["columns"]), lock["seed"]
    common = dict(family=lock["chosen_family"], latent_k=c, seed=seed, epochs=list(epochs or [.25, 1, 3]),
                  lr=.001, fold=fold, microbatch=4, effective_batch=16, device=device,
                  family_lock_sha=training.fingerprint(lock))
    return dict(shared=dict(common, run_id=f"{fold}_{task['name']}_E4_ORDERED_MULTI_K_k{c}_s{seed}", k_grid=lock["k_grid"]),
                max_only=dict(common, run_id=f"{fold}_{task['name']}_E4_MAXK_PCA_TRAINED_k{c}_s{seed}", k_grid=[c]))


def prefix_prediction(model, x, foundation, latent_k):
    """Model's D remains [7,Kmax,C]; encode/F/decode all use requested K."""
    return model(x, foundation, latent_k=int(latent_k))


def _nested_artifact(task, cache_root, fold):
    folder = Path(cache_root) / "ordered" / f"{fold}_{task['name']}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "nested_pca.npz"
    end = data_contract.split_ranges(task, fold)["A_stop"][1]
    mean, std, _ = data_contract.fit_statistics(task, end)
    if path.exists():
        contract_path = folder / "nested_pca_contract.json"
        if not contract_path.exists():
            raise ValueError("Nested PCA artifact has no immutable contract")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if data_contract.sha256_file(path) != contract["artifact_sha256"] or contract["source_sha"] != training.fingerprint(_sources()):
            raise ValueError("Nested PCA source/artifact fingerprint differs")
        with np.load(path, allow_pickle=False) as handle:
            value = {key: handle[key] for key in handle.files}
        if str(value["data_sha"].item()) != task["data_sha256"] or int(value["fit_end"].item()) != end:
            raise ValueError("Existing nested PCA data/time contract differs")
        if not np.array_equal(value["mean"], mean) or not np.array_equal(value["std"], std):
            raise ValueError("Existing nested PCA normalization differs")
    else:
        pca = data_contract.initialize_pca(task, end, len(task["columns"]), mean, std)
        value = dict(basis=pca["E"], center=pca["pca_center"], mean=mean, std=std,
                     fit_end=np.asarray(end), data_sha=np.asarray(task["data_sha256"]),
                     source_sha=np.asarray(training.fingerprint(_sources())))
        with path.open("xb") as handle:
            np.savez_compressed(handle, **value)
        fit_base._json_new(folder / "nested_pca_contract.json", dict(artifact_sha256=data_contract.sha256_file(path),
                          source_sha=training.fingerprint(_sources()), data_sha=task["data_sha256"], fit_end=end))
    if value["basis"].shape != (len(task["columns"]), len(task["columns"])) or value["center"].shape != (len(task["columns"]),):
        raise ValueError("Nested PCA basis/center axes differ")
    if not np.isfinite(value["basis"]).all() or not np.isfinite(value["center"]).all():
        raise ValueError("Nonfinite nested PCA")
    return value, path


def predict_prefix(alias_run_id, task, latent_k, cache_root, public_root, foundation, *,
                   parent_receipt=None, variant="shared", fold="primary", deadline=None):
    """Frozen V-only inference; alias does not alter the parent artifact ID."""
    alias_run_id = str(alias_run_id)
    if Path(alias_run_id).name != alias_run_id or alias_run_id in {"", ".", ".."}:
        raise ValueError("Safe prefix alias required")
    if variant not in {"shared", "max_only", "posthoc", "nested_pca"} or latent_k not in k_grid(task):
        raise ValueError("Registered E4 variant and q/half/C K required")
    bounds = data_contract.split_ranges(task, fold)
    c = len(task["columns"])
    foundation_sha = fit_base._model_sha(foundation)
    if variant == "nested_pca":
        artifact, path = _nested_artifact(task, cache_root, fold)
        own = dict(mean=artifact["mean"], std=artifact["std"], fit_end=int(artifact["fit_end"].item()))
        artifact_sha = data_contract.sha256_file(path)
        parent_run_id, family, training_source = "nested_pca", "NESTED_PCA", str(artifact["source_sha"].item())
        model_sha = training.fingerprint(dict(full_basis=evaluation.array_fingerprint(artifact["basis"]),
                                             center=evaluation.array_fingerprint(artifact["center"]),
                                             artifact_sha=artifact_sha, foundation_sha=foundation_sha))
        device = getattr(foundation, "device", "cpu")
        saved, spec = None, None
    else:
        if parent_receipt is None:
            raise ValueError("Trained prefix needs its exact immutable parent receipt")
        parent_run_id = parent_receipt["run_id"]
        saved, _ = prediction_cache._load_parent(parent_run_id, task, cache_root, public_root, parent_receipt, fold)
        own, spec = saved["normalization"], copy.deepcopy(saved["spec"])
        if spec["family"] not in {"B0", "B2"} or spec["latent_k"] != c:
            raise ValueError("E4 parent must be full-C B0/B2")
        trained_grid = spec.get("k_grid")
        if variant == "shared" and trained_grid != k_grid(task):
            raise ValueError("Shared prefix requires its genuine multi-K trained parent")
        if variant in {"posthoc", "max_only"} and trained_grid != [c]:
            raise ValueError("Posthoc prefix requires matched full-PCA max-only parent")
        if variant == "max_only" and latent_k != c:
            raise ValueError("max_only score is full C; use posthoc for smaller untrained prefixes")
        if int(own.get("train_end", own["fit_end"])) > bounds["M_cal"][0]:
            raise ValueError("E4 parent trained beyond A60")
        if foundation_sha != saved["contract"]["model_sha"]:
            raise ValueError("E4 foundation identity differs from trained parent")
        artifact_sha = parent_receipt["artifact_sha256"]
        family, training_source = spec["family"], saved["contract"]["source_sha"]
        model_sha = training.fingerprint(dict(artifact_sha=artifact_sha, state_sha=saved["model_state_sha"]))
        device = spec.get("device", getattr(foundation, "device", "cpu"))
        # A deployment-only allowed-prefix registry, never a weight/train edit.
        spec["k_grid"] = sorted(set(trained_grid + [int(latent_k)]))
    if foundation is None or not hasattr(foundation, "streams"):
        raise ValueError("E4 requires a counted caller-owned frozen row foundation")
    mean, std, _ = data_contract.fit_statistics(task, bounds["A_stop"][1])
    if not np.array_equal(own["mean"], mean) or not np.array_equal(own["std"], std):
        raise ValueError("E4 comparisons require identical canonical A60 input/scoring normalization")
    origins = data_contract.origins(bounds["V_select"], task["L"], task["H"], task["H"])
    truths, masks = zip(*(data_contract.target_window(task, int(origin), mean, std) for origin in origins))
    arrays = dict(target_normalized=np.stack(truths), observed=np.stack(masks), origins=origins,
                  channel_ids=np.asarray(task["columns"], dtype=str))
    sources = _sources()
    identity = dict(run_id=alias_run_id, split="V_select", cohort=f"{task['name']}:{fold}:V_select:stride{task['H']}",
                    source_sha=training.fingerprint(dict(sources=sources, training_source_sha=training_source)),
                    config_sha=training.fingerprint(dict(parent_run_id=parent_run_id, variant=variant, latent_k=latent_k, fold=fold)),
                    data_sha=task["data_sha256"], model_sha=model_sha, scaler_sha=fit_base.scaler_fingerprint(mean, std))
    expected_streams = len(origins) * latent_k
    reference = dict(identity, expected_F_streams=expected_streams,
                     arrays={name: evaluation.array_fingerprint(value) for name, value in arrays.items()})
    public = Path(public_root) / "ordered_predictions" / alias_run_id
    public.mkdir(parents=True, exist_ok=True)
    private = Path(cache_root) / "ordered_predictions" / alias_run_id
    private.mkdir(parents=True, exist_ok=True)
    cache_path, manifest_path = private / "V_select.npz", public / "PREDICTION_MANIFEST.json"
    prediction_cache._json_new_or_same(public / "REFERENCE_CONTRACT.json", reference)
    if cache_path.exists():
        if not manifest_path.exists():
            raise ValueError("Existing prefix cache incomplete; preserve it")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items()) or data_contract.sha256_file(cache_path) != manifest["archive_sha256"]:
            raise ValueError("Existing prefix cache contract differs")
        with np.load(cache_path, allow_pickle=False) as archive:
            cached = {key: archive[key] for key in archive.files}
        if set(cached) != set(prediction_cache.KEYS):
            raise ValueError("Prefix cache keys differ")
        for name, value in cached.items():
            if evaluation.array_fingerprint(value) != manifest["arrays"][name]:
                raise ValueError("Prefix cache fingerprint differs")
            if name != "pred_normalized" and evaluation.array_fingerprint(value) != reference["arrays"][name]:
                raise ValueError("Prefix cache target cohort differs")
        return dict(cached, manifest=manifest, reference_contract=reference, metadata=manifest["metadata"],
                    score=manifest["score"], cost_record=manifest["cost_record"],
                    public_stats_relative_path=manifest["public_stats_relative_path"])
    prefix = dict(task, physical=task["physical"][:int(origins[-1]) + task["H"]],
                  observed=task["observed"][:int(origins[-1]) + task["H"]])
    windows = training.FastWindows(prefix, mean, std, device)
    model = None
    before = int(foundation.streams)
    predictions = []
    try:
        if variant == "nested_pca":
            model = fit_base._FixedPCA(artifact["center"], artifact["basis"]).to(device)
        else:
            model, _ = fit_base.create_model(spec, task, int(own["fit_end"]), foundation)
            model.load_state_dict(saved["model"], strict=True)
        model.eval()
        with torch.no_grad():
            for lo in range(0, len(origins), 4):
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Ordered prediction deadline reached before microbatch")
                x, _, _ = windows.get(origins[lo:lo + 4])
                y = prefix_prediction(model, x, foundation, latent_k)
                if tuple(y.shape) != (len(x), task["H"], c) or not torch.isfinite(y).all():
                    raise ValueError("Invalid E4 original-C forecast")
                predictions.append(y.cpu().numpy().astype(np.float32))
    finally:
        if model is not None and callable(getattr(model, "close", None)):
            model.close()
        if callable(getattr(foundation, "assert_frozen_unchanged", None)):
            foundation.assert_frozen_unchanged()
    actual_streams = int(foundation.streams) - before
    if actual_streams != expected_streams:
        raise ValueError("Prefix F row count differs: max-K execution cannot be labeled small K")
    if _sources() != sources:
        raise ValueError("E4 source changed during inference")
    arrays["pred_normalized"] = np.concatenate(predictions)
    with cache_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    score, statistics = evaluation.error_statistics(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"],
                   origins, alias_run_id, "V_select", identity["cohort"], arrays["channel_ids"])
    stats_path = public / "PUBLIC_ERROR_STATS.csv.gz"
    prediction_cache._stats_file(stats_path, statistics)
    with (public / "SCORES.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score)); writer.writeheader(); writer.writerow(score)
    metadata = dict(identity, data_sha256=task["data_sha256"], fold=fold, parent_run_id=parent_run_id,
                    parent_artifact_sha256=artifact_sha, inference_latent_k=latent_k, stored_latent_k=c,
                    variant=variant, family=family, parent_fit_end=int(own.get("train_end", own["fit_end"])),
                    foundation_sha=foundation_sha,
                    actual_F_streams=actual_streams, prefix_decoder="D[lag,:K,:], then flatten", T_final_accessed=False,
                    posthoc_prefix_was_trained=variant == "shared" or (variant in {"posthoc", "max_only"} and latent_k == c))
    cost = dict(run_id=alias_run_id, batch=len(origins), latent_k=latent_k,
                actual_F_streams=actual_streams, expected_F_streams=expected_streams, F_calls=1)
    relative_stats = stats_path.relative_to(Path(public_root)).as_posix()
    manifest = dict(identity, cache_relative_path=cache_path.relative_to(Path(cache_root)).as_posix(),
                    archive_sha256=data_contract.sha256_file(cache_path), arrays={name:evaluation.array_fingerprint(value) for name,value in arrays.items()},
                    score=score, cost_record=cost, metadata=metadata, public_stats_relative_path=relative_stats)
    prediction_cache._json_new_or_same(manifest_path, manifest)
    return dict(arrays, metadata=metadata, manifest=manifest, reference_contract=reference, score=score,
                cost_record=cost, public_stats_relative_path=relative_stats)


def run_ordered_pilot(task, v_caches, cache_root, public_root, foundation, deadline, *, epochs=None, fold="primary", device="cuda"):
    """Root-callable full E4 bundle; root applies returned registry actions."""
    lock = lock_family(task, v_caches, public_root, fold)
    specs = make_specs(task, lock, epochs=epochs, device=device)
    receipts = {}
    reused_fit_runs = []
    for label, spec in specs.items():
        if _sources() != lock["source_files"]:
            raise ValueError("E4 source changed after the pretraining lock")
        receipts[label] = fit_base.fit_base(spec, task, cache_root, public_root, foundation, deadline)
        if receipts[label].get("reused_complete_artifact"):
            reused_fit_runs.append(spec["run_id"])
        # Keep the immutable scientific receipt independent of this call's reuse.
        receipts[label] = dict(receipts[label], reused_complete_artifact=False)
        if receipts[label]["status"] != "COMPLETE_VALID":
            return dict(status=receipts[label]["status"], family_lock=lock, receipts=receipts,
                        pending_phase=label, T_final_accessed=False)
    outputs, actions = {}, []
    seed, c = lock["seed"], len(task["columns"])
    for k in lock["k_grid"]:
        shared_id = specs["shared"]["run_id"] if k == c else specs["shared"]["run_id"] + f"__prefix_k{k}"
        post_id = f"{fold}_{task['name']}_E4_POSTHOC_PREFIX_k{k}_s{seed}"
        nested_id = f"{fold}_{task['name']}_E4_NESTED_PCA_k{k}_s{seed}"
        for alias, variant, parent in ((shared_id, "shared", receipts["shared"]), (post_id, "posthoc", receipts["max_only"]),
                                        (nested_id, "nested_pca", None)):
            outputs[alias] = predict_prefix(alias, task, k, cache_root, public_root, foundation,
                    parent_receipt=parent, variant=variant, fold=fold, deadline=deadline)
            actions.append(dict(experiment_id=alias, action="ADD_DERIVED_SHARED_PREFIX_ROW" if "__prefix_" in alias else "UPDATE_EXISTING_E4_ROW",
                                family=variant, latent_k=k, execution_status="COMPLETE_VALID", prediction_rescoring="PENDING_INDEPENDENT_AUDIT"))
    max_id = specs["max_only"]["run_id"]
    outputs[max_id] = predict_prefix(max_id, task, c, cache_root, public_root, foundation,
                                    parent_receipt=receipts["max_only"], variant="max_only", fold=fold, deadline=deadline)
    actions.append(dict(experiment_id=max_id, action="UPDATE_EXISTING_E4_ROW", family="max_only", latent_k=c,
                        execution_status="COMPLETE_VALID", prediction_rescoring="PENDING_INDEPENDENT_AUDIT"))
    independent = {_run_id(cache):cache for cache in v_caches if cache["metadata"].get("family") == lock["chosen_family"] and _candidate_k(cache) in lock["k_grid"]}
    science_source = _sources()
    receipt = dict(status="COMPLETE_VALID", dataset=task["name"], fold=fold, chosen_family=lock["chosen_family"],
                   k_grid=lock["k_grid"], requested_epochs=list(epochs or [.25, 1, 3]),
                   family_lock_sha=training.fingerprint(lock), training_receipts=receipts,
                   prediction_aliases=list(outputs), reused_independent_run_ids=list(independent),
                   registry_actions=actions, source_files=science_source,
                   comparison_budget=dict(search_steps_equal=receipts["shared"]["search_updates"] == receipts["max_only"]["search_updates"],
                       search_origin_exposures_equal=receipts["shared"]["search_exposures"] == receipts["max_only"]["search_exposures"],
                       equal_GPU_time_or_equal_perK_exposure_not_claimed=True),
                   training_status="LIMITED_SCREEN" if max(epochs or [.25, 1, 3]) < 3 else "SCREEN_NOT_CONVERGENCE_PROVEN",
                   T_final_accessed=False)
    prediction_cache._json_new_or_same(_folder(task, public_root, fold) / "ORDERED_PILOT_RECEIPT.json", receipt)
    return dict(receipt, prefix_predictions=outputs, independent_predictions=independent,
                operational_reused_fit_runs=reused_fit_runs)
