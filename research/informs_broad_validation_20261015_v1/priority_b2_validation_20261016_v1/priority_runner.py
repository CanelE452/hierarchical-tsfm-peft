"""Explicit Jena no-Bolt priority execution; default opens no input or model."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

NAME = "priority_b2_validation_20261016_v1"
L, H, C, K, SEED = 512, 48, 21, 6, 92601
ROLES = ("R5a", "R5b")
MODES = ("fit", "rescore", "replay", "cost")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Symlink in explicit input/output path")
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Explicit path escapes its root")
    return path


def new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv_new(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Priority operation deadline reached")


def state_sha(model_state, data):
    return fingerprint({name: data.array_sha256(value.detach().cpu().numpy()) for name, value in model_state.items()})


def forward_wrapper(model, torch):
    class OriginalStageWrapper(torch.nn.Module):
        def __init__(self, direct):
            super().__init__()
            self.direct = direct

        def forward(self, x, foundation=None, latent_k=None):
            if foundation is not None or latent_k not in (None, K):
                raise ValueError("No foundation and exact K6 required")
            return self.direct(x)

    return OriginalStageWrapper(model)


def stage_normalization(task, fit_end, data, fit):
    mean, std, count = data.fit_statistics(task, fit_end)
    pca = data.initialize_pca(task, fit_end, K, mean, std)
    return dict(mean=mean, std=std, observed_counts=count, fit_end=fit_end, train_end=fit_end,
                basis_initialization="A_fitted_centered_PCA", basis=pca["E"],
                pca_center=pca["pca_center"], explained_variance=pca["explained_variance"])


def original_adapter(bundle, networks, torch):
    spec, norm = bundle["spec"], bundle["normalization"]
    if spec["family"] != "B2" or spec["latent_k"] != K or spec["seed"] != SEED:
        raise ValueError("Pinned R4 is not the exact Jena B2/K6/seed parent")
    adapter = networks.ForecastAdapter("B2", L, H, C, K, norm["pca_center"], norm["basis"], "pointwise", None)
    adapter.load_state_dict(bundle["model"], strict=True)
    return adapter.to("cpu")


def build_model(role, norm, r4_bundle, direct, networks, torch):
    if role == "R5a":
        model = direct.fixed_model(original_adapter(r4_bundle, networks, torch))
    elif role == "R5b":
        model = direct.fresh_model(norm["pca_center"], norm["basis"], seed=SEED)
    else:
        raise ValueError("Only preregistered R5a/R5b roles")
    return forward_wrapper(model, torch)


def verify_fixed_adapter(model, r4_bundle, role, torch):
    adapter = model.direct.adapter
    if role == "R5a":
        actual = adapter.state_dict()
        if set(actual) != set(r4_bundle["model"]) or any(not torch.equal(actual[key].cpu(), r4_bundle["model"][key].cpu()) for key in actual):
            raise ValueError("Fixed R5a P/E/D changed from exact R4 artifact")
        if any(p.requires_grad or p.grad is not None for p in adapter.parameters()):
            raise ValueError("Fixed R5a adapter acquired trainability/gradient")
    head = model.direct.head
    if tuple(head.weight.shape) != (H, L) or head.bias.numel() != H:
        raise ValueError("Full-context shared Linear(512,48) head required")
    return dict(full_latent_history=L, latent_k=K, head_parameters=H * L + H,
                exact_R4_PED_frozen=role == "R5a", same_architecture_different_weights=role == "R5b",
                actual_F_streams=0, foundation_instances_created=0)


def save_bundle(path, bundle, torch):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp_" + str(os.getpid()))
    if path.exists() or temporary.exists():
        raise FileExistsError("Preserve existing private artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        torch.save(bundle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError("Artifact appeared during write")
    temporary.replace(path)


def write_stage(public_stage, stage):
    new(public_stage / "RECEIPT.json", stage["summary"])
    csv_new(public_stage / "CURVE.csv", stage["curves"])


def run_fit(role, run_id, task, bounds, r4_bundle, context, public_run, private_run, deadline):
    torch, data, fit, direct, networks = (context[name] for name in ("torch", "data", "fit", "direct", "networks"))
    spec = dict(run_id=run_id, dataset="jena", family="PRIORITY_NO_BOLT", latent_k=K, seed=SEED,
                fold="primary", lr=.001, microbatch=4, effective_batch=16, device="cpu",
                role=role, head="shared_Linear_512_to_48_persistence_init", priority_plan_sha256=context["plan_sha"],
                new_source_sha256=context["new_sources"], epochs=[3.] if role == "R5a" else [.25, 1., 3.])
    source_sha, source_files = fit._source_contract()
    base_contract = dict(source_sha=source_sha,
        config_sha=fingerprint(dict(spec=spec, axes=dict(L=L, H=H, C=C, K=K, columns=task["columns"]))),
        data_sha=task["data_sha256"], model_sha="NO_FOUNDATION")
    new(public_run / "SOURCE_CONTRACT.json", dict(**base_contract, files=source_files, spec=spec))
    if role == "R5a":
        norm = r4_bundle["normalization"]
        if norm["train_end"] != bounds["A_stop"][1]:
            raise ValueError("R5a exact representation must use original A60 scaler")
        model = build_model(role, norm, r4_bundle, direct, networks, torch)
        verify_fixed_adapter(model, r4_bundle, role, torch)
        stage = fit._stage_body(spec, task, bounds["A_stop"][1], None, [3.], private_run / "fixed_refit",
                                public_run / "fixed_refit", None, deadline, base_contract, model, norm)
        if stage["summary"]["status"] != "SEGMENT_COMPLETE":
            return dict(status="BUDGET_LIMIT_PRESERVED", phase="fixed_refit", summary=stage["summary"])
        write_stage(public_run / "fixed_refit", stage)
        selected_epoch, search = 3., None
    else:
        norm = stage_normalization(task, bounds["A_fit"][1], data, fit)
        model = build_model(role, norm, r4_bundle, direct, networks, torch)
        search = fit._stage_body(spec, task, bounds["A_fit"][1], bounds["A_stop"], [.25, 1., 3.],
                                 private_run / "search", public_run / "search", None, deadline, base_contract, model, norm)
        if search["summary"]["status"] != "SEGMENT_COMPLETE":
            return dict(status="BUDGET_LIMIT_PRESERVED", phase="search", summary=search["summary"])
        write_stage(public_run / "search", search)
        selected_epoch = search["summary"]["selected_epoch"]
        norm = stage_normalization(task, bounds["A_stop"][1], data, fit)
        model = build_model(role, norm, r4_bundle, direct, networks, torch)
        stage = fit._stage_body(spec, task, bounds["A_stop"][1], None, [selected_epoch] if selected_epoch else [],
                                private_run / "refit", public_run / "refit", None, deadline, base_contract, model, norm)
        if stage["summary"]["status"] != "SEGMENT_COMPLETE":
            return dict(status="BUDGET_LIMIT_PRESERVED", phase="refit", summary=stage["summary"])
        write_stage(public_run / "refit", stage)
    model.load_state_dict(stage["model"], strict=True)
    diagnosis = verify_fixed_adapter(model, r4_bundle, role, torch)
    bundle = dict(model=stage["model"], normalization=norm, spec=spec, contract=stage["summary"]["contract"],
                  model_state_sha=state_sha(stage["model"], data), selected_epoch=selected_epoch,
                  search_summary=None if search is None else search["summary"], refit_summary=stage["summary"],
                  R4_artifact_sha256=context["r4_artifact_sha"], plan_sha256=context["plan_sha"])
    artifact = private_run / "selected_bundle.pt"
    save_bundle(artifact, bundle, torch)
    receipt = dict(status="COMPLETE_PRIORITY_NO_BOLT_FIT", role=role, run_id=run_id, selected_epoch=selected_epoch,
                   artifact_sha256=sha(artifact), model_state_sha=bundle["model_state_sha"],
                   scaler_sha=fit.scaler_fingerprint(norm["mean"], norm["std"]),
                   search_summary=bundle["search_summary"], refit_summary=bundle["refit_summary"], **diagnosis,
                   A_stop_unseen_head_selection_claimed=False, new_M_target_training=False,
                   training_convergence_claimed=False, protected_T_target_operations=0)
    new(public_run / "BASE_FIT_RECEIPT.json", receipt)
    return dict(receipt=receipt, bundle=bundle, model=model, artifact=artifact)


def cohort_arrays(task, bounds, mean, std, data, np):
    origins = data.origins(bounds["V_select"], L, H, H)
    if len(origins) != 43 or int(origins[-1]) + H > bounds["V_select"][1]:
        raise ValueError("Original complete V43 cohort required")
    truths = [data.target_window(task, int(origin), mean, std) for origin in origins]
    return dict(target_normalized=np.stack([value[0] for value in truths]),
                observed=np.stack([value[1] for value in truths]), origins=origins,
                channel_ids=np.asarray(task["columns"], dtype=str))


def validate_cohort(arrays, reference, evaluation):
    for name in ("target_normalized", "observed", "origins", "channel_ids"):
        if evaluation.array_fingerprint(arrays[name]) != reference["arrays"][name]:
            raise ValueError("Priority V differs from pinned original R4 cohort: " + name)


def predict_normalized(model, task, mean, std, origins, training, torch, np, deadline):
    model.eval()
    windows = training.FastWindows(task, mean, std, "cpu")
    output = []
    with torch.no_grad():
        for offset in range(0, len(origins), 4):
            check_deadline(deadline)
            x, _, _ = windows.get(origins[offset:offset + 4])
            prediction = model(x, None, K)
            if tuple(prediction.shape) != (len(x), H, C) or not torch.isfinite(prediction).all():
                raise ValueError("No-Bolt original-C forecast invalid")
            output.append(prediction.cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def save_prediction(run_id, role, fitted, task, bounds, reference, context, public_folder, private_folder, deadline):
    data, evaluation, np = (context[name] for name in ("data", "evaluation", "np"))
    norm = fitted["bundle"]["normalization"]
    arrays = cohort_arrays(task, bounds, norm["mean"], norm["std"], data, np)
    validate_cohort(arrays, reference, evaluation)
    identity = dict(run_id=run_id, split="V_select", cohort="jena:primary:V_select:stride48",
                    data_sha=task["data_sha256"], scaler_sha=context["fit"].scaler_fingerprint(norm["mean"], norm["std"]),
                    source_sha=fingerprint(context["new_sources"]), config_sha=fingerprint(fitted["bundle"]["spec"]),
                    model_sha=fingerprint(dict(artifact_sha=fitted["receipt"]["artifact_sha256"], model_state_sha=fitted["bundle"]["model_state_sha"])))
    public_folder.mkdir(parents=True, exist_ok=False)
    private_folder.mkdir(parents=True, exist_ok=False)
    new(public_folder / "REFERENCE_CONTRACT.json", dict(**identity,
        arrays={name: evaluation.array_fingerprint(value) for name, value in arrays.items()}, expected_F_streams=0))
    arrays["pred_normalized"] = predict_normalized(fitted["model"], task, norm["mean"], norm["std"], arrays["origins"],
                                                context["training"], context["torch"], np, deadline)
    path = private_folder / "V_select.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    score, statistics = evaluation.error_statistics(arrays["pred_normalized"], arrays["target_normalized"], arrays["observed"],
        arrays["origins"], run_id, "V_select", identity["cohort"], arrays["channel_ids"])
    fields = list(statistics[0])
    with gzip.open(public_folder / "PUBLIC_ERROR_STATS.csv.gz", "xt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(statistics)
    csv_new(public_folder / "SCORES.csv", [score])
    manifest = dict(**identity, archive_sha256=sha(path), arrays={name: evaluation.array_fingerprint(value) for name, value in arrays.items()},
        score=score, role=role, artifact_sha256=fitted["receipt"]["artifact_sha256"], device="cpu", actual_F_streams=0,
        original_full_L_history_used=True, same_lag_D_current_anchor=True, post_T_development=True,
        new_T_target_operations=0, independent_generalization_claimed=False)
    new(public_folder / "PREDICTION_MANIFEST.json", manifest)
    return manifest


def independent_rescore(arrays, manifest, statistics, score_row, np):
    prediction = np.asarray(arrays["pred_normalized"], np.float64)
    target = np.asarray(arrays["target_normalized"], np.float64)
    mask = arrays["observed"]
    if prediction.shape != target.shape or prediction.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Saved prediction/target/mask axes differ")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite saved predictions/observed targets")
    counts = mask.sum((0, 1))
    per_channel = []
    for channel in range(prediction.shape[2]):
        errors = (prediction[:, :, channel] - target[:, :, channel])[mask[:, :, channel]].tolist()
        if counts[channel]:
            per_channel.append((math.fsum(value * value for value in errors) / int(counts[channel]),
                                math.fsum(abs(value) for value in errors) / int(counts[channel])))
    metrics = dict(mse=math.fsum(value[0] for value in per_channel) / len(per_channel),
                   mae=math.fsum(value[1] for value in per_channel) / len(per_channel), n_observed=int(counts.sum()))
    expected = {}
    edges = sorted({0, math.ceil(prediction.shape[1] / 4), math.ceil(prediction.shape[1] / 2), prediction.shape[1]})
    for index, origin in enumerate(arrays["origins"]):
        for channel, channel_id in enumerate(arrays["channel_ids"]):
            for start, stop in zip(edges[:-1], edges[1:]):
                errors = (prediction[index, start:stop, channel] - target[index, start:stop, channel])[mask[index, start:stop, channel]].tolist()
                expected[int(origin), str(channel_id), start, stop] = (math.fsum(value * value for value in errors), math.fsum(abs(value) for value in errors), len(errors))
    seen = set()
    for row in statistics:
        if any(row[name] != manifest[name] for name in ("run_id", "split", "cohort")):
            raise ValueError("Public statistic identity differs")
        key = int(row["origin"]), row["channel"], int(row["horizon_start"]), int(row["horizon_end"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected public statistic cell")
        seen.add(key)
        values = expected[key]
        if int(row["count"]) != values[2] or any(not math.isclose(float(row[name]), values[index], abs_tol=1e-9, rel_tol=1e-7) for index, name in enumerate(("sse", "sae"))):
            raise ValueError("Public sufficient statistic differs from independent errors")
    if seen != set(expected):
        raise ValueError("Incomplete public sufficient statistics")
    for name in ("mse", "mae"):
        if not math.isclose(metrics[name], manifest["score"][name], abs_tol=1e-9, rel_tol=1e-7) or float(score_row[name]) != manifest["score"][name]:
            raise ValueError("Independent metric, manifest and score CSV disagree")
    if metrics["n_observed"] != manifest["score"]["n_observed"]:
        raise ValueError("Observed count differs")
    return dict(status="PASS_INDEPENDENT_FULL_V_FP64", **metrics, public_stats_rows=len(seen),
                saved_predictions_rescored=True, new_model_inference=False, protected_T_target_operations=0)


def replay_indices(count, np):
    fixed = {0, count // 2, count - 1}
    other = np.asarray([index for index in range(count) if index not in fixed], np.int64)
    random = np.random.default_rng(SEED).choice(other, min(3, len(other)), replace=False)
    return np.asarray(sorted(fixed | set(map(int, random))), np.int64)


def cost_route_class(deployment):
    class NoBoltPhysicalRoute(deployment.Route):
        def __init__(self, model, norm, task, run_id, artifact_sha):
            self.spec = dict(run_id=run_id, family="PRIORITY_NO_BOLT")
            self.task, self.run_id, self.family, self.device = task, run_id, "PRIORITY_NO_BOLT", "cpu"
            self.model, self.mean, self.std = model, norm["mean"], norm["std"]
            self.pool, self.owns_pool = deployment.FoundationPool(None, "cpu"), True
            self.g_route = self.b_route = self.bundle = self.ridge_artifact = None
            self._closed, self._stage_mode, self._stages, self.last_receipt = False, False, {}, {}
            self.latent_k, self.artifact_sha256 = K, artifact_sha
            self.mean_t = deployment.torch.as_tensor(self.mean, device="cpu")
            self.std_t = deployment.torch.as_tensor(self.std, device="cpu")
            self.model.eval()

        def _predict_normalized(self, x, physical, observed):
            direct = self.model.direct
            z = self._step("P_E", lambda: direct.adapter.encode(x, K))
            future = self._step("direct_head", lambda: direct.head(z.transpose(1, 2)).transpose(1, 2))
            return self._step("D", lambda: direct.adapter.decode(x, z, future, K))

    return NoBoltPhysicalRoute


def assert_old_identities_dead(identities, process_api):
    for identity in identities:
        try:
            actual = process_api.Process(int(identity["pid"]))
            if abs(actual.create_time() - float(identity["create_time"])) < .01:
                raise RuntimeError("An original owned scheduler/worker identity is still alive")
        except process_api.NoSuchProcess:
            continue


def metadata(public, cache, plan_sha, process_api, plan_path=None):
    path = inside(public, NAME + "/PREREG_PLAN.json")
    if plan_path is not None and Path(plan_path).resolve() != path:
        raise ValueError("Explicit plan must identify the current priority PREREG_PLAN")
    if not plan_sha or sha(path) != plan_sha:
        raise ValueError("Explicit ROOT-sealed plan SHA required")
    plan = read(path)
    if (plan.get("schema") != "PRIORITY_B2_NO_BOLT_PLAN_v1" or plan.get("status") != "SEALED_BY_ROOT"
            or plan.get("namespace") != NAME or plan.get("stage") != "POST_T_V_ONLY_DEVELOPMENT"
            or plan.get("materialized_prefix_end_exclusive") != 16772
            or plan.get("original_parent_task_identity_inherited_not_recomputed") is not True
            or plan.get("protected_tail_numeric_parse_allowed") is not False):
        raise ValueError("Executable Jena prefix priority contract required")
    expected_optimizer = dict(name="AdamW", lr=.001, weight_decay=0., effective_batch=16, microbatch=4, gradient_clip=1.)
    if plan["optimizer"] != expected_optimizer or plan["epochs"] != {"R5a": [3.], "R5b": [.25, 1., 3.]} or plan["seed"] != SEED:
        raise ValueError("Preregistered learning recipe changed")
    if len(plan["source39"]) != 39 or len(plan["immutable_public"]) != 9:
        raise ValueError("Full original source and immutable metadata pins required")
    for name, digest in {**plan["source39"], **plan["immutable_public"]}.items():
        if sha(inside(public, name)) != digest:
            raise ValueError("Original source/immutable contract changed: " + name)
    for name, key in (("direct_no_bolt.py", "direct_model_sha256"), ("safe_jena_prefix.py", "prefix_loader_sha256"),
                      ("priority_runner.py", "runner_sha256")):
        if sha(inside(public, NAME + "/" + name)) != plan[key]:
            raise ValueError("Priority source changed: " + name)
    for key in ("stop_proof", "scheduler_blocked", "stop_identity_snapshot"):
        item = plan[key]
        if sha(inside(cache, item["cache_relative_path"])) != item["sha256"]:
            raise ValueError("Actual stop/dispatch proof changed")
    stop = read(inside(cache, plan["stop_proof"]["cache_relative_path"]))
    blocked = read(inside(cache, plan["scheduler_blocked"]["cache_relative_path"]))
    before = read(inside(cache, plan["stop_identity_snapshot"]["cache_relative_path"]))
    if (stop.get("status") != "OWNED_SCHEDULER_AND_WORKERS_STOPPED_CHECKPOINTS_PRESERVED"
            or stop.get("remaining_owned_processes") != [] or stop.get("scheduler_dispatch_disabled") is not True
            or blocked.get("existing_queued_commands_must_not_be_resumed_automatically") is not True
            or len(before["owned_tree"]) != stop["owned_identities_checked"]):
        raise ValueError("Original queue closure and dispatch block are not established")
    assert_old_identities_dead(before["owned_tree"], process_api)
    if not all(isinstance(plan["role_run_ids"][role], str) and Path(plan["role_run_ids"][role]).name == plan["role_run_ids"][role] for role in ROLES):
        raise ValueError("Safe exact preregistered run IDs required")
    return plan


def require_environment():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or any(os.environ.get(key) != "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            or any(key.startswith("INFORMS_GPU_") for key in os.environ)):
        raise ValueError("Explicit offline CPU2 environment without GPU lease required")


def bound_R4(plan, public, cache):
    r4 = plan["R4"]
    for root, name, key in ((cache, "artifact_cache_relative_path", "artifact_sha256"),
                            (public, "fit_receipt_public_relative_path", "fit_receipt_sha256"),
                            (public, "V_manifest_public_relative_path", "V_manifest_sha256")):
        if sha(inside(root, r4[name])) != r4[key]:
            raise ValueError("Original exact R4 artifact/metadata changed")
    receipt = read(inside(public, r4["fit_receipt_public_relative_path"]))
    manifest = read(inside(public, r4["V_manifest_public_relative_path"]))
    if (receipt["status"] != "COMPLETE_VALID" or receipt["run_id"] != r4["run_id"]
            or receipt["artifact_sha256"] != r4["artifact_sha256"] or receipt["model_state_sha"] != r4["model_state_sha"]
            or receipt["refit_scaler_sha"] != r4["refit_scaler_sha"]
            or manifest["run_id"] != r4["run_id"] or manifest["split"] != "V_select"
            or manifest["scaler_sha"] != r4["refit_scaler_sha"]
            or manifest["data_sha"] != plan["task_contract"]["data_sha256"]):
        raise ValueError("Pinned R4 recipe and saved V identity differ")
    return manifest


def load_context(public, plan, plan_sha, model_operations):
    sys.path.insert(0, str(public))
    import numpy as np
    import evaluation
    import data_contract as data
    if Path(data.__file__).resolve() != (public / "data_contract.py").resolve():
        raise ValueError("Original scientific modules must come from calling campaign")
    context = dict(np=np, evaluation=evaluation, data=data, plan_sha=plan_sha,
        new_sources={name: plan[key] for name, key in (("direct_no_bolt.py", "direct_model_sha256"),
            ("safe_jena_prefix.py", "prefix_loader_sha256"), ("priority_runner.py", "runner_sha256"))},
        r4_artifact_sha=plan["R4"]["artifact_sha256"])
    if model_operations:
        import torch
        import fit_base as fit
        import networks
        import training
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only priority process unexpectedly initialized CUDA")
        direct = module("priority_direct_model_library", inside(public, NAME + "/direct_no_bolt.py"))
        context.update(torch=torch, fit=fit, networks=networks, training=training, direct=direct)
    return context


def load_R4_bundle(plan, cache, context):
    torch, data, fit = (context[key] for key in ("torch", "data", "fit"))
    bundle = torch.load(inside(cache, plan["R4"]["artifact_cache_relative_path"]), map_location="cpu", weights_only=False)
    if (bundle["model_state_sha"] != plan["R4"]["model_state_sha"] or state_sha(bundle["model"], data) != bundle["model_state_sha"]
            or bundle["contract"]["data_sha"] != plan["task_contract"]["data_sha256"]
            or bundle["contract"]["source_sha"] != fit._source_contract()[0]
            or fit.scaler_fingerprint(bundle["normalization"]["mean"], bundle["normalization"]["std"]) != plan["R4"]["refit_scaler_sha"]):
        raise ValueError("R4 tensor/state/normalization contract differs")
    return bundle


def load_task_prefix(asset, plan, context, public):
    loader = module("priority_safe_jena_prefix", inside(public, NAME + "/safe_jena_prefix.py"))
    original_contract = read(inside(public, "DATA_CONTRACT.json"))["assets"]["jena"]
    task, receipt = loader.load_jena_prefix(asset, original_contract)
    contract, data, fit = plan["task_contract"], context["data"], context["fit"]
    bounds = {key: tuple(value) for key, value in contract["primary_split_ranges"].items()}
    if (task["physical"].shape != (16772, C) or task["observed"].shape != (16772, C)
            or task["L"] != L or task["H"] != H or list(task["columns"]) != contract["columns"]
            or task["data_sha256"] != contract["data_sha256"] or task["array_sha256"] != contract["array_sha256"]
            or bounds["V_select"][1] != 16772):
        raise ValueError("Prefix materialization or inherited original task identity differs")
    mean, std, count = data.fit_statistics(task, bounds["A_fit"][1])
    if ({key: data.array_sha256(value) for key, value in (("mean", mean), ("std", std), ("count", count))}
            != contract["A_fit_statistics_fingerprints"] or fit.scaler_fingerprint(mean, std) != contract["search_scaler_sha256"]):
        raise ValueError("Materialized original A50 statistics differ")
    mean, std, _ = data.fit_statistics(task, bounds["A_stop"][1])
    if fit.scaler_fingerprint(mean, std) != contract["refit_scaler_sha256"]:
        raise ValueError("Materialized original A60 normalization differs")
    return task, bounds, receipt


def load_saved_prediction(public_folder, private_folder, r4_reference, context):
    manifest = read(public_folder / "PREDICTION_MANIFEST.json")
    reference = read(public_folder / "REFERENCE_CONTRACT.json")
    path = private_folder / "V_select.npz"
    if sha(path) != manifest["archive_sha256"]:
        raise ValueError("Saved private V bytes changed")
    with context["np"].load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
            raise ValueError("Unexpected saved V members")
        arrays = {name: archive[name] for name in archive.files}
    for name, array in arrays.items():
        if context["evaluation"].array_fingerprint(array) != manifest["arrays"][name]:
            raise ValueError("Saved V array fingerprint differs")
    validate_cohort(arrays, r4_reference, context["evaluation"])
    validate_cohort(arrays, reference, context["evaluation"])
    if any(reference[key] != manifest[key] for key in ("run_id", "split", "cohort", "data_sha", "scaler_sha", "source_sha", "config_sha", "model_sha")):
        raise ValueError("Saved V reference identity differs from completed manifest")
    if (arrays["pred_normalized"].shape != (43, H, C) or manifest["split"] != "V_select"
            or manifest["source_sha"] != fingerprint(context["new_sources"])):
        raise ValueError("Saved V split or source identity differs")
    return arrays, manifest


def validate_accuracy_binding(manifest, receipt, role, plan, context, source_contract):
    expected_model = fingerprint(dict(artifact_sha=receipt["artifact_sha256"], model_state_sha=receipt["model_state_sha"]))
    if (receipt["run_id"] != plan["role_run_ids"][role] or receipt["role"] != role
            or manifest["run_id"] != receipt["run_id"] or manifest["role"] != role
            or manifest["artifact_sha256"] != receipt["artifact_sha256"] or manifest["model_sha"] != expected_model
            or manifest["config_sha"] != fingerprint(source_contract["spec"])
            or manifest["source_sha"] != fingerprint(context["new_sources"])
            or manifest["data_sha"] != plan["task_contract"]["data_sha256"]
            or manifest["scaler_sha"] != plan["task_contract"]["refit_scaler_sha256"]):
        raise ValueError("Accuracy route is not the exact preregistered trained role/model/scaler")


def load_priority_model(public_run, private_run, role, plan, r4_bundle, context):
    receipt = read(public_run / "BASE_FIT_RECEIPT.json")
    artifact = private_run / "selected_bundle.pt"
    if receipt["status"] != "COMPLETE_PRIORITY_NO_BOLT_FIT" or receipt["role"] != role or sha(artifact) != receipt["artifact_sha256"]:
        raise ValueError("Completed exact priority checkpoint required")
    bundle = context["torch"].load(artifact, map_location="cpu", weights_only=False)
    source = read(public_run / "SOURCE_CONTRACT.json")
    if (bundle["plan_sha256"] != context["plan_sha"] or bundle["R4_artifact_sha256"] != plan["R4"]["artifact_sha256"]
            or state_sha(bundle["model"], context["data"]) != receipt["model_state_sha"]
            or bundle["spec"] != source["spec"]
            or any(bundle["contract"][key] != source[key] for key in ("source_sha", "config_sha", "data_sha", "model_sha"))
            or context["fit"].scaler_fingerprint(bundle["normalization"]["mean"], bundle["normalization"]["std"]) != plan["R4"]["refit_scaler_sha"]):
        raise ValueError("Selected priority model/scaler/source contract differs")
    model = build_model(role, bundle["normalization"], r4_bundle, context["direct"], context["networks"], context["torch"])
    model.load_state_dict(bundle["model"], strict=True)
    model.eval()
    verify_fixed_adapter(model, r4_bundle, role, context["torch"])
    return dict(model=model, bundle=bundle, receipt=receipt, artifact=artifact)


def run_replay(fitted, arrays, manifest, task, context, output, deadline):
    import deployment
    np = context["np"]
    indices = replay_indices(len(arrays["origins"]), np)
    selected = arrays["origins"][indices]
    norm = fitted["bundle"]["normalization"]
    for index, origin in zip(indices, selected):
        target, mask = context["data"].target_window(task, int(origin), norm["mean"], norm["std"])
        if not np.array_equal(target, arrays["target_normalized"][index]) or not np.array_equal(mask, arrays["observed"][index]):
            raise ValueError("Replay Task V target/mask differs from saved cohort")
    route = cost_route_class(deployment)(fitted["model"], norm, task, manifest["run_id"], fitted["receipt"]["artifact_sha256"])
    predictions = []
    try:
        for offset in range(0, len(selected), 4):
            check_deadline(deadline)
            origins = selected[offset:offset + 4]
            physical = np.stack([task["physical"][int(o) - L:int(o)] for o in origins])
            observed = np.stack([task["observed"][int(o) - L:int(o)] for o in origins])
            raw = route.predict_physical(physical, observed)
            predictions.append(((raw - norm["mean"].astype(np.float64)) / norm["std"].astype(np.float64)).astype(np.float32))
        actual = np.concatenate(predictions)
        expected = arrays["pred_normalized"][indices]
        passed = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-4))
        if route.pool.instances_created != 0 or any(route.pool.counters().values()):
            raise ValueError("No-Bolt replay instantiated/invoked a foundation")
        receipt = dict(status="PASS_FRESH_CPU6_REPLAY" if passed else "FAIL_REPLAY_PARITY", origins=selected.tolist(),
            actual_origins=len(selected), microbatches=[4, 2], max_abs=float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max()),
            atol=1e-5, rtol=1e-4, artifact_sha256=fitted["receipt"]["artifact_sha256"],
            full_L_history=L, latent_k=K, same_lag_D_current_anchor=True, actual_F_streams=0, foundation_instances_created=0,
            new_T_target_operations=0, executed_now=True, independent_training_rerun=False)
    finally:
        route.close()
    new(output / "REPLAY_RECEIPT.json", receipt)
    if not passed:
        raise ValueError("Fresh checkpoint replay differs from saved V")
    return receipt


def run_cost(fitted, task, bounds, batch, context, output, deadline):
    import deployment
    import measure_costs
    import psutil
    np = context["np"]
    origins = context["data"].origins(bounds["V_select"], L, H, H)[:batch]
    if batch not in (1, 8) or len(origins) != batch:
        raise ValueError("Fresh batch1 or batch8 first original V histories required")
    physical = np.stack([task["physical"][int(o) - L:int(o)] for o in origins])
    observed = np.stack([task["observed"][int(o) - L:int(o)] for o in origins])
    route = cost_route_class(deployment)(fitted["model"], fitted["bundle"]["normalization"], task,
        fitted["receipt"]["run_id"], fitted["receipt"]["artifact_sha256"])
    check_deadline(deadline)
    class DeadlineCheck:
        def check(self):
            check_deadline(deadline)
    try:
        rows = measure_costs.measure_balanced([route], physical, observed, warmups=10, timed_trials=20, stage_trials=20, lease=DeadlineCheck())
        memory = route.memory_state()
        process_memory = psutil.Process().memory_info()
        memory.update(process_RSS_bytes=process_memory.rss, OS_peak_working_set_bytes=getattr(process_memory, "peak_wset", None))
        if route.pool.instances_created != 0 or any(route.pool.counters().values()) or any(row["F_logical_streams"] or row["F_physical_streams"] or row["F_calls"] for row in rows):
            raise ValueError("No-Bolt cost instantiated/invoked foundation")
    finally:
        route.close()
    trials = output / "COST_TRIALS.csv.gz"
    measure_costs._write_trials(trials, rows)
    receipt = dict(status="COMPLETE_PROTOCOL", run_id=fitted["receipt"]["run_id"], role=fitted["receipt"]["role"],
        device="cpu", CPU_threads=2, batch_size=batch, origins=list(map(int, origins)),
        warmups=10, timed_trials=20, stage_trials=20, physical_input_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
        observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(), physical_output_dtype="float64",
        inverse_transform_dtype="float64", artifact_sha256=fitted["receipt"]["artifact_sha256"],
        summaries=measure_costs.summarize(rows), memory=memory, actual_F_streams=0, foundation_instances_created=0,
        cached_Z_or_forecast_reads=0, trials_sha256=sha(trials), actual_trial_rows=len(rows),
        peak_process_RSS_guard_and_training_quiet_required_from_ROOT=True,
        stage_probe_is_intrusive_separate_scope=True, training_peak_included=False, protected_T_target_operations=0)
    new(output / "COST_RECEIPT.json", receipt)
    return receipt


def execute(public, cache, asset, plan_sha, role, mode, deadline, output, batch=None, plan_path=None):
    import psutil
    public, cache, asset, output = map(lambda path: Path(path).resolve(), (public, cache, asset, output))
    plan = metadata(public, cache, plan_sha, psutil, plan_path)
    if role not in ROLES or mode not in MODES or deadline is None or not time.time() < deadline <= plan["deadline_unix"]:
        raise ValueError("Explicit role/mode/live original deadline required")
    if mode == "cost" and batch not in (1, 8) or mode != "cost" and batch is not None:
        raise ValueError("Only cost mode accepts explicit batch1/8")
    require_environment()
    root = public / NAME
    if output.exists() or output == root or not output.is_relative_to(root):
        raise ValueError("Fresh diagnostic output within priority namespace required")
    run_id = plan["role_run_ids"][role]
    public_run, private_run = root / "runs" / run_id, cache / NAME / "runs" / run_id
    public_prediction = root / "predictions" / run_id / "V_select"
    private_prediction = cache / NAME / "predictions" / run_id
    if mode == "fit" and (public_run.exists() or private_run.exists()):
        raise FileExistsError("No automatic reuse/resumption of an existing priority attempt")
    r4_manifest = bound_R4(plan, public, cache)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    new(output / "STARTED.json", dict(role=role, mode=mode, run_id=run_id, plan_sha256=plan_sha,
        actual_pid=os.getpid(), actual_create_time=psutil.Process().create_time(), deadline_unix=deadline,
        protected_T_target_operations=0, materialized_prefix_end_exclusive=16772))
    before_cpu = psutil.Process().cpu_times()
    context, result, failed = None, None, None
    try:
        context = load_context(public, plan, plan_sha, mode != "rescore")
        if mode == "rescore":
            arrays, manifest = load_saved_prediction(public_prediction, private_prediction, r4_manifest, context)
            fit_receipt = read(public_run / "BASE_FIT_RECEIPT.json")
            if sha(private_run / "selected_bundle.pt") != fit_receipt["artifact_sha256"]:
                raise ValueError("Saved accuracy artifact bytes changed")
            validate_accuracy_binding(manifest, fit_receipt, role, plan, context, read(public_run / "SOURCE_CONTRACT.json"))
            with (public_prediction / "SCORES.csv").open(encoding="utf-8", newline="") as handle:
                scores = list(csv.DictReader(handle))
            if len(scores) != 1:
                raise ValueError("Exactly one saved score row required")
            with gzip.open(public_prediction / "PUBLIC_ERROR_STATS.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                statistics = list(csv.DictReader(handle))
            result = independent_rescore(arrays, manifest, statistics, scores[0], context["np"])
            new(output / "RESCORE_RECEIPT.json", result)
        else:
            task, bounds, access = load_task_prefix(asset, plan, context, public)
            new(output / "PREFIX_ACCESS_RECEIPT.json", access)
            r4_bundle = load_R4_bundle(plan, cache, context)
            if mode == "fit":
                public_run.mkdir(parents=True, exist_ok=False)
                private_run.mkdir(parents=True, exist_ok=False)
                fitted = run_fit(role, run_id, task, bounds, r4_bundle, context, public_run, private_run, deadline)
                if "model" not in fitted:
                    result = fitted
                else:
                    manifest = save_prediction(run_id, role, fitted, task, bounds, r4_manifest, context,
                        public_prediction, private_prediction, deadline)
                    result = dict(status="COMPLETE_FIT_AND_FULL_V", fit_receipt=fitted["receipt"], V_score=manifest["score"],
                                  full_V_completed=True, independent_rescore_pending=True, checkpoint_replay_pending=True)
            else:
                fitted = load_priority_model(public_run, private_run, role, plan, r4_bundle, context)
                manifest = read(public_prediction / "PREDICTION_MANIFEST.json")
                validate_accuracy_binding(manifest, fitted["receipt"], role, plan, context, read(public_run / "SOURCE_CONTRACT.json"))
                if mode == "replay":
                    arrays, manifest = load_saved_prediction(public_prediction, private_prediction, r4_manifest, context)
                    result = run_replay(fitted, arrays, manifest, task, context, output, deadline)
                else:
                    result = run_cost(fitted, task, bounds, batch, context, output, deadline)
                    result["accuracy_manifest_sha256"] = sha(public_prediction / "PREDICTION_MANIFEST.json")
        check_deadline(deadline)
    except Exception as error:
        failed = type(error).__name__
        raise
    finally:
        metadata(public, cache, plan_sha, psutil, plan_path)
        if mode != "rescore" and context is not None and context["torch"].cuda.is_initialized():
            raise RuntimeError("Priority run initialized CUDA")
        after_cpu = psutil.Process().cpu_times()
        memory = psutil.Process().memory_info()
        new(output / "RESULT.json", dict(status="FAILED_PRESERVED" if failed else (result or {}).get("status", "INCOMPLETE"),
            role=role, mode=mode, run_id=run_id, error_class=failed, scientific_result=result,
            elapsed_process_wall_seconds=time.time() - started, actual_F_streams=0, foundation_instances_created=0,
            actual_CPU_process_seconds=after_cpu.user + after_cpu.system - before_cpu.user - before_cpu.system,
            RSS_bytes=memory.rss, OS_peak_working_set_bytes=getattr(memory, "peak_wset", None),
            new_T_target_operations=0, original_source39_and_immutable9_preserved=True,
            checkpoint_on_exception_supported=False, original_stage_checkpoint_protocol_retained=True,
            old_full_task_fingerprints_inherited_not_recomputed=True, independent_generalization_claimed=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for key in ("public-root", "cache-root", "asset-root", "plan", "plan-sha", "output"):
        parser.add_argument("--" + key)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--batch", type=int, choices=(1, 8))
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED", input_reads=0, files_written=0,
            torch_imported="torch" in sys.modules, numpy_imported="numpy" in sys.modules,
            actual_Task_checkpoint_forward_training_cost_or_child_operations=0)))
        return 0
    if not all((args.public_root, args.cache_root, args.asset_root, args.plan_sha, args.role, args.mode, args.deadline, args.output)):
        parser.error("Explicit sealed plan, roots, role, mode, deadline and fresh output required")
    result = execute(args.public_root, args.cache_root, args.asset_root, args.plan_sha, args.role, args.mode, args.deadline, args.output, args.batch, args.plan)
    print(json.dumps(dict(status=result["status"], role=args.role, mode=args.mode)))
    return 0 if result["status"].startswith(("COMPLETE", "PASS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
