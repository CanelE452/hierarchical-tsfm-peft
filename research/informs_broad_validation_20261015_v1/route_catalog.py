"""Hash-bound portable deployment routes assembled from actual fit receipts."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

import data_contract as data
import fit_base
from training import fingerprint

DIRECT = {"DIRECT_G", "DIRECT_B2", "DIRECT_B4", "CURRENT", "SEASONAL", "RAW_RIDGE"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("Route artifact escapes its declared root")
    return path


def _bound(cache, relative, expected):
    path = _inside(cache, relative)
    if not expected or not path.is_file() or data.sha256_file(path) != expected:
        raise ValueError("Route artifact SHA is absent or changed")
    return path


def _base(run_id, row, task, cache, public):
    folder = public/"runs"/run_id
    receipt_path = next((folder/name for name in ("BASE_FIT_RECEIPT.json", "CONTROL_FIT_RECEIPT.json", "RIDGE_FIT_RECEIPT.json")
                         if (folder/name).is_file()), None)
    if receipt_path is None:
        raise FileNotFoundError("Route has no complete trained artifact receipt")
    receipt = _json(receipt_path)
    if receipt.get("run_id") != run_id or receipt.get("status") != "COMPLETE_VALID":
        raise ValueError("Route requires this run's complete valid fit receipt")
    relative, sha = receipt["artifact_cache_relative_path"], receipt["artifact_sha256"]
    path = _bound(cache, relative, sha)
    family = receipt.get("family", row["family"])
    if family == "RAW_RIDGE":
        with np.load(path, allow_pickle=False) as archive:
            contract = json.loads(str(archive["contract_json"].item()))
            fit_end = int(archive["actual_fit_end"].item())
        if contract["data_sha"] != task["data_sha256"]:
            raise ValueError("Ridge route data fingerprint differs")
        size, prediction_sha, foundation_sha = None, receipt["model_sha"], "NO_FOUNDATION"
        k = 0
    else:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["spec"]["run_id"] != run_id or saved["contract"]["data_sha"] != task["data_sha256"]:
            raise ValueError("Route bundle belongs to another run or data")
        if saved["spec"].get("fold", "primary") != row.get("fold", "primary"):
            raise ValueError("Route bundle belongs to another time fold")
        family = saved["spec"]["family"]
        fit_end = int(saved["normalization"].get("train_end", saved["normalization"]["fit_end"]))
        foundation_sha = saved["contract"]["model_sha"]
        size = None if family in DIRECT else "small"
        prediction_sha = fingerprint(dict(artifact_sha=sha, model_state_sha=saved["model_state_sha"],
                                           normalization=saved["contract"]["scaler_sha"]))
        k = int(saved["spec"].get("latent_k", len(task["columns"])))
    fold = row.get("fold", "primary")
    ranges = data.split_ranges(task, fold)
    is_control = row["family"] in {"G_REFIT_APLUSM", "LORA_REFIT_APLUSM"}
    if fit_end > ranges["M_cal"][1 if is_control else 0]:
        raise ValueError("Route parent trained beyond its registered information boundary")
    spec = dict(run_id=run_id, dataset=task["name"], family=family, fold=fold,
                artifact_cache_relative_path=relative, artifact_sha256=sha, latent_k=0 if family in DIRECT else k,
                canonical_fit_end=ranges["A_stop"][1], foundation_sha=foundation_sha,
                expected_prediction_model_sha=prediction_sha,
                fit_receipt_public_relative_path=receipt_path.relative_to(public).as_posix(),
                fit_receipt_sha256=data.sha256_file(receipt_path))
    return spec, size


def _registry(public):
    with (public/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        if row["experiment_id"] in result:
            raise ValueError("Duplicate registered route ID")
        result[row["experiment_id"]] = row
    return result


def _size(*sizes):
    active = {value for value in sizes if value is not None}
    if len(active) > 1:
        raise ValueError("A deployment route cannot silently share different backbone sizes")
    return next(iter(active)) if active else None


def build_route(run_id, task, cache_root, public_root):
    """Return ``(deployment.Route spec, small/mini/tiny or None)``; no F loads."""
    cache, public = Path(cache_root), Path(public_root)
    rows = _registry(public)

    def resolve(identity, ancestors):
        if identity in ancestors or identity not in rows or Path(identity).name != identity:
            raise ValueError("Unknown or cyclic portable route ID")
        row = rows[identity]
        if row["dataset"] != task["name"]:
            raise ValueError("Registered route task differs")
        family, fold = row["family"], row.get("fold", "primary")
        ancestors = set(ancestors) | {identity}
        canonical = data.split_ranges(task, fold)["A_stop"][1]
        ordered_path = public/"ordered_predictions"/identity/"PREDICTION_MANIFEST.json"
        if ordered_path.exists():
            manifest = _json(ordered_path)
            meta = manifest["metadata"]
            if manifest["data_sha"] != task["data_sha256"] or manifest["run_id"] != identity:
                raise ValueError("Ordered route identity differs")
            k = int(meta["inference_latent_k"])
            if k != int(row["latent_k"]) or meta.get("fold",fold) != fold:
                raise ValueError("Ordered prefix K/fold differs from its registered row")
            if meta["variant"] == "nested_pca":
                relative = f"ordered/{fold}_{task['name']}/nested_pca.npz"
                _bound(cache, relative, meta["parent_artifact_sha256"])
                if not meta.get("foundation_sha"):
                    raise ValueError("Nested PCA route needs its actual foundation identity")
                spec = dict(run_id=identity, dataset=task["name"], family="NESTED_PCA", fold=fold,
                            artifact_cache_relative_path=relative, artifact_sha256=meta["parent_artifact_sha256"],
                            foundation_sha=meta["foundation_sha"], canonical_fit_end=canonical, latent_k=k)
                size = "small"
            else:
                parent = meta["parent_run_id"]
                if parent not in rows:
                    raise ValueError("Ordered parent is not registered")
                spec, size = _base(parent, rows[parent], task, cache, public)
                if spec["artifact_sha256"] != meta["parent_artifact_sha256"]:
                    raise ValueError("Ordered alias points to different selected weights")
                spec.update(run_id=identity, source_parent_run_id=parent, latent_k=k,
                            prefix_variant=meta["variant"])
            spec["ordered_manifest_public_relative_path"] = ordered_path.relative_to(public).as_posix()
            spec["ordered_manifest_sha256"] = data.sha256_file(ordered_path)
        elif family.startswith("RAW_BOLT_"):
            size = family.removeprefix("RAW_BOLT_").lower()
            if size not in {"small", "mini", "tiny"}:
                raise ValueError("Unregistered Bolt size")
            spec = dict(run_id=identity, dataset=task["name"], family="RAW", fold=fold,
                        canonical_fit_end=canonical, latent_k=len(task["columns"]))
            if size in {"mini", "tiny"}:
                assets_path = public/"MODEL_ASSETS.json"
                entry = _json(assets_path)["models"][size]
                for file in entry["files"].values():
                    _bound(cache, file["cache_relative_path"], file["sha256"])
                spec["foundation_asset"] = dict(repository=entry["repository"], revision=entry["revision"],
                    files={name:dict(sha256=value["sha256"], bytes=value["bytes"]) for name,value in entry["files"].items()})
            else:
                probe = public/"PROBE_ACTUAL_GPU.json"
                if probe.exists():
                    value = _json(probe)
                    spec["foundation_sha"] = value["model_state_sha256"]
                    spec["foundation_revision"] = value["model_revision"]
                    mean,std,_ = data.fit_statistics(task,canonical)
                    spec["expected_prediction_model_sha"] = fingerprint(dict(
                        baseline=dict(kind="RAW",fold=fold,L=task["L"],H=task["H"]),
                        foundation_sha=spec["foundation_sha"],scaler_sha=fit_base.scaler_fingerprint(mean,std),
                        data_sha=task["data_sha256"]))
        elif family in {"CURRENT", "SEASONAL_NAIVE"}:
            spec = dict(run_id=identity, dataset=task["name"], family="CURRENT" if family == "CURRENT" else "SEASONAL",
                        fold=fold, canonical_fit_end=canonical, latent_k=0, period=int(task.get("period", 24)))
            size = None
        elif family in {"ALPHA0", "ALPHA05", "ALPHA1", "ALPHA_FIT", "CONDITIONAL", "PAST_CONTROL", "LEGACY_RESIDUAL"}:
            group = identity
            label = {"ALPHA0":"alpha0", "ALPHA05":"alpha05", "ALPHA1":"alpha1", "ALPHA_FIT":"fitted"}.get(family)
            if label:
                group = identity.replace("_"+family+"_", "_ALPHA_FIT_")
            receipt_path = public/"combinations"/group/"receipt.json"
            receipt = _json(receipt_path)
            if receipt.get("execution_status") != "COMPLETE_VALID" or receipt["contract"]["data_sha"] != task["data_sha256"]:
                raise ValueError("Combination route is not a complete valid fit")
            fit_spec_path = cache/"combinations"/group/"spec.json"
            fit_spec = _json(fit_spec_path)
            variant = receipt["variant"]
            scientific = {key:value for key,value in fit_spec.items() if key not in {"run_id","resume_checkpoint"}}
            if fit_spec["run_id"] != group or fit_spec["variant"] != variant or fingerprint(scientific) != receipt["contract"]["config_sha"]:
                raise ValueError("Saved calibration specification differs from its fit contract")
            g = b = None
            gs = bs = None
            if variant != "legacy_residual" and label != "alpha1":
                g, gs = resolve(fit_spec["g_run_id"], ancestors)
            if variant != "past_control" and label != "alpha0":
                b, bs = resolve(fit_spec["parent_run_id"], ancestors)
            for key,parent in (("g_v",g),("parent_v",b)):
                expected = receipt.get("parent_models",{}).get(key,{}).get("model_sha")
                if expected and parent is not None:
                    actual = parent.get("standard_V_binding",{}).get("model_sha",parent.get("expected_prediction_model_sha"))
                    if not actual or actual != expected:
                        raise ValueError("Actual deployment parent differs from the calibrated parent forecast")
            if label in {"alpha0", "alpha1"}:
                spec = copy.deepcopy(g if label == "alpha0" else b)
                spec.update(source_parent_run_id=spec["run_id"], run_id=identity)
            elif label:
                alpha = .5 if label == "alpha05" else float(receipt["alpha_fit"]["alpha_fit"])
                if not 0 <= alpha <= 1:
                    raise ValueError("Fitted alpha is outside its registered interval")
                spec = dict(run_id=identity, dataset=task["name"], family="ALPHA", fold=fold,
                            canonical_fit_end=canonical, alpha=alpha, g_route=g, b_route=b)
            else:
                relative = f"combinations/{group}/selected.pt"
                sha = receipt["selected_checkpoint_sha256"]
                _bound(cache, relative, sha)
                spec = dict(run_id=identity, dataset=task["name"], family="COMBINATION", fold=fold,
                            canonical_fit_end=canonical, variant=variant,
                            artifact_cache_relative_path=relative, artifact_sha256=sha)
                if g is not None: spec["g_route"] = g
                if b is not None: spec["b_route"] = b
            spec["calibration_receipt_public_relative_path"] = receipt_path.relative_to(public).as_posix()
            spec["calibration_receipt_sha256"] = data.sha256_file(receipt_path)
            spec["calibration_spec_sha256"] = data.sha256_file(fit_spec_path)
            size = _size(gs, bs)
        elif family == "RAW_CALIBRATED":
            receipt_path = public/"raw_calibration"/identity/"receipt.json"
            receipt = _json(receipt_path)
            if receipt.get("status") != "COMPLETE_VALID" or receipt["contract"]["data_sha"] != task["data_sha256"]:
                raise ValueError("Raw bias route is not a complete valid fit")
            b, size = resolve(receipt["parent_run_id"], ancestors)
            expected = receipt["contract"].get("parent_model_sha")
            actual = b.get("standard_V_binding",{}).get("model_sha",b.get("expected_prediction_model_sha"))
            if expected and expected != actual:
                raise ValueError("Raw bias deployment parent differs from its calibrated forecast")
            _bound(cache, receipt["artifact_cache_relative_path"], receipt["artifact_sha256"])
            spec = dict(run_id=identity, dataset=task["name"], family="RAW_CALIBRATED", fold=fold,
                        canonical_fit_end=canonical, b_route=b,
                        artifact_cache_relative_path=receipt["artifact_cache_relative_path"], artifact_sha256=receipt["artifact_sha256"],
                        calibration_receipt_public_relative_path=receipt_path.relative_to(public).as_posix(),
                        calibration_receipt_sha256=data.sha256_file(receipt_path))
        else:
            spec, size = _base(identity, row, task, cache, public)
        standard = public/"predictions"/identity/"V_select/PREDICTION_MANIFEST.json"
        if standard.exists():
            value = _json(standard)
            if value["run_id"] != identity or value["data_sha"] != task["data_sha256"]:
                raise ValueError("Standard V model binding belongs to another route")
            _bound(cache,value["cache_relative_path"],value["archive_sha256"])
            spec["standard_V_binding"] = dict(model_sha=value["model_sha"], archive_sha256=value["archive_sha256"],
                manifest_public_relative_path=standard.relative_to(public).as_posix(), manifest_sha256=data.sha256_file(standard))
        spec["model_size"] = size
        return spec, size

    return resolve(str(run_id), set())


def expected_F_streams(spec, batch, channels):
    family = spec["family"]
    if family in DIRECT:
        return 0
    if family in {"RAW", "LORA"}:
        return int(batch)*int(channels)
    if family in {"ALPHA", "COMBINATION", "RAW_CALIBRATED"}:
        return sum(expected_F_streams(spec[key], batch, channels) for key in ("g_route", "b_route") if key in spec)
    if family in {"B0", "B2", "B4", "LAG_CAUSAL", "LAG_BIDIR", "ADAPTS", "PCA_FIXED_BOLT", "NESTED_PCA"}:
        return int(batch)*int(spec["latent_k"])
    raise ValueError("Unknown family for actual F row accounting")


def artifact_bindings(spec):
    result = {}
    def visit(route, prefix):
        if "artifact_cache_relative_path" in route:
            result[prefix] = dict(cache_relative_path=route["artifact_cache_relative_path"], sha256=route["artifact_sha256"])
        for key in ("g_route", "b_route"):
            if key in route:visit(route[key], prefix+"."+key)
    visit(spec, "route")
    return result
