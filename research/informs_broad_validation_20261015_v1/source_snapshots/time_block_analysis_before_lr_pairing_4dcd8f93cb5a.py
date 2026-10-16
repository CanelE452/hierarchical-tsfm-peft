"""Descriptive matched time-block errors; no bootstrap, selection, or inference."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import audit_predictions as audit
import channel_patterns as patterns
import data_contract as data


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _array_contract(array):
    value = np.ascontiguousarray(array)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def block_plan(origins, horizon, period):
    origins = np.asarray(origins)
    horizon, period = int(horizon), int(period)
    if horizon < 1 or period < 1 or origins.ndim != 1 or not len(origins) or origins.dtype.kind not in "iu":
        raise ValueError("Positive metadata period/H and integer evaluation origins required")
    if len(origins) > 1 and not np.all(np.diff(origins) == horizon):
        raise ValueError("Time blocks require consecutive nonoverlapping strideH origins")
    size = int(math.ceil(max(period, horizon) / horizon))
    blocks = []
    for start in range(0, len(origins), size):
        stop = min(start + size, len(origins))
        blocks.append(dict(block_index=len(blocks), indices=list(range(start, stop)), n_origins=stop - start,
            origin_start=int(origins[start]), origin_last=int(origins[stop - 1]),
            target_end_exclusive=int(origins[stop - 1]) + horizon, short_tail=stop - start < size))
    return dict(period=period, horizon=horizon, block_origins=size, n_origins=len(origins),
                origin_contract=_array_contract(origins), blocks=blocks)


def record_from_arrays(arrays, identity):
    p = np.asarray(arrays["pred_normalized"], np.float64)
    y = np.asarray(arrays["target_normalized"], np.float64)
    mask = np.asarray(arrays["observed"])
    if p.ndim != 3 or p.shape != y.shape or p.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Matching prediction/target/original bool mask [N,H,C] required")
    if not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError("Nonfinite observed prediction/target")
    if np.asarray(arrays["origins"]).shape != (p.shape[0],) or np.asarray(arrays["channel_ids"]).shape != (p.shape[2],):
        raise ValueError("Prediction axes and origin/channel axes differ")
    difference = np.zeros_like(p)
    np.subtract(p, y, out=difference, where=mask)
    return dict(identity=dict(identity), array_contract={key: _array_contract(arrays[key]) for key in sorted(audit.REFERENCE_KEYS)},
        sse=(difference * difference).sum(axis=1, dtype=np.float64),
        sae=np.abs(difference).sum(axis=1, dtype=np.float64), counts=mask.sum(axis=1, dtype=np.int64),
        origins=np.asarray(arrays["origins"]).copy(), horizon=p.shape[1], channels=p.shape[2])


def _compatible(left, right):
    for key in ("dataset", "fold", "cohort", "data_sha", "scaler_sha"):
        if left["identity"][key] != right["identity"][key]:
            raise ValueError("Paired identity differs: " + key)
    for key in audit.REFERENCE_KEYS:
        if left["array_contract"][key] != right["array_contract"][key]:
            raise ValueError("Paired target/mask/channel/origins contract differs: " + key)
    if not np.array_equal(left["counts"], right["counts"]):
        raise ValueError("Paired observed counts differ")


def paired_summary(left, right, indices):
    _compatible(left, right)
    indices = np.asarray(indices, np.int64)
    counts = left["counts"][indices].sum(axis=0, dtype=np.int64)
    active = np.flatnonzero(counts)
    result = {}
    for metric, field in (("mse", "sse"), ("mae", "sae")):
        a = left[field][indices].sum(axis=0, dtype=np.float64)
        b = right[field][indices].sum(axis=0, dtype=np.float64)
        result["left_" + metric] = math.fsum(float(a[i]) / int(counts[i]) for i in active) / len(active) if len(active) else None
        result["right_" + metric] = math.fsum(float(b[i]) / int(counts[i]) for i in active) / len(active) if len(active) else None
        result["delta_" + metric] = math.fsum((float(a[i]) - float(b[i])) / int(counts[i]) for i in active) / len(active) if len(active) else None
    possible = int(len(indices) * left["horizon"] * left["channels"])
    return dict(result, n_origins=len(indices), total_channels=left["channels"], active_channels=len(active),
        zero_observed_channels=left["channels"] - len(active), observed_target_cells=int(counts.sum()),
        possible_target_cells=possible, observed_fraction=float(counts.sum()) / possible if possible else None)


def _recipe_pair(left, right):
    a, b = left["identity"], right["identity"]
    strict = {("B2", "B0"), ("B4", "B2"), ("LAG_CAUSAL", "B0"), ("LAG_BIDIR", "LAG_CAUSAL")}
    if (a["family"], b["family"]) in strict and any(a[key] != b[key] for key in ("seed", "k", "budget")):
        return False
    if a["family"] == "CONDITIONAL":
        if b["family"] in {"PAST_CONTROL", "ALPHA_FIT"} and (not a["parent_signature"] or a["parent_signature"] != b["parent_signature"]):
            return False
        if b["family"] == "PAST_CONTROL" and any(a[key] != b[key] for key in ("seed", "budget")):
            return False
        if b["family"] == "DIRECT_G" and a["anchors"].get("g", {}).get("run_id") != b["run_id"]:
            return False
    if a["family"] == "RAW_CALIBRATED" and a["anchors"].get("b", {}).get("run_id") != b["run_id"]:
        return False
    return True


def _budget_inputs(public, run_id, manifest):
    meta = manifest.get("metadata", {})
    parent = meta.get("parent_run_id", meta.get("family_run_id", meta.get("route_spec", {}).get("source_parent_run_id", run_id)))
    for candidate in dict.fromkeys((run_id, parent)):
        for path in (public / "runs" / candidate / "BASE_FIT_RECEIPT.json",
                     public / "runs" / candidate / "CONTROL_FIT_RECEIPT.json",
                     public / "combinations" / candidate / "receipt.json"):
            if path.is_file():
                receipt = _json(path)
                if receipt.get("run_id") != candidate:
                    raise ValueError("Epoch receipt belongs to a different run")
                artifact_sha = receipt.get("artifact_sha256", receipt.get("selected_checkpoint_sha256"))
                expected = meta.get("artifact_sha256", meta.get("parent_artifact_sha256", meta.get("realized_model_binding", {}).get("selected_artifact_sha256")))
                if expected and artifact_sha and artifact_sha != expected:
                    raise ValueError("Epoch receipt artifact differs from actual prediction model")
                return [dict(scope="public", file=path.relative_to(public).as_posix(), sha256=data.sha256_file(path))]
    return []


def generate(public, cache, split="V_select"):
    public, cache = Path(public).resolve(), Path(cache).resolve()
    if split not in {"V_select", "T_final"}:
        raise ValueError("Only V_select or locked T_final is supported")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    lock = {}
    if split == "T_final":
        path = public / "SELECTION_LOCK.json"
        if not path.is_file() or _json(path).get("locked") is not True:
            raise ValueError("T blocks require locked SELECTION_LOCK.json")
        lock = _json(path)
    raw_registry = (public / "EXPERIMENT_REGISTRY.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw_registry.decode("utf-8-sig"))))
    registry = {row["experiment_id"]: row for row in rows}
    if len(registry) != len(rows):
        raise ValueError("Duplicate registered experiment IDs")
    headers, excluded = [], []
    for path in sorted(public.rglob("PREDICTION_MANIFEST.json")):
        try:
            manifest = _json(path)
            if manifest.get("split") != split or manifest.get("score") is None:
                continue
            registered = registry[manifest["run_id"]]
            if split == "T_final" and not patterns._t_allowed(lock, manifest):
                raise ValueError("T run/artifact/route not authorized by SELECTION_LOCK")
            headers.append((path, manifest, registered))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            excluded.append(dict(manifest=path.relative_to(public).as_posix(), reason=str(exc)))
    if split == "T_final" and not headers:
        raise ValueError("No existing allowed T_final prediction manifests")
    # Metadata fixes block size before any private forecast or target cache is opened.
    tasks, rules = {}, {}
    for name in sorted({row["dataset"] for _, _, row in headers}):
        task = data.load_task(name, cache_dir=cache)
        period, horizon = int(task["period"]), int(task["H"])
        if min(period, horizon) < 1:
            raise ValueError("Positive predefined period/H metadata is required")
        tasks[name] = dict(period=period, horizon=horizon, columns=list(task["columns"]), data_sha=task["data_sha256"])
        rules[name] = dict(period=period, horizon=horizon,
                          block_origins=int(math.ceil(max(period, horizon) / horizon)), metadata_only=True)
    directory = public / "diagnostics" / ("time_blocks_" + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "BLOCK_RULES_PRE_SCORE.json", dict(created_unix=time.time(), split=split,
        formula="ceil(max(task.period,H)/H) consecutive strideH origins", datasets=rules,
        chosen_before_private_prediction_cache_open=True, chosen_using_forecast_errors=False))
    records, inputs, plans, seen = [], [], {}, set()
    for path, manifest, registered in headers:
        try:
            run_id, name = manifest["run_id"], registered["dataset"]
            metadata = tasks[name]
            if manifest["data_sha"] != metadata["data_sha"]:
                raise ValueError("Prediction data differs from metadata Task")
            reference_path = path.with_name("REFERENCE_CONTRACT.json")
            reference = _json(reference_path)
            if any(manifest.get(key) != reference.get(key) or not str(manifest.get(key, "")) for key in audit.IDENTITY_KEYS):
                raise ValueError("Independent reference identity differs")
            arrays = audit._load_arrays(manifest, reference, cache)
            if list(map(str, arrays["channel_ids"])) != list(map(str, metadata["columns"])):
                raise ValueError("Prediction channels differ from metadata Task")
            if arrays["pred_normalized"].shape[1] != metadata["horizon"]:
                raise ValueError("Prediction H differs from predefined metadata")
            plan = block_plan(arrays["origins"], metadata["horizon"], metadata["period"])
            cohort_key = manifest["cohort"] + ":" + plan["origin_contract"]["sha256"]
            plans[cohort_key] = plan
            identity_key = (run_id, manifest["cohort"])
            if identity_key in seen:
                raise ValueError("Duplicate run/cohort; no automatic cache or score selection")
            anchors = manifest.get("metadata", {}).get("actual_parent_manifests", {})
            identity = dict(run_id=run_id, dataset=name, fold=registered.get("fold", "primary"), cohort=manifest["cohort"],
                family=registered["family"], seed=int(registered.get("seed") or 0), k=int(registered.get("latent_k") or 0),
                budget=patterns._budget(public, run_id, manifest), data_sha=manifest["data_sha"], scaler_sha=manifest["scaler_sha"],
                parent_signature=patterns._fingerprint(anchors) if anchors else "", anchors=anchors)
            budget_inputs = _budget_inputs(public, run_id, manifest)
            record = record_from_arrays(arrays, identity)
            self_score = paired_summary(record, record, np.arange(len(record["origins"])))
            for metric in ("mse", "mae"):
                if self_score["left_" + metric] is None or not math.isclose(self_score["left_" + metric], float(manifest["score"][metric]), rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError("Independent own FP64 score differs from stored metric: " + metric)
            record["plan_key"] = cohort_key
            records.append(record); seen.add(identity_key)
            inputs.extend(budget_inputs)
            for item in (path, reference_path):
                inputs.append(dict(scope="public", file=item.relative_to(public).as_posix(), sha256=data.sha256_file(item)))
            inputs.append(dict(scope="private", file=manifest["cache_relative_path"], sha256=manifest["archive_sha256"]))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            reason = str(exc)
            for root, label in ((public, "<public root>"), (cache, "<private cache>")):
                reason = reason.replace(str(root), label).replace(root.as_posix(), label)
            excluded.append(dict(manifest=path.relative_to(public).as_posix(), reason=reason))
    if split == "T_final" and not records:
        raise ValueError("No numerically valid sealed authorized T prediction remains")
    overall_rows, block_rows, pair_index, unmatched = [], [], [], []
    for left_family, right_family in patterns.COMPARISONS:
        valid_pairs = 0
        for left in [r for r in records if r["identity"]["family"] == left_family]:
            for right in [r for r in records if r["identity"]["family"] == right_family]:
                a, b = left["identity"], right["identity"]
                if any(a[key] != b[key] for key in ("dataset", "fold", "cohort", "data_sha", "scaler_sha")) or not _recipe_pair(left, right):
                    continue
                try:
                    _compatible(left, right)
                except ValueError as exc:
                    unmatched.append(dict(left_run_id=a["run_id"], right_run_id=b["run_id"], reason=str(exc)))
                    continue
                valid_pairs += 1
                common = dict(left_run_id=a["run_id"], right_run_id=b["run_id"], dataset=a["dataset"], fold=a["fold"], split=split,
                    cohort=a["cohort"], comparison=left_family + " - " + right_family, left_seed=a["seed"], right_seed=b["seed"],
                    left_k=a["k"], right_k=b["k"], left_budget=a["budget"], right_budget=b["budget"],
                    metric_scale="A_normalized_original_channels", data_sha=a["data_sha"], scaler_sha=a["scaler_sha"])
                plan = plans[left["plan_key"]]
                overall_rows.append(dict(**common, **paired_summary(left, right, np.arange(len(left["origins"]))),
                    block_origins=plan["block_origins"], n_blocks=len(plan["blocks"]),
                    origin_start=int(left["origins"][0]), origin_last=int(left["origins"][-1]),
                    target_end_exclusive=int(left["origins"][-1]) + left["horizon"], overall_not_mean_of_blocks=True))
                for block in plan["blocks"]:
                    block_rows.append(dict(**common, **{k: v for k, v in block.items() if k not in {"indices", "n_origins"}},
                        block_origins=plan["block_origins"], **paired_summary(left, right, np.asarray(block["indices"]))))
                pair_index.append(dict(left_run_id=a["run_id"], right_run_id=b["run_id"], cohort_plan=left["plan_key"],
                    exact_target_mask_channel_origins_verified=True, same_recipe_when_required=True))
        if not valid_pairs:
            unmatched.append(dict(comparison=left_family + " - " + right_family, reason="No available exact-cohort pair with the registered recipe/parent conditions"))
    _write_csv(directory / "OVERALL_PAIRED.csv", overall_rows)
    _write_csv(directory / "TIME_BLOCK_PAIRED.csv", block_rows)
    _write_json(directory / "COHORT_BLOCKS.json", plans)
    manifest = dict(created_unix=time.time(), split=split, source_sha256=data.sha256_file(Path(__file__)),
        dependencies_sha256={name: data.sha256_file(Path(__file__).with_name(name)) for name in ("channel_patterns.py", "audit_predictions.py", "data_contract.py")},
        registry_sha256=hashlib.sha256(raw_registry).hexdigest(), prediction_count=len(records), paired_comparisons=len(pair_index),
        overall_rows=len(overall_rows), time_block_rows=len(block_rows), exact_pair_index=pair_index,
        unmatched_or_unavailable=unmatched, excluded_prediction_inputs=excluded, inputs=inputs,
        outputs={p.name: data.sha256_file(p) for p in directory.iterdir() if p.is_file()},
        full_cohort_reaggregated_separately=True, blocks_assumed_IID=False, metadata_period_proves_independence=False,
        confidence_intervals_computed=False, bootstrap_performed=False, significance_test_performed=False,
        method_K_seed_or_case_selection=False, inference_or_training_executed=False, GPU_execution=False,
        T_selection_lock_checked=split == "T_final", full_registry_audit_performed=False,
        CPU_process_seconds=time.process_time() - started_cpu, wall_seconds=time.perf_counter() - started_wall,
        runtime_environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")})
    _write_json(directory / "TIME_BLOCK_MANIFEST.json", manifest)
    (directory / "README_KO.md").write_text("# 고정 시간 블록의 대응 예측 차이\n\n블록 길이는 task.period와 H 메타데이터로 오차 계산 전에 정했다. 모든 유효한 대응 run/K/seed를 보존하며 마지막 짧은 블록도 포함한다. 전체 오차는 전체 관측수로 별도 계산했고 block 평균으로 대체하지 않았다. 블록은 IID 표본이 아니며 CI·bootstrap·유의성 검정을 수행하지 않았다.\n\n- [전체 cohort 차이](OVERALL_PAIRED.csv)\n- [블록별 차이](TIME_BLOCK_PAIRED.csv)\n- [선고정 길이](BLOCK_RULES_PRE_SCORE.json)\n- [원점 분할](COHORT_BLOCKS.json)\n- [검산 증빙](TIME_BLOCK_MANIFEST.json)\n", encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", required=True); parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="V_select", choices=("V_select", "T_final"))
    args = parser.parse_args()
    print(str(generate(args.public_root, args.cache_root, args.split)))


if __name__ == "__main__":
    main()
