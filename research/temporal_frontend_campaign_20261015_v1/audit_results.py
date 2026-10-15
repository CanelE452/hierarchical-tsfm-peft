"""Independent saved-result arithmetic audit; never reruns training or timing.

ROOT may execute only after SCORE_COMPLETE. Importing does not read artifacts.
Only this campaign, its prediction cache, and selected small checkpoints are
read. Original data archives, foundation-model weights, GPU APIs, and production
runtime/data/model modules are never opened or imported.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
H, L = 48, 512
CSV_FIELD_LIMIT = 16 * 1024 * 1024
RUN_FIELDS = (
    "attempt_id", "logical_id", "category", "block", "dataset", "fold", "type", "kind", "r", "seed",
    "status", "source", "source_sha256", "logical_fit_count", "fitting_configurations", "optimizer_updates",
    "incremental_updates", "receipt_endpoint", "updates_exact", "selected_update", "parameter_count",
    "legal_A_origins", "distinct_A_origins", "unique_input_times", "unique_target_times",
    "observed_target_entries", "origin_exposures", "alpha", "zero_no_fit", "prediction_origins",
    "prediction_scalar_entries", "cost_measure_trials", "cost_warmup_trials", "cost_error_trials",
    "cost_failed_queue_attempts", "cost_environment_probes", "cost_guard_errors_observed",
    "reuse_source", "reason",
)


def safe_path(path):
    path = Path(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise PermissionError(f"Audit read outside new campaign: {path}")
    return path


def digest(path):
    result = hashlib.sha256()
    with safe_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    csv.field_size_limit(CSV_FIELD_LIMIT)
    with safe_path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value is True or str(value).lower() == "true"


def number(value):
    return None if value in (None, "") else float(value)


def utc_seconds(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class Audit:
    def __init__(self):
        self.findings = []
        self.checks = Counter()
        self.run_rows = []
        self.details = {}
        self.inventory = {}

    def check(self, condition, code, scope, detail="", severity="ERROR"):
        self.checks[code] += 1
        if not bool(condition):
            finding = dict(severity=severity, code=code, scope=str(scope), detail=str(detail))
            self.findings.append(finding)
            print(json.dumps(dict(event="AUDIT_FINDING", **finding), ensure_ascii=False), flush=True)
            return False
        return True

    def equal(self, actual, expected, scope, code="ARITHMETIC", atol=1e-9, rtol=1e-9):
        if isinstance(expected, dict):
            if not self.check(isinstance(actual, dict), code, scope, "expected mapping"):
                return False
            good = True
            for key, value in expected.items():
                if not self.check(key in actual, code, scope, f"missing {key}"):
                    good = False
                else:
                    good = self.equal(actual[key], value, f"{scope}.{key}", code, atol, rtol) and good
            return good
        if expected is None or isinstance(expected, (str, bool)):
            return self.check(actual == expected, code, scope, f"actual={actual!r}, expected={expected!r}")
        if isinstance(expected, (list, tuple)) and any(not isinstance(value, (int, float, np.number)) for value in expected):
            if not self.check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), code, scope, "sequence shape differs"):
                return False
            good = True
            for index, value in enumerate(expected):
                good = self.equal(actual[index], value, f"{scope}[{index}]", code, atol, rtol) and good
            return good
        try:
            left, right = np.asarray(actual), np.asarray(expected)
            valid = left.shape == right.shape and np.allclose(left.astype(float), right.astype(float),
                                                              atol=atol, rtol=rtol, equal_nan=False)
        except (TypeError, ValueError):
            valid = False
        return self.check(valid, code, scope, f"actual={str(actual)[:160]}, expected={str(expected)[:160]}")

    def row(self, **values):
        unknown = set(values) - set(RUN_FIELDS)
        if unknown:
            raise ValueError(f"Unknown RUN_INDEX fields: {unknown}")
        self.run_rows.append(values)

    def section(self, name, function):
        try:
            function()
        except Exception as error:
            scope = self.inventory.pop("active_path", name)
            self.check(False, "SECTION_EXCEPTION", scope, f"{type(error).__name__}: {error}")
            self.details.setdefault("exceptions", {})[name] = traceback.format_exc()


# CORE_AUDIT_FUNCTIONS

def metric64(prediction, target, mask):
    """Channel-first reductions independently derived from valid scalar pairs."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    mask = np.asarray(mask)
    if prediction.shape != target.shape or target.shape != mask.shape or prediction.ndim != 3:
        raise ValueError("prediction/target/mask must be matching [N,H,C]")
    if mask.dtype != np.bool_ or not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("nonfinite observed pair or nonboolean mask")
    counts, mse, mae = [], [], []
    for channel in range(prediction.shape[-1]):
        selected = mask[..., channel]
        difference = prediction[..., channel][selected].astype(np.float64) - target[..., channel][selected].astype(np.float64)
        count = len(difference)
        counts.append(count)
        mse.append(float(np.dot(difference, difference) / count) if count else 0.)
        mae.append(float(np.abs(difference).sum(dtype=np.float64) / count) if count else 0.)
    active = np.asarray(counts) > 0
    if not active.any():
        raise ValueError("no observed target")
    return dict(mse=float(np.asarray(mse)[active].mean()), mae=float(np.asarray(mae)[active].mean()),
                channel_mse=mse, channel_mae=mae, counts=counts, n_observed=sum(counts))


def metadata_for(spec):
    typ = spec["type"]
    if typ == "G":
        return dict(metadata_for(spec["base"]), G=True)
    if typ == "SWAP":
        return dict(metadata_for(spec["base"]), F="native_shared_3feature_head", G=False)
    if typ == "ADAPTER":
        return dict(P=spec["kind"], E="learned_pointwise_C_to_r", decoder="origin_reference_lag7", F="Bolt-small", G=False)
    if typ == "R1":
        return dict(P=spec["variant"], E="identity", decoder="R1_original_grid_restore", F="Bolt-small", G=False)
    if typ == "DIRECT":
        return dict(P=spec["kind"], E="none", decoder="direct_prediction_head", F="none", G=False)
    if typ == "RIDGE":
        return dict(P="block33", E="none", decoder="self_channel_linear_head", F="none", G=False)
    return dict(P="identity", E="identity", decoder="persistence" if typ == "CURRENT" else "identity",
                F="none" if typ == "CURRENT" else "Bolt-LoRA-qv8" if typ == "LORA" else "Bolt-small", G=False)


def interval_exposure(origins, rows):
    origins = np.asarray(origins, dtype=np.int64)
    past, future = np.zeros(rows + 1, dtype=np.int64), np.zeros(rows + 1, dtype=np.int64)
    if len(origins):
        np.add.at(past, origins - L, 1); np.add.at(past, origins, -1)
        np.add.at(future, origins, 1); np.add.at(future, origins + H, -1)
    past, future = past.cumsum()[:-1], future.cumsum()[:-1]
    return past, future


def expected_schedule(n, seed):
    rng = np.random.default_rng(seed)
    total = 2400 * 4
    if total <= n:
        sample = np.floor((np.arange(total) + .5) * n / total).astype(np.int64)
        rng.shuffle(sample)
    else:
        sample = np.concatenate([rng.permutation(n) for _ in range(math.ceil(total / n))])[:total]
    return sample.reshape(2400, 4)


def audit_contract(a, contract, stats):
    for dataset, asset in contract["assets"].items():
        a.check(not asset["historically_unseen"], "EXPOSURE_DISCLOSED", dataset)
        lo, hi = asset["permitted_bounds"]
        rows = hi - lo
        a.check(hi <= asset["original_tune_bounds"][0], "ORIGINAL_TUNE_EXCLUDED", dataset)
        for fold, fraction in (("primary", (.6, .8)), ("secondary", (.5, .7))):
            item = contract["folds"][dataset][fold]
            expected_bounds = dict(A=[0, int(rows * fraction[0])],
                                   B=[int(rows * fraction[0]), int(rows * fraction[1])],
                                   C=[int(rows * fraction[1]), rows])
            a.equal(item["bounds"], expected_bounds, f"{dataset}/{fold}", "CUTOFFS", atol=0, rtol=0)
            for split, (start, stop) in expected_bounds.items():
                first, last = max(start, L), stop - H
                expected = np.arange(first, last + 1, 1 if split == "A" else H, dtype=np.int64)
                if split != "A" and len(expected) and expected[-1] != last:
                    expected = np.r_[expected, last]
                actual = np.asarray(item["origins"][split], dtype=np.int64)
                a.equal(actual, expected, f"{dataset}/{fold}/{split}", "ORIGIN_COHORT", atol=0, rtol=0)
                a.equal(item["absolute_origins"][split], actual + lo, f"{dataset}/{fold}/{split}", "ABSOLUTE_COORDINATES", atol=0, rtol=0)
                a.equal(item["absolute_bounds"][split], [start + lo, stop + lo], f"{dataset}/{fold}/{split}", "ABSOLUTE_BOUNDS", atol=0, rtol=0)
            for before, after in (("A", "B"), ("B", "C")):
                first, second = item["origins"][before], item["origins"][after]
                a.check(not first or not second or max(first) + H - 1 < min(second), "TARGET_PURGE", f"{dataset}/{fold}/{before}{after}")
            st = stats[dataset][fold]
            for field in ("mean", "std"):
                values = np.asarray(st[f"A_{field}"], dtype=np.float32)
                a.check(values.shape == (asset["channels"],) and np.isfinite(values).all(), "STATISTICS_SHAPE_FINITE", f"{dataset}/{fold}/{field}")
                if field == "std": a.check((values > 0).all(), "POSITIVE_STD", f"{dataset}/{fold}")
                a.equal(st[f"{field}_sha256"], array_digest(values), f"{dataset}/{fold}/{field}", "STATISTICS_HASH")
            a.check(all(value > 0 for value in st["A_true_observed_counts"]), "A_HAS_OBSERVATIONS", f"{dataset}/{fold}")
            origins = item["origins"]["A"]
            inputs, targets = interval_exposure(origins, rows)
            exposure = dict(origins=len(origins), unique_input_times=int((inputs > 0).sum()),
                            unique_target_times=int((targets > 0).sum()), target_max_repetition=int(targets.max(initial=0)),
                            input_row_minmax=[min(origins) - L, max(origins) - 1],
                            target_row_minmax=[min(origins), max(origins) + H - 1])
            a.equal(st["A_exposure"], exposure, f"{dataset}/{fold}", "A_EXPOSURE", atol=0, rtol=0)
    a.check(contract["protocol"]["protected_TEST"] is False, "PROTECTED_TEST_DISABLED", "contract")
    # Inspect code and pinned literals; never open any of the original assets.
    tree = ast.parse((HERE / "data.py").read_text(encoding="utf-8-sig"))
    assets_node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "ASSETS" for target in node.targets))
    pinned = ast.literal_eval(assets_node)
    for name, item in pinned.items():
        a.equal(contract["assets"][name], {key: item[key] for key in ("path", "sha256", "permitted_bounds", "channels")},
                name, "LOADER_EXACT_ALLOWLIST")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    a.check({"install_data_access_guard", "load_dataset"}.issubset(functions), "LOADER_GUARD_SOURCE", "data.py")
    a.details["data_access_scope"] = dict(original_archive_reads=0, foundation_weight_reads=0,
        original_data_hashes_recomputed=False, statistics_refit=False,
        checked="Pinned metadata, source AST and new prediction caches only",
        limit="Python audit hook is not an OS sandbox; source review cannot prove absence of native-library or historical reads")


def audit_locks(a, selection, cost_selection, cost_lock, first, complete, stats):
    a.check(selection["status"] == "ALL_SELECTIONS_LOCKED_BEFORE_C" and not selection["C_scored"], "SELECTION_LOCK", "selection")
    a.check(cost_lock["status"] == "COMPLETE_VALID", "COST_GATE", "cost")
    general = read_json(HERE / "GENERAL_COST_COMPLETE.json")
    for key, filename in (("selection_lock", "SELECTION_LOCK.json"), ("cost_selection_sha256", "COST_SELECTION.json"),
                          ("general_cost", "GENERAL_COST_COMPLETE.json")):
        a.equal(cost_lock[key], digest(HERE / filename), filename, "LOCK_HASH")
    a.equal(first["selection_lock"], digest(HERE / "SELECTION_LOCK.json"), "first_C", "LOCK_HASH")
    a.equal(first["cost_lock"], digest(HERE / "COST_LOCK.json"), "first_C", "LOCK_HASH")
    a.equal(general["selection_lock_sha256"], digest(HERE / "SELECTION_LOCK.json"), "general", "LOCK_HASH")
    a.equal(general["cost_selection_sha256"], digest(HERE / "COST_SELECTION.json"), "general", "LOCK_HASH")
    a.equal(cost_selection["selection_lock_sha256"], digest(HERE / "SELECTION_LOCK.json"), "cost_selection", "LOCK_HASH")
    a.equal(selection["statistics_hash"], digest(HERE / "FOLD_STATISTICS.json"), "statistics", "LOCK_HASH")
    for name, expected in selection["source_hashes"].items():
        a.equal(digest(HERE / name), expected, name, "SCIENTIFIC_SOURCE_HASH")
    selected_artifacts = set()
    def collect_artifacts(spec):
        for key in ("checkpoint", "weights", "initial_checkpoint"):
            if spec.get(key): selected_artifacts.add(spec[key])
        if spec.get("base"): collect_artifacts(spec["base"])
    for spec in selection["routes"]: collect_artifacts(spec)
    a.equal(sorted(selection["artifact_hashes"]), sorted(selected_artifacts), "selection",
            "SELECTED_ARTIFACT_HASH_COVERAGE")
    for name, expected in selection["artifact_hashes"].items():
        a.equal(digest(CACHE / name), expected, name, "SELECTED_ARTIFACT_HASH")
    source_r2 = read_json(HERE / "SOURCE_R2.json")
    for name, expected in source_r2.items():
        snapshot = HERE / "sources_R2" / name
        a.check(snapshot.is_file(), "R2_SNAPSHOT_EXISTS", name)
        if snapshot.is_file(): a.equal(digest(snapshot), expected, name, "R2_SNAPSHOT_HASH")
        if name in ("runtime.py", "campaign.py", "models.py", "data.py"):
            a.equal(digest(HERE / name), expected, name, "R2_CORE_SOURCE_UNCHANGED")
    a.check(selection["time_unix"] <= cost_selection["time_unix"] <= cost_lock["time_unix"] <= first["time_unix"] <= complete["time_unix"],
            "SELECTION_COST_C_ORDER", "timestamps")
    a.check(first["algorithmically_held_out_this_run"] and not first["historically_unseen"], "C_EXPOSURE_LABEL", "first_C")
    a.check(complete["F_unchanged"], "F_BASE_UNCHANGED", "score")
    a.check(.5 in selection["F_quantiles"] and selection["F_head_length"] >= H,
            "F_MEDIAN_HEAD_CONTRACT", "selection")
    a.details["lock_order"] = dict(selection=selection["time_unix"], cost_selection=cost_selection["time_unix"],
                                  cost_complete=cost_lock["time_unix"], first_C=first["time_unix"], score_complete=complete["time_unix"])
    for spec in selection["routes"]:
        if spec["fold"] == "secondary" and spec["type"] in ("ADAPTER", "G"):
            a.equal(spec["seed"], 92601, spec["id"], "SECONDARY_PREDECLARED_SEED", atol=0, rtol=0)


def audit_training(a, selection, contract, stats):
    by_id = defaultdict(list)
    events = []
    if (HERE / "EVENTS.jsonl").is_file():
        for line in (HERE / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    for path in sorted((HERE / "fits").glob("*.json")):
        value = read_json(path)
        by_id[value["id"]].append((path, value))
    latest, ridge = {}, {}
    for identifier, records in by_id.items():
        a.inventory["active_path"] = identifier
        iterative = sorted([(path, v) for path, v in records if "checkpoints" in v], key=lambda pair: pair[1]["updates"])
        if not iterative:
            for path, value in records:
                ridge[identifier] = value
                candidates = value["candidate_diagnostics"]
                chosen = min(enumerate(candidates), key=lambda pair: (pair[1]["B"]["mse"], pair[0]))[1]
                a.equal(value["selected_alpha"], chosen["alpha"], identifier, "RIDGE_B_SELECTION")
                a.equal(value["selected_B"], chosen["B"], identifier, "RIDGE_B_SELECTION")
                a.equal(value["fit_candidates"], sum(v["alpha"] is not None for v in candidates), identifier, "RIDGE_FIT_COUNTS", atol=0, rtol=0)
                expected_alphas = [0., .01, None] if value["type"] == "SWAP" else [.01, None]
                a.equal([v["alpha"] for v in candidates], expected_alphas, identifier, "RIDGE_PREDECLARED_ALPHAS")
                numeric = chosen["numeric"]
                unresolved = numeric.get("numeric_status") == "NUMERIC_UNRESOLVED" or numeric.get("numeric_unresolved_equations", 0) > 0
                a.equal(value["training_status"], "UNDERTRAINED_OR_UNRESOLVED" if unresolved else "COMPLETE_VALID", identifier, "RIDGE_NUMERIC_STATUS")
                weights = np.load(safe_path(CACHE / value["weights"]), allow_pickle=False)
                shape = (3, H) if value["type"] == "SWAP" else (contract["assets"][value["dataset"]]["channels"], 33, H)
                a.check(weights.shape == shape and np.isfinite(weights).all(), "NATIVE_WEIGHT_SHAPE_FINITE", identifier)
                if chosen["alpha"] is None:
                    a.check(np.count_nonzero(weights) == 0, "ZERO_NO_FIT_WEIGHT", identifier)
                for index, candidate in enumerate(candidates):
                    zero = candidate["alpha"] is None
                    a.row(attempt_id=f"{identifier}/alpha{index}", logical_id=identifier, category="RIDGE_CANDIDATE",
                          block=value["block"], dataset=value["dataset"], fold=value["fold"], type=value["type"],
                          kind=value.get("kind", ""), r=value["r"], seed=value["seed"],
                          status=value.get("training_status", value["status"]), source=str(path.relative_to(HERE)), source_sha256=digest(path),
                          logical_fit_count=int(index == 0), fitting_configurations=0 if zero else 1, optimizer_updates=0,
                          updates_exact=True, alpha="ZERO" if zero else candidate["alpha"], zero_no_fit=zero,
                          legal_A_origins=value["fit_A_origins"])
            continue
        previous = 0
        for index, (path, value) in enumerate(iterative):
            scope = f"{identifier}/endpoint{value['updates']}"
            cps, curve = value["checkpoints"], value["learning_curve"]
            best = min(cps, key=lambda row: (row["B"]["mse"], row["update"]))
            a.equal(value["selected_update"], best["update"], scope, "CHECKPOINT_SELECTION", atol=0, rtol=0)
            a.equal(value["selected_B"], best["B"], scope, "CHECKPOINT_SELECTION")
            a.equal(value["selected_state_sha256"], best["state_sha256"], scope, "CHECKPOINT_STATE_CONNECTION")
            a.check(value["updates"] in (600, 1200, 2400), "APPROVED_UPDATE_ENDPOINT", scope)
            a.equal([row["update"] for row in curve], list(range(1, value["updates"] + 1)), scope, "UPDATE_TRAJECTORY", atol=0, rtol=0)
            a.check(all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) and math.isfinite(row["seconds"])
                        for row in curve), "FINITE_TRAINING_RECEIPTS", scope)
            a.equal(value["clipping_fraction"], sum(bool(row["clipped"]) for row in curve) / len(curve), scope, "CLIPPING_FRACTION")
            a.equal(value["update_seconds_median"], statistics.median(row["seconds"] for row in curve), scope, "UPDATE_TIMER_METADATA")
            decreasing = len(cps) >= 3 and cps[-3]["B"]["mse"] > cps[-2]["B"]["mse"] > cps[-1]["B"]["mse"]
            expected_status = "UNDERTRAINED_OR_UNRESOLVED" if value["updates"] == 2400 and (best["update"] == 2400 or decreasing) else "COMPLETE_VALID"
            a.equal(value["status"], expected_status, scope, "UNDERTRAINED_STATUS")
            origins = np.asarray(contract["folds"][value["dataset"]][value["fold"]]["origins"]["A"], dtype=np.int64)
            order = expected_schedule(len(origins), value["seed"])[:value["updates"]]
            a.equal(array_digest(order), value["origin_schedule_sha256"], scope, "ORIGIN_SCHEDULE_HASH")
            ids = np.unique(order)
            used = origins[ids]
            rows = contract["assets"][value["dataset"]]["permitted_bounds"]
            inputs, targets = interval_exposure(used, rows[1] - rows[0])
            exposure = dict(legal_A_origins=len(origins), distinct_A_origins=len(ids), unique_input_times=int((inputs > 0).sum()),
                            unique_target_times=int((targets > 0).sum()), repeated_origin_exposures=4 * value["updates"] - len(ids))
            a.equal(value, exposure, scope, "ACTUAL_TRAINING_EXPOSURE", atol=0, rtol=0)
            all_inputs, all_targets = interval_exposure(origins, rows[1] - rows[0])
            if np.array_equal(targets > 0, all_targets > 0):
                wanted = sum(stats[value["dataset"]][value["fold"]]["A_exposure"]["target_observed_unique"])
                a.equal(value["unique_true_observed_target_entries"], wanted, scope, "UNIQUE_TARGET_OBSERVED_COUNTS", atol=0, rtol=0)
            final = index == len(iterative) - 1
            a.row(attempt_id=scope, logical_id=identifier, category="OPTIMIZER_ENDPOINT", block=value["block"],
                  dataset=value["dataset"], fold=value["fold"], type=value.get("type", "ADAPTER"), kind=value.get("kind", ""),
                  r=value["r"], seed=value["seed"], status=value["status"], source=str(path.relative_to(HERE)), source_sha256=digest(path),
                  logical_fit_count=int(final), fitting_configurations=int(final), optimizer_updates=value["updates"] if final else 0,
                  incremental_updates=value["updates"] - previous, receipt_endpoint=value["updates"], updates_exact=True,
                  selected_update=value["selected_update"], parameter_count=value["parameter_count"],
                  observed_target_entries=value["unique_true_observed_target_entries"], origin_exposures=4 * value["updates"],
                  **{key: exposure[key] for key in ("legal_A_origins", "distinct_A_origins", "unique_input_times", "unique_target_times")})
            previous = value["updates"]
            if final: latest[identifier] = value
    # Selected small checkpoints only; never load the foundation model or a model class.
    import torch
    for spec in selection["routes"]:
        if spec["id"] not in latest: continue
        value = latest[spec["id"]]
        a.equal(spec["selected_update"], value["selected_update"], spec["id"], "LOCKED_CP_CONNECTION", atol=0, rtol=0)
        a.equal(spec["selected_B"], value["selected_B"], spec["id"], "LOCKED_CP_CONNECTION")
        a.equal(spec.get("training_status"), value["status"], spec["id"], "LOCKED_TRAINING_STATUS")
        path = safe_path(CACHE / spec["checkpoint"])
        state = torch.load(path, map_location="cpu", weights_only=True)
        state_hash = hashlib.sha256()
        for key, tensor in state.items():
            state_hash.update(key.encode()); state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        a.equal(state_hash.hexdigest(), value["selected_state_sha256"], spec["id"], "SELECTED_TENSOR_STATE_HASH")
        if spec["type"] == "G" and spec["base"].get("checkpoint"):
            base_hash = hashlib.sha256()
            for key, tensor in state.items():
                if key.startswith("base."):
                    base_hash.update(key[5:].encode()); base_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            base_id = spec["base"]["id"]
            if base_id in latest:
                a.equal(base_hash.hexdigest(), latest[base_id]["selected_state_sha256"], spec["id"], "G_BASE_FROZEN_STATE")
        del state
    for path in sorted((HERE / "failures").glob("*.json")):
        value = read_json(path)
        progress = max([0] + [int(row.get("update", 0)) for row in events if row.get("id") == value["id"]])
        a.row(attempt_id=f"failure/{value['id']}", logical_id=value["id"], category="FAILURE", block=value.get("block", ""),
              dataset=value.get("dataset", ""), fold=value.get("fold", ""), type=value.get("type", "ADAPTER"),
              kind=value.get("kind", ""), r=value.get("r", ""), seed=value.get("seed", ""), status=value["status"],
              source=str(path.relative_to(HERE)), source_sha256=digest(path), logical_fit_count=0, fitting_configurations=0,
              optimizer_updates=0, receipt_endpoint=progress, updates_exact=False, reason=value.get("error", ""))
        a.check(False, "RECORDED_PATH_FAILURE", value["id"], value.get("error", ""), severity="WARNING")
    known_attempts = set(by_id)
    known_attempts.update(read_json(path)["id"] for path in (HERE / "failures").glob("*.json"))
    orphan_attempts = []
    checkpoint_root = CACHE / "checkpoints"
    if checkpoint_root.is_dir():
        for folder in checkpoint_root.iterdir():
            if not folder.is_dir() or folder.name in known_attempts: continue
            checkpoints = [int(match[1]) for path in folder.glob("cp*.pt")
                           if (match := re.fullmatch(r"cp(\d+)\.pt", path.name))]
            lower_bound = max(checkpoints, default=0)
            orphan_attempts.append(dict(id=folder.name, checkpoint_update_lower_bound=lower_bound))
            a.check(False, "UNREGISTERED_CHECKPOINT_ATTEMPT", folder.name,
                    f"No completed-fit/failure receipt; checkpoint lower bound {lower_bound}", severity="WARNING")
            a.row(attempt_id=f"orphan/{folder.name}", logical_id=folder.name, category="ORPHAN_ATTEMPT", status="PARTIAL_RECEIPT_ONLY",
                  source=str(folder.relative_to(CACHE)), logical_fit_count=0, fitting_configurations=0,
                  optimizer_updates=0, receipt_endpoint=lower_bound, updates_exact=False,
                  reason="Checkpoint filename lower bound only; actual additional updates are unknown")
    core = [v for v in latest.values() if v["block"] == "R2"]
    expected_grid = {(d, r, kind, seed) for d in ("beijing", "jena")
                     for r in (contract["assets"][d]["channels"], contract["assets"][d]["small_k"])
                     for kind in ("B0", "B1", "B2", "B3", "B4") for seed in (92601, 92602)}
    a.check({(v["dataset"], v["r"], v["kind"], v["seed"]) for v in core} == expected_grid and len(core) == 40,
            "R2_CORE_40_COVERAGE", "training")
    g = [v for v in latest.values() if v.get("type") == "G"]
    a.equal(len(g), 20, "training", "R3_G_20_COVERAGE", atol=0, rtol=0)
    a.equal(Counter(v["fold"] for v in g), {"primary": 10, "secondary": 10}, "training", "R3_G_20_COVERAGE", atol=0, rtol=0)
    r4_optimizer = [v for v in latest.values() if v["block"] == "R4"]
    r4_configs = len(r4_optimizer) + sum(v["fit_candidates"] for v in ridge.values() if v["block"] == "R4")
    a.equal(r4_configs, 16, "training", "R4_16_FITTING_CONFIGURATIONS", atol=0, rtol=0)
    a.equal(Counter(v["type"] for v in r4_optimizer), {"DIRECT": 4, "LORA": 2}, "R4", "R4_OPTIMIZER_TYPES", atol=0, rtol=0)
    a.equal(Counter(v["type"] for v in ridge.values() if v["block"] == "R4"), {"SWAP": 4, "RIDGE": 2}, "R4", "R4_DETERMINISTIC_TYPES", atol=0, rtol=0)
    for value in r4_optimizer:
        if value["type"] == "LORA": a.equal(value["learning_rate"], .0001, value["id"], "LORA_PREDECLARED_LR")
    r4_receipt = read_json(HERE / "R4_SELECTIONS.json")
    a.equal(r4_receipt["planned_fitting_configurations"], 16, "R4", "R4_PLANNED_CONFIGURATION_CEILING", atol=0, rtol=0)
    a.equal(r4_receipt["actual_fitting_configurations"], r4_configs, "R4", "R4_ACTUAL_CONFIGURATION_RECEIPT", atol=0, rtol=0)
    a.details["training_counts"] = dict(unique_optimizer_fits=len(latest), completed_optimizer_updates=sum(v["updates"] for v in latest.values()),
        endpoint_receipts=sum(len(v) for v in by_id.values()), R2_unique_fits=len(core), G_unique_fits=len(g),
        R4_optimizer_fits=len(r4_optimizer), R4_fitting_configurations=r4_configs,
        ridge_logical_routes=len(ridge), ridge_fitting_configurations=sum(v["fit_candidates"] for v in ridge.values()),
        zero_no_fit_candidates=sum(sum(r["alpha"] is None for r in v["candidate_diagnostics"]) for v in ridge.values()),
        R5_aux_optimizer_fits=sum(v["block"].startswith("R5") and v.get("type") != "G" for v in latest.values()),
        orphan_attempts=orphan_attempts,
        A_checkpoint_monitor="Fixed up-to-32 uniformly spread A origins; not complete-A evaluation",
        checkpoint_metric_scope="Selection recomputed from checkpoint metadata; selected B metrics reconnected to cached B predictions. Intermediate CP predictions are not replayed.",
        training_and_timer_reruns=0)
    a.inventory["latest_fits"], a.inventory["ridge_fits"] = latest, ridge
    a.inventory.pop("active_path", None)


def audit_attempts(a, selection, contract):
    r1 = read_json(HERE / "R1_B.json")
    a.inventory["r1_by_id"] = {value["id"]: value for value in r1["records"]}
    expected = {(dataset, f"A{index}") for dataset in contract["assets"] for index in range(14)}
    seen = set()
    for value in r1["records"]:
        seen.add((value["dataset"], value["variant"]))
        wanted = np.asarray(contract["folds"][value["dataset"]]["primary"]["origins"]["B"], dtype=np.int64)
        wanted = wanted[wanted >= value["length"]]
        if value["status"] == "COMPLETE_VALID":
            a.equal(value["B_origins"], wanted, value["id"], "R1_INITIAL_B_COHORT", atol=0, rtol=0)
            a.check(len(wanted) > 0, "R1_NONEMPTY_INFERENCE", value["id"])
        elif value["status"] == "BLOCKED":
            a.check(len(wanted) == 0, "R1_LOOKBACK_BLOCKED_LEGAL", value["id"])
        a.row(attempt_id=f"R1_initial_B/{value['id']}", logical_id=value["id"], category="R1_INITIAL_B", block="R1",
              dataset=value["dataset"], fold="primary", type=value["type"], kind=value["variant"], r=value["r"], seed=0,
              status=value["status"], source="R1_B.json", source_sha256=digest(HERE / "R1_B.json"), logical_fit_count=0,
              fitting_configurations=0, optimizer_updates=0, updates_exact=True,
              prediction_origins=len(value.get("B_origins", [])), reason=value.get("reason", value.get("error", "")))
        for field in ("matched_Raw512_B", "matched_effective_lookback_Raw_B"):
            if field in value:
                a.row(attempt_id=f"R1_initial_B/{value['id']}/{field}", logical_id=value["id"], category="R1_INITIAL_MATCHED_B",
                      block="R1_CONTROL", dataset=value["dataset"], fold="primary", type="RAW", kind=field,
                      status="COMPLETE_VALID", source="R1_B.json", source_sha256=digest(HERE / "R1_B.json"),
                      logical_fit_count=0, fitting_configurations=0, optimizer_updates=0, updates_exact=True,
                      prediction_origins=len(value["B_origins"]), reason="Initial B matched reference; cached-B reconnection is checked separately")
    a.check(seen == expected, "R1_42_ATTEMPT_COVERAGE", "R1")
    probe = read_json(HERE / "PROBE.json")
    total_probe = sum(row["updates"] for row in probe["rows"])
    a.equal(total_probe, 200, "probe", "PROBE_200_UPDATES", atol=0, rtol=0)
    for index, row in enumerate(probe["rows"]):
        a.row(attempt_id=f"probe/{index}", logical_id=f"probe/{index}", category="PROBE", block="PROBE", dataset="beijing",
              fold="primary", type="ADAPTER", kind=row["kind"], r=row["r"], seed=92601, status="COMPLETE_VALID",
              source="PROBE.json", source_sha256=digest(HERE / "PROBE.json"), logical_fit_count=0, fitting_configurations=0,
              optimizer_updates=row["updates"], incremental_updates=row["updates"], updates_exact=True,
              parameter_count=row["parameters"], reason="Probe is separate from R2 fit/update totals; memory peaks are process cumulative")
    r5 = read_json(HERE / "R5_SELECTIONS.json")
    latest = a.inventory["latest_fits"]
    for reuse in r5["reuses"]:
        left, right = latest[reuse["source"]], latest[reuse["comparator"]]
        a.equal({k: left[k] for k in ("dataset", "fold", "r", "seed", "updates", "origin_schedule_sha256", "learning_rate", "batch")},
                {k: right[k] for k in ("dataset", "fold", "r", "seed", "updates", "origin_schedule_sha256", "learning_rate", "batch")},
                reuse["source"], "B3_B4_MATCHED_REUSE")
        a.equal(left["parameter_count"], right["parameter_count"], reuse["source"], "B3_B4_CAPACITY_MATCH", atol=0, rtol=0)
        a.check(left["kind"] == "B3" and right["kind"] == "B4", "B3_B4_REUSE_CLASS", reuse["source"])
        a.row(attempt_id=f"reuse/{reuse['source']}", logical_id=reuse["comparator"], category="REUSE", block="R5_ABLATION",
              dataset=left["dataset"], fold=left["fold"], type="ADAPTER", kind="B4_without_decomposition", r=left["r"],
              seed=left["seed"], status=reuse["status"], source="R5_SELECTIONS.json", source_sha256=digest(HERE / "R5_SELECTIONS.json"),
              logical_fit_count=0, fitting_configurations=0, optimizer_updates=0, reuse_source=reuse["source"],
              reason="Recorded matching + source/unit-test architecture equivalence; training trajectories not independently rerun")
    a.details["attempt_counts"] = dict(R1_records=len(r1["records"]), R1_success=sum(v["status"] == "COMPLETE_VALID" for v in r1["records"]),
        R1_blocked=sum(v["status"] == "BLOCKED" for v in r1["records"]), probe_updates=total_probe, reuses=len(r5["reuses"]))


def audit_predictions(a, selection, contract, stats, complete):
    rows = read_csv(HERE / "SCORES.csv")
    indexed = {(row["id"], row["split"]): row for row in rows}
    a.check(len(indexed) == len(rows), "UNIQUE_SCORE_KEYS", "SCORES.csv")
    a.equal(len(rows), complete["rows"], "SCORES.csv", "SCORE_ROW_COUNT", atol=0, rtol=0)
    consumed = set()
    for spec in selection["routes"]:
        identifier = spec["id"]
        path = safe_path(CACHE / "predictions" / f"{identifier}.npz")
        if not a.check(path.is_file(), "PREDICTION_CACHE_EXISTS", identifier): continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                st = stats[spec["dataset"]][spec["fold"]]
                mean, std = (np.asarray(st[f"A_{name}"], dtype=np.float32) for name in ("mean", "std"))
                for split in ("B", "C"):
                    expected = np.asarray(contract["folds"][spec["dataset"]][spec["fold"]]["origins"][split], dtype=np.int64)
                    expected = expected[expected >= spec.get("length", L)]
                    if not len(expected): continue
                    prefix = f"{split}_"
                    origins, prediction, target, mask = (archive[prefix + name] for name in ("origins", "pred", "target", "mask"))
                    raw_target = archive[prefix + "target_physical"]
                    scope = f"{identifier}/{split}"
                    a.equal(origins, expected, scope, "PREDICTION_ORIGIN_COHORT", atol=0, rtol=0)
                    a.check(prediction.shape == (len(origins), H, contract["assets"][spec["dataset"]]["channels"]), "PREDICTION_SHAPE", scope)
                    a.check(np.isfinite(prediction).all(), "ALL_PREDICTIONS_FINITE", scope)
                    a.equal(np.where(mask, (raw_target - mean) / std, 0).astype(np.float32), target, scope, "NORMALIZED_PHYSICAL_TARGET", atol=0, rtol=0)
                    metrics = metric64(prediction, target, mask)
                    row = indexed[(identifier, split)]; consumed.add((identifier, split))
                    a.equal(float(row["mse"]), metrics["mse"], scope, "CACHED_MSE")
                    a.equal(float(row["mae"]), metrics["mae"], scope, "CACHED_MAE")
                    a.equal(int(row["n_observed"]), metrics["n_observed"], scope, "CACHED_OBSERVED_COUNT", atol=0, rtol=0)
                    a.equal(row["cohort"], array_digest(origins), scope, "COHORT_HASH")
                    a.equal(int(row["n_origins"]), len(origins), scope, "ORIGIN_COUNT", atol=0, rtol=0)
                    wanted_meta = metadata_for(spec)
                    for key, value in wanted_meta.items():
                        actual = truth(row[key]) if isinstance(value, bool) else row[key]
                        a.equal(actual, value, scope + "/" + key, "METHOD_METADATA")
                    a.check(truth(row["algorithmically_held_out_this_run"]) == (split == "C") and not truth(row["historically_unseen"]), "SCORE_EXPOSURE_LABEL", scope)
                    details = read_json(HERE / "scores" / f"{identifier}_{split}.json")
                    a.equal(details["normalized"], metrics, scope, "SCORE_JSON_NORMALIZED")
                    # Recreate the deployed FP32 output; raw physical Y is cached independently.
                    physical = metric64(prediction * std + mean, raw_target, mask)
                    a.equal(details["physical_per_channel"], physical, scope, "SCORE_JSON_PHYSICAL")
                    for label, begin, end in (("1-12", 0, 12), ("13-24", 12, 24), ("25-48", 24, 48)):
                        a.equal(details["horizon_metrics"][label], metric64(prediction[:, begin:end], target[:, begin:end], mask[:, begin:end]),
                                scope + "/" + label, "HORIZON_METRICS")
                    if split == "B" and "selected_B" in spec and not spec.get("row_microbatch"):
                        a.equal(metrics, spec["selected_B"], scope, "SELECTED_B_RECONNECTION", atol=1e-9, rtol=1e-9)
                    expected_microbatch = spec.get("row_microbatch", "full") if wanted_meta["F"].startswith("Bolt") else "none"
                    a.equal(str(row["F_row_microbatch"]), str(expected_microbatch), scope, "F_ROW_MICROBATCH")
                    a.row(attempt_id=f"score/{scope}", logical_id=identifier, category="SCORING", block=spec.get("block", "CONTROL"),
                          dataset=spec["dataset"], fold=spec["fold"], type=spec["type"], kind=spec.get("kind", spec.get("variant", "")),
                          r=spec.get("r", len(std)), seed=spec.get("seed", 0), status=row["status"], source=str(path.relative_to(CACHE)),
                          source_sha256=digest(path), logical_fit_count=0, fitting_configurations=0, optimizer_updates=0,
                          prediction_origins=len(origins), prediction_scalar_entries=prediction.size, reason=split)
                    for key in archive.files:
                        match = re.fullmatch(rf"{split}_matched_RawL(\d+)", key)
                        if not match: continue
                        context = int(match[1]); ref_id = f"{identifier}_matched_RawL{context}"
                        reference = indexed[(ref_id, split)]; consumed.add((ref_id, split))
                        ref_metric = metric64(archive[key], target, mask)
                        a.equal(float(reference["mse"]), ref_metric["mse"], ref_id + "/" + split, "MATCHED_RAW_MSE")
                        a.equal(float(reference["mae"]), ref_metric["mae"], ref_id + "/" + split, "MATCHED_RAW_MAE")
                        a.equal(reference["cohort"], row["cohort"], ref_id, "MATCHED_RAW_COHORT")
                        a.equal(int(reference["n_observed"]), metrics["n_observed"], ref_id, "MATCHED_RAW_MASK", atol=0, rtol=0)
                        a.equal(int(reference["F_context_length"]), context, ref_id, "MATCHED_RAW_CONTEXT", atol=0, rtol=0)
                        for k, v in metadata_for(dict(type="RAW")).items():
                            a.equal(truth(reference[k]) if isinstance(v, bool) else reference[k], v, ref_id + "/" + k, "MATCHED_RAW_METADATA")
                        if split == "B":
                            r1_initial = a.inventory.get("r1_by_id", {}).get(identifier, {})
                            field = "matched_effective_lookback_Raw_B" if spec.get("variant") == "A11" else "matched_Raw512_B"
                            if field in r1_initial:
                                a.equal(ref_metric, r1_initial[field], ref_id, "R1_MATCHED_B_RECONNECTION")
                        a.row(attempt_id=f"score/{ref_id}/{split}", logical_id=ref_id, category="MATCHED_RAW_SCORING", block="R1_CONTROL",
                              dataset=spec["dataset"], fold=spec["fold"], type="RAW", kind="MATCHED_COHORT", status=reference["status"],
                              source=str(path.relative_to(CACHE)), source_sha256=digest(path), logical_fit_count=0, fitting_configurations=0,
                              optimizer_updates=0, prediction_origins=len(origins), prediction_scalar_entries=archive[key].size, reason=split)
        except Exception as error:
            a.check(False, "PREDICTION_PATH_EXCEPTION", identifier, f"{type(error).__name__}: {error}")
    a.check(consumed == set(indexed), "ALL_SCORE_ROWS_RECOMPUTED", "SCORES.csv", str(sorted(set(indexed) - consumed)))
    micro_receipts = []
    for spec in selection["routes"]:
        if spec.get("row_microbatch") != 4: continue
        canonical_id = spec["id"].removesuffix("_micro4")
        candidates = [r for r in selection["routes"] if r["id"] == canonical_id and r["type"] == "RAW"]
        if not a.check(len(candidates) == 1, "LOCKED_RAW_MICRO_CANONICAL", spec["id"]): continue
        try:
            with (
                np.load(safe_path(CACHE / "predictions" / f"{spec['id']}.npz"), allow_pickle=False) as micro,
                np.load(safe_path(CACHE / "predictions" / f"{canonical_id}.npz"), allow_pickle=False) as canonical,
            ):
                for split in ("B", "C"):
                    scope = f"{spec['id']}/{split}"
                    for key in ("origins", "target", "mask", "target_physical"):
                        a.equal(micro[f"{split}_{key}"], canonical[f"{split}_{key}"], scope + "/" + key, "ACTUAL_RAW_MICRO_COHORT", atol=0, rtol=0)
                    p, q = micro[f"{split}_pred"], canonical[f"{split}_pred"]
                    a.equal(p, q, scope, "ACTUAL_RAW_MICRO_PREDICTION_PARITY", atol=1e-5, rtol=2e-6)
                    micro_receipts.append(dict(id=spec["id"], split=split, max_abs=float(np.max(np.abs(p.astype(np.float64) - q.astype(np.float64))))))
        except Exception as error:
            a.check(False, "RAW_MICRO_CACHE_EXCEPTION", spec["id"], f"{type(error).__name__}: {error}")
    a.equal(len(micro_receipts), 6, "raw_micro", "ACTUAL_RAW_MICRO_B_C_COVERAGE", atol=0, rtol=0)
    a.details["prediction_audit"] = dict(score_rows=len(rows), independently_recomputed_rows=len(consumed),
        arithmetic="Independent valid-pair FP64 reductions; physical predictions reconstructed with deployed FP32 A mean/std",
        scalar_tolerance=dict(atol=1e-9, rtol=1e-9), physical_target_origin="New cached original physical Y; original archives not opened",
        reference_limit="Saved target/mask correctness is tied to pinned source/contract and origin metadata, not independent rereading of original assets",
        actual_raw_micro_receipts=micro_receipts,
        raw_micro_selected_B_note="Predeclared micro4 inherits canonical Raw B metadata, not a separately selected CP. Actual B/C cache parity uses producer's fixed atol1e-5/rtol2e-6 for FP32 batch-shape variation.")


def audit_parity(a, selection, contract, filename="ACTUAL_PARITY.json"):
    path = HERE / filename
    receipt = read_json(path)
    a.equal(receipt["status"], "PASS", filename, "RAW_PARITY_RECEIPT")
    a.check(receipt["C_scored"] is False, "PARITY_PRE_C_LABEL", filename)
    a.equal(receipt["F_model_hash"], selection["F_model_hash"], filename, "PARITY_MODEL_BINDING")
    a.equal(receipt["statistics_sha256"], digest(HERE / "FOLD_STATISTICS.json"), filename, "PARITY_STATISTICS_BINDING")
    a.equal(receipt["atol"], 1e-5, filename, "PARITY_FIXED_TOLERANCE", atol=0, rtol=0)
    a.equal(receipt["rtol"], 2e-6, filename, "PARITY_FIXED_TOLERANCE", atol=0, rtol=0)
    expected = {(dataset, batch) for dataset in contract["assets"] for batch in (1, 8)}
    actual, cp0 = set(), set()
    for row in receipt["receipts"]:
        scope = f"parity/{path.stem}/{row['dataset']}/{row.get('batch', row.get('kind'))}"
        if "checks" in row:
            actual.add((row["dataset"], row["batch"]))
            a.check(set(row["checks"]) == {"physical_pipeline", "microbatch", "physical_microbatch"}, "RAW_PARITY_COVERAGE", scope)
            for label, item in row["checks"].items():
                a.check(item["pass_"] is True and math.isfinite(item["max_abs_normalized"]) and item["max_abs_normalized"] >= 0,
                        "RAW_MICROBATCH_PARITY", scope + "/" + label)
                if label in ("physical_pipeline", "physical_microbatch"):
                    a.check(item.get("physical_FP32_quantization_bound_used") is True,
                            "RAW_PHYSICAL_PARITY_QUANTIZATION_BOUND", scope + "/" + label)
        else:
            cp0.add((row["dataset"], row["kind"]))
            a.check(row["CP0_raw_parity"] is True and math.isfinite(row["max_abs"]) and row["max_abs"] >= 0,
                    "ACTUAL_CP0_PARITY", scope)
        a.row(attempt_id=scope, logical_id=scope, category="PARITY_DIAGNOSTIC", block="R5_COST_PRECHECK",
              dataset=row["dataset"], fold="primary", type="RAW" if "checks" in row else "ADAPTER", kind=row.get("kind", "RAW_MICRO4"),
              status="PASS", source=filename, source_sha256=digest(path),
              logical_fit_count=0, fitting_configurations=0, optimizer_updates=0, reason="Producer actual parity receipt; no F reexecution in audit")
    a.check(actual == expected, "RAW_PARITY_COVERAGE", filename)
    expected_cp0 = {(r["dataset"], r["kind"]) for r in selection["routes"] if r["type"] == "ADAPTER"
                    and r["fold"] == "primary" and r["r"] == contract["assets"][r["dataset"]]["channels"]}
    a.check(cp0 == expected_cp0, "ACTUAL_CP0_COVERAGE", filename)
    a.equal(len(expected_cp0), 10, filename, "ACTUAL_CP0_EXPECTED_COUNT", atol=0, rtol=0)
    a.equal(len(receipt["receipts"]), len(expected) + len(expected_cp0), filename, "PARITY_UNIQUE_ROW_COUNT", atol=0, rtol=0)
    a.details.setdefault("parity_receipts", {})[filename] = dict(raw_rows=len(actual), CP0_rows=len(cp0),
        scope="Actual producer receipts + source binding; no independent F replay and no saved parity-array arithmetic claim",
        time_limit="Parity file has no internal timestamp. Direct-interpreter chronology additionally uses its separate self-produced execution receipt; audit does not independently replay F.")


# COST_AUDIT_START
def audit_costs(audit, selection, cost_selection, cost_lock):
    """Recalculate saved cost arithmetic and inventory every recorded trial."""
    stages = ("preprocessing", "PE", "F", "DG", "return")
    statistic_fields = (
        "whole_seconds", *(f"{stage}_seconds" for stage in stages),
        "stage_sum_seconds", "whole_minus_stage_sum_seconds", "peak_allocated_bytes",
        "peak_reserved_bytes", "peak_allocated_minus_F_resident_bytes",
        "peak_allocated_minus_pretrial_allocated_bytes", "rss_before_bytes", "rss_after_bytes",
        "process_lifetime_peak_rss_bytes",
    )
    numeric_fields = set(statistic_fields) | {
        "batch", "round", "position", "F_resident_parameter_buffer_bytes", "active_parameter_buffer_bytes",
        "allocated_before_bytes", "reserved_before_bytes", "allocated_after_bytes", "reserved_after_bytes",
    }
    trial_fields = {
        "route_id", "dataset", "method", "batch", "phase", "round", "position", "status", "device",
        *(f"{stage}_seconds" for stage in stages), "whole_seconds", "stage_coverage_complete",
        "stage_sum_seconds", "whole_minus_stage_sum_seconds", "stage_sum_exceeds_whole",
        "F_resident_parameter_buffer_bytes", "active_parameter_buffer_bytes", "active_bytes_scope",
        "allocated_before_bytes", "reserved_before_bytes", "allocated_after_bytes", "reserved_after_bytes",
        "peak_allocated_bytes", "peak_reserved_bytes", "peak_allocated_minus_F_resident_bytes",
        "peak_allocated_minus_pretrial_allocated_bytes", "rss_before_bytes", "rss_after_bytes",
        "process_lifetime_peak_rss_bytes", "output_dtype", "output_shape_json", "error_type", "error_message",
        "route_metadata_json",
    }
    channels = {"beijing": 11, "jena": 21, "robin": 17}
    counters = Counter()
    details = {
        "counters": counters, "groups": [], "folders": [],
        "limitations": [
            "Saved timers and memory counters are checked arithmetically; timings are not rerun.",
            "Distinct PIDs and serial timestamps do not prove absence of unrelated GPU work.",
            "Allocated-minus-F includes active weights, buffers and temporaries; it is not activation-only memory.",
            "Windows peak RSS is a process-lifetime high-water mark, not a per-route peak.",
            "Gradient and LoRA cleanup receipts have no internal route/PID/timestamp binding beyond their paths.",
        ],
    }
    audit.details["cost"] = details

    def load(path, scope, csv_file=False):
        try:
            payload = read_csv(path) if csv_file else read_json(path)
            counters["files_read"] += 1
            return payload
        except FileNotFoundError:
            counters["missing_files"] += 1
            audit.check(False, "COST_MISSING_FILE", scope, str(path))
        except Exception as error:
            counters["unreadable_files"] += 1
            audit.check(False, "COST_FILE_READ", scope, f"{path}: {type(error).__name__}: {error}")
        return None

    def finite(value, scope, *, required=True, nonnegative=False, integer=False):
        if value in (None, ""):
            audit.check(not required, "COST_NUMERIC", scope, "missing numeric value")
            return None
        try:
            result = float(value)
            valid = math.isfinite(result) and (not nonnegative or result >= 0) and (not integer or result.is_integer())
        except (TypeError, ValueError, OverflowError):
            valid, result = False, None
        audit.check(valid, "COST_NUMERIC", scope, f"value={value!r}")
        return result if valid else None

    def uses_f(spec):
        if spec.get("type") == "G":
            return uses_f(spec["base"])
        return spec.get("type") in ("RAW", "R1", "ADAPTER", "LORA")

    def stats(values):
        values = [value for value in values if value is not None]
        if not values:
            return dict(count=0, median=None, mean=None, population_variance=None, min=None, max=None)
        return dict(count=len(values), median=statistics.median(values), mean=statistics.fmean(values),
                    population_variance=statistics.pvariance(values), min=min(values), max=max(values))

    def expected_cost_routes():
        chosen = []
        routes = selection.get("routes", [])
        for dataset in channels:
            pool = [spec for spec in routes if spec.get("dataset") == dataset and spec.get("fold") == "primary"]
            for typ in ("RAW", "CURRENT", "G", "DIRECT", "SWAP", "LORA", "RIDGE"):
                matching = [spec for spec in pool if spec.get("type") == typ and spec.get("length", 512) == 512]
                chosen.extend(matching if typ in ("G", "DIRECT", "SWAP") else matching[:1])
            for kind in ("B0", "B1", "B2", "B3", "B4"):
                matching = [spec for spec in pool if spec.get("type") == "ADAPTER" and spec.get("kind") == kind
                            and spec.get("block", "R2") == "R2"]
                for rank in sorted({spec["r"] for spec in matching}):
                    chosen.append(min((spec for spec in matching if spec["r"] == rank),
                                      key=lambda spec: spec["selected_B"]["mse"]))
            if dataset == "robin":
                chosen.extend(spec for spec in pool if spec.get("type") == "ADAPTER")
            chosen.extend(spec for spec in pool if spec.get("type") == "R1" or
                          (spec.get("type") == "RAW" and spec.get("length", 512) > 512))
            raw = next((spec for spec in pool if spec.get("type") == "RAW" and spec.get("length", 512) == 512), None)
            if raw is not None:
                chosen.append(next(spec for spec in pool if spec["id"] == raw["id"] + "_micro4"
                                   and spec.get("row_microbatch") == 4))
        return list({spec["id"]: spec for spec in chosen}.values())

    try:
        expected_routes = expected_cost_routes()
        audit.check(cost_selection.get("routes") == expected_routes, "COST_SELECTION_ROUTES", "COST_SELECTION",
                    "Full route dictionaries/order must include all locked R1/long Raw routes and the locked micro4 specs")
    except Exception as error:
        audit.check(False, "COST_SELECTION_DERIVATION", "COST_SELECTION", f"{type(error).__name__}: {error}")
    selected = cost_selection.get("routes", [])
    if not audit.check(isinstance(selected, list) and all(isinstance(spec, dict) and "id" in spec for spec in selected),
                       "COST_SELECTION_SCHEMA", "COST_SELECTION"):
        selected = []
    ids = [spec["id"] for spec in selected]
    audit.check(len(ids) == len(set(ids)) and bool(ids), "COST_SELECTION_IDS", "COST_SELECTION", str(ids))
    audit.equal(cost_selection.get("C_scored"), False, "COST_SELECTION.C_scored", code="COST_PRE_C")
    audit.equal(cost_selection.get("training_competitors"), [], "COST_SELECTION.training_competitors", code="COST_COMPETITORS")
    audit.equal(cost_lock.get("status"), "COMPLETE_VALID", "COST_LOCK.status", code="COST_LOCK_STATUS")
    audit.equal(cost_lock.get("C_scored"), False, "COST_LOCK.C_scored", code="COST_PRE_C")
    bindings = {}
    for filename in ("SELECTION_LOCK.json", "COST_SELECTION.json", "GENERAL_COST_COMPLETE.json", "COST_LOCK.json"):
        try:
            bindings[filename] = digest(HERE / filename)
        except Exception as error:
            audit.check(False, "COST_BINDING_READ", filename, f"{type(error).__name__}: {error}")
    for payload, field, filename, label in (
        (cost_selection, "selection_lock_sha256", "SELECTION_LOCK.json", "COST_SELECTION"),
        (cost_lock, "selection_lock", "SELECTION_LOCK.json", "COST_LOCK"),
        (cost_lock, "cost_selection_sha256", "COST_SELECTION.json", "COST_LOCK"),
        (cost_lock, "general_cost", "GENERAL_COST_COMPLETE.json", "COST_LOCK"),
    ):
        if filename in bindings:
            audit.equal(payload.get(field), bindings[filename], f"{label}.{field}", code="COST_HASH_BINDING")
    essential = []
    for dataset in channels:
        pool = [spec for spec in selected if spec.get("dataset") == dataset]
        for typ in ("RAW", "G", "DIRECT", "SWAP", "RIDGE", "LORA"):
            essential.extend(spec for spec in pool if spec.get("type") == typ)
        essential.extend(spec for spec in pool if spec.get("type") == "ADAPTER" and spec.get("kind") in ("B0", "B2", "B4"))
    essential = list({spec["id"]: spec for spec in essential}.values())
    expected_outcomes = [dict(id=spec["id"], exit_code=0, status="COMPLETE_VALID") for spec in essential]
    audit.check(cost_lock.get("exclusive") == expected_outcomes, "COST_EXCLUSIVE_COVERAGE", "COST_LOCK.exclusive",
                "Expected exact essential IDs/order and successful separate-process exits")
    details["expected_general_routes"] = len(selected)
    details["expected_exclusive_routes"] = len(essential)
    counters["expected_route_batch_groups"] = 2 * (len(selected) + len(essential))
    input_lengths = {dataset: max((spec.get("length", 512) for spec in selected if spec.get("dataset") == dataset), default=512)
                     for dataset in channels}
    details["common_input_lengths"] = input_lengths
    recomputed_flat, folder_metadata, active_by_id, inputs_by_dataset, scalers_by_dataset = {}, {}, {}, {}, {}
    f_residents = []

    def inspect_folder(scope, name, specs):
        folder = HERE / ("cost" if scope == "general" else "cost_exclusive") / name
        label = f"cost/{scope}/{name}"
        meta = load(folder / "COST_RUN_METADATA.json", label + "/metadata")
        summaries = load(folder / "COST_SUMMARY.json", label + "/summary")
        raw = load(folder / "RAW_COST_TRIALS.csv", label + "/raw", csv_file=True)
        counters["folders_inspected"] += 1
        if meta is not None and not isinstance(meta, dict):
            audit.check(False, "COST_METADATA_SCHEMA", label, "metadata must be an object")
            meta = None
        if summaries is not None and not isinstance(summaries, list):
            audit.check(False, "COST_SUMMARY_SCHEMA", label, "summary must be a list")
            summaries = None
        expected_specs = {spec["id"]: spec for spec in specs}
        order = list(expected_specs)
        q = len(order)
        rounds = math.ceil(10 / q) * q if q else None
        expected_groups = {(identifier, batch) for identifier in order for batch in (1, 8)}
        interval = None
        resident = None
        if meta is not None:
            folder_metadata[(scope, name)] = meta
            audit.equal(meta, dict(warmups_per_route_batch=5, requested_measured_repeats=10,
                                  candidate_count=q, batch_sizes=[1, 8], active_candidates=1,
                                  factory_load_unload_in_whole_timer=False,
                                  physical_preprocessing_CPU_return_in_whole_timer=True,
                                  exclusive_process_claimed_by_caller=scope == "exclusive",
                                  exclusive_process_not_verified_by_this_module=True),
                        label + "/metadata", code="COST_RUN_CONTRACT")
            if rounds is not None:
                audit.equal(meta.get("measured_rounds"), rounds, label + "/rounds", code="COST_ROUNDS")
            audit.check(meta.get("routes") == specs, "COST_ROUTE_ORDER", label, "Metadata route dictionaries/order differ")
            audit.equal(meta.get("status"), "COMPLETE_VALID", label + "/status", code="COST_RUN_STATUS")
            finite(meta.get("pid"), label + "/pid", nonnegative=True, integer=True)
            audit.check(isinstance(meta.get("pid"), int) and meta["pid"] > 0, "COST_PID", label, str(meta.get("pid")))
            audit.check(str(meta.get("device", "")).startswith("cuda"), "COST_DEVICE", label, str(meta.get("device")))
            resident = finite(meta.get("F_resident_parameter_buffer_bytes"), label + "/resident", nonnegative=True, integer=True)
            needs_f = scope == "general" or any(uses_f(spec) for spec in specs)
            audit.equal(meta.get("F_instances_supplied_by_caller"), int(needs_f), label + "/F_instances", code="COST_F_RESIDENCY")
            audit.check(resident is not None and (resident > 0 if needs_f else resident == 0),
                        "COST_F_RESIDENCY", label, f"needs_F={needs_f}, resident={resident}")
            if resident is not None and needs_f:
                f_residents.append(resident)
            try:
                start_text, end_text = meta["started_utc"], meta["finished_utc"]
                start, end = utc_seconds(start_text), utc_seconds(end_text)
                aware = datetime.fromisoformat(start_text.replace("Z", "+00:00")).utcoffset() is not None
                aware = aware and datetime.fromisoformat(end_text.replace("Z", "+00:00")).utcoffset() is not None
                audit.check(aware and math.isfinite(start) and math.isfinite(end) and start <= end,
                            "COST_TIME_ORDER", label, f"start={start_text}, end={end_text}")
                interval = (start, end)
            except Exception as error:
                audit.check(False, "COST_TIME_ORDER", label, f"{type(error).__name__}: {error}")
            contracts = meta.get("input_contracts", [])
            if not isinstance(contracts, list):
                audit.check(False, "COST_INPUT_SCHEMA", label, "input_contracts must be a list")
                contracts = []
            expected_contracts = {(spec["dataset"], batch) for spec in specs for batch in (1, 8)}
            actual_contracts = Counter((item.get("dataset"), item.get("batch")) for item in contracts if isinstance(item, dict))
            audit.check(actual_contracts == Counter({key: 1 for key in expected_contracts}),
                        "COST_INPUT_COVERAGE", label, str(actual_contracts))
            for contract in contracts:
                if not isinstance(contract, dict):
                    audit.check(False, "COST_INPUT_SCHEMA", label, repr(contract))
                    continue
                dataset, batch = contract.get("dataset"), contract.get("batch")
                if dataset in channels and batch in (1, 8):
                    input_length = input_lengths[dataset]
                    audit.equal(contract.get("physical_shape"), [batch, input_length, channels[dataset]],
                                label + f"/input/{dataset}/{batch}", code="COST_INPUT_SHAPE")
                    observed = finite(contract.get("past_observed_values"), label + "/observed", nonnegative=True, integer=True)
                    audit.check(observed is not None and observed <= batch * input_length * channels[dataset],
                                "COST_INPUT_MASK_COUNT", label, str(observed))
                hashes = contract.get("physical_observed_mean_std_sha256", [])
                audit.check(isinstance(hashes, list) and len(hashes) == 4 and
                            all(isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef") for value in hashes),
                            "COST_INPUT_HASH_SCHEMA", label, "Expected four saved input hashes")
                input_key = (dataset, batch, tuple(contract.get("physical_shape", [])))
                if (dataset, batch) in expected_contracts:
                    if input_key in inputs_by_dataset:
                        audit.check(contract == inputs_by_dataset[input_key], "COST_MATCHED_INPUTS", label,
                                    f"Saved physical/mask/scaler input contract differs for {input_key}")
                    else:
                        inputs_by_dataset[input_key] = contract
                    if isinstance(hashes, list) and len(hashes) == 4:
                        if dataset in scalers_by_dataset:
                            audit.check(hashes[2:] == scalers_by_dataset[dataset], "COST_MATCHED_SCALERS", label,
                                        f"A-only mean/std hashes differ for {dataset}")
                        else:
                            scalers_by_dataset[dataset] = hashes[2:]
        details["folders"].append(dict(scope=scope, name=name, raw_available=raw is not None,
                                        metadata_available=meta is not None, summary_available=summaries is not None,
                                        interval=interval))
        if raw is None:
            for spec in specs:
                for batch in (1, 8):
                    audit.row(attempt_id=f"cost_{scope}_{spec['id']}_batch{batch}", logical_id=spec["id"],
                              category="cost", dataset=spec.get("dataset"), fold=spec.get("fold"),
                              type=spec.get("type"), kind=spec.get("kind", spec.get("variant", "")),
                              r=spec.get("r"), seed=spec.get("seed"), status="MISSING_OR_UNREADABLE",
                              source=str(folder / "RAW_COST_TRIALS.csv"), reason=f"{scope}, batch={batch}; trial counts unavailable")
            return
        counters["raw_files"] += 1
        counters["raw_trials"] += len(raw)
        grouped = defaultdict(list)
        raw_sha = digest(folder / "RAW_COST_TRIALS.csv")
        for index, trial in enumerate(raw):
            identifier = trial.get("route_id", "<missing>")
            trial_scope = f"{label}/{identifier}/row{index + 2}"
            before = len(audit.findings)
            audit.check(set(trial) == trial_fields, "COST_RAW_SCHEMA", trial_scope,
                        f"missing={sorted(trial_fields - set(trial))}; extra={sorted(set(trial) - trial_fields)}")
            parsed = dict(trial)
            parsed["_raw_index"] = index
            for field in numeric_fields:
                parsed[field] = finite(trial.get(field), trial_scope + "/" + field,
                                       required=field in ("batch", "round", "position"),
                                       integer=field in ("batch", "round", "position") or field.endswith("bytes"))
            batch = parsed["batch"]
            if batch is not None:
                batch = int(batch)
            parsed["batch"] = batch
            grouped[(identifier, batch)].append(parsed)
            phase, status = trial.get("phase"), trial.get("status")
            counters[f"{phase}_trials"] += 1
            counters["error_trials"] += int(status == "ERROR")
            counters["cleanup_error_trials"] += int(phase == "cleanup" and status == "ERROR")
            audit.check((identifier, batch) in expected_groups, "COST_RAW_COVERAGE", trial_scope, f"batch={batch}")
            audit.check(phase in ("warmup", "measure", "cleanup"), "COST_PHASE", trial_scope, str(phase))
            audit.check(status in ("VALID", "ERROR") and (phase != "cleanup" or status == "ERROR"),
                        "COST_TRIAL_STATUS", trial_scope, f"phase={phase}, status={status}")
            spec = expected_specs.get(identifier)
            if spec is not None:
                audit.equal(trial.get("dataset"), spec.get("dataset"), trial_scope + "/dataset", code="COST_ROUTE_BINDING")
                method = spec.get("method", spec.get("variant", identifier))
                if status == "VALID":
                    audit.equal(trial.get("method"), method, trial_scope + "/method", code="COST_ROUTE_BINDING")
                try:
                    declared = json.loads(trial.get("route_metadata_json", ""))
                    audit.check(declared == spec, "COST_ROUTE_BINDING", trial_scope, "route_metadata_json differs from locked route")
                except Exception as error:
                    audit.check(False, "COST_ROUTE_BINDING", trial_scope, f"{type(error).__name__}: {error}")
                if phase == "warmup":
                    expected_position = order.index(identifier)
                    audit.equal(parsed["position"], expected_position, trial_scope + "/position", code="COST_ROTATION")
                    audit.check(parsed["round"] in range(5), "COST_WARMUP_ROUND", trial_scope, str(parsed["round"]))
                elif phase == "measure" and rounds is not None:
                    current_round = parsed["round"]
                    audit.check(current_round in range(rounds), "COST_MEASURE_ROUND", trial_scope, str(current_round))
                    if current_round is not None:
                        expected_position = (order.index(identifier) - int(current_round)) % q
                        audit.equal(parsed["position"], expected_position, trial_scope + "/position", code="COST_ROTATION")
            if meta is not None:
                audit.equal(trial.get("device"), meta.get("device"), trial_scope + "/device", code="COST_DEVICE")
            if status == "ERROR":
                audit.check(False, "COST_TRIAL_ERROR", trial_scope,
                            f"{phase}: {trial.get('error_type')}: {trial.get('error_message')}")
                audit.check(bool(trial.get("error_type")), "COST_ERROR_RECEIPT", trial_scope, "missing error_type")
            elif status == "VALID":
                for field in ("stage_coverage_complete", "stage_sum_exceeds_whole"):
                    audit.check(str(trial.get(field)).lower() in ("true", "false"),
                                "COST_BOOLEAN_SCHEMA", trial_scope + "/" + field, repr(trial.get(field)))
                for field in ("whole_seconds", *(f"{stage}_seconds" for stage in stages),
                              "F_resident_parameter_buffer_bytes", "active_parameter_buffer_bytes", "allocated_before_bytes",
                              "reserved_before_bytes", "allocated_after_bytes", "reserved_after_bytes", "peak_allocated_bytes",
                              "peak_reserved_bytes", "rss_before_bytes", "rss_after_bytes"):
                    value = parsed.get(field)
                    audit.check(value is not None and value >= 0, "COST_VALID_VALUE", trial_scope + "/" + field, str(value))
                whole = parsed["whole_seconds"]
                complete = all(parsed[f"{stage}_seconds"] is not None for stage in stages)
                stage_sum = sum(parsed[f"{stage}_seconds"] or 0 for stage in stages)
                audit.equal(truth(trial.get("stage_coverage_complete")), complete, trial_scope + "/coverage", code="COST_STAGE_ACCOUNTING")
                audit.check(complete, "COST_STAGE_COMPLETE", trial_scope, "All five same-trial stage timers required")
                audit.equal(parsed["stage_sum_seconds"], stage_sum, trial_scope + "/stage_sum", code="COST_STAGE_ACCOUNTING")
                if whole is not None:
                    difference = whole - stage_sum
                    exceeds = difference < -max(0.001, whole * 0.05)
                    audit.equal(parsed["whole_minus_stage_sum_seconds"], difference, trial_scope + "/whole_minus_stages", code="COST_STAGE_ACCOUNTING")
                    audit.equal(truth(trial.get("stage_sum_exceeds_whole")), exceeds, trial_scope + "/stage_exceeds", code="COST_STAGE_ACCOUNTING")
                    audit.check(not exceeds, "COST_STAGE_EXCEEDS_WHOLE", trial_scope, str(difference))
                if resident is not None:
                    audit.equal(parsed["F_resident_parameter_buffer_bytes"], resident, trial_scope + "/F_resident", code="COST_F_RESIDENCY")
                if spec is not None:
                    base = spec["base"] if spec.get("type") == "G" else spec
                    if not uses_f(spec):
                        audit.equal(parsed["F_seconds"], 0, trial_scope + "/F_seconds", code="COST_NO_F_STAGE")
                    else:
                        audit.check(parsed["F_seconds"] is not None and parsed["F_seconds"] > 0,
                                    "COST_F_STAGE", trial_scope, str(parsed["F_seconds"]))
                    if base.get("type") in ("RAW", "LORA"):
                        audit.equal(parsed["PE_seconds"], 0, trial_scope + "/PE_seconds", code="COST_STAGE_ROUTE")
                    active = parsed["active_parameter_buffer_bytes"]
                    audit.check(active is not None and (active == 0 if spec.get("type") in ("RAW", "CURRENT", "R1") else active > 0),
                                "COST_ACTIVE_BYTES", trial_scope, str(active))
                    if active is not None:
                        if identifier in active_by_id:
                            audit.equal(active, active_by_id[identifier], trial_scope + "/active_bytes", code="COST_ACTIVE_BYTES")
                        else:
                            active_by_id[identifier] = active
                    audit.equal(trial.get("active_bytes_scope"), "route_declared_active_parameter_bytes",
                                trial_scope + "/active_scope", code="COST_ACTIVE_BYTES")
                    try:
                        shape = json.loads(trial.get("output_shape_json", ""))
                        audit.equal(shape, [batch, 48, channels[spec["dataset"]]], trial_scope + "/output_shape", code="COST_OUTPUT_CONTRACT")
                    except Exception as error:
                        audit.check(False, "COST_OUTPUT_CONTRACT", trial_scope, f"{type(error).__name__}: {error}")
                peak = parsed["peak_allocated_bytes"]
                f_bytes, before_bytes = parsed["F_resident_parameter_buffer_bytes"], parsed["allocated_before_bytes"]
                if peak is not None and f_bytes is not None:
                    audit.equal(parsed["peak_allocated_minus_F_resident_bytes"], peak - f_bytes,
                                trial_scope + "/peak_minus_F", code="COST_MEMORY_ACCOUNTING")
                if peak is not None and before_bytes is not None:
                    audit.equal(parsed["peak_allocated_minus_pretrial_allocated_bytes"], peak - before_bytes,
                                trial_scope + "/peak_minus_before", code="COST_MEMORY_ACCOUNTING")
                for low, high in (("allocated_before_bytes", "reserved_before_bytes"),
                                  ("allocated_after_bytes", "reserved_after_bytes"),
                                  ("allocated_before_bytes", "peak_allocated_bytes"),
                                  ("allocated_after_bytes", "peak_allocated_bytes"),
                                  ("reserved_before_bytes", "peak_reserved_bytes"),
                                  ("reserved_after_bytes", "peak_reserved_bytes"),
                                  ("peak_allocated_bytes", "peak_reserved_bytes"),
                                  ("F_resident_parameter_buffer_bytes", "allocated_before_bytes")):
                    left, right = parsed[low], parsed[high]
                    audit.check(left is not None and right is not None and left <= right,
                                "COST_MEMORY_ORDER", trial_scope, f"{low}={left}, {high}={right}")
                lifetime = parsed["process_lifetime_peak_rss_bytes"]
                if lifetime is not None:
                    audit.check(all(parsed[field] is not None and parsed[field] <= lifetime for field in ("rss_before_bytes", "rss_after_bytes")),
                                "COST_RSS_ORDER", trial_scope, str(lifetime))
            parsed["_audit_valid"] = len(audit.findings) == before
        if q:
            expected_sequence = []
            for batch in (1, 8):
                for position, identifier in enumerate(order):
                    expected_sequence.extend((identifier, batch, "warmup", round_index, position) for round_index in range(5))
                for round_index in range(rounds):
                    offset = round_index % q
                    rotated = order[offset:] + order[:offset]
                    expected_sequence.extend((identifier, batch, "measure", round_index, position)
                                             for position, identifier in enumerate(rotated))
            actual_sequence = [(trial.get("route_id"), trial.get("batch"), trial.get("phase"), trial.get("round"), trial.get("position"))
                               for trial in sorted((item for group in grouped.values() for item in group), key=lambda item: item["_raw_index"])]
            mismatch = next((index for index, pair in enumerate(zip(actual_sequence, expected_sequence)) if pair[0] != pair[1]), None)
            audit.check(actual_sequence == expected_sequence, "COST_SERIAL_TRIAL_ORDER", label,
                        f"actual_rows={len(actual_sequence)}, expected_rows={len(expected_sequence)}, first_mismatch={mismatch}")
        audit.check(set(grouped) == expected_groups, "COST_GROUP_COVERAGE", label,
                    f"missing={sorted(expected_groups - set(grouped), key=str)}, extra={sorted(set(grouped) - expected_groups, key=str)}")
        if meta is not None:
            audit.equal(meta.get("raw_trial_count_including_warmups_errors"), len(raw), label + "/raw_count", code="COST_TRIAL_COUNTS")
            audit.equal(meta.get("error_count"), sum(trial.get("status") == "ERROR" for trial in raw), label + "/error_count", code="COST_TRIAL_COUNTS")
            audit.equal(meta.get("cleanup_failure_count"), sum(trial.get("phase") == "cleanup" and trial.get("status") == "ERROR" for trial in raw),
                        label + "/cleanup_count", code="COST_TRIAL_COUNTS")
        saved_summaries = defaultdict(list)
        if summaries is not None:
            counters["summary_rows"] += len(summaries)
            for item in summaries:
                if isinstance(item, dict):
                    saved_summaries[(item.get("route_id"), item.get("batch"))].append(item)
                else:
                    audit.check(False, "COST_SUMMARY_SCHEMA", label, repr(item))
            audit.check(Counter({key: len(value) for key, value in saved_summaries.items()}) == Counter({key: 1 for key in expected_groups}),
                        "COST_SUMMARY_COVERAGE", label, str({str(key): len(value) for key, value in saved_summaries.items()}))
        computed_statuses = []
        for identifier, batch in sorted(set(grouped) | expected_groups, key=str):
            group_scope = f"{label}/{identifier}/batch{batch}"
            trials = grouped.get((identifier, batch), [])
            measures = [trial for trial in trials if trial.get("phase") == "measure"]
            warmups = [trial for trial in trials if trial.get("phase") == "warmup"]
            valid = [trial for trial in measures if trial.get("status") == "VALID"]
            errors = [trial for trial in trials if trial.get("status") == "ERROR"]
            coverage = len(warmups) == 5 and rounds is not None and len(measures) == rounds
            if (identifier, batch) in expected_groups:
                audit.equal(len(warmups), 5, group_scope + "/warmups", code="COST_TRIAL_COUNTS")
                audit.equal(len(measures), rounds, group_scope + "/measures", code="COST_TRIAL_COUNTS")
                audit.check(Counter(trial["round"] for trial in warmups) == Counter(range(5)), "COST_WARMUP_COVERAGE", group_scope)
                audit.check(Counter(trial["round"] for trial in measures) == Counter(range(rounds)), "COST_ROUND_COVERAGE", group_scope)
                if q:
                    audit.check(Counter(trial["position"] for trial in measures) == Counter({position: rounds // q for position in range(q)}),
                                "COST_POSITION_BALANCE", group_scope)
            group_valid = coverage and len(valid) == rounds and not errors and all(trial.get("_audit_valid") for trial in trials)
            spec = expected_specs.get(identifier, {})
            group_detail = dict(scope=scope, id=identifier, batch=batch, measure_trials=len(measures),
                                warmup_trials=len(warmups), error_trials=len(errors), valid_measure_trials=len(valid))
            details["groups"].append(group_detail)
            counters["route_batch_groups"] += 1
            audit.row(attempt_id=f"cost_{scope}_{identifier}_batch{batch}", logical_id=identifier, category="cost",
                      block=spec.get("block"), dataset=spec.get("dataset", trials[0].get("dataset") if trials else None),
                      fold=spec.get("fold"), type=spec.get("type"), kind=spec.get("kind", spec.get("variant", "")),
                      r=spec.get("r"), seed=spec.get("seed"), status="COMPLETE_VALID" if group_valid else "INVALID_OR_INCOMPLETE",
                      source=str(folder / "RAW_COST_TRIALS.csv"), source_sha256=raw_sha,
                      logical_fit_count=0, fitting_configurations=0, optimizer_updates=0, incremental_updates=0,
                      updates_exact=True, zero_no_fit=True, cost_measure_trials=len(measures), cost_warmup_trials=len(warmups),
                      cost_error_trials=len(errors), reason=f"scope={scope}; batch={batch}; saved timing trials, no optimizer update")
            if not measures:
                audit.check(False, "COST_NO_MEASUREMENTS", group_scope, "No raw measure rows; summary cannot be reconstructed")
                continue
            summary = dict(
                route_id=identifier, dataset=measures[0].get("dataset"), method=measures[0].get("method"), batch=batch,
                status="COMPLETE_VALID" if len(valid) == rounds else "INVALID_OR_INCOMPLETE",
                measured_trials=len(measures), valid_trials=len(valid), errors=len(measures) - len(valid),
                exclusive_process_claimed_by_caller=scope == "exclusive",
                F_resident_parameter_buffer_bytes=resident if resident is not None else measures[0].get("F_resident_parameter_buffer_bytes"),
                active_parameter_buffer_bytes=valid[0].get("active_parameter_buffer_bytes") if valid else None,
                all_stage_timers_complete=bool(valid) and all(all(trial.get(f"{stage}_seconds") is not None for stage in stages) for trial in valid),
                stage_sum_exceeds_whole_trials=sum(
                    trial.get("whole_seconds") is not None and
                    trial["whole_seconds"] - sum(trial.get(f"{stage}_seconds") or 0 for stage in stages) < -max(0.001, trial["whole_seconds"] * 0.05)
                    for trial in valid),
                statistics={field: stats([trial.get(field) for trial in valid]) for field in statistic_fields},
            )
            computed_statuses.append(summary["status"])
            for saved in saved_summaries.get((identifier, batch), []):
                audit.equal(saved, summary, group_scope + "/summary", code="COST_SUMMARY_ARITHMETIC", atol=1e-12, rtol=1e-9)
            values = summary["statistics"]
            flat = dict(id=identifier, dataset=summary["dataset"], batch=batch, scope=scope, status=summary["status"],
                        whole_seconds_median=values["whole_seconds"]["median"],
                        whole_seconds_variance=values["whole_seconds"]["population_variance"],
                        peak_allocated_bytes_median=values["peak_allocated_bytes"]["median"],
                        peak_reserved_bytes_median=values["peak_reserved_bytes"]["median"],
                        F_resident_bytes=summary["F_resident_parameter_buffer_bytes"],
                        active_parameter_buffer_bytes=summary["active_parameter_buffer_bytes"],
                        valid_trials=summary["valid_trials"], measured_trials=summary["measured_trials"])
            flat.update({f"{stage}_seconds_median": values[f"{stage}_seconds"]["median"] for stage in stages})
            recomputed_flat[(scope, identifier, batch)] = flat
        if meta is not None:
            computed_status = "COMPLETE_VALID" if computed_statuses and all(value == "COMPLETE_VALID" for value in computed_statuses) and not any(trial.get("status") == "ERROR" for trial in raw) else "INVALID_OR_INCOMPLETE"
            audit.equal(meta.get("status"), computed_status, label + "/recomputed_status", code="COST_RUN_STATUS")

    expected_folders = {
        "general": {dataset: [spec for spec in selected if spec.get("dataset") == dataset] for dataset in channels},
        "exclusive": {spec["id"]: [spec] for spec in essential},
    }
    for scope, folders in expected_folders.items():
        base = HERE / ("cost" if scope == "general" else "cost_exclusive")
        observed = {path.name for path in safe_path(base).glob("*") if path.is_dir()}
        audit.check(observed == set(folders), "COST_FOLDER_COVERAGE", scope,
                    f"missing={sorted(set(folders) - observed)}, extra={sorted(observed - set(folders))}")
        for name in list(folders) + sorted(observed - set(folders)):
            try:
                inspect_folder(scope, name, folders.get(name, []))
            except Exception as error:
                audit.check(False, "COST_GROUP_EXCEPTION", f"cost/{scope}/{name}", f"{type(error).__name__}: {error}")
                counters["folder_exceptions"] += 1
    if f_residents:
        audit.check(len(set(f_residents)) == 1, "COST_F_SHARED_STORAGE", "cost", str(sorted(set(f_residents))))
    general = load(HERE / "GENERAL_COST_COMPLETE.json", "GENERAL_COST_COMPLETE")
    if isinstance(general, dict):
        audit.equal(general, dict(status="COMPLETE_VALID", C_scored=False, F_unchanged=True), "GENERAL_COST_COMPLETE", code="COST_GENERAL_RECEIPT")
        for field, filename in (("selection_lock_sha256", "SELECTION_LOCK.json"), ("cost_selection_sha256", "COST_SELECTION.json")):
            if filename in bindings:
                audit.equal(general.get(field), bindings[filename], "GENERAL_COST_COMPLETE." + field, code="COST_HASH_BINDING")
        complete_metadata = [folder_metadata.get(("general", dataset)) for dataset in channels]
        audit.check(all(item is not None for item in complete_metadata) and general.get("metadata") == complete_metadata,
                    "COST_GENERAL_METADATA_BINDING", "GENERAL_COST_COMPLETE", "Receipt must match all three actual metadata objects")
    first_c = load(HERE / "FIRST_C_SCORE.json", "FIRST_C_SCORE")
    if isinstance(first_c, dict):
        for field, filename in (("selection_lock", "SELECTION_LOCK.json"), ("cost_lock", "COST_LOCK.json")):
            if filename in bindings:
                audit.equal(first_c.get(field), bindings[filename], "FIRST_C_SCORE." + field, code="COST_HASH_BINDING")
    selected_time = finite(selection.get("time_unix"), "SELECTION_LOCK.time_unix")
    cost_selected_time = finite(cost_selection.get("time_unix"), "COST_SELECTION.time_unix")
    locked_time = finite(cost_lock.get("time_unix"), "COST_LOCK.time_unix")
    first_c_time = finite(first_c.get("time_unix"), "FIRST_C_SCORE.time_unix") if isinstance(first_c, dict) else None
    for before, after, scope in ((selected_time, cost_selected_time, "selection_before_cost_selection"),
                                 (cost_selected_time, locked_time, "cost_selection_before_cost_lock"),
                                 (locked_time, first_c_time, "cost_lock_before_C")):
        audit.check(before is not None and after is not None and before <= after, "COST_TIME_ORDER", scope, f"{before} <= {after}")
    previous_end = cost_selected_time
    general_pids = []
    for scope, names in (("general", list(channels)), ("exclusive", [spec["id"] for spec in essential])):
        for name in names:
            meta = folder_metadata.get((scope, name))
            if meta is None:
                continue
            if scope == "general":
                general_pids.append(meta.get("pid"))
            else:
                audit.check(meta.get("pid") not in general_pids, "COST_EXCLUSIVE_PID", name,
                            f"exclusive pid={meta.get('pid')}, general={general_pids}")
            try:
                start, end = utc_seconds(meta["started_utc"]), utc_seconds(meta["finished_utc"])
                audit.check(previous_end is not None and previous_end <= start, "COST_SERIAL_TIME", f"{scope}/{name}",
                            f"previous finish={previous_end}, start={start}")
                audit.check(locked_time is not None and end <= locked_time, "COST_TIME_ORDER", f"{scope}/{name}/before_lock",
                            f"end={end}, lock={locked_time}")
                previous_end = end
            except Exception as error:
                audit.check(False, "COST_TIME_ORDER", f"{scope}/{name}", f"{type(error).__name__}: {error}")
    audit.check(len(general_pids) == 3 and len(set(general_pids)) == 1, "COST_GENERAL_PID", "cost/general", str(general_pids))
    details["general_pids"] = general_pids
    details["exclusive_pids"] = [folder_metadata[("exclusive", spec["id"])].get("pid")
                                  for spec in essential if ("exclusive", spec["id"]) in folder_metadata]
    flat_rows = load(HERE / "COST.csv", "COST.csv", csv_file=True)
    expected_flat_keys = {("general", spec["id"], batch) for spec in selected for batch in (1, 8)}
    expected_flat_keys |= {("exclusive", spec["id"], batch) for spec in essential for batch in (1, 8)}
    if flat_rows is not None:
        counters["flat_rows"] = len(flat_rows)
        observed_flat_keys = Counter()
        string_fields = {"id", "dataset", "scope", "status"}
        for index, row in enumerate(flat_rows):
            label = f"COST.csv/row{index + 2}/{row.get('scope')}/{row.get('id')}"
            try:
                actual = {key: value if key in string_fields else finite(value, label + "/" + key, required=False)
                          for key, value in row.items()}
                key = (row.get("scope"), row.get("id"), actual.get("batch"))
                observed_flat_keys[key] += 1
                if audit.check(key in recomputed_flat, "COST_FLAT_RAW_BINDING", label, str(key)):
                    expected = recomputed_flat[key]
                    audit.check(set(actual) == set(expected), "COST_FLAT_SCHEMA", label,
                                f"missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}")
                    audit.equal(actual, expected, label, code="COST_FLAT_ARITHMETIC", atol=1e-12, rtol=1e-9)
            except Exception as error:
                audit.check(False, "COST_FLAT_EXCEPTION", label, f"{type(error).__name__}: {error}")
        audit.check(observed_flat_keys == Counter({key: 1 for key in expected_flat_keys}), "COST_FLAT_COVERAGE", "COST.csv",
                    f"missing={sorted(expected_flat_keys - set(observed_flat_keys), key=str)}, extra_or_duplicate={[(key, count) for key, count in observed_flat_keys.items() if key not in expected_flat_keys or count != 1]}")
    for dataset in ("beijing", "jena"):
        label = f"LORA_CLEANUP_{dataset}.json"
        receipt = load(HERE / label, label)
        if not isinstance(receipt, dict):
            if receipt is not None:
                audit.check(False, "COST_LORA_CLEANUP_SCHEMA", label, "receipt must be an object")
            continue
        counters["lora_cleanup_receipts"] += 1
        audit.equal(receipt, dict(status="PASS", F_base_weights_unchanged=True), label, code="COST_LORA_CLEANUP")
        before = finite(receipt.get("hooks_before"), label + "/hooks_before", nonnegative=True, integer=True)
        after = finite(receipt.get("hooks_after"), label + "/hooks_after", nonnegative=True, integer=True)
        audit.check(before is not None and after is not None and before == after, "COST_LORA_HOOKS", label, f"{before} -> {after}")
        raw_before, raw_after = receipt.get("raw_before"), receipt.get("raw_after")
        metric_fields = {"mse", "mae", "channel_mse", "channel_mae", "counts", "n_observed"}
        if audit.check(isinstance(raw_before, dict) and isinstance(raw_after, dict), "COST_LORA_RAW_SCHEMA", label):
            audit.check(set(raw_before) == metric_fields and set(raw_after) == metric_fields, "COST_LORA_RAW_SCHEMA", label,
                        "Expected complete before/after score dictionaries")
            audit.equal(raw_after, raw_before, label + "/raw_parity", code="COST_LORA_RAW_PARITY", atol=1e-10, rtol=0)
            for field in ("mse", "mae", "n_observed"):
                finite(raw_before.get(field), label + "/raw_before/" + field, nonnegative=True, integer=field == "n_observed")
    for spec in essential:
        if spec.get("type") != "ADAPTER":
            continue
        path = HERE / "cost_exclusive" / spec["id"] / "TRAINING_GRADIENT_MEMORY.json"
        label = f"cost/exclusive/{spec['id']}/gradient_memory"
        receipt = load(path, label)
        if not isinstance(receipt, dict):
            if receipt is not None:
                audit.check(False, "COST_GRADIENT_SCHEMA", label, "receipt must be an object")
            continue
        counters["gradient_memory_receipts"] += 1
        before = len(audit.findings)
        audit.equal(receipt, dict(diagnostic="Selected weights, one real A batch forward/backward, no optimizer update",
                                  batch=4, optimizer_updates=0, F_parameters_frozen=True, F_parameter_gradients_none=True),
                    label, code="COST_GRADIENT_RECEIPT")
        values = {field: finite(receipt.get(field), label + "/" + field, nonnegative=True,
                                integer=field in ("peak_allocated_bytes", "peak_reserved_bytes"))
                  for field in ("wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "E_gradient_norm", "D_gradient_norm")}
        audit.check(all(values[field] is not None and values[field] > 0 for field in values),
                    "COST_GRADIENT_FLOW", label, str(values))
        allocated, reserved = values["peak_allocated_bytes"], values["peak_reserved_bytes"]
        audit.check(allocated is not None and reserved is not None and allocated <= reserved,
                    "COST_GRADIENT_MEMORY_ORDER", label, f"allocated={allocated}, reserved={reserved}")
        meta = folder_metadata.get(("exclusive", spec["id"]), {})
        resident = finite(meta.get("F_resident_parameter_buffer_bytes"), label + "/F_resident", required=False, nonnegative=True, integer=True)
        active = active_by_id.get(spec["id"])
        if resident is not None and active is not None:
            audit.check(allocated is not None and allocated >= resident + active, "COST_GRADIENT_MEMORY_ORDER", label,
                        f"allocated={allocated}, F+active={resident + active}")
        audit.row(attempt_id=f"cost_gradient_{spec['id']}", logical_id=spec["id"], category="cost_gradient_diagnostic",
                  block=spec.get("block"), dataset=spec.get("dataset"), fold=spec.get("fold"), type=spec.get("type"),
                  kind=spec.get("kind"), r=spec.get("r"), seed=spec.get("seed"), source=str(path), source_sha256=digest(path),
                  status="RECORDED_NO_OPTIMIZER_UPDATE" if len(audit.findings) == before else "INVALID",
                  logical_fit_count=0, fitting_configurations=0, optimizer_updates=0, incremental_updates=0,
                  updates_exact=True, zero_no_fit=True, reason="One selected-route A-batch backward; no optimizer update or refit")
    details["counters"] = dict(sorted(counters.items()))
# COST_AUDIT_END


def audit_direct_parity_execution(a, selection, cost_lock, first):
    path = HERE / "DIRECT_PARITY_EXECUTION.json"
    if not a.check(path.is_file(), "DIRECT_PARITY_EXECUTION_RECEIPT", path.name): return
    receipt = read_json(path)
    a.equal(receipt, dict(status="PASS", C_before=False, C_after=False, scientific_sources_changed=False),
            path.name, "DIRECT_PARITY_EXECUTION_STATUS")
    for field, filename in (("selection_lock_sha256", "SELECTION_LOCK.json"), ("cost_lock_sha256", "COST_LOCK.json"),
                            ("parity_sha256", "ACTUAL_PARITY_DIRECT_INTERPRETER.json"), ("statistics_sha256", "FOLD_STATISTICS.json")):
        a.equal(receipt[field], digest(HERE / filename), path.name + "/" + field, "DIRECT_PARITY_EXECUTION_HASH")
    a.equal(receipt["F_model_hash"], selection["F_model_hash"], path.name, "DIRECT_PARITY_EXECUTION_MODEL")
    parity = read_json(HERE / "ACTUAL_PARITY_DIRECT_INTERPRETER.json")
    a.equal(receipt["F_model_hash"], parity["F_model_hash"], path.name, "DIRECT_PARITY_EXECUTION_MODEL")
    a.equal(receipt["statistics_sha256"], parity["statistics_sha256"], path.name, "DIRECT_PARITY_EXECUTION_HASH")
    a.check(cost_lock["time_unix"] < receipt["started_unix"] <= receipt["finished_unix"] < first["time_unix"],
            "DIRECT_PARITY_EXECUTION_TIME_ORDER", path.name)
    a.row(attempt_id="parity/direct_interpreter_execution_receipt", logical_id="direct_interpreter_actual_parity",
          category="PARITY_EXECUTION_RECEIPT", block="R5_COST_PRECHECK", status=receipt["status"],
          source=path.name, source_sha256=digest(path), logical_fit_count=0, fitting_configurations=0,
          optimizer_updates=0, incremental_updates=0, updates_exact=True, zero_no_fit=True,
          cost_measure_trials=0, cost_warmup_trials=0, cost_error_trials=0,
          reason="Self-produced actual F-parity execution receipt after cost lock and before first C; audit only validates saved bindings/order")
    a.details["direct_parity_execution"] = dict(receipt=receipt,
        limitation="Root-produced execution timestamps and hashes are audited; this is not an independent replay of F or independent process tracing.")


def audit_launcher_attempt(a, selection, cost_lock, first):
    failure_path = HERE / "EXCLUSIVE_LAUNCHER_FAILURE.json"
    parity_path = HERE / "DIRECT_INTERPRETER_PARITY.json"
    if not failure_path.is_file() and not parity_path.is_file():
        return
    if not a.check(failure_path.is_file() and parity_path.is_file(), "LAUNCHER_ATTEMPT_RECEIPTS", "exclusive_launcher"):
        return
    failure, parity = read_json(failure_path), read_json(parity_path)
    original = read_json(HERE / "RUNTIME_ENVIRONMENT.json")
    general = read_json(HERE / "GENERAL_COST_COMPLETE.json")
    a.equal(failure, dict(status="FAILED_PRESERVED", general_cost_preserved=True,
                         exclusive_cost_folders=[], C_scored=False), "EXCLUSIVE_LAUNCHER_FAILURE", "LAUNCHER_FAILURE_RECEIPT")
    a.check("COMPETING_RESEARCH_PROCESS" in failure.get("observed_error", ""),
            "LAUNCHER_FAILURE_GUARD", "EXCLUSIVE_LAUNCHER_FAILURE")
    tree = failure["process_tree"]
    pids = [row["pid"] for row in tree]
    a.check(bool(tree) and len(pids) == len(set(pids)), "LAUNCHER_PROCESS_OBSERVATION", "process_tree")
    observed_routes, modes, wrapper_edges = set(), set(), []
    by_pid = {row["pid"]: row for row in tree}
    for row in tree:
        command = row["cmdline"]
        a.check(isinstance(row["pid"], int) and row["pid"] > 0 and isinstance(row["ppid"], int) and row["ppid"] > 0,
                "LAUNCHER_PID_SCHEMA", row["pid"])
        normalized = [str(value).replace("\\", "/").lower() for value in command]
        a.check(any(value.endswith("cost_and_score.py") for value in normalized), "LAUNCHER_COMMAND_SCOPE", row["pid"])
        mode = "exclusive" if "exclusive" in command else "exclusive_all" if "exclusive_all" in command else None
        modes.add(mode)
        if "--id" in command:
            observed_routes.add(command[command.index("--id") + 1])
        parent = by_pid.get(row["ppid"])
        if parent and parent["cmdline"][1:] == command[1:]:
            wrapper_edges.append([parent["pid"], row["pid"]])
    a.check(modes == {"exclusive", "exclusive_all"} and bool(wrapper_edges),
            "LAUNCHER_ANCESTRY_OBSERVATION", "process_tree")
    a.check(bool(observed_routes) and observed_routes.issubset({spec["id"] for spec in selection["routes"]}),
            "LAUNCHER_ROUTE_SCOPE", "process_tree", str(sorted(observed_routes)))
    source = ast.parse((HERE / "cost_and_score.py").read_text(encoding="utf-8-sig"))
    exclusive = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "exclusive")
    first_call = exclusive.body[0].value if isinstance(exclusive.body[0], ast.Expr) else None
    guarded_before_model = isinstance(first_call, ast.Call) and isinstance(first_call.func, ast.Name) and first_call.func.id == "assert_no_training"
    guarded_before_model = guarded_before_model and any(keyword.arg == "check_exclusive" and isinstance(keyword.value, ast.Constant)
                                                       and keyword.value.value is True for keyword in first_call.keywords)
    a.check(guarded_before_model, "LAUNCHER_FAILURE_BEFORE_MODEL_SOURCE", "exclusive")
    a.equal(parity, dict(status="SAME_RUNTIME_MODULE_PATHS_PASS", scientific_sources_changed=False,
                        first_C_score_exists=False, cuda_available=True), "DIRECT_INTERPRETER_PARITY", "DIRECT_RUNTIME_PROBE")
    a.equal(parity["selection_lock_sha256"], digest(HERE / "SELECTION_LOCK.json"), "direct_runtime", "LOCK_HASH")
    a.equal(parity["version"], original["python"], "direct_runtime/python", "DIRECT_RUNTIME_VERSION")
    a.equal(parity["cuda_runtime"], original["cuda_runtime"], "direct_runtime/cuda", "DIRECT_RUNTIME_VERSION")
    modules = parity["modules"]
    a.check(set(modules) == {"torch", "numpy", "chronos", "transformers", "psutil", "sklearn"},
            "DIRECT_RUNTIME_MODULE_COVERAGE", "direct_runtime")
    venv_site = ROOT / ".venv" / "Lib" / "site-packages"
    for name, module in modules.items():
        expected = venv_site / name / "__init__.py"
        a.equal(str(Path(module["path"]).resolve()).lower(), str(expected.resolve()).lower(),
                "direct_runtime/" + name, "DIRECT_RUNTIME_MODULE_PATH")
        a.check(isinstance(module.get("version"), str) and bool(module["version"]),
                "DIRECT_RUNTIME_MODULE_VERSION", "direct_runtime/" + name)
    a.equal(modules["torch"]["version"], original["torch"], "direct_runtime/torch", "DIRECT_RUNTIME_VERSION")
    executable = str(Path(parity["executable"]).resolve()).lower()
    a.check(any(str(Path(row["cmdline"][0]).resolve()).lower() == executable for row in tree),
            "DIRECT_RUNTIME_BASE_EXECUTABLE", "direct_runtime")
    a.equal(str(Path(parity["prefix"]).resolve()).lower(), str(Path(parity["executable"]).parent.resolve()).lower(),
            "direct_runtime/prefix", "DIRECT_RUNTIME_BASE_EXECUTABLE")
    general_end = max(utc_seconds(meta["finished_utc"]) for meta in general["metadata"])
    a.check(general_end <= failure["time_unix"] <= parity["time_unix"] <= cost_lock["time_unix"] <= first["time_unix"],
            "LAUNCHER_REPAIR_TIME_ORDER", "exclusive_launcher")
    final_receipts = []
    for outcome in cost_lock["exclusive"]:
        path = HERE / "cost_exclusive" / outcome["id"] / "COST_RUN_METADATA.json"
        try:
            metadata = read_json(path)
            a.equal(metadata, dict(status="COMPLETE_VALID", python=parity["version"].split()[0],
                                   torch=modules["torch"]["version"], cuda_runtime=parity["cuda_runtime"]),
                    outcome["id"], "DIRECT_RUNTIME_FINAL_COST")
            a.check(parity["time_unix"] <= utc_seconds(metadata["started_utc"]),
                    "DIRECT_RUNTIME_BEFORE_COST", outcome["id"])
            final_receipts.append(dict(id=outcome["id"], metadata_sha256=digest(path)))
        except Exception as error:
            a.check(False, "DIRECT_RUNTIME_FINAL_COST_READ", path, f"{type(error).__name__}: {error}")
    a.row(attempt_id="cost_launcher/exclusive_all/initial", logical_id="exclusive_all_initial_launcher",
          category="COST_LAUNCHER_FAILURE", block="R5_COST", status=failure["status"],
          source=failure_path.name, source_sha256=digest(failure_path), logical_fit_count=0,
          fitting_configurations=0, optimizer_updates=0, incremental_updates=0, updates_exact=True,
          zero_no_fit=True, cost_measure_trials=0, cost_warmup_trials=0, cost_error_trials=0, cost_failed_queue_attempts=1,
          reason="Preserved queue failed at pre-model guard; process-tree observations are not a complete child-attempt count")
    a.row(attempt_id="cost_environment/direct_interpreter", logical_id="direct_interpreter_parity",
          category="COST_ENVIRONMENT_PROBE", block="R5_COST", status="DIRECT_INTERPRETER_PREFLIGHT_RECORDED",
          source=parity_path.name, source_sha256=digest(parity_path), logical_fit_count=0,
          fitting_configurations=0, optimizer_updates=0, incremental_updates=0, updates_exact=True,
          zero_no_fit=True, cost_measure_trials=0, cost_warmup_trials=0, cost_error_trials=0, cost_environment_probes=1,
          reason="Runtime module-path/version probe; successful cost execution is assessed separately from this probe")
    supporting = {}
    guard_errors = 0
    for filename in ("EXCLUSIVE_LAUNCHER_FAILURE_LOG.txt", "EXECUTION_REPAIR_KO.md"):
        path = HERE / filename
        if not a.check(path.is_file(), "LAUNCHER_SUPPORTING_RECEIPT", filename): continue
        text = safe_path(path).read_text(encoding="utf-8-sig")
        supporting[filename] = digest(path)
        if filename.endswith("_LOG.txt"):
            guard_errors = len(re.findall(r"^RuntimeError: COMPETING_RESEARCH_PROCESS:\[[^\r\n]*\]$", text, re.MULTILINE))
            a.check(guard_errors > 0, "LAUNCHER_PRESERVED_ERROR_LOG", filename)
            a.row(attempt_id="cost_launcher/preserved_error_log", logical_id="exclusive_all_initial_launcher",
                  category="COST_LAUNCHER_ERROR_LOG", block="R5_COST", status="PARTIAL_LOG_PRESERVED",
                  source=filename, source_sha256=supporting[filename], logical_fit_count=0,
                  fitting_configurations=0, optimizer_updates=0, incremental_updates=0, updates_exact=True,
                  zero_no_fit=True, cost_measure_trials=0, cost_warmup_trials=0, cost_error_trials=0,
                  cost_guard_errors_observed=guard_errors,
                  reason="Count of guard-error lines in retained terminal tail; not a complete child/route attempt census")
    a.details["exclusive_launcher_attempt"] = dict(failure=failure, direct_interpreter_probe=parity,
        observed_processes=len(tree), observed_route_ids=sorted(observed_routes), same_argv_parent_child_edges=wrapper_edges,
        failed_queue_attempts=1, failed_child_attempt_count=None, final_exclusive_cost_receipts=final_receipts,
        partial_log_guard_errors_observed=guard_errors, supporting_receipt_sha256=supporting,
        limitation="Process-tree snapshot is not a complete execution log. Non-torch historical module versions were not independently recorded; saved paths/versions are checked without importing or hashing external modules. No environment probe is counted as a timing trial.")


# MAIN_AUDIT_ENTRY

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-saved-results-audit", action="store_true",
                        help="ROOT explicitly authorizes post-SCORE_COMPLETE saved-artifact recomputation")
    parser.add_argument("--attempt", choices=("v1", "v2"), default="v1",
                        help="v2 preserves attempt1 files and records the CSV-reader-only repair")
    args = parser.parse_args()
    if not args.execute_saved_results_audit:
        parser.error("Preparation only: execution requires ROOT authorization and --execute-saved-results-audit")
    required = ("SCORE_COMPLETE.json", "SELECTION_LOCK.json", "COST_LOCK.json", "FIRST_C_SCORE.json", "SCORES.csv")
    if any(not (HERE / name).is_file() for name in required):
        raise SystemExit("Not ready: completed scoring and selection/cost/first-C receipts are required")
    audit_file, index_file = ("AUDIT_v2.json", "RUN_INDEX_v2.csv") if args.attempt == "v2" else ("AUDIT.json", "RUN_INDEX.csv")
    output_names = (audit_file, index_file, "AUDIT_LATEST.json") if args.attempt == "v2" else (audit_file, index_file)
    for name in output_names:
        if (HERE / name).exists():
            raise FileExistsError(f"Preserve previous audit; ROOT must select a fresh attempt before rerunning: {name}")
    prior_files = {}
    if args.attempt == "v2":
        prior_files = {name: digest(HERE / name) for name in ("AUDIT.json", "RUN_INDEX.csv", "AUDIT_EXECUTION_01.log")}
    started = time.time()
    a = Audit()
    selection = read_json(HERE / "SELECTION_LOCK.json")
    cost_selection = read_json(HERE / "COST_SELECTION.json")
    cost_lock = read_json(HERE / "COST_LOCK.json")
    first = read_json(HERE / "FIRST_C_SCORE.json")
    complete = read_json(HERE / "SCORE_COMPLETE.json")
    contract = read_json(HERE / "DATA_CONTRACT.json")
    stats = read_json(HERE / "FOLD_STATISTICS.json")
    a.check(complete["status"] == "COMPLETE_VALID", "SCORING_COMPLETION", "SCORE_COMPLETE", complete.get("failures", []))
    a.section("contract", lambda: audit_contract(a, contract, stats))
    a.section("locks", lambda: audit_locks(a, selection, cost_selection, cost_lock, first, complete, stats))
    a.section("training", lambda: audit_training(a, selection, contract, stats))
    a.section("attempts", lambda: audit_attempts(a, selection, contract))
    a.section("predictions", lambda: audit_predictions(a, selection, contract, stats, complete))
    a.section("parity", lambda: audit_parity(a, selection, contract))
    if (HERE / "DIRECT_INTERPRETER_PARITY.json").is_file():
        direct_parity = "ACTUAL_PARITY_DIRECT_INTERPRETER.json"
        if a.check((HERE / direct_parity).is_file(), "DIRECT_ACTUAL_PARITY_RECEIPT", direct_parity):
            a.section("parity_direct_interpreter", lambda: audit_parity(a, selection, contract, direct_parity))
        a.section("direct_parity_execution", lambda: audit_direct_parity_execution(a, selection, cost_lock, first))
    a.section("cost", lambda: audit_costs(a, selection, cost_selection, cost_lock))
    a.section("launcher_attempt", lambda: audit_launcher_attempt(a, selection, cost_lock, first))
    for failure in complete.get("failures", []):
        a.check(False, "RECORDED_SCORING_FAILURE", failure["id"], failure.get("error", ""))
    totals = dict(logical_fits=sum(int(row.get("logical_fit_count", 0)) for row in a.run_rows),
                  fitting_configurations=sum(int(row.get("fitting_configurations", 0)) for row in a.run_rows),
                  optimizer_updates_excluding_probe=sum(int(row.get("optimizer_updates", 0)) for row in a.run_rows if row["category"] != "PROBE"),
                  probe_updates=sum(int(row.get("optimizer_updates", 0)) for row in a.run_rows if row["category"] == "PROBE"),
                  cached_prediction_origins=sum(int(row.get("prediction_origins", 0)) for row in a.run_rows if "SCORING" in row["category"]),
                  precision_measure_trials=sum(int(row.get("cost_measure_trials", 0)) for row in a.run_rows),
                  precision_warmup_trials=sum(int(row.get("cost_warmup_trials", 0)) for row in a.run_rows),
                  precision_error_trials=sum(int(row.get("cost_error_trials", 0)) for row in a.run_rows),
                  cost_failed_queue_attempts=sum(int(row.get("cost_failed_queue_attempts", 0)) for row in a.run_rows),
                  cost_environment_probes=sum(int(row.get("cost_environment_probes", 0)) for row in a.run_rows),
                  cost_guard_errors_observed=sum(int(row.get("cost_guard_errors_observed", 0)) for row in a.run_rows),
                  cost_launcher_failure_receipts=sum(row["category"] == "COST_LAUNCHER_FAILURE" for row in a.run_rows),
                  failure_receipts=sum(row["category"] == "FAILURE" for row in a.run_rows),
                  orphan_checkpoint_attempts=sum(row["category"] == "ORPHAN_ATTEMPT" for row in a.run_rows))
    manifest_names = set(required) | {"COST_SELECTION.json", "GENERAL_COST_COMPLETE.json", "DATA_CONTRACT.json",
                                    "FOLD_STATISTICS.json", "PROBE.json", "R1_B.json", "R2_SELECTIONS.json",
                                    "R3_SELECTIONS.json", "R4_SELECTIONS.json", "R5_SELECTIONS.json", "COST.csv", "ACTUAL_PARITY.json",
                                    "EXCLUSIVE_LAUNCHER_FAILURE.json", "DIRECT_INTERPRETER_PARITY.json", "RUNTIME_ENVIRONMENT.json",
                                    "EXECUTION_REPAIR_KO.md", "EXCLUSIVE_LAUNCHER_FAILURE_LOG.txt", "ACTUAL_PARITY_DIRECT_INTERPRETER.json",
                                    "DIRECT_PARITY_EXECUTION.json"}
    inputs = {name: digest(HERE / name) for name in sorted(manifest_names) if (HERE / name).is_file()}
    errors = [item for item in a.findings if item["severity"] == "ERROR"]
    result = dict(status="PASS" if not errors else "PATH_FINDINGS", started_unix=started, finished_unix=time.time(),
                  audit_attempt=args.attempt, audit_file=audit_file, run_index_file=index_file,
                  csv_field_limit_characters=CSV_FIELD_LIMIT, prior_files=prior_files,
                  audit_source_sha256=digest(Path(__file__)), checks=dict(a.checks), error_count=len(errors),
                  warning_count=len(a.findings) - len(errors), findings=a.findings, details=a.details,
                  run_index_rows=len(a.run_rows), run_index_totals=totals, input_sha256=inputs,
                  validation_scope="Saved prediction/mask/metadata/cost arithmetic and pinned new source/checkpoints; no original archive or F weights read",
                  original_data_reads=0, foundation_model_loads=0, GPU_calls=0,
                  independent_training_reruns=0, independent_timer_reruns=0,
                  interpretation="Findings identify affected paths/receipts; they do not erase valid paths or establish method failure")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RUN_FIELDS)
    writer.writeheader(); writer.writerows(a.run_rows)
    index_text = buffer.getvalue()
    result["run_index_sha256"] = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    json_text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with (HERE / index_file).open("x", encoding="utf-8", newline="") as stream:
        stream.write(index_text)
    with (HERE / audit_file).open("x", encoding="utf-8", newline="") as stream:
        stream.write(json_text)
    if args.attempt == "v2" and not errors:
        for name, expected in prior_files.items():
            if digest(HERE / name) != expected:
                raise RuntimeError(f"Prior audit artifact changed during re-audit: {name}")
        latest = dict(status=result["status"], audit_file=audit_file, audit_sha256=digest(HERE / audit_file),
                      run_index_file=index_file, run_index_sha256=digest(HERE / index_file), prior_files=prior_files,
                      repair_reason="CSV reader default field limit only; no scientific/cost/prediction change")
        with (HERE / "AUDIT_LATEST.json").open("x", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(latest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], errors=len(errors), run_index_rows=len(a.run_rows), totals=totals), ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
