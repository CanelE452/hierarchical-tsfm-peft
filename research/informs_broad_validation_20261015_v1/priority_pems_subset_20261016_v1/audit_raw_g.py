"""Independent PeMS C32 Raw/G audit; default zero input reads.

Only explicitly bound metadata, public sufficient statistics and physical cost
trials are interpreted. No experiment, model, array, T target or OS query runs.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

NAME = "priority_pems_subset_20261016_v1"
RUNNER_SHA = "caf332f445121ae03249c23023fde916fefb17ab0eb7685dbd9cefe7be566381"
DRAFT_SHA = "b255fec55dcdfe49898d024193cbc7f0efe60100679b29e8e1cdd89bb7969693"
GUARD_REL = "postT_cpu_lag_preparation_20261016_v1/cpu_group_supervisor.py"
GUARD_SHA = "82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
DEADLINE = 1792222550
COHORT = "pems08:C32_hash:primary:V_select:stride12"
ORIGINS = tuple(range(12499, 14284 - 12 + 1, 12))
BANDS = ((0, 3), (3, 6), (6, 12))
RUNS = {family: "postT_cpu_pems08_E1_SUBSET_" + family + "_k0_s92601_C32_priority_v1" for family in ("RAW", "DIRECT_G")}
PREPARATION = "priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/PEMS_RAW_G_DISPATCH_PREP_v1/"
ORCHESTRATION_PINS = {
    "dispatcher_source": (PREPARATION + "RUN_DISPATCH_DRAFT_v1.ps1", "22d7cfef749b065350a6afe157076e404b449d974c2259e75f6fc7e4280a940b"),
    "post_source": (PREPARATION + "RUN_POST_DRAFT_v1.ps1", "a1d7dc403e5492fd67e40407f5aeb30ffd267518cd4062ac7e28bd2791f91435"),
}
PRIVATE_HELPER_PINS = {
    "priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/FAMILY_INITIAL_DISPATCH_PREP_v1/RUN_ROOT_DRAFT_v1.ps1": "67dfce149dc8c3f8b84221d5b7c8da1e3f2233a3e7058ee2182c09fdc63cc1cc",
    "priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/FAMILY_INITIAL_POST_PREP_v1/RUN_ROOT_DRAFT_v1.ps1": "57cf98eda8f8d1b2c6303ba336860733a41454c470dd28c0e18485d81e257982",
}
HELPERS = {
    "inputs": ("priority_valid_minima_20261016_v1/build_beijing_report.py", "8bb730a55e0f0f924de0616bf917ca4b47e4117503027cfc4656c6ef502750dd"),
    "arithmetic": ("priority_valid_minima_20261016_v1/independent_beijing_audit.py", "933883a052b02e6a41ccdadaa5ec45e80b977dbfaada007607ac1093f518dd04"),
    "guard": ("priority_valid_family_20261016_v1/audit_initial_family.py", "f92a3a7b4bc03677f67d3bb7c42758765a95f49a378e2dd9d768ffc7f4de6241"),
    "runner": (NAME + "/subset_minimum_runner.py", RUNNER_SHA),
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def counts(family, batches, full=False):
    width = 32 if family == "RAW" else 0
    if family not in RUNS:
        raise ValueError("Only the original Raw/G C32 recipes")
    streams = sum(batches) * width
    return dict(streams=streams, physical_streams=streams, physical_calls=sum(1 if full and width else math.ceil(b * width / 128) for b in batches))


def aggregate_statistics(rows, identifier, channels, observed):
    if len(channels) != 32 or len(set(channels)) != 32 or identifier not in RUNS.values():
        raise ValueError("Exactly the actual hash-ordered C32 channels required")
    fields = {"run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count"}
    expected = {(o, str(c), a, b) for o in ORIGINS for c in channels for a, b in BANDS}
    seen = {}
    for row in rows:
        if set(row) != fields or (row["run_id"], row["split"], row["cohort"]) != (identifier, "V_select", COHORT):
            raise ValueError("Actual C32 sufficient-statistic schema and identity required")
        key = (int(row["origin"]), row["channel"], int(row["horizon_start"]), int(row["horizon_end"]))
        n, sse, sae = int(row["count"]), float(row["sse"]), float(row["sae"])
        if key not in expected or key in seen or not 0 <= n <= key[3] - key[2] or any(not math.isfinite(v) or v < 0 for v in (sse, sae)) or (n == 0 and (sse != 0 or sae != 0)):
            raise ValueError("Foreign, duplicate, nonfinite or impossible band statistic")
        seen[key] = (sse, sae, n)
    if set(seen) != expected:
        raise ValueError("All 14208 actual origin/channel/band rows, including empty bands, required")
    per_channel = []
    for channel in channels:
        group = [v for k, v in seen.items() if k[1] == str(channel)]
        n = sum(v[2] for v in group)
        sse, sae = math.fsum(v[0] for v in group), math.fsum(v[1] for v in group)
        per_channel.append(dict(channel=channel, count=n, sse=sse, sae=sae, mse=sse / n if n else None, mae=sae / n if n else None))
    active = [r for r in per_channel if r["count"]]
    n = sum(r["count"] for r in per_channel)
    if not active or n != observed:
        raise ValueError("Observed count is taken from actual evidence, never assumed fully observed")
    return dict(mse=math.fsum(r["mse"] for r in active) / len(active), mae=math.fsum(r["mae"] for r in active) / len(active),
        n_observed=n, active_channels=len(active), public_stats_rows=len(seen), per_channel=per_channel,
        count_by_key=[dict(origin=k[0], channel=k[1], horizon_start=k[2], horizon_end=k[3], count=v[2]) for k, v in sorted(seen.items())])


def aggregate_trials(rows, identifier, family, batch):
    if identifier != RUNS.get(family) or batch not in (1, 8):
        raise ValueError("Actual Raw/G B1/B8 routes only")
    stages = tuple("stage_" + name + "_seconds" for name in ("CPU_preprocess", "host_to_device", "F" if family == "RAW" else "G_or_direct", "inverse_and_CPU"))
    fields = {"run_id", "phase", "trial", "order_position", "batch_size", "whole_seconds", "F_logical_streams", "F_physical_streams", "F_calls", "prediction_sha256", "row_microbatch", "stage_instrumentation", *stages}
    phases = dict(warmup=10, timed=20, stage_probe=20)
    expected = {(p, i) for p, n in phases.items() for i in range(n)}
    wanted = counts(family, [batch]); seen = {}; totals = [0, 0, 0]; hashes = set()
    for row in rows:
        key = (row.get("phase"), int(row.get("trial", -1))); instrumented = key[0] == "stage_probe"
        if set(row) != fields or key not in expected or key in seen or row["run_id"] != identifier or int(row["batch_size"]) != batch or int(row["order_position"]) != 0 or int(row["row_microbatch"]) != 128 or row["stage_instrumentation"] != str(instrumented) or not digest(row["prediction_sha256"]):
            raise ValueError("Exact family-specific four-stage 50-trial schema/identity required")
        counter = tuple(int(row[k]) for k in ("F_logical_streams", "F_physical_streams", "F_calls"))
        if counter != (wanted["streams"], wanted["physical_streams"], wanted["physical_calls"]):
            raise ValueError("Actual per-trial logical/physical F counters differ")
        seconds = float(row["whole_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Finite positive whole timing required")
        for field in stages:
            if instrumented:
                value = float(row[field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Finite recorded stage timing required")
            elif row[field] != "":
                raise ValueError("Ordinary whole trials cannot contain intrusive stage timings")
        seen[key] = seconds; totals = [a + b for a, b in zip(totals, counter)]; hashes.add(row["prediction_sha256"])
    if set(seen) != expected:
        raise ValueError("Complete 10 warmup/20 ordinary/20 separate stage trials required")
    summaries = []
    for phase in ("timed", "stage_probe"):
        values = sorted(v for k, v in seen.items() if k[0] == phase); position = 19 * .9; low = math.floor(position); fraction = position - low
        summaries.append(dict(run_id=identifier, phase=phase, batch_size=batch, trials=20, median_seconds=statistics.median(values), mean_seconds=math.fsum(values) / 20,
            min_seconds=values[0], max_seconds=values[-1], p90_seconds=values[low] + (values[low + 1] - values[low]) * fraction,
            stage_instrumentation=phase == "stage_probe", stage_sum_equals_whole_claimed=False))
    return dict(trial_rows=50, phase_rows=phases, summaries=summaries, F_logical_streams=totals[0], F_physical_streams=totals[1], F_calls=totals[2],
        derived_trial_F_scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS", unique_prediction_hashes=len(hashes), actual_stage_fields=list(stages))


def validate_training(receipt, search=None, refit=None):
    family = receipt.get("family")
    want = dict(status="COMPLETE_VALID", run_id=RUNS.get(family), seed=92601, actual_C=32, L=512, H=12, registered_latent_k=0,
        A50_end=8928, A60_end=10713, full_parent_C=170, full_parent_identity_inherited=True, new_T_numeric_values_parsed=0,
        independent_generalization_claimed=False, old_C170_training_budget_matched=False, PCA_fit_time_separately_recorded=False)
    if family not in RUNS or any(receipt.get(k) != v for k, v in want.items()) or not all(digest(receipt.get(k)) for k in ("artifact_sha256", "model_state_sha", "refit_scaler_sha", "actual_subset_prefix_sha256", "data_sha256")):
        raise ValueError("Actual original C32 recipe, artifact and scaler lineage required")
    if family == "RAW":
        want = dict(selected_epoch=None, search_updates=0, refit_updates=0, search_exposures=0, refit_exposures=0, training_status="NO_TRAINING",
            artifact_kind="RAW_METADATA_IDENTITY_NO_TRAINED_ADAPTER", trained_adapter_checkpoint_exists=False)
        if search is not None or refit is not None or any(receipt.get(k) != v for k, v in want.items()):
            raise ValueError("Raw is optimizer-zero metadata identity, not a trained adapter")
        return
    epoch = receipt.get("selected_epoch")
    if epoch not in (0., .25, 1., 3.):
        raise ValueError("Original selected endpoint, including zero, required")
    exposures = math.ceil(epoch * 10190); updates = math.ceil(exposures / 16)
    want = dict(search_updates=1577, search_exposures=25215, refit_updates=updates, refit_exposures=exposures,
        artifact_kind="TRAINED_SUBSET_BUNDLE", source_stage_API="original_fit_base._stage",
        training_status="UNDERTRAINED_OR_BUDGET_LIMIT" if epoch == 3 else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
    if any(receipt.get(k) != v for k, v in want.items()):
        raise ValueError("Segmented 1577-update search and fresh selected refit, with unresolved convergence, required")
    for stage, expected in ((search, dict(status="SEGMENT_COMPLETE", completed_epochs=[0., .25, 1., 3.], train_origins=8405, exposures=25215, updates=1577)),
            (refit, dict(status="SEGMENT_COMPLETE", completed_epochs=[0.] + ([epoch] if epoch else []), train_origins=10190, exposures=exposures, updates=updates))):
        if not isinstance(stage, dict) or any(stage.get(k) != v for k, v in expected.items()):
            raise ValueError("Original search/refit stage activity differs")
    for label in ("A50", "A60"):
        values = receipt.get("initialization_by_stage", {}).get(label, {}).get("fingerprints", {})
        if set(values) != {"mean", "std", "observed_counts", "basis", "pca_center"} or values["basis"] is not None or values["pca_center"] is not None or any(not digest(values[k]) for k in ("mean", "std", "observed_counts")):
            raise ValueError("Direct G has no PCA basis/center; actual A50/A60 normalization hashes required")


def validate_validation(value, mode, family, identity, observed):
    if any(value.get(k) != v for k, v in identity.items()) or value.get("new_T_numeric_values_parsed") != 0:
        raise ValueError("Validation must bind the actual selected C32 route")
    if mode == "rescore":
        want = dict(status="PASS_INDEPENDENT_FULL_V_FP64", full_V_origins=148, actual_C=32, H=12, n_observed=observed, public_stats_rows=14208,
            saved_predictions_rescored=True, new_model_inference=False, protected_T_target_operations=0, torch_imported=False)
    else:
        origins = value.get("origins", [])
        want = dict(status="PASS_FRESH_CPU6_REPLAY", executed_now=True, microbatches=[4, 2], replay_selection_seed=92601,
            F_counters=counts(family, [4, 2]), atol=1e-5, rtol=1e-4, full_V_reinference_claimed=False,
            foundation_instances_created=1 if family == "RAW" else 0)
        if len(origins) != 6 or len(set(origins)) != 6 or not set(origins).issubset(ORIGINS) or not {ORIGINS[0], ORIGINS[74], ORIGINS[-1]}.issubset(origins):
            raise ValueError("Actual deterministic six origins including fixed endpoints required")
    if any(value.get(k) != v for k, v in want.items()):
        raise ValueError("Separate FP64 full-V / fresh CPU6 evidence differs")


def validate_parity(value, batch, identity):
    variants = [dict(row_microbatch="full", F_counters=counts("RAW", [batch], full=True)), dict(row_microbatch=128, F_counters=counts("RAW", [batch]))]
    wanted = dict(status="PASS_RAW_ROW128_VS_FULL", batch_size=batch, all_passed=True, actual_raw_forward=True, actual_optimizer_updates=0,
        historical_original_Raw_failures_resolved=False, new_T_numeric_values_parsed=0, per_variant_F_counters=variants,
        F_counters={key: sum(v["F_counters"][key] for v in variants) for key in ("streams", "physical_streams", "physical_calls")})
    original = value.get("original_raw_row_parity", {})
    if any(value.get(k) != v for k, v in dict(identity, **wanted).items()) or original.get("all_passed") is not True or original.get("atol") != 1e-5 or original.get("rtol") != 1e-4 or [r.get("row_microbatch") for r in original.get("variants", [])] != ["full", 128] or any(r.get("status") != "PASS" for r in original.get("variants", [])):
        raise ValueError("This batch's actual full-versus-row128 physical outputs and independently observed counters must pass")


def validate_foundation_state(value, family, foundation_sha):
    if value.get("foundation_instances_created") != (1 if family == "RAW" else 0):
        raise ValueError("Actual Raw single instance / G zero foundation instances required")
    if family == "RAW" and value.get("F_final_state_sha256") != foundation_sha:
        raise ValueError("Actual Raw foundation must remain frozen")
    if family == "DIRECT_G" and "F_final_state_sha256" in value:
        raise ValueError("Absent G foundation cannot be represented by a fabricated F hash")


def load_helpers(public):
    result = {}
    for key, (name, expected) in HELPERS.items():
        path = Path(public) / name
        if sha(path) != expected:
            raise ValueError("Frozen stdlib helper/source changed")
        spec = importlib.util.spec_from_file_location("pems_raw_g_" + key, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); result[key] = module
    return result


def bound_json(inputs, ref, root):
    if set(ref) != {"root", "relative_path", "sha256"} or ref["root"] != root or not digest(ref["sha256"]):
        raise ValueError("Actual explicit root/relative-path/SHA descriptor required, never pending null")
    return inputs.json(root, ref["relative_path"], ref["sha256"])


def native_json(inputs, ref, expected, absent):
    if ref is None:
        if inputs.path("public", expected).exists():
            raise ValueError("Existing native evidence cannot be hidden behind a missing descriptor")
        absent.append(expected)
        return None
    if ref.get("relative_path") != expected:
        raise ValueError("Actual native evidence path must match its observed guard ID")
    return bound_json(inputs, ref, "public")


def require_arg(command, flag, expected, path=False):
    if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise ValueError("One actual command argument required: " + flag)
    value = command[command.index(flag) + 1]
    differs = Path(value).resolve() != Path(expected).resolve() if path else value != str(expected)
    if differs:
        raise ValueError("Actual command binding differs: " + flag)


def validate_deadline(command, recorded):
    if command.count("--deadline") != 1 or command.index("--deadline") + 1 >= len(command):
        raise ValueError("One explicit actual worker deadline argument required")
    deadline = float(command[command.index("--deadline") + 1])
    if type(recorded) not in (int, float) or not math.isfinite(deadline) or deadline != recorded or not 0 < deadline <= DEADLINE:
        raise ValueError("Actual bounded soft deadline and original hard upper limit must remain unchanged")


def closed_binding(inputs, binding):
    values = {key: bound_json(inputs, entry, "cache") for key, entry in binding.items() if key in ("spec", "guard", "closure")}
    if set(values) != {"spec", "guard", "closure"} or any(binding[k]["root"] != "cache" for k in values):
        raise ValueError("Actual private guard/SPEC/current-owner closure required")
    spec, guard, closure = (values[k] for k in ("spec", "guard", "closure"))
    if (guard.get("schema") != "CPU_GROUP_SUPERVISOR_RESULT_v1" or guard.get("status") not in ("COMPLETE", "STOPPED_RESOURCE_OR_WORKER_FAILURE")
            or guard.get("helper_sha256") != GUARD_SHA or guard.get("helper_unchanged") is not True or guard.get("spec_file_sha256") != binding["spec"]["sha256"]
            or guard.get("supervisor_failure_class") is not None or guard.get("remaining_owned_processes") != [] or spec.get("schema") != "CPU_GROUP_SUPERVISOR_v1"
            or spec.get("deadline_unix") != DEADLINE or not guard.get("processes") or len(guard["processes"]) != len(spec.get("processes", []))):
        raise ValueError("Actual terminal CPU2/RSS3GiB guard/SPEC binding required")
    effective = guard.get("effective_spec", {})
    if len(spec["processes"]) not in (1, 2) or {p["id"] for p in spec["processes"]} != {p["id"] for p in guard["processes"]} or effective.get("max_each_RSS") != 3221225472 or effective.get("max_group_RSS") not in (3221225472, 6442450944) or guard.get("sampled_group_peak_RSS_bytes", math.inf) > effective["max_group_RSS"]:
        raise ValueError("Actual original one/two-worker group limits and identity set required")
    for key in ("schema", "deadline_unix", "max_each_RSS", "max_group_RSS", "poll_seconds", "terminate_grace_seconds", "kill_grace_seconds"):
        if effective.get(key) != spec.get(key):
            raise ValueError("Recorded effective guard settings differ from actual SPEC")
    for key in ("asset_root", "cache_root", "output_root"):
        if Path(effective[key]).resolve() != Path(spec[key]).resolve():
            raise ValueError("Recorded effective guard root differs from actual SPEC")
    for original in spec["processes"]:
        actual = [p for p in effective.get("processes", []) if p["id"] == original["id"]]
        if len(actual) != 1 or actual[0].get("command") != original["command"] or actual[0].get("max_seconds") != original["max_seconds"] or Path(actual[0]["cwd"]).resolve() != Path(original["cwd"]).resolve():
            raise ValueError("Actual effective worker argv/cwd/time limit differs")
    owned = []; last = 0
    for child in guard["processes"]:
        if (child.get("returncode") is None or child.get("end_unix") is None or child.get("end_unix") > DEADLINE or child.get("observed_tree_join_confirmed") is not True
                or child.get("process_tree_errors") != [] or child.get("remaining_owned_processes") != [] or not child.get("owned_processes") or child.get("sampled_peak_RSS_bytes", math.inf) > 3221225472):
            raise ValueError("Every actually executed worker must have preserved strong tree closure")
        root = child.get("root", {})
        if sum(p.get("pid") == root.get("pid") and p.get("create_time") == root.get("create_time") for p in child["owned_processes"]) != 1:
            raise ValueError("Actual owned root PID/create_time must occur exactly once")
        if any(type(p.get("pid")) is not int or type(p.get("create_time")) not in (int, float) for p in child["owned_processes"]):
            raise ValueError("Typed actual PID/create_time identity required")
        owned += [dict(pid=p["pid"], create_time=p["create_time"]) for p in child["owned_processes"]]
        last = max(last, child["end_unix"])
    owned.sort(key=lambda p: (p["pid"], p["create_time"]))
    if len({(p["pid"], p["create_time"]) for p in owned}) != len(owned):
        raise ValueError("Observed ownership identities cannot be duplicated")
    want = dict(schema="ROOT_OPAQUE_METADATA_OWNER_ABSENCE_v2", status="VERIFIED_STRONG_OBSERVED_TREE_CLOSURE_AND_CURRENT_ABSENCE",
        guard_sha256=binding["guard"]["sha256"], spec_sha256=binding["spec"]["sha256"], actual_PID_creation_time_check=True,
        remaining_exact_owners=[], errors=[], absent_identities=owned)
    if any(closure.get(k) != v for k, v in want.items()) or not last <= closure.get("checked_unix", -1) <= time.time():
        raise ValueError("Saved actual ROOT post-terminal PID/create-time absence required")
    workers = binding.get("workers", [])
    if len(workers) != len(spec["processes"]) or {w["id"] for w in workers} != {w["id"] for w in spec["processes"]}:
        raise ValueError("Only actual observed worker records, never invented skipped guards")
    return spec, guard, owned


def collect_guards(inputs, helpers, execution, plan_sha):
    records = {}; costs = []; other = []; owners = []; evidence = []; absent = []
    seen_guards = set()
    for binding in execution["executed_guards"]:
        if binding["guard"]["sha256"] in seen_guards:
            raise ValueError("Each actual guard group must appear once")
        seen_guards.add(binding["guard"]["sha256"])
        spec, guard, owned = closed_binding(inputs, binding); owners += owned
        for record in binding["workers"]:
            identity = record["identity"]; run_id, mode, batch = (identity.get(k) for k in ("run_id", "mode", "batch"))
            if run_id not in RUNS.values() or mode not in ("fit", "rescore", "replay", "raw_parity", "cost") or ((mode in ("raw_parity", "cost")) != (batch in (1, 8))) or (mode == "raw_parity" and run_id != RUNS["RAW"]):
                raise ValueError("Exactly original Raw/G fit/validation/batch-parity/cost workers only")
            key = (run_id, mode, batch)
            if key in records:
                raise ValueError("No repeated attempt or cherry-picked same-condition worker")
            outer_ref, started_ref = record["outer"], record["started"]
            if not isinstance(record["id"], str) or not record["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in record["id"]):
                raise ValueError("Actual canonical native diagnostic ID required")
            folder = NAME + "/diagnostics/" + record["id"]
            outer = native_json(inputs, outer_ref, folder + "/RESULT.json", absent)
            started = native_json(inputs, started_ref, folder + "/STARTED.json", absent)
            selected = [p for p in spec["processes"] if p["id"] == record["id"]]
            if len(selected) != 1:
                raise ValueError("Actual selected child must exist once in SPEC")
            command = selected[0]["command"]
            for flag, value in (("--run-id", run_id), ("--mode", mode), ("--plan-sha", plan_sha)):
                require_arg(command, flag, value)
            for flag, value in (("--public-root", inputs.roots["public"]), ("--cache-root", inputs.roots["cache"]), ("--output", inputs.path("public", folder))):
                require_arg(command, flag, value, True)
            require_arg(command, "--asset-root", spec["asset_root"], True)
            if sum(Path(a).resolve() == inputs.path("public", NAME + "/subset_minimum_runner.py").resolve() for a in command if not a.startswith("-")) != 1 or command.count("--execute") != 1 or command.count("--root-triggered") != 1:
                raise ValueError("Actual original N6 runner/explicit ROOT invocation required")
            if batch is not None:
                require_arg(command, "--batch", batch)
            recorded_deadline = started["deadline_unix"] if started is not None else float(command[command.index("--deadline") + 1])
            validate_deadline(command, recorded_deadline)
            if any(v.get(k) != x for v in (outer, started) if v is not None for k, x in dict(run_id=run_id, mode=mode, plan_sha256=plan_sha).items()):
                raise ValueError("Actual outer/STARTED/argv identities differ")
            child = next(p for p in guard["processes"] if p["id"] == record["id"])
            if started is not None and not any(p["pid"] == started.get("actual_pid") and abs(p["create_time"] - started.get("actual_create_time", -1)) <= 1e-6 for p in child["owned_processes"]):
                raise ValueError("Native STARTED PID/create_time must match actual owned child tree")
            if child["returncode"] == 0 and child.get("stop_reason") is None and started is not None and outer is not None:
                helpers["guard"].validate_guard(guard, spec, binding["spec"]["sha256"], record["id"], command, started)
            else:
                if outer is not None and outer.get("status") not in ("FAILED_PRESERVED", "BUDGET_LIMIT_PRESERVED", "INCOMPLETE"):
                    raise ValueError("Failed/partial worker cannot be represented as a successful native outcome")
            interval = dict(id=record["id"], start=child["start_unix"], end=child["end_unix"])
            (costs if mode == "cost" else other).append(interval)
            proof = dict(id=record["id"], run_id=run_id, mode=mode, batch=batch, spec_sha256=binding["spec"]["sha256"], guard_sha256=binding["guard"]["sha256"],
                outer_sha256=outer_ref["sha256"] if outer_ref else None, STARTED_sha256=started_ref["sha256"] if started_ref else None, closure_sha256=binding["closure"]["sha256"],
                outer_public_relative_path=folder + "/RESULT.json", native_outer_exists=outer is not None, native_STARTED_exists=started is not None,
                outer_status=outer.get("status") if outer else "MISSING_PRESERVED", error_class=outer.get("error_class") if outer else None, postflight_error_class=outer.get("postflight_error_class") if outer else None,
                sampled_peak_RSS_bytes=child["sampled_peak_RSS_bytes"], returncode=child["returncode"], current_PID_query_performed_by_auditor=False)
            evidence.append(proof)
            records[key] = dict(outer=outer, started=started, command=command, child=child, folder=folder, evidence=proof)
    helpers["arithmetic"].validate_intervals(costs, other)
    unique = sorted({(p["pid"], p["create_time"]) for p in owners})
    current = execution["ROOT_current_owner_absence"]
    current = bound_json(inputs, current, "cache")
    wanted = dict(schema="ROOT_PEMS_RAW_G_CURRENT_ABSENCE_v1", status="VERIFIED_CURRENT_EMPTY",
        actual_PID_creation_time_check=True, remaining_exact_owners=[], errors=[], active_model_workers=[], active_cost_workers=[],
        absent_identities=[dict(pid=p, create_time=c) for p, c in unique], executed_guards_sha256=fingerprint(execution["executed_guards"]))
    if any(current.get(k) != v for k, v in wanted.items()) or not 0 <= time.time() - current.get("checked_unix", -1) <= 120:
        raise ValueError("Actual fresh ROOT whole-current-worker/CIM absence proof required")
    return records, evidence, costs, absent


def successful(record, status):
    outer = record["outer"]
    wanted = dict(status=status, error_class=None, postflight_error_class=None, original_source39_immutable9_preserved=True,
        new_T_numeric_values_parsed=0, actual_CPU_threads=2, actual_GPU_seconds=0, original_registry_changed=False, independent_generalization_claimed=False)
    if outer is None or record["started"] is None or record.get("native_wrapper_success") is not True or record["child"]["returncode"] != 0 or record["child"].get("stop_reason") is not None or any(outer.get(k) != v for k, v in wanted.items()):
        raise ValueError("Actual successful native child/outer/source boundary required")
    return outer["scientific_result"]


def validation_record(inputs, records, family, mode, identity, observed):
    record = records[(RUNS[family], mode, None)]
    status = "PASS_INDEPENDENT_FULL_V_FP64" if mode == "rescore" else "PASS_FRESH_CPU6_REPLAY"
    value = successful(record, status)
    native = inputs.json("public", record["folder"] + ("/RESCORE_RECEIPT.json" if mode == "rescore" else "/REPLAY_RECEIPT.json"))
    if any(value.get(k) != v for k, v in native.items()):
        raise ValueError("Native receipt must match actual outer scientific result")
    validate_validation(value, mode, family, identity, observed)
    return value, record


def accuracy_row(inputs, helpers, plan, family, records, plan_sha):
    rid = RUNS[family]; folder = NAME + "/runs/" + rid; prediction = NAME + "/predictions/" + rid + "/V_select"
    condition = next(c for c in plan["conditions"] if c["run_id"] == rid)
    receipt = inputs.json("public", folder + "/BASE_FIT_RECEIPT.json")
    stages = [inputs.json("public", folder + "/" + stage + "/RECEIPT.json") for stage in ("search", "refit")] if family == "DIRECT_G" else [None, None]
    validate_training(receipt, *stages)
    if receipt.get("plan_sha256") != plan_sha:
        raise ValueError("Selected fit belongs to another C32 PLAN")
    identity = receipt["subset_identity_payload"]
    expected = helpers["runner"].subset_identity_payload(plan["parent_task_contract"]["data_sha256"], plan["column_selection"], identity["actual_subset_prefix_array_sha256"])
    if identity != expected or fingerprint(identity) != receipt["data_sha256"] or receipt["actual_subset_prefix_sha256"] != fingerprint(identity["actual_subset_prefix_array_sha256"]):
        raise ValueError("Actual pre-T subset identity must be newly derived, never relabeled C170 identity")
    fit_record = records[(rid, "fit", None)]
    access = inputs.json("public", fit_record["folder"] + "/PREFIX_ACCESS_RECEIPT.json")
    access_expected = dict(materialized_rows=14284, materialized_bounds=[0, 14284], actual_C=32, new_T_numeric_values_parsed=0,
        inherited_identity_status="INHERITED_NOT_RECOMPUTED", full_subset_T_identity_computed=False, column_selection=plan["column_selection"],
        subset_identity_payload=identity, derived_subset_data_sha256=receipt["data_sha256"], actual_subset_prefix_array_sha256=identity["actual_subset_prefix_array_sha256"])
    if any(access.get(k) != v for k, v in access_expected.items()) or len(access.get("members", [])) != 1 or access["members"][0].get("tail_numeric_values_parsed") != 0 or access["members"][0].get("materialized_source_shape") != [14284, 170, 3]:
        raise ValueError("Actual prefix-only parent source and new C32 subset access evidence required")
    source = inputs.json("public", folder + "/SOURCE_CONTRACT.json")
    source_files = {name: plan["source39"][name] for name in ("fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py")}
    expected_model = plan["foundation_tensor_sha256"] if family == "RAW" else "NO_FOUNDATION"
    if (source.get("spec") != condition["spec"] or source.get("plan_sha256") != plan_sha or source.get("subset_identity_payload") != identity
            or source.get("config_sha") != fingerprint(dict(spec=condition["spec"], subset_identity=identity)) or source.get("data_sha") != receipt["data_sha256"]
            or source.get("model_sha") != expected_model or source.get("files") != source_files or source.get("source_sha") != fingerprint(source["files"]) or receipt["source_sha"] != source["source_sha"]):
        raise ValueError("Actual G/Raw recipe/source/derived-task ancestry differs")
    for name, expected_sha in source["files"].items():
        inputs.capture("public", name, expected_sha)
    artifact_root = "public" if family == "RAW" else "cache"
    artifact_name = folder + ("/RAW_CONTRACT.json" if family == "RAW" else "/selected_bundle.pt")
    inputs.capture(artifact_root, artifact_name, receipt["artifact_sha256"])
    if family == "RAW":
        raw = inputs.json("public", artifact_name, receipt["artifact_sha256"])
        if raw.get("optimizer_updates") != 0 or raw.get("foundation_sha256") != plan["foundation_tensor_sha256"] or raw.get("scaler_sha") != receipt["refit_scaler_sha"] or receipt["model_state_sha"] != plan["foundation_tensor_sha256"]:
            raise ValueError("Actual Raw metadata identity/frozen foundation/zero optimizer required")
    else:
        for stage, n in (("search", 4), ("refit", 1 + int(receipt["selected_epoch"] != 0))):
            for index in range(n):
                inputs.capture("cache", folder + "/" + stage + "/checkpoint_" + str(index).zfill(6) + ".pt")
        for stage, end in zip(stages, (8928, 10713)):
            if any(stage.get("contract", {}).get(k) != source[k] for k in ("source_sha", "config_sha", "data_sha", "model_sha")):
                raise ValueError("Actual original stage source/NO_FOUNDATION contract differs")
            contract = stage["contract"]
            if contract.get("initialization_sha") != fingerprint(dict(scaler_sha=contract["scaler_sha"], fit_end=end, basis_sha=None, center_sha=None)):
                raise ValueError("Actual A50/A60 normalization-only initialization must bind the correct prefix")
        if stages[1].get("contract", {}).get("scaler_sha") != receipt["refit_scaler_sha"]:
            raise ValueError("Actual A60 scaler must match selected fresh refit")
    fitted = records[(rid, "fit", None)]; fit_science = successful(fitted, "COMPLETE_FIT_AND_FULL_V")
    if fit_science.get("fit_receipt") != receipt or fit_science.get("full_V_origins") != 148:
        raise ValueError("Actual full-V fit must bind the same selected receipt")
    manifest = inputs.json("public", prediction + "/PREDICTION_MANIFEST.json")
    manifest_sha = inputs.files[("public", prediction + "/PREDICTION_MANIFEST.json")]
    canonical = helpers["runner"].canonical_route(condition, receipt, plan["foundation_tensor_sha256"])
    wanted = dict(run_id=rid, family=family, seed=92601, split="V_select", cohort=COHORT, device="cpu", actual_C=32, H=12, K=0,
        data_sha=receipt["data_sha256"], scaler_sha=receipt["refit_scaler_sha"], artifact_sha256=receipt["artifact_sha256"],
        source_sha=fingerprint(dict(runner=RUNNER_SHA, **plan["dependencies"])), config_sha=fingerprint(condition["spec"]),
        model_sha=fingerprint(dict(artifact_sha=receipt["artifact_sha256"], model_state_sha=receipt["model_state_sha"])), canonical_route_sha256=fingerprint(canonical),
        F_counters=counts(family, [4] * 37), prefix_rows=14284, new_T_numeric_values_parsed=0, old_fullC_training_budget_matched=False, independent_generalization_claimed=False)
    if any(manifest.get(k) != v for k, v in wanted.items()) or fit_science.get("V_score") != manifest["score"]:
        raise ValueError("Actual V148/C32/H12/model/scaler/F accounting differs")
    score = manifest["score"]
    if any(score.get(k) != v for k, v in dict(n_origins=148, num_channels=32, horizon=12, metric_scale="A_normalized_original_channels").items()) or not 0 < score["n_observed"] <= 148 * 12 * 32:
        raise ValueError("Actual full-V geometry and observed-label metadata required")
    reference = inputs.json("public", prediction + "/REFERENCE_CONTRACT.json")
    for key in ("run_id", "split", "cohort", "data_sha", "scaler_sha", "source_sha", "config_sha", "model_sha", "canonical_route_sha256"):
        if reference.get(key) != manifest.get(key):
            raise ValueError("Pre-inference reference and final manifest identity differ")
    axes = ("target_normalized", "observed", "origins", "channel_ids")
    if reference.get("arrays") != {key: manifest["arrays"][key] for key in axes} or reference.get("subset_identity_payload") != identity or reference.get("canonical_route") != canonical:
        raise ValueError("Actual pre-inference reference axes and canonical route required")
    shapes = dict(pred_normalized=[148, 12, 32], target_normalized=[148, 12, 32], observed=[148, 12, 32], origins=[148], channel_ids=[32])
    if set(manifest["arrays"]) != set(shapes) or any(manifest["arrays"][k].get("shape") != v or not digest(manifest["arrays"][k].get("sha256")) for k, v in shapes.items()):
        raise ValueError("Exact saved C32 axes and array fingerprints required")
    for key, dtype in (("pred_normalized", "float32"), ("target_normalized", "float32"), ("observed", "bool"), ("origins", "int64")):
        if manifest["arrays"][key].get("dtype") != dtype:
            raise ValueError("Original saved FP32/bool/int64 axes must retain their declared dtype")
    channel_dtype = manifest["arrays"]["channel_ids"].get("dtype", "")
    if channel_dtype not in ("<U" + str(max(map(len, plan["column_selection"]["selected_ids"]))), ">U" + str(max(map(len, plan["column_selection"]["selected_ids"])) )):
        raise ValueError("Saved hash-ordered channel IDs must retain actual fixed-width Unicode metadata")
    inputs.capture("cache", NAME + "/predictions/" + rid + "/V_select.npz", manifest["archive_sha256"])
    result_identity = dict(run_id=rid, artifact_sha256=receipt["artifact_sha256"], manifest_sha256=manifest_sha, data_sha256=receipt["data_sha256"], canonical_route_sha256=fingerprint(canonical))
    rescore, rescore_record = validation_record(inputs, records, family, "rescore", result_identity, score["n_observed"])
    replay, replay_record = validation_record(inputs, records, family, "replay", result_identity, score["n_observed"])
    for value in (fit_science, replay):
        validate_foundation_state(value, family, plan["foundation_tensor_sha256"])
    stats = helpers["arithmetic"].gzip_rows(inputs, "public", prediction + "/PUBLIC_ERROR_STATS.csv.gz", 14208)
    aggregate = aggregate_statistics(stats, rid, plan["column_selection"]["selected_ids"], score["n_observed"])
    scores = inputs.csv("public", prediction + "/SCORES.csv")
    if len(scores) != 1 or any(scores[0].get(k) != manifest[k] for k in ("run_id", "split", "cohort")):
        raise ValueError("One exact original full-V score row required")
    deltas = helpers["arithmetic"].compare_accuracy(aggregate, manifest, scores[0], rescore)
    row = dict(run_id=rid, family=family, actual_C=32, L=512, H=12, seed=92601, mse=aggregate["mse"], mae=aggregate["mae"],
        n_observed=aggregate["n_observed"], active_channels=aggregate["active_channels"], public_stats_rows=14208, full_V_origins=148,
        fit_receipt=receipt, search_stage=stages[0], refit_stage=stages[1], artifact_sha256=receipt["artifact_sha256"], manifest_sha256=manifest_sha,
        reference_axis_fingerprints={key: manifest["arrays"][key] for key in axes}, source_identity=source["model_sha"], metric_deltas=deltas,
        accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6", cost_status={}, costs={}, F_only_resident_bytes=None,
        F_only_resident_scope="NOT_SEPARATELY_RECORDED", actual_parameter_numel=None, parameter_numel_scope="NOT_RECORDED_BY_NATIVE_ROUTE_MEMORY_API",
        full_C170_or_B2_comparison=False, independent_generalization_claimed=False)
    proofs = {"rescore": rescore_record, "replay": replay_record}
    return row, aggregate, result_identity, proofs


def cost_row(inputs, helpers, family, batch, record, identity, proofs, records, dispatch_sha, foundation_sha):
    value = successful(record, "COMPLETE_PROTOCOL")
    native = inputs.json("public", record["folder"] + "/COST_RECEIPT.json")
    if any(value.get(k) != v for k, v in native.items()):
        raise ValueError("Native cost receipt and actual outer differ")
    prerequisites = dict(proofs)
    if family == "RAW":
        parity = records[(RUNS[family], "raw_parity", batch)]
        actual = successful(parity, "PASS_RAW_ROW128_VS_FULL")
        saved = inputs.json("public", parity["folder"] + "/RAW_PARITY_RECEIPT.json")
        if any(actual.get(k) != v for k, v in saved.items()):
            raise ValueError("Actual native per-batch Raw parity receipt differs")
        validate_parity(actual, batch, identity); prerequisites["raw_parity"] = parity
    validation_hashes = {mode: p["evidence"]["outer_sha256"] for mode, p in prerequisites.items()}
    for mode, proof in prerequisites.items():
        require_arg(record["command"], "--" + mode.replace("_", "-") + "-proof", inputs.path("public", proof["folder"] + "/RESULT.json"), True)
        require_arg(record["command"], "--" + mode.replace("_", "-") + "-sha", proof["evidence"]["outer_sha256"])
    command = record["command"]
    if command.count("--quiet-proof") != 1:
        raise ValueError("Actual fresh quiet proof argv required")
    quiet_path = Path(command[command.index("--quiet-proof") + 1]).resolve()
    if not quiet_path.is_relative_to(inputs.roots["cache"]):
        raise ValueError("Actual quiet proof stays private")
    quiet = inputs.json("cache", quiet_path.relative_to(inputs.roots["cache"]).as_posix(), value["quiet_proof_sha256"])
    require_arg(command, "--quiet-sha", value["quiet_proof_sha256"])
    if quiet.get("status") != "QUIET_CPU_COST_WINDOW" or quiet.get("active_model_workers") != [] or quiet.get("active_cost_workers") != [] or not 0 <= record["child"]["start_unix"] - quiet.get("captured_unix", -1) <= 120:
        raise ValueError("Actual CPU2 cost must have fresh 120-second empty model/cost window")
    expected = dict(identity, status="COMPLETE_PROTOCOL", device="cpu", CPU_threads=2, batch_size=batch, warmups=10, timed_trials=20, stage_trials=20,
        actual_trial_rows=50, validation_result_sha256=validation_hashes, F_counters_per_trial=counts(family, [batch]), row_microbatch=128,
        physical_output_and_inverse_dtype="float64", cached_Z_or_forecast_reads=0, training_peak_included=False,
        stage_probe_separate_intrusive_scope=True, new_T_numeric_values_parsed=0, origins=list(ORIGINS[:batch]))
    if any(value.get(k) != v for k, v in expected.items()) or not all(digest(value.get(k)) for k in ("trials_sha256", "physical_input_sha256", "observation_mask_sha256")):
        raise ValueError("Actual full physical route/batch/validation/precision/counters differ")
    memory = value.get("memory", {})
    if memory.get("foundation_instances_created") != (1 if family == "RAW" else 0) or not math.isfinite(float(memory.get("resident_weights_cpu_MiB", math.nan))) or memory["resident_weights_cpu_MiB"] <= 0 or memory.get("resident_weights_gpu_MiB") != 0:
        raise ValueError("Actual per-route parameter/buffer resident bytes and separate F instance count required")
    if not math.isfinite(float(value.get("model_load_seconds", math.nan))) or value["model_load_seconds"] < 0:
        raise ValueError("Cold model construction/load is separately recorded, never inferred as zero")
    validate_foundation_state(value, family, foundation_sha)
    trials = helpers["arithmetic"].gzip_rows(inputs, "public", record["folder"] + "/COST_TRIALS.csv.gz", 50, value["trials_sha256"])
    aggregate = aggregate_trials(trials, RUNS[family], family, batch)
    helpers["arithmetic"].compare_summaries(aggregate["summaries"], value["summaries"])
    return dict(run_id=RUNS[family], family=family, batch=batch, receipt=native, **aggregate,
        memory_scope=memory, guard_sampled_peak_RSS_bytes=record["evidence"]["sampled_peak_RSS_bytes"],
        recorded_scope="WARM_WHOLE_PHYSICAL_ROUTE_WITH_SEPARATE_INTRUSIVE_STAGE_PROBE", cold_model_load_in_whole_trial=False,
        F_only_resident_bytes=None, F_only_resident_scope="NOT_SEPARATELY_RECORDED", model_load_seconds=value["model_load_seconds"],
        dispatch_terminal_sha256=dispatch_sha, native_F_counters_per_trial=value["F_counters_per_trial"])


def expected_keys():
    return {(rid, mode, batch) for family, rid in RUNS.items() for mode, batch in
        (("fit", None), ("rescore", None), ("replay", None), ("cost", 1), ("cost", 8))} | {(RUNS["RAW"], "raw_parity", b) for b in (1, 8)}


def bind_native_execution(inputs, execution, records):
    sources = {}
    for key in ("dispatcher_source", "post_source"):
        ref = execution[key]
        if ref != dict(root="cache", relative_path=ORCHESTRATION_PINS[key][0], sha256=ORCHESTRATION_PINS[key][1]):
            raise ValueError("Exact reviewed private Raw/G orchestration source required")
        inputs.capture(ref["root"], ref["relative_path"], ref["sha256"]); sources[key + "_sha256"] = ref["sha256"]
    native = {}
    for key in ("dispatch", "post"):
        ref = execution[key]
        expected_name = "RESULT.json" if key == "dispatch" else "POST_RESULT.json"
        if ref["root"] != "cache" or ref["relative_path"] != PREPARATION + expected_name:
            raise ValueError("Original terminal ROOT wrapper remains private and unchanged")
        native[key] = bound_json(inputs, ref, "cache")
        if (native[key].get("error") is not None or native[key].get("plan_sha256") != execution["plan_sha256"] or native[key].get("pins_before_after_unchanged") is not True
                or native[key].get("changed_or_unreadable_pinned_paths") != [] or native[key].get("original_worker_deadline") != DEADLINE or native[key].get("original_campaign_deadline") != 1792223150):
            raise ValueError("Original actual wrapper source/resource/PLAN boundary not verified")
    if native["post"].get("fit_dispatch_result_sha256") != execution["dispatch"]["sha256"]:
        raise ValueError("Actual POST must bind the original fit terminal result")
    if native["dispatch"].get("schema") != "PEMS_RAW_G_TWO_FIT_RESULT_v1" or native["post"].get("schema") != "PEMS_RAW_G_POST_RESULT_v1":
        raise ValueError("Exact native Raw/G wrapper schemas required")
    for wrapper in native.values():
        if wrapper.get("dispatcher_source_sha256") != sources["dispatcher_source_sha256"] or wrapper.get("source39_immutable9_count") != 48 or wrapper.get("actual_C") != 32 or wrapper.get("full_parent_C") != 170 or wrapper.get("B0_B2_automatically_started") is not False or wrapper.get("whole_MASTER_complete") is not False:
            raise ValueError("Actual wrapper must retain original source48/C32-only scope and admission")
    if native["post"].get("post_source_sha256") != sources["post_source_sha256"] or native["dispatch"].get("strong_observed_tree_closure_verified") is not True or native["dispatch"].get("original_four_conditions_preserved") is not True or native["post"].get("new_training") != 0:
        raise ValueError("Actual native sources/closure/original recipes/no-new-POST-training required")
    steps, attempts = native["post"].get("steps", []), native["post"].get("attempts", [])
    by_id = {row["id"]: row for row in steps}; attempt_by_id = {row["id"]: row for row in attempts}
    if len(by_id) != len(steps) or len(attempt_by_id) != len(attempts) or set(by_id) != set(attempt_by_id):
        raise ValueError("All actual successful and failed attempts remain in both original native lists")
    for identifier, row in by_id.items():
        if any(attempt_by_id[identifier].get(k) != row.get(k) for k in ("run_id", "family", "mode", "batch", "success", "spec_sha256", "guard_sha256", "outer_sha256", "started_sha256")) or row.get("strong_observed_tree_closure_verified") is not True:
            raise ValueError("Original attempt/step SHA, status and observed-tree closure must agree")
    if native["post"].get("fit_dispatch_SPEC_sha256") != native["dispatch"].get("dispatch_spec_sha256") or native["post"].get("actual_fit_guard_sha256") != native["dispatch"].get("dispatch_guard_sha256"):
        raise ValueError("Actual fit guard/SPEC must remain bound across both native wrappers")
    context_values = {}
    for key in ("admission", "predecessor", "preparation"):
        ref = execution[key]
        if ref["root"] != "cache" or native["dispatch"].get(key + "_sha256") != ref["sha256"]:
            raise ValueError("Actual original admission/predecessor byte binding required")
        context_values[key] = bound_json(inputs, ref, "cache")
    admission, predecessor = (context_values[k] for k in ("admission", "predecessor"))
    if execution["preparation"]["relative_path"] != PREPARATION + "PLAN_DRAFT.json" or context_values["preparation"].get("status") != "PREP_ONLY_NOT_ADMITTED" or context_values["preparation"].get("run_ids") != [RUNS["RAW"], RUNS["DIRECT_G"]]:
        raise ValueError("Exact original private Raw/G preparation must remain unchanged")
    want = dict(schema="PEMS_RAW_G_ADMISSION_v1", status="SEALED_AND_METADATA_VERIFIED_NOT_EXECUTED", plan_sha256=execution["plan_sha256"],
        dispatcher_sha256=sources["dispatcher_source_sha256"], preparation_sha256=execution["preparation"]["sha256"], predecessor_sha256=execution["predecessor"]["sha256"],
        original_four_conditions_preserved=True, B0_B2_automatically_started=False, original_worker_deadline=DEADLINE, budgets_reinitialized_or_extended=False,
        new_T_numeric_parsed=0, GPU_added_seconds=0)
    if any(admission.get(k) != v for k, v in want.items()) or sorted(admission.get("run_ids", [])) != sorted(RUNS.values()):
        raise ValueError("Exact separate Raw/G-only admission and original resource/T scope required")
    if predecessor.get("schema") != "PEMS_RAW_G_N5_PREDECESSOR_v1" or predecessor.get("status") != "ROOT_RECONCILED_CLOSED_REPORTED_AND_PUBLISHED" or predecessor.get("remote_tree_and_HTTP_verified") is not True or predecessor.get("whole_MASTER_complete") is not False:
        raise ValueError("Actual closed/reported/published current N5 predecessor, including honest partial, required")
    for key, wanted in (("dispatch_started", dict(script_sha256=sources["dispatcher_source_sha256"], plan_sha256=execution["plan_sha256"],
            admission_sha256=execution["admission"]["sha256"], predecessor_sha256=execution["predecessor"]["sha256"], preparation_sha256=execution["preparation"]["sha256"])),
            ("post_started", dict(script_sha256=sources["post_source_sha256"], dispatcher_source_sha256=sources["dispatcher_source_sha256"],
                fit_dispatch_result_sha256=execution["dispatch"]["sha256"], fit_dispatch_SPEC_sha256=native["dispatch"]["dispatch_spec_sha256"],
                actual_fit_guard_sha256=native["dispatch"]["dispatch_guard_sha256"], plan_sha256=execution["plan_sha256"], new_training=0))):
        ref = execution[key]
        if ref["root"] != "cache" or ref["relative_path"] != PREPARATION + ("STARTED.json" if key == "dispatch_started" else "POST_STARTED.json"):
            raise ValueError("Actual wrapper STARTED remains private")
        value = bound_json(inputs, ref, "cache")
        if any(value.get(k) != v for k, v in wanted.items()):
            raise ValueError("Actual wrapper STARTED/source/admission/fit provenance differs")
    rows = [(r, "fit") for r in native["dispatch"].get("conditions", [])] + [(r, r.get("mode")) for r in native["post"].get("steps", [])]
    native_ids = set()
    for row, mode in rows:
        batch = row.get("batch") if mode in ("cost", "raw_parity") else None
        key = (row.get("run_id"), mode, batch)
        if key not in records or key in native_ids:
            raise ValueError("Native terminal rows and actual observed workers must match exactly")
        if row.get("family") != next(family for family, rid in RUNS.items() if rid == key[0]):
            raise ValueError("Native terminal family label must match its actual original run")
        native_ids.add(key); actual = records[key]["evidence"]
        expected = dict(id=row.get("id"), spec_sha256=native["dispatch"].get("dispatch_spec_sha256"), guard_sha256=native["dispatch"].get("dispatch_guard_sha256"), outer_sha256=row.get("outer_sha256")) if mode == "fit" else row
        for field in ("id", "spec_sha256", "guard_sha256", "outer_sha256"):
            if expected.get(field) != actual[field]:
                raise ValueError("Actual native terminal/source SPEC/guard/outer binding differs")
        if row.get("started_sha256") != actual["STARTED_sha256"]:
            raise ValueError("Actual native STARTED byte binding differs")
        records[key]["native_wrapper_success"] = row.get("worker_science_complete") if mode == "fit" else row.get("success")
    if native_ids != set(records):
        raise ValueError("No actual executed guard may be silently omitted")
    skipped = native["dispatch"].get("skipped", []) + native["post"].get("skipped", [])
    skip_keys = {(r.get("run_id"), r.get("mode"), r.get("batch")) for r in skipped}
    if len(skip_keys) != len(skipped) or skip_keys & set(records) or set(records) | skip_keys != expected_keys() or any(row.get("actual_guard_created") is not False for row in skipped):
        raise ValueError("Every original condition must be actually executed or truthfully skipped without invented guards")
    return native, skipped


def collect(inputs, helpers, execution):
    if execution.get("schema") != "PEMS_RAW_G_ROOT_EXECUTION_BINDINGS_v1" or execution.get("status") != "SEALED_BY_ROOT" or execution.get("whole_MASTER_complete") is not False:
        raise ValueError("Actual ROOT metadata seal required; DRAFT never supplies successful evidence")
    for path, expected in HELPERS.values():
        inputs.capture("public", path, expected)
    inputs.capture("cache", GUARD_REL, GUARD_SHA)
    for name, expected in PRIVATE_HELPER_PINS.items():
        inputs.capture("cache", name, expected)
    draft = inputs.json("public", NAME + "/PLAN_DRAFT.json", DRAFT_SHA)
    plan = inputs.json("public", NAME + "/PLAN.json", execution["plan_sha256"])
    if plan.get("status") != "SEALED_BY_ROOT" or plan.get("runner_sha256") != RUNNER_SHA or any(plan.get(k) != v for k, v in draft.items() if k not in ("status", "runner_sha256", "actual_training_prediction_replay_cost_or_child_operations", "ROOT_seal_required")):
        raise ValueError("Original four-recipe PLAN/rules/source48/resources/T lock must remain unchanged")
    if plan.get("conditions") != helpers["runner"].conditions() or len(plan["source39"]) != 39 or len(plan["immutable_public"]) != 9 or len(set(plan["source39"]) | set(plan["immutable_public"])) != 48:
        raise ValueError("Exact four original recipes and immutable source48 required, not four executed conditions")
    if plan["column_selection"] != helpers["runner"].select_columns(plan["parent_task_contract"]["columns"]):
        raise ValueError("Actual ID-only SHA ordered C32 rule differs")
    for name, expected in {**plan["source39"], **plan["immutable_public"], **plan["dependencies"]}.items():
        inputs.capture("public", name, expected)
    records, guard_evidence, cost_intervals, absent = collect_guards(inputs, helpers, execution, execution["plan_sha256"])
    native, skipped = bind_native_execution(inputs, execution, records)
    rows = []; costs = []; accuracy = []; failures = []
    for family in ("RAW", "DIRECT_G"):
        try:
            row, stats, identity, proofs = accuracy_row(inputs, helpers, plan, family, records, execution["plan_sha256"])
        except Exception as error:
            failures.append(dict(run_id=RUNS[family], stage="ACCURACY", status="NOT_COMPLETE", reason_class=type(error).__name__, failed_audit_check=str(error) if type(error) is ValueError else None, reason="Actual native fit/full-V/FP64/CPU6 identity or arithmetic not all accepted; original evidence preserved"))
            continue
        for batch in (1, 8):
            try:
                cost = cost_row(inputs, helpers, family, batch, records[(RUNS[family], "cost", batch)], identity, proofs, records, execution["dispatch"]["sha256"], plan["foundation_tensor_sha256"])
                row["costs"][str(batch)] = cost; row["cost_status"][str(batch)] = "PASS_ACTUAL_PHYSICAL_TRIALS"; costs.append(cost)
            except Exception as error:
                row["cost_status"][str(batch)] = "NOT_COMPLETE"
                failures.append(dict(run_id=RUNS[family], stage="COST_B" + str(batch), status="NOT_COMPLETE", reason_class=type(error).__name__, failed_audit_check=str(error) if type(error) is ValueError else None, reason="Same-batch prerequisite, Raw parity, whole-route trial or saved summary not accepted; other paths remain independent"))
        rows.append(row); accuracy.append(stats)
    if len(rows) == 2:
        if rows[0]["reference_axis_fingerprints"] != rows[1]["reference_axis_fingerprints"] or rows[0]["fit_receipt"]["data_sha256"] != rows[1]["fit_receipt"]["data_sha256"] or rows[0]["fit_receipt"]["refit_scaler_sha"] != rows[1]["fit_receipt"]["refit_scaler_sha"]:
            raise ValueError("Two methods cannot be paired with mismatched target/mask/origin/channel/scaler/subset identity")
        helpers["arithmetic"].require_same_counts(accuracy)
        for batch in ("1", "8"):
            if all(batch in row["costs"] for row in rows):
                if any(rows[0]["costs"][batch]["receipt"][key] != rows[1]["costs"][batch]["receipt"][key] for key in ("physical_input_sha256", "observation_mask_sha256")):
                    raise ValueError("Paired Raw/G costs must use exactly the same physical batch and mask")
    for row, stats in zip(rows, accuracy):
        row["band_count_binding_sha256"] = fingerprint(stats.pop("count_by_key"))
    native_complete = native["dispatch"].get("status") == "COMPLETE_PEMS_RAW_G_FITS_AND_FULL_V" and native["post"].get("status") == "COMPLETE_PEMS_RAW_G_TEN_POST_STEPS"
    if not native_complete:
        failures.append(dict(stage="NATIVE_WRAPPERS", status="NOT_COMPLETE", reason="Original terminal wrapper status remains incomplete; no automatic reconciliation or status replacement"))
    complete = native_complete and len(rows) == 2 and len(costs) == 4 and not failures and skipped == []
    incomplete = []
    for (run_id, mode, batch), record in records.items():
        if record["native_wrapper_success"] is not True:
            outer = record["outer"] or {}; science = outer.get("scientific_result") or {}; summary = science.get("summary") or {}
            incomplete.append(dict(run_id=run_id, mode=mode, batch=batch, native_outer_status=outer.get("status", "MISSING_PRESERVED"),
                native_scientific_status=science.get("status"), error_class=outer.get("error_class"), worker_returncode=record["child"]["returncode"],
                phase=science.get("phase"), recorded_progress={k: summary[k] for k in ("status", "selected_epoch", "updates", "exposures", "completed_epochs", "train_origins") if k in summary},
                outer_public_relative_path=record["folder"] + "/RESULT.json", saved_point_scope="NATIVE_SUMMARY_ONLY; LAST_CHECKPOINT_NOT_INFERRED"))
    return dict(status="PASS" if complete else "PARTIAL_VALID_EVIDENCE_PRESERVED" if rows else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED", rows=rows, accuracy=accuracy,
        cost_conditions=costs, failures=failures, skipped=skipped, native_incomplete_outcomes=incomplete, missing_public_native_paths=absent, recorded_guards=guard_evidence, recorded_cost_intervals=cost_intervals,
        costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals=True, actual_current_owner_absence_bound=True,
        native_dispatch_status_preserved=native["dispatch"].get("status"), native_POST_status_preserved=native["post"].get("status"),
        totals=dict(new_accuracy_conditions=len(rows), training_conditions_with_audited_accuracy=sum(r["family"] == "DIRECT_G" for r in rows),
            full_V_origins_per_route=148, actual_C=32, public_stat_rows=sum(a["public_stats_rows"] for a in accuracy), cost_conditions=len(costs),
            trial_rows=sum(c["trial_rows"] for c in costs), summary_groups=sum(len(c["summaries"]) for c in costs), summary_numeric_fields_checked=10 * len(costs)),
        source48_binding_sha256=fingerprint({"research/informs_broad_validation_20261015_v1/" + k: v for k, v in {**plan["source39"], **plan["immutable_public"]}.items()}))


def execute(public, cache, bindings_path, bindings_sha, audit_sha, output):
    if sha(__file__) != audit_sha or not digest(bindings_sha) or time.time() >= DEADLINE:
        raise ValueError("Exact actual auditor and private ROOT bindings SHA required")
    helpers = load_helpers(public); inputs = helpers["inputs"].Inputs(public, cache)
    path = Path(bindings_path).resolve()
    if not path.is_relative_to(inputs.roots["cache"]):
        raise ValueError("Actual execution binding remains private")
    execution = inputs.json("cache", path.relative_to(inputs.roots["cache"]).as_posix(), bindings_sha)
    inputs.capture("public", NAME + "/audit_raw_g.py", audit_sha)
    result = {}; error = None
    try:
        result = collect(inputs, helpers, execution)
    except Exception as exception:
        error = dict(reason_class=type(exception).__name__, failed_audit_check=str(exception) if type(exception) is ValueError else None, reason="Required actual execution/source/resource/cohort evidence rejected; inspect preserved native receipts")
    preserved = inputs.preserved() and all(not inputs.path("public", name).exists() for name in result.get("missing_public_native_paths", []))
    status = result.get("status", "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED") if error is None and preserved else "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"
    receipt = dict(result, schema="INDEPENDENT_PEMS_RAW_G_AUDIT_v1", status=status, audit_source_sha256=audit_sha,
        ROOT_bound_execution_sha256=bindings_sha, execution_plan_sha256=execution.get("plan_sha256"), error=error,
        source48_preserved=error is None and preserved, input_SHA_before_after_equal=preserved,
        inputs=[dict(root=root, relative_path=name, sha256=d) for (root, name), d in sorted(inputs.files.items())],
        independent_generalization_claimed=False, full_C170_or_B2_or_Bolt_necessity_proven=False, whole_MASTER_complete=False,
        limitations=["Repeated development V; one seed; upstream preprocessing and time interpolation ancestry UNKNOWN.",
            "Raw has no optimizer; G uses three-search endpoints and a selected fresh refit. Structure and training differ.",
            "C32 is a new hash-ID prefix subset. The old C170 one-epoch run is not a C-only causal contrast.",
            "Neither B2 value nor Bolt necessity is tested by Raw/G alone. Original subset and MASTER coverage remain incomplete.",
            "FP64 saved-array rescore and fresh CPU6 are prior actual bound executions; this auditor recomputes public statistics/trials only.",
            "Whole trial excludes cold model load. CPU model parameter/buffer bytes, process RSS, training peak and F-only residency are distinct.",
            "Native route memory does not record parameter numel or isolated F resident bytes; these remain unavailable."])
    target = Path(output).resolve()
    if not target.is_relative_to(inputs.roots["public"] / NAME) or target.exists():
        raise FileExistsError("CreateNew audit output only inside this public namespace")
    helpers["runner"].new(target, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--root-triggered", action="store_true")
    for name in ("public-root", "cache-root", "bindings", "bindings-sha", "audit-sha", "output"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREPARATION_ONLY_NOT_EXECUTED", input_reads=0, files_written=0, actual_experiment_operations=0))); return 0
    if not args.root_triggered or any(getattr(args, k.replace("-", "_")) is None for k in ("public-root", "cache-root", "bindings", "bindings-sha", "audit-sha", "output")):
        parser.error("ROOT actual bindings/source/output SHA required")
    value = execute(args.public_root, args.cache_root, args.bindings, args.bindings_sha, args.audit_sha, args.output)
    print(json.dumps(dict(status=value["status"], error=value["error"])))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
