"""ROOT-reviewed CPU queue. Default: immutable metadata only; no child starts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
DEADLINE = 1792222550
LAG_NAME = "postT_V_only_cpu_lag_20261016_v1"
PCA_NAME = "postT_V_only_cpu_fixed_pca_20261016_v1"
FAMILIES = ("B0", "B2", "LAG_CAUSAL", "LAG_BIDIR")
PLAN_SHAS = ("fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35",
             "b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930")
GUARD_SHA = "82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"


class UnsafeClosure(RuntimeError):
    pass


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def relative(root, name):
    root, item = Path(root).resolve(), Path(name)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("Root-relative path required")
    result = root / item
    if any(path.is_symlink() for path in [result] + list(result.parents)):
        raise ValueError("Symlink paths are refused")
    result = result.resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path escaped declared root")
    return result


def new_output(path):
    path = Path(path)
    if any(item.is_symlink() for item in [path] + list(path.parents)):
        raise ValueError("Symlink output is refused")
    path.mkdir(parents=True, exist_ok=False)


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_argv(argv, python, script):
    if (not isinstance(argv, list) or len(argv) < 6 or not all(isinstance(x, str) and x for x in argv)
            or argv[1:5] != ["-X", "utf8", "-u", "-B"]
            or Path(argv[0]).resolve() != Path(python).resolve() or Path(argv[5]).resolve() != Path(script).resolve()):
        raise ValueError("Exact Python argv list required; shell commands are forbidden")
    if any("\x00" in x or "\n" in x or "\r" in x for x in argv):
        raise ValueError("Invalid argv bytes")


def expand_tokens(argv, tokens):
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError("Argument list required")
    output = []
    for argument in argv:
        for token, value in tokens.items():
            argument = argument.replace(token, str(value))
        if "{" in argument or "}" in argument:
            raise ValueError("Unknown cost argument token")
        output.append(argument)
    return output


def identity_state(binding):
    import psutil
    try:
        process = psutil.Process(binding["pid"])
        if process.create_time() != binding["create_time"]:
            return "DEAD"
        return "DEAD" if process.status() == psutil.STATUS_ZOMBIE else "LIVE"
    except psutil.NoSuchProcess:
        return "DEAD"
    except (psutil.AccessDenied, KeyError, TypeError):
        return "UNKNOWN"


def closure_state(result, spec_sha, helper_sha, ids, alive, external=()):
    rows = result.get("processes", [])
    if (result.get("schema") != "CPU_GROUP_SUPERVISOR_RESULT_v1"
            or result.get("helper_sha256") != helper_sha or result.get("helper_unchanged") is not True
            or result.get("spec_file_sha256") != spec_sha
            or result.get("status") not in {"COMPLETE", "STOPPED_RESOURCE_OR_WORKER_FAILURE"}
            or len(rows) != len(ids) or {row.get("id") for row in rows} != set(ids)
            or result.get("remaining_owned_processes") != []):
        return "UNSAFE", {"reason": "missing_or_mismatched_terminal_contract"}
    bindings = list(external)
    for row in rows:
        if (row.get("returncode") is None or row.get("end_unix") is None
                or row.get("observed_tree_join_confirmed") is not True
                or row.get("remaining_owned_processes") != [] or row.get("process_tree_errors") != []
                or not row.get("root") or not row.get("owned_processes")):
            return "UNSAFE", {"reason": "worker_closure_unconfirmed", "id": row.get("id")}
        bindings.extend([row["root"]] + row["owned_processes"])
    states = [dict(pid=b.get("pid"), create_time=b.get("create_time"), state=alive(b)) for b in bindings]
    if any(row["state"] == "UNKNOWN" for row in states):
        return "UNSAFE", {"reason": "identity_unknown", "identities": states}
    return ("WAIT" if any(row["state"] == "LIVE" for row in states) else "READY"), {"identities": states}


def wait_for_existing(load_state, check_pins, now, sleep, deadline, interval, event):
    while now() < deadline:
        check_pins()
        status, detail = load_state()
        if status == "READY":
            return detail
        if status == "UNSAFE":
            raise UnsafeClosure(json.dumps(detail))
        event(dict(event="WAIT_EXISTING", at_unix=now(), detail=detail))
        sleep(min(interval, max(0, deadline - now())))
    raise TimeoutError("Deadline reached while waiting; no existing job was stopped")


def sequence(lag_ids, pca_ids, ready, perform, prepare, event):
    rows = []
    def action(phase, run_id, mode, payload=None):
        row = dict(phase=phase, run_id=run_id, mode=mode, **perform(phase, run_id, mode, payload))
        rows.append(row)
        event(dict(event="ACTION_TERMINAL", **row))
        if row.get("closed") is not True:
            raise UnsafeClosure("A newly owned process tree has unknown or live members")
    for run_id in lag_ids:
        good, reason = ready("lag", run_id)
        if good:
            for mode in ("rescore", "replay"):
                action("lag", run_id, mode)
        else:
            rows.append(dict(phase="lag", run_id=run_id, status="SKIPPED_NOT_READY", reason=reason))
    jobs, reason = prepare()
    if reason:
        rows.append(dict(phase="cost", status="SKIPPED_ORIGINAL_ALL4_GATE", reason=reason))
    for job in jobs:
        action("cost", job["job_id"], "measure", job)
    for run_id in pca_ids:
        good, reason = ready("pca", run_id)
        if good:
            rows.append(dict(phase="pca", run_id=run_id, status="EXISTING_COMPLETE_PRESERVED_NO_REFIT"))
        elif reason == "ABSENT_FRESH_PATHS":
            action("pca", run_id, "fit")
        else:
            rows.append(dict(phase="pca", run_id=run_id, status="EXISTING_OR_INVALID_PATH_PRESERVED", reason=reason))
    for run_id in pca_ids:
        good, reason = ready("pca", run_id)
        if good:
            for mode in ("rescore", "replay"):
                action("pca", run_id, mode)
        else:
            rows.append(dict(phase="pca_verify", run_id=run_id, status="SKIPPED_NOT_READY", reason=reason))
    return rows


def metadata(config_path, config_sha):
    config_path = Path(config_path).resolve()
    if not config_sha or sha(config_path) != config_sha:
        raise ValueError("Explicit immutable configuration SHA required")
    config = read(config_path)
    if (config.get("schema") != "POST_T_CPU_SEQUENCE_v1" or config.get("deadline_unix") != DEADLINE
            or config.get("storage_limit_bytes") != 30 * 1024**3
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,70}", config.get("queue_id", ""))
            or config.get("quiet_cost_exclusive_ROOT_authorization") is not True
            or not 1 <= config.get("wait_poll_seconds", 0) <= 30):
        raise ValueError("Queue, deadline, quiet-cost or resource contract differs")
    public, cache, asset = (Path(config[key]).resolve() for key in ("public_root", "cache_root", "asset_root"))
    if not Path(config["python"]).is_file() or config["pins"]["helper"]["file"] != str(Path(__file__).resolve()):
        raise ValueError("Python or queue source identity differs")
    for pin in config["pins"].values():
        if sha(pin["file"]) != pin["sha256"]:
            raise ValueError("Pinned source/input changed: " + Path(pin["file"]).name)
    if config["pins"]["guard"]["sha256"] != GUARD_SHA:
        raise ValueError("Unreviewed CPU guard")
    lag = load_source(config["pins"]["lag_runner"]["file"], "queue_lag_metadata")
    pca = load_source(config["pins"]["pca_runner"]["file"], "queue_pca_metadata")
    lag_plan = lag.metadata(public, cache, asset, PLAN_SHAS[0])
    pca_plan = pca.metadata(public, cache, asset, PLAN_SHAS[1])
    fit_spec = read(config["pins"]["fit_spec"]["file"])
    if ([row["id"] for row in fit_spec["processes"]] != list(FAMILIES)
            or len(config.get("existing_supervisor_bindings", [])) != 2
            or Path(config["existing_result"]) != Path(fit_spec["output_root"]) / "RESULT.json"):
        raise ValueError("Original four-worker or supervisor lifecycle binding differs")
    return config, (lag_plan, pca_plan)


class Runtime:
    def __init__(self, config_path, config_sha, output):
        self.config_path, self.config_sha, self.output = config_path, config_sha, Path(output)
        self.config, self.plans = metadata(config_path, config_sha)
        self.public, self.cache, self.asset = (Path(self.config[key]).resolve() for key in ("public_root", "cache_root", "asset_root"))
        self.guard = load_source(self.config["pins"]["guard"]["file"], "queue_reviewed_CPU_guard")
        self.preparer = load_source(self.config["pins"]["cost_prepare"]["file"], "queue_original_cost_preparer")
        self.counter, self.jobs, self.preparation_output = 0, [], None

    def event(self, value):
        with (self.output / "PROGRESS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def check(self):
        metadata(self.config_path, self.config_sha)
        if time.time() >= DEADLINE - 20:
            raise TimeoutError("Queue deadline reserve reached")

    def storage(self):
        used = sum(path.stat().st_size for root in (self.public, self.cache) for path in root.rglob("*") if path.is_file())
        reserve = 256 * 1024**2
        if used + reserve > self.config["storage_limit_bytes"] or shutil.disk_usage(self.cache).free < reserve:
            raise RuntimeError("Combined 30GiB or free-space admission failed")
        return used

    def ready(self, phase, run_id):
        name = LAG_NAME if phase == "lag" else PCA_NAME
        public, cache = self.public / name, self.cache / name
        receipt_path = public / "runs" / run_id / "BASE_FIT_RECEIPT.json"
        manifest_path = public / "predictions" / run_id / "V_select" / "PREDICTION_MANIFEST.json"
        exists = any((root / kind / run_id).exists() for root in (self.public, self.cache, public, cache) for kind in ("runs", "predictions"))
        if not exists:
            return False, "ABSENT_FRESH_PATHS"
        try:
            receipt, manifest = read(receipt_path), read(manifest_path)
            if receipt.get("status") != "COMPLETE_VALID" or receipt.get("run_id") != run_id:
                return False, "NONCOMPLETE_OR_WRONG_BASE_RECEIPT"
            if manifest.get("run_id") != run_id or manifest.get("split") != "V_select":
                return False, "WRONG_V_MANIFEST_IDENTITY"
            return True, "COMPLETE_FIT_AND_V_METADATA_PRESENT_NOT_YET_INDEPENDENTLY_VALIDATED"
        except (OSError, ValueError, KeyError) as error:
            return False, "MISSING_OR_INVALID_METADATA_" + type(error).__name__

    def initial_state(self):
        path = Path(self.config["existing_result"])
        try:
            result = read(path)
        except (OSError, ValueError):
            states = [dict(**binding, state=identity_state(binding)) for binding in self.config["existing_supervisor_bindings"]]
            if any(row["state"] == "UNKNOWN" for row in states):
                return "UNSAFE", dict(reason="ORIGINAL_SUPERVISOR_IDENTITY_UNKNOWN_WITHOUT_READABLE_RESULT", identities=states)
            if all(row["state"] == "DEAD" for row in states):
                return "UNSAFE", dict(reason="ORIGINAL_SUPERVISORS_ENDED_WITHOUT_READABLE_RESULT", identities=states)
            return "WAIT", dict(reason="original_RESULT_unavailable_supervisor_live", identities=states)
        status, detail = closure_state(result, self.config["pins"]["fit_spec"]["sha256"], GUARD_SHA,
            FAMILIES, identity_state, self.config["existing_supervisor_bindings"])
        return status, dict(detail, result_file=str(path), result_sha256=sha(path), original_status=result.get("status"))

    def command(self, phase, run_id, mode, output):
        plan_sha = PLAN_SHAS[0 if phase == "lag" else 1]
        key = "cost_prepare" if mode == "prepare" else ("pca_runner" if mode == "fit" else phase+"_verifier")
        script = self.config["pins"][key]["file"]
        argv = [self.config["python"], "-X", "utf8", "-u", "-B", script,
            "--public-root", str(self.public), "--cache-root", str(self.cache), "--asset-root", str(self.asset), "--plan-sha", plan_sha]
        if mode != "prepare":
            argv.extend(["--mode", mode, "--run-id", run_id, "--deadline", str(DEADLINE)])
        argv.extend(["--output", str(output), "--execute"])
        validate_argv(argv, self.config["python"], script)
        return argv

    def perform(self, phase, run_id, mode, payload=None):
        self.check()
        used_before = self.storage()
        self.counter += 1
        job_id = "j%02d_%s_%s" % (self.counter, phase, mode)
        name = PCA_NAME if phase == "pca" else LAG_NAME
        private, public = self.cache / name, self.public / name
        if not private.is_dir():
            private.mkdir(parents=True, exist_ok=False)
        rss = 3 * 1024**3 if phase == "pca" else 4 * 1024**3
        token = self.config["queue_id"] + "_" + job_id
        application = public / "diagnostics" / token
        if payload is not None:
            application = relative(public, payload["output_public_relative_path"])
            argv = payload["expanded_arguments"]
            validate_argv(argv, self.config["python"], str(self.public / "measure_costs.py"))
            if application.with_suffix(".trials.csv.gz").exists():
                raise FileExistsError("Existing cost trials must be preserved")
        else:
            argv = self.command(phase, run_id, mode, application)
        if application.exists():
            raise FileExistsError("New output path required")
        spec = dict(schema="CPU_GROUP_SUPERVISOR_v1", asset_root=str(self.asset), cache_root=str(private),
            output_root=str(private / ("supervisor_" + token)), deadline_unix=DEADLINE,
            processes=[dict(id=job_id, command=argv, cwd=self.config["cwd"], max_seconds=max(1, DEADLINE-time.time()-20))],
            max_each_RSS=rss, max_group_RSS=rss, poll_seconds=2, terminate_grace_seconds=10, kill_grace_seconds=5)
        spec_path = self.output / (job_id + "_SPEC.json")
        write_new(spec_path, spec)
        spec_sha = sha(spec_path)
        self.event(dict(event="JOB_ADMITTED", id=job_id, phase=phase, run_id=run_id, mode=mode,
            argv=argv, spec_file=str(spec_path), spec_sha256=spec_sha, source_pins=self.config["pins"], storage_before_bytes=used_before))
        result = self.guard.supervise(spec, spec_file_sha256=spec_sha)
        closure, evidence = closure_state(result, spec_sha, GUARD_SHA, [job_id], identity_state)
        receipt = application if payload is not None else application / ("COST_EXECUTION_PLAN.JSON" if mode == "prepare" else "RESULT.json")
        application_status, receipt_sha = None, None
        if receipt.is_file():
            try:
                application_status, receipt_sha = read(receipt).get("status"), sha(receipt)
            except (OSError, ValueError):
                application_status = "UNREADABLE_PRESERVED"
        required_status = ("COMPLETE_PROTOCOL" if payload is not None else
            "PREPARED_NOT_EXECUTED_CPU_COST_WORKERS" if mode == "prepare" else
            "COMPLETE_FRESH_CPU_FIXED_PCA_AND_FULL_V" if mode == "fit" else "PASS")
        row = dict(id=job_id, closed=closure == "READY", success=result.get("status") == "COMPLETE" and application_status == required_status,
            supervisor_status=result.get("status"), application_status=application_status, argv=argv,
            receipt_file=str(receipt), receipt_sha256=receipt_sha, closure=evidence,
            supervisor_result_file=str(Path(spec["output_root"]) / "RESULT.json"),
            supervisor_result_sha256=sha(Path(spec["output_root"]) / "RESULT.json"))
        self.jobs.append(row)
        self.event(dict(event="GUARD_RETURNED", **row))
        if row["closed"]:
            self.check()
            row["storage_after_bytes"] = self.storage()
        if mode == "prepare":
            self.preparation_output = receipt
        return row

    def prepare(self):
        self.check()
        try:
            state = self.preparer.metadata(self.public, self.cache, self.asset, PLAN_SHAS[0])
        except Exception as error:
            self.event(dict(event="COST_PREPARATION_METADATA_REJECTED", error_class=type(error).__name__, message=str(error)))
            return [], "original_all4_metadata_rejected_" + type(error).__name__
        if not state["ready"]:
            return [], dict(reason="original_all4_fit_V_alias_gate_not_ready", missing=state["missing"], alias_ready=state["alias_ready"])
        row = self.perform("lag", "all4", "prepare")
        if not row["closed"]:
            raise UnsafeClosure("Cost preparation closure unconfirmed")
        if not row["success"]:
            return [], "original_preparation_failed_closed_and_preserved"
        self.check()
        try:
            jobs = self.validate_cost_plan(read(self.preparation_output))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.check()
            self.event(dict(event="SKIPPED_COST_PLAN_INVALID", file=str(self.preparation_output),
                error_class=type(error).__name__, message=str(error)))
            return [], "SKIPPED_COST_PLAN_INVALID_" + type(error).__name__
        self.check()
        self.event(dict(event="COST_PLAN_BOUND", file=str(self.preparation_output), sha256=sha(self.preparation_output), exact_jobs=8))
        return jobs, None

    def validate_cost_plan(self, plan):
        jobs = plan["jobs"]
        expected = [(family, batch) for batch in (1, 8) for family in FAMILIES]
        if (plan.get("post_T_plan_sha256") != PLAN_SHAS[0] or plan.get("worker_count") != 8
                or len(jobs) != 8 or [(row["family"], row["batch"]) for row in jobs] != expected):
            raise ValueError("Original exact eight-job cost queue differs")
        tokens = {"{PYTHON}": self.config["python"], "{ORIGINAL_PUBLIC_ROOT}": str(self.public),
            "{POST_T_PUBLIC_ROOT}": str(self.public / LAG_NAME), "{POST_T_CACHE_ROOT}": str(self.cache / LAG_NAME), "{ASSET_ROOT}": str(self.asset)}
        for job, (family, batch) in zip(jobs, expected):
            if (job["run_id"] != self.plans[0]["conditions"][FAMILIES.index(family)]["run_id"]
                    or job["accuracy_run_id"] != job["run_id"] or job["job_id"] != family+"_B"+str(batch)
                    or any(job.get(key) != value for key, value in dict(device="cpu", L=512, H=48, C=21, K=6,
                        row_microbatch=128, checkpoint_inputs=False, warmups=10, timed_trials=20, stage_trials=20).items())):
                raise ValueError("Cost job science or fixed protocol differs")
            relative(self.public / LAG_NAME, job["output_public_relative_path"])
            if sha(relative(self.cache / LAG_NAME, job["spec_cache_relative_path"])) != job["spec_sha256"]:
                raise ValueError("Prepared route spec changed")
            expected_argv = self.preparer.worker_arguments(job["spec_cache_relative_path"], job["output_public_relative_path"], batch, self.plans[0]["model_asset_relative_path"])
            if job["arguments"] != expected_argv:
                raise ValueError("Prepared worker argv differs from original API")
            job["expanded_arguments"] = expand_tokens(job["arguments"], tokens)
        return jobs


def execute(config_path, config_sha, output):
    runtime = Runtime(config_path, config_sha, output)
    if not output.resolve().is_relative_to(HERE) or output.resolve() == HERE:
        raise ValueError("Queue output must be a new private handoff child folder")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any(k.startswith("INFORMS_GPU_") for k in os.environ)
            or any(os.environ.get(k) != "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))):
        raise ValueError("Explicit offline CPU-only parent environment required")
    import psutil
    identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time())
    write_new(HERE / ("CLAIM_" + runtime.config["queue_id"] + ".json"), dict(config_sha256=config_sha, queue_identity=identity, claimed_unix=time.time(), output=str(output)))
    new_output(output)
    write_new(output / "STARTED.json", dict(config_sha256=config_sha, helper_sha256=sha(__file__), queue_identity=identity, started_unix=time.time(), new_children_started=0))
    (output / "PROGRESS.jsonl").open("x", encoding="utf-8").close()
    rows, failure, initial, preserved = [], None, None, False
    try:
        initial = wait_for_existing(runtime.initial_state, runtime.check, time.time, time.sleep,
            DEADLINE-20, runtime.config["wait_poll_seconds"], runtime.event)
        runtime.event(dict(event="ORIGINAL_FOUR_AND_SUPERVISORS_CLOSED", **initial))
        rows = sequence([row["run_id"] for row in runtime.plans[0]["conditions"]],
            [row["run_id"] for row in runtime.plans[1]["conditions"]], runtime.ready, runtime.perform, runtime.prepare, runtime.event)
    except BaseException as error:
        failure = type(error).__name__
        with (output / "PRIVATE_ERROR_TRACE.txt").open("x", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
    finally:
        try:
            metadata(config_path, config_sha)
            preserved = True
        except Exception as error:
            runtime.event(dict(event="POSTFLIGHT_PIN_FAILURE", error_class=type(error).__name__))
        result = dict(status="SEQUENCE_TERMINAL" if failure is None and preserved else "STOPPED_PRESERVED",
            all_planned_actions_success=failure is None and preserved and len(runtime.jobs) == 35 and all(row["success"] for row in runtime.jobs),
            failure_class=failure, immutable_inputs_preserved=preserved, rows=rows, actual_guard_jobs=runtime.jobs,
            original_closure=initial, config_sha256=config_sha, helper_sha256=sha(__file__), ended_unix=time.time(),
            new_GPU_owner_or_lease_created=False, old_job_stopped_or_restarted=False, new_T_target_or_scoring=False,
            external_quiet_window_is_ROOT_authorized_not_global_process_absence_proof=True,
            memory_is_sampled_and_unobserved_reparented_children_absence_is_not_proven=True,
            storage_scope="C+R byte-size admission and post-job checks with 256MiB reserve; not continuous disk telemetry")
        write_new(output / "RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-triggered", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        config, plans = metadata(Path(args.config), args.config_sha)
        print(json.dumps(dict(status="METADATA_ONLY_READY_FOR_ROOT_REVIEW", lag_conditions=len(plans[0]["conditions"]),
            PCA_conditions=len(plans[1]["conditions"]), cost_jobs_expected=8, children_started=0,
            numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules,
            psutil_imported="psutil" in sys.modules, helper_sha256=sha(__file__), config_sha256=args.config_sha)))
        return 0
    if not args.root_triggered or not args.output:
        parser.error("Actual execution requires --root-triggered and a new private --output")
    result = execute(Path(args.config), args.config_sha, Path(args.output).resolve())
    print(json.dumps(dict(status=result["status"], actual_guard_jobs=len(result["actual_guard_jobs"]), failure_class=result["failure_class"])))
    return 0 if result["status"] == "SEQUENCE_TERMINAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
