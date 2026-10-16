"""Private CPU-only supervisor. Default invocation inspects no experiment inputs.

Process ownership is restricted to observed descendants of newly spawned children,
bound by exact PID/create_time. A child that reparents before any observation cannot
be discovered safely without an OS job object; no global process-name scan is used.
RSS peaks and CPU totals are sampled observations, not continuous/accounting proofs.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HARD_DEADLINE = 1792222550
EACH_RSS = 4294967296
GROUP_RSS = 8589934592


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def safe_path(value, *, directory=True):
    p = Path(value)
    if not p.is_absolute():
        raise ValueError("Absolute paths are required")
    for part in [p] + list(p.parents):
        if part.is_symlink():
            raise ValueError("Symlink paths are not accepted")
    if directory and not p.is_dir():
        raise ValueError("Required directory is absent")
    return p.resolve()


def bounded_number(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError("Invalid or excessive resource setting: " + name)
    return value


def validate_spec(spec):
    if not isinstance(spec, dict) or spec.get("schema") != "CPU_GROUP_SUPERVISOR_v1":
        raise ValueError("CPU_GROUP_SUPERVISOR_v1 object required")
    allowed = {"schema", "processes", "asset_root", "cache_root", "output_root", "deadline_unix",
               "max_each_RSS", "max_group_RSS", "poll_seconds", "terminate_grace_seconds", "kill_grace_seconds"}
    if set(spec) - allowed:
        raise ValueError("Unknown supervisor configuration key")
    asset, cache = safe_path(spec["asset_root"]), safe_path(spec["cache_root"])
    output = safe_path(spec["output_root"], directory=False)
    if output == cache or not output.is_relative_to(cache) or output.exists():
        raise ValueError("Output must be a new private directory below explicit cache_root")
    deadline = bounded_number(spec.get("deadline_unix"), "deadline_unix", HARD_DEADLINE)
    if deadline <= time.time():
        raise ValueError("Deadline has already elapsed")
    processes = spec.get("processes")
    if not isinstance(processes, list) or not 1 <= len(processes) <= 4:
        raise ValueError("One through four explicit commands required")
    rows, ids = [], set()
    for row in processes:
        if not isinstance(row, dict) or set(row) - {"id", "command", "cwd", "max_seconds"}:
            raise ValueError("Invalid process specification")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or len(identity) > 100 or not all(c.isascii() and (c.isalnum() or c in "_-") for c in identity) or identity in ids:
            raise ValueError("Distinct safe process IDs required")
        ids.add(identity)
        command = row.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v or "\0" in v for v in command):
            raise ValueError("Command must be a nonempty argv list; no shell strings")
        cwd = safe_path(row["cwd"])
        maximum = bounded_number(row.get("max_seconds", deadline - time.time()), "max_seconds", HARD_DEADLINE)
        rows.append(dict(id=identity, command=list(command), cwd=str(cwd), max_seconds=float(maximum)))
    result = dict(schema=spec["schema"], asset_root=str(asset), cache_root=str(cache), output_root=str(output),
                  deadline_unix=float(deadline), processes=rows,
                  max_each_RSS=int(bounded_number(spec.get("max_each_RSS", EACH_RSS), "max_each_RSS", EACH_RSS)),
                  max_group_RSS=int(bounded_number(spec.get("max_group_RSS", GROUP_RSS), "max_group_RSS", GROUP_RSS)),
                  poll_seconds=float(bounded_number(spec.get("poll_seconds", 2), "poll_seconds", 2)),
                  terminate_grace_seconds=10, kill_grace_seconds=5)
    if spec.get("terminate_grace_seconds", 10) != 10 or spec.get("kill_grace_seconds", 5) != 5:
        raise ValueError("Teardown protocol is fixed to terminate10 + kill5 seconds")
    return result


def child_environment(spec, original=None):
    env = dict(os.environ if original is None else original)
    for key in list(env):
        if key.startswith("INFORMS_GPU_") or (key.startswith("CUDA_") and "DEVICE" in key) or key == "NVIDIA_VISIBLE_DEVICES":
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               INFORMS_ASSET_ROOT=spec["asset_root"], INFORMS_CACHE_ROOT=spec["cache_root"])
    return env


class OwnedTree:
    def __init__(self, api, root):
        self.api, self.root = api, dict(root)
        self.rows = {(root["pid"], root["create_time"]): dict(root, peak_RSS_bytes=0, last_CPU_seconds=0.)}
        self.errors, self.actions = [], []

    def identities(self):
        return [dict(pid=p, create_time=c) for p, c in self.rows]

    def error(self, row, operation, error):
        event = dict(**row, operation=operation, failure_class=type(error).__name__)
        if event not in self.errors:
            self.errors.append(event)

    def exact(self, row):
        try:
            p = self.api.Process(row["pid"])
            if p.create_time() != row["create_time"]:
                return None
            if p.status() == getattr(self.api, "STATUS_ZOMBIE", "zombie"):
                return None
            return p
        except self.api.NoSuchProcess:
            return None
        except self.api.AccessDenied as error:
            self.error(row, "identity", error)
            return None

    def observe(self):
        pending, seen = self.identities(), set()
        while pending:
            row = pending.pop(0)
            key = row["pid"], row["create_time"]
            if key in seen:
                continue
            seen.add(key)
            p = self.exact(row)
            if p is None:
                continue
            try:
                children = p.children(recursive=False)
            except self.api.NoSuchProcess:
                continue
            except self.api.AccessDenied as error:
                self.error(row, "children", error)
                continue
            for child in children:
                try:
                    child_row = dict(pid=child.pid, create_time=child.create_time())
                except self.api.NoSuchProcess:
                    continue
                except self.api.AccessDenied as error:
                    self.error(dict(pid=child.pid, create_time=None), "child_identity", error)
                    continue
                child_key = child_row["pid"], child_row["create_time"]
                if child_key not in self.rows:
                    self.rows[child_key] = dict(child_row, peak_RSS_bytes=0, last_CPU_seconds=0.)
                pending.append(child_row)

    def sample(self):
        rss, live = 0, []
        for key, record in self.rows.items():
            row = dict(pid=key[0], create_time=key[1])
            p = self.exact(row)
            if p is None:
                continue
            live.append(p.pid)
            try:
                memory, cpu = p.memory_info(), p.cpu_times()
                rss += memory.rss
                record["peak_RSS_bytes"] = max(record["peak_RSS_bytes"], memory.rss)
                record["last_CPU_seconds"] = max(record["last_CPU_seconds"], cpu.user + cpu.system)
            except self.api.NoSuchProcess:
                continue
            except self.api.AccessDenied as error:
                self.error(row, "resource_sample", error)
        return dict(RSS_bytes=rss, observed_CPU_seconds=sum(v["last_CPU_seconds"] for v in self.rows.values()), live_pids=live)

    def stop(self, reason):
        stop_trees([self], reason)
        return self.closure()

    def closure(self):
        remaining = [row for row in self.identities() if self.exact(row) is not None]
        return dict(remaining_owned_processes=remaining, observed_tree_join_confirmed=not remaining and not self.errors,
                    process_tree_errors=list(self.errors), signals=list(self.actions))


def stop_trees(trees, reason):
    if not trees:
        return
    api = trees[0].api
    waiting = []
    for tree in trees:
        tree.observe()
        tree.sample()
        for row in reversed(tree.identities()):
            p = tree.exact(row)
            if p is None:
                continue
            try:
                p.terminate()
                tree.actions.append(dict(**row, action="terminate", reason=reason, unix=time.time()))
                waiting.append(p)
            except api.NoSuchProcess:
                continue
            except api.AccessDenied as error:
                tree.error(row, "terminate", error)
                waiting.append(p)
    if waiting:
        try:
            api.wait_procs(waiting, timeout=10)
        except api.Error as error:
            for tree in trees:
                tree.error(tree.root, "terminate_wait", error)
    killing = []
    for tree in trees:
        for row in reversed(tree.identities()):
            p = tree.exact(row)
            if p is None:
                continue
            try:
                p.kill()
                tree.actions.append(dict(**row, action="kill", reason=reason, unix=time.time()))
                killing.append(p)
            except api.NoSuchProcess:
                continue
            except api.AccessDenied as error:
                tree.error(row, "kill", error)
    if killing:
        try:
            api.wait_procs(killing, timeout=5)
        except api.Error as error:
            for tree in trees:
                tree.error(tree.root, "kill_wait", error)


def supervise(spec, *, spec_file_sha256=None):
    spec = validate_spec(spec)
    import psutil
    started, monotonic_started = time.time(), time.monotonic()
    helper_sha = sha_bytes(Path(__file__).read_bytes())
    output = Path(spec["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    env = child_environment(spec)
    rows, group_peak, failure = [], 0, None
    progress_path = output / "PROGRESS.jsonl"
    with contextlib.ExitStack() as stack:
        progress = stack.enter_context(progress_path.open("x", encoding="utf-8"))
        def event(value):
            progress.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
        try:
            for item in spec["processes"]:
                if time.time() >= spec["deadline_unix"]:
                    raise TimeoutError("Group deadline reached before launch")
                log = stack.enter_context((output / (item["id"] + ".log")).open("xb"))
                child = subprocess.Popen(item["command"], cwd=item["cwd"], env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                launched = time.time()
                try:
                    p = psutil.Process(child.pid)
                    root = dict(pid=child.pid, create_time=p.create_time())
                except psutil.NoSuchProcess:
                    child.wait(timeout=5)
                    raise RuntimeError("Child ended before ownership identity could be captured")
                row = dict(id=item["id"], child=child, tree=OwnedTree(psutil, root), start_unix=launched,
                           worker_deadline_unix=min(spec["deadline_unix"], launched + item["max_seconds"]),
                           peak_RSS_bytes=0, stop_reason=None, end_unix=None)
                rows.append(row)
                row["tree"].observe()
                event(dict(event="STARTED", id=item["id"], unix=launched, root=root))
            while any(row["end_unix"] is None for row in rows):
                now, group_rss, due = time.time(), 0, []
                for row in rows:
                    if row["end_unix"] is not None:
                        continue
                    tree = row["tree"]
                    tree.observe()
                    sample = tree.sample()
                    row["peak_RSS_bytes"] = max(row["peak_RSS_bytes"], sample["RSS_bytes"])
                    group_rss += sample["RSS_bytes"]
                    code = row["child"].poll()
                    event(dict(event="SAMPLE", id=row["id"], unix=now, returncode=code, **sample))
                    if code is not None and not sample["live_pids"]:
                        row["end_unix"] = now
                        if code != 0:
                            row["stop_reason"] = "WORKER_NONZERO_EXIT"
                        continue
                    if tree.errors:
                        row["stop_reason"] = "OWNERSHIP_OR_MONITOR_ERROR"
                    elif code not in (None, 0):
                        row["stop_reason"] = "WORKER_NONZERO_EXIT"
                    elif sample["RSS_bytes"] > spec["max_each_RSS"]:
                        row["stop_reason"] = "WORKER_RSS_LIMIT"
                    elif now >= row["worker_deadline_unix"]:
                        row["stop_reason"] = "WORKER_WALL_LIMIT"
                    if row["stop_reason"]:
                        due.append(row)
                group_peak = max(group_peak, group_rss)
                group_reason = "GROUP_WALL_LIMIT" if now >= spec["deadline_unix"] else "GROUP_RSS_LIMIT" if group_rss > spec["max_group_RSS"] else None
                if group_reason:
                    due = [row for row in rows if row["end_unix"] is None]
                    for row in due:
                        row["stop_reason"] = group_reason
                if due:
                    stop_trees([row["tree"] for row in due], group_reason or "WORKER_LIMIT")
                    for row in due:
                        row["child"].poll()
                        row["end_unix"] = time.time()
                        event(dict(event="STOPPED", id=row["id"], unix=row["end_unix"], reason=row["stop_reason"], **row["tree"].closure()))
                if any(row["end_unix"] is None for row in rows):
                    remaining = min([spec["deadline_unix"]] + [row["worker_deadline_unix"] for row in rows if row["end_unix"] is None]) - time.time()
                    time.sleep(max(0, min(spec["poll_seconds"], remaining)))
        except BaseException as error:
            failure = type(error).__name__
            with (output / "PRIVATE_ERROR_TRACE.txt").open("x", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            stop_trees([row["tree"] for row in rows], "SUPERVISOR_EXCEPTION")
            for row in rows:
                if row["end_unix"] is None:
                    row["end_unix"], row["stop_reason"] = time.time(), "SUPERVISOR_EXCEPTION"
        finished = time.time()
        results = []
        for row in rows:
            closure = row["tree"].closure()
            results.append(dict(id=row["id"], root=row["tree"].root, start_unix=row["start_unix"], end_unix=row["end_unix"],
                worker_deadline_unix=row["worker_deadline_unix"], returncode=row["child"].poll(), stop_reason=row["stop_reason"],
                sampled_peak_RSS_bytes=row["peak_RSS_bytes"], observed_CPU_seconds=row["tree"].sample()["observed_CPU_seconds"],
                owned_processes=list(row["tree"].rows.values()), **closure))
        remaining = [dict(worker=row["id"], **p) for row in results for p in row["remaining_owned_processes"]]
        passed = failure is None and len(results) == len(spec["processes"]) and all(row["returncode"] == 0 and row["stop_reason"] is None and row["observed_tree_join_confirmed"] for row in results)
        result = dict(schema="CPU_GROUP_SUPERVISOR_RESULT_v1", status="COMPLETE" if passed else "STOPPED_RESOURCE_OR_WORKER_FAILURE",
            spec_sha256=sha_bytes(json_bytes(spec)), spec_file_sha256=spec_file_sha256, helper_sha256=helper_sha,
            helper_unchanged=sha_bytes(Path(__file__).read_bytes()) == helper_sha, effective_spec=spec,
            start_unix=started, end_unix=finished, elapsed_seconds=time.monotonic() - monotonic_started,
            sampled_group_peak_RSS_bytes=group_peak, observed_group_CPU_seconds=sum(row["observed_CPU_seconds"] for row in results),
            processes=results, remaining_owned_processes=remaining, supervisor_failure_class=failure,
            CUDA_VISIBLE_DEVICES="-1", GPU_owner_or_token_created=False,
            process_tree_scope="Only observed exact PID/create_time descendants of newly spawned children; no global/name scan",
            unobserved_reparented_descendants_absence_proven=False, RSS_and_CPU_are_sampled_observations=True,
            resource_poll_seconds=spec["poll_seconds"], group_teardown_protocol="terminate_all10_then_kill_alive5_bounded")
        if not result["helper_unchanged"]:
            result["status"] = "STOPPED_HELPER_CHANGED"
        with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--spec")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY", schema="CPU_GROUP_SUPERVISOR_v1", child_processes_started=0,
            maximum_concurrent_children=4, hard_deadline_unix=HARD_DEADLINE, max_each_RSS=EACH_RSS,
            max_group_RSS=GROUP_RSS, poll_seconds=2, psutil_imported="psutil" in sys.modules,
            execution_requires="--execute --spec NEWJSON", helper_sha256=sha_bytes(Path(__file__).read_bytes()))))
        return 0
    if not args.spec:
        parser.error("--execute requires an explicit --spec JSON")
    raw = Path(args.spec).read_bytes()
    result = supervise(json.loads(raw.decode("utf-8-sig")), spec_file_sha256=sha_bytes(raw))
    print(json.dumps(dict(status=result["status"], output_root=result["effective_spec"]["output_root"])))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
