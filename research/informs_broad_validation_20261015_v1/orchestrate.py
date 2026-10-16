"""Sequential minimum-coverage integration; failed routes do not gate others."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resources import check_storage_budget

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("INFORMS_CACHE_ROOT", HERE.parents[1] / ".cache" / HERE.name))


def integration_commands():
    commands = []
    def phase(name, ds, epochs=None):
        command = ["postfit.py", "--phase", name, "--dataset", ds]
        if epochs:
            command += ["--epochs"] + list(map(str, epochs))
        commands.append(command)
    phase("predict", "beijing")
    phase("combine", "beijing")
    phase("ordered", "beijing", [.25])
    phase("controls", "beijing", [.25])
    commands.append(["campaign.py", "--dataset", "electricity", "--minimum-only", "--epochs", ".25",
                     "--family", "DIRECT_G", "--family", "B0", "--family", "B2", "--family", "B4", "--family", "PCA_FIXED_BOLT"])
    phase("predict", "electricity")
    phase("ordered", "electricity", [.25])
    phase("combine", "electricity")
    phase("predict", "pems08")
    phase("combine", "pems08")
    phase("controls", "pems08", [.25])
    commands.append(["campaign.py", "--dataset", "electricity", "--minimum-only", "--epochs", ".25"])
    phase("controls", "electricity", [.25])
    for ds in ("jena", "robin"):
        commands.append(["campaign.py", "--dataset", ds, "--minimum-only", "--epochs", ".25"])
        phase("predict", ds)
        if ds != "robin":
            phase("combine", ds)
            phase("controls", ds, [.25])
    return commands


def used_gpu_seconds():
    path = CACHE / "GPU_USAGE.jsonl"
    return sum(json.loads(line)["charged_process_seconds"] for line in path.read_text().splitlines() if line.strip()) if path.is_file() else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["integration"], required=True)
    p.add_argument("--from-command", type=int, default=0)
    a = p.parse_args()
    commands = integration_commands()
    stamp = str(time.time_ns())
    private = CACHE / "orchestration" / stamp
    public = HERE / "execution_batches" / ("integration_" + stamp)
    private.mkdir(parents=True, exist_ok=False); public.mkdir(parents=True, exist_ok=False)
    plan = dict(stage=a.stage, commands=[["python", "-X", "utf8", "-u"] + c for c in commands],
                starting_command=a.from_command, registered_unix=time.time(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                no_numeric_result_gate=True, actual_completion=False)
    with (public / "COMMANDS.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    for i, command in enumerate(commands):
        if i < a.from_command:
            continue
        if used_gpu_seconds() >= 43200 - 3600 or time.time() >= 1792050350 + 172800:
            with (public / "DEFERRED_RESOURCE.json").open("x", encoding="utf-8") as handle:
                json.dump(dict(status="DEFERRED_RESOURCE", next_command=i, remaining=plan["commands"][i:],
                               GPU_used_seconds=used_gpu_seconds(), actual_completion=False), handle, indent=2)
            break
        check_storage_budget(CACHE, pending_bytes=512 * 1024 ** 2)
        started = time.time()
        print(json.dumps(dict(event="COMMAND_START", index=i, command=plan["commands"][i])), flush=True)
        actual = [sys.executable, "-X", "utf8", "-u", str(HERE / command[0])] + command[1:]
        log = private / f"{i:03d}.log"
        with log.open("x", encoding="utf-8") as stream:
            proc = subprocess.Popen(actual, cwd=HERE, env=dict(os.environ), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                stream.write(line); stream.flush()
                if line.startswith("{"):
                    print(line, end="", flush=True)
            code = proc.wait()
        receipt = dict(command_index=i, command=plan["commands"][i], returncode=code,
                       status="PROCESS_EXIT_ZERO_REQUIRES_RECEIPT_REVIEW" if code == 0 else "PROCESS_FAILED_OTHER_COMMANDS_CONTINUE",
                       started_unix=started, completed_unix=time.time(), private_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                       stopped_other_jobs=False, T_final_accessed_by_this_stage=False)
        with (public / f"COMMAND_{i:03d}.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
        print(json.dumps(dict(event="COMMAND_DONE", index=i, returncode=code)), flush=True)


if __name__ == "__main__":
    main()
