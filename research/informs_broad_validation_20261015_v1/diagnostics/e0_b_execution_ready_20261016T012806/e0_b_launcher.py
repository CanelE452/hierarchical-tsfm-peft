"""Pinned historical-B replay preparation. Default invocation is stdlib plan only.

ROOT may later call run_with_owner with its existing central audit lease. This
module never creates, replaces, recovers, or closes a GPU owner itself.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parents[1]
PINS_SHA256 = "0a4bdd3870fc938b79e0ab5895620d273a921bd0f88e14d646c9668825d33f18"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def metadata_plan():
    if sha(HERE / "INPUT_PINS.json") != PINS_SHA256:
        raise ValueError("Pinned preparation metadata differs")
    pins = read(HERE / "INPUT_PINS.json")
    for name, expected in pins["source30"].items():
        if sha(PUBLIC / name) != expected:
            raise ValueError("Frozen source differs: " + name)
    for pathkey, hashkey in (("inventory_public_relative_path", "inventory_sha256"),
                             ("checkpoint_inventory_public_relative_path", "checkpoint_inventory_sha256"),
                             ("inventory_verification_relative_path", "inventory_verification_sha256"),
                             ("patch_receipt_relative_path", "patch_receipt_sha256")):
        if sha(PUBLIC / pins[pathkey]) != pins[hashkey]:
            raise ValueError("Pinned inventory evidence differs")
    if sha(HERE / "PINNED_INVENTORY.json") != pins["pinned_inventory_sha256"]:
        raise ValueError("Combined exact inventory differs")
    return pins


def _asset_bindings(pins, asset, cache):
    for entry in pins["asset_metadata"]:
        if sha(asset / entry["relative_path"]) != entry["sha256"]:
            raise ValueError("Legacy source/receipt/statistics metadata differs")
    for entry in pins["public_bindings"]:
        if sha(PUBLIC / entry["relative_path"]) != entry["sha256"]:
            raise ValueError("Continuation endpoint receipt differs")
    for entry in pins["checkpoints"]:
        root = asset if entry["scope"] == "asset" else cache
        path = root / entry["checkpoint_relative_path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["file_sha256"]:
            raise ValueError("Pinned checkpoint bytes differ")
    if not (asset / pins["foundation_snapshot_relative_path"]).is_dir():
        raise FileNotFoundError("Pinned local foundation snapshot is absent")


def run_with_owner(lease, asset_root, cache_root, *, max_seconds, gpu_devices="0"):
    """Future explicit ROOT action; borrow one existing owner and spawn a child."""
    if (lease is None or getattr(lease, "purpose", None) != "audit"
            or not getattr(lease, "_entered", False)
            or getattr(getattr(lease, "lock", None), "file", None) is None
            or getattr(lease, "pid", None) != os.getpid()):
        raise ValueError("Existing active ROOT audit owner is required; no nested/new owner")
    lease.check()
    pins = metadata_plan()
    if (lease.total_gpu_seconds > pins["total_GPU_process_wall_ceiling_seconds"]
            or lease.wall_deadline > pins["hard_wall_deadline_unix"]):
        raise ValueError("Existing owner exceeds the registered campaign ceiling")
    if not 0 < float(max_seconds) <= lease.remaining or str(gpu_devices) == "-1":
        raise ValueError("Explicit positive bounded child wall and enabled owner GPU required")
    asset, cache = Path(asset_root).resolve(), Path(cache_root).resolve()
    if cache != asset / ".cache" / PUBLIC.name:
        raise ValueError("Cache must match the unchanged legacy guard allowance")
    _asset_bindings(pins, asset, cache)
    sys.path.insert(0, str(PUBLIC))
    from resources import check_storage_budget
    public_bytes = sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file() and not p.is_symlink())
    pending = pins["pending_storage_allowance_bytes"]
    check_storage_budget(cache, max_bytes=pins["combined_cache_and_public_limit_bytes"] - public_bytes,
                         pending_bytes=pending)
    if shutil.disk_usage(cache).free < pending or shutil.disk_usage(PUBLIC).free < pending:
        raise ValueError("Pending storage allowance is unavailable")
    stamp = str(time.time_ns())
    folder = cache / "final_execution" / ("e0_B_pinned_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    output, log = folder / "E0_B_RESULT.json", folder / "E0_B.log"
    deadline = min(lease.deadline, time.time() + float(max_seconds))
    registry_before = sha(PUBLIC / "EXPERIMENT_REGISTRY.csv")
    env = dict(os.environ, **lease.child_environment(), INFORMS_ASSET_ROOT=str(asset),
               INFORMS_CACHE_ROOT=str(cache), CUDA_VISIBLE_DEVICES=str(gpu_devices),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [sys.executable, "-X", "utf8", "-u", "-B", str(Path(__file__).resolve()),
           "--authenticated-child-execute", "--deadline", str(deadline), "--result-json", str(output)]
    started = time.time()
    with log.open("xb") as stream:
        code = subprocess.Popen(cmd, cwd=PUBLIC, env=env, stdout=stream, stderr=subprocess.STDOUT).wait()
    metadata_plan()
    raw = read(output) if output.is_file() else {}
    registry_unchanged = registry_before == sha(PUBLIC / "EXPERIMENT_REGISTRY.csv")
    complete = (code == 0 and raw.get("status") == "EVALUATION_COMPLETE"
                and raw.get("available_checkpoint_count") == pins["checkpoint_count"]
                and raw.get("checkpoints_rescored") == pins["checkpoint_count"]
                and raw.get("checkpoints_matching_recorded_curve") == pins["checkpoint_count"]
                and registry_unchanged)
    receipt = dict(status="E0_B_EXACT55_COMPLETE" if complete else "E0_B_INCOMPLETE_REVIEW_PRIVATE_LOG",
                   split=pins["legacy_split"], pins_sha256=PINS_SHA256,
                   exact_legacy_ids=pins["legacy_ids"], expected_checkpoints=pins["checkpoint_count"],
                   actual_process_exit_code=code, actual_result_status=raw.get("status"),
                   available_checkpoint_count=raw.get("available_checkpoint_count"),
                   checkpoints_replayed=raw.get("checkpoints_replayed"),
                   checkpoints_rescored=raw.get("checkpoints_rescored"),
                   checkpoints_matching_recorded_curve=raw.get("checkpoints_matching_recorded_curve"),
                   actual_F_calls=raw.get("actual_F_calls"), actual_F_streams=raw.get("actual_F_streams"),
                   registry_relative_path=raw.get("registry_relative_path"),
                   registry_bytes_unchanged=registry_unchanged, source30_unchanged=True,
                   private_log_cache_relative_path=log.relative_to(cache).as_posix(),
                   private_log_sha256=sha(log), private_result_sha256=sha(output) if output.is_file() else None,
                   child_wall_seconds=time.time() - started, batch_size=8, row_microbatch=128,
                   historically_unseen=False, independent_training_rerun=False,
                   T_final_accessed=False, protected_TEST_accessed=False, C_scored=False,
                   campaign_completion_claimed=False)
    write_new(PUBLIC / "execution_batches" / ("E0_B_PINNED_" + stamp + ".json"), receipt)
    return receipt


def _execute_child(args):
    pins = metadata_plan()
    asset, cache = Path(os.environ["INFORMS_ASSET_ROOT"]).resolve(), Path(os.environ["INFORMS_CACHE_ROOT"]).resolve()
    output = Path(args.result_json).resolve()
    if not output.is_relative_to(cache / "final_execution") or output.exists():
        raise ValueError("Fresh private result under final_execution is required")
    sys.path.insert(0, str(PUBLIC))
    from resources import validate_gpu_child
    lease = validate_gpu_child(cache)
    lease.check()
    _asset_bindings(pins, asset, cache)
    import legacy_evaluation
    expected = read(HERE / "PINNED_INVENTORY.json")
    actual = legacy_evaluation.inventory_checkpoints(asset, cache, PUBLIC, pins["legacy_ids"])
    if actual != expected:
        raise ValueError("Fresh inventory differs from the frozen exact9/55 scope")
    raw = legacy_evaluation.evaluate_legacy(asset, cache, PUBLIC, pins["legacy_ids"], lease=lease,
                                           deadline=args.deadline, batch_size=8, row_microbatch=128)
    write_new(output, raw)
    metadata_plan()
    if raw["inventory"] != expected:
        raise ValueError("Execution inventory scope changed")
    if (raw["status"] != "EVALUATION_COMPLETE" or raw["available_checkpoint_count"] != 55
            or raw["checkpoints_rescored"] != 55 or raw["checkpoints_matching_recorded_curve"] != 55):
        raise RuntimeError("Historical-B evaluation is partial or a recorded curve differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authenticated-child-execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.authenticated_child_execute:
        if args.deadline is None or args.result_json is None:
            parser.error("Authenticated child requires explicit deadline and private result")
        _execute_child(args)
    else:
        pins = metadata_plan()
        print(json.dumps(dict(status="PINNED_METADATA_PLAN_ONLY", legacy_ids=pins["legacy_ids"],
                              checkpoint_count=pins["checkpoint_count"], sources_pinned=len(pins["source30"]),
                              split=pins["legacy_split"], GPU_execution_started=False,
                              dataset_or_checkpoint_arrays_loaded=False,
                              numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules)))


if __name__ == "__main__":
    main()
