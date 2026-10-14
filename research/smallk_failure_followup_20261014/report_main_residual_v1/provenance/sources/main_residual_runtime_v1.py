"""Main-forecast residual fresh approval runtime.

SOURCE-only until ROOT writes main_residual_authorization_v1.json and activates the
fixed manifest. This runtime preserves the exhausted continuation/stage/matched ledgers,
uses one exclusive lease for CPU and GPU jobs, forbids CUDA/F access for CPU fixture jobs,
and allows only explicitly-scoped frozen-F forward cost jobs on GPU.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUTHORIZATION = HERE / "main_residual_authorization_v1.json"
RUN_MANIFEST = HERE / "main_residual_run_manifest_v1.json"
CORE_RUNTIME = Path(__file__).resolve()
APPROVED_PLAN = HERE / "PLAN_MAIN_FORECAST_RESIDUAL_KO_v1.md"
METHOD_LOCK = HERE / "METHOD_LOCK_MAIN_RESIDUAL_v1.md"
FAST_STORAGE_SCANNER = HERE / "fast_storage_scan_v1.py"
RESOURCES = HERE / "main_residual_resources_v1"
ACTIVE = RESOURCES / "ACTIVE.json"
PHASE = "main_forecast_residual_v1"
EXPECTED_RESPONSE_IDENTIFIER = "call_c2e3607e77684a40bbd8e6a06612d9b9:0"
EXPECTED_USER_RESPONSE = "아니 왜 자꾸 물어봐 하라고 지금까지 한거 main에 push하고 쭉하라고"
MAX_LIMITS = {
    "gpu_process_wall_seconds": 300,
    "cpu_process_wall_seconds": 900,
    "campaign_wall_seconds": 2400,
    "additional_storage_bytes": 1024**3,
    "process_rss_bytes": 8 * 1024**3,
}
TERMINAL = ("COMPLETE", "FAILED_PRESERVED", "BUDGET_HELD_PRESERVED")
PRIOR_ACTIVE_LEASES = (
    HERE / "continuation_resources_v1" / "ACTIVE.json",
    HERE / "stage_separation_resources_v1" / "ACTIVE.json",
    HERE / "matched_readout_resources_v1" / "ACTIVE.json",
)
LAST_CONSTRUCTOR_TERMINAL = None


class BudgetHeld(RuntimeError):
    pass


class AuthorizationRequired(RuntimeError):
    pass


class ForbiddenAccess(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from fast_storage_scan_v1 import storage_bytes  # noqa: E402


def _limits_valid(limits):
    if set(limits) != set(MAX_LIMITS):
        return False
    for key, cap in MAX_LIMITS.items():
        value = limits.get(key)
        if not isinstance(value, (int, float)):
            return False
        if value <= 0 or value > cap:
            return False
    return True


def authorized_records():
    if not AUTHORIZATION.is_file() or not RUN_MANIFEST.is_file():
        raise AuthorizationRequired("MAIN_RESIDUAL_HUMAN_APPROVAL_AND_FIXED_MANIFEST_REQUIRED")
    auth, run = read(AUTHORIZATION), read(RUN_MANIFEST)
    limits = auth.get("limits", {})
    if (
        auth.get("schema") != "smallk_main_residual_authorization_v1"
        or auth.get("status") != "APPROVED"
        or auth.get("origin") != "human_user_response"
        or auth.get("response_identifier") != EXPECTED_RESPONSE_IDENTIFIER
        or auth.get("user_response") != EXPECTED_USER_RESPONSE
        or auth.get("phase") != PHASE
        or auth.get("approved_plan_sha256") != sha(APPROVED_PLAN)
        or auth.get("method_lock_sha256") != sha(METHOD_LOCK)
        or auth.get("runtime_source_sha256") != sha(CORE_RUNTIME)
        or auth.get("gpu_allowed") is not True
        or auth.get("cuda_allowed_for_gpu_cost") is not True
        or auth.get("cpu_cuda_allowed") is not False
        or auth.get("F_access_allowed") is not True
        or auth.get("training_allowed") is not False
        or not _limits_valid(limits)
    ):
        raise AuthorizationRequired("MAIN_RESIDUAL_APPROVAL_RECORD_INVALID_OR_LIMITS_EXCEEDED")
    if (
        run.get("schema") != "smallk_main_residual_run_manifest_v1"
        or run.get("authorization_sha256") != sha(AUTHORIZATION)
        or run.get("runtime_source_sha256") != sha(CORE_RUNTIME)
        or run.get("phase") != PHASE
        or run.get("started_unix", 0) < auth.get("approved_unix", float("inf"))
        or run.get("expires_unix") != run.get("started_unix", 0) + limits["campaign_wall_seconds"]
        or not isinstance(run.get("baseline_storage_bytes"), int)
        or run.get("gpu_allowed") is not True
        or run.get("cuda_allowed_for_gpu_cost") is not True
        or run.get("cpu_cuda_allowed") is not False
        or run.get("F_access_allowed") is not True
        or run.get("training_allowed") is not False
    ):
        raise AuthorizationRequired("MAIN_RESIDUAL_FIXED_MANIFEST_INVALID")
    protected = run.get("protected_prior_receipts")
    roots = run.get("storage_roots")
    if (
        not isinstance(protected, list)
        or not protected
        or any(not isinstance(r, dict) or not r.get("path") or not isinstance(r.get("sha256"), str) or len(r["sha256"]) != 64 for r in protected)
        or not isinstance(roots, list)
        or not roots
    ):
        raise AuthorizationRequired("MAIN_RESIDUAL_PRIOR_RECEIPTS_AND_STORAGE_ROOTS_REQUIRED")
    return auth, run


class _FastShared:
    def __init__(self, limit):
        self.limit = limit

    def check(self):
        return self.limit.base_check()

    def checkfast(self):
        return self.check()


class ContinuationLimit:
    def __init__(self, tag, gpu=False):
        global LAST_CONSTRUCTOR_TERMINAL
        LAST_CONSTRUCTOR_TERMINAL = None
        try:
            self.authorization, self.run_manifest = authorized_records()
        except Exception as error:
            LAST_CONSTRUCTOR_TERMINAL = {
                "terminal_status": "FAILED_PRESERVED",
                "terminal_error": {"type": type(error).__name__, "message": str(error)},
                "scope": "pre-authorization failure; no main-residual resource receipt or scientific assets opened",
            }
            raise
        if not tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in tag):
            raise ValueError("SAFE_MAIN_RESIDUAL_RESOURCE_TAG_REQUIRED")
        import psutil

        self.tag, self.gpu = tag, bool(gpu)
        self.limits = self.authorization["limits"]
        self.authorization_record = {"path": str(AUTHORIZATION), "sha256": sha(AUTHORIZATION)}
        self.run_manifest_record = {"path": str(RUN_MANIFEST), "sha256": sha(RUN_MANIFEST)}
        self.guard_adapter_record = {
            "runtime_source": {"path": str(CORE_RUNTIME), "sha256": sha(CORE_RUNTIME)},
            "approved_plan": {"path": str(APPROVED_PLAN), "sha256": sha(APPROVED_PLAN)},
            "method_lock": {"path": str(METHOD_LOCK), "sha256": sha(METHOD_LOCK)},
            "storage_scanner": {"path": str(FAST_STORAGE_SCANNER), "sha256": sha(FAST_STORAGE_SCANNER)},
            "scope": "fresh main-residual CPU/GPU limits; prior continuation/stage/matched ledgers preserved and not reset",
        }
        self.process = psutil.Process()
        self.started_unix = self.process.create_time()
        self.start = time.perf_counter()
        self.initial_elapsed = max(0.0, time.time() - self.started_unix)
        self.rss = 0
        self.closed = False
        self.forbidden_access = None
        self.training_access = None
        self.gpu_forward_scope = None
        self.storage = self.run_manifest["baseline_storage_bytes"]
        self.last_check = None
        self.path = RESOURCES / f"{os.getpid()}_{time.time_ns()}_{tag}.json"
        self.rt = SimpleNamespace(
            BudgetHeld=BudgetHeld,
            ROOT=ROOT,
            HERE=HERE,
            GPU_SECONDS=self.limits["gpu_process_wall_seconds"],
            CPU_SECONDS=self.limits["cpu_process_wall_seconds"],
        )
        self.shared = _FastShared(self)
        RESOURCES.mkdir(exist_ok=True)
        for prior_active in PRIOR_ACTIVE_LEASES:
            if Path(prior_active).exists():
                raise RuntimeError("PRIOR_PHASE_ACTIVE_LEASE_REQUIRES_TERMINAL_REVIEW:" + str(prior_active))
        self.previous = []
        for path in RESOURCES.glob("*.json"):
            if path == ACTIVE:
                raise RuntimeError("EXISTING_MAIN_RESIDUAL_ACTIVE_LEASE_REQUIRES_REVIEW")
            receipt = read(path)
            if receipt.get("status") not in TERMINAL:
                raise RuntimeError("NONTERMINAL_MAIN_RESIDUAL_RECEIPT:" + str(path))
            if (
                receipt.get("authorization", {}).get("sha256") != self.authorization_record["sha256"]
                or receipt.get("run_manifest", {}).get("sha256") != self.run_manifest_record["sha256"]
            ):
                raise RuntimeError("MAIN_RESIDUAL_RESOURCE_FOLDER_OTHER_APPROVAL_OR_MANIFEST:" + str(path))
            self.previous.append(receipt)
        self.lease = {
            "pid": os.getpid(),
            "process_created_unix": self.started_unix,
            "tag": tag,
            "gpu": self.gpu,
            "manifest_sha256": self.run_manifest_record["sha256"],
        }
        write_new(ACTIVE, self.lease)
        self.lease_hash = sha(ACTIVE)
        try:
            self.check()
        except Exception as error:
            status = "BUDGET_HELD_PRESERVED" if isinstance(error, BudgetHeld) else "FAILED_PRESERVED"
            LAST_CONSTRUCTOR_TERMINAL = self.close(status, {"type": type(error).__name__, "message": str(error)})
            raise

    def _elapsed(self):
        return self.initial_elapsed + time.perf_counter() - self.start

    def flag_forbidden_access(self, kind):
        self.forbidden_access = {"kind": str(kind), "message": "CUDA/F access is forbidden for CPU main-residual jobs"}
        raise ForbiddenAccess("MAIN_RESIDUAL_CPU_CUDA_OR_F_ACCESS_FORBIDDEN:" + str(kind))

    def flag_training_access(self, kind="training"):
        self.training_access = {"kind": str(kind), "message": "Training is forbidden in the main-residual phase"}
        raise ForbiddenAccess("MAIN_RESIDUAL_TRAINING_FORBIDDEN:" + str(kind))

    def declare_gpu_forward_scope(self, scope):
        if not self.gpu:
            raise ForbiddenAccess("MAIN_RESIDUAL_GPU_FORWARD_SCOPE_ONLY_VALID_FOR_GPU_JOBS")
        if scope not in ("frozen_F", "frozen_F_cost"):
            raise ForbiddenAccess("MAIN_RESIDUAL_GPU_FORWARD_SCOPE_INVALID:" + str(scope))
        self.gpu_forward_scope = scope
        return {"gpu_forward_scope": scope, "training_allowed": False}

    def _cuda_initialized(self):
        torch = sys.modules.get("torch")
        return bool(torch is not None and getattr(getattr(torch, "cuda", None), "is_initialized", lambda: False)())

    def _mode_seconds_key(self):
        return "gpu_process_wall_seconds" if self.gpu else "cpu_process_wall_seconds"

    def base_check(self):
        self.rss = max(self.rss, self.process.memory_info().rss)
        if not self.gpu and (self._cuda_initialized() or self.forbidden_access is not None):
            raise ForbiddenAccess("MAIN_RESIDUAL_CPU_CUDA_OR_F_ACCESS_FORBIDDEN")
        if self.training_access is not None:
            raise ForbiddenAccess("MAIN_RESIDUAL_TRAINING_FORBIDDEN")
        elapsed = self._elapsed()
        charged = sum(r["wall_seconds"] for r in self.previous if bool(r.get("gpu")) is self.gpu) + elapsed
        if charged > self.limits[self._mode_seconds_key()]:
            raise BudgetHeld("MAIN_RESIDUAL_" + ("GPU" if self.gpu else "CPU") + "_PROCESS_WALL_BUDGET")
        if time.time() > self.run_manifest["expires_unix"]:
            raise BudgetHeld("MAIN_RESIDUAL_CAMPAIGN_WALL_BUDGET")
        if self.rss > self.limits["process_rss_bytes"]:
            raise BudgetHeld("MAIN_RESIDUAL_RSS_BUDGET")
        return {"elapsed_seconds": elapsed, "budget_counted_seconds": charged, "storage_bytes": self.storage, "tag": self.tag, "gpu": self.gpu}

    def check(self):
        value = self.base_check()
        if sha(AUTHORIZATION) != self.authorization_record["sha256"] or sha(RUN_MANIFEST) != self.run_manifest_record["sha256"]:
            raise RuntimeError("MAIN_RESIDUAL_AUTHORIZATION_OR_MANIFEST_CHANGED")
        for record in self.guard_adapter_record.values():
            if isinstance(record, dict) and sha(record["path"]) != record["sha256"]:
                raise RuntimeError("MAIN_RESIDUAL_GUARD_SOURCE_CHANGED")
        for record in self.run_manifest.get("protected_prior_receipts", []):
            if sha(record["path"]) != record["sha256"]:
                raise RuntimeError("MAIN_RESIDUAL_PROTECTED_PRIOR_RECEIPT_CHANGED")
        self.storage = storage_bytes(self.run_manifest["storage_roots"])
        growth = max(0, self.storage - self.run_manifest["baseline_storage_bytes"])
        if growth > self.limits["additional_storage_bytes"] or shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise BudgetHeld("MAIN_RESIDUAL_STORAGE_GROWTH_OR_FREE_DISK_BUDGET")
        value.update(
            storage_bytes=self.storage,
            additional_storage_bytes=growth,
            authorization=self.authorization_record,
            run_manifest=self.run_manifest_record,
            guard_adapter=self.guard_adapter_record,
            gpu_forward_scope=self.gpu_forward_scope,
            training_allowed=False,
        )
        self.last_check = value
        self.base_check()
        return dict(value)

    def checkfull(self):
        return self.check()

    def close(self, status="COMPLETE", error=None):
        if self.closed:
            return dict(self.terminal)
        requested = status
        if status == "COMPLETE" and error is not None:
            status = "FAILED_PRESERVED"
        check_failure = None
        try:
            self.check()
        except Exception as failure:
            status = "BUDGET_HELD_PRESERVED" if isinstance(failure, BudgetHeld) else "FAILED_PRESERVED"
            check_failure = {"type": type(failure).__name__, "message": str(failure)}
        if status not in TERMINAL:
            raise ValueError("MAIN_RESIDUAL_TERMINAL_STATUS_REQUIRED")
        elapsed = self._elapsed()
        receipt = {
            "tag": self.tag,
            "gpu": self.gpu,
            "pid": os.getpid(),
            "status": status,
            "started_unix": self.started_unix,
            "finished_unix": time.time(),
            "wall_seconds": elapsed,
            "sampled_peak_rss": self.rss,
            "storage_bytes": self.storage,
            "authorization": self.authorization_record,
            "run_manifest": self.run_manifest_record,
            "guard_adapter": self.guard_adapter_record,
            "requested_status": requested,
            "caller_error": error,
            "check_failure": check_failure,
            "previous_receipts": len(self.previous),
            "gpu_forward_scope": self.gpu_forward_scope,
            "training_allowed": False,
            "wall_scope": "process creation to terminal; CPU jobs forbid CUDA/F access, GPU jobs allow frozen-F forward cost only",
            "prior_ledgers_preserved_and_not_reset": True,
        }
        write_new(self.path, receipt)
        if sha(ACTIVE) != self.lease_hash:
            raise RuntimeError("MAIN_RESIDUAL_ACTIVE_LEASE_CHANGED")
        ACTIVE.unlink()
        record = {"path": str(self.path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(self.path), "bytes": self.path.stat().st_size}
        terminal_error = check_failure or error
        self.terminal = {
            **record,
            "terminal_status": status,
            "requested_status": requested,
            "terminal_error": terminal_error,
            "check_failure": check_failure,
            "continuation_limit": self.last_check,
            "authorization": self.authorization_record,
            "run_manifest": self.run_manifest_record,
        }
        self.closed = True
        print(json.dumps({"resource_terminal": record, "status": status, "gpu": self.gpu, "wall_seconds": elapsed}), flush=True)
        return dict(self.terminal)
