"""Shared Bi runtime adapter for small-K follow-up work.

Importing this module does not import the Bi runtime, torch, old engines, or
scientific arrays.  ``FollowupLimit`` imports and instantiates the frozen Bi
``runtime.Limit`` only when an actual caller creates the adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import importlib.util
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FOLLOWUP_CACHE = ROOT / ".cache" / HERE.name
BI_RUNTIME_PATH = ROOT / "research" / "bihash_future_prediction_20261013" / "runtime.py"
BI_RUNTIME_SHA256 = "c8852f0811ce76f6ea70b3f97e014b3794e2250a94bf6f0cc02d17977b0a7d1a"
INDEPENDENT_AUDIT_CPU_BODY_SECONDS = 407.8757245
REPORT_BODY_CPU_RECORDS = (
    {
        "label": "canonical_final_v1_report_body",
        "manifest_path": ROOT / "research" / "bihash_future_prediction_20261013" / "report_canonical_final_v1" / "MANIFEST.json",
        "manifest_sha256": "7eca401fde5cfa3c91e5639fee97076bd7577f93582820ace4aef70828a2f6f8",
        "seconds": 49.6082627,
    },
    {
        "label": "canonical_final_v2_report_body",
        "manifest_path": ROOT / "research" / "bihash_future_prediction_20261013" / "report_canonical_final_v2" / "MANIFEST.json",
        "manifest_sha256": "2d97930bc83be56932b517c72afe49666a61bc165a240f5f4d54a6b8630a8595",
        "seconds": 51.25142159965,
    },
)
REPORT_BODY_CPU_SECONDS = sum(float(record["seconds"]) for record in REPORT_BODY_CPU_RECORDS)
DEFAULT_SUPPLEMENTARY_CPU_SECONDS = INDEPENDENT_AUDIT_CPU_BODY_SECONDS + REPORT_BODY_CPU_SECONDS
UNKNOWN_ADHOC_STARTUP_OVERHEAD = "unknown_not_assumed_zero"
LAST_CONSTRUCTOR_TERMINAL: dict[str, Any] | None = None


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bi_runtime() -> Any:
    actual = sha256_path(BI_RUNTIME_PATH)
    if actual != BI_RUNTIME_SHA256:
        raise RuntimeError(f"BI_RUNTIME_SHA_MISMATCH:{actual}!={BI_RUNTIME_SHA256}")
    spec = importlib.util.spec_from_file_location("bi_runtime_for_smallk_followup", BI_RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BI_RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_report_body_records() -> list[dict[str, Any]]:
    records = []
    for record in REPORT_BODY_CPU_RECORDS:
        path = Path(record["manifest_path"])
        actual = sha256_path(path)
        expected = str(record["manifest_sha256"])
        if actual != expected:
            raise RuntimeError(f"REPORT_MANIFEST_SHA_MISMATCH:{path}:{actual}!={expected}")
        records.append(
            {
                "label": record["label"],
                "manifest_path": str(path.relative_to(ROOT)),
                "manifest_sha256": actual,
                "seconds": float(record["seconds"]),
            }
        )
    return records


def verified_report_body_seconds() -> float:
    return sum(record["seconds"] for record in verified_report_body_records())


def _receipt_status_terminal(status: str | None) -> bool:
    return status in {"COMPLETE", "FAILED_PRESERVED", "BUDGET_HELD_PRESERVED"}


def _previous_cpu_seconds(rt: Any) -> float:
    total = 0.0
    resource_dir = rt.HERE / "resources"
    if not resource_dir.exists():
        return 0.0
    for path in resource_dir.glob("*.json"):
        receipt = rt.read(path)
        if not _receipt_status_terminal(receipt.get("status")):
            raise RuntimeError("NONTERMINAL_RESOURCE_FILE: " + str(path))
        if receipt.get("gpu") is False:
            total += float(receipt.get("wall_seconds", 0.0))
    return total


def _storage_bytes(roots: Iterable[Path]) -> int:
    seen: set[Path] = set()
    total = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            total += resolved.stat().st_size
    return total


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class FollowupLimit:
    """Adapter over the frozen Bi ``runtime.Limit`` with follow-up accounting.

    API for callers: ``FollowupLimit(tag, gpu)``, then cooperative ``.check()``
    and terminal ``.close(status='COMPLETE', error=None)``.  Extra keyword
    arguments only narrow accounting and are optional for runners.
    """

    def __init__(
        self,
        tag: str,
        gpu: bool,
        *,
        followup_roots: Iterable[str | Path] = (),
        supplementary_cpu_seconds: float = INDEPENDENT_AUDIT_CPU_BODY_SECONDS,
        include_report_body_seconds: bool = True,
    ) -> None:
        global LAST_CONSTRUCTOR_TERMINAL
        LAST_CONSTRUCTOR_TERMINAL = None
        self.constructor_failure_scope = None
        try:
            self.rt = load_bi_runtime()
            self.tag = tag
            self.gpu = bool(gpu)
            self.audit_supplementary_cpu_seconds = float(supplementary_cpu_seconds)
            self.report_body_cpu_seconds = float(verified_report_body_seconds()) if include_report_body_seconds else 0.0
            self.supplementary_cpu_seconds = self.audit_supplementary_cpu_seconds + self.report_body_cpu_seconds
            self.followup_roots = tuple(Path(p) for p in followup_roots) or (HERE, FOLLOWUP_CACHE)
        except Exception as error:
            self.constructor_failure_scope = {
                "scope": "adapter pre-shared construction failed before shared receipt assignment; caller cannot close a shared receipt",
                "type": type(error).__name__,
                "message": str(error),
            }
            LAST_CONSTRUCTOR_TERMINAL = {
                "terminal_status": "FAILED_PRESERVED",
                "constructor_failure_scope": self.constructor_failure_scope,
            }
            raise
        self.shared = None
        try:
            self.shared = self.rt.Limit(tag, gpu=self.gpu)
        except Exception as error:
            self.constructor_failure_scope = {
                "scope": "rt.Limit construction failed before shared receipt assignment; caller cannot close a shared receipt",
                "type": type(error).__name__,
                "message": str(error),
            }
            LAST_CONSTRUCTOR_TERMINAL = {
                "terminal_status": "FAILED_PRESERVED",
                "constructor_failure_scope": self.constructor_failure_scope,
            }
            raise
        self.closed = False
        self.last_check: dict[str, Any] | None = None
        self.terminal: dict[str, Any] | None = None
        try:
            self.check()
        except Exception as error:
            LAST_CONSTRUCTOR_TERMINAL = self._close_after_adapter_failure("constructor_check", error)
            raise

    def _is_budget_error(self, error: Exception) -> bool:
        budget_type = getattr(self.rt, "BudgetHeld", ())
        try:
            return isinstance(error, budget_type)
        except TypeError:
            return False

    def _status_for_adapter_failure(self, error: Exception) -> str:
        return "BUDGET_HELD_PRESERVED" if self._is_budget_error(error) else "FAILED_PRESERVED"

    def _failure_record(self, phase: str, error: Exception) -> dict[str, Any]:
        return {
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
        }

    def _close_after_adapter_failure(self, phase: str, error: Exception) -> dict[str, Any]:
        status = self._status_for_adapter_failure(error)
        failure = self._failure_record(phase, error)
        terminal = self.shared.close(status, failure)
        self.closed = True
        self.terminal = {
            **dict(terminal),
            "terminal_status": status,
            "requested_status": None,
            "terminal_error": failure,
            "check_failure": failure,
            "shared_resource_terminal": dict(terminal),
            "followup_limit": self.last_check,
            "supplementary_cpu_seconds_counted": self.supplementary_cpu_seconds,
            "independent_audit_cpu_seconds_counted": self.audit_supplementary_cpu_seconds,
            "report_body_cpu_seconds_counted": self.report_body_cpu_seconds,
            "unknown_adhoc_startup_overhead": UNKNOWN_ADHOC_STARTUP_OVERHEAD,
        }
        return dict(self.terminal)

    def check(self) -> dict[str, Any]:
        shared = dict(self.shared.check())
        previous_cpu = _previous_cpu_seconds(self.rt)
        cpu_counted = previous_cpu + self.supplementary_cpu_seconds
        if not self.gpu:
            cpu_counted = float(shared["budget_counted_seconds"]) + self.supplementary_cpu_seconds
        if cpu_counted > self.rt.CPU_SECONDS:
            raise self.rt.BudgetHeld(f"FOLLOWUP_CPU_BUDGET_WITH_SUPPLEMENT:{cpu_counted:.6f}>{self.rt.CPU_SECONDS}")
        followup_storage = _storage_bytes(self.followup_roots)
        combined_storage = int(shared.get("storage_bytes", 0)) + followup_storage
        if combined_storage > self.rt.STORAGE_BYTES or shutil.disk_usage(self.rt.ROOT).free < 2 * 1024**3:
            raise self.rt.BudgetHeld("FOLLOWUP_STORAGE_BUDGET_WITH_BI_TOTAL")
        self.last_check = {
            **shared,
            "tag": self.tag,
            "gpu": self.gpu,
            "supplementary_cpu_seconds_counted": self.supplementary_cpu_seconds,
            "independent_audit_cpu_seconds_counted": self.audit_supplementary_cpu_seconds,
            "report_body_cpu_seconds_counted": self.report_body_cpu_seconds,
            "supplementary_cpu_source": (
                "independent audit body 405.3414399994 + own 2.53428449994, "
                "canonical report measured body v1 49.6082627 + v2 51.25142159965; "
                "root verification RT CPU already in shared ledger"
            ),
            "unknown_adhoc_startup_overhead": UNKNOWN_ADHOC_STARTUP_OVERHEAD,
            "cpu_budget_counted_with_supplement": cpu_counted,
            "followup_storage_bytes": followup_storage,
            "combined_bi_and_followup_storage_bytes": combined_storage,
            "storage_budget_bytes": self.rt.STORAGE_BYTES,
            "bi_runtime_path": str(BI_RUNTIME_PATH.relative_to(ROOT)),
            "bi_runtime_sha256": BI_RUNTIME_SHA256,
        }
        return dict(self.last_check)

    def close(self, status: str = "COMPLETE", error: Any | None = None) -> dict[str, Any]:
        if self.closed:
            return dict(self.terminal)
        check = None
        check_failure = None
        terminal_status = status
        terminal_error = error
        try:
            check = self.check()
        except Exception as check_error:
            terminal_status = self._status_for_adapter_failure(check_error)
            check_failure = self._failure_record("terminal_check", check_error)
            terminal_error = {
                "caller_status": status,
                "caller_error": _jsonable(error),
                "check_failure": check_failure,
            }
        terminal = self.shared.close(terminal_status, terminal_error)
        self.terminal = {
            **terminal,
            "terminal_status": terminal_status,
            "requested_status": status,
            "terminal_error": _jsonable(terminal_error),
            "check_failure": check_failure,
            "shared_resource_terminal": terminal,
            "followup_limit": check if check is not None else self.last_check,
            "supplementary_cpu_seconds_counted": self.supplementary_cpu_seconds,
            "independent_audit_cpu_seconds_counted": self.audit_supplementary_cpu_seconds,
            "report_body_cpu_seconds_counted": self.report_body_cpu_seconds,
            "unknown_adhoc_startup_overhead": UNKNOWN_ADHOC_STARTUP_OVERHEAD,
        }
        self.closed = True
        return _jsonable(dict(self.terminal))
