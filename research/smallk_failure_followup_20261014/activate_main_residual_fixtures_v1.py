"""Activate the main-residual approval window and run initial CPU fixtures.

SOURCE-only until ROOT writes main_residual_authorization_v1.json and executes this script.
The activator verifies the exact human approval, preserves all prior continuation/stage/
matched receipts, writes one immutable main_residual_run_manifest_v1.json from process
birth, then constructs main_residual_runtime_v1.ContinuationLimit in the same process so
fixture import and pytest collection are charged to the first CPU receipt.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Iterable


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DEFAULT_TAG = "main_residual_initial_fixtures_v1"
DEFAULT_OUT = HERE / "validation_v1" / "MAIN_RESIDUAL_INITIAL_FIXTURES_v1.json"
DEFAULT_TESTS = (
    HERE / "test_main_residual_runtime_v1.py",
    HERE / "test_main_forecast_residual_readout_v1.py",
    HERE / "test_run_main_residual_v1.py",
)
MATCHED_RUN_MANIFEST = HERE / "matched_readout_run_manifest_v1.json"
RESOURCE_DIRS_TO_PROTECT = (
    HERE / "continuation_resources_v1",
    HERE / "stage_separation_resources_v1",
    HERE / "matched_readout_resources_v1",
)
SAFE_TAG_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class ActivationContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_tag(tag: str) -> str:
    if not tag or any(char not in SAFE_TAG_CHARS for char in tag):
        raise ActivationContractError("SAFE_MAIN_RESIDUAL_FIXTURE_TAG_REQUIRED")
    return tag


def _resolve_output(value: str | None) -> Path:
    path = DEFAULT_OUT if value is None else Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ActivationContractError("OUTPUT_MUST_BE_UNDER_REPOSITORY_ROOT") from exc
    return resolved


def _resolve_test(value: str) -> Path:
    raw = Path(value)
    candidates = [raw.resolve()] if raw.is_absolute() else [(HERE / raw).resolve(), (ROOT / raw).resolve()]
    existing = [path for path in candidates if path.is_file()]
    resolved = existing[0] if existing else candidates[0]
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ActivationContractError("TEST_FILE_MUST_BE_UNDER_REPOSITORY_ROOT") from exc
    if resolved.suffix != ".py" or not resolved.name.startswith("test_"):
        raise ActivationContractError("TEST_FILE_MUST_BE_TEST_PYTHON_SOURCE")
    if not resolved.is_file():
        raise FileNotFoundError("MAIN_RESIDUAL_FIXTURE_SOURCE_NOT_READY:" + str(resolved))
    return resolved


def _resolve_tests(values: list[str] | None) -> list[Path]:
    tests = [_resolve_test(value) for value in values] if values else [_resolve_test(str(path)) for path in DEFAULT_TESTS]
    unique: list[Path] = []
    seen = set()
    for path in tests:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ActivationContractError("AT_LEAST_ONE_MAIN_RESIDUAL_FIXTURE_REQUIRED")
    return unique


def _source_record(path: Path) -> dict:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _snapshot(paths: Iterable[Path]) -> list[dict]:
    return [_source_record(path) for path in paths]


def _snapshot_changes(before: list[dict], after: list[dict]) -> list[dict]:
    prior = {record["path"]: record for record in before}
    changed = []
    for record in after:
        old = prior.get(record["path"])
        if old is None or old.get("sha256") != record.get("sha256") or old.get("bytes") != record.get("bytes"):
            changed.append({"before": old, "after": record})
    return changed


def _validate_authorization(runtime) -> dict:
    auth = runtime.read(runtime.AUTHORIZATION)
    if (
        auth.get("schema") != "smallk_main_residual_authorization_v1"
        or auth.get("status") != "APPROVED"
        or auth.get("origin") != "human_user_response"
        or auth.get("response_identifier") != runtime.EXPECTED_RESPONSE_IDENTIFIER
        or auth.get("user_response") != runtime.EXPECTED_USER_RESPONSE
        or auth.get("phase") != runtime.PHASE
        or auth.get("approved_plan_sha256") != runtime.sha(runtime.APPROVED_PLAN)
        or auth.get("method_lock_sha256") != runtime.sha(runtime.METHOD_LOCK)
        or auth.get("runtime_source_sha256") != runtime.sha(runtime.CORE_RUNTIME)
        or auth.get("limits") != runtime.MAX_LIMITS
        or auth.get("gpu_allowed") is not True
        or auth.get("cuda_allowed_for_gpu_cost") is not True
        or auth.get("cpu_cuda_allowed") is not False
        or auth.get("F_access_allowed") is not True
        or auth.get("training_allowed") is not False
    ):
        raise PermissionError("ACTUAL_MAIN_RESIDUAL_HUMAN_APPROVAL_REQUIRED")
    return auth


def _record_path(path_value: str) -> Path:
    raw = Path(path_value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def _add_protected_record(runtime, protected: dict[str, dict], path_value: str, sha256: str) -> None:
    resolved = _record_path(path_value)
    if runtime.sha(resolved) != sha256:
        raise RuntimeError("PRIOR_RECEIPT_CHANGED_BEFORE_MAIN_RESIDUAL_ACTIVATION:" + str(resolved))
    protected[str(resolved)] = {"path": str(resolved), "sha256": sha256}


def _collect_protected_prior_receipts(runtime, matched_manifest: dict) -> list[dict]:
    protected: dict[str, dict] = {}
    for record in matched_manifest.get("protected_prior_receipts", []):
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise RuntimeError("MATCHED_MANIFEST_PROTECTED_RECEIPT_RECORD_INVALID")
        _add_protected_record(runtime, protected, record["path"], record["sha256"])
    for folder in RESOURCE_DIRS_TO_PROTECT:
        if not folder.is_dir():
            raise FileNotFoundError("PRIOR_RESOURCE_FOLDER_MISSING:" + str(folder))
        for path in sorted(folder.glob("*.json")):
            if path.name == "ACTIVE.json":
                raise RuntimeError("PRIOR_PHASE_ACTIVE_LEASE_PRESENT:" + str(path))
            receipt = _read_json(path)
            if receipt.get("status") not in runtime.TERMINAL:
                raise RuntimeError("PRIOR_RECEIPT_NOT_TERMINAL:" + str(path))
            protected[str(path.resolve())] = {"path": str(path.resolve()), "sha256": runtime.sha(path)}
    if not protected:
        raise RuntimeError("NO_PRIOR_RECEIPTS_TO_PROTECT")
    for record in protected.values():
        if runtime.sha(record["path"]) != record["sha256"]:
            raise RuntimeError("PRIOR_RECEIPT_CHANGED_AFTER_MAIN_RESIDUAL_COLLECTION")
    return [protected[key] for key in sorted(protected)]


def _fail_if_existing(runtime, output: Path) -> None:
    if runtime.RUN_MANIFEST.exists():
        raise FileExistsError("DO_NOT_REPLACE_MAIN_RESIDUAL_RUN_MANIFEST:" + str(runtime.RUN_MANIFEST))
    if output.exists():
        raise FileExistsError("DO_NOT_REPLACE_MAIN_RESIDUAL_FIXTURE_RESULT:" + str(output))
    if runtime.ACTIVE.exists():
        raise RuntimeError("MAIN_RESIDUAL_ACTIVE_LEASE_PRESENT:" + str(runtime.ACTIVE))
    for active in runtime.PRIOR_ACTIVE_LEASES:
        if Path(active).exists():
            raise RuntimeError("PRIOR_PHASE_ACTIVE_LEASE_PRESENT:" + str(active))


def _write_manifest(runtime, auth: dict, matched_manifest: dict, protected: list[dict], birth: float) -> dict:
    storage_roots = matched_manifest.get("storage_roots")
    if not isinstance(storage_roots, list) or not storage_roots:
        raise RuntimeError("MATCHED_RUN_MANIFEST_STORAGE_ROOTS_REQUIRED")
    baseline_storage = runtime.storage_bytes(storage_roots)
    manifest = {
        "schema": "smallk_main_residual_run_manifest_v1",
        "phase": runtime.PHASE,
        "authorization_sha256": runtime.sha(runtime.AUTHORIZATION),
        "runtime_source_sha256": runtime.sha(runtime.CORE_RUNTIME),
        "started_unix": birth,
        "expires_unix": birth + auth["limits"]["campaign_wall_seconds"],
        "baseline_storage_bytes": baseline_storage,
        "storage_roots": storage_roots,
        "protected_prior_receipts": protected,
        "gpu_allowed": True,
        "cuda_allowed_for_gpu_cost": True,
        "cpu_cuda_allowed": False,
        "F_access_allowed": True,
        "training_allowed": False,
        "scope": "fresh main-residual phase approval; first activation runs CPU fixtures only while preserving previous continuation/stage/matched receipts and clocks",
        "bootstrap_process_birth_wall_in_first_CPU_receipt": True,
        "activation_source": _source_record(Path(__file__).resolve()),
        "runtime_source": _source_record(runtime.CORE_RUNTIME),
        "matched_run_manifest": {"path": _relative(MATCHED_RUN_MANIFEST), "sha256": runtime.sha(MATCHED_RUN_MANIFEST)},
        "human_response_identifier": runtime.EXPECTED_RESPONSE_IDENTIFIER,
    }
    runtime.write_new(runtime.RUN_MANIFEST, manifest)
    return manifest


def _error_record(error: BaseException) -> dict:
    return {"type": type(error).__name__, "message": str(error)}


def execute(args: argparse.Namespace) -> int:
    import psutil
    import main_residual_runtime_v1 as runtime

    tag = _validate_tag(args.tag)
    output = _resolve_output(args.out)
    _fail_if_existing(runtime, output)
    auth = _validate_authorization(runtime)
    birth = psutil.Process().create_time()
    if birth < auth["approved_unix"]:
        raise PermissionError("PROCESS_STARTED_BEFORE_MAIN_RESIDUAL_APPROVAL_RECORD")
    if not MATCHED_RUN_MANIFEST.is_file():
        raise FileNotFoundError("MATCHED_RUN_MANIFEST_REQUIRED:" + str(MATCHED_RUN_MANIFEST))
    matched_manifest = runtime.read(MATCHED_RUN_MANIFEST)
    protected = _collect_protected_prior_receipts(runtime, matched_manifest)
    manifest = _write_manifest(runtime, auth, matched_manifest, protected, birth)
    limit = runtime.ContinuationLimit(tag, gpu=False)
    result = {
        "schema": "smallk_main_residual_initial_fixtures_v1",
        "status": "RUNNING",
        "tag": tag,
        "scope": "metered synthetic CPU fixtures only; no scientific data/CP/model/F/GPU execution",
        "run_manifest": limit.run_manifest_record,
        "authorization": limit.authorization_record,
        "guard_adapter": limit.guard_adapter_record,
        "fixed_expiry_unix": manifest["expires_unix"],
        "GPU_execution": False,
    }
    output_ready = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(str(output))
        output_ready = True
        limit.checkfull()
        tests = _resolve_tests(args.tests)
        sources = [Path(__file__).resolve(), runtime.CORE_RUNTIME, *tests]
        before = _snapshot(sources)
        result["source_before"] = before
        result["tests"] = [_relative(path) for path in tests]
        limit.checkfull()
        import pytest
        limit.checkfull()
        code = pytest.main(["-q", *[str(path) for path in tests]])
        result["pytest_exit_code"] = int(code)
        limit.checkfull()
        after = _snapshot(sources)
        result["source_after"] = after
        changes = _snapshot_changes(before, after)
        result["source_changes"] = changes
        if changes:
            result["status"] = "FAILED_PRESERVED"
            result["failure"] = {"type": "SourceChangedDuringFixtures", "message": "main-residual fixture source changed while metered"}
        else:
            result["status"] = "COMPLETE" if int(code) == 0 else "FAILED_PRESERVED"
            if int(code) != 0:
                result["failure"] = {"type": "PytestFailed", "message": f"pytest exit code {int(code)}"}
        limit.checkfull()
    except Exception as error:
        result["status"] = "BUDGET_HELD_PRESERVED" if isinstance(error, runtime.BudgetHeld) else "FAILED_PRESERVED"
        result["failure"] = _error_record(error)
    finally:
        receipt = limit.close(result["status"], result.get("failure"))
        result["resource_receipt"] = receipt
        if receipt.get("terminal_status") != "COMPLETE":
            result["status"] = receipt.get("terminal_status", "FAILED_PRESERVED")
            result["resource_terminal_error"] = receipt.get("terminal_error")
        result["finished_unix"] = time.time()
        if output_ready:
            runtime.write_new(output, result)
    print(json.dumps({"status": result["status"], "out": _relative(output), "fixed_expiry_unix": manifest["expires_unix"]}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="verify AUTH, write manifest, and run metered CPU fixtures")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--tests", nargs="+", help="explicit fixture test files; default is runtime/math/runner main-residual fixtures")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "SOURCE_ONLY_NOT_EXECUTED", "activation_source": _source_record(Path(__file__).resolve()), "default_tag": DEFAULT_TAG, "default_out": _relative(DEFAULT_OUT)}, ensure_ascii=False, indent=2), flush=True)
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
